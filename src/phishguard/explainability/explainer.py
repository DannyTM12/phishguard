# src/phishguard/explainability/explainer.py
"""
Explicabilidad XAI para PhishGuard — Hito 3.

Responsabilidad única: dado un `MetaSubModel` entrenado y un diccionario
de features, calcular los valores SHAP (SHapley Additive exPlanations)
que explican POR QUÉ el modelo tomó la decisión que tomó.

Por qué SHAP sobre el submodelo de metadatos
---------------------------------------------
• El Random Forest del submodelo de metadatos tiene un `TreeExplainer`
  dedicado en SHAP, que es EXACTO (no una aproximación) para los árboles
  de decisión.
• El submodelo de texto (LR + TF-IDF) produce explicaciones por tokens,
  accesibles directamente desde `coef_` — no requiere SHAP (Hito 3+).
• Los 32 features de metadatos son semánticamente interpretables por
  humanos: "has_ip_url", "num_suspicious_tlds", etc. La explicación SHAP
  sobre ellos es directamente accionable para el destinatario del correo.

Diseño defensivo
----------------
• shap es una dependencia OPCIONAL: si no está instalada, el método
  `explain_metadata()` devuelve {} sin propagar ninguna excepción.
• Si el modelo inyectado es un dummy (sin `._clf`), el resultado también
  es {}.
• Cualquier otra excepción SHAP interna (memoria, compatibilidad de
  versiones, datos malformados) se atrapa y se loguea como WARNING.
• El engine nunca rompe por un fallo en la explicación.

Formato de salida
-----------------
Diccionario ordenado de mayor a menor valor SHAP positivo:

    {
        "has_ip_url":          0.312,   # feature que más empuja hacia phishing
        "has_url_shortener":   0.201,
        "num_suspicious_tlds": 0.098,
        "has_urgency_words":   0.044,
        "has_form":            0.031,   # top 5 por defecto
    }

Los valores negativos (features que alejan de phishing) se excluyen del
top-K para mantener la explicación centrada en "por qué es phishing".

Para la defensa
---------------
SHAP TreeExplainer calcula contribuciones exactas de Shapley para cada
feature usando el algoritmo TreeSHAP (Lundberg et al., 2020, NeurIPS).
Los valores son consistentes, localmente exactos y satisfacen la propiedad
de eficiencia (suman al gap entre la predicción y el valor esperado).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from phishguard.features.extractor import get_metadata_feature_names

if TYPE_CHECKING:
    # Import solo para type-checking (evita error si shap no instalado)
    from phishguard.models.meta_submodel import MetaSubModel

log = logging.getLogger(__name__)

# Lista canónica de features — misma fuente de verdad que MetaSubModel
_FEATURE_NAMES: list[str] = get_metadata_feature_names()
_N_FEATURES: int = len(_FEATURE_NAMES)   # 32


class PhishGuardExplainer:
    """
    Calcula explicaciones SHAP para el submodelo de metadatos (Random Forest).

    Wrappea `shap.TreeExplainer` de forma defensiva: si SHAP no está
    disponible o el modelo no es compatible, todos los métodos devuelven
    resultados vacíos sin excepciones.

    Attributes:
        _meta_model:  El MetaSubModel cuyo clasificador interno se explica.
        _shap_explainer: La instancia de shap.TreeExplainer (o None si SHAP
                         no está disponible o el modelo es incompatible).
        _top_k:       Número de features a devolver en el top-K.
        _enabled:     False si la inicialización de SHAP falló.
    """

    def __init__(
        self,
        meta_model: "MetaSubModel",
        top_k: int = 5,
    ) -> None:
        """
        Inicializa el explainer intentando construir un TreeExplainer.

        El constructor es NO-FAILABLE: cualquier problema con SHAP se
        registra como WARNING y `_enabled` queda en False.

        Args:
            meta_model: MetaSubModel ya cargado (con su ._clf interno).
                        Si es un dummy (sin ._clf), el explainer queda
                        deshabilitado silenciosamente.
            top_k:      Número máximo de features en la explicación.
                        Default = 5 (configurable desde model_config.yaml).
        """
        self._meta_model = meta_model
        self._top_k: int = max(1, top_k)
        self._shap_explainer: Any = None
        self._enabled: bool = False

        self._try_init_shap()

    def _try_init_shap(self) -> None:
        """
        Intenta construir el TreeExplainer de SHAP sobre el RF interno.

        Falla silenciosamente si:
          a) shap no está instalado (ImportError).
          b) El modelo no tiene ._clf (dummy model).
          c) El clasificador no es un árbol compatible con TreeExplainer.
          d) Cualquier otra excepción durante la construcción del explainer.
        """
        # ── Verificar que el modelo tiene un clf interno real ────────────────
        clf = getattr(self._meta_model, "_clf", None)
        if clf is None:
            log.info(
                "PhishGuardExplainer: el MetaSubModel no tiene ._clf "
                "(¿modelo dummy?). Explicabilidad SHAP deshabilitada."
            )
            return

        # ── Verificar que shap está instalado ────────────────────────────────
        try:
            import shap  # noqa: PLC0415 — import tardío intencional
        except ImportError:
            log.warning(
                "PhishGuardExplainer: 'shap' no está instalado. "
                "Instala con: pip install shap\n"
                "Explicabilidad SHAP deshabilitada hasta que se instale."
            )
            return

        # ── Construir TreeExplainer ───────────────────────────────────────────
        try:
            # check_additivity=False acelera la inferencia sin sacrificar
            # exactitud en RandomForest (el check es redundante para TreeSHAP)
            self._shap_explainer = shap.TreeExplainer(
                clf,
                feature_perturbation="tree_path_dependent",
            )
            self._enabled = True
            log.info(
                "PhishGuardExplainer inicializado — clf=%s | top_k=%d",
                type(clf).__name__,
                self._top_k,
            )
        except Exception as exc:
            log.warning(
                "PhishGuardExplainer: fallo al construir TreeExplainer: %s. "
                "Explicabilidad SHAP deshabilitada.",
                exc,
            )

    # ── API pública ─────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True si el explainer está activo y puede generar explicaciones."""
        return self._enabled

    def explain_metadata(
        self,
        features_dict: dict[str, Any],
    ) -> dict[str, float]:
        """
        Calcula los valores SHAP para un correo y devuelve el top-K.

        El diccionario de entrada se convierte a un array numpy 2D con el
        mismo orden de columnas que se usó en el entrenamiento (definido
        por `get_metadata_feature_names()`), se pasan por el TreeExplainer,
        y se devuelven las top-K features con contribución POSITIVA al
        score de phishing, ordenadas de mayor a menor contribución.

        Args:
            features_dict: dict[str, Any] con los 32 features de metadatos,
                           tal como devuelve
                           FeatureExtractor.extract_metadata_features().

        Returns:
            dict[str, float] ordenado: {feature_name: shap_value} para el
            top-K con valores SHAP positivos (contribuyentes a phishing).
            Devuelve {} si:
              - el explainer está deshabilitado,
              - features_dict está vacío o no es un dict,
              - SHAP falla internamente por cualquier razón.
        """
        if not self._enabled or self._shap_explainer is None:
            return {}

        if not isinstance(features_dict, dict) or not features_dict:
            log.debug("explain_metadata: features_dict vacío o no es dict.")
            return {}

        try:
            X = _dict_to_array(features_dict)
            return self._compute_shap_top_k(X)

        except Exception as exc:
            log.warning(
                "explain_metadata: excepción durante el cálculo SHAP: %s. "
                "Devolviendo explicación vacía.",
                exc,
            )
            return {}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compute_shap_top_k(self, X: np.ndarray) -> dict[str, float]:
        """
        Ejecuta TreeExplainer.shap_values() y extrae el top-K positivo.

        Maneja las dos APIs de SHAP:
          - API clásica (shap < 0.42): shap_values devuelve una lista
            [class_0_array, class_1_array] donde cada array es (n_samples, n_features).
          - API moderna (shap >= 0.42): shap_values puede devolver un
            Explanation object con .values de shape (n_samples, n_features)
            o (n_samples, n_features, n_classes).

        En ambos casos se extraen los valores para la CLASE 1 (phishing).

        Args:
            X: Array numpy (1, 32) con el vector de features del correo.

        Returns:
            dict top-K ordenado de mayor a menor valor SHAP positivo.
        """
        # Obtener índice de la clase phishing en el clasificador
        clf = self._meta_model._clf
        classes = list(clf.classes_)
        phish_idx = classes.index(1) if 1 in classes else len(classes) - 1

        raw = self._shap_explainer.shap_values(X, check_additivity=False)

        # ── Resolver el formato de salida de shap_values ────────────────────
        phish_shap: np.ndarray

        if isinstance(raw, list):
            # API clásica: raw[phish_idx] tiene shape (n_samples, n_features)
            phish_shap = np.asarray(raw[phish_idx][0], dtype=np.float64)

        elif isinstance(raw, np.ndarray):
            if raw.ndim == 3:
                # (n_samples, n_features, n_classes) — toma slice de la clase
                phish_shap = raw[0, :, phish_idx].astype(np.float64)
            elif raw.ndim == 2:
                # (n_samples, n_features) — binario, valores ya para clase 1
                phish_shap = raw[0].astype(np.float64)
            else:
                # (n_features,) — edge case
                phish_shap = raw.astype(np.float64)

        else:
            # API moderna con Explanation object — intentar .values
            values = getattr(raw, "values", None)
            if values is None:
                log.warning("SHAP devolvió un tipo desconocido: %s", type(raw))
                return {}
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 3:
                phish_shap = arr[0, :, phish_idx]
            elif arr.ndim == 2:
                phish_shap = arr[0]
            else:
                phish_shap = arr.flatten()

        # ── Validar longitud ─────────────────────────────────────────────────
        if len(phish_shap) != _N_FEATURES:
            log.warning(
                "SHAP devolvió %d valores, se esperaban %d. "
                "Devolviendo explicación vacía.",
                len(phish_shap),
                _N_FEATURES,
            )
            return {}

        # ── Construir pares (nombre, valor) para los positivos ───────────────
        paired = [
            (_FEATURE_NAMES[i], float(phish_shap[i]))
            for i in range(_N_FEATURES)
            if phish_shap[i] > 0.0   # solo contribuciones hacia phishing
        ]

        # Ordenar de mayor a menor y tomar top-K
        paired.sort(key=lambda x: x[1], reverse=True)
        top_k = paired[: self._top_k]

        result = {name: round(value, 6) for name, value in top_k}

        log.debug(
            "explain_metadata: top-%d SHAP positivos: %s",
            self._top_k,
            result,
        )

        return result

    def __repr__(self) -> str:
        return (
            f"PhishGuardExplainer("
            f"enabled={self._enabled}, "
            f"top_k={self._top_k}, "
            f"clf={type(getattr(self._meta_model, '_clf', None)).__name__})"
        )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _dict_to_array(features: dict[str, Any]) -> np.ndarray:
    """
    Convierte un dict de features a array numpy (1, 32) con orden canónico.

    Idéntico en semántica a `meta_submodel._dict_to_array`, pero duplicado
    aquí para mantener `explainer.py` sin dependencias circulares con
    `meta_submodel.py` (que a su vez importa el extractor).

    Los booleanos se convierten a float. Las claves ausentes se rellenan
    con 0.0 con un warning.
    """
    missing: list[str] = []
    vector: list[float] = []

    for name in _FEATURE_NAMES:
        value = features.get(name)
        if value is None:
            missing.append(name)
            vector.append(0.0)
        elif isinstance(value, bool):
            vector.append(1.0 if value else 0.0)
        else:
            vector.append(float(value))

    if missing:
        log.warning(
            "_dict_to_array (explainer): %d features ausentes → 0.0: %s",
            len(missing),
            missing,
        )

    return np.array(vector, dtype=np.float32).reshape(1, -1)