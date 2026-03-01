# src/phishguard/models/meta_submodel.py
"""
Submodelo de Metadatos para PhishGuard — Hito 2.

Responsabilidad única: envolver un clasificador sklearn entrenado y
exponerlo a través del protocolo `SubModel` definido en `fusion_engine.py`.

Este módulo NO entrena nada. El entrenamiento ocurre en
`scripts/train_metadata_model.py`. Aquí solo se deserializa el
artefacto ya entrenado y se usa para inferencia.

Protocolo cumplido
------------------
  predict_proba(features: dict) -> float
  model_name: str                          (property)

Uso típico (Hito 2 en lifespan de FastAPI)
------------------------------------------
    from phishguard.models.meta_submodel import MetaSubModel

    meta = MetaSubModel.load(Path("artifacts/meta_model.pkl"))
    engine = PhishGuardEngine(meta_model=meta, ...)

Garantía de orden de columnas
------------------------------
El vector de features se construye SIEMPRE a partir de
`get_metadata_feature_names()`, nunca de `dict.keys()`. Esto garantiza
que el orden de columnas en inferencia sea idéntico al orden usado
durante el entrenamiento, incluso si Python cambia el orden de
iteración de dicts en versiones futuras (aunque desde 3.7 es estable,
la garantía explícita es más robusta y documentable en la defensa).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import ClassifierMixin

from phishguard.features.extractor import get_metadata_feature_names

log = logging.getLogger(__name__)

# Lista canónica y estable de features — importada una sola vez al cargar el módulo.
# Todos los métodos de este módulo la usan como fuente de verdad.
_FEATURE_NAMES: list[str] = get_metadata_feature_names()
_N_FEATURES: int = len(_FEATURE_NAMES)   # 32


# ---------------------------------------------------------------------------
# Serialization envelope
# ---------------------------------------------------------------------------


class _ModelArtifact:
    """
    Sobre de serialización que se persiste en el .pkl.

    Almacena el clasificador sklearn junto con metadatos de trazabilidad:
    versión del esquema de features, fecha de entrenamiento y nombre del
    estimador. Esto permite detectar incompatibilidades al cargar un
    artefacto antiguo con una versión nueva del código.

    Versión del esquema: se incrementa cuando cambia la lista de features
    o su orden en `get_metadata_feature_names()`. Al cargar, se valida
    que la versión del artefacto coincida con la del código.
    """

    # Incrementar cada vez que se cambie la lista/orden de features.
    SCHEMA_VERSION: str = "1.0"

    def __init__(
        self,
        classifier: ClassifierMixin,
        feature_names: list[str],
        trained_at: str,
        estimator_type: str,
        train_metrics: dict[str, float],
    ) -> None:
        self.schema_version = self.SCHEMA_VERSION
        self.classifier = classifier
        self.feature_names = feature_names
        self.trained_at = trained_at
        self.estimator_type = estimator_type
        self.train_metrics = train_metrics


# ---------------------------------------------------------------------------
# MetaSubModel
# ---------------------------------------------------------------------------


class MetaSubModel:
    """
    Submodelo de metadatos basado en un clasificador sklearn entrenado.

    Implementa el protocolo `SubModel` de `fusion_engine.py`:
      • predict_proba(features: dict) -> float
      • model_name: str  (property)

    Attributes (internos, no modificar directamente):
        _artifact:    Sobre de serialización con el clf y los metadatos.
        _clf:         Clasificador sklearn extraído del artefacto.
        _phish_idx:   Índice de la clase positiva (phishing=1) en
                      clf.classes_, necesario para extraer la probabilidad
                      correcta del array que devuelve sklearn.
    """

    def __init__(self, artifact: _ModelArtifact) -> None:
        """
        Construye el submodelo a partir de un artefacto ya validado.

        No usar directamente en producción. Usar `MetaSubModel.load()`.
        """
        self._artifact = artifact
        self._clf: ClassifierMixin = artifact.classifier

        # Determinar el índice de la clase phishing (label=1) en clf.classes_
        # RandomForest puede ordenar las clases como [0, 1] o [1, 0] dependiendo
        # de los datos de entrenamiento, así que no asumimos índice fijo.
        classes = list(self._clf.classes_)
        if 1 not in classes:
            raise ValueError(
                f"El clasificador no tiene la clase positiva (1) en sus clases: {classes}. "
                "Verifica que el modelo se entrenó con label=1 para phishing."
            )
        self._phish_idx: int = classes.index(1)

        log.info(
            "MetaSubModel cargado — clf=%s | schema=%s | trained_at=%s | "
            "phish_class_idx=%d | metrics=%s",
            artifact.estimator_type,
            artifact.schema_version,
            artifact.trained_at,
            self._phish_idx,
            artifact.train_metrics,
        )

    # ── Protocolo SubModel ──────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """
        Identificador único del modelo para logs, trazabilidad y la API.

        Formato: "<estimador>_meta_v<schema>_<fecha>".
        """
        return (
            f"{self._artifact.estimator_type}_meta"
            f"_v{self._artifact.schema_version}"
            f"_{self._artifact.trained_at}"
        )

    def predict_proba(self, features: dict[str, Any] | str) -> float:
        """
        Devuelve la probabilidad de phishing según el clasificador entrenado.

        El dict de entrada se convierte a un array 2D numpy respetando
        el orden EXACTO de `get_metadata_feature_names()`. Los valores
        booleanos se convierten a float (True→1.0, False→0.0).

        Args:
            features: dict[str, Any] con exactamente los 32 features
                      de metadatos producidos por FeatureExtractor.
                      Si se pasa un str (uso incorrecto), devuelve 0.0
                      con warning en el log.

        Returns:
            Probabilidad ∈ [0.0, 1.0] de que el correo sea phishing.

        Raises:
            ValueError: si el dict tiene una versión de schema de features
                        incompatible (número de features distinto).
        """
        if not isinstance(features, dict):
            log.warning(
                "MetaSubModel.predict_proba() recibió %s en lugar de dict. "
                "Devolviendo 0.0.",
                type(features).__name__,
            )
            return 0.0

        # Construir el vector respetando el orden canónico
        feature_vector = _dict_to_array(features)

        # Inferencia sklearn: predict_proba devuelve [[prob_0, prob_1]]
        proba_matrix: np.ndarray = self._clf.predict_proba(feature_vector)
        phish_proba: float = float(proba_matrix[0, self._phish_idx])

        return float(min(max(phish_proba, 0.0), 1.0))

    # ── Serialización ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "MetaSubModel":
        """
        Carga un MetaSubModel desde un archivo .pkl serializado con joblib.

        Valida que:
          1. El archivo existe y puede deserializarse.
          2. El contenido es un _ModelArtifact.
          3. El schema_version del artefacto coincide con el del código.
          4. La lista de features del artefacto coincide con la actual.

        Args:
            path: Ruta al archivo .pkl generado por `train_metadata_model.py`.

        Returns:
            Instancia de MetaSubModel lista para inferencia.

        Raises:
            FileNotFoundError: si el archivo no existe.
            TypeError:         si el .pkl no contiene un _ModelArtifact.
            ValueError:        si hay incompatibilidad de schema o features.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Artefacto del modelo no encontrado: {path}\n"
                "Ejecuta primero: python scripts/train_metadata_model.py"
            )

        log.info("Cargando MetaSubModel desde: %s", path)

        try:
            artifact: _ModelArtifact = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo deserializar el artefacto en {path}: {exc}"
            ) from exc

        # Validar tipo
        if not isinstance(artifact, _ModelArtifact):
            raise TypeError(
                f"El archivo {path} no contiene un _ModelArtifact válido. "
                f"Tipo encontrado: {type(artifact)}. "
                "Regenera el artefacto con train_metadata_model.py."
            )

        # Validar versión del schema
        if artifact.schema_version != _ModelArtifact.SCHEMA_VERSION:
            raise ValueError(
                f"Incompatibilidad de schema: el artefacto tiene v{artifact.schema_version} "
                f"pero el código espera v{_ModelArtifact.SCHEMA_VERSION}. "
                "Reentrena el modelo con la versión actual del extractor."
            )

        # Validar lista de features
        if artifact.feature_names != _FEATURE_NAMES:
            added = set(_FEATURE_NAMES) - set(artifact.feature_names)
            removed = set(artifact.feature_names) - set(_FEATURE_NAMES)
            raise ValueError(
                f"La lista de features del artefacto no coincide con la actual.\n"
                f"  Features nuevas (no en artefacto): {added}\n"
                f"  Features eliminadas (en artefacto): {removed}\n"
                "Reentrena el modelo con la versión actual del extractor."
            )

        return cls(artifact)

    @staticmethod
    def save(
        classifier: ClassifierMixin,
        path: Path,
        trained_at: str,
        train_metrics: dict[str, float],
    ) -> None:
        """
        Serializa un clasificador sklearn entrenado como _ModelArtifact.

        Llamado por `train_metadata_model.py` al finalizar el entrenamiento.
        No debe llamarse directamente en inferencia.

        Args:
            classifier:    Clasificador sklearn ya ajustado (post-fit).
            path:          Destino del archivo .pkl.
            trained_at:    Timestamp de entrenamiento (ISO 8601, sin zona).
            train_metrics: Métricas de validación {precision, recall, f1}.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = _ModelArtifact(
            classifier=classifier,
            feature_names=_FEATURE_NAMES,
            trained_at=trained_at,
            estimator_type=type(classifier).__name__,
            train_metrics=train_metrics,
        )

        # Escritura atómica: .tmp → rename
        tmp_path = path.with_suffix(".tmp.pkl")
        try:
            joblib.dump(artifact, tmp_path, compress=3)
            tmp_path.rename(path)
            size_kb = path.stat().st_size / 1024
            log.info(
                "Artefacto guardado: %s (%.1f KB) — schema=v%s | metrics=%s",
                path,
                size_kb,
                _ModelArtifact.SCHEMA_VERSION,
                train_metrics,
            )
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Error al guardar el artefacto: {exc}") from exc

    # ── Introspección ───────────────────────────────────────────────────────

    @property
    def feature_importances(self) -> dict[str, float]:
        """
        Devuelve la importancia de cada feature según el clasificador.

        Solo disponible para estimadores con `.feature_importances_`
        (RandomForest, GradientBoosting, XGBoost, etc.).

        Returns:
            dict ordenado de mayor a menor importancia,
            o dict vacío si el estimador no soporta feature_importances_.
        """
        if not hasattr(self._clf, "feature_importances_"):
            log.warning(
                "%s no soporta feature_importances_.",
                type(self._clf).__name__,
            )
            return {}

        importances = self._clf.feature_importances_
        paired = dict(zip(_FEATURE_NAMES, importances.tolist()))
        return dict(sorted(paired.items(), key=lambda x: x[1], reverse=True))

    @property
    def train_metrics(self) -> dict[str, float]:
        """Métricas de validación registradas durante el entrenamiento."""
        return dict(self._artifact.train_metrics)

    def __repr__(self) -> str:
        return (
            f"MetaSubModel("
            f"clf={self._artifact.estimator_type}, "
            f"schema=v{self._artifact.schema_version}, "
            f"trained_at={self._artifact.trained_at!r}, "
            f"metrics={self._artifact.train_metrics})"
        )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _dict_to_array(features: dict[str, Any]) -> np.ndarray:
    """
    Convierte un dict de features a un array numpy 2D con orden canónico.

    El orden de columnas sigue EXACTAMENTE `get_metadata_feature_names()`.
    Los valores booleanos se convierten a float32 (True→1.0, False→0.0).
    Los valores ausentes (key no encontrada) se completan con 0.0 y
    se registra un warning — esto nunca debería ocurrir en producción si
    el extractor está actualizado.

    Args:
        features: dict producido por FeatureExtractor.extract_metadata_features().

    Returns:
        numpy array de shape (1, 32) y dtype float32, listo para sklearn.

    Raises:
        ValueError: si el número de features difiere del esperado después
                    de completar los valores ausentes (indica schema drift).
    """
    missing_keys: list[str] = []
    vector: list[float] = []

    for name in _FEATURE_NAMES:
        value = features.get(name)
        if value is None:
            missing_keys.append(name)
            vector.append(0.0)
        elif isinstance(value, bool):
            vector.append(1.0 if value else 0.0)
        else:
            vector.append(float(value))

    if missing_keys:
        log.warning(
            "_dict_to_array: %d features ausentes en el dict de entrada "
            "(completadas con 0.0): %s",
            len(missing_keys),
            missing_keys,
        )

    # Reshape a (1, n_features) para sklearn
    arr = np.array(vector, dtype=np.float32).reshape(1, -1)

    if arr.shape[1] != _N_FEATURES:
        raise ValueError(
            f"Array de features tiene {arr.shape[1]} columnas, "
            f"se esperaban {_N_FEATURES}. "
            "Verifica que get_metadata_feature_names() esté sincronizado."
        )

    return arr