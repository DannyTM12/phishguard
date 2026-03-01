# src/phishguard/models/fusion_engine.py
"""
Motor de inferencia híbrido de PhishGuard — Fusión Tardía con Gating.

Flujo de decisión
-----------------

  email (subject + body + sender)
          │
          ▼
  ┌─────────────────────┐
  │  FeatureExtractor   │  → metadata_features: dict[str, float]
  └─────────────────────┘
          │
          ▼
  ┌─────────────────────┐
  │  SubmodelMeta       │  → score_meta ∈ [0, 1]
  │  (técnico/URLs)     │
  └─────────────────────┘
          │
          ▼
  score_meta >= θ_meta?  ──── SÍ ──► label=PHISHING  (gating activado)
          │
          NO
          │
          ▼
  ┌─────────────────────┐
  │  SubmodelText       │  → score_text ∈ [0, 1]
  │  (NLP/TF-IDF/BERT)  │
  └─────────────────────┘
          │
          ▼
  score_final = α·score_meta + (1-α)·score_text
          │
          ▼
  score_final >= θ_final? → label ∈ {PHISHING, LEGITIMATE}

Diseño de inyección de dependencias
-------------------------------------
Los submodelos se inyectan a través del protocolo `SubModel`.
En el Hito 1 se usan implementaciones dummy/heurísticas.
En el Hito 2, `SubModel` se cumple con los pickles reales:

    from phishguard.models.fusion_engine import PhishGuardEngine
    from phishguard.models.meta_submodel import MetaSubModel
    from phishguard.models.text_submodel import TextSubModel

    engine = PhishGuardEngine(
        meta_model=MetaSubModel.load("models/meta.pkl"),
        text_model=TextSubModel.load("models/text.pkl"),
    )

Sin cambiar ninguna firma pública, ningún test, ni la API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from phishguard.config import PhishGuardConfig, get_config
from phishguard.features.extractor import FeatureExtractor, ExtractorConfig
from phishguard.preprocessing.text_cleaner import (
    count_urgency_words,
    has_urgency_words,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocolo público para los submodelos (contrato del Hito 2)
# ---------------------------------------------------------------------------


@runtime_checkable
class SubModel(Protocol):
    """
    Protocolo que deben cumplir TODOS los submodelos de PhishGuard.

    Cualquier objeto con un método `predict_proba(features) -> float`
    satisface este protocolo, tanto los dummies del Hito 1 como los
    modelos entrenados del Hito 2.

    El uso de `runtime_checkable` permite validar la interfaz en
    `isinstance()` durante la inicialización del engine.
    """

    def predict_proba(self, features: dict[str, Any] | str) -> float:
        """
        Devuelve la probabilidad de que el correo sea phishing.

        Args:
            features: Para el submodelo de metadatos, un dict con los
                      32 features numéricos. Para el de texto, un str
                      con el texto preparado.

        Returns:
            Probabilidad ∈ [0.0, 1.0] donde 1.0 = phishing seguro.
        """
        ...

    @property
    def model_name(self) -> str:
        """Identificador del modelo para logs y trazabilidad."""
        ...


# ---------------------------------------------------------------------------
# Tipos de datos de resultados
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GatingResult:
    """
    Resultado de la etapa de gating por metadatos.

    Attributes:
        activated:    True si score_meta superó el umbral θ_meta y se
                      tomó una decisión inmediata sin consultar el texto.
        score_meta:   Score producido por el submodelo de metadatos.
        threshold:    Valor de θ_meta utilizado.
    """

    activated: bool
    score_meta: float
    threshold: float


@dataclass(frozen=True)
class PredictionResult:
    """
    Resultado completo de una inferencia de PhishGuard.

    Contiene todos los valores intermedios para trazabilidad,
    depuración y para que la API devuelva una respuesta rica.

    Attributes:
        label:          Etiqueta final: "phishing" | "legitimate".
        is_phishing:    Booleano equivalente a label == "phishing".
        score_final:    Score de fusión usado para la decisión final.
                        Igual a score_meta si el gating se activó.
        score_meta:     Score del submodelo de metadatos ∈ [0, 1].
        score_text:     Score del submodelo de texto ∈ [0, 1], o None
                        si el gating se activó (texto no evaluado).
        gating:         Detalles de la etapa de gating.
        metadata_features: Dict con los 32 features extraídos.
        prepared_text:  Texto preparado para el submodelo NLP (o "" si
                        el gating se activó).
        latency_ms:     Tiempo de inferencia total en milisegundos.
        meta_model_name: Identificador del submodelo de metadatos.
        text_model_name: Identificador del submodelo de texto, o None.
    """

    label: str
    is_phishing: bool
    score_final: float
    score_meta: float
    score_text: float | None
    gating: GatingResult
    metadata_features: dict[str, Any]
    prepared_text: str
    latency_ms: float
    meta_model_name: str
    text_model_name: str | None


# ---------------------------------------------------------------------------
# Submodelos dummy / heurísticos (Hito 1)
# ---------------------------------------------------------------------------


class _DummyMetaModel:
    """
    Submodelo de metadatos heurístico para el Hito 1.

    NO usa ningún modelo entrenado. Calcula un score de riesgo
    sumando pesos fijos asignados a features de alto valor predictivo
    según la literatura de detección de phishing.

    Esta implementación es útil para:
      1. Verificar el flujo completo del engine sin modelos pkl.
      2. Servir como baseline interpretable en el Hito 2.
      3. Probar la API con respuestas coherentes (no puramente aleatorias).

    Los pesos reflejan la importancia relativa documentada en trabajos
    como Shahrivari et al. (2020) y Maturure et al. (2024).
    """

    # Pesos de features → contribución al score de riesgo.
    # La suma de todos los pesos es > 1.0 para permitir saturación;
    # el score se clipea a [0, 1] al final.
    _FEATURE_WEIGHTS: dict[str, float] = {
        # Señales de alta confianza (pesos altos)
        "has_ip_url":           0.35,  # IP directa en URL → muy sospechoso
        "has_form":             0.25,  # formulario embebido
        "has_iframe":           0.20,  # iframe oculto
        "has_url_shortener":    0.20,  # acortador de URL
        # Señales medias
        "has_urgency_words":    0.15,
        "has_brand_mention":    0.12,
        "num_suspicious_tlds":  0.10,  # por TLD sospechoso (acumulable)
        "num_redirect_params":  0.10,  # por parámetro redirect (acumulable)
        "subject_has_urgency":  0.08,
        "ratio_uppercase_body": 0.06,  # normalizado abajo
        # Señales débiles (corroboradoras)
        "has_html":             0.04,
        "subject_has_brand":    0.05,
        "subject_is_empty":     0.03,
    }

    @property
    def model_name(self) -> str:
        return "DummyMetaModel_heuristic_v1"

    def predict_proba(self, features: dict[str, Any] | str) -> float:
        """
        Calcula el score de riesgo de metadatos usando pesos fijos.

        Args:
            features: dict[str, Any] con los 32 features de metadatos.

        Returns:
            Score ∈ [0.0, 1.0].
        """
        if not isinstance(features, dict):
            log.warning("DummyMetaModel esperaba dict, recibió %s. Score=0.", type(features))
            return 0.0

        score = 0.0

        for feat_name, weight in self._FEATURE_WEIGHTS.items():
            value = features.get(feat_name, 0)

            # Booleanos y enteros/floats se manejan igual:
            # True → 1.0, False → 0.0, int/float → valor directo
            if isinstance(value, bool):
                numeric = float(value)
            elif isinstance(value, (int, float)):
                # Para ratios como ratio_uppercase_body (0..1):
                # amplificar si es > 0.5 (mucho uppercase)
                if feat_name.startswith("ratio_"):
                    numeric = 1.0 if value > 0.5 else value
                # Para conteos acumulables (num_suspicious_tlds, etc.):
                # capear en 3 para evitar que un solo email domine
                else:
                    numeric = min(float(value), 3.0) / 3.0 if value > 1 else float(value)
            else:
                numeric = 0.0

            score += weight * numeric

        # Clipear a [0, 1]
        return float(min(max(score, 0.0), 1.0))


class _DummyTextModel:
    """
    Submodelo de texto heurístico para el Hito 1.

    Estima el score de riesgo del texto usando el conteo de palabras
    de urgencia en el texto preparado, normalizado por la longitud.

    Es deliberadamente simple: el Hito 2 lo reemplaza con TF-IDF + SVM
    o DistilBERT fine-tuned.
    """

    # Umbral de densidad de palabras de urgencia considerado "alto riesgo"
    _HIGH_URGENCY_DENSITY: float = 0.02  # 2 palabras urgentes por 100 palabras

    @property
    def model_name(self) -> str:
        return "DummyTextModel_heuristic_v1"

    def predict_proba(self, features: dict[str, Any] | str) -> float:
        """
        Estima riesgo de phishing basándose en la densidad de urgencia.

        Args:
            features: str con el texto preparado (subject [SEP] body).

        Returns:
            Score ∈ [0.0, 1.0].
        """
        if not isinstance(features, str):
            log.warning("DummyTextModel esperaba str, recibió %s. Score=0.", type(features))
            return 0.0

        if not features.strip():
            return 0.0

        text = features
        word_count = max(len(text.split()), 1)
        urgency_count = count_urgency_words(text)

        # Densidad de urgencia normalizada
        density = urgency_count / word_count

        # Score base desde densidad: densidad >= 0.02 → score 0.5+
        density_score = min(density / self._HIGH_URGENCY_DENSITY, 1.0) * 0.6

        # Bonus por urgency absoluta (aunque el texto sea largo)
        urgency_bonus = min(urgency_count / 10.0, 0.4)

        return float(min(density_score + urgency_bonus, 1.0))


# ---------------------------------------------------------------------------
# Lógica de fusión (pura, sin estado)
# ---------------------------------------------------------------------------


def compute_fusion_score(
    score_meta: float,
    score_text: float,
    alpha: float,
) -> float:
    """
    Fórmula de fusión tardía ponderada.

    score_final = α · score_meta + (1 - α) · score_text

    Args:
        score_meta: Score del submodelo técnico ∈ [0, 1].
        score_text: Score del submodelo de texto ∈ [0, 1].
        alpha:      Peso del submodelo de metadatos (de la config).

    Returns:
        Score final ∈ [0, 1].
    """
    result = alpha * score_meta + (1.0 - alpha) * score_text
    # Clipear por seguridad frente a errores numéricos de punto flotante
    return float(min(max(result, 0.0), 1.0))


def apply_gating(
    score_meta: float,
    theta_meta: float,
) -> bool:
    """
    Evalúa si el gating debe activarse.

    El gating evita invocar el submodelo de texto cuando la evidencia
    técnica es suficientemente fuerte por sí sola.

    Args:
        score_meta:  Score del submodelo de metadatos ∈ [0, 1].
        theta_meta:  Umbral de activación del gating (config.gating).

    Returns:
        True si el gating se activa (score_meta >= theta_meta).
    """
    return score_meta >= theta_meta


def score_to_label(score: float, threshold: float) -> str:
    """
    Convierte un score numérico en una etiqueta de texto.

    Args:
        score:     Score final ∈ [0, 1].
        threshold: Umbral de decisión θ_final (config.fusion).

    Returns:
        "phishing" si score >= threshold, "legitimate" en caso contrario.
    """
    return "phishing" if score >= threshold else "legitimate"


# ---------------------------------------------------------------------------
# Motor principal
# ---------------------------------------------------------------------------


class PhishGuardEngine:
    """
    Motor de inferencia híbrido de PhishGuard.

    Orquesta el flujo completo:
      extracción de features → scoring de metadatos → gating →
      scoring de texto (condicional) → fusión tardía → decisión.

    Inyección de modelos
    --------------------
    Acepta cualquier objeto que cumpla el protocolo `SubModel`.
    Si no se proveen, usa las implementaciones dummy heurísticas
    del Hito 1.

    Para el Hito 2, inyecta los modelos entrenados:

        engine = PhishGuardEngine(
            meta_model=MetaSubModel.load("models/meta.pkl"),
            text_model=TextSubModel.load("models/text.pkl"),
        )

    Singleton recomendado
    ---------------------
    Usa `PhishGuardEngine.get_instance()` para obtener un singleton
    thread-safe adecuado para FastAPI (evita recargar modelos en
    cada request).
    """

    _instance: PhishGuardEngine | None = None

    def __init__(
        self,
        config: PhishGuardConfig | None = None,
        meta_model: SubModel | None = None,
        text_model: SubModel | None = None,
        extractor_config: ExtractorConfig | None = None,
    ) -> None:
        """
        Inicializa el engine con configuración y submodelos opcionales.

        Args:
            config:           Configuración de PhishGuard. Si es None,
                              usa get_config() (carga desde YAML).
            meta_model:       Submodelo de metadatos. Si es None, usa
                              _DummyMetaModel (heurístico).
            text_model:       Submodelo de texto. Si es None, usa
                              _DummyTextModel (heurístico).
            extractor_config: Config del FeatureExtractor. Si es None,
                              usa ExtractorConfig() con defaults.
        """
        self._config: PhishGuardConfig = config or get_config()
        self._meta_model: SubModel = meta_model or _DummyMetaModel()
        self._text_model: SubModel = text_model or _DummyTextModel()
        self._extractor: FeatureExtractor = FeatureExtractor(extractor_config)

        # Validar que los modelos inyectados cumplen el protocolo
        for name, model in [("meta_model", self._meta_model), ("text_model", self._text_model)]:
            if not isinstance(model, SubModel):
                raise TypeError(
                    f"'{name}' no cumple el protocolo SubModel. "
                    f"Debe tener predict_proba() y model_name."
                )

        log.info(
            "PhishGuardEngine inicializado — meta=%s | text=%s | "
            "θ_meta=%.2f | α=%.2f | θ_final=%.2f",
            self._meta_model.model_name,
            self._text_model.model_name,
            self._config.gating.metadata_threshold,
            self._config.fusion.alpha,
            self._config.fusion.decision_threshold,
        )

    # ── Singleton ───────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        meta_model: SubModel | None = None,
        text_model: SubModel | None = None,
    ) -> "PhishGuardEngine":
        """
        Devuelve el singleton del engine, creándolo si no existe.

        Thread-safe para entornos de producción con múltiples workers
        (en FastAPI con Uvicorn, la inicialización ocurre en el
        evento `startup` antes de aceptar requests).

        Args:
            meta_model: Submodelo de metadatos a inyectar en la
                        primera inicialización. Ignorado si el
                        singleton ya existe.
            text_model: Submodelo de texto a inyectar en la
                        primera inicialización.

        Returns:
            Instancia única de PhishGuardEngine.
        """
        if cls._instance is None:
            cls._instance = cls(meta_model=meta_model, text_model=text_model)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Destruye el singleton actual.

        Útil en tests para crear un engine limpio con distintos modelos
        o configuraciones sin interferencia entre casos de prueba.
        """
        cls._instance = None

    # ── Método principal de inferencia ─────────────────────────────────────

    def predict(
        self,
        subject: str,
        body: str,
        sender: str = "",
    ) -> PredictionResult:
        """
        Clasifica un correo electrónico como phishing o legítimo.

        Implementa el flujo completo de fusión tardía con gating:
          1. Extrae las 32 features de metadatos.
          2. Obtiene score_meta del submodelo técnico.
          3. Evalúa el gating: si score_meta >= θ_meta, decide phishing
             sin invocar el submodelo de texto.
          4. Si el gating no se activa, obtiene score_text y calcula
             score_final = α·score_meta + (1-α)·score_text.
          5. Decide label según score_final >= θ_final.

        Args:
            subject: Asunto del correo (texto crudo).
            body:    Cuerpo del correo (puede contener HTML).
            sender:  Dirección del remitente (reservado para features
                     de cabeceras en el Hito 2; no usado actualmente).

        Returns:
            PredictionResult con todos los scores, flags y metadata.
        """
        t_start = time.perf_counter()

        # ── Paso 1: Extracción de features ─────────────────────────────────
        urls = self._extractor.get_urls_from_body(body)
        metadata_features = self._extractor.extract_metadata_features(
            subject=subject,
            body=body,
            urls=urls,
        )

        # ── Paso 2: Score de metadatos ──────────────────────────────────────
        score_meta = self._meta_model.predict_proba(metadata_features)
        score_meta = float(min(max(score_meta, 0.0), 1.0))  # clipear por seguridad

        log.debug(
            "score_meta=%.4f | θ_meta=%.2f | model=%s",
            score_meta,
            self._config.gating.metadata_threshold,
            self._meta_model.model_name,
        )

        # ── Paso 3: Gating ──────────────────────────────────────────────────
        gating_activated = apply_gating(
            score_meta=score_meta,
            theta_meta=self._config.gating.metadata_threshold,
        )
        gating_result = GatingResult(
            activated=gating_activated,
            score_meta=score_meta,
            threshold=self._config.gating.metadata_threshold,
        )

        if gating_activated:
            # El submodelo de texto no se invoca → ahorro computacional
            log.debug("Gating activado — decisión directa sin submodelo de texto.")
            score_text = None
            score_final = score_meta
            prepared_text = ""
            text_model_name = None
        else:
            # ── Paso 4: Score de texto ──────────────────────────────────────
            prepared_text = self._extractor.extract_text_features(
                subject=subject,
                body=body,
            )
            score_text_raw = self._text_model.predict_proba(prepared_text)
            score_text = float(min(max(score_text_raw, 0.0), 1.0))
            text_model_name = self._text_model.model_name

            log.debug(
                "score_text=%.4f | model=%s",
                score_text,
                self._text_model.model_name,
            )

            # ── Paso 5: Fusión tardía ───────────────────────────────────────
            score_final = compute_fusion_score(
                score_meta=score_meta,
                score_text=score_text,
                alpha=self._config.fusion.alpha,
            )

        # ── Paso 6: Decisión final ──────────────────────────────────────────
        label = score_to_label(
            score=score_final,
            threshold=self._config.fusion.decision_threshold,
        )
        is_phishing = label == "phishing"
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        log.info(
            "predict() → label=%s | score_final=%.4f | score_meta=%.4f | "
            "score_text=%s | gating=%s | latency=%.1fms",
            label,
            score_final,
            score_meta,
            f"{score_text:.4f}" if score_text is not None else "N/A",
            gating_activated,
            latency_ms,
        )

        return PredictionResult(
            label=label,
            is_phishing=is_phishing,
            score_final=round(score_final, 6),
            score_meta=round(score_meta, 6),
            score_text=round(score_text, 6) if score_text is not None else None,
            gating=gating_result,
            metadata_features=metadata_features,
            prepared_text=prepared_text,
            latency_ms=round(latency_ms, 3),
            meta_model_name=self._meta_model.model_name,
            text_model_name=text_model_name,
        )

    # ── Propiedades de introspección ────────────────────────────────────────

    @property
    def config(self) -> PhishGuardConfig:
        """Acceso de solo lectura a la configuración del engine."""
        return self._config

    @property
    def is_using_dummy_models(self) -> bool:
        """True si alguno de los submodelos es un dummy heurístico."""
        return isinstance(self._meta_model, _DummyMetaModel) or isinstance(
            self._text_model, _DummyTextModel
        )

    def __repr__(self) -> str:
        return (
            f"PhishGuardEngine("
            f"meta={self._meta_model.model_name!r}, "
            f"text={self._text_model.model_name!r}, "
            f"dummy={self.is_using_dummy_models})"
        )