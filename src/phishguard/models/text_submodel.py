# src/phishguard/models/text_submodel.py
"""
Submodelo de Texto para PhishGuard — Hito 2.

Responsabilidad única: envolver un Pipeline sklearn (TF-IDF + clasificador
lineal) ya entrenado y exponerlo a través del protocolo `SubModel`
definido en `fusion_engine.py`.

Este módulo NO entrena nada. El entrenamiento ocurre en
`scripts/train_text_model.py`. Aquí solo se deserializa el artefacto
y se usa para inferencia.

Ventaja del Pipeline sklearn
----------------------------
Al serializar el Pipeline completo (TfidfVectorizer + LogisticRegression),
la inferencia se reduce a una sola llamada:

    pipeline.predict_proba(["asunto [SEP] cuerpo limpio"])

No hay que guardar el vocabulario por separado ni hacer el transform
manualmente. El Pipeline garantiza que el texto de inferencia pasa
exactamente la misma transformación que en entrenamiento.

Protocolo cumplido
------------------
  predict_proba(features: str) -> float
  model_name: str                        (property)

Uso típico (Hito 2 en lifespan de FastAPI)
------------------------------------------
    from phishguard.models.text_submodel import TextSubModel

    text = TextSubModel.load(Path("artifacts/text_model.pkl"))
    engine = PhishGuardEngine(text_model=text, ...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sobre de serialización
# ---------------------------------------------------------------------------


class _TextModelArtifact:
    """
    Envelope de serialización para el Pipeline de texto.

    Almacena el Pipeline sklearn junto con metadatos de trazabilidad
    para detectar artefactos obsoletos al cargar.

    Schema version
    --------------
    Incrementar si cambia la interfaz del extractor de texto
    (separador [SEP], truncamiento, normalización unicode, etc.)
    que afecte la distribución de tokens esperada por el modelo.
    """

    SCHEMA_VERSION: str = "1.0"

    def __init__(
        self,
        pipeline: Pipeline,
        trained_at: str,
        vectorizer_params: dict[str, Any],
        classifier_type: str,
        train_metrics: dict[str, float],
        text_separator: str,
        max_text_length: int,
    ) -> None:
        self.schema_version: str = self.SCHEMA_VERSION
        self.pipeline: Pipeline = pipeline
        self.trained_at: str = trained_at
        self.vectorizer_params: dict[str, Any] = vectorizer_params
        self.classifier_type: str = classifier_type
        self.train_metrics: dict[str, float] = train_metrics
        # Guardar los params del extractor de texto para detectar
        # incompatibilidades si cambia el preprocesamiento en el futuro.
        self.text_separator: str = text_separator
        self.max_text_length: int = max_text_length


# ---------------------------------------------------------------------------
# TextSubModel
# ---------------------------------------------------------------------------


class TextSubModel:
    """
    Submodelo de texto basado en un Pipeline sklearn (TF-IDF + clasificador).

    Implementa el protocolo `SubModel` de `fusion_engine.py`:
      • predict_proba(features: str) -> float
      • model_name: str  (property)

    El Pipeline encapsula la vectorización y la clasificación en un
    único objeto, lo que garantiza que el preprocesamiento de texto
    en inferencia sea idéntico al de entrenamiento.
    """

    def __init__(self, artifact: _TextModelArtifact) -> None:
        """
        Construye el submodelo desde un artefacto ya validado.

        No usar directamente en producción. Usar `TextSubModel.load()`.
        """
        self._artifact = artifact
        self._pipeline: Pipeline = artifact.pipeline

        # Determinar el índice de la clase phishing (1) en el Pipeline.
        # LogisticRegression y SVC guardan las clases en .classes_.
        clf_step = self._pipeline.steps[-1][1]
        if not hasattr(clf_step, "classes_"):
            raise ValueError(
                f"El clasificador del Pipeline ({type(clf_step).__name__}) "
                "no tiene atributo '.classes_'. "
                "Asegúrate de que el Pipeline esté entrenado (fit)."
            )
        classes = list(clf_step.classes_)
        if 1 not in classes:
            raise ValueError(
                f"El clasificador no tiene la clase positiva (1) en: {classes}. "
                "Verifica que el modelo se entrenó con label=1 para phishing."
            )
        self._phish_idx: int = classes.index(1)
        self._clf_name: str = type(clf_step).__name__

        log.info(
            "TextSubModel cargado — clf=%s | schema=%s | trained_at=%s | "
            "vocab_size=%s | phish_class_idx=%d | metrics=%s",
            artifact.classifier_type,
            artifact.schema_version,
            artifact.trained_at,
            self._vocab_size(),
            self._phish_idx,
            artifact.train_metrics,
        )

    # ── Protocolo SubModel ──────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """
        Identificador único del modelo.

        Formato: "TfIdf_<clasificador>_text_v<schema>_<fecha>".
        """
        return (
            f"TfIdf_{self._artifact.classifier_type}_text"
            f"_v{self._artifact.schema_version}"
            f"_{self._artifact.trained_at}"
        )

    def predict_proba(self, features: dict[str, Any] | str) -> float:
        """
        Devuelve la probabilidad de phishing para un texto preparado.

        El Pipeline aplica internamente:
          1. TfidfVectorizer.transform([text])  -> sparse matrix (1, vocab)
          2. LogisticRegression.predict_proba() -> [[prob_0, prob_1]]

        Args:
            features: str con el texto preparado por
                      FeatureExtractor.extract_text_features().
                      Si se recibe un dict (uso incorrecto), devuelve 0.0.

        Returns:
            Probabilidad en [0.0, 1.0] de que el correo sea phishing.
        """
        if not isinstance(features, str):
            log.warning(
                "TextSubModel.predict_proba() recibió %s en lugar de str. "
                "Devolviendo 0.0.",
                type(features).__name__,
            )
            return 0.0

        if not features.strip():
            log.debug("Texto vacío recibido en TextSubModel. Devolviendo 0.0.")
            return 0.0

        # El Pipeline espera un iterable de strings.
        proba_matrix: np.ndarray = self._pipeline.predict_proba([features])
        phish_proba: float = float(proba_matrix[0, self._phish_idx])
        return float(min(max(phish_proba, 0.0), 1.0))

    # ── Serialización ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "TextSubModel":
        """
        Carga un TextSubModel desde un archivo .pkl serializado con joblib.

        Valida:
          1. El archivo existe y puede deserializarse.
          2. El contenido es un _TextModelArtifact.
          3. El schema_version coincide con el del código.

        Args:
            path: Ruta al archivo .pkl generado por train_text_model.py.

        Returns:
            Instancia de TextSubModel lista para inferencia.

        Raises:
            FileNotFoundError: si el archivo no existe.
            TypeError:         si el pkl no contiene un _TextModelArtifact.
            ValueError:        si hay incompatibilidad de schema.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Artefacto del modelo de texto no encontrado: {path}\n"
                "Ejecuta primero: python scripts/train_text_model.py"
            )

        log.info("Cargando TextSubModel desde: %s", path)

        try:
            artifact: _TextModelArtifact = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo deserializar el artefacto en {path}: {exc}"
            ) from exc

        if not isinstance(artifact, _TextModelArtifact):
            raise TypeError(
                f"El archivo {path} no contiene un _TextModelArtifact válido. "
                f"Tipo encontrado: {type(artifact)}. "
                "Regenera el artefacto con train_text_model.py."
            )

        if artifact.schema_version != _TextModelArtifact.SCHEMA_VERSION:
            raise ValueError(
                f"Incompatibilidad de schema: artefacto v{artifact.schema_version} "
                f"vs código v{_TextModelArtifact.SCHEMA_VERSION}. "
                "Reentrena el modelo con la versión actual del extractor."
            )

        return cls(artifact)

    @staticmethod
    def save(
        pipeline: Pipeline,
        path: Path,
        trained_at: str,
        train_metrics: dict[str, float],
        text_separator: str = " [SEP] ",
        max_text_length: int = 10_000,
    ) -> None:
        """
        Serializa el Pipeline entrenado como _TextModelArtifact.

        Llamado por train_text_model.py. No llamar directamente en inferencia.

        Args:
            pipeline:        Pipeline sklearn ya ajustado (TF-IDF + clf).
            path:            Destino del .pkl.
            trained_at:      Timestamp ISO 8601 del entrenamiento.
            train_metrics:   Métricas {precision, recall, f1, roc_auc}.
            text_separator:  Separador usado en extract_text_features().
            max_text_length: Truncamiento aplicado en extract_text_features().
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extraer params del vectorizador para registro en el artefacto.
        tfidf = pipeline.named_steps.get("tfidf") or pipeline.steps[0][1]
        vectorizer_params: dict[str, Any] = {
            "max_features": getattr(tfidf, "max_features", None),
            "ngram_range":  getattr(tfidf, "ngram_range", None),
            "sublinear_tf": getattr(tfidf, "sublinear_tf", None),
        }
        clf_step = pipeline.steps[-1][1]
        classifier_type = type(clf_step).__name__

        artifact = _TextModelArtifact(
            pipeline=pipeline,
            trained_at=trained_at,
            vectorizer_params=vectorizer_params,
            classifier_type=classifier_type,
            train_metrics=train_metrics,
            text_separator=text_separator,
            max_text_length=max_text_length,
        )

        # Escritura atómica: tmp → rename
        tmp_path = path.with_suffix(".tmp.pkl")
        try:
            joblib.dump(artifact, tmp_path, compress=3)
            tmp_path.rename(path)
            size_kb = path.stat().st_size / 1024
            log.info(
                "Artefacto de texto guardado: %s (%.1f KB) — "
                "clf=%s | schema=v%s | vocab=%s | metrics=%s",
                path, size_kb, classifier_type,
                _TextModelArtifact.SCHEMA_VERSION,
                vectorizer_params, train_metrics,
            )
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Error al guardar el artefacto: {exc}") from exc

    # ── Introspección ───────────────────────────────────────────────────────

    @property
    def train_metrics(self) -> dict[str, float]:
        """Métricas de validación registradas durante el entrenamiento."""
        return dict(self._artifact.train_metrics)

    @property
    def vectorizer_params(self) -> dict[str, Any]:
        """Parámetros del TfidfVectorizer usados en el entrenamiento."""
        return dict(self._artifact.vectorizer_params)

    def top_tokens_by_class(self, top_k: int = 20) -> dict[str, list[str]]:
        """
        Devuelve los tokens más discriminativos por clase.

        Usa los coeficientes del clasificador lineal (coef_) para identificar
        qué palabras/bigramas tienen más peso hacia phishing o legítimo.
        Útil para el análisis de interpretabilidad en la tesina.

        Args:
            top_k: Número de tokens a devolver por clase.

        Returns:
            Dict con "phishing" y "legitimate", cada uno con lista de tokens
            ordenados por peso descendente.
            Dict vacío si el clasificador no tiene coef_ (ej. RandomForest).
        """
        clf_step = self._pipeline.steps[-1][1]
        if not hasattr(clf_step, "coef_"):
            log.warning(
                "%s no tiene coef_ — top_tokens_by_class no disponible.",
                self._clf_name,
            )
            return {}

        tfidf_step = self._pipeline.steps[0][1]
        if not hasattr(tfidf_step, "get_feature_names_out"):
            return {}

        feature_names = tfidf_step.get_feature_names_out()
        # coef_ shape: (1, n_features) para clasificadores binarios
        coef = clf_step.coef_[0] if clf_step.coef_.ndim > 1 else clf_step.coef_

        top_phish_idx = np.argsort(coef)[-top_k:][::-1]
        top_legit_idx = np.argsort(coef)[:top_k]

        return {
            "phishing":   [str(feature_names[i]) for i in top_phish_idx],
            "legitimate": [str(feature_names[i]) for i in top_legit_idx],
        }

    def _vocab_size(self) -> int | str:
        """Tamaño del vocabulario aprendido por el TF-IDF."""
        try:
            tfidf = self._pipeline.steps[0][1]
            return len(tfidf.vocabulary_)
        except Exception:
            return "unknown"

    def __repr__(self) -> str:
        return (
            f"TextSubModel("
            f"clf={self._artifact.classifier_type}, "
            f"vocab_size={self._vocab_size()}, "
            f"schema=v{self._artifact.schema_version}, "
            f"trained_at={self._artifact.trained_at!r}, "
            f"metrics={self._artifact.train_metrics})"
        )