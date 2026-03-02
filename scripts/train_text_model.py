#!/usr/bin/env python3
# scripts/train_text_model.py
"""
Hito 2 — Entrenamiento del Submodelo de Texto (NLP).

Lee train.parquet y val.parquet, extrae el texto preparado con
FeatureExtractor.extract_text_features(), entrena un Pipeline
TF-IDF + LogisticRegression y guarda el artefacto en
artifacts/text_model.pkl.

Flujo
-----
  train.parquet  →  extract_text_features (vectorizado pandas)  →  Pipeline.fit()
  val.parquet    →  extract_text_features                        →  evaluación
                                                                 →  artifacts/text_model.pkl

Ventaja del Pipeline completo
------------------------------
Guardar el Pipeline (TF-IDF + LR) en lugar de solo el clasificador
garantiza que en inferencia el texto pasa exactamente la misma
transformación que en entrenamiento: no hay vectorizador separado
ni riesgo de desincronización del vocabulario.

Elección de LogisticRegression
--------------------------------
Frente a SVM, LR ofrece predict_proba() calibrada de forma nativa,
que es lo que necesita el motor de fusión tardía. Además es la opción
más interpretable (coef_ → tokens más importantes) y rápida de
entrenar para el volumen de datos del proyecto.

Uso
---
  python scripts/train_text_model.py
  python scripts/train_text_model.py --force
  python scripts/train_text_model.py --max-features 20000 --ngram-max 2
  python scripts/train_text_model.py --c 0.5 --solver saga
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phishguard.train_text")

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "text_model.pkl"

DEFAULT_TRAIN = PROCESSED_DIR / "train.parquet"
DEFAULT_VAL   = PROCESSED_DIR / "val.parquet"

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------

# TF-IDF
DEFAULT_MAX_FEATURES: int   = 10_000
DEFAULT_NGRAM_MIN:    int   = 1
DEFAULT_NGRAM_MAX:    int   = 2
DEFAULT_SUBLINEAR_TF: bool  = True   # log(tf) + 1: atenúa términos muy frecuentes
DEFAULT_MIN_DF:       int   = 2      # ignorar tokens que aparecen en < 2 docs

# LogisticRegression
DEFAULT_C:          float = 1.0     # regularización inversa (más bajo = más regularización)
DEFAULT_SOLVER:     str   = "saga"  # saga: eficiente para corpus grandes y L1/L2
DEFAULT_MAX_ITER:   int   = 1_000
DEFAULT_PENALTY:    str   = "l2"

# ---------------------------------------------------------------------------
# Tipos internos
# ---------------------------------------------------------------------------


class TextCorpus(NamedTuple):
    """Corpus de texto extraído de un split."""
    texts:  list[str]    # Textos preparados por extract_text_features()
    labels: list[int]    # Etiquetas correspondientes (0/1)
    n_rows: int          # Filas procesadas
    n_errors: int        # Filas descartadas por error


class EvaluationMetrics(NamedTuple):
    """Métricas de evaluación en validación."""
    precision: float
    recall:    float
    f1:        float
    roc_auc:   float


# ---------------------------------------------------------------------------
# Paso 1 — Carga de datos
# ---------------------------------------------------------------------------


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """
    Lee un split Parquet y valida las columnas mínimas requeridas.

    Args:
        path:       Ruta al archivo .parquet.
        split_name: Nombre del split para mensajes de error.

    Returns:
        DataFrame con columnas [subject, body, label].
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Split '{split_name}' no encontrado: {path}\n"
            "Ejecuta primero: python scripts/make_splits.py"
        )

    log.info("Leyendo %s: %s", split_name, path)
    df = pd.read_parquet(path)

    required = {"subject", "body", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"El split '{split_name}' no tiene las columnas requeridas: {missing}"
        )

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"]    = df["body"].fillna("").astype(str)
    df["label"]   = df["label"].astype(int)

    n_phish = int((df["label"] == 1).sum())
    n_legit = int((df["label"] == 0).sum())
    log.info(
        "  %s: %d registros — phishing=%d (%.1f%%) | legítimo=%d (%.1f%%)",
        split_name, len(df),
        n_phish, 100 * n_phish / max(len(df), 1),
        n_legit, 100 * n_legit / max(len(df), 1),
    )
    return df


# ---------------------------------------------------------------------------
# Paso 2 — Extracción de texto
# ---------------------------------------------------------------------------


def extract_text_corpus(df: pd.DataFrame, split_name: str) -> TextCorpus:
    """
    Aplica FeatureExtractor.extract_text_features() a todo el DataFrame.

    A diferencia del submodelo de metadatos (que necesita URLs y usa
    extracción por filas), la extracción de texto es una operación
    vectorizada pura con pandas: no hay red, no hay parsing HTML complejo,
    solo llamadas a funciones de string.

    Se usa pandas.apply() para mantener el código claro y correcto.
    Para corpus muy grandes (>1M filas), se puede paralelizar con
    multiprocessing o Dask, pero para el volumen del proyecto esto
    es innecesario.

    Args:
        df:         DataFrame con columnas [subject, body, label].
        split_name: Nombre del split para mensajes de progreso.

    Returns:
        TextCorpus con listas de textos y etiquetas.
    """
    from phishguard.features.extractor import FeatureExtractor

    extractor = FeatureExtractor()
    total = len(df)
    log.info("Extrayendo texto de %s: %d registros...", split_name, total)

    texts:    list[str] = []
    labels:   list[int] = []
    n_errors: int       = 0

    t_start = time.perf_counter()

    # Reportar progreso en bloques de 20% para corpus grandes.
    report_step = max(total // 5, 1)

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            text = extractor.extract_text_features(
                subject=row["subject"],
                body=row["body"],
            )
            texts.append(text)
            labels.append(int(row["label"]))
        except Exception as exc:
            n_errors += 1
            log.debug("Error en fila %d: %s", i, exc)

        if (i + 1) % report_step == 0 or i == total - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta  = (total - i - 1) / rate if rate > 0 else 0
            log.info(
                "  [%s] %d/%d (%.0f%%) — %.0f docs/s — ETA: %.0fs",
                split_name, i + 1, total,
                100 * (i + 1) / total, rate, eta,
            )

    elapsed_total = time.perf_counter() - t_start

    if not texts:
        raise RuntimeError(
            f"No se pudo extraer ningún texto del split '{split_name}'."
        )

    # Advertir sobre documentos vacíos (tras clean_text + strip)
    n_empty = sum(1 for t in texts if not t.strip())
    if n_empty:
        log.warning(
            "  [%s] %d documentos vacíos tras la extracción (%.1f%%). "
            "TF-IDF los tratará como vectores cero.",
            split_name, n_empty, 100 * n_empty / len(texts),
        )

    log.info(
        "  [%s] Extracción completada: %d textos | %d errores | %.1fs | "
        "longitud media: %.0f chars",
        split_name, len(texts), n_errors, elapsed_total,
        sum(len(t) for t in texts) / max(len(texts), 1),
    )

    return TextCorpus(
        texts=texts,
        labels=labels,
        n_rows=len(texts),
        n_errors=n_errors,
    )


# ---------------------------------------------------------------------------
# Paso 3 — Construcción y entrenamiento del Pipeline
# ---------------------------------------------------------------------------


def build_pipeline(
    max_features: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
    min_df: int,
    c: float,
    solver: str,
    max_iter: int,
    penalty: str,
    random_state: int,
) -> Pipeline:
    """
    Construye el Pipeline TF-IDF + LogisticRegression.

    Justificación de cada parámetro (para la tesina):

    TfidfVectorizer:
      • max_features=10000:  limita el vocabulario a los tokens más frecuentes,
                             controlando la dimensionalidad y el tiempo de inferencia.
      • ngram_range=(1,2):   incluye bigramas ("click here", "verify account")
                             que son señales clave en phishing no capturables con unigramas.
      • sublinear_tf=True:   log(tf)+1 atenúa el efecto de repetición masiva de
                             tokens (táctica de spam: repetir palabras clave 100 veces).
      • min_df=2:            ignora hapax legomena (tokens únicos), que son ruido.

    LogisticRegression:
      • class_weight='balanced': compensa el desbalance de clases calculando
                                 automáticamente pesos inversamente proporcionales
                                 a la frecuencia de cada clase en el train set.
      • solver='saga':         eficiente para corpus grandes con regularización L1/L2.
      • C=1.0:                 regularización moderada; ajustar con grid search en Hito 3.
      • max_iter=1000:         suficiente para convergencia incluso con corpus grandes.

    Args:
        (ver signaturas de los parámetros, mapeados desde la CLI)

    Returns:
        Pipeline sklearn sin entrenar, listo para .fit().
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",  # tokens de al menos 2 chars
        dtype=np.float32,
    )

    # penalty= y n_jobs= están deprecados en sklearn >= 1.8.
    # La regularización L2 es el comportamiento por defecto de LR;
    # para L1 o ElasticNet pasar l1_ratio explícitamente (Hito 3).
    lr_kwargs: dict = {
        "C":            c,
        "solver":       solver,
        "max_iter":     max_iter,
        "class_weight": "balanced",
        "random_state": random_state,
    }
    # Solo pasar penalty si el usuario eligió algo distinto de l2 (default).
    # Nota: sklearn >= 1.8 prefiere omitir penalty y usar l1_ratio en su lugar,
    # pero la CLI lo expone por compatibilidad con versiones anteriores.
    if penalty and penalty not in ("l2", "none"):
        lr_kwargs["penalty"] = penalty
    clf = LogisticRegression(**lr_kwargs)

    return Pipeline(steps=[("tfidf", tfidf), ("clf", clf)])


def train_pipeline(
    pipeline: Pipeline,
    corpus: TextCorpus,
) -> Pipeline:
    """
    Ajusta el Pipeline sobre el corpus de entrenamiento.

    Args:
        pipeline: Pipeline sin entrenar (de build_pipeline()).
        corpus:   TextCorpus con textos y etiquetas de train.

    Returns:
        Pipeline ajustado.
    """
    log.info("─── Entrenamiento Pipeline TF-IDF + LR ─────────────")
    log.info(
        "  Documentos: %d | phishing=%d | legítimo=%d",
        corpus.n_rows,
        sum(1 for l in corpus.labels if l == 1),
        sum(1 for l in corpus.labels if l == 0),
    )

    tfidf_params = pipeline.named_steps["tfidf"].get_params()
    clf_params   = pipeline.named_steps["clf"].get_params()
    log.info("  TF-IDF params: %s", {k: v for k, v in tfidf_params.items()
                                     if k in ("max_features","ngram_range","sublinear_tf","min_df")})
    log.info("  LR params    : %s", {k: v for k, v in clf_params.items()
                                     if k in ("C","solver","penalty","class_weight","max_iter")})

    t_start = time.perf_counter()
    pipeline.fit(corpus.texts, corpus.labels)
    elapsed = time.perf_counter() - t_start

    # Tamaño real del vocabulario aprendido
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    log.info(
        "  Entrenamiento completado en %.1fs | vocabulario: %d tokens",
        elapsed, vocab_size,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Paso 4 — Evaluación
# ---------------------------------------------------------------------------


def evaluate(
    pipeline: Pipeline,
    corpus: TextCorpus,
) -> EvaluationMetrics:
    """
    Evalúa el Pipeline en el corpus de validación.

    Misma lógica de métricas que en el submodelo de metadatos:
    Recall es la métrica prioritaria (objetivo >= 0.95).

    Args:
        pipeline: Pipeline ya entrenado.
        corpus:   TextCorpus con textos y etiquetas de val.

    Returns:
        EvaluationMetrics con precision, recall, f1, roc_auc.
    """
    log.info("─── Evaluación en validación ───────────────────────")

    y_val  = np.array(corpus.labels)
    y_pred = pipeline.predict(corpus.texts)

    clf_step = pipeline.steps[-1][1]
    classes  = list(clf_step.classes_)
    phish_idx = classes.index(1) if 1 in classes else 0
    y_proba   = pipeline.predict_proba(corpus.texts)[:, phish_idx]

    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall    = float(recall_score(y_val, y_pred, zero_division=0))
    f1        = float(f1_score(y_val, y_pred, zero_division=0))

    unique_classes = np.unique(y_val)
    if len(unique_classes) < 2:
        log.warning("  y_val tiene una sola clase — ROC-AUC no definido.")
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_val, y_proba))

    log.info("  ┌────────────────────────────────────────────")
    log.info("  │  Precision : %.4f", precision)
    log.info("  │  Recall    : %.4f  ← métrica prioritaria", recall)
    log.info("  │  F1-Score  : %.4f", f1)
    if roc_auc == roc_auc:  # NaN check
        log.info("  │  ROC-AUC   : %.4f", roc_auc)
    else:
        log.info("  │  ROC-AUC   : N/A (una sola clase en val)")
    log.info("  └────────────────────────────────────────────")

    present_labels = sorted(unique_classes.tolist())
    present_names  = [
        f"{'phishing' if lbl == 1 else 'legitimate'} ({lbl})"
        for lbl in present_labels
    ]
    report = classification_report(
        y_val, y_pred,
        labels=present_labels,
        target_names=present_names,
        digits=4,
        zero_division=0,
    )
    log.info("\n%s", report)

    recall_target = 0.95
    if recall < recall_target:
        log.warning(
            "⚠  Recall=%.4f por debajo del objetivo (>= %.2f). "
            "Prueba: reducir C, aumentar max_features, o usar ngram_range=(1,3).",
            recall, recall_target,
        )
    else:
        log.info("✓  Recall=%.4f cumple el objetivo (>= %.2f).", recall, recall_target)

    return EvaluationMetrics(precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)


# ---------------------------------------------------------------------------
# Paso 5 — Log de tokens más discriminativos
# ---------------------------------------------------------------------------


def log_top_tokens(pipeline: Pipeline, top_k: int = 15) -> None:
    """
    Registra los tokens con mayor peso hacia phishing y hacia legítimo.

    Equivalente al log de feature importances del submodelo de metadatos,
    pero para el espacio léxico. Útil para la interpretabilidad de la tesina
    y como punto de partida para análisis LIME/SHAP en el Hito 3.
    """
    clf_step = pipeline.steps[-1][1]
    if not hasattr(clf_step, "coef_"):
        log.info("  (clasificador sin coef_ — tokens no disponibles)")
        return

    tfidf_step = pipeline.steps[0][1]
    feature_names = tfidf_step.get_feature_names_out()
    coef = clf_step.coef_[0] if clf_step.coef_.ndim > 1 else clf_step.coef_

    top_phish_idx = np.argsort(coef)[-top_k:][::-1]
    top_legit_idx = np.argsort(coef)[:top_k]

    log.info("─── Top %d tokens → PHISHING ────────────────────────", top_k)
    for i, idx in enumerate(top_phish_idx, 1):
        bar = "█" * min(int(coef[idx] * 20), 40)
        log.info("  %2d. %-30s  %+.4f  %s", i, feature_names[idx], coef[idx], bar)

    log.info("─── Top %d tokens → LEGÍTIMO ────────────────────────", top_k)
    for i, idx in enumerate(top_legit_idx, 1):
        bar = "█" * min(int(abs(coef[idx]) * 20), 40)
        log.info("  %2d. %-30s  %+.4f  %s", i, feature_names[idx], coef[idx], bar)


# ---------------------------------------------------------------------------
# Idempotencia
# ---------------------------------------------------------------------------


def artifact_exists(output_path: Path) -> bool:
    """True si el artefacto ya existe en disco."""
    return output_path.exists()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def run_training(
    train_path: Path,
    val_path: Path,
    output_path: Path,
    pipeline_params: dict,
    force: bool,
) -> None:
    """
    Orquesta el pipeline completo de entrenamiento del submodelo de texto.

    Pasos:
      1. Verificar idempotencia.
      2. Cargar splits.
      3. Extraer texto preparado para train y val.
      4. Construir y entrenar el Pipeline.
      5. Evaluar en validación.
      6. Log de tokens discriminativos.
      7. Guardar artefacto.
      8. Smoke test.
    """
    if not force and artifact_exists(output_path):
        log.info(
            "El artefacto ya existe: %s\n"
            "  → Usa --force para reentrenar.",
            output_path,
        )
        return

    t_start = time.perf_counter()
    log.info("═══ Iniciando entrenamiento del Submodelo de Texto ═══")

    # ── Carga ────────────────────────────────────────────────────────────────
    df_train = load_split(train_path, "train")
    df_val   = load_split(val_path,   "val")

    # ── Extracción de texto ──────────────────────────────────────────────────
    log.info("─── Extracción de texto — train ─────────────────────")
    train_corpus = extract_text_corpus(df_train, "train")

    log.info("─── Extracción de texto — val ───────────────────────")
    val_corpus = extract_text_corpus(df_val, "val")

    # ── Construcción y entrenamiento del Pipeline ────────────────────────────
    pipeline = build_pipeline(**pipeline_params)
    trained_pipeline = train_pipeline(pipeline, train_corpus)

    # ── Evaluación ───────────────────────────────────────────────────────────
    metrics = evaluate(trained_pipeline, val_corpus)

    # ── Log de tokens más discriminativos ────────────────────────────────────
    log_top_tokens(trained_pipeline, top_k=15)

    # ── Guardar artefacto ────────────────────────────────────────────────────
    from phishguard.models.text_submodel import TextSubModel

    trained_at = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    TextSubModel.save(
        pipeline=trained_pipeline,
        path=output_path,
        trained_at=trained_at,
        train_metrics={
            "precision": round(metrics.precision, 4),
            "recall":    round(metrics.recall,    4),
            "f1":        round(metrics.f1,         4),
            "roc_auc":   round(metrics.roc_auc,    4)
                        if metrics.roc_auc == metrics.roc_auc else 0.0,
        },
    )

    # ── Smoke test ────────────────────────────────────────────────────────────
    log.info("─── Smoke test del artefacto guardado ──────────────")
    loaded = TextSubModel.load(output_path)
    test_score = loaded.predict_proba(
        "URGENT verify your PayPal account immediately or it will be suspended"
    )
    assert 0.0 <= test_score <= 1.0, f"Smoke test fallido: score={test_score}"
    log.info("  ✓ predict_proba() funciona (score=%.4f)", test_score)

    # Tokens top para confirmar que aprendió algo razonable
    top_tokens = loaded.top_tokens_by_class(top_k=5)
    if top_tokens:
        log.info("  Top 5 tokens phishing: %s", top_tokens.get("phishing", []))
        log.info("  Top 5 tokens legítimo: %s", top_tokens.get("legitimate", []))

    # ── Resumen final ────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start
    log.info("═══════════════════════════════════════════════════════")
    log.info("ENTRENAMIENTO COMPLETADO en %.1fs", elapsed_total)
    log.info("  Artefacto : %s", output_path)
    log.info("  Precision : %.4f", metrics.precision)
    log.info("  Recall    : %.4f  ← objetivo >= 0.95", metrics.recall)
    log.info("  F1-Score  : %.4f", metrics.f1)
    if metrics.roc_auc == metrics.roc_auc:
        log.info("  ROC-AUC   : %.4f", metrics.roc_auc)
    log.info("  Modelo    : %s", loaded.model_name)
    log.info("═══════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhishGuard Hito 2 — Entrenamiento del Submodelo de Texto.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train",       type=Path,  default=DEFAULT_TRAIN)
    parser.add_argument("--val",         type=Path,  default=DEFAULT_VAL)
    parser.add_argument("--output",      type=Path,  default=DEFAULT_OUTPUT)
    parser.add_argument("--max-features",type=int,   default=DEFAULT_MAX_FEATURES,
                        help="Tamaño máximo del vocabulario TF-IDF. Default: 10000")
    parser.add_argument("--ngram-min",   type=int,   default=DEFAULT_NGRAM_MIN)
    parser.add_argument("--ngram-max",   type=int,   default=DEFAULT_NGRAM_MAX,
                        help="Máximo de n en n-gramas. Default: 2 (unigramas + bigramas)")
    parser.add_argument("--sublinear-tf",action="store_true", default=DEFAULT_SUBLINEAR_TF,
                        help="Usar log(tf)+1 en lugar de tf. Default: True")
    parser.add_argument("--min-df",      type=int,   default=DEFAULT_MIN_DF,
                        help="Mínimo de documentos para incluir un token. Default: 2")
    parser.add_argument("--c",           type=float, default=DEFAULT_C,
                        help="Regularización inversa de LR. Default: 1.0")
    parser.add_argument("--solver",      type=str,   default=DEFAULT_SOLVER,
                        choices=["saga","lbfgs","liblinear","newton-cg","sag"])
    parser.add_argument("--max-iter",    type=int,   default=DEFAULT_MAX_ITER)
    parser.add_argument("--penalty",     type=str,   default=DEFAULT_PENALTY,
                        choices=["l1","l2","elasticnet","none"])
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--force",       action="store_true",
                        help="Reentrenar aunque el artefacto ya exista.")
    parser.add_argument("--verbose",     action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    repo_src = _REPO_ROOT / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    pipeline_params = {
        "max_features":  args.max_features,
        "ngram_range":   (args.ngram_min, args.ngram_max),
        "sublinear_tf":  args.sublinear_tf,
        "min_df":        args.min_df,
        "c":             args.c,
        "solver":        args.solver,
        "max_iter":      args.max_iter,
        "penalty":       args.penalty,
        "random_state":  args.seed,
    }

    try:
        run_training(
            train_path=args.train,
            val_path=args.val,
            output_path=args.output,
            pipeline_params=pipeline_params,
            force=args.force,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        log.error("Error en el pipeline: %s", exc)
        sys.exit(1)
    except AssertionError as exc:
        log.error("Smoke test fallido: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()