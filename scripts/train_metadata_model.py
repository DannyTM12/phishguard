#!/usr/bin/env python3
# scripts/train_metadata_model.py
"""
Hito 2 — Entrenamiento del Submodelo de Metadatos.

Lee los splits de train y val, extrae las 32 features técnicas con
`FeatureExtractor`, entrena un `RandomForestClassifier` calibrado para
recall alto, evalúa en validación y guarda el artefacto en
`artifacts/meta_model.pkl`.

Flujo
-----
  train.parquet  ──►  extracción de features (lotes)  ──►  fit RF
  val.parquet    ──►  extracción de features (lotes)  ──►  evaluación
                                                       ──►  artifacts/meta_model.pkl

Reproducibilidad
----------------
  • La semilla aleatoria del RF se pasa por --seed (default 42).
  • El orden de features es siempre el de `get_metadata_feature_names()`.
  • El script es idempotente: si el artefacto existe y no se pasa --force,
    sale sin reentrenar.

Uso
---
  python scripts/train_metadata_model.py
  python scripts/train_metadata_model.py --force
  python scripts/train_metadata_model.py --n-estimators 200 --max-depth 12
  python scripts/train_metadata_model.py --batch-size 2000 --seed 0
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phishguard.train_meta")

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "meta_model.pkl"

DEFAULT_TRAIN = PROCESSED_DIR / "train.parquet"
DEFAULT_VAL = PROCESSED_DIR / "val.parquet"

# ---------------------------------------------------------------------------
# Constantes de entrenamiento
# ---------------------------------------------------------------------------

# Parámetros conservadores para evitar sobreajuste en RF.
# Justificación tesina (Shahrivari et al., 2020; Maturure et al., 2024):
#   • max_depth=10     limita la profundidad para no memorizar patrones
#   • min_samples_leaf=5 evita splits en nodos con muy pocos ejemplos
#   • class_weight='balanced' compensa el desbalance de clases
#   • n_jobs=-1        usa todos los cores disponibles
DEFAULT_RF_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "oob_score": True,   # out-of-bag estimate gratis durante el entrenamiento
}

DEFAULT_BATCH_SIZE = 1000   # filas por lote en la extracción de features

# ---------------------------------------------------------------------------
# Tipos internos
# ---------------------------------------------------------------------------


class FeatureMatrix(NamedTuple):
    """Resultado de la extracción de features para un split."""
    X: np.ndarray      # (n_samples, 32), dtype float32
    y: np.ndarray      # (n_samples,),    dtype int8
    n_rows: int        # filas procesadas exitosamente
    n_errors: int      # filas descartadas por error en la extracción


class EvaluationMetrics(NamedTuple):
    """Métricas de evaluación en el conjunto de validación."""
    precision: float
    recall: float
    f1: float
    roc_auc: float
    oob_score: float | None   # None si el clf no calculó OOB


# ---------------------------------------------------------------------------
# Paso 1 — Carga de datos
# ---------------------------------------------------------------------------


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    """
    Lee un split Parquet y valida las columnas mínimas requeridas.

    Args:
        path:       Ruta al archivo .parquet.
        split_name: Nombre del split ('train' | 'val') para mensajes de error.

    Returns:
        DataFrame con columnas [subject, body, label] como mínimo.

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError:        si faltan columnas requeridas.
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

    # Asegurar tipos correctos
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    n_phish = int((df["label"] == 1).sum())
    n_legit = int((df["label"] == 0).sum())
    log.info(
        "  %s: %d registros — phishing=%d (%.1f%%) | legítimo=%d (%.1f%%)",
        split_name,
        len(df),
        n_phish, 100 * n_phish / max(len(df), 1),
        n_legit, 100 * n_legit / max(len(df), 1),
    )
    return df


# ---------------------------------------------------------------------------
# Paso 2 — Extracción de features en lotes
# ---------------------------------------------------------------------------


def extract_features_batched(
    df: pd.DataFrame,
    batch_size: int,
    split_name: str,
) -> FeatureMatrix:
    """
    Extrae las 32 features de metadatos para cada fila del DataFrame.

    Procesa los datos en lotes para poder reportar el progreso mediante
    logging sin necesidad de tqdm u otras dependencias externas.

    Diseño anti-leakage: el FeatureExtractor no tiene estado interno que
    se ajuste a los datos (no hay fit). Es un extractor puro y determinista,
    por lo que aplicarlo en train y val de forma independiente es seguro.

    Args:
        df:         DataFrame con columnas [subject, body, label].
        batch_size: Número de filas a procesar por lote.
        split_name: Nombre del split para mensajes de progreso.

    Returns:
        FeatureMatrix con X (n_samples, 32), y (n_samples,) y conteos.
    """
    # Importar aquí para mantener el script ejecutable sin el paquete en PYTHONPATH
    # cuando se usa `python scripts/train_metadata_model.py` directamente.
    # El sys.path se modifica al inicio de main().
    from phishguard.features.extractor import FeatureExtractor, get_metadata_feature_names

    extractor = FeatureExtractor()
    feature_names = get_metadata_feature_names()
    n_features = len(feature_names)

    total = len(df)
    n_batches = (total + batch_size - 1) // batch_size

    # Pre-alocar arrays para eficiencia de memoria
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    n_errors = 0

    t_start = time.perf_counter()
    log.info(
        "Extrayendo features de %s: %d filas en %d lotes de %d",
        split_name, total, n_batches, batch_size,
    )

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        batch_vectors: list[list[float]] = []
        batch_labels: list[int] = []

        for _, row in batch.iterrows():
            try:
                urls = extractor.get_urls_from_body(row["body"])
                feat_dict = extractor.extract_metadata_features(
                    subject=row["subject"],
                    body=row["body"],
                    urls=urls,
                )

                # Construir vector con orden canónico
                vector = _feat_dict_to_row(feat_dict, feature_names)
                batch_vectors.append(vector)
                batch_labels.append(int(row["label"]))

            except Exception as exc:
                n_errors += 1
                log.debug(
                    "Error extrayendo features en lote %d, fila %d: %s",
                    batch_idx, _, exc,
                )
                # Continuar — no abortar por una fila problemática

        if batch_vectors:
            X_list.append(np.array(batch_vectors, dtype=np.float32))
            y_list.extend(batch_labels)

        # Progreso cada 10 lotes o en el último
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            processed = min((batch_idx + 1) * batch_size, total)
            elapsed = time.perf_counter() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            log.info(
                "  [%s] %d/%d filas (%.0f%%) — %.0f filas/s — ETA: %.0fs",
                split_name, processed, total,
                100 * processed / total, rate, eta,
            )

    elapsed_total = time.perf_counter() - t_start

    if not X_list:
        raise RuntimeError(
            f"No se pudo extraer ningún feature del split '{split_name}'. "
            "Revisa los datos de entrada."
        )

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int8)

    log.info(
        "  [%s] Extracción completada: X=%s | y=%s | errors=%d | tiempo=%.1fs",
        split_name, X.shape, y.shape, n_errors, elapsed_total,
    )

    if n_errors > 0:
        log.warning(
            "  [%s] %d filas descartadas por errores en la extracción (%.1f%% del total).",
            split_name, n_errors, 100 * n_errors / total,
        )

    return FeatureMatrix(X=X, y=y, n_rows=len(y), n_errors=n_errors)


def _feat_dict_to_row(
    feat_dict: dict[str, float | int | bool],
    feature_names: list[str],
) -> list[float]:
    """
    Convierte un dict de features a una lista ordenada de floats.

    Función separada de `_dict_to_array` de meta_submodel.py para evitar
    la sobrecarga de crear un array numpy por cada fila durante el
    entrenamiento (es más eficiente acumular listas y hacer vstack al final).
    """
    row: list[float] = []
    for name in feature_names:
        value = feat_dict.get(name, 0)
        if isinstance(value, bool):
            row.append(1.0 if value else 0.0)
        else:
            row.append(float(value))
    return row


# ---------------------------------------------------------------------------
# Paso 3 — Entrenamiento
# ---------------------------------------------------------------------------


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rf_params: dict,
) -> RandomForestClassifier:
    """
    Ajusta un RandomForestClassifier con los parámetros dados.

    Args:
        X_train:   Array (n_samples, 32) de features de entrenamiento.
        y_train:   Array (n_samples,) de etiquetas (0=legit, 1=phishing).
        rf_params: Diccionario de hiperparámetros para RandomForest.

    Returns:
        Clasificador entrenado.
    """
    log.info("─── Entrenamiento RandomForest ─────────────────────")
    log.info("  Parámetros: %s", rf_params)
    log.info("  X_train: %s | y_train: %s", X_train.shape, y_train.shape)
    log.info(
        "  Distribución train — phishing=%d | legítimo=%d",
        int((y_train == 1).sum()),
        int((y_train == 0).sum()),
    )

    clf = RandomForestClassifier(**rf_params)

    t_start = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t_start

    log.info("  Entrenamiento completado en %.1fs", elapsed)

    if hasattr(clf, "oob_score_"):
        log.info("  OOB score (estimación interna): %.4f", clf.oob_score_)

    return clf


# ---------------------------------------------------------------------------
# Paso 4 — Evaluación en validación
# ---------------------------------------------------------------------------


def evaluate(
    clf: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> EvaluationMetrics:
    """
    Evalúa el clasificador en el conjunto de validación.

    Métricas reportadas (justificación en la tesina):
      • Recall:    métrica primaria — minimizar falsos negativos es
                   más crítico que minimizar falsos positivos en phishing.
      • Precision: complementaria — controla el ratio de FP.
      • F1:        balance entre precision y recall.
      • ROC-AUC:   robustez frente al umbral de decisión.
      • OOB score: estimación interna del RF (bonus, gratis).

    Args:
        clf:   Clasificador sklearn ya entrenado.
        X_val: Array (n_val, 32) de features de validación.
        y_val: Array (n_val,)  de etiquetas de validación.

    Returns:
        EvaluationMetrics con todos los valores calculados.
    """
    log.info("─── Evaluación en validación ───────────────────────")

    y_pred = clf.predict(X_val)
    # Si label=1 (phishing) no está en clf.classes_ (split sin phishing),
    # usamos el índice de la clase disponible para y_proba.
    classes = list(clf.classes_)
    proba_idx = classes.index(1) if 1 in classes else 0
    y_proba = clf.predict_proba(X_val)[:, proba_idx]

    precision = float(precision_score(y_val, y_pred, zero_division=0))
    recall = float(recall_score(y_val, y_pred, zero_division=0))
    f1 = float(f1_score(y_val, y_pred, zero_division=0))
    oob = float(clf.oob_score_) if hasattr(clf, "oob_score_") else None

    # roc_auc_score requiere al menos 2 clases en y_val
    unique_classes = np.unique(y_val)
    if len(unique_classes) < 2:
        log.warning(
            "  y_val tiene una sola clase %s — ROC-AUC no definido.",
            unique_classes.tolist(),
        )
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
    if oob is not None:
        log.info("  │  OOB Score : %.4f", oob)
    log.info("  └────────────────────────────────────────────")

    # Reporte por clase — labels explícitos para evitar ValueError
    # cuando y_val no tiene todas las clases (split desbalanceado/pequeño)
    present_labels = sorted(unique_classes.tolist())
    present_names = [
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

    # Alerta si el recall está por debajo del objetivo de la tesina
    recall_target = 0.95
    if recall < recall_target:
        log.warning(
            "⚠️  Recall=%.4f está por debajo del objetivo de la tesina (≥%.2f). "
            "Considera: ajustar class_weight, bajar el umbral de decisión en "
            "config.yaml, o usar más datos de entrenamiento.",
            recall, recall_target,
        )
    else:
        log.info(
            "✓ Recall=%.4f cumple el objetivo de la tesina (≥%.2f).",
            recall, recall_target,
        )

    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        oob_score=oob,
    )


# ---------------------------------------------------------------------------
# Paso 5 — Log de importancia de features
# ---------------------------------------------------------------------------


def log_feature_importances(
    clf: RandomForestClassifier,
    top_k: int = 15,
) -> None:
    """
    Registra las features más importantes según el Random Forest.

    Útil para validar que el modelo usa las señales esperadas
    y como punto de partida para el análisis SHAP del Hito 3.
    """
    from phishguard.features.extractor import get_metadata_feature_names

    feature_names = get_metadata_feature_names()
    importances = clf.feature_importances_

    paired = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    log.info("─── Top %d features por importancia ────────────────", top_k)
    for i, (name, imp) in enumerate(paired[:top_k], 1):
        bar = "█" * int(imp * 200)   # barra visual proporcional
        log.info("  %2d. %-30s  %.4f  %s", i, name, imp, bar)


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
    rf_params: dict,
    batch_size: int,
    force: bool,
) -> None:
    """
    Orquesta el pipeline completo de entrenamiento del submodelo de metadatos.

    Pasos:
      1. Verificar idempotencia.
      2. Cargar splits train y val.
      3. Extraer features en lotes para ambos splits.
      4. Entrenar RandomForestClassifier.
      5. Evaluar en validación.
      6. Registrar importancia de features.
      7. Guardar artefacto .pkl.

    Args:
        train_path:  Ruta a train.parquet.
        val_path:    Ruta a val.parquet.
        output_path: Ruta de destino para meta_model.pkl.
        rf_params:   Hiperparámetros del RandomForest.
        batch_size:  Tamaño del lote para la extracción de features.
        force:       Si True, reentrenar aunque el artefacto ya exista.
    """
    # ── Idempotencia ────────────────────────────────────────────────────────
    if not force and artifact_exists(output_path):
        log.info(
            "El artefacto ya existe: %s\n"
            "  → Usa --force para reentrenar. Saliendo sin cambios.",
            output_path,
        )
        return

    t_pipeline_start = time.perf_counter()
    log.info("═══ Iniciando entrenamiento del Submodelo de Metadatos ═══")

    # ── Carga ────────────────────────────────────────────────────────────────
    df_train = load_split(train_path, "train")
    df_val = load_split(val_path, "val")

    # ── Extracción de features ───────────────────────────────────────────────
    log.info("─── Extracción de features — train ─────────────────")
    train_matrix = extract_features_batched(df_train, batch_size, "train")

    log.info("─── Extracción de features — val ───────────────────")
    val_matrix = extract_features_batched(df_val, batch_size, "val")

    # ── Entrenamiento ────────────────────────────────────────────────────────
    clf = train_random_forest(
        X_train=train_matrix.X,
        y_train=train_matrix.y,
        rf_params=rf_params,
    )

    # ── Evaluación ───────────────────────────────────────────────────────────
    metrics = evaluate(clf, val_matrix.X, val_matrix.y)

    # ── Importancia de features ──────────────────────────────────────────────
    log_feature_importances(clf, top_k=15)

    # ── Guardar artefacto ────────────────────────────────────────────────────
    from phishguard.models.meta_submodel import MetaSubModel

    trained_at = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    MetaSubModel.save(
        classifier=clf,
        path=output_path,
        trained_at=trained_at,
        train_metrics={
            "precision": round(metrics.precision, 4),
            "recall":    round(metrics.recall, 4),
            "f1":        round(metrics.f1, 4),
            "roc_auc":   round(metrics.roc_auc, 4),
        },
    )

    # ── Verificar que el artefacto se puede cargar y usar ────────────────────
    log.info("─── Smoke test del artefacto guardado ──────────────")
    loaded = MetaSubModel.load(output_path)
    dummy_features = {name: 0.0 for name in __import__(
        "phishguard.features.extractor", fromlist=["get_metadata_feature_names"]
    ).get_metadata_feature_names()}
    test_score = loaded.predict_proba(dummy_features)
    assert 0.0 <= test_score <= 1.0, f"Smoke test fallido: score={test_score}"
    log.info("  ✓ Artefacto cargado y predict_proba() funciona (score=%.4f)", test_score)

    # ── Resumen final ────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_pipeline_start
    log.info("═══════════════════════════════════════════════════════")
    log.info("ENTRENAMIENTO COMPLETADO en %.1fs", elapsed_total)
    log.info("  Artefacto: %s", output_path)
    log.info("  Precision: %.4f", metrics.precision)
    log.info("  Recall:    %.4f  ← objetivo ≥ 0.95", metrics.recall)
    log.info("  F1-Score:  %.4f", metrics.f1)
    log.info("  ROC-AUC:   %.4f", metrics.roc_auc)
    log.info("  Modelo:    %s", loaded.model_name)
    log.info("═══════════════════════════════════════════════════════")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhishGuard Hito 2 — Entrenamiento del Submodelo de Metadatos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train",    type=Path, default=DEFAULT_TRAIN,
                        help=f"Ruta a train.parquet. Default: {DEFAULT_TRAIN}")
    parser.add_argument("--val",      type=Path, default=DEFAULT_VAL,
                        help=f"Ruta a val.parquet. Default: {DEFAULT_VAL}")
    parser.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT,
                        help=f"Destino del artefacto .pkl. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Número de árboles en el RF. Default: 100")
    parser.add_argument("--max-depth",    type=int, default=10,
                        help="Profundidad máxima de los árboles. Default: 10")
    parser.add_argument("--min-samples-leaf", type=int, default=5,
                        help="Mínimo de muestras por hoja. Default: 5")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad. Default: 42")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Filas por lote en la extracción. Default: {DEFAULT_BATCH_SIZE}")
    parser.add_argument("--force",    action="store_true",
                        help="Reentrenar aunque el artefacto ya exista.")
    parser.add_argument("--verbose",  action="store_true",
                        help="Activar logging DEBUG.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Asegurar que el paquete phishguard sea importable desde el repo
    repo_src = _REPO_ROOT / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    # Construir parámetros del RF desde los args
    rf_params = {
        **DEFAULT_RF_PARAMS,
        "n_estimators":     args.n_estimators,
        "max_depth":        args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state":     args.seed,
    }

    try:
        run_training(
            train_path=args.train,
            val_path=args.val,
            output_path=args.output,
            rf_params=rf_params,
            batch_size=args.batch_size,
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