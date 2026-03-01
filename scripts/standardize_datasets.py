#!/usr/bin/env python3
# scripts/standardize_datasets.py
"""
Hito 1 — Estandarización de datasets crudos.

Lee los 7 CSV de data/raw/, unifica el esquema y escribe
data/processed/unified_dataset.parquet (idempotente).

Esquema de salida
-----------------
  dataset_source : str   — nombre del CSV de origen (sin extensión)
  subject        : str   — asunto del correo (vacío si no existe)
  body           : str   — cuerpo del correo (vacío si no existe)
  label          : int8  — 1 = phishing/spam, 0 = legítimo
  content_hash   : str   — MD5 de (subject + body) para deduplicación

Ejecución
---------
  python scripts/standardize_datasets.py
  python scripts/standardize_datasets.py --output data/processed/custom.parquet
  python scripts/standardize_datasets.py --force   # re-procesa aunque exista el output

Idempotencia
------------
  Si el archivo de salida ya existe y --force no está presente,
  el script termina sin reescribir nada. El resultado es siempre
  reproducible a partir de los mismos datos de entrada.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración del logger
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phishguard.standardize")

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _REPO_ROOT / "data" / "raw"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROCESSED_DIR / "unified_dataset.parquet"

# ---------------------------------------------------------------------------
# Tipos y constantes
# ---------------------------------------------------------------------------

# Función que toma un DataFrame crudo y devuelve uno con el esquema unificado.
DatasetParser = Callable[[pd.DataFrame], pd.DataFrame]

# Columnas requeridas en el DataFrame unificado (antes de añadir content_hash).
UNIFIED_COLUMNS: list[str] = ["dataset_source", "subject", "body", "label"]

# ---------------------------------------------------------------------------
# Helpers de bajo nivel
# ---------------------------------------------------------------------------


def _md5_of_text(text: str) -> str:
    """Devuelve el hash MD5 hexadecimal de una cadena UTF-8."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def _compute_content_hash(df: pd.DataFrame) -> pd.Series:
    """
    Genera un hash MD5 por fila sobre la concatenación de subject y body.

    La concatenación usa un separador de control (\\x00) para evitar
    colisiones entre ('ab', 'c') y ('a', 'bc').
    """
    combined: pd.Series = df["subject"].fillna("") + "\x00" + df["body"].fillna("")
    return combined.apply(_md5_of_text)


def _coerce_label_to_int8(series: pd.Series) -> pd.Series:
    """
    Normaliza la columna de etiqueta a int8 {0, 1}.

    Acepta enteros (0/1), strings numéricos, 'spam'/'ham', 'phishing'/'legit',
    y booleanos. Lanza ValueError si encuentra valores fuera de rango.
    """
    PHISHING_STRINGS = {"1", "spam", "phishing", "malicious", "true", "yes", "1.0"}
    LEGIT_STRINGS = {"0", "ham", "legit", "legitimate", "benign", "false", "no", "0.0"}

    def _map_single(val: object) -> int:
        s = str(val).strip().lower()
        if s in PHISHING_STRINGS:
            return 1
        if s in LEGIT_STRINGS:
            return 0
        raise ValueError(
            f"Valor de etiqueta desconocido: {val!r}. "
            f"Esperado uno de {PHISHING_STRINGS | LEGIT_STRINGS}."
        )

    return series.map(_map_single).astype("int8")


def _pick_column(df: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    """
    Devuelve la primera columna del DataFrame que coincida con `candidates`
    (comparación case-insensitive). Si ninguna existe, devuelve una serie
    de strings vacíos con el mismo índice.
    """
    lower_map: dict[str, str] = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        actual = lower_map.get(candidate.lower())
        if actual is not None:
            log.debug("  columna '%s' encontrada como '%s'", candidate, actual)
            return df[actual].fillna(default).astype(str)

    log.warning(
        "  Ninguna de las columnas candidatas %s encontrada. "
        "Se usará string vacío para todas las filas.",
        candidates,
    )
    return pd.Series(default, index=df.index, dtype=str)


# ---------------------------------------------------------------------------
# Parsers específicos por dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Especificación de un dataset crudo: nombre, archivo y función de parseo."""

    name: str
    filename: str
    parser: DatasetParser
    encoding: str = "utf-8"
    sep: str = ","


def _make_unified(
    df: pd.DataFrame,
    source: str,
    subject_cols: list[str],
    body_cols: list[str],
    label_cols: list[str],
    all_phishing: bool = False,
    all_legit: bool = False,
) -> pd.DataFrame:
    """
    Constructor genérico de un DataFrame unificado.

    Args:
        df:            DataFrame crudo del CSV.
        source:        Nombre del dataset (para la columna dataset_source).
        subject_cols:  Candidatos a columna de asunto (orden de preferencia).
        body_cols:     Candidatos a columna de cuerpo.
        label_cols:    Candidatos a columna de etiqueta (ignorado si
                       all_phishing o all_legit son True).
        all_phishing:  Si True, todos los registros son etiquetados como 1.
        all_legit:     Si True, todos los registros son etiquetados como 0.

    Returns:
        DataFrame con columnas [dataset_source, subject, body, label].
    """
    result = pd.DataFrame()
    result["dataset_source"] = source
    result["subject"] = _pick_column(df, subject_cols)
    result["body"] = _pick_column(df, body_cols)

    if all_phishing:
        result["label"] = pd.Series(1, index=df.index, dtype="int8")
    elif all_legit:
        result["label"] = pd.Series(0, index=df.index, dtype="int8")
    else:
        raw_label = _pick_column(df, label_cols, default="0")
        result["label"] = _coerce_label_to_int8(raw_label)

    return result


# --- Parsers individuales ---


def _parse_ceas08(df: pd.DataFrame) -> pd.DataFrame:
    """
    CEAS 2008 Spam Filtering Challenge.

    Esquema conocido: sender, receiver, date, subject, body, label
    donde label = 1 indica spam (tratado como clase positiva).
    Columna alternativa observada: 'spam' (bool/int).
    """
    return _make_unified(
        df,
        source="CEAS_08",
        subject_cols=["subject"],
        body_cols=["body", "message", "content"],
        label_cols=["label", "spam", "class", "category"],
    )


def _parse_enron(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enron Email Dataset (versión con etiquetas spam/ham).

    Variantes observadas en la comunidad:
      - subject, message, Spam/Ham
      - Subject, Body, Label
      - Message-ID, Date, From, To, Subject, X-FileName, label
    """
    return _make_unified(
        df,
        source="Enron",
        subject_cols=["subject", "Subject"],
        body_cols=["message", "body", "Body", "text", "content"],
        label_cols=["label", "Label", "spam", "Spam", "class"],
    )


def _parse_ling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ling-Spam corpus.

    Variantes: Subject/Body, subject/body, o solo 'text'/'message'.
    Etiqueta como spam=1/ham=0 o spamclass.
    """
    return _make_unified(
        df,
        source="Ling",
        subject_cols=["subject", "Subject"],
        body_cols=["body", "Body", "message", "text", "content"],
        label_cols=["label", "Label", "spam", "spamclass", "class"],
    )


def _parse_nazario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corpus Nazario de phishing.

    Este corpus contiene únicamente correos de phishing; se etiquetan
    todos como clase 1 independientemente de si existe una columna label.
    Esquemas observados: sender, receiver, date, urls, body  |  body, label.
    """
    has_label = any(c.lower() in {"label", "class", "spam"} for c in df.columns)
    return _make_unified(
        df,
        source="Nazario",
        subject_cols=["subject", "Subject"],
        body_cols=["body", "Body", "text", "message", "content", "urls"],
        label_cols=["label", "class"],
        all_phishing=(not has_label),
    )


def _parse_nigerian_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nigerian Fraud (419 scam) corpus.

    Típicamente solo emails de fraude (clase positiva).
    Columnas observadas: subject, body / text / message.
    """
    has_label = any(c.lower() in {"label", "class", "spam"} for c in df.columns)
    return _make_unified(
        df,
        source="Nigerian_Fraud",
        subject_cols=["subject", "Subject"],
        body_cols=["body", "Body", "text", "message", "content"],
        label_cols=["label", "class"],
        all_phishing=(not has_label),
    )


def _parse_phishing_email(df: pd.DataFrame) -> pd.DataFrame:
    """
    phishing_email.csv — dataset principal del proyecto (Módulo 2).

    Esquema documentado en la tesina:
      - text_combined : texto completo del correo (asunto + cuerpo pre-concatenados)
      - label         : 1 = phishing, 0 = safe
    También puede tener columnas separadas 'subject' y 'body'.
    """
    # Si tiene columnas separadas, las usamos; si no, todo va a body.
    has_subject = any(c.lower() == "subject" for c in df.columns)
    has_body = any(c.lower() in {"body", "message"} for c in df.columns)

    if has_subject or has_body:
        subject_cols = ["subject", "Subject"]
        body_cols = ["body", "Body", "message", "text_combined", "text"]
    else:
        # text_combined va como cuerpo; asunto queda vacío.
        subject_cols = ["subject"]  # fallback → serie vacía
        body_cols = ["text_combined", "text", "message", "body"]

    return _make_unified(
        df,
        source="phishing_email",
        subject_cols=subject_cols,
        body_cols=body_cols,
        label_cols=["label", "Label", "class", "spam"],
    )


def _parse_spamassasin(df: pd.DataFrame) -> pd.DataFrame:
    """
    SpamAssassin Public Corpus.

    Columnas observadas: label (spam=1/ham=0), subject, body / text / message.
    Variante alternativa: 'v1' para la etiqueta (0=ham, 1=spam).
    """
    return _make_unified(
        df,
        source="SpamAssasin",
        subject_cols=["subject", "Subject"],
        body_cols=["body", "Body", "message", "text", "content"],
        label_cols=["label", "Label", "spam", "v1", "class"],
    )


# ---------------------------------------------------------------------------
# Registro de datasets
# ---------------------------------------------------------------------------

DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec("CEAS_08", "CEAS_08.csv", _parse_ceas08),
    DatasetSpec("Enron", "Enron.csv", _parse_enron),
    DatasetSpec("Ling", "Ling.csv", _parse_ling),
    DatasetSpec("Nazario", "Nazario.csv", _parse_nazario),
    DatasetSpec("Nigerian_Fraud", "Nigerian_Fraud.csv", _parse_nigerian_fraud),
    DatasetSpec("phishing_email", "phishing_email.csv", _parse_phishing_email),
    DatasetSpec("SpamAssasin", "SpamAssasin.csv", _parse_spamassasin),
]

# ---------------------------------------------------------------------------
# Carga individual de un CSV
# ---------------------------------------------------------------------------


def load_raw_csv(spec: DatasetSpec, raw_dir: Path) -> pd.DataFrame | None:
    """
    Lee un CSV crudo y devuelve un DataFrame o None si hay error.

    Los errores de lectura (archivo inexistente, encoding incorrecto,
    CSV malformado) se registran pero no abortan el pipeline completo.
    """
    csv_path = raw_dir / spec.filename
    if not csv_path.exists():
        log.error("[%s] Archivo no encontrado: %s", spec.name, csv_path)
        return None

    try:
        df = pd.read_csv(
            csv_path,
            encoding=spec.encoding,
            sep=spec.sep,
            on_bad_lines="warn",
            low_memory=False,
        )
    except UnicodeDecodeError:
        log.warning(
            "[%s] Error de encoding UTF-8. Reintentando con latin-1.", spec.name
        )
        try:
            df = pd.read_csv(
                csv_path,
                encoding="latin-1",
                sep=spec.sep,
                on_bad_lines="warn",
                low_memory=False,
            )
        except Exception as exc:
            log.error("[%s] No se pudo leer el archivo: %s", spec.name, exc)
            return None
    except Exception as exc:
        log.error("[%s] Error inesperado al leer CSV: %s", spec.name, exc)
        return None

    log.info("[%s] Cargado: %d filas, %d columnas", spec.name, len(df), len(df.columns))
    log.debug("[%s] Columnas detectadas: %s", spec.name, df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# Parseo de un dataset individual
# ---------------------------------------------------------------------------


def parse_dataset(spec: DatasetSpec, df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Aplica el parser del spec y valida el esquema de salida.

    Devuelve None si el parser lanza una excepción no recuperable.
    """
    try:
        unified = spec.parser(df)
    except Exception as exc:
        log.error("[%s] Fallo en el parser: %s", spec.name, exc)
        return None

    # Verificar que todas las columnas requeridas estén presentes.
    missing = [c for c in UNIFIED_COLUMNS if c not in unified.columns]
    if missing:
        log.error(
            "[%s] El parser no generó las columnas requeridas: %s", spec.name, missing
        )
        return None

    # Eliminar filas donde subject Y body están vacíos (sin contenido útil).
    before = len(unified)
    mask_empty = (unified["subject"].str.strip() == "") & (
        unified["body"].str.strip() == ""
    )
    unified = unified[~mask_empty].copy()
    dropped = before - len(unified)
    if dropped:
        log.warning("[%s] %d filas eliminadas por tener subject y body vacíos.", spec.name, dropped)

    log.info(
        "[%s] Parseado: %d registros (phishing=%d, legit=%d)",
        spec.name,
        len(unified),
        (unified["label"] == 1).sum(),
        (unified["label"] == 0).sum(),
    )
    return unified


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_unified_dataset(
    raw_dir: Path = RAW_DIR,
    specs: list[DatasetSpec] | None = None,
) -> pd.DataFrame:
    """
    Orquesta la carga, parseo y unificación de todos los datasets.

    Args:
        raw_dir: Directorio con los CSVs crudos.
        specs:   Lista de DatasetSpec a procesar. Usa DATASET_SPECS por defecto.

    Returns:
        DataFrame unificado con las columnas:
        [dataset_source, subject, body, label, content_hash]

    Raises:
        RuntimeError: si ningún dataset pudo procesarse correctamente.
    """
    if specs is None:
        specs = DATASET_SPECS

    frames: list[pd.DataFrame] = []

    for spec in specs:
        log.info("─── Procesando: %s ───", spec.name)
        df_raw = load_raw_csv(spec, raw_dir)
        if df_raw is None:
            continue

        df_unified = parse_dataset(spec, df_raw)
        if df_unified is None:
            continue

        frames.append(df_unified)

    if not frames:
        raise RuntimeError(
            "No se pudo procesar ningún dataset. "
            "Revisa los logs para ver los errores por dataset."
        )

    log.info("Concatenando %d datasets...", len(frames))
    combined = pd.concat(frames, ignore_index=True)

    # Añadir content_hash ANTES de cualquier deduplicación posterior.
    # El hash se calcula aquí para preservar el campo en el parquet;
    # la deduplicación real se hará en split_builder.py usando este campo.
    log.info("Calculando content_hash (MD5 de subject + body)...")
    combined["content_hash"] = _compute_content_hash(combined)

    # Reordenar columnas para consistencia.
    combined = combined[["dataset_source", "subject", "body", "label", "content_hash"]]

    # Asegurar tipos correctos.
    combined["dataset_source"] = combined["dataset_source"].astype("category")
    combined["label"] = combined["label"].astype("int8")

    _log_summary(combined)
    return combined


def _log_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen estadístico del dataset unificado."""
    log.info("=" * 60)
    log.info("RESUMEN DEL DATASET UNIFICADO")
    log.info("  Total de registros : %d", len(df))
    log.info("  Phishing (label=1) : %d (%.1f%%)", (df["label"] == 1).sum(), 100 * (df["label"] == 1).mean())
    log.info("  Legítimo (label=0) : %d (%.1f%%)", (df["label"] == 0).sum(), 100 * (df["label"] == 0).mean())
    log.info("  Hashes únicos      : %d", df["content_hash"].nunique())
    log.info("  Hashes duplicados  : %d (pendiente deduplicación en split_builder)", len(df) - df["content_hash"].nunique())
    log.info("")
    log.info("  Por fuente:")
    for source, group in df.groupby("dataset_source", observed=True):
        log.info(
            "    %-20s %6d registros  (phishing=%d, legit=%d)",
            source,
            len(group),
            (group["label"] == 1).sum(),
            (group["label"] == 0).sum(),
        )
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Escritura del archivo de salida (idempotente)
# ---------------------------------------------------------------------------


def save_unified_dataset(
    df: pd.DataFrame,
    output_path: Path = DEFAULT_OUTPUT,
    force: bool = False,
) -> None:
    """
    Escribe el DataFrame unificado en formato Parquet.

    Idempotencia: si el archivo ya existe y force=False, no se sobreescribe.
    El formato Parquet preserva los tipos (int8, category) y es
    significativamente más eficiente que CSV para este volumen de datos.

    Args:
        df:          DataFrame a guardar.
        output_path: Ruta del archivo de salida.
        force:       Si True, sobreescribe el archivo existente.
    """
    if output_path.exists() and not force:
        log.info(
            "El archivo de salida ya existe: %s\n"
            "  → Usa --force para sobreescribir. Saliendo sin cambios.",
            output_path,
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Escribir a una ruta temporal primero para evitar corrupción
    # si el proceso se interrumpe a mitad de la escritura.
    tmp_path = output_path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp_path, index=False, compression="snappy")
        tmp_path.rename(output_path)
        log.info("Dataset unificado guardado en: %s", output_path)
        log.info("Tamaño del archivo: %.1f MB", output_path.stat().st_size / 1e6)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Error al guardar el archivo Parquet: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PhishGuard Hito 1 — Estandarización de datasets crudos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directorio con los CSV crudos. Default: {RAW_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ruta del Parquet de salida. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobreescribir el archivo de salida si ya existe.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="DATASET",
        help=(
            "Procesar solo los datasets indicados. "
            f"Opciones: {[s.name for s in DATASET_SPECS]}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Filtrar specs si se usó --only
    specs = DATASET_SPECS
    if args.only:
        valid_names = {s.name for s in DATASET_SPECS}
        unknown = set(args.only) - valid_names
        if unknown:
            log.error(
                "Datasets desconocidos en --only: %s. Válidos: %s",
                unknown,
                valid_names,
            )
            sys.exit(1)
        specs = [s for s in DATASET_SPECS if s.name in args.only]

    try:
        df = build_unified_dataset(raw_dir=args.raw_dir, specs=specs)
        save_unified_dataset(df, output_path=args.output, force=args.force)
    except RuntimeError as exc:
        log.error("Pipeline fallido: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()