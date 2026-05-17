# PhishGuard — Hybrid Phishing Email Detection System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-6C3483?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)

> **Tesina:** *Modelo Híbrido Basado en Análisis de Texto y Metadatos para la Detección de Correos Electrónicos de Phishing*
> **Autor:** Carlos Daniel Torres Macías · Ingeniería en Computación Inteligente · Universidad Autónoma de Aguascalientes

---

## Overview

PhishGuard is a hybrid machine learning system for phishing email detection using **late fusion with a metadata gating mechanism**. It combines two independent specialized submodels:

- **MetaSubModel** — Random Forest over 32 technical metadata features
- **TextSubModel** — TF-IDF vectorization + Logistic Regression on cleaned email body text

A weighted fusion engine (`score_final = α·score_meta + (1−α)·score_text`) combines both scores, and a **gating mechanism** short-circuits to a phishing decision when metadata confidence alone exceeds a configurable threshold — reducing unnecessary NLP inference on high-confidence cases.

Explainability is provided via **SHAP TreeExplainer**, propagated through the FastAPI microservice response.

---

## Results

| Submodel | ROC-AUC | Recall | F1-Score | Precision |
|---|---|---|---|---|
| Technical (Random Forest) | 0.9121 | 0.8117 | 0.8277 | 0.8443 |
| Semantic (TF-IDF + LR) | **0.9988** | **0.9897** | 0.9871 | 0.9845 |

| Component | Avg Latency |
|---|---|
| Base pipeline (no SHAP) | ~15 ms |
| SHAP overhead | ~12 ms |
| **Total with explainability** | **< 30 ms** |

SHAP explanations matched human expert judgment in **94%** of evaluated cases.

---

## Architecture

```
email (subject + body)
        │
        ▼
┌───────────────────┐
│  FeatureExtractor │  → 32 metadata features (heuristic, no-fit)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  MetaSubModel     │  → score_meta ∈ [0, 1]
│  (Random Forest)  │
└───────────────────┘
        │
score_meta ≥ θ_meta? ──── YES ──► label = PHISHING  (gating)
        │
        NO
        ▼
┌───────────────────┐
│  TextSubModel     │  → score_text ∈ [0, 1]
│  (TF-IDF + LR)    │
└───────────────────┘
        │
        ▼
score_final = α·score_meta + (1−α)·score_text
        │
        ▼
score_final ≥ θ_final → label ∈ {phishing, legitimate}
        │
        ▼
SHAP explanation (top-K features → phishing)
```

**Configured values:** `θ_meta = 0.70` · `α = 0.60` · `θ_final = 0.50`

---

## Quickstart

### Prerequisites

```bash
git lfs install
git lfs pull
pip install -r requirements.txt
```

### 1. Standardize datasets

```bash
python scripts/standardize_datasets.py
```

### 2. Create train/val/test splits

```bash
python scripts/make_splits.py
```

### 3. Train submodels

```bash
# Metadata submodel (Random Forest)
python scripts/train_metadata_model.py

# Text submodel (TF-IDF + Logistic Regression)
python scripts/train_text_model.py
```

### 4. Start the API

```bash
uvicorn phishguard.api.main:app --reload --port 8000
```

Interactive docs available at `http://localhost:8000/docs`.

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "engine_ready": true,
  "using_dummy_models": false,
  "meta_model": "RandomForestClassifier_meta_v1.0_20250501T...",
  "text_model": "TfIdf_LogisticRegression_text_v1.0_..."
}
```

### `POST /classify`

```json
// Request
{
  "subject": "URGENT: Verify your PayPal account",
  "body": "Dear Customer, click here immediately: http://192.168.1.1/login",
  "sender": "noreply@paypa1-security.tk"
}
```

```json
// Response
{
  "label": "phishing",
  "is_phishing": true,
  "score_final": 0.8923,
  "score_meta": 0.9341,
  "score_text": null,
  "gating": { "activated": true, "score_meta": 0.9341, "threshold": 0.70 },
  "latency_ms": 3.2,
  "explanation": {
    "has_ip_url": 0.312,
    "has_url_shortener": 0.201,
    "body_digit_count": 0.098,
    "has_urgency_words": 0.044,
    "has_form": 0.031
  }
}
```

---

## Datasets

Seven public datasets were consolidated, deduplicated by MD5 hash, and split into stratified 70/15/15 partitions:

| Dataset | Records | Description |
|---|---|---|
| phishing_email.csv | ~82k | Primary phishing corpus |
| Enron.csv | ~33k | Legitimate email corpus |
| CEAS_08.csv | ~17k | Spam filtering challenge |
| SpamAssasin.csv | ~6k | SpamAssassin public corpus |
| Ling.csv | ~2k | Ling-Spam corpus |
| Nazario.csv | ~1.5k | Phishing-only corpus |
| Nigerian_Fraud.csv | ~1k | 419 fraud corpus |

**Final corpus:** 164,563 deduplicated records (408 duplicates removed).

> Raw CSVs are tracked via **Git LFS**. See [`docs/data_policy.md`](docs/data_policy.md).

---

## Project Structure

```
phishguard/
├── configs/
│   └── model_config.yaml       # Thresholds, fusion weights, XAI config
├── data/
│   ├── raw/                    # Source CSVs (Git LFS)
│   └── processed/              # Parquet splits (gitignored)
├── artifacts/                  # Trained model .pkl files (gitignored)
├── scripts/
│   ├── standardize_datasets.py # Unify 7 datasets → unified_dataset.parquet
│   ├── make_splits.py          # Stratified 70/15/15 split with MD5 dedup
│   ├── train_metadata_model.py # Train Random Forest submodel
│   └── train_text_model.py     # Train TF-IDF + LR submodel
├── src/phishguard/
│   ├── preprocessing/
│   │   └── text_cleaner.py     # HTML stripping, Unicode normalization, URL extraction
│   ├── features/
│   │   └── extractor.py        # 32 metadata features + text preparation
│   ├── models/
│   │   ├── fusion_engine.py    # Late fusion engine with gating (PhishGuardEngine)
│   │   ├── meta_submodel.py    # MetaSubModel wrapper + serialization
│   │   └── text_submodel.py    # TextSubModel wrapper + serialization
│   ├── explainability/
│   │   └── explainer.py        # PhishGuardExplainer (SHAP TreeExplainer)
│   ├── api/
│   │   └── main.py             # FastAPI microservice (/health, /classify)
│   └── config.py               # Pydantic-validated config loader
└── tests/
    └── unit/
        └── test_imports.py
```

---

## Configuration

```yaml
gating:
  metadata_threshold: 0.70   # θ_meta: short-circuit threshold

fusion:
  alpha: 0.60                # weight of metadata submodel
  decision_threshold: 0.50   # θ_final: final decision cutoff

explainability:
  enabled: true
  top_k: 10                  # SHAP features to return per prediction
```

---

## Explainability

PhishGuard uses **SHAP TreeExplainer** (Lundberg et al., 2020, NeurIPS) for the Random Forest submodel. Values are **exact** (not approximations), satisfying local accuracy, consistency, and efficiency properties.

If SHAP is not installed, the system degrades gracefully: predictions continue normally and `explanation` returns `null`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn 1.4 (Random Forest, TF-IDF, Logistic Regression) |
| Explainability | SHAP 0.44 (TreeExplainer) |
| API | FastAPI 0.110 + Uvicorn |
| Validation | Pydantic v2 |
| Serialization | joblib (schema-versioned artifacts) |
| Data | pandas 2.1, pyarrow, Parquet |
| CI | GitHub Actions (Python 3.11, Ubuntu latest) |
| Data versioning | Git LFS |

---

## Known Architectural Limitations

1. **Static α** — fusion weight does not adapt to email profile (URL-heavy vs. text-heavy).
2. **Probability calibration mismatch** — Random Forest emits uncalibrated probabilities; Logistic Regression is intrinsically calibrated. Future work: `CalibratedClassifierCV(cv='prefit')`.
3. **Monolingual vocabulary** — TF-IDF vocabulary is English-only. Planned extension: multilingual sentence embeddings (LaBSE or `paraphrase-multilingual-MiniLM-L12-v2`).

---

## License

Academic project — Universidad Autónoma de Aguascalientes, 2025–2026.
Dataset usage is subject to each dataset's original license terms.
