# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Install dependencies
pip install -r requirements.txt

# Optional: install XGBoost for forecasting model
pip install xgboost

# Step 1: Train ML models (once, or after data updates)
python train_models.py "IQVIA - Leah .xlsx"
# Models are saved to models/v1/

# Step 2: Run the pipeline (loads pre-trained models from models/v1/)
python pipeline.py
```

The pipeline can also be called as a function:

```python
import pandas as pd
from pipeline import run_pipeline

df = pd.read_excel("IQVIA - Leah .xlsx", sheet_name="Sheet1")
results = run_pipeline(df, models_dir="models/v1")
# results keys: ranked_molecules, tier_distribution, analysis_summary,
#               execution_log, validation
```

## Architecture

This is a pharmaceutical molecule investment ranking system ("Molecule Decision System") that processes IQVIA MAT (Moving Annual Total) Excel data and ranks molecules for market entry decisions.

**Pipeline flow** — each step is a standalone class that receives the previous step's output:

| Module | Class | Responsibility |
|--------|-------|----------------|
| `data_layer.py` | `IQVIADataLoader` | Validate and clean the IQVIA DataFrame; compute data quality scores |
| `feature_engineering.py` | `FeatureEngineer` | Build 20+ market metrics (CAGR, HHI, price elasticity, volatility, lifecycle stage, etc.) |
| `scoring_engine.py` | `ScoringEngine` | Dual-score framework: Opportunity score (growth 25%, market size 20%) minus Risk score (competition 25%, concentration 25%, volatility 15%, pricing 15%), with business penalties |
| `ml_models.py` | `MLModels` | Train or load RF classifier, XGBoost forecaster, K-Means; `from_disk()` for inference, `save_models()` for training |
| `ensemble.py` | `EnsembleDecisionEngine` | Combine rule-based (40%) + RF (35%) + XGBoost (25%) into a final score with confidence |
| `output_formatter.py` | `OutputFormatter` | Rank molecules, assign tiers, generate entry strategy recommendations |
| `pipeline.py` | `run_pipeline(df)` | Callable function: accepts raw DataFrame, returns results dict; ML failures are non-fatal |
| `train_models.py` | — | One-shot training script: trains all models and serializes them to `models/v1/` |

**Train vs. inference separation:** `train_models.py` trains and saves RF, XGBoost, and K-Means models (plus the RF `StandardScaler`) using `joblib`. At inference time, `MLModels.from_disk(models_dir, features_df)` loads them. If model files are missing, inference falls back to neutral defaults.

## Configuration

All scoring weights, business thresholds, ML hyperparameters, confidence penalties, and tier cutoffs live in **`config.py`** — edit there rather than in individual modules.

Key config sections: `WEIGHTS`, `RISK_WEIGHTS`, `THRESHOLDS`, `ML`, `CONFIDENCE`, `TIER_CUTOFFS`.

Tier cutoffs: High Potential ≥ 70, Moderate ≥ 50, Avoid < 50.

## Docker

```powershell
# Build and start all three services (api + prometheus + grafana)
docker compose up --build

# API:        http://localhost:8000/docs
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / admin)
```

Models must be trained before starting (run `python train_models.py` on the host first — they are mounted read-only via `./models:/app/models`). XGBoost is not in `requirements.txt`; its `ImportError` is caught silently in `ml_models.py`.

Override host/port/workers at runtime:
```powershell
docker run -e PORT=9000 -e WORKERS=1 -v ./models:/app/models:ro pharma-analytics
```

`WORKERS > 1` is unsupported because the job store is in-memory per process.

## API server

```powershell
# Start the server (train models first)
uvicorn main:app --reload
```

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/analyze` | yes | Multipart `.xlsx` upload → returns `job_id` (202) |
| POST | `/analyze/json` | yes | JSON molecule records → returns `job_id` (202) |
| POST | `/train` | yes | Retrain models from disk and hot-reload; `data_path` query param or `TRAINING_DATA_PATH` env var |
| GET | `/jobs/{job_id}` | yes | Poll status; `result` is populated when `status == "complete"` |
| GET | `/model/info` | yes | Training metadata from `models/v1/metadata.json` |
| GET | `/health` | no | Liveness; `status` is `"degraded"` if models aren't loaded |
| GET | `/metrics` | no | Prometheus metrics (auto-instrumented) |
| GET | `/docs` | yes | Swagger UI |

All error responses include `request_id`, `error`, `detail`, and `timestamp`. Every response carries `X-Request-ID` header. Both analyze endpoints return immediately with a `poll_url`; the pipeline runs in a background thread. `schemas.py` contains all Pydantic v2 models.

**Env vars:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_KEY` | _(unset)_ | Static API key; if unset, auth is bypassed (dev mode) |
| `LOG_FORMAT` | `json` | Set to `console` for human-readable output |
| `TRAINING_DATA_PATH` | _(unset)_ | Default data path for `POST /train` |
| `HOST` / `PORT` / `WORKERS` | `0.0.0.0` / `8000` / `1` | uvicorn runtime config |

**Custom Prometheus metrics** (in addition to http_* auto-metrics):
- `pharma_pipeline_runs_total{status}` — success/failure counts
- `pharma_tier_molecules_total{tier}` — molecules per investment tier
- `pharma_confidence_molecules_total{confidence_class}` — molecules per confidence class
- `pharma_model_info` — trained_at, n_samples, models_dir

Logging uses structlog (JSON by default, threaded with `request_id` and `job_id` context vars). Configure via `logging_config.py` / `get_logger()`.

## Outputs

The `--output-dir` (default `./analytics_output/`) receives:

- `molecules_ranked.csv` — all molecules with scores, tiers, drivers, risks
- `molecules_top50.xlsx` — top 50 formatted for stakeholder review
- `analysis_summary.json` — portfolio-level statistics and metadata
- `execution_log.txt` — timestamped step-by-step log

## Data

Input is an Excel workbook with molecule × country rows and three MAT quarter columns (revenue in LCD Manufacturer units, volume in Standard Units). The loader handles negative revenue flags, structural zeros (no imputation), and anomaly detection before computing pre-aggregated market metrics (market share, competition count, HHI) at the manufacturer level.
