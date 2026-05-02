# Molecule Decision System

Professional analytics pipeline for pharmaceutical market decisioning.

Summary
- Data ingestion and cleaning for IQVIA MAT datasets
- Feature engineering: market share, HHI proxy, CAGR, price/unit, volatility
- Rule-based scoring engine (opportunity vs risk) with configurable thresholds
- ML layer: classification, forecasting, clustering (RF, XGBoost optional, KMeans)
- Ensemble decision engine combining rule-based and ML signals
- Output formatter: ranked molecules, tiers, strategies, explainability

Quickstart
1. Create a Python virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Place the IQVIA workbook in this folder and update the `data_path` when running.

3. Run the pipeline from `analytics` folder:

```powershell
python pipeline.py "IQVIA - Leah .xlsx" --output-dir ./output
```

Notes
- Configurable parameters live in `config.py` (weights, thresholds, ML options).
- XGBoost is optional; the pipeline will continue without it.
- This repository contains audit fixes to data aggregation, leakage, scaling, and scoring. See commit history for details.

Files of interest
- `data_layer.py` — ingestion, cleaning, market-level aggregation
- `feature_engineering.py` — derived metrics and lifecycle classification
- `scoring_engine.py` — opportunity and risk computation
- `ml_models.py` — ML training and clustering
- `ensemble.py` — combines scoring + ML, computes confidence
- `output_formatter.py` — ranking, tiers, strategies, explainability
- `pipeline.py` — orchestration
git push origin main