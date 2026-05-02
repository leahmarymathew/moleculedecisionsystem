# Molecule Decision System — Analytics

> **End-to-end pharmaceutical molecule investment intelligence platform**  
> Ingests IQVIA MAT data, engineers strategic metrics, and produces ranked investment decisions with explainability.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Feature Catalogue](#feature-catalogue)
5. [Scoring Model](#scoring-model)
6. [Machine Learning Layer](#machine-learning-layer)
7. [Ensemble & Confidence](#ensemble--confidence)
8. [Output & Investment Tiers](#output--investment-tiers)
9. [Configuration Reference](#configuration-reference)
10. [Getting Started](#getting-started)
11. [File Reference](#file-reference)
12. [Dependencies](#dependencies)

---

## Overview

The Molecule Decision System analytics pipeline transforms raw IQVIA Moving Annual Total (MAT) pharmaceutical market data into ranked, investment-ready molecule recommendations. It combines rule-based scoring with three machine learning models and produces actionable outputs including investment tiers, entry strategies, and per-molecule explainability.

**Key capabilities:**

| Capability | Description |
|---|---|
| Data ingestion & cleaning | Load IQVIA Excel workbooks, handle negatives, impute missing values, aggregate to market level |
| Feature engineering | 18 strategic metrics covering growth, competition, pricing, volatility, and lifecycle |
| Dual-score framework | Independent Opportunity and Risk scores with configurable weights and penalty mechanisms |
| ML classification | Random Forest classifies molecules as High / Medium / Low potential |
| Growth forecasting | XGBoost predicts future revenue growth (optional) |
| Market segmentation | K-Means clustering identifies GROWTH / MATURE / DECLINING market segments |
| Ensemble decisioning | Weighted combination of rule-based scores and ML outputs with confidence scoring |
| Investment output | Ranked molecules with tiers, entry strategies, key drivers, and risk flags |

---

## Architecture

```
IQVIA Excel Workbook
        │
        ▼
┌──────────────────┐
│   data_layer.py  │  Load → Validate → Clean → Aggregate (market level)
└────────┬─────────┘
         │
         ▼
┌────────────────────────┐
│ feature_engineering.py │  18 strategic metrics (base + advanced + derived)
└──────────┬─────────────┘
           │
     ┌─────┴──────┐
     │             │
     ▼             ▼
┌──────────┐  ┌────────────┐
│ scoring_ │  │ ml_models  │  Rule-based scoring     ML: RF + XGBoost + K-Means
│ engine   │  │    .py     │
└────┬─────┘  └─────┬──────┘
     │               │
     └──────┬────────┘
            │
            ▼
    ┌──────────────┐
    │  ensemble.py │  Weighted combination + confidence calculation
    └──────┬───────┘
           │
           ▼
  ┌──────────────────┐
  │ output_formatter │  Rankings, tiers, strategies, explainability
  └──────────────────┘
           │
           ▼
  CSV / Excel / JSON output
```

All weights, thresholds, and model parameters are centralised in `config.py`.

---

## Pipeline Stages

The `PharmaceuticalInvestmentPipeline` class in `pipeline.py` orchestrates the following eight stages in sequence:

| Stage | Module | Description |
|---|---|---|
| 1. Data Ingestion | `data_layer.py` | Load IQVIA workbook from `Sheet1` |
| 2. Data Cleaning | `data_layer.py` | Flag negatives, impute via segment median, detect anomalies, aggregate to molecule-country level |
| 3. Feature Engineering | `feature_engineering.py` | Compute all 18 metrics |
| 4. Scoring | `scoring_engine.py` | Compute Opportunity and Risk scores; apply penalty mechanisms |
| 5. ML Modeling | `ml_models.py` | Train Random Forest, XGBoost, and K-Means |
| 6. Ensemble | `ensemble.py` | Combine all signals into a single ensemble score with confidence |
| 7. Output Formatting | `output_formatter.py` | Rank molecules, assign tiers, generate strategy recommendations |
| 8. Validation | `pipeline.py` | Sanity checks on score range, tier distribution, ranking uniqueness, and sensitivity analysis |

If any stage fails, the pipeline halts and logs the failing step. The ML stage has a fallback: if ML training fails, the pipeline continues with neutral ML values so that the scoring model output is preserved.

---

## Feature Catalogue

All features are computed by `FeatureEngineer` in `feature_engineering.py` using three MAT periods: **Q2 2023**, **Q2 2024**, **Q2 2025**.

### Base Metrics

| Feature | Description |
|---|---|
| `revenue_growth` | Total revenue change from Q2 2023 to Q2 2025 (%), capped at ±500% |
| `volume_growth` | Standard unit volume change over the same period (%), capped at ±500% |
| `market_size` | Latest MAT Q2 2025 revenue in millions (USD/local currency) |
| `competition_count` | Unique manufacturers offering the same molecule in the same country |
| `market_share` | Molecule's revenue share of its own Country × Molecule market (%) |
| `price_per_unit` | Revenue ÷ volume for Q2 2025; `NaN` when volume ≈ 0 |
| `price_change` | YoY price per unit change (%), capped at ±100% |
| `volatility` | Standard deviation of period-over-period revenue growth rates |

### Advanced Metrics

| Feature | Description |
|---|---|
| `hhi` | HHI-proxy for market concentration using revenue coefficient of variation; scale 2,500–10,000 |
| `cagr` | 2-year Compound Annual Growth Rate (2023–2025, %) |
| `revenue_sustainability` | Proportion of periods with positive growth (0–100) |
| `generic_erosion_index` | Proxy for generic pressure: high when price declines alongside volume growth (0–100, higher is better) |
| `market_penetration` | Market-share-weighted growth factor (0–100) |
| `channel_dependency` | Proxy for distribution channel concentration using Sector (HOSPITAL/RETAIL); higher = more concentrated |

### Derived Metrics

| Feature | Description |
|---|---|
| `lifecycle_stage` | Rule-based classification: **EMERGING** / **GROWTH** / **MATURE** / **DECLINING** (thresholds in `config.py`) |
| `risk_score` | Composite risk (0–100): competition + HHI + volatility + price erosion |
| `opportunity_score` | Composite opportunity (0–100): growth + market size + penetration |
| `entry_barrier_score` | Barrier to entry (0–100): HHI + incumbent market share + price stability |

---

## Scoring Model

`ScoringEngine` in `scoring_engine.py` implements a **dual-score framework**:

```
Final Score = Opportunity Score − Risk Score + Penalties
```

Scores are clipped to `[−100, 100]` and rescaled to `[0, 100]` for output (50 = neutral).

### Opportunity Score Components

| Component | Weight | Signal |
|---|---|---|
| Growth | 25% | S-curve applied to `revenue_growth`; midpoint calibrated to dataset median |
| Market Size | 20% | Log-scaled `market_size` with min-max normalisation |
| Demand / Penetration | 55% (residual) | `market_penetration` + volume growth momentum |

### Risk Score Components

| Component | Weight | Signal |
|---|---|---|
| Competition | 25% | S-curve on competitor count + monopoly fraction |
| Market Concentration | 25% | Exponential penalty for HHI > 2,500 |
| Volatility | 15% | Normalised revenue volatility |
| Pricing Risk | 15% | Triggered when price change < −20% |

### Penalty Mechanisms

| Penalty | Trigger | Max Deduction |
|---|---|---|
| Monopoly | Market share > 80% | −30 pts (scaled by excess over threshold) |
| Declining Revenue | Revenue growth below −20% | −20 pts (scaled by decline depth) |
| Small Market | Market size < 1 M | −15 pts (inverse tanh of size) |
| Generic Erosion | Price decline × volume gain | −10 pts |
| **Total Cap** | All penalties combined | **−50 pts maximum** |

---

## Machine Learning Layer

`MLModels` in `ml_models.py` trains three models on the feature matrix. The feature matrix excludes `revenue_growth` to prevent data leakage (it is used as the XGBoost target).

### Random Forest Classifier

- **Target:** Volume growth buckets — High (> 20%), Medium (−5% to 20%), Low (< −5%)
- **Architecture:** 100 trees, max depth 10, class weight boosted for High
- **Validation:** Stratified 80/20 train-test split; scaler fitted on training fold only
- **Output:** Per-molecule classification label (`High` / `Medium` / `Low`)

### XGBoost Growth Regressor *(optional)*

- **Target:** `revenue_growth` (continuous)
- **Architecture:** 100 estimators, max depth 5, learning rate 0.1
- **Validation:** 80/20 split; test RMSE and R² reported
- **Fallback:** Pipeline continues with zero forecast if XGBoost is not installed

### K-Means Segmentation

- **Optimal k:** Determined automatically using silhouette score (range 2–10)
- **Output:** Cluster labels mapped to `GROWTH` / `MATURE` / `DECLINING` based on cluster average revenue growth
- **Limitation:** K-Means assumes spherical, equal-variance clusters; interpret as coarse segments only

---

## Ensemble & Confidence

`EnsembleDecisionEngine` in `ensemble.py` produces the final investment score:

```
Ensemble Score = 0.40 × Scoring Model + 0.35 × ML Classification + 0.25 × XGBoost Forecast
```

All three inputs are normalised to [0, 100] before combining (percentile scaling for Scoring Model and XGBoost forecast; fixed mapping for RF classification: High=80, Medium=50, Low=20).

### Confidence Score

Confidence (0.0–1.0) reflects how much to trust the ensemble score:

| Component | Weight | Source |
|---|---|---|
| Data quality | 40% | Penalised for negative flags (−20%), imputation (−15%), growth anomalies (−25%) |
| Model agreement | 35% | Low standard deviation across the three component scores |
| Volatility | 25% | Inverse of normalised revenue volatility |

Confidence is clamped to a minimum of 0.30. Confidence classes: **High** (≥ 0.75) / **Medium** (≥ 0.50) / **Low** (< 0.50).

---

## Output & Investment Tiers

`OutputFormatter` in `output_formatter.py` produces the final deliverable:

### Investment Tiers

| Tier | Ensemble Score Threshold |
|---|---|
| **High Potential** | ≥ 70 |
| **Moderate** | 50–69 |
| **Avoid** | < 50 |

### Entry Strategy Recommendations

| Strategy | Trigger Conditions |
|---|---|
| `MONOPOLY_DEFENSE` | Market share > 80% |
| `TENDER_STRATEGY` | Sector = HOSPITAL and < 5 competitors |
| `DIFFERENTIATION` | Revenue growth > 30% and price change > 5% |
| `COST_LEADERSHIP` | Market size > $500 M and risk score > 50 |
| `NICHE_PARTNERSHIP` | Market size < $10 M and revenue growth > 10% |
| `PORTFOLIO_BUNDLING` | Market penetration > 50 and price declining > −5% |
| `MONITOR` | Default (none of the above) |

### Output Files

| File | Contents |
|---|---|
| `output/molecules_ranked.csv` | Full ranked list of all molecules |
| `output/molecules_top50.xlsx` | Top 50 High Potential molecules |
| `output/analysis_summary.json` | Executive summary: tier counts, top 5, portfolio recommendation |
| `output/execution_log.txt` | Timestamped log of all pipeline steps |

---

## Configuration Reference

All tunable parameters are in `config.py`:

| Section | Key Parameters |
|---|---|
| `WEIGHTS` | Opportunity component weights: growth (0.25), market_size (0.20), competition (0.25), pricing_power (0.15), stability (0.15) |
| `RISK_WEIGHTS` | Risk component weights: competition (0.25), concentration (0.25), volatility (0.15), pricing (0.15) |
| `THRESHOLDS` | Business rules: monopoly threshold (80%), decline CAGR (−20%), penalty caps, lifecycle size/growth cutoffs, score bounds |
| `ML` | Model hyperparameters: RF trees/depth, XGBoost depth/learning rate, K-Means cluster range |
| `CONFIDENCE` | Confidence weights and data quality penalty rates |
| `TIER_CUTOFFS` | Score thresholds for High Potential (70), Moderate (50), Avoid (0) |

---

## Getting Started

### Prerequisites

- Python 3.9+
- An IQVIA workbook (`.xlsx`) with a `Sheet1` containing the required MAT columns

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

Place the IQVIA workbook in the `analytics/` directory, then run:

```bash
cd analytics
python pipeline.py "IQVIA - Leah .xlsx" --output-dir ./output
```

The pipeline will log each stage to the console and write all output files to `./output/`.

### Inspecting the Workbook Schema

To verify the column layout of a new IQVIA workbook before running the pipeline:

```bash
python _inspect_iqvia.py
```

---

## File Reference

| File | Class / Purpose |
|---|---|
| `pipeline.py` | `PharmaceuticalInvestmentPipeline` — end-to-end orchestrator (8 stages) |
| `data_layer.py` | `IQVIADataLoader` — load, validate, clean, and aggregate IQVIA data |
| `feature_engineering.py` | `FeatureEngineer` — compute all 18 strategic metrics |
| `scoring_engine.py` | `ScoringEngine` — dual-score framework with penalties |
| `ml_models.py` | `MLModels` — Random Forest, XGBoost, and K-Means models |
| `ensemble.py` | `EnsembleDecisionEngine` — weighted ensemble + confidence scoring |
| `output_formatter.py` | `OutputFormatter` — rankings, tiers, strategies, explainability |
| `config.py` | Central configuration: weights, thresholds, ML hyperparameters |
| `performance_metrics.py` | Reserved for future model performance tracking |
| `sql_queries.py` | Reserved for future database query templates |
| `_inspect_iqvia.py` | Utility script for inspecting IQVIA workbook schema |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading, manipulation, and aggregation |
| `numpy` | Numerical computations and transformations |
| `scipy` | Statistical functions (used in feature engineering) |
| `scikit-learn` | Random Forest, K-Means, StandardScaler, train-test split, metrics |
| `xgboost` | XGBoost growth regressor *(optional — pipeline continues without it)* |
| `openpyxl` | Reading `.xlsx` workbooks via pandas |

Install all required packages:

```bash
pip install pandas numpy scipy scikit-learn openpyxl
# Optional:
pip install xgboost
```

---

*For questions about data interpretation, scoring weights, or ML configuration, refer to `config.py` as the single source of truth for all tunable parameters.*
