# Molecule Decision System

An end-to-end analytics pipeline for pharmaceutical market investment decisioning, built on IQVIA MAT (Moving Annual Total) data.

---

## Overview

The pipeline ingests raw IQVIA pharmaceutical sales data, engineers strategic features, applies a dual-score (opportunity vs. risk) framework, augments decisions with machine learning, and produces a ranked molecule list with investment tiers and strategy recommendations.

**Key capabilities:**

- IQVIA MAT workbook ingestion with structural data-quality scoring
- Feature engineering: market share, HHI, CAGR, price-per-unit, volatility, lifecycle stage, and more
- Rule-based scoring engine with configurable weights, thresholds, and penalty caps
- ML layer: Random Forest classifier, XGBoost growth forecaster (optional), K-Means segmentation
- Ensemble decision engine (40 % scoring model · 35 % RF · 25 % XGBoost)
- Confidence scoring that accounts for data quality flags and model agreement
- Ranked output with investment tiers, entry strategies, and heuristic explainability
- Sensitivity analysis and stability checks in the validation stage

---

## Architecture

```
IQVIA Workbook (.xlsx)
        │
        ▼
┌─────────────────┐
│  Data Layer     │  Load → Validate → Clean → Aggregate → Quality Score
│  data_layer.py  │  (manufacturer rows → molecule-market level)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Feature Engineering │  Revenue/volume growth · Market size · Competition count
│  feature_engineering │  Market share · HHI · CAGR · Price change · Volatility
│       .py            │  Sustainability · Generic erosion · Lifecycle stage
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐
│ Scoring Engine  │  Opportunity score − Risk score + Penalty adjustments
│ scoring_engine  │  Final score range: −100 to +100
│      .py        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Models     │  Random Forest (classifier) · XGBoost (forecaster, optional)
│  ml_models.py   │  K-Means segmentation (2–10 clusters, silhouette-optimised)
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Ensemble Engine     │  Weighted combination → Confidence scoring → Tier assignment
│   ensemble.py        │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Output Formatter    │  Ranked list · Investment tiers · Entry strategies
│ output_formatter.py  │  Heuristic explainability per molecule
└────────┬─────────────┘
         │
         ▼
  CSV · Excel · JSON · Execution log
```

---

## Quickstart

**Requirements:** Python 3.9+

### 1. Set up the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Place the data file

Copy your IQVIA workbook into the project root. The default expected filename is:

```
IQVIA - Leah .xlsx
```

The workbook must contain a sheet named `Sheet1` with the following columns:

| Column group | Columns |
|---|---|
| Hierarchy | `Country`, `Sector`, `Manufacturer`, `Molecule List` |
| Classification | `ATC1`, `ATC2`, `ATC3`, `ATC4`, `Innovation Insights` |
| Revenue (LCD MNF) | `MAT Q2 2023_LCD MNF`, `MAT Q2 2024_LCD MNF`, `MAT Q2 2025_LCD MNF` |
| Volume (Std Units) | `MAT Q2 2023_Standard Units`, `MAT Q2 2024_Standard Units`, `MAT Q2 2025_Standard Units` |

### 3. Run the pipeline

```powershell
python pipeline.py
```

Results are written to `./output/` by default.

---

## Output Files

| File | Description |
|---|---|
| `molecules_ranked.csv` | All molecules, ranked by ensemble score |
| `molecules_top50.xlsx` | Top-50 **High Potential** molecules |
| `analysis_summary.json` | Aggregate statistics and tier distribution |
| `execution_log.txt` | Timestamped log of every pipeline step |

---

## Scoring Model

### Opportunity Score (0–100)

| Driver | Weight |
|---|---|
| Revenue growth | 25 % |
| Market size | 20 % |
| Competition (inverse) | 25 % |
| Pricing power | 15 % |
| Stability | 15 % |

### Risk Score (0–100)

| Driver | Weight |
|---|---|
| Competition intensity | 25 % |
| Market concentration (HHI) | 25 % |
| Revenue volatility | 15 % |
| Price erosion | 15 % |

### Penalty Adjustments

| Condition | Penalty |
|---|---|
| Monopoly (market share > 80 %) | −30 pts |
| Declining revenue | −20 pts |
| Small market | −15 pts |
| Generic erosion | −10 pts |
| **Maximum total penalty** | **−50 pts** |

**Final score = Opportunity − Risk + Penalties**, capped at [−100, +100].

### Ensemble Weights

| Component | Weight |
|---|---|
| Scoring model | 40 % |
| Random Forest classifier | 35 % |
| XGBoost growth forecast | 25 % |

---

## Investment Tiers

| Tier | Ensemble Score |
|---|---|
| **High Potential** | ≥ 70 |
| **Moderate** | 50 – 69 |
| **Avoid** | < 50 |

---

## Entry Strategies

The output formatter assigns one of the following strategies per molecule based on its feature profile:

| Strategy | Trigger condition |
|---|---|
| **Cost Leadership** | High risk + large market (> $500 M) |
| **Tender Strategy** | Hospital sector + < 5 competitors |
| **Portfolio Bundling** | High penetration + declining price |
| **Differentiation** | Strong growth (> 30 %) + positive price trend |
| **Monopoly Defense** | Market share > 80 % |
| **Niche Partnership** | Small market (< $10 M) + growth > 10 % |

---

## Lifecycle Classification

Molecules are classified into one of five lifecycle stages:

| Stage | Criteria |
|---|---|
| **EMERGING** | New market entry, or market size < $50 M with growth > 20 % |
| **GROWTH** | Mid/large market (≥ $50 M) with strong growth (> 15–20 %) |
| **MATURE** | Stable growth, established market |
| **DECLINING** | Revenue growth < −10 % |
| **EXITED** | Zero revenue across all periods |

---

## Confidence Scoring

Each molecule receives a confidence score (0.0 – 1.0) reflecting:

- **Data quality** — penalties for negative revenue flags, imputed values, and growth anomalies
- **Structural flags** — new entrants (−20 %), exits (−60 %), inactive molecules (−90 %), partial presence (−15 %)
- **Model agreement** — standard deviation across the three ensemble components
- **Volatility** — high revenue volatility reduces confidence

| Confidence class | Score |
|---|---|
| High | ≥ 0.75 |
| Medium | 0.50 – 0.74 |
| Low | < 0.50 |

Minimum confidence is capped at 0.30 to avoid over-penalising data-sparse molecules.

---

## Configuration

All scoring weights, ML hyperparameters, and thresholds are centralised in `config.py`.

```python
# Example: adjust opportunity weights
WEIGHTS = {
    'growth': 0.25,
    'market_size': 0.20,
    'competition': 0.25,
    'pricing_power': 0.15,
    'stability': 0.15,
}

# Example: tier cutoffs
TIER_CUTOFFS = {
    'high_potential': 70,
    'moderate': 50,
    'avoid': 0,
}
```

XGBoost is optional. If the package is not installed, the pipeline continues using only the Random Forest and scoring model components.

---

## Module Reference

| Module | Responsibility |
|---|---|
| `data_layer.py` | IQVIA workbook loading, structural validation, cleaning, manufacturer-to-market aggregation, data quality scoring |
| `feature_engineering.py` | All derived metrics (growth, HHI, CAGR, lifecycle, opportunity/risk sub-scores, etc.) |
| `scoring_engine.py` | Dual-score (opportunity vs. risk) computation with penalty logic |
| `ml_models.py` | Random Forest classifier, XGBoost forecaster, K-Means segmentation, data-leakage-safe feature matrix |
| `ensemble.py` | Weighted ensemble combination, confidence scoring, confidence classification |
| `output_formatter.py` | Molecule ranking, investment tier assignment, entry strategy selection, heuristic explainability |
| `pipeline.py` | End-to-end orchestration, validation, sensitivity analysis, export |
| `config.py` | Central source-of-truth for all weights, thresholds, and ML hyperparameters |

---

## Validation & Quality Checks

The pipeline runs the following checks after every execution:

- Output is non-empty and contains no null ensemble scores
- All scores fall within [0, 100]
- More than one investment tier is represented (distribution not collapsed)
- All molecule ranks are unique
- Feature sanity: competition ≥ 0, market share ∈ [0, 100], revenue growth present
- Tier distribution is not over-concentrated (no tier > 95 % of molecules)
- Opportunity and risk scores are not collinear (correlation < 0.95)
- Sensitivity analysis: score coefficient of variation checked at ±10 % weight perturbation

---

## Dependencies

| Package | Minimum version |
|---|---|
| pandas | 2.0.0 |
| numpy | 1.24.0 |
| scikit-learn | 1.3.0 |
| scipy | 1.11.0 |
| openpyxl | 3.1.0 |
| xgboost | *(optional)* |

Install with:

```powershell
pip install -r requirements.txt
# Optional: pip install xgboost
```