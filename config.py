"""
Configuration and weights for analytics pipeline.
Central source-of-truth for scoring weights and thresholds.
All values can be overridden via environment variables.
"""
import os


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


# ===== OPPORTUNITY & RISK WEIGHTS =====
WEIGHTS = {
    'growth': _env_float('WEIGHT_GROWTH', 0.25),
    'market_size': _env_float('WEIGHT_MARKET_SIZE', 0.20),
    'competition': _env_float('WEIGHT_COMPETITION', 0.25),
    'pricing_power': _env_float('WEIGHT_PRICING_POWER', 0.15),
    'stability': _env_float('WEIGHT_STABILITY', 0.15),
}

# Risk component weights (FIX #10)
RISK_WEIGHTS = {
    'competition': _env_float('RISK_WEIGHT_COMPETITION', 0.25),
    'concentration': _env_float('RISK_WEIGHT_CONCENTRATION', 0.25),
    'volatility': _env_float('RISK_WEIGHT_VOLATILITY', 0.15),
    'pricing': _env_float('RISK_WEIGHT_PRICING', 0.15),
}

# ===== BUSINESS THRESHOLDS =====
THRESHOLDS = {
    'monopoly_share': _env_float('THRESHOLD_MONOPOLY_SHARE', 0.8),
    'small_market_size': _env_float('THRESHOLD_SMALL_MARKET_SIZE', 1.0),
    'decline_cagr_pct': _env_float('THRESHOLD_DECLINE_CAGR_PCT', -0.20),

    # FIX #25: Additional thresholds from hardcoded values
    'lifecycle_emerging_growth': _env_float('THRESHOLD_LIFECYCLE_EMERGING_GROWTH', 0.20),
    'lifecycle_growth_growth': _env_float('THRESHOLD_LIFECYCLE_GROWTH_GROWTH', 0.15),
    'lifecycle_decline_threshold': _env_float('THRESHOLD_LIFECYCLE_DECLINE', -0.10),
    'lifecycle_emerging_market_size': _env_float('THRESHOLD_LIFECYCLE_EMERGING_MARKET_SIZE', 50),
    'lifecycle_mid_market_min': _env_float('THRESHOLD_LIFECYCLE_MID_MARKET_MIN', 50),
    'lifecycle_mid_market_max': _env_float('THRESHOLD_LIFECYCLE_MID_MARKET_MAX', 500),

    'penalty_monopoly': _env_float('PENALTY_MONOPOLY', -30.0),
    'penalty_declining': _env_float('PENALTY_DECLINING', -20.0),
    'penalty_small_market': _env_float('PENALTY_SMALL_MARKET', -15.0),
    'penalty_generic_erosion': _env_float('PENALTY_GENERIC_EROSION', -10.0),
    'penalty_cap': _env_float('PENALTY_CAP', -50.0),

    'final_score_min': _env_float('FINAL_SCORE_MIN', -100),
    'final_score_max': _env_float('FINAL_SCORE_MAX', 100),
}

# ===== ML CONFIGURATION =====
ML = {
    'ensemble_weights': {
        'scoring_model': _env_float('ML_ENSEMBLE_SCORING', 0.4),
        'random_forest': _env_float('ML_ENSEMBLE_RF', 0.35),
        'xgboost': _env_float('ML_ENSEMBLE_XGB', 0.25),
    },
    'ml_contrib_weight': _env_float('ML_CONTRIB_WEIGHT', 0.25),
    'rf_max_depth': _env_int('ML_RF_MAX_DEPTH', 10),
    'rf_n_estimators': _env_int('ML_RF_N_ESTIMATORS', 100),
    'xgb_max_depth': _env_int('ML_XGB_MAX_DEPTH', 5),
    'xgb_n_estimators': _env_int('ML_XGB_N_ESTIMATORS', 100),
    'xgb_learning_rate': _env_float('ML_XGB_LEARNING_RATE', 0.1),
    'kmeans_min_clusters': _env_int('ML_KMEANS_MIN_CLUSTERS', 2),
    'kmeans_max_clusters': _env_int('ML_KMEANS_MAX_CLUSTERS', 10),
}

# ===== CONFIDENCE SETTINGS =====
CONFIDENCE = {
    'min_confidence': _env_float('CONFIDENCE_MIN', 0.3),
    'quality_penalty_negative': _env_float('CONFIDENCE_PENALTY_NEGATIVE', 0.20),
    'quality_penalty_imputed': _env_float('CONFIDENCE_PENALTY_IMPUTED', 0.15),
    'quality_penalty_anomaly': _env_float('CONFIDENCE_PENALTY_ANOMALY', 0.25),
    'data_confidence_weight': _env_float('CONFIDENCE_DATA_WEIGHT', 0.40),
    'model_agreement_weight': _env_float('CONFIDENCE_AGREEMENT_WEIGHT', 0.35),
    'volatility_weight': _env_float('CONFIDENCE_VOLATILITY_WEIGHT', 0.25),
}

# ===== TIER CUTOFFS =====
TIER_CUTOFFS = {
    'high_potential': _env_float('TIER_HIGH_POTENTIAL', 70),
    'moderate': _env_float('TIER_MODERATE', 50),
    'avoid': _env_float('TIER_AVOID', 0),
}
