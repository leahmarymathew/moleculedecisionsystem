"""
Configuration and weights for analytics pipeline.
Central source-of-truth for scoring weights and thresholds.
"""

# ===== OPPORTUNITY & RISK WEIGHTS =====
WEIGHTS = {
    'growth': 0.25,
    'market_size': 0.20,
    'competition': 0.25,
    'pricing_power': 0.15,
    'stability': 0.15
}

# Risk component weights (FIX #10)
RISK_WEIGHTS = {
    'competition': 0.25,
    'concentration': 0.25,
    'volatility': 0.15,
    'pricing': 0.15
}

# ===== BUSINESS THRESHOLDS =====
THRESHOLDS = {
    'monopoly_share': 0.8,   # fraction (80%)
    'small_market_size': 1.0, # in same units as market_size column
    'decline_cagr_pct': -0.20, # -20% decline
    
    # FIX #25: Additional thresholds from hardcoded values
    'lifecycle_emerging_growth': 0.20,  # 20% growth for EMERGING
    'lifecycle_growth_growth': 0.15,  # 15% growth threshold for GROWTH
    'lifecycle_decline_threshold': -0.10,  # -10% decline for DECLINING
    'lifecycle_emerging_market_size': 50,  # < 50M for EMERGING
    'lifecycle_mid_market_min': 50,  # >= 50M for mid-size
    'lifecycle_mid_market_max': 500,  # < 500M for mid-size
    
    'penalty_monopoly': -30.0,  # Monopoly penalty (FIX #14)
    'penalty_declining': -20.0,  # Declining revenue penalty
    'penalty_small_market': -15.0,  # Small market penalty
    'penalty_generic_erosion': -10.0,  # Generic erosion penalty
    'penalty_cap': -50.0,  # Maximum total penalty (FIX #14)
    
    'final_score_min': -100,  # FIX #13: Explicit bounds
    'final_score_max': 100,
}

# ===== ML CONFIGURATION =====
ML = {
    'ensemble_weights': {
        'scoring_model': 0.4,
        'random_forest': 0.35,
        'xgboost': 0.25
    },
    'ml_contrib_weight': 0.25,
    'rf_max_depth': 10,  # FIX #25: Random Forest config
    'rf_n_estimators': 100,
    'xgb_max_depth': 5,  # FIX #25: XGBoost config
    'xgb_n_estimators': 100,
    'xgb_learning_rate': 0.1,
    'kmeans_min_clusters': 2,  # FIX #25: K-Means config
    'kmeans_max_clusters': 10,
}

# ===== CONFIDENCE SETTINGS =====
CONFIDENCE = {
    'min_confidence': 0.3,
    'quality_penalty_negative': 0.20,  # FIX #20: Data quality penalties
    'quality_penalty_imputed': 0.15,
    'quality_penalty_anomaly': 0.25,
    'data_confidence_weight': 0.40,
    'model_agreement_weight': 0.35,
    'volatility_weight': 0.25,
}

# ===== TIER CUTOFFS =====
TIER_CUTOFFS = {
    'high_potential': 70,
    'moderate': 50,
    'avoid': 0,
}
