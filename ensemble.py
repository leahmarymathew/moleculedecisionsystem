"""
Ensemble Decision Engine: Combine scoring model + ML outputs (Section 7).
Final score = 40% Scoring Model + 35% ML Classification + 25% Forecast.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class EnsembleDecisionEngine:
    """
    Ensemble combines:
    - 40% Scoring Model (rule-based + penalties)
    - 35% ML Classification (RF: High/Medium/Low)
    - 25% XGBoost Forecast (predicted growth)
    
    Outputs confidence level based on model agreement.
    """
    
    WEIGHTS = {
        'scoring_model': 0.40,
        'ml_classification': 0.35,
        'xgb_forecast': 0.25
    }
    
    CONFIDENCE_THRESHOLDS = {
        'high': 0.75,
        'medium': 0.50,
        'low': 0.0
    }
    
    def __init__(self, 
                 scores_df: pd.DataFrame,
                 ml_df: pd.DataFrame):
        """
        Args:
            scores_df: From ScoringEngine.get_scores_dataframe()
            ml_df: From MLModels.get_ml_predictions_dataframe()
        """
        self.scores = scores_df.copy()
        self.ml = ml_df.copy()
        self.ensemble_df = pd.DataFrame()
    
    def normalize_scoring_model(self) -> pd.Series:
        """
        Normalize final_score to 0-100 range for ensemble.
        Input typically ranges -100 to +200.
        Output: 0-100 where 50 = neutral.
        """
        final_score = self.scores['final_score']

        # Robust percentile scaling avoids mean/std instability
        p5 = np.nanpercentile(final_score, 5)
        p95 = np.nanpercentile(final_score, 95)
        if p95 - p5 <= 1e-9:
            return pd.Series(50, index=final_score.index)

        normalized = 100 * (final_score - p5) / (p95 - p5)
        return normalized.clip(0, 100)
    
    def normalize_ml_classification(self) -> pd.Series:
        """
        Convert RF classification (High/Medium/Low) to 0-100 score.
        High: 80 | Medium: 50 | Low: 20
        """
        classification = self.ml['rf_classification'].fillna('Medium')
        
        class_scores = {
            'High': 80,
            'Medium': 50,
            'Low': 20
        }
        
        normalized = classification.map(class_scores)
        return normalized
    
    def normalize_xgb_forecast(self) -> pd.Series:
        """
        Use percentile scaling instead of hardcoded formula.
        Convert XGBoost growth forecast to 0-100 score.
        Normalize to 0-100 where 50 = median growth.
        """
        forecast = self.ml['xgb_growth_forecast'].fillna(0)
        
        # Use percentile-based normalization instead of hardcoded division by 2
        p5 = np.nanpercentile(forecast, 5)
        p95 = np.nanpercentile(forecast, 95)
        
        if p95 - p5 <= 1e-9:
            normalized = pd.Series(50, index=forecast.index)
        else:
            # Scale [p5, p95] to [0, 100]
            normalized = 100 * (forecast - p5) / (p95 - p5)
            normalized = normalized.clip(0, 100)
        
        return normalized
    
    def compute_ensemble_score(self) -> Tuple[pd.Series, Dict]:
        """
        Compute final ensemble score.
        Weighted combination of three components.
        """
        scoring_model_norm = self.normalize_scoring_model()
        ml_class_norm = self.normalize_ml_classification()
        xgb_forecast_norm = self.normalize_xgb_forecast()
        
        # Weighted ensemble
        ensemble_score = (
            self.WEIGHTS['scoring_model'] * scoring_model_norm +
            self.WEIGHTS['ml_classification'] * ml_class_norm +
            self.WEIGHTS['xgb_forecast'] * xgb_forecast_norm
        )
        
        components = {
            'scoring_model': scoring_model_norm,
            'ml_classification': ml_class_norm,
            'xgb_forecast': xgb_forecast_norm
        }
        
        return ensemble_score, components
    
    def compute_confidence(self, 
                          ensemble_score: pd.Series,
                          components: Dict) -> pd.Series:
        """
        Include data quality flags in confidence calculation.
        Confidence level (0.0 - 1.0) based on:
        - Data quality flags (negative values, imputation, anomalies)
        - Data confidence from scoring model
        - Model agreement (std dev of component scores)
        - Volatility
        """
        data_confidence = self.scores['confidence'].fillna(0.5)
        volatility = self.scores.get('volatility', pd.Series(0, index=self.scores.index)).fillna(0)
        vol_norm = (volatility / (volatility.max() + 1e-10)).clip(0, 1)

        # Include data quality flags in confidence calculation
        data_quality_penalty = pd.Series(1.0, index=self.scores.index)

        # Negative revenue flag penalty (20%) applied per-row
        if 'negative_revenue_flag' in self.scores.columns:
            neg = self.scores['negative_revenue_flag'].fillna(False).astype(float)
            data_quality_penalty = data_quality_penalty * (1.0 - 0.20 * neg)

        # Imputed penalty (15%)
        if 'imputed_flag' in self.scores.columns:
            imputed = self.scores['imputed_flag'].fillna(False).astype(float)
            data_quality_penalty = data_quality_penalty * (1.0 - 0.15 * imputed)

        # Growth anomaly penalty (25%)
        if 'growth_anomaly_flag' in self.scores.columns:
            anomaly = self.scores['growth_anomaly_flag'].fillna(False).astype(float)
            data_quality_penalty = data_quality_penalty * (1.0 - 0.25 * anomaly)

        # New entry / exit / inactive / partial presence penalties (reduce confidence)
        if 'is_new_entry' in self.scores.columns:
            new_e = self.scores['is_new_entry'].fillna(False).astype(float)
            data_quality_penalty = data_quality_penalty * (1.0 - 0.20 * new_e)
        if 'is_exit' in self.scores.columns:
            exit_f = self.scores['is_exit'].fillna(False).astype(float)
            data_quality_penalty = data_quality_penalty * (1.0 - 0.60 * exit_f)
        if 'is_inactive' in self.scores.columns:
            inactive = self.scores['is_inactive'].fillna(False).astype(float)
            data_quality_penalty = data_quality_penalty * (1.0 - 0.90 * inactive)
        if 'has_partial_presence' in self.scores.columns:
            partial = self.scores['has_partial_presence'].fillna(False).astype(float)
            data_quality_penalty = data_quality_penalty * (1.0 - 0.15 * partial)

        # Model agreement: low std dev = high agreement = high confidence
        component_scores = pd.DataFrame({
            'scoring': components['scoring_model'],
            'ml_class': components['ml_classification'],
            'xgb': components['xgb_forecast']
        })

        agreement = 1.0 - (component_scores.std(axis=1) / 50).clip(0, 1)

        # Combined confidence: data quality, data confidence, model agreement, and volatility penalty
        confidence = (
            0.40 * data_confidence * data_quality_penalty + 
            0.35 * agreement + 
            0.25 * (1 - vol_norm)
        )

        # Cap confidence minimum at 0.3 to avoid over-penalizing
        confidence = confidence.clip(lower=0.3, upper=1.0)
        return confidence
    
    def classify_confidence(self, confidence: pd.Series) -> pd.Series:
        """Classify confidence into High / Medium / Low."""
        # Use simple logic instead of pd.cut to avoid bin edge issues
        classification = pd.Series('Medium', index=confidence.index)
        classification[confidence >= self.CONFIDENCE_THRESHOLDS['high']] = 'High'
        classification[confidence < self.CONFIDENCE_THRESHOLDS['medium']] = 'Low'
        return classification
    
    def get_ensemble_dataframe(self) -> pd.DataFrame:
        """Return final ensemble scores + metadata."""
        ensemble_score, components = self.compute_ensemble_score()
        confidence = self.compute_confidence(ensemble_score, components)
        confidence_class = self.classify_confidence(confidence)
        
        # Build result
        result = pd.DataFrame(index=self.scores.index)
        
        # Ensemble output
        result['ensemble_score'] = ensemble_score
        result['ensemble_confidence'] = confidence
        result['ensemble_confidence_class'] = confidence_class
        
        # Component breakdown
        result['component_scoring_model'] = components['scoring_model']
        result['component_ml_classification'] = components['ml_classification']
        result['component_xgb_forecast'] = components['xgb_forecast']
        
        # Original scores
        result['final_score'] = self.scores['final_score']
        result['opportunity'] = self.scores['opportunity']
        result['risk'] = self.scores['risk']
        result['data_confidence'] = self.scores['confidence']
        
        # ML predictions
        result['rf_classification'] = self.ml['rf_classification']
        result['xgb_growth_forecast'] = self.ml['xgb_growth_forecast']
        result['cluster_segment'] = self.ml['cluster_segment']
        
        # Metadata
        result['molecule_id'] = self.scores['molecule_id']
        result['country'] = self.scores['country']
        result['sector'] = self.scores['sector']
        result['manufacturer'] = self.scores['manufacturer']
        result['molecule'] = self.scores['molecule']
        
        self.ensemble_df = result
        return result


# Example usage
if __name__ == "__main__":
    from data_layer import IQVIADataLoader
    from feature_engineering import FeatureEngineer
    from scoring_engine import ScoringEngine
    from ml_models import MLModels
    
    base_dir = Path(__file__).resolve().parent
    loader = IQVIADataLoader(base_dir / 'IQVIA - Leah .xlsx')
    loader.load()
    loader.validate_structure()
    clean_df = loader.clean()
    
    engineer = FeatureEngineer(clean_df)
    features = engineer.build_features()
    
    scorer = ScoringEngine(features)
    scores_df = scorer.get_scores_dataframe()
    
    ml = MLModels(features)
    ml.train_all_models()
    ml_df = ml.get_ml_predictions_dataframe()
    
    ensemble = EnsembleDecisionEngine(scores_df, ml_df)
    ensemble_result = ensemble.get_ensemble_dataframe()
    
    print("=== Ensemble Results ===")
    print(ensemble_result[[
        'molecule_id', 'ensemble_score', 'ensemble_confidence',
        'component_scoring_model', 'component_ml_classification',
        'component_xgb_forecast'
    ]].head(15))
    
    print("\n=== Score Distribution ===")
    print(ensemble_result['ensemble_score'].describe())
    
    print("\n=== Confidence Distribution ===")
    print(ensemble_result['ensemble_confidence_class'].value_counts())
