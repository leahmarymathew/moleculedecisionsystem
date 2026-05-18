"""
Scoring Engine: Dual-score framework per roadmap Section 5.
Implements business rules, penalties, and final ranking score.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Import weights and thresholds from central config
try:
    from config import WEIGHTS
    from config import THRESHOLDS
    from config import RISK_WEIGHTS
except Exception:
    WEIGHTS = {
        'growth': 0.25,
        'market_size': 0.20,
        'competition': 0.25,
        'pricing_power': 0.15,
        'stability': 0.15
    }
    THRESHOLDS = {
        'monopoly_share': 0.8,
        'small_market_size': 1.0,
        'decline_cagr_pct': -0.20
    }
    RISK_WEIGHTS = {
        'competition': 0.25,
        'concentration': 0.25,
        'volatility': 0.15,
        'pricing': 0.15
    }


class ScoringEngine:
    """
    Dual-score model: Opportunity - Risk = Final Score.
    Incorporates penalty mechanisms per Section 5.5.
    """
    
    # ===== WEIGHTS  =====
    WEIGHTS = WEIGHTS
    
    # ===== BUSINESS LOGIC CONSTRAINTS  =====
    OPTIMAL_COMPETITION_MIN = 3
    OPTIMAL_COMPETITION_MAX = 15
    MONOPOLY_SHARE_THRESHOLD = 0.80  # > 80% = penalty
    
    def __init__(self, features_df: pd.DataFrame):
        """
        Args:
            features_df: DataFrame from FeatureEngineer.build_features()
        """
        self.features = features_df.copy()
        self.scores = pd.DataFrame(index=features_df.index)
        self.penalties = {}
    
    # ===== NORMALIZATION & SCALING  =====
    
    @staticmethod
    def log_scale(values: pd.Series, base: float = 10) -> pd.Series:
        """Log transform for market size."""
        return np.log10(values + 1)
    
    @staticmethod
    def s_curve(values: pd.Series, midpoint: float = 50, steepness: float = 0.1) -> pd.Series:
        """
        S-curve for growth saturation.
        Flattens extreme values to 0-1 range.
        """
        return 1 / (1 + np.exp(-steepness * (values - midpoint)))
    
    @staticmethod
    def exponential_penalty(hhi: pd.Series, threshold: float = 0.80) -> pd.Series:
        """
        Exponential penalty for monopolistic markets (HHI > 2500).
        hhi: normalized HHI 0-1 (where 1 = monopoly)
        Returns: penalty multiplier (0-1, where 0 = severe penalty)
        """
        # Normalize HHI to 0-1 (assuming max HHI = 10000)
        hhi_norm = hhi / 10000
        
        # Exponential penalty if HHI > 0.25 (2500)
        penalty = np.where(hhi_norm > 0.25, np.exp(-2 * (hhi_norm - 0.25)), 1.0)
        return penalty
    
    # ===== OPPORTUNITY SCORE =====
    
    def compute_growth_component(self) -> Tuple[pd.Series, Dict]:
        """
        Use dataset-dependent midpoint instead of fixed 50.
        Growth component: 25% weight.
        Uses S-curve to saturate extreme growth.
        """
        revenue_growth = self.features['revenue_growth']
        
        # Use dataset-dependent midpoint instead of fixed 50
        median_growth = revenue_growth.median()
        
        # S-curve transformation: cap unrealistic growth
        growth_scaled = self.s_curve(revenue_growth, midpoint=median_growth, steepness=0.05)
        
        # Penalize declining (< -10%) hard
        is_declining = revenue_growth < -10
        growth_scaled[is_declining] *= 0.3
        
        # Normalize to 0-100
        # Handle NaNs for structural cases: new entries -> allow high growth signal; inactive -> low
        is_new = self.features.get('is_new_entry', pd.Series(False, index=self.features.index)).fillna(False)
        is_inactive = self.features.get('is_inactive', pd.Series(False, index=self.features.index)).fillna(False)

        # Set default for NaN scaled values
        # For new entries, encourage higher scaled growth (e.g., 0.8) but confidence is reduced elsewhere
        growth_scaled = growth_scaled.copy()
        growth_scaled[pd.isna(growth_scaled) & is_new] = 0.8
        # For inactive or explicit NaNs not new, set to 0
        growth_scaled[pd.isna(growth_scaled)] = 0.0

        # Inactive markets should have very low growth score
        growth_scaled[is_inactive] = 0.0

        growth_score = growth_scaled * 100
        
        return growth_score, {'raw': revenue_growth, 'scaled': growth_scaled, 'midpoint_used': median_growth}
    
    def compute_market_size_component(self) -> Tuple[pd.Series, Dict]:
        """
        Use min-max normalization instead of hardcoded division by 6.0.
        Market size component: 20% weight.
        Uses log scale (diminishing returns for huge markets).
        """
        market_size = self.features['market_size']
        
        # Log scale
        size_log = self.log_scale(market_size)
        
        # Use min-max normalization instead of division by 6.0
        log_min = size_log.min()
        log_max = size_log.max()
        log_range = log_max - log_min + 1e-10
        
        size_scaled = ((size_log - log_min) / log_range).clip(0, 1)
        
        # Penalize very small markets (< 1M)
        is_small = market_size < 1
        size_scaled[is_small] *= 0.5
        
        size_score = size_scaled * 100
        
        return size_score, {'raw': market_size, 'log_scaled': size_log, 'normalized': size_scaled}
    
    def compute_demand_component(self) -> Tuple[pd.Series, Dict]:
        """
        Demand expansion component: Market penetration proxy.
        Part of Opportunity (growth + size + demand = opportunity).
        """
        penetration = self.features['market_penetration'].fillna(0)
        volume_growth = self.features['volume_growth'].fillna(0)
        
        # Demand = penetration + volume growth momentum
        demand_score = (0.6 * (penetration / 100) + 0.4 * self.s_curve(volume_growth, midpoint=25)) * 100
        
        return demand_score, {'penetration': penetration, 'volume_growth': volume_growth}
    
    def compute_opportunity_score(self) -> pd.Series:
        """
        Opportunity Score = f(Growth 25%, Size 20%, Demand/Penetration)
        """
        growth, _ = self.compute_growth_component()
        size, _ = self.compute_market_size_component()
        demand, _ = self.compute_demand_component()
        
        # Weighted combination
        opportunity = (
            self.WEIGHTS['growth'] * growth +
            self.WEIGHTS['market_size'] * size +
            (1 - sum([self.WEIGHTS['growth'], self.WEIGHTS['market_size']])) * demand
        )
        
        return opportunity
    
    # ===== RISK SCORE =====
    
    def compute_competition_risk(self) -> Tuple[pd.Series, Dict]:
        """
        Competition risk: 25% weight.
        Optimal: 3-15 competitors. Penalty for < 3 or > 15.
        Severe penalty for > 80% share (monopoly).
        """
        competition = self.features['competition_count']
        market_share = self.features['market_share']
        
        # Continuous competition risk using s-curve on competitor count
        # More competitors => generally lower competition risk up to a point
        comp_count = competition.fillna(0).astype(float)

        # Normalize competitor count around an empirical midpoint
        midpoint = max(3.0, comp_count.median())
        comp_strength = 1.0 / (1.0 + np.exp(-(comp_count - midpoint) / (max(1.0, comp_count.std()) + 1e-6)))

        # Market share as fraction (ensure consistent units)
        ms = market_share.copy().astype(float)
        if ms.max() <= 1.0:
            ms = ms * 100.0

        monopoly_frac = ms / 100.0

        # Higher monopoly_frac increases risk
        monopoly_risk = np.where(monopoly_frac > THRESHOLDS.get('monopoly_share', 0.8),
                                 (monopoly_frac - THRESHOLDS.get('monopoly_share', 0.8)) / (1 - THRESHOLDS.get('monopoly_share', 0.8)),
                                 0.0)

        # Combine: low comp_strength (few competitors) increases risk, monopoly increases risk
        competition_risk = (1.0 - comp_strength) + monopoly_risk
        competition_risk = np.clip(competition_risk, 0.0, 1.0)

        competition_risk_score = competition_risk * 100

        return competition_risk_score, {'competition': competition, 'monopoly_risk': monopoly_risk}
    
    def compute_concentration_risk(self) -> Tuple[pd.Series, Dict]:
        """
        Concentration risk: HHI-based.
        HHI > 2500 = highly concentrated (risky).
        """
        hhi = self.features['hhi'].fillna(2500)
        
        # Exponential penalty for high HHI
        hhi_penalty = self.exponential_penalty(hhi, threshold=2500)
        
        # Risk = 1 - penalty
        concentration_risk = (1 - hhi_penalty) * 100
        
        return concentration_risk, {'hhi': hhi, 'penalty': hhi_penalty}
    
    def compute_volatility_risk(self) -> Tuple[pd.Series, Dict]:
        """
        Volatility risk: 15% weight.
        High volatility = unpredictable market.
        """
        volatility = self.features['volatility'].fillna(0)
        
        # Normalize volatility (cap at 1.0 for very volatile)
        volatility_norm = (volatility / (volatility.max() + 1e-10)).clip(0, 1)
        
        volatility_risk = volatility_norm * 100
        
        return volatility_risk, {'volatility': volatility, 'normalized': volatility_norm}
    
    def compute_pricing_risk(self) -> Tuple[pd.Series, Dict]:
        """
        Pricing risk: 15% weight.
        High price erosion (< -20%) = high risk.
        """
        price_change = self.features['price_change'].fillna(0)
        
        # Risk from aggressive price decline
        pricing_risk = np.where(price_change < -20, np.abs(price_change) / 100, 0)
        pricing_risk = pricing_risk.clip(0, 1) * 100
        
        return pricing_risk, {'price_change': price_change}
    
    def compute_risk_score(self) -> pd.Series:
        """
        Use config weights, not hardcoded 0.25 values.
        Risk Score = f(Competition 25%, Concentration 25%, Volatility 15%, Pricing 15%)
        """
        comp_risk, _ = self.compute_competition_risk()
        conc_risk, _ = self.compute_concentration_risk()
        vol_risk, _ = self.compute_volatility_risk()
        price_risk, _ = self.compute_pricing_risk()
        
        # Use config weights instead of hardcoded values
        risk_weights = RISK_WEIGHTS
        
        # Normalize weights to sum to 1
        total_weight = sum(risk_weights.values())
        risk_weights = {k: v/total_weight for k, v in risk_weights.items()}
        
        risk = (
            risk_weights.get('competition', 0.25) * comp_risk +
            risk_weights.get('concentration', 0.25) * conc_risk +
            risk_weights.get('volatility', 0.15) * vol_risk +
            risk_weights.get('pricing', 0.15) * price_risk
        )
        
        return risk
    
    # ===== FINAL SCORE & PENALTIES (Section 5.5) =====
    
    def apply_penalties(self) -> pd.DataFrame:
        """
        Use config thresholds to normalize penalties and prevent stacking from distorting scores.
        Apply penalty mechanisms:
        - Monopoly (> 80%) → heavy penalty
        - Declining revenue (< -20%) → negative adjustment
        - Small market (< 1M) → capped score
        - Data anomaly → confidence reduction
        """
        penalties = pd.DataFrame(index=self.features.index)

        # Continuous monopoly penalty scaled to how far above threshold
        ms = self.features['market_share'].astype(float)
        if ms.max() <= 1.0:
            ms = ms * 100.0
        monopoly_thresh = THRESHOLDS.get('monopoly_share', 0.8) * 100.0
        monopoly_excess = (ms - monopoly_thresh).clip(lower=0)
        penalties['monopoly_penalty'] = -30.0 * (monopoly_excess / (100.0 - monopoly_thresh + 1e-6))

        # Declining revenue penalty scaled by decline depth
        decline_thresh = THRESHOLDS.get('decline_cagr_pct', -0.20) * 100.0
        rev_growth = self.features.get('revenue_growth', pd.Series(0, index=self.features.index)).astype(float)
        decline_amount = (-rev_growth - (-decline_thresh)).clip(lower=0)
        penalties['declining_penalty'] = -20.0 * (decline_amount / (abs(decline_thresh) + 1e-6))

        # Small market penalty (continuous inverse of log size)
        small_thresh = THRESHOLDS.get('small_market_size', 1.0)
        msz = self.features.get('market_size', pd.Series(0, index=self.features.index)).astype(float)
        penalties['small_market_penalty'] = -15.0 * (1.0 - np.tanh(msz / (small_thresh + 1e-6)))

        # Generic erosion: continuous based on price decline vs volume growth
        price_drop = (-self.features.get('price_change', 0).clip(upper=0)).astype(float)
        vol_gain = self.features.get('volume_growth', 0).astype(float).clip(lower=0)
        erosion_score = (price_drop / (price_drop.max() + 1e-6)) * (vol_gain / (vol_gain.max() + 1e-6))
        penalties['generic_erosion_penalty'] = -10.0 * erosion_score

        # Use config thresholds to cap total penalty and prevent stacking distortion
        # Normalize by dividing by number of penalties (soft normalization)
        individual_penalties = penalties[[c for c in penalties.columns if 'penalty' in c]]
        penalties['total_penalty'] = individual_penalties.sum(axis=1)
        
        # Additional structural penalties based on lifecycle flags
        # Strong penalty for exits, small penalty for new entries and partial presence
        is_exit = self.features.get('is_exit', pd.Series(False, index=self.features.index)).astype(bool)
        is_new = self.features.get('is_new_entry', pd.Series(False, index=self.features.index)).astype(bool)
        partial = self.features.get('has_partial_presence', pd.Series(False, index=self.features.index)).astype(bool)
        is_inactive = self.features.get('is_inactive', pd.Series(False, index=self.features.index)).astype(bool)

        # Define penalties (negative values)
        penalties['exit_penalty'] = -80.0 * is_exit.astype(float)
        penalties['new_entry_penalty'] = -10.0 * is_new.astype(float)
        penalties['partial_presence_penalty'] = -10.0 * partial.astype(float)
        # inactive handled as override in final scoring (force near-zero)
        penalties['inactive_penalty'] = -100.0 * is_inactive.astype(float)

        # Include exit/new/partial in total penalty sum but not inactive (override)
        penalties['total_penalty'] = penalties['total_penalty'] + penalties['exit_penalty'] + penalties['new_entry_penalty'] + penalties['partial_presence_penalty']

        # Cap total penalty to -100 (max deduction)
        penalties['total_penalty'] = penalties['total_penalty'].clip(lower=-100)

        self.penalties = penalties
        return penalties
    
    def compute_final_score(self) -> pd.Series:
        """
        Use config thresholds to explicitly bound final score to [-100, 100].
        Final Score = Opportunity - Risk + Penalties
        """
        opportunity = self.compute_opportunity_score()
        risk = self.compute_risk_score()
        penalties = self.apply_penalties()
        
        # Raw final score (may be unbounded)
        raw_final = opportunity - risk + penalties['total_penalty']

        # Store raw components
        self.scores['opportunity_raw'] = opportunity
        self.scores['risk_raw'] = risk
        self.scores['penalties_raw'] = penalties['total_penalty']
        self.scores['final_raw'] = raw_final

        # Use config thresholds to explicitly bound final score to [-100, 100]
        # Then rescale to 0-100 for output (where 50 = neutral)
        clipped = np.clip(raw_final, -100, 100)

        # Use config bounds instead of hardcoded values
        score_min = THRESHOLDS.get('final_score_min', -100)
        score_max = THRESHOLDS.get('final_score_max', 100)
        clipped = np.clip(raw_final, score_min, score_max)

        # Rescale to [0, 100] (where 50 = 0 raw)
        midpoint = (score_max - score_min) / 2
        scaled = 50 + ((clipped - score_min - midpoint) / (score_max - score_min)) * 100
        scaled = np.clip(scaled, 0, 100)

        # Override: For inactive molecules, force near-zero score
        is_inactive = self.features.get('is_inactive', pd.Series(False, index=self.features.index)).astype(bool)
        if is_inactive.any():
            # set to very low score (1) to keep ranking but near-zero
            scaled.loc[is_inactive] = 1.0

        # For exits, ensure strong down-weight if not already low
        is_exit = self.features.get('is_exit', pd.Series(False, index=self.features.index)).astype(bool)
        if is_exit.any():
            # reduce exit scores to 20% of original (but at least 1)
            scaled.loc[is_exit] = (scaled.loc[is_exit] * 0.2).clip(lower=1.0)
        self.scores['final_score'] = scaled
        return scaled
    
    def compute_confidence(self) -> pd.Series:
        """
        Confidence in the score (0.0 to 1.0).
        Based on: data completeness, model stability, volatility.
        """
        volatility = self.features['volatility']
        
        # Start with full confidence
        confidence = pd.Series(1.0, index=self.features.index)
        
        # Reduce confidence for high volatility
        confidence -= (volatility / (volatility.max() + 1e-10)) * 0.3
        
        # Reduce confidence for small markets (less reliable data)
        market_size_norm = self.features['market_size'] / (self.features['market_size'].max() + 1e-10)
        confidence -= (1 - market_size_norm) * 0.2
        
        confidence = confidence.clip(lower=0.3, upper=1.0)  # Min 30% confidence
        
        return confidence
    
    def get_scores_dataframe(self) -> pd.DataFrame:
        """Return comprehensive scores DataFrame."""
        result = self.scores.copy()
        # Backward-compatible aliases for downstream layers
        if 'opportunity' not in result.columns and 'opportunity_raw' in result.columns:
            result['opportunity'] = result['opportunity_raw']
        if 'risk' not in result.columns and 'risk_raw' in result.columns:
            result['risk'] = result['risk_raw']
        if 'penalties' not in result.columns and 'penalties_raw' in result.columns:
            result['penalties'] = result['penalties_raw']
        result['confidence'] = self.compute_confidence()
        # Expose a few raw drivers for downstream calibration and validation
        if 'volatility' not in result.columns and 'volatility' in self.features.columns:
            result['volatility'] = self.features['volatility']
        if 'revenue_growth' not in result.columns and 'revenue_growth' in self.features.columns:
            result['revenue_growth'] = self.features['revenue_growth']
        if 'market_size' not in result.columns and 'market_size' in self.features.columns:
            result['market_size'] = self.features['market_size']
        result['molecule_id'] = self.features['molecule_id']
        result['country'] = self.features['country']
        result['sector'] = self.features['sector']
        result['manufacturer'] = self.features['manufacturer']
        result['molecule'] = self.features['molecule']
        # Expose structural flags for downstream layers
        for flag in ['is_new_entry', 'is_exit', 'is_inactive', 'has_partial_presence', 'negative_revenue_flag', 'growth_anomaly_flag', 'imputed_flag']:
            if flag in self.features.columns:
                result[flag] = self.features[flag]
            elif flag in self.scores.index and flag in self.scores.columns:
                result[flag] = self.scores.get(flag)
            else:
                # default False if unavailable
                result[flag] = False
        
        return result


# Example usage
if __name__ == "__main__":
    from data_layer import IQVIADataLoader
    from feature_engineering import FeatureEngineer
    
    base_dir = Path(__file__).resolve().parent
    loader = IQVIADataLoader(base_dir / 'IQVIA - Leah .xlsx')
    loader.load()
    loader.validate_structure()
    clean_df = loader.clean()
    
    engineer = FeatureEngineer(clean_df)
    features = engineer.build_features()
    
    scorer = ScoringEngine(features)
    final_scores = scorer.compute_final_score()
    
    print("=== Final Scores ===")
    scores_df = scorer.get_scores_dataframe()
    print(scores_df[['molecule_id', 'opportunity', 'risk', 'final_score', 'confidence']].head(20))
    
    print("\n=== Score Distribution ===")
    print(final_scores.describe())
