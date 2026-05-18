"""
Feature Engineering: Transform IQVIA data into strategic metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple

try:
    from config import THRESHOLDS
except ImportError:
    THRESHOLDS = {}

from logging_config import get_logger

_logger = get_logger(__name__)


class FeatureEngineer:
    """Build strategic metrics for pharmaceutical molecules."""
    
    # MAT revenue columns (LCD Manufacturer)
    REVENUE_COLS = [
        'MAT Q2 2023_LCD MNF',
        'MAT Q2 2024_LCD MNF',
        'MAT Q2 2025_LCD MNF'
    ]
    
    # MAT volume columns (Standard Units)
    VOLUME_COLS = [
        'MAT Q2 2023_Standard Units',
        'MAT Q2 2024_Standard Units',
        'MAT Q2 2025_Standard Units'
    ]
    
    # Time periods for growth calculation (years apart)
    TIME_PERIODS = [2023, 2024, 2025]
    
    def __init__(self, clean_df: pd.DataFrame):
        self.df = clean_df.copy()
        self.features = pd.DataFrame()
        # Ensure molecule_id exists
        if 'molecule_id' not in self.df.columns:
            manufacturer = self.df['Manufacturer'] if 'Manufacturer' in self.df.columns else pd.Series('Unknown', index=self.df.index)
            self.df['molecule_id'] = manufacturer.astype(str) + '_' + self.df['Molecule List'].astype(str) + '_' + self.df['Country'].astype(str)
    
    # ===== BASE METRICS  =====
    
    def compute_revenue_growth(self) -> pd.Series:
        """YoY revenue growth %."""
        revenue = self.df[self.REVENUE_COLS]
        # If base year is zero, we should not compute growth (emerging)
        base = revenue.iloc[:, 0]
        end = revenue.iloc[:, -1]
        growth = pd.Series(index=self.df.index, dtype=float)
        # Compute growth only where base > 0
        mask = base > 0
        growth.loc[mask] = ((end.loc[mask] - base.loc[mask]) / (base.loc[mask])) * 100
        growth.loc[~mask] = np.nan
        growth = growth.replace([np.inf, -np.inf], np.nan)
        return growth.clip(-100, 500)
    
    def compute_volume_growth(self) -> pd.Series:
        """YoY volume growth %."""
        volume = self.df[self.VOLUME_COLS]
        base = volume.iloc[:, 0]
        end = volume.iloc[:, -1]
        growth = pd.Series(index=self.df.index, dtype=float)
        mask = base > 0
        growth.loc[mask] = ((end.loc[mask] - base.loc[mask]) / (base.loc[mask])) * 100
        growth.loc[~mask] = np.nan
        growth = growth.replace([np.inf, -np.inf], np.nan)
        return growth.clip(-100, 500)
    
    def compute_market_size(self) -> pd.Series:
        """Latest MAT revenue (Q2 2025, millions)."""
        latest_revenue = self.df['MAT Q2 2025_LCD MNF'].fillna(0)
        return latest_revenue / 1e6  # Convert to millions
    
    def compute_competition_count(self) -> pd.Series:
        """Number of competitors (unique manufacturers) offering the same molecule in the same country.
        This counts distinct manufacturers for each `Country` x `Molecule List` combination.
        """
        if 'competition_count' in self.df.columns:
            return self.df['competition_count'].fillna(0)

        comp_count = self.df.groupby(['Country', 'Molecule List'])['Manufacturer'].transform('nunique')
        return comp_count.fillna(0)
    
    def compute_market_share(self) -> pd.Series:
        """
        Use market_share from data_layer (computed before aggregation).
        If data already aggregated, market_share = dominant manufacturer's share.
        Only recompute if market_share column not provided by data_layer.
        """
        # If data_layer already provided market_share, use it
        if 'market_share' in self.df.columns:
            return self.df['market_share'].clip(0, 100).fillna(0)
        
        # Fallback: recompute if not provided (shouldn't happen with fixed pipeline)
        # Group by Country + Molecule to get molecule-level totals
        molecule_totals = self.df.groupby(['Country', 'Molecule List'])['MAT Q2 2025_LCD MNF'].transform('sum')
        
        # Each row's revenue / molecule total = market share
        molecule_rev = self.df['MAT Q2 2025_LCD MNF'].fillna(0)
        market_share = (molecule_rev / (molecule_totals + 1e-10)) * 100
        
        # Clip to valid range [0, 100] to handle negative adjustments or rounding
        market_share = market_share.clip(0, 100)
        return market_share.fillna(0)
    
    def compute_price_per_unit(self) -> pd.Series:
        """
        Handle zero volume properly - set price = NaN when volume is 0.
        Don't use fillna(1) which creates fake prices.
        """
        revenue = self.df['MAT Q2 2025_LCD MNF'].fillna(0)
        volume = self.df['MAT Q2 2025_Standard Units'].fillna(0)
        
        price_per_unit = revenue / (volume + 1e-10)
        
        # Where volume is 0 or very small, set price to NaN (not a fake price)
        price_per_unit = price_per_unit.where(volume > 0.1, np.nan)
        
        return price_per_unit.replace([np.inf, -np.inf], np.nan)
    
    def compute_price_change(self) -> pd.Series:
        """Price trend: YoY price per unit change %.
        FIX #4: Handle zero volume properly - set price = NaN when volume is 0.
        """
        revenue_2023 = self.df['MAT Q2 2023_LCD MNF'].fillna(0)
        volume_2023 = self.df['MAT Q2 2023_Standard Units'].fillna(0)
        
        # Where volume is zero, price is NaN (not fake 1)
        price_2023 = revenue_2023 / (volume_2023 + 1e-10)
        price_2023 = price_2023.where(volume_2023 > 0.1, np.nan)
        
        revenue_2025 = self.df['MAT Q2 2025_LCD MNF'].fillna(0)
        volume_2025 = self.df['MAT Q2 2025_Standard Units'].fillna(0)
        price_2025 = revenue_2025 / (volume_2025 + 1e-10)
        price_2025 = price_2025.where(volume_2025 > 0.1, np.nan)
        
        price_change = ((price_2025 - price_2023) / (price_2023 + 1e-10)) * 100
        price_change = price_change.replace([np.inf, -np.inf], np.nan).fillna(0)
        return price_change.clip(-100, 100)
    
    def compute_volatility(self) -> pd.Series:
        """Revenue volatility: std dev of YoY growth."""
        revenue = self.df[self.REVENUE_COLS].fillna(0)
        
        # Calculate YoY growth for each period
        yoy_growth = []
        for i in range(len(revenue.columns) - 1):
            growth = (revenue.iloc[:, i + 1] - revenue.iloc[:, i]) / (revenue.iloc[:, i] + 1e-10)
            yoy_growth.append(growth)
        
        yoy_df = pd.DataFrame(yoy_growth).T
        volatility = yoy_df.std(axis=1).fillna(0)
        return volatility
    
    # ===== ADVANCED METRICS  =====
    
    def compute_hhi_market_concentration(self) -> pd.Series:
        """
        Real HHI using market share (Herfindahl-Hirschman Index).
        HHI = sum(market_share_i^2) for all competitors
        > 2500 = highly concentrated, < 1500 = competitive
        """
        if 'hhi' in self.df.columns:
            return self.df['hhi'].fillna(2500)

        # Get market shares at country-molecule level
        market_share = self.compute_market_share()  # Returns 0-100 percentage
        market_share_frac = market_share / 100  # Convert to fraction
        
        # Group by Country + Molecule, compute HHI as sum of squared shares
        def compute_row_hhi(group):
            return (group ** 2).sum() * 10000  # Scale to 0-10000 range
        
        hhi = market_share_frac.groupby([self.df['Country'], self.df['Molecule List']]).transform(lambda g: compute_row_hhi(g))
        return hhi.fillna(2500)  # Default: moderate concentration
    
    def compute_cagr(self) -> pd.Series:
        """
        Compound Annual Growth Rate (2023-2025, 2 years).
        CAGR = (Ending / Beginning) ^ (1 / years) - 1
        """
        revenue_start = self.df['MAT Q2 2023_LCD MNF']
        revenue_end = self.df['MAT Q2 2025_LCD MNF']

        # If start is zero, set CAGR to NaN (handled as emerging)
        cagr = pd.Series(index=self.df.index, dtype=float)
        mask = revenue_start > 0
        cagr.loc[mask] = np.power(revenue_end.loc[mask] / revenue_start.loc[mask], 1/2) - 1
        cagr.loc[~mask] = np.nan
        cagr = cagr.replace([np.inf, -np.inf], np.nan)
        return cagr * 100  # Return as percentage
    
    def compute_revenue_sustainability_index(self) -> pd.Series:
        """
        Sustainability = consistency of growth across periods.
        High if no reversals; low if oscillating.
        Score 0-100.
        """
        revenue = self.df[self.REVENUE_COLS].fillna(0)
        
        # Count periods with positive growth
        growth_positive = []
        for i in range(len(revenue.columns) - 1):
            pos = (revenue.iloc[:, i + 1] > revenue.iloc[:, i]).astype(int)
            growth_positive.append(pos)
        
        growth_df = pd.DataFrame(growth_positive).T
        # Sustainability = proportion of periods with positive growth
        sustainability = (growth_df.sum(axis=1) / len(growth_df.columns)) * 100
        return sustainability
    
    def compute_generic_erosion_index(self) -> pd.Series:
        """
        Proxy for generic pressure.
        If: price declining + volume growing = generic erosion.
        Index: price_change - volume_growth (inverted, lower is worse).
        """
        price_chg = self.compute_price_change()
        vol_growth = self.compute_volume_growth()
        
        # Erosion = (price decline - volume gain) normalized
        erosion_index = price_chg - vol_growth
        erosion_index = 100 - np.abs(erosion_index)  # Invert so higher is better
        return erosion_index.clip(0, 100)
    
    def compute_market_penetration_proxy(self) -> pd.Series:
        """
        Proxy for market penetration within segment.
        High if: market_share > segment average AND growing.
        """
        market_share = self.compute_market_share()
        revenue_growth = self.compute_revenue_growth()
        
        # Simple penetration = market_share * growth factor
        # Normalize growth to 1.0 baseline
        growth_factor = 1 + (revenue_growth / 100).clip(-0.5, 2.0)
        
        # Penetration proxy
        penetration = (market_share / 100) * growth_factor * 100
        penetration = penetration.fillna(0).clip(0, 100)
        return penetration
    
    def compute_channel_dependency(self) -> pd.Series:
        """
        Channel dependency is a PROXY using Sector, which may not represent real channels.
        
        ASSUMPTION: Sector (HOSPITAL/RETAIL) encodes distribution channel.
        If this is incorrect, this feature should be removed from scoring.
        
        Returns max channel share per molecule (0-1).
        Higher = dependent on single channel = risky.
        """
        # Note: This assumes Sector = distribution channel (HOSPITAL/RETAIL)
        # If Sector is not a true channel indicator in your data, remove this feature
        # from the feature matrix before modeling
        
        rev = self.df.copy()
        rev_col = 'MAT Q2 2025_LCD MNF'
        
        # Skip if Sector is missing
        if 'Sector' not in rev.columns or rev['Sector'].isna().all():
            # Return neutral value if no channel data
            return pd.Series(0.5, index=self.df.index)
        
        grp = rev.groupby(['Country', 'Molecule List', 'Sector'])[rev_col].transform('sum')
        total_by_mol = rev.groupby(['Country', 'Molecule List'])[rev_col].transform('sum') + 1e-10

        channel_share = grp / total_by_mol
        channel_dependency = channel_share.groupby([rev['Country'], rev['Molecule List']]).transform('max')

        return channel_dependency.fillna(0.5)  # Neutral if no data
    
    # ===== DERIVED METRICS =====
    
    def compute_lifecycle_stage(self) -> pd.Series:
        """
        Use explicit, interpretable thresholds instead of rank-based bins.
        Market Lifecycle Classification: Emerging / Growth / Mature / Declining.
        Based on: revenue growth + market size thresholds.
        """
        market_size = self.compute_market_size()  # In millions
        revenue_growth = self.compute_revenue_growth()  # In %
        
        # Clean inputs
        rg = revenue_growth.replace([np.inf, -np.inf], np.nan)
        ms = market_size.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Incorporate structural flags if available
        is_new = self.df.get('is_new_entry', pd.Series(False, index=self.df.index))
        is_exit = self.df.get('is_exit', pd.Series(False, index=self.df.index))
        is_inactive = self.df.get('is_inactive', pd.Series(False, index=self.df.index))
        # Use explicit thresholds for interpretability
        # Use config thresholds instead of hardcoded values
        emerging_growth = THRESHOLDS.get('lifecycle_emerging_growth', 0.20) * 100
        growth_threshold = THRESHOLDS.get('lifecycle_growth_growth', 0.15) * 100
        decline_threshold = THRESHOLDS.get('lifecycle_decline_threshold', -0.10) * 100
        emerging_size = THRESHOLDS.get('lifecycle_emerging_market_size', 50)
        mid_min = THRESHOLDS.get('lifecycle_mid_market_min', 50)
        mid_max = THRESHOLDS.get('lifecycle_mid_market_max', 500)

        stage = pd.Series('MATURE', index=self.df.index)

        # Exited and inactive overrides
        stage[is_inactive] = 'EXITED'
        stage[is_exit] = 'EXITED'

        # EMERGING: New markets or structural zero-base with subsequent growth
        stage[is_new] = 'EMERGING'
        stage[(ms < emerging_size) & (rg > emerging_growth) & (~is_new) & (~is_exit)] = 'EMERGING'

        # GROWTH: Mid/large markets with strong growth
        stage[(~is_inactive) & (ms >= mid_min) & (ms < mid_max) & (rg > growth_threshold)] = 'GROWTH'
        stage[(~is_inactive) & (ms >= mid_max) & (rg > emerging_growth)] = 'GROWTH'

        # DECLINING: Negative growth and not exited
        stage[(~is_inactive) & (rg < decline_threshold) & (~is_exit)] = 'DECLINING'

        # MATURE: default

        return stage
    
    def compute_risk_score(self) -> pd.Series:
        """
        Risk Score (0-100, higher = more risk).
        Inputs: Competition, HHI, Volatility, Price erosion.
        """
        competition = self.compute_competition_count()
        hhi = self.compute_hhi_market_concentration()
        volatility = self.compute_volatility()
        price_change = self.compute_price_change()
        
        # Normalize to 0-1
        comp_norm = (competition / (competition.max() + 1e-10)).fillna(0)
        hhi_norm = (hhi / 10000).clip(0, 1)
        vol_norm = (volatility / (volatility.max() + 1e-10)).fillna(0)
        price_norm = (np.abs(price_change) / 100).clip(0, 1)
        
        # Weighted risk
        risk = (0.25 * comp_norm + 0.30 * hhi_norm + 0.25 * vol_norm + 0.20 * price_norm) * 100
        return risk
    
    def compute_opportunity_score(self) -> pd.Series:
        """
        Opportunity Score (0-100, higher = more opportunity).
        Inputs: Growth, Market Size, Market Penetration.
        """
        revenue_growth = self.compute_revenue_growth()
        market_size = self.compute_market_size()
        
        # Normalize
        growth_norm = ((revenue_growth + 100) / 200).clip(0, 1)  # -100% to +100% range
        size_norm = (market_size / (market_size.max() + 1e-10)).fillna(0)
        
        # Weighted opportunity
        opportunity = (0.50 * growth_norm + 0.50 * size_norm) * 100
        return opportunity
    
    def compute_entry_barrier_score(self) -> pd.Series:
        """
        Entry Barrier Score (0-100, higher = harder to enter).
        Inputs: HHI (monopoly), Market Share (incumbent strength), Price stability.
        """
        hhi = self.compute_hhi_market_concentration()
        market_share = self.compute_market_share()
        price_stability = 100 - np.abs(self.compute_price_change())
        
        # Normalize
        hhi_norm = (hhi / 10000).clip(0, 1)
        share_norm = (market_share / 100).clip(0, 1)
        stability_norm = (price_stability / 100).clip(0, 1)
        
        # Weighted barrier
        barrier = (0.40 * hhi_norm + 0.35 * share_norm + 0.25 * stability_norm) * 100
        return barrier
    
    def build_features(self) -> pd.DataFrame:
        """Build and return all features as DataFrame."""
        features = pd.DataFrame(index=self.df.index)
        
        # Base Metrics
        features['revenue_growth'] = self.compute_revenue_growth()
        features['volume_growth'] = self.compute_volume_growth()
        features['market_size'] = self.compute_market_size()
        features['competition_count'] = self.compute_competition_count()
        features['market_share'] = self.compute_market_share()
        features['price_per_unit'] = self.compute_price_per_unit()
        features['price_change'] = self.compute_price_change()
        features['volatility'] = self.compute_volatility()
        
        # Advanced Metrics
        features['hhi'] = self.compute_hhi_market_concentration()
        features['cagr'] = self.compute_cagr()
        features['revenue_sustainability'] = self.compute_revenue_sustainability_index()
        features['generic_erosion_index'] = self.compute_generic_erosion_index()
        features['market_penetration'] = self.compute_market_penetration_proxy()
        features['channel_dependency'] = self.compute_channel_dependency()
        
        # Derived Metrics
        features['lifecycle_stage'] = self.compute_lifecycle_stage()
        features['risk_score'] = self.compute_risk_score()
        features['opportunity_score'] = self.compute_opportunity_score()
        features['entry_barrier_score'] = self.compute_entry_barrier_score()
        
        # Add hierarchy info
        if 'molecule_id' in self.df.columns:
            features['molecule_id'] = self.df['molecule_id']
        else:
            manufacturer = self.df['Manufacturer'] if 'Manufacturer' in self.df.columns else pd.Series('Unknown', index=self.df.index)
            features['molecule_id'] = manufacturer.astype(str) + '_' + self.df['Molecule List'].astype(str) + '_' + self.df['Country'].astype(str)
        features['country'] = self.df['Country'].fillna('Unknown')
        features['sector'] = self.df['Sector'].fillna('Unknown')
        features['manufacturer'] = self.df['Manufacturer'].fillna('Unknown') if 'Manufacturer' in self.df.columns else pd.Series('Unknown', index=self.df.index)
        features['molecule'] = self.df['Molecule List'].fillna('Unknown')
        
        self.features = features
        
        # Validate feature consistency
        self._validate_feature_consistency()
        return features
    
    def _validate_feature_consistency(self):
        """
        Check that features are not dominated by zeros/NaNs.
        Warn if any numeric feature has >70% zeros or NaNs.
        """
        numeric_features = self.features.select_dtypes(include=[np.number]).columns
        
        for feat in numeric_features:
            non_null_count = self.features[feat].notna().sum()
            zero_count = (self.features[feat] == 0).sum()
            zero_pct = zero_count / len(self.features)
            
            if zero_pct > 0.70:
                _logger.warning("feature_high_zero_pct", feature=feat, zero_pct=round(zero_pct * 100, 1))


# Example usage
if __name__ == "__main__":
    from data_layer import IQVIADataLoader
    
    base_dir = Path(__file__).resolve().parent
    loader = IQVIADataLoader(base_dir / 'IQVIA - Leah .xlsx')
    loader.load()
    loader.validate_structure()
    clean_df = loader.clean()
    
    engineer = FeatureEngineer(clean_df)
    features = engineer.build_features()
    
    print("=== Features Sample ===")
    print(features[['molecule_id', 'revenue_growth', 'market_size', 'risk_score', 'opportunity_score']].head(10))
    
    print("\n=== Feature Statistics ===")
    print(features.describe())

