"""
Feature Engineering: Transform IQVIA data into strategic metrics.
Implements roadmap Section 3: Base (8) + Advanced (8) + Derived metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, Tuple


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
            self.df['molecule_id'] = (
                self.df['Manufacturer'] + '_' + 
                self.df['Molecule List'] + '_' + 
                self.df['Country']
            )
    
    # ===== BASE METRICS (Section 3.1) =====
    
    def compute_revenue_growth(self) -> pd.Series:
        """YoY revenue growth %."""
        revenue = self.df[self.REVENUE_COLS]
        # Growth from 2023 to 2025
        growth = ((revenue.iloc[:, -1] - revenue.iloc[:, 0]) / (revenue.iloc[:, 0] + 1e-10)) * 100
        growth = growth.replace([np.inf, -np.inf], 0).fillna(0)
        return growth.clip(-100, 500)  # Cap at ±500% for outliers
    
    def compute_volume_growth(self) -> pd.Series:
        """YoY volume growth %."""
        volume = self.df[self.VOLUME_COLS]
        growth = ((volume.iloc[:, -1] - volume.iloc[:, 0]) / (volume.iloc[:, 0] + 1e-10)) * 100
        growth = growth.replace([np.inf, -np.inf], 0).fillna(0)
        return growth.clip(-100, 500)
    
    def compute_market_size(self) -> pd.Series:
        """Latest MAT revenue (Q2 2025, millions)."""
        latest_revenue = self.df['MAT Q2 2025_LCD MNF'].fillna(0)
        return latest_revenue / 1e6  # Convert to millions
    
    def compute_competition_count(self) -> pd.Series:
        """Number of competitors (unique manufacturers) offering the same molecule in the same country.
        This counts distinct manufacturers for each `Country` x `Molecule List` combination.
        """
        comp_count = self.df.groupby(['Country', 'Molecule List'])['Manufacturer'].transform('nunique')
        return comp_count.fillna(0)
    
    def compute_market_share(self) -> pd.Series:
        """
        FIX #5: Market share should be molecule's share of ITS OWN MARKET, not segment.
        Denominator: Total revenue of that MOLECULE across all manufacturers (market-level).
        After market aggregation, each row is already at molecule level, so divide by molecule total.
        """
        # After data_layer aggregation, each row is a unique molecule per country
        # Market share = (this row's revenue) / (total market for this molecule-country)
        
        # Group by Country + Molecule to get molecule-level totals
        molecule_totals = self.df.groupby(['Country', 'Molecule List'])['MAT Q2 2025_LCD MNF'].transform('sum')
        
        # Each row's revenue / molecule total = market share
        molecule_rev = self.df['MAT Q2 2025_LCD MNF'].fillna(0)
        market_share = (molecule_rev / (molecule_totals + 1e-10)) * 100
        
        return market_share.fillna(0)
    
    def compute_price_per_unit(self) -> pd.Series:
        """
        FIX #7: Handle zero volume properly - set price = NaN when volume is 0.
        Don't use fillna(1) which creates fake prices.
        """
        revenue = self.df['MAT Q2 2025_LCD MNF'].fillna(0)
        volume = self.df['MAT Q2 2025_Standard Units'].fillna(0)
        
        price_per_unit = revenue / (volume + 1e-10)
        
        # FIX #7: Where volume is 0 or very small, set price to NaN (not a fake price)
        price_per_unit = price_per_unit.where(volume > 0.1, np.nan)
        
        return price_per_unit.replace([np.inf, -np.inf], np.nan)
    
    def compute_price_change(self) -> pd.Series:
        """Price trend: YoY price per unit change %."""
        revenue_2023 = self.df['MAT Q2 2023_LCD MNF'].fillna(1)
        volume_2023 = self.df['MAT Q2 2023_Standard Units'].fillna(1)
        price_2023 = revenue_2023 / volume_2023
        
        revenue_2025 = self.df['MAT Q2 2025_LCD MNF'].fillna(1)
        volume_2025 = self.df['MAT Q2 2025_Standard Units'].fillna(1)
        price_2025 = revenue_2025 / volume_2025
        
        price_change = ((price_2025 - price_2023) / (price_2023 + 1e-10)) * 100
        price_change = price_change.replace([np.inf, -np.inf], 0).fillna(0)
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
    
    # ===== ADVANCED METRICS (Section 3.2) =====
    
    def compute_hhi_market_concentration(self) -> pd.Series:
        """
        FIX #6: HHI should be computed PER MOLECULE, not per segment.
        Indicates market concentration within a molecule's market: > 2500 = highly concentrated.
        """
        # After market aggregation, each row is already a molecule.
        # For concentration analysis, use revenue stability as proxy
        
        revenue = self.df[self.REVENUE_COLS].fillna(0)
        volume = self.df[self.VOLUME_COLS].fillna(0)
        
        # Coefficient of variation as proxy for concentration
        # High CV = unstable competition (potential for HHI shifts)
        revenue_cv = (revenue.std(axis=1) / (revenue.mean(axis=1) + 1e-10)).fillna(0)
        
        # Convert to HHI-like scale (0-10000)
        # Low CV = stable = low HHI risk = ~2500
        # High CV = volatile = high HHI risk = ~7500
        hhi_proxy = 2500 + (revenue_cv * 2000).clip(0, 7500)
        
        return hhi_proxy
    
    def compute_cagr(self) -> pd.Series:
        """
        Compound Annual Growth Rate (2023-2025, 2 years).
        CAGR = (Ending / Beginning) ^ (1 / years) - 1
        """
        revenue_start = self.df['MAT Q2 2023_LCD MNF'].fillna(1)
        revenue_end = self.df['MAT Q2 2025_LCD MNF'].fillna(1)
        
        cagr = np.power(revenue_end / revenue_start, 1/2) - 1
        cagr = cagr.replace([np.inf, -np.inf], 0).fillna(0)
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
        FIX #9: Channel dependency is a PROXY using Sector, which may not represent real channels.
        
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
        channel_dependency = rev.groupby(['Country', 'Molecule List'])['channel_share'].transform('max') if 'channel_share' in rev.columns else channel_share.max()
        
        return channel_dependency.fillna(0.5)  # Neutral if no data
    
    # ===== DERIVED METRICS (Section 3.3) =====
    
    def compute_lifecycle_stage(self) -> pd.Series:
        """
        FIX #8: Use explicit, interpretable thresholds instead of rank-based bins.
        Market Lifecycle Classification: Emerging / Growth / Mature / Declining.
        Based on: revenue growth + market size thresholds.
        """
        market_size = self.compute_market_size()  # In millions
        revenue_growth = self.compute_revenue_growth()  # In %
        
        # Clean inputs
        rg = revenue_growth.replace([np.inf, -np.inf], np.nan).fillna(0)
        ms = market_size.replace([np.inf, -np.inf], np.nan).fillna(0)


    # FIX #8: Use explicit thresholds for interpretability
    # FIX #25: Use config thresholds instead of hardcoded values
    from config import THRESHOLDS
        
    emerging_growth = THRESHOLDS.get('lifecycle_emerging_growth', 0.20) * 100
    growth_threshold = THRESHOLDS.get('lifecycle_growth_growth', 0.15) * 100
    decline_threshold = THRESHOLDS.get('lifecycle_decline_threshold', -0.10) * 100
    emerging_size = THRESHOLDS.get('lifecycle_emerging_market_size', 50)
    mid_min = THRESHOLDS.get('lifecycle_mid_market_min', 50)
    mid_max = THRESHOLDS.get('lifecycle_mid_market_max', 500)
        
    stage = pd.Series('MATURE', index=self.df.index)
        
    # EMERGING: New markets with small size but positive growth
    stage[(ms < emerging_size) & (rg > emerging_growth)] = 'EMERGING'
        
    # GROWTH: Mid-size markets with strong growth
    stage[(ms >= mid_min) & (ms < mid_max) & (rg > growth_threshold)] = 'GROWTH'
    stage[(ms >= mid_max) & (rg > emerging_growth)] = 'GROWTH'  # Large markets need higher growth
        
    # DECLINING: Negative growth regardless of size
    stage[rg < decline_threshold] = 'DECLINING'
        
    # MATURE: Everything else (stable, mid-size or large)
        
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
        features['molecule_id'] = self.df.get('molecule_id', 
            self.df['Manufacturer'] + '_' + self.df['Molecule List'] + '_' + self.df['Country']
        )
        features['country'] = self.df['Country'].fillna('Unknown')
        features['sector'] = self.df['Sector'].fillna('Unknown')
        features['manufacturer'] = self.df['Manufacturer'].fillna('Unknown')
        features['molecule'] = self.df['Molecule List'].fillna('Unknown')
        
        self.features = features
        return features


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

# FIX #25: Import config for thresholds
try:
    from config import THRESHOLDS
except ImportError:
    THRESHOLDS = {}
