"""
Data Layer: Load, validate, clean IQVIA pharmaceutical molecule data.
Handles MAT time series, anomalies, and data quality scoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, List

from logging_config import get_logger

_logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """Data quality assessment for molecules."""
    molecule_id: str
    is_valid: bool
    anomalies: List[str] = field(default_factory=list)
    confidence_score: float = 1.0  # 0.0 to 1.0
    null_count: int = 0
    notes: str = ""


class IQVIADataLoader:
    """Load and preprocess IQVIA pharmaceutical dataset."""
    
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
    
    # Hierarchy columns
    HIERARCHY_COLS = ['Country', 'Sector', 'Manufacturer', 'Molecule List']
    
    # Classification columns
    CLASS_COLS = ['ATC1', 'ATC2', 'ATC3', 'ATC4', 'Innovation Insights']
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.raw_df = None
        self.clean_df = None
        self.quality_report = {}
    
    def load(self) -> pd.DataFrame:
        """Load IQVIA workbook."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        try:
            self.raw_df = pd.read_excel(self.file_path, sheet_name='Sheet1')
            _logger.info("data_loaded", n_rows=len(self.raw_df))
            return self.raw_df
        except Exception as e:
            raise RuntimeError(f"Failed to load workbook: {e}")
    
    def validate_structure(self) -> bool:
        """Validate required columns exist."""
        required = self.HIERARCHY_COLS + self.REVENUE_COLS + self.VOLUME_COLS
        missing = [c for c in required if c not in self.raw_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
    
    def clean(self) -> pd.DataFrame:
        """
        Clean data according to roadmap rules:
        - Preserve original negative adjustments
        - Treat NULL as structural zero (no imputation for revenue/volume)
        - Add structural presence flags
        - Compute growth/anomaly indicators
        - Aggregate manufacturer rows to molecule-market level
        """
        if self.raw_df is None:
            raise RuntimeError("Call load() before clean()")

        df = self.raw_df.copy()

        # Create molecule identifier at manufacturer-row level (traceability)
        df['molecule_id'] = (
            df['Manufacturer'].astype(str) + '_' + 
            df['Molecule List'].astype(str) + '_' + 
            df['Country'].astype(str)
        )

        revenue_cols = self.REVENUE_COLS
        volume_cols = self.VOLUME_COLS

        # ===== RULE 1: Negative revenue/units =====
        has_negative = (df[revenue_cols] < 0).any(axis=1) | (df[volume_cols] < 0).any(axis=1)
        _logger.info("negative_revenue_units", count=int(has_negative.sum()))
        for col in revenue_cols + volume_cols:
            raw_col = f"{col}_raw"
            if raw_col not in df.columns:
                df[raw_col] = df[col]
        df['negative_revenue_flag'] = has_negative

        # ===== RULE 2: Fully exited molecules (manufacturer rows) =====
        revenue_filled = df[revenue_cols].fillna(0)
        volume_filled = df[volume_cols].fillna(0)
        is_exited = (revenue_filled.sum(axis=1) == 0) & (volume_filled.sum(axis=1) == 0)
        _logger.info("exited_molecules", count=int(is_exited.sum()))
        df['exited_flag'] = is_exited

        # ===== RULE 3: NULLs are structural -> convert to 0 (no imputation) =====
        # FIX #8: Track which values were originally null before conversion
        df['missing_flag'] = False
        for col in revenue_cols + volume_cols:
            df['missing_flag'] |= df[col].isna()
        
        # Preserve raw columns above, then convert
        df[revenue_cols] = df[revenue_cols].fillna(0)
        df[volume_cols] = df[volume_cols].fillna(0)
        
        # Mark structural zeros (converted from NULL)
        df['structural_zero_flag'] = df['missing_flag']

        # Maintain an imputed_flag for compatibility (no structural imputation applied)
        df['imputed_flag'] = False

        # Data quality flag base
        if 'data_quality_flag' in df.columns:
            df['data_quality_flag'] = df['data_quality_flag'].fillna('').astype(str)
        else:
            df['data_quality_flag'] = ''

        neg_idx = df['negative_revenue_flag'] == True
        if neg_idx.any():
            df.loc[neg_idx, 'data_quality_flag'] = df.loc[neg_idx, 'data_quality_flag'].apply(lambda x: (';'.join([s for s in [x, 'NEGATIVE'] if s])).strip(';'))

        # ===== Growth anomaly detection & zero-base handling =====
        start_col = revenue_cols[0]
        end_col = revenue_cols[-1]
        df['rev_start'] = df[start_col]
        df['rev_end'] = df[end_col]

        # Zero-base detection: emerging when start == 0
        df['emerging_flag'] = df['rev_start'].fillna(0) == 0

        def compute_cagr_row(a, b):
            try:
                if pd.isna(a) or pd.isna(b):
                    return np.nan
                if a == 0:
                    return np.nan
                years = 2.0
                return (b / a) ** (1.0 / years) - 1.0
            except Exception:
                return np.nan

        df['cagr_2y'] = df.apply(lambda r: compute_cagr_row(r['rev_start'], r['rev_end']), axis=1)

        # Detect anomalies using sector-level median and MAD on cagr
        df['cagr_median_sector'] = df.groupby('Sector')['cagr_2y'].transform('median')
        df['cagr_mad'] = df.groupby('Sector')['cagr_2y'].transform(lambda x: np.median(np.abs(x - np.nanmedian(x))) if x.notna().any() else np.nan)
        df['cagr_std'] = df.groupby('Sector')['cagr_2y'].transform('std')

        def compute_z_score(row):
            mad = row['cagr_mad']
            std = row['cagr_std']
            cagr = row['cagr_2y']
            median = row['cagr_median_sector']
            if pd.isna(cagr) or pd.isna(median):
                return 0
            if not pd.isna(mad) and mad > 1e-12:
                return (cagr - median) / (1.4826 * mad)
            elif not pd.isna(std) and std > 1e-12:
                return (cagr - median) / std
            else:
                return 0

        df['cagr_mad_z'] = df.apply(compute_z_score, axis=1)
        df['growth_anomaly_flag'] = df['cagr_mad_z'].abs() > 3

        _logger.info("cleaning_complete", n_rows=len(df))

        # ===== STRUCTURAL PRESENCE FLAGS (manufacturer rows) =====
        rev_later = df[revenue_cols[1:]]
        df['is_new_entry'] = (df[revenue_cols[0]] == 0) & (rev_later.sum(axis=1) > 0)
        df['is_exit'] = (df[revenue_cols[:-1]].sum(axis=1) > 0) & (df[revenue_cols[-1]] == 0)
        df['is_inactive'] = (df[revenue_cols].sum(axis=1) == 0) & (df[volume_cols].sum(axis=1) == 0)
        has_zero = (df[revenue_cols] == 0).any(axis=1)
        has_pos = (df[revenue_cols] > 0).any(axis=1)
        df['has_partial_presence'] = has_zero & has_pos & ~(df['is_new_entry'] | df['is_exit'])

        # ===== PRE-AGGREGATION MARKET METRICS =====
        # Compute manufacturer-level market share before aggregation.
        # This uses molecule-level totals, not sector totals.
        molecule_revenue = df.groupby(['Country', 'Molecule List'])['MAT Q2 2025_LCD MNF'].transform('sum')
        df['market_share'] = (df['MAT Q2 2025_LCD MNF'] / (molecule_revenue + 1e-10)) * 100
        df['market_share'] = df['market_share'].fillna(0).clip(0, 100)

        # Competition count must also be computed before aggregation.
        # After aggregation, Manufacturer becomes "Market" and would otherwise collapse to 1.
        df['competition_count'] = df.groupby(['Country', 'Molecule List'])['Manufacturer'].transform('nunique').fillna(0)

        # Pre-compute HHI at the molecule-country level from manufacturer shares.
        # This is stored on every raw row and aggregated later with first().
        share_frac = df['market_share'] / 100
        df['hhi'] = share_frac.groupby([df['Country'], df['Molecule List']]).transform(lambda g: (g ** 2).sum() * 10000)
        df['hhi'] = df['hhi'].fillna(2500)

        # Group by Country + Molecule List (aggregate manufacturers within each market)
        df = self._aggregate_to_molecule_market(df, revenue_cols, volume_cols)

        _logger.info("aggregation_complete", n_rows=len(df))

        self.clean_df = df
        return df
    
    def _aggregate_to_molecule_market(self, df: pd.DataFrame, revenue_cols: list, volume_cols: list) -> pd.DataFrame:
        """
         Aggregate manufacturer rows to molecule-market level (Country + Molecule List).
         Compute market_share BEFORE aggregation, then take dominant player's share.
        
        Aggregation rules:
        - Revenue/Volume: SUM across manufacturers
        - Flags: OR (logical max)
        - Market share: MAX across manufacturers (dominant player's share of sector)
        - molecule_id: Create new ID = Molecule List_Country
        """
        
        # Now aggregate by Country + Molecule List
        groupby_cols = ['Country', 'Molecule List']
        
        # Define aggregation functions
        agg_dict = {}
        
        # Sum numeric columns (revenue, volume)
        for col in revenue_cols + volume_cols:
            agg_dict[col] = 'sum'
        
        # Boolean flags: take OR (max)
        for flag_col in ['exited_flag', 'emerging_flag', 'growth_anomaly_flag', 
                         'is_new_entry', 'is_exit', 'is_inactive', 'has_partial_presence',
                         'negative_revenue_flag', 'missing_flag', 'structural_zero_flag', 'imputed_flag']:
            if flag_col in df.columns:
                agg_dict[flag_col] = 'max'  # OR logic: True if any manufacturer has it
        
        # Precomputed market metrics
        if 'market_share' in df.columns:
            agg_dict['market_share'] = 'max'
        if 'competition_count' in df.columns:
            agg_dict['competition_count'] = 'first'
        if 'hhi' in df.columns:
            agg_dict['hhi'] = 'first'
        
        # Sector: take first (should be same for molecule)
        if 'Sector' in df.columns:
            agg_dict['Sector'] = 'first'
        
        # ATC columns: take first
        for col in ['ATC1', 'ATC2', 'ATC3', 'ATC4', 'Innovation Insights']:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # Confidence/quality: take max
        for col in ['data_quality_flag', 'cagr_2y']:
            if col in df.columns:
                agg_dict[col] = 'first' if col == 'data_quality_flag' else 'mean'
        
        # Raw columns: preserve first (for traceability)
        for col in df.columns:
            if col.endswith('_raw') and col not in agg_dict:
                agg_dict[col] = 'first'
        
        # Perform aggregation
        agg_df = df.groupby(groupby_cols, as_index=False).agg(agg_dict)
        
        # Create new molecule_id at aggregated level
        agg_df['molecule_id'] = agg_df['Molecule List'].astype(str) + '_' + agg_df['Country'].astype(str)
        
        # Keep Manufacturer column as 'Market' to indicate aggregation
        agg_df['Manufacturer'] = 'Market'
        
        return agg_df

    def compute_quality_scores(self) -> Dict[str, DataQualityReport]:
        """
        Assign data confidence score per molecule (0.0 to 1.0).
        Penalize for:
        - Null values
        - Anomalies (extreme growth spikes)
        - Structural inconsistencies
        """
        if self.clean_df is None:
            raise RuntimeError("Call clean() before computing quality scores")
        
        df = self.clean_df
        
        for mol_id in df['molecule_id'].unique():
            mol_data = df[df['molecule_id'] == mol_id]
            
            report = DataQualityReport(molecule_id=mol_id, is_valid=True)
            
            # ===== Count nulls =====
            null_count = mol_data[self.REVENUE_COLS + self.VOLUME_COLS].isnull().sum().sum()
            report.null_count = int(null_count)
            
            # ===== Detect anomalies: extreme growth spikes =====
            revenue_vals = mol_data[self.REVENUE_COLS].values.flatten()
            revenue_vals = revenue_vals[~np.isnan(revenue_vals)]
            
            if len(revenue_vals) >= 2:
                growth_rates = np.diff(revenue_vals) / (revenue_vals[:-1] + 1e-10)
                extreme_growth = np.abs(growth_rates) > 2.0  # > 200% change
                
                if extreme_growth.any():
                    report.anomalies.append(f"Extreme growth spike detected")
            
            # ===== Confidence score =====
            confidence = 1.0
            confidence -= null_count * 0.05  # Each null = -5%
            confidence -= len(report.anomalies) * 0.10  # Each anomaly = -10%
            confidence = max(0.0, min(1.0, confidence))
            
            report.confidence_score = confidence
            report.is_valid = confidence >= 0.3  # Threshold: 30% confidence minimum
            
            if report.confidence_score < 0.5:
                report.notes = "Low confidence data"
            
            self.quality_report[mol_id] = report
        
        _logger.info("quality_scores_computed", n_molecules=len(self.quality_report))
        return self.quality_report
    
    def get_summary_stats(self) -> Dict:
        """Return summary of data cleaning and quality."""
        if self.clean_df is None:
            return {}
        
        revenue_cols = self.REVENUE_COLS
        volume_cols = self.VOLUME_COLS
        
        return {
            'total_molecules': len(self.clean_df),
            'countries': self.clean_df['Country'].unique().tolist(),
            'sectors': self.clean_df['Sector'].unique().tolist(),
            'revenue_range': {
                'min': self.clean_df[revenue_cols].min().min(),
                'max': self.clean_df[revenue_cols].max().max(),
                'mean': self.clean_df[revenue_cols].mean().mean()
            },
            'volume_range': {
                'min': self.clean_df[volume_cols].min().min(),
                'max': self.clean_df[volume_cols].max().max(),
                'mean': self.clean_df[volume_cols].mean().mean()
            },
            'avg_confidence': np.mean([r.confidence_score for r in self.quality_report.values()]) if self.quality_report else None
        }


# Example usage
if __name__ == "__main__":
    loader = IQVIADataLoader(r'C:\Users\Leah\Documents\git\analytics\IQVIA - Leah .xlsx')
    loader.load()
    loader.validate_structure()
    clean = loader.clean()
    quality = loader.compute_quality_scores()
    
    print("\n=== Summary Stats ===")
    stats = loader.get_summary_stats()
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    print("\n=== Quality Report Sample ===")
    for mol_id, report in list(quality.items())[:5]:
        print(f"{mol_id}: confidence={report.confidence_score:.2f}, valid={report.is_valid}")
