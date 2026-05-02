"""
Data Layer: Load, validate, clean IQVIA pharmaceutical molecule data.
Handles MAT time series, anomalies, and data quality scoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, List


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
            print(f"Loaded {len(self.raw_df)} molecules from IQVIA workbook")
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
        - Remove negative revenue/units (adjustment entries)
        - Remove fully exited molecules
        - Handle zero-base entries
        - Impute missing values + flag
        """
        df = self.raw_df.copy()
        
        # Create molecule identifier
        df['molecule_id'] = (
            df['Manufacturer'] + '_' + 
            df['Molecule List'] + '_' + 
            df['Country']
        )
        
        # ===== RULE 1: Negative revenue/units =====
        # FIX #2: KEEP original negative values, DO NOT convert to NaN
        # Create negative flag for later handling in scoring, not in cleaning
        revenue_cols = self.REVENUE_COLS
        volume_cols = self.VOLUME_COLS

        has_negative = (df[revenue_cols] < 0).any(axis=1) | (df[volume_cols] < 0).any(axis=1)

        print(f"Rows with negative revenue/units: {has_negative.sum()}")
        # preserve original values - DO NOT CONVERT NEGATIVES TO NAN
        for col in revenue_cols + volume_cols:
            raw_col = f"{col}_raw"
            if raw_col not in df.columns:
                df[raw_col] = df[col]  # Keep originals intact

        df['negative_revenue_flag'] = has_negative
        # Original negative values are preserved; downstream scoring will handle with flag
        
        # ===== RULE 2: Fully exited molecules =====
        # Molecule is fully exited if all revenue AND volume are null/zero across all periods
        revenue_filled = df[revenue_cols].fillna(0)
        volume_filled = df[volume_cols].fillna(0)

        is_exited = (revenue_filled.sum(axis=1) == 0) & (volume_filled.sum(axis=1) == 0)

        print(f"Fully exited molecules: {is_exited.sum()}")
        df['exited_flag'] = is_exited
        # Keep exited rows but mark for downstream handling
        
        # ===== RULE 3: Handle missing values =====
        # Impute missing revenue/volume robustly across time (row-wise interpolation across MAT columns)
        # Preserve raw columns created above; create imputed flags for each cell
        for col in revenue_cols + volume_cols:
            imputed_col = f"{col}_imputed"
            df[imputed_col] = False

        # FIX #3: Use median imputation per segment INSTEAD of interpolation
        # (Pharma data is NOT linear, interpolation distorts trends)
        # For remaining missing values after row-wise check, use segment median
        
        # For any fully-null rows (row still all NaN), fallback to segment median
        segment_cols = ['Country', 'Sector', 'ATC1']
        for col in revenue_cols + volume_cols:
            missing_mask = df[col].isna()
            if missing_mask.any():
                seg_median = df.groupby(segment_cols)[col].transform('median')
                df.loc[missing_mask, col] = seg_median[missing_mask]
                df.loc[missing_mask, f"{col}_imputed"] = True

        # Mark imputed at row-level if any MAT cell was imputed
        df['imputed_flag'] = df[[f for f in df.columns if f.endswith('_imputed')]].any(axis=1)

        # Clean combined data_quality_flag to preserve previous flags
        if 'data_quality_flag' in df.columns:
            df['data_quality_flag'] = df['data_quality_flag'].fillna('').astype(str)
        else:
            df['data_quality_flag'] = ''

        # Append flags in a robust way
        neg_idx = df['negative_revenue_flag'] == True
        if neg_idx.any():
            df.loc[neg_idx, 'data_quality_flag'] = df.loc[neg_idx, 'data_quality_flag'].apply(lambda x: (';'.join([s for s in [x, 'NEGATIVE'] if s])).strip(';'))

        imp_idx = df['imputed_flag'] == True
        if imp_idx.any():
            df.loc[imp_idx, 'data_quality_flag'] = df.loc[imp_idx, 'data_quality_flag'].apply(lambda x: (';'.join([s for s in [x, 'IMPUTED'] if s])).strip(';'))

        # ===== Growth anomaly detection & zero-base handling =====
        # Compute simple CAGR between MAT Q2 2023 and MAT Q2 2025 (2-year interval)
        start_col = self.REVENUE_COLS[0]
        end_col = self.REVENUE_COLS[-1]
        df['rev_start'] = df[start_col]
        df['rev_end'] = df[end_col]
        # Zero-base detection: mark emerging if start <= 1
        df['emerging_flag'] = df['rev_start'].fillna(0) <= 1

        # Compute CAGR where possible; if base small, set NaN and mark emerging
        def compute_cagr(row):
            a = row['rev_start']
            b = row['rev_end']
            if pd.isna(a) or pd.isna(b):
                return np.nan
            if a <= 1:
                return np.nan
            years = 2.0
            try:
                return (b / a) ** (1.0 / years) - 1.0
            except Exception:
                return np.nan

        df['cagr_2y'] = df.apply(compute_cagr, axis=1)

        # FIX #4: Add fallback when MAD is 0 (unstable anomaly detection)
        # Detect anomalies using sector-level median and MAD on cagr
        df['cagr_median_sector'] = df.groupby('Sector')['cagr_2y'].transform('median')
        df['cagr_mad'] = df.groupby('Sector')['cagr_2y'].transform(lambda x: np.median(np.abs(x - np.nanmedian(x))) if x.notna().any() else np.nan)
        
        # FIX #4: Use standard deviation as fallback when MAD == 0
        df['cagr_std'] = df.groupby('Sector')['cagr_2y'].transform('std')
        
        def compute_z_score(row):
            mad = row['cagr_mad']
            std = row['cagr_std']
            cagr = row['cagr_2y']
            median = row['cagr_median_sector']
            
            if pd.isna(cagr) or pd.isna(median):
                return 0
            
            # Use MAD if available and non-zero, else use std
            if not pd.isna(mad) and mad > 1e-12:
                return (cagr - median) / (1.4826 * mad)
            elif not pd.isna(std) and std > 1e-12:
                return (cagr - median) / std
            else:
                return 0
        
        df['cagr_mad_z'] = df.apply(compute_z_score, axis=1)
        df['growth_anomaly_flag'] = df['cagr_mad_z'].abs() > 3
        
        print(f"After cleaning: {len(df)} molecules remain")
        
        # FIX #1: ADD MARKET-LEVEL AGGREGATION BEFORE RETURNING
        # Aggregate by Country + Molecule (not manufacturer level)
        # This ensures features are computed at molecule market level
        revenue_cols_to_agg = self.REVENUE_COLS
        volume_cols_to_agg = self.VOLUME_COLS
        
        agg_dict = {}
        for col in revenue_cols_to_agg + volume_cols_to_agg:
            agg_dict[col] = 'sum'  # Sum revenue/volume across manufacturers
            if f"{col}_raw" in df.columns:
                agg_dict[f"{col}_raw"] = 'first'
        
        # Keep other important fields
        for col in self.CLASS_COLS:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        agg_dict['Sector'] = 'first'
        agg_dict['Country'] = 'first'
        agg_dict['data_quality_flag'] = lambda x: ';'.join([s for s in x if s])
        agg_dict['negative_revenue_flag'] = 'any'
        agg_dict['imputed_flag'] = 'any'
        agg_dict['exited_flag'] = 'all'
        agg_dict['emerging_flag'] = 'any'
        agg_dict['cagr_2y'] = 'mean'  # Average CAGR across manufacturers
        agg_dict['growth_anomaly_flag'] = 'any'
        agg_dict['cagr_median_sector'] = 'first'
        agg_dict['cagr_std'] = 'first'
        agg_dict['cagr_mad_z'] = 'mean'
        
        # Aggregate at molecule-country level (market level)
        df['molecule_id'] = df['Molecule List'] + '_' + df['Country']
        df_agg = df.groupby(['Country', 'Molecule List'], as_index=False).agg(agg_dict)
        df_agg['molecule_id'] = df_agg['Molecule List'] + '_' + df_agg['Country']
        
        print(f"After market-level aggregation: {len(df_agg)} molecules (from {len(df)} manufacturer rows)")
        
        self.clean_df = df_agg
        return df_agg
    
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
            report.null_count = null_count
            
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
        
        print(f"Quality scores computed for {len(self.quality_report)} molecules")
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
            'avg_confidence': np.mean([r.confidence_score for r in self.quality_report.values()])
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
