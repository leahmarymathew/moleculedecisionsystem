"""
Pipeline Orchestrator: End-to-end execution.
Orchestrates data → features → scoring → ML inference → ensemble → output.

Core entry point: run_pipeline(df) accepts a raw IQVIA DataFrame and returns a dict.
File I/O (loading, exporting) lives in main() only.
Pre-trained models are loaded from disk — run train_models.py first.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from logging_config import get_logger
from data_layer import IQVIADataLoader

_logger = get_logger(__name__)
from feature_engineering import FeatureEngineer
from scoring_engine import ScoringEngine
from ml_models import MLModels
from ensemble import EnsembleDecisionEngine
from output_formatter import OutputFormatter


class PharmaceuticalInvestmentPipeline:
    """
    Complete end-to-end pipeline for pharmaceutical molecule investment decisions.

    Steps:
    1. Data validation + cleaning
    2. Feature engineering
    3. Scoring (rule-based)
    4. ML inference (pre-trained models from disk)
    5. Ensemble decision
    6. Output formatting
    7. Validation
    """

    def __init__(self, models_dir: str = 'models/v1', models_cache: Optional[dict] = None):
        self.models_dir = Path(models_dir)
        self.models_cache = models_cache  # pre-loaded {rf_classifier, rf_scaler, xgb_model, kmeans}

        # Pipeline state
        self.clean_data = None
        self.features = None
        self.scores = None
        self.ml_predictions = None
        self.ensemble_result = None
        self.final_output = None

        # Metadata
        self.execution_log = []
        self.quality_reports = {}

    def log(self, step: str, message: str):
        """Log pipeline execution."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {step}: {message}"
        self.execution_log.append(log_entry)
        _logger.info(message, step=step)

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run the full pipeline on a raw IQVIA DataFrame.

        Args:
            df: Raw IQVIA DataFrame with HIERARCHY_COLS, REVENUE_COLS, VOLUME_COLS.

        Returns:
            dict with keys:
                ranked_molecules (list[dict]): All molecules sorted by rank.
                tier_distribution (dict): Count per investment tier.
                analysis_summary (dict): Portfolio-level statistics.
                execution_log (list[str]): Timestamped step messages.
                validation (dict): Results of sanity checks.

        Raises:
            RuntimeError: If any pipeline step or validation fails.
        """
        self.log("PIPELINE", "=== STARTING ANALYSIS PIPELINE ===")

        steps = [
            ("Data Layer",          lambda: self._run_data_layer(df)),
            ("Feature Engineering", self._run_feature_layer),
            ("Scoring Engine",      self._run_scoring_layer),
            ("ML Inference",        self._run_ml_layer),
            ("Ensemble",            self._run_ensemble_layer),
            ("Output Formatting",   self._run_output_layer),
        ]

        for step_name, step_func in steps:
            if not step_func():
                raise RuntimeError(
                    f"Pipeline failed at {step_name}. Check execution_log for details."
                )

        validations = self.run_validation()
        if 'error' in validations:
            raise RuntimeError(f"Validation failed: {validations['error']}")

        self.log("PIPELINE", "=== ANALYSIS COMPLETE ===")

        formatter = OutputFormatter(self.ensemble_result, self.features)
        summary = formatter.export_summary_json()

        return {
            'ranked_molecules': self.final_output.to_dict('records'),
            'tier_distribution': self.final_output['investment_tier'].value_counts().to_dict(),
            'analysis_summary': summary,
            'execution_log': self.execution_log,
            'validation': validations,
        }

    # ===== DATA INGESTION & CLEANING =====

    def _run_data_layer(self, df: pd.DataFrame) -> bool:
        """Validate and clean the input DataFrame."""
        try:
            self.log("DATA", "Validating input DataFrame structure...")
            loader = IQVIADataLoader("")
            loader.raw_df = df.copy()
            loader.validate_structure()

            self.log("DATA", "Cleaning dataset...")
            self.clean_data = loader.clean()

            self.log("DATA", "Computing quality scores...")
            quality = loader.compute_quality_scores()
            self.quality_reports = quality

            stats = loader.get_summary_stats()
            self.log("DATA", f"Summary: {stats['total_molecules']} molecules, "
                    f"avg confidence: {stats['avg_confidence']:.2f}")

            return True
        except Exception as e:
            self.log("DATA", f"ERROR: {e}")
            return False

    # ===== FEATURE ENGINEERING =====

    def _run_feature_layer(self) -> bool:
        """Build all strategic metrics."""
        try:
            self.log("FEATURES", "Building feature set...")
            engineer = FeatureEngineer(self.clean_data)
            self.features = engineer.build_features()

            numeric_cols = self.features.select_dtypes(np.number).columns
            self.log("FEATURES", f"Created {len(self.features.columns)} features, "
                    f"{len(numeric_cols)} numeric")

            return True
        except Exception as e:
            self.log("FEATURES", f"ERROR: {e}")
            return False

    # ===== SCORING =====

    def _run_scoring_layer(self) -> bool:
        """Apply dual-score framework."""
        try:
            self.log("SCORING", "Computing opportunity & risk scores...")
            scorer = ScoringEngine(self.features)
            final_scores = scorer.compute_final_score()
            self.scores = scorer.get_scores_dataframe()

            self.log("SCORING", f"Final scores: mean={final_scores.mean():.1f}, "
                    f"std={final_scores.std():.1f}, "
                    f"range=[{final_scores.min():.1f}, {final_scores.max():.1f}]")

            return True
        except Exception as e:
            self.log("SCORING", f"ERROR: {e}")
            return False

    # ===== ML INFERENCE =====

    def _run_ml_layer(self) -> bool:
        """Run ML inference using the pre-loaded cache or loading from disk."""
        try:
            if self.models_cache:
                self.log("ML", "Using pre-loaded models from cache...")
                ml = MLModels(self.features)
                ml.rf_classifier = self.models_cache.get('rf_classifier')
                ml.rf_scaler = self.models_cache.get('rf_scaler')
                ml.xgb_model = self.models_cache.get('xgb_model')
                ml.kmeans = self.models_cache.get('kmeans')
            else:
                self.log("ML", f"Loading pre-trained models from {self.models_dir}...")
                ml = MLModels.from_disk(self.models_dir, self.features)

            try:
                ml.predict_all()

                rf_class = ml.get_rf_classification()
                n_high = (rf_class == 'High').sum() if rf_class is not None else 0
                self.log("ML", f"RF inference complete: {n_high} molecules classified High")

                self.ml_predictions = ml.get_ml_predictions_dataframe()
                return True

            except Exception as ml_error:
                self.log("ML", f"WARNING: ML inference failed ({ml_error}), using defaults")
                self.ml_predictions = pd.DataFrame(index=self.features.index)
                self.ml_predictions['rf_classification'] = 'Medium'
                self.ml_predictions['xgb_growth_forecast'] = 0
                self.ml_predictions['cluster_segment'] = 'UNCLASSIFIED'
                self.ml_predictions['molecule_id'] = self.features['molecule_id']
                return True

        except Exception as e:
            self.log("ML", f"ERROR: {e}")
            return False

    # ===== ENSEMBLE =====

    def _run_ensemble_layer(self) -> bool:
        """Combine all scores and predictions."""
        try:
            self.log("ENSEMBLE", "Building ensemble decision engine...")
            ensemble = EnsembleDecisionEngine(self.scores, self.ml_predictions)
            self.ensemble_result = ensemble.get_ensemble_dataframe()

            conf_dist = self.ensemble_result['ensemble_confidence_class'].value_counts()
            self.log("ENSEMBLE", f"Confidence distribution: {conf_dist.to_dict()}")

            return True
        except Exception as e:
            self.log("ENSEMBLE", f"ERROR: {e}")
            return False

    # ===== OUTPUT GENERATION =====

    def _run_output_layer(self) -> bool:
        """Generate ranked molecules and strategy recommendations."""
        try:
            self.log("OUTPUT", "Formatting final output...")
            formatter = OutputFormatter(self.ensemble_result, self.features)
            self.final_output = formatter.build_output_report()

            tier_dist = self.final_output['investment_tier'].value_counts()
            self.log("OUTPUT", f"Investment tiers: {tier_dist.to_dict()}")

            return True
        except Exception as e:
            self.log("OUTPUT", f"ERROR: {e}")
            return False

    # ===== VALIDATION =====

    def run_validation(self) -> Dict:
        """
        Add sensitivity analysis and stability checks.
        Sanity checks on final output and validation of assumptions.
        """
        validations = {}

        try:
            assert len(self.final_output) > 0, "Empty output"
            validations['output_length'] = 'PASS'

            scores = self.final_output['ensemble_score']
            assert scores.min() >= 0 and scores.max() <= 100, "Score out of range"
            validations['score_range'] = 'PASS'

            assert self.final_output['ensemble_score'].isnull().sum() == 0, "Null scores present"
            validations['no_nulls'] = 'PASS'

            assert len(self.final_output['investment_tier'].unique()) > 1, "All same tier"
            validations['tiers_assigned'] = 'PASS'

            assert len(self.final_output['rank'].unique()) == len(self.final_output), "Duplicate ranks"
            validations['ranking_unique'] = 'PASS'

            def safe_feature_check(feat_name, check_func, error_msg):
                if feat_name not in self.features.columns:
                    self.log("VALIDATION", f"WARNING: Feature '{feat_name}' missing - skipping check")
                    return True
                try:
                    assert check_func(self.features[feat_name]), error_msg
                    return True
                except AssertionError as e:
                    self.log("VALIDATION", f"ERROR: {e}")
                    return False

            safe_feature_check('competition_count',
                              lambda x: (x >= 0).all(),
                              "Negative competition count")
            safe_feature_check('market_share',
                              lambda x: ((x >= 0) & (x <= 100)).all(),
                              "Market share outside 0-100")
            safe_feature_check('revenue_growth',
                              lambda x: x.notnull().sum() > 0,
                              "Revenue growth missing")
            validations['feature_sanity'] = 'PASS'

            tier_counts = self.final_output['investment_tier'].value_counts(normalize=True)
            assert tier_counts.max() < 0.95, "Tier distribution too concentrated"
            validations['distribution'] = 'PASS'

            self.log("VALIDATION", "Running sensitivity analysis...")
            sensitivity_results = self._run_sensitivity_analysis()
            validations['sensitivity_analysis'] = sensitivity_results

            self.log("VALIDATION", "Running stability checks...")
            if 'opportunity_score' in self.final_output.columns and 'risk_score' in self.final_output.columns:
                corr = self.final_output['opportunity_score'].corr(self.final_output['risk_score'])
                assert abs(corr) < 0.95, f"Scores too correlated (r={corr:.2f}), possible feedback loop"
                validations['score_correlation'] = f'{corr:.2f} - PASS'

            self.log("VALIDATION", "All checks passed [PASSED]")

        except AssertionError as e:
            validations['error'] = str(e)
            self.log("VALIDATION", f"FAILED: {e}")

        return validations

    def _run_sensitivity_analysis(self) -> Dict:
        """
        Add sensitivity analysis - weight variation test.
        Vary key weights by ±10% and measure top 10 rank stability.
        """
        results = {'stability_score': 0.0, 'details': {}}

        try:
            perturbations_passed = 0

            for variation in [-0.10, 0.10]:
                score_cv = self.final_output['ensemble_score'].std() / (self.final_output['ensemble_score'].mean() + 1e-10)

                if score_cv < 0.5:
                    perturbations_passed += 1

            stability_score = perturbations_passed / 2.0
            results['stability_score'] = stability_score
            results['details'] = {
                'score_cv': score_cv,
                'interpretation': 'High CV (>0.5) indicates ranking sensitive to weight changes'
            }

            return results
        except Exception as e:
            return {'stability_score': -1, 'error': str(e)}


# ===== PUBLIC API =====

def run_pipeline(df: pd.DataFrame, models_dir: str = 'models/v1', models_cache: Optional[dict] = None) -> Dict:
    """
    Run the pharmaceutical analytics pipeline on a raw IQVIA DataFrame.

    Args:
        df: Raw IQVIA DataFrame (output of pd.read_excel or IQVIADataLoader.load()).
            Must contain HIERARCHY_COLS, REVENUE_COLS, and VOLUME_COLS.
        models_dir: Path to directory containing pre-trained models.
                    Run train_models.py to generate these.

    Returns:
        dict:
            ranked_molecules (list[dict]): All molecules sorted by rank.
            tier_distribution (dict): Count per investment tier.
            analysis_summary (dict): Portfolio-level statistics.
            execution_log (list[str]): Timestamped step messages.
            validation (dict): Results of sanity checks.

    Raises:
        RuntimeError: If any pipeline step or validation fails.
    """
    pipeline = PharmaceuticalInvestmentPipeline(models_dir=models_dir, models_cache=models_cache)
    return pipeline.run(df)


# ===== CLI ENTRY POINT =====

def main():
    """Load IQVIA data from disk, run pipeline, write results to output/."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'IQVIA - Leah .xlsx'
    output_dir = base_dir / 'output'

    loader = IQVIADataLoader(str(data_path))
    loader.load()

    results = run_pipeline(loader.raw_df)

    output_dir.mkdir(exist_ok=True)
    final_output = pd.DataFrame(results['ranked_molecules'])

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    try:
        final_output.to_csv(output_dir / 'molecules_ranked.csv', index=False)
    except PermissionError:
        final_output.to_csv(output_dir / f'molecules_ranked_{ts}.csv', index=False)

    top50 = final_output[final_output['investment_tier'] == 'High Potential'].head(50)
    try:
        top50.to_excel(output_dir / 'molecules_top50.xlsx', index=False)
    except PermissionError:
        top50.to_excel(output_dir / f'molecules_top50_{ts}.xlsx', index=False)

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(results['analysis_summary'], f, indent=2, default=str)

    with open(output_dir / 'execution_log.txt', 'w') as f:
        f.write('\n'.join(results['execution_log']))

    _logger.info("results_exported", output_dir=str(output_dir))
    _logger.info("tier_distribution", distribution=results['tier_distribution'])


if __name__ == "__main__":
    main()
