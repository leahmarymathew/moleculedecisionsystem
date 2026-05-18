"""
Train ML models on IQVIA data and serialize them to models/v1/.

Run this once (or after data updates) before running the inference pipeline.
Produces rf_classifier.joblib, rf_scaler.joblib, xgb_model.joblib,
kmeans.joblib, feature_names.json, and metadata.json.

Usage:
    python train_models.py "IQVIA - Leah .xlsx"
    python train_models.py "IQVIA - Leah .xlsx" --models-dir models/v2
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from logging_config import configure_logging, get_logger
from data_layer import IQVIADataLoader
from feature_engineering import FeatureEngineer
from ml_models import MLModels

_logger = get_logger(__name__)


def train_and_save(data_path: str, models_dir: str = 'models/v1') -> dict:
    """
    Load data, train all models, serialize to disk, and write metadata.json.

    Returns:
        metadata dict written to metadata.json.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    _logger.info("loading_data", data_path=str(data_path))
    loader = IQVIADataLoader(data_path)
    loader.load()
    loader.validate_structure()
    clean_df = loader.clean()

    _logger.info("engineering_features")
    engineer = FeatureEngineer(clean_df)
    features = engineer.build_features()

    _logger.info("training_models")
    ml = MLModels(features)
    results = ml.train_all_models()

    _logger.info("saving_models", models_dir=str(models_dir))
    ml.save_models(models_dir)

    xgb_info = results['xgboost']
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'data_path': str(Path(data_path).resolve()),
        'n_samples': len(features),
        'models_dir': str(models_dir.resolve()),
        'rf': {
            'holdout_accuracy': results['random_forest']['holdout_accuracy'],
            'n_estimators': results['random_forest']['model'].n_estimators,
            'train_size': results['random_forest']['train_size'],
            'test_size': results['random_forest']['test_size'],
        },
        'xgboost': {
            'available': 'error' not in xgb_info,
            'test_score': xgb_info.get('test_score'),
            'test_rmse': xgb_info.get('test_rmse'),
        },
        'kmeans': {
            'n_clusters': results['kmeans']['n_clusters_optimal'],
            'silhouette_score': results['kmeans']['silhouette_score'],
            'inertia': results['kmeans']['inertia'],
        },
    }

    metadata_path = models_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    _logger.info("metadata_written", path=str(metadata_path))
    return metadata


if __name__ == '__main__':
    configure_logging()
    parser = argparse.ArgumentParser(
        description='Train ML models for the pharmaceutical analytics pipeline'
    )
    parser.add_argument('data_path', help='Path to IQVIA Excel workbook')
    parser.add_argument(
        '--models-dir', default='models/v1',
        help='Output directory for serialized models (default: models/v1)'
    )
    args = parser.parse_args()

    metadata = train_and_save(args.data_path, args.models_dir)

    print(f"\n=== Training Complete ===")
    print(f"Samples:       {metadata['n_samples']}")
    print(f"RF accuracy:   {metadata['rf']['holdout_accuracy']:.3f}")
    if metadata['xgboost']['available']:
        print(f"XGBoost R²:    {metadata['xgboost']['test_score']:.3f}")
        print(f"XGBoost RMSE:  {metadata['xgboost']['test_rmse']:.4f}")
    else:
        print("XGBoost:       not available (install xgboost to enable)")
    print(f"K-Means k:     {metadata['kmeans']['n_clusters']}")
    if metadata['kmeans']['silhouette_score'] is not None:
        print(f"Silhouette:    {metadata['kmeans']['silhouette_score']:.3f}")
    print(f"Models saved:  {metadata['models_dir']}/")
