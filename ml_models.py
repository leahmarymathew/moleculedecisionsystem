"""
ML Models Layer: RandomForest, XGBoost, and clustering.
Supporting layer (40% weight in ensemble, not primary).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from typing import Tuple, Dict, List

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class MLModels:
    """
    Machine learning models for:
    1. Classification: High / Medium / Low potential
    2. Forecasting: Future growth prediction
    3. Segmentation: Market clustering
    """
    
    def __init__(self, features_df: pd.DataFrame):
        """
        Args:
            features_df: DataFrame from FeatureEngineer with all metrics
        """
        self.features = features_df.copy()
        self.scaler = StandardScaler()
        
        # ML model storage
        self.rf_classifier = None
        self.xgb_model = None
        self.kmeans = None
        
        # Predictions
        self.rf_predictions = None
        self.xgb_predictions = None
        self.cluster_labels = None
    
    def prepare_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Remove revenue_growth from features (DATA LEAKAGE).
        revenue_growth is used as target, should NOT be in feature matrix.
        
        Extract numeric features for ML modeling.
        Exclude ID columns and non-numeric data.
        """
        numeric_cols = [
            # Remove 'revenue_growth' - it's the target!
            'volume_growth', 'market_size', 'competition_count',
            'market_share', 'price_per_unit', 'price_change', 'volatility',
            'hhi', 'cagr', 'revenue_sustainability', 'generic_erosion_index',
            'market_penetration', 'entry_barrier_score'
        ]
        
        # Include structural flags as numeric features if present
        flag_cols = ['is_new_entry', 'is_exit', 'is_inactive', 'has_partial_presence']
        for f in flag_cols:
            if f in self.features.columns and f not in numeric_cols:
                numeric_cols.append(f)

        # Filter to available columns
        available_cols = [c for c in numeric_cols if c in self.features.columns]

        # Fill NaNs with 0 for ML (models must handle NaN safely)
        X = self.features[available_cols].fillna(0).astype(float).values
        
        # Fit scaler on train only - do this AFTER train/test split
        # For now, just return raw data; scaling will happen in train methods
        
        return X, available_cols
    
    # ===== CLASSIFICATION: RANDOM FOREST =====
    
    def create_classification_labels(self) -> np.ndarray:
        """
        Use a smoothed business proxy instead of a hard rule copy.
        Create target labels for classification (High / Medium / Low potential).
        The score blends growth, market size, and competition, then adds a tiny
        deterministic jitter so the target is not an exact threshold rule.
        """
        volume_growth = self.features['volume_growth'].fillna(0)
        market_size = self.features['market_size'].fillna(0)
        competition = self.features['competition_count'].fillna(5)

        # Rank-based smoothing reduces sensitivity to raw scale differences.
        growth_rank = volume_growth.rank(pct=True)
        size_rank = market_size.rank(pct=True)
        comp_rank = competition.rank(pct=True)
        
        labels = []
        for i in range(len(self.features)):
            # Higher growth and market size should help; higher competition should hurt.
            score = (
                0.50 * growth_rank.iloc[i] +
                0.30 * size_rank.iloc[i] +
                0.20 * (1.0 - comp_rank.iloc[i])
            )

            # Tiny deterministic jitter avoids making labels a hard copy of the rule engine.
            stable_key = str(self.features.index[i])
            jitter = ((sum(ord(ch) for ch in stable_key) % 100) - 50) / 1000.0
            score = score + jitter

            if score >= 0.67:
                label = 2
            elif score <= 0.40:
                label = 0
            else:
                label = 1
            
            labels.append(label)
        
        return np.array(labels)
    
    def train_random_forest(self) -> Dict:
        """
        Scale data AFTER train/test split, not before.
        Train Random Forest classifier.
        Role: Classify molecules into High / Medium / Low potential.
        """
        X, feature_names = self.prepare_feature_matrix()
        y = self.create_classification_labels()

        class_counts = pd.Series(y).value_counts()
        stratify_y = y if class_counts.min() >= 2 and len(class_counts) > 1 else None

        # Scale data AFTER train/test split to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_y
        )
        
        # Scale data AFTER train/test split to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Also apply to full dataset for predictions (use train scaler)
        X_scaled = scaler.transform(X)
        
        # Handle class imbalance
        class_weights = {
            0: 1.0,  # Low
            1: 1.0,  # Medium
            2: 1.5   # High (boost)
        }
        
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_classifier.fit(X_train_scaled, y_train)
        
        # Get predictions and probabilities on the full matrix for downstream use
        predictions = self.rf_classifier.predict(X_scaled)
        probabilities = self.rf_classifier.predict_proba(X_scaled)
        test_predictions = self.rf_classifier.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        self.rf_predictions = predictions
        
        # Feature importance
        importances = self.rf_classifier.feature_importances_
        feature_importance_dict = dict(zip(feature_names, importances))
        
        return {
            'model': self.rf_classifier,
            'predictions': predictions,
            'probabilities': probabilities,
            'feature_importance': feature_importance_dict,
            'score': test_accuracy,
            'holdout_accuracy': test_accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def get_rf_classification(self) -> pd.Series:
        """Return RF predicted class as strings (High / Medium / Low)."""
        if self.rf_predictions is None:
            return None
        
        class_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        return pd.Series([class_map[p] for p in self.rf_predictions], index=self.features.index)
    
    # ===== FORECASTING: XGBOOST =====
    
    def train_xgboost_growth_model(self) -> Dict:
        """
        Train XGBoost to predict future growth.
        Target: revenue_growth (continuous).
        """
        if not HAS_XGBOOST:
            return {'error': 'XGBoost not installed', 'model': None}
        
        X, feature_names = self.prepare_feature_matrix()
        y = self.features['revenue_growth'].fillna(0).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.xgb_model.fit(X_train, y_train, verbose=False)
        
        # Predictions on full dataset
        predictions = self.xgb_model.predict(X)
        
        self.xgb_predictions = predictions
        
        # Test score
        test_predictions = self.xgb_model.predict(X_test)
        test_score = self.xgb_model.score(X_test, y_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
        
        return {
            'model': self.xgb_model,
            'predictions': predictions,
            'test_score': test_score,
            'test_rmse': rmse,
            'feature_names': feature_names
        }
    
    def get_xgb_growth_forecast(self) -> pd.Series:
        """Return XGBoost predicted growth."""
        if self.xgb_predictions is None:
            return None
        return pd.Series(self.xgb_predictions, index=self.features.index)
    
    # ===== CLUSTERING: K-MEANS =====
    
    def train_kmeans_clustering(self, n_clusters: int = None) -> Dict:
        """
        Use silhouette score to determine optimal clusters, not fixed n_clusters=3.
        Train K-Means for market segmentation.
        Clusters: Growth / Mature / Decline.
        """
        X, feature_names = self.prepare_feature_matrix()
        
        # Use silhouette score to determine optimal clusters
        if n_clusters is None:
            best_score = -1
            best_k = 3
            
            # Test k from 2 to min(10, len(X)//2)
            max_k = min(10, max(3, len(X) // 5))
            
            for k in range(2, max_k + 1):
                try:
                    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_temp = km_temp.fit_predict(X)
                    
                    if len(np.unique(labels_temp)) > 1 and len(X) > len(np.unique(labels_temp)):
                        sil_score = silhouette_score(X, labels_temp)
                        if sil_score > best_score:
                            best_score = sil_score
                            best_k = k
                except Exception:
                    pass
            
            n_clusters = best_k
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        labels = self.kmeans.fit_predict(X)
        self.cluster_labels = labels

        cluster_silhouette = None
        if len(np.unique(labels)) > 1 and len(X) > len(np.unique(labels)):
            try:
                cluster_silhouette = float(silhouette_score(X, labels))
            except Exception:
                cluster_silhouette = None
        
        # Characterize clusters
        cluster_centers = self.kmeans.cluster_centers_
        cluster_chars = {}
        
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_chars[i] = {
                'size': cluster_mask.sum(),
                'avg_revenue_growth': self.features['revenue_growth'].iloc[cluster_mask].mean(),
                'avg_market_size': self.features['market_size'].iloc[cluster_mask].mean(),
                'avg_opportunity': self.features['opportunity_score'].iloc[cluster_mask].mean(),
                'avg_risk': self.features['risk_score'].iloc[cluster_mask].mean()
            }
        
        # Assign names based on characteristics
        cluster_names = {}
        for i, chars in cluster_chars.items():
            if chars['avg_revenue_growth'] > 30:
                cluster_names[i] = 'GROWTH'
            elif chars['avg_revenue_growth'] < -10:
                cluster_names[i] = 'DECLINING'
            else:
                cluster_names[i] = 'MATURE'
        
        return {
            'model': self.kmeans,
            'labels': labels,
            'cluster_characteristics': cluster_chars,
            'cluster_names': cluster_names,
            'n_clusters_optimal': n_clusters,
            'inertia': self.kmeans.inertia_,
            'silhouette_score': cluster_silhouette,
            'limitations': 'K-Means assumes spherical clusters and similar variance; interpret clusters as coarse segments only.'
        }
    
    def get_cluster_assignment(self) -> pd.Series:
        """Return cluster assignment as strings (GROWTH / MATURE / DECLINING)."""
        if self.cluster_labels is None:
            return None
        
        # Map to GROWTH / MATURE / DECLINING based on characteristics
        cluster_map = {}
        for i in range(len(np.unique(self.cluster_labels))):
            cluster_mask = self.cluster_labels == i
            avg_growth = self.features['revenue_growth'].iloc[cluster_mask].mean()
            
            if avg_growth > 30:
                cluster_map[i] = 'GROWTH'
            elif avg_growth < -10:
                cluster_map[i] = 'DECLINING'
            else:
                cluster_map[i] = 'MATURE'
        
        return pd.Series(
            [cluster_map[label] for label in self.cluster_labels],
            index=self.features.index
        )
    
    # ===== BUILD ALL MODELS =====
    
    def train_all_models(self) -> Dict:
        """Train all models (RF, XGBoost, K-Means)."""
        results = {}
        
        print("Training RandomForest...")
        results['random_forest'] = self.train_random_forest()
        
        print("Training XGBoost (if available)...")
        results['xgboost'] = self.train_xgboost_growth_model()
        
        print("Training K-Means clustering...")
        results['kmeans'] = self.train_kmeans_clustering()
        
        return results
    
    def get_ml_predictions_dataframe(self) -> pd.DataFrame:
        """Return all ML predictions as DataFrame."""
        ml_df = pd.DataFrame(index=self.features.index)
        
        ml_df['rf_classification'] = self.get_rf_classification()
        ml_df['xgb_growth_forecast'] = self.get_xgb_growth_forecast()
        ml_df['cluster_segment'] = self.get_cluster_assignment()
        
        ml_df['molecule_id'] = self.features['molecule_id']
        
        return ml_df


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
    
    ml = MLModels(features)
    results = ml.train_all_models()
    
    print("\n=== Random Forest ===")
    print(f"Score: {results['random_forest']['score']:.3f}")
    print(f"Feature importance (top 5):")
    imp = sorted(results['random_forest']['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, score in imp:
        print(f"  {feat}: {score:.3f}")
    
    print("\n=== XGBoost ===")
    if 'error' not in results['xgboost']:
        print(f"Test score: {results['xgboost']['test_score']:.3f}")
    else:
        print(f"Note: {results['xgboost']['error']}")
    
    print("\n=== K-Means Clustering ===")
    chars = results['kmeans']['cluster_characteristics']
    for cluster_id, char in chars.items():
        print(f"Cluster {cluster_id}: {char['size']} molecules")
        print(f"  Avg Growth: {char['avg_revenue_growth']:.1f}%")
        print(f"  Avg Market Size: ${char['avg_market_size']:.1f}M")
    
    print("\n=== ML Predictions Sample ===")
    ml_df = ml.get_ml_predictions_dataframe()
    print(ml_df.head(10))
