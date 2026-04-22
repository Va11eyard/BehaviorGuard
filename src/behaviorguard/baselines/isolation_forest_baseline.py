"""Isolation Forest baseline for anomaly detection."""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional
import pickle


class IsolationForestBaseline:
    """
    Isolation Forest anomaly detector on feature vectors.
    
    This baseline uses sklearn's Isolation Forest algorithm on the same
    features as BehaviorGuard but without the structured multi-dimensional
    scoring approach.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of trees
            max_samples: Number of samples to draw for each tree
            contamination: Expected proportion of anomalies
            random_state: Random seed
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # parallel training; predict() is single-sample in evaluation
        )
        self.is_fitted = False
        self.feature_mean = None
        self.feature_std = None
        # Store training score statistics for normalization
        self.train_score_mean = None
        self.train_score_std = None
    
    def fit(self, feature_vectors: np.ndarray):
        """
        Train Isolation Forest on normal data.
        
        Args:
            feature_vectors: (n_samples, n_features) array
        """
        # Normalize features
        self.feature_mean = np.mean(feature_vectors, axis=0)
        self.feature_std = np.std(feature_vectors, axis=0) + 1e-8
        
        normalized_features = (feature_vectors - self.feature_mean) / self.feature_std
        
        self.model.fit(normalized_features)
        
        # Compute training score statistics for normalization
        train_scores = self.model.score_samples(normalized_features)
        self.train_score_mean = np.mean(train_scores)
        self.train_score_std = np.std(train_scores) + 1e-8
        
        self.is_fitted = True
    
    def predict(self, feature_vectors: np.ndarray) -> Dict:
        """
        Predict anomaly scores.
        
        Args:
            feature_vectors: (n_samples, n_features) array
        
        Returns:
            Dict with:
                - anomaly_scores: array of scores in [0, 1]
                - predictions: array of -1 (anomaly) or 1 (normal)
                - is_anomaly: boolean array
                - component_scores: dict with overall score
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Normalize features
        normalized_features = (feature_vectors - self.feature_mean) / self.feature_std
        
        # Get anomaly scores (more negative = more normal, less negative = more anomalous)
        raw_scores = self.model.score_samples(normalized_features)
        
        # Normalize using training statistics (z-score approach)
        z_scores = (raw_scores - self.train_score_mean) / self.train_score_std
        
        # Convert z-scores to [0, 1] using sigmoid
        # Negative z-scores (below training mean) = more anomalous
        anomaly_scores = 1 / (1 + np.exp(z_scores))  # Sigmoid of negative z-score
        
        # Get binary predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(normalized_features)
        
        return {
            'anomaly_scores': anomaly_scores,
            'predictions': predictions,
            'is_anomaly': predictions == -1,
            'component_scores': {
                'semantic': 0.0,  # Not separated
                'linguistic': 0.0,  # Not separated
                'temporal': 0.0,  # Not separated
                'overall': anomaly_scores
            }
        }
    
    def detect_single(self, feature_vector: np.ndarray) -> Dict:
        """
        Detect anomaly for a single sample.
        
        Args:
            feature_vector: (n_features,) array
        
        Returns:
            Dict with anomaly_score, is_anomaly, component_scores
        """
        result = self.predict(feature_vector.reshape(1, -1))
        
        return {
            'anomaly_score': float(result['anomaly_scores'][0]),
            'is_anomaly': bool(result['is_anomaly'][0]),
            'component_scores': {
                'semantic': 0.0,
                'linguistic': 0.0,
                'temporal': 0.0,
                'overall': float(result['anomaly_scores'][0])
            }
        }
    
    def save(self, filepath: str):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_fitted': self.is_fitted,
                'feature_mean': self.feature_mean,
                'feature_std': self.feature_std,
                'train_score_mean': self.train_score_mean,
                'train_score_std': self.train_score_std
            }, f)
    
    def load(self, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.is_fitted = data['is_fitted']
            self.feature_mean = data['feature_mean']
            self.feature_std = data['feature_std']
            self.train_score_mean = data['train_score_mean']
            self.train_score_std = data['train_score_std']
