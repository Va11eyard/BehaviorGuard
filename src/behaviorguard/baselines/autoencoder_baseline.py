"""Autoencoder baseline for anomaly detection."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm


class Autoencoder(nn.Module):
    """Simple feedforward autoencoder."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderBaseline:
    """
    Autoencoder-based anomaly detector.
    
    Anomaly score = reconstruction error (MSE between input and output).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize autoencoder detector.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent space dimension
            learning_rate: Learning rate for Adam optimizer
            epochs: Training epochs
            batch_size: Batch size for training
            device: 'cuda' or 'cpu'
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = Autoencoder(input_dim, hidden_dims, latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_fitted = False
        
        # For normalization
        self.feature_mean = None
        self.feature_std = None
        
        # For score normalization
        self.reconstruction_error_mean = None
        self.reconstruction_error_std = None
    
    def fit(self, feature_vectors: np.ndarray, verbose: bool = True):
        """
        Train autoencoder on normal data.
        
        Args:
            feature_vectors: (n_samples, n_features) array of normal data
            verbose: Whether to show progress bar
        """
        # Normalize features
        self.feature_mean = np.mean(feature_vectors, axis=0)
        self.feature_std = np.std(feature_vectors, axis=0) + 1e-8
        
        normalized_features = (feature_vectors - self.feature_mean) / self.feature_std
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(normalized_features).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Training loop
        self.model.train()
        epoch_iterator = tqdm(range(self.epochs), desc="Training") if verbose else range(self.epochs)
        
        for epoch in epoch_iterator:
            total_loss = 0.0
            for batch in dataloader:
                X_batch = batch[0]
                
                # Forward pass
                reconstruction = self.model(X_batch)
                loss = self.criterion(reconstruction, X_batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if verbose and epoch % 10 == 0:
                if hasattr(epoch_iterator, 'set_postfix'):
                    epoch_iterator.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # Compute reconstruction errors on training data for normalization
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            errors = torch.mean((reconstruction - X_tensor) ** 2, dim=1).cpu().numpy()
            self.reconstruction_error_mean = np.mean(errors)
            self.reconstruction_error_std = np.std(errors) + 1e-8
        
        self.is_fitted = True
    
    def predict(self, feature_vectors: np.ndarray) -> Dict:
        """
        Predict anomaly scores based on reconstruction error.
        
        Args:
            feature_vectors: (n_samples, n_features) array
        
        Returns:
            Dict with anomaly_scores, is_anomaly, component_scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Normalize features
        normalized_features = (feature_vectors - self.feature_mean) / self.feature_std
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(normalized_features).to(self.device)
        
        # Get reconstruction
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            
            # Compute reconstruction error (MSE per sample)
            errors = torch.mean((reconstruction - X_tensor) ** 2, dim=1).cpu().numpy()
        
        # Normalize errors to [0, 1] using z-score
        z_scores = (errors - self.reconstruction_error_mean) / self.reconstruction_error_std
        
        # Convert z-scores to [0, 1] using sigmoid
        anomaly_scores = 1 / (1 + np.exp(-z_scores))
        
        # Threshold at 0.5 for binary classification
        is_anomaly = anomaly_scores > 0.5
        
        return {
            'anomaly_scores': anomaly_scores,
            'reconstruction_errors': errors,
            'is_anomaly': is_anomaly,
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
            'reconstruction_error': float(result['reconstruction_errors'][0]),
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'reconstruction_error_mean': self.reconstruction_error_mean,
            'reconstruction_error_std': self.reconstruction_error_std,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.feature_mean = checkpoint['feature_mean']
        self.feature_std = checkpoint['feature_std']
        self.reconstruction_error_mean = checkpoint['reconstruction_error_mean']
        self.reconstruction_error_std = checkpoint['reconstruction_error_std']
        self.is_fitted = checkpoint['is_fitted']
