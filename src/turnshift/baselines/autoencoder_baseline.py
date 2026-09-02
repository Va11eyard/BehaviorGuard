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

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
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

    Anomaly score = min-max normalized reconstruction error (MSE) computed on
    the training set's error distribution. Higher score = more anomalous.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = "cpu",
        random_seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model: Optional[Autoencoder] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.MSELoss()
        self.is_fitted = False

        self.feature_mean = None
        self.feature_std = None
        self.reconstruction_error_min: Optional[float] = None
        self.reconstruction_error_max: Optional[float] = None

    def _set_training_seed(self, seed: int) -> torch.Generator:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _init_model(self) -> None:
        self.model = Autoencoder(
            self.input_dim, self.hidden_dims, self.latent_dim
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _normalize_errors_to_scores(self, errors: np.ndarray) -> np.ndarray:
        """Map reconstruction error to a non-negative score; higher = more anomalous.

        Uses training-set min-max without an upper clip so test errors above the
        training maximum retain relative ordering (avoids sigmoid saturation).
        """
        err_range = self.reconstruction_error_max - self.reconstruction_error_min + 1e-8
        return np.maximum((errors - self.reconstruction_error_min) / err_range, 0.0)

    def fit(
        self,
        feature_vectors: np.ndarray,
        verbose: bool = True,
        random_seed: Optional[int] = None,
    ):
        """
        Train autoencoder on normal data.

        Model weights are initialized after seeding so consecutive fits with
        the same seed produce identical parameters and scores.
        """
        seed = self.random_seed if random_seed is None else random_seed
        generator = self._set_training_seed(seed)
        self._init_model()

        self.feature_mean = np.mean(feature_vectors, axis=0)
        self.feature_std = np.std(feature_vectors, axis=0) + 1e-8

        normalized_features = (feature_vectors - self.feature_mean) / self.feature_std
        X_tensor = torch.FloatTensor(normalized_features).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        self.model.train()
        epoch_iterator = (
            tqdm(range(self.epochs), desc="Training") if verbose else range(self.epochs)
        )

        for _epoch in epoch_iterator:
            total_loss = 0.0
            for batch in dataloader:
                X_batch = batch[0]
                reconstruction = self.model(X_batch)
                loss = self.criterion(reconstruction, X_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if verbose and hasattr(epoch_iterator, "set_postfix"):
                epoch_iterator.set_postfix({"loss": f"{avg_loss:.4f}"})

        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            errors = torch.mean((reconstruction - X_tensor) ** 2, dim=1).cpu().numpy()
            self.reconstruction_error_min = float(np.min(errors))
            self.reconstruction_error_max = float(np.max(errors))

        self.is_fitted = True

    def predict(self, feature_vectors: np.ndarray) -> Dict:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        normalized_features = (feature_vectors - self.feature_mean) / self.feature_std
        X_tensor = torch.FloatTensor(normalized_features).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            errors = torch.mean((reconstruction - X_tensor) ** 2, dim=1).cpu().numpy()

        anomaly_scores = self._normalize_errors_to_scores(errors)
        is_anomaly = anomaly_scores > 0.5

        return {
            "anomaly_scores": anomaly_scores,
            "reconstruction_errors": errors,
            "is_anomaly": is_anomaly,
            "component_scores": {
                "semantic": 0.0,
                "linguistic": 0.0,
                "temporal": 0.0,
                "overall": anomaly_scores,
            },
        }

    def detect_single(self, feature_vector: np.ndarray) -> Dict:
        result = self.predict(feature_vector.reshape(1, -1))

        return {
            "anomaly_score": float(result["anomaly_scores"][0]),
            "reconstruction_error": float(result["reconstruction_errors"][0]),
            "is_anomaly": bool(result["is_anomaly"][0]),
            "component_scores": {
                "semantic": 0.0,
                "linguistic": 0.0,
                "temporal": 0.0,
                "overall": float(result["anomaly_scores"][0]),
            },
        }

    def save(self, filepath: str):
        torch.save({
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "reconstruction_error_min": self.reconstruction_error_min,
            "reconstruction_error_max": self.reconstruction_error_max,
            "is_fitted": self.is_fitted,
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]
        self.latent_dim = checkpoint["latent_dim"]
        self.learning_rate = checkpoint.get("learning_rate", 0.001)
        self.epochs = checkpoint.get("epochs", 50)
        self.batch_size = checkpoint.get("batch_size", 32)
        self.random_seed = checkpoint.get("random_seed", 42)
        self._init_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.feature_mean = checkpoint["feature_mean"]
        self.feature_std = checkpoint["feature_std"]
        self.reconstruction_error_min = checkpoint.get(
            "reconstruction_error_min",
            checkpoint.get("reconstruction_error_mean"),
        )
        self.reconstruction_error_max = checkpoint.get(
            "reconstruction_error_max",
            checkpoint.get("reconstruction_error_mean"),
        )
        self.is_fitted = checkpoint["is_fitted"]
