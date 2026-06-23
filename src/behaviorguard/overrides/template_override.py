"""Embedding-based template override (Override 4) for jailbreak / ATO phrases."""

from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from behaviorguard.utils.torch_device import embedding_device


class TemplateOverrideProvider:
    """Match message embeddings against curated jailbreak template embeddings."""

    def __init__(
        self,
        template_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        theta: float = 0.82,
    ):
        self.theta = theta
        self.model = SentenceTransformer(model_name, device=embedding_device())
        self.templates, self.labels = self._load_and_embed(template_path)

    def _load_and_embed(self, path: str) -> tuple[np.ndarray, list[str]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)  # expects [{text: str, category: str}]
        texts = [d["text"] for d in data]
        labels = [d["category"] for d in data]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float64), labels

    def _normalize(self, message_embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(message_embedding, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def max_similarity(self, message_embedding: np.ndarray) -> float:
        """Return maximum cosine similarity to any template."""
        sims = self.templates @ self._normalize(message_embedding)
        return float(np.max(sims))

    def check_override(self, message_embedding: np.ndarray) -> Tuple[float, Optional[str]]:
        """Return (max_similarity, matched_category) or (sim, None) if below theta."""
        vec = self._normalize(message_embedding)
        sims = self.templates @ vec
        max_idx = int(np.argmax(sims))
        max_sim = float(sims[max_idx])
        category = self.labels[max_idx] if max_sim >= self.theta else None
        return max_sim, category
