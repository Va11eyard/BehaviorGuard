"""Pinned embedding-model identity for reproducible scoring.

The model name alone is not enough for bit-exact reproducibility across
Hugging Face cache refreshes; pin the Hub revision hash as well.
"""

from __future__ import annotations

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_MODEL_HF_ID = f"sentence-transformers/{EMBEDDING_MODEL_NAME}"


def load_sentence_transformer(**kwargs):
    """Load the pinned MiniLM encoder (revision-locked)."""
    from sentence_transformers import SentenceTransformer

    from behaviorguard.utils.torch_device import embedding_device

    device = kwargs.pop("device", None) or embedding_device()
    return SentenceTransformer(
        EMBEDDING_MODEL_HF_ID,
        revision=EMBEDDING_MODEL_REVISION,
        device=device,
        **kwargs,
    )
