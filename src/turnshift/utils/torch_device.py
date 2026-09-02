"""Pick a safe PyTorch device for embedding models (SentenceTransformer)."""

from __future__ import annotations

import os
import warnings


def embedding_device() -> str:
    """
    Return ``cpu`` or ``cuda`` for sentence-transformers / torch models.

  Honors ``TURNSHIFT_DEVICE`` (``cpu`` | ``cuda`` | ``auto``).
  When ``auto`` (default), uses CUDA only if the current GPU compute
  capability is listed in ``torch.cuda.get_arch_list()`` — otherwise CPU.
  This avoids fatal kernel errors on Blackwell (sm_120) with older cu118 wheels.
    """
    env = os.environ.get("TURNSHIFT_DEVICE", "auto").strip().lower()
    if env == "cpu":
        return "cpu"
    if env == "cuda":
        return "cuda"

    try:
        import torch
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability()
    sm = f"sm_{major}{minor}"
    supported = set(torch.cuda.get_arch_list())
    if sm in supported or f"compute_{major}{minor}" in supported:
        return "cuda"

    name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA GPU"
    warnings.warn(
        f"[TurnShift] {name} ({sm}) is not supported by this PyTorch build "
        f"(arch list: {sorted(supported)}). Using CPU for embeddings. "
        "Install PyTorch cu128 for RTX 50-series, or set TURNSHIFT_DEVICE=cpu.",
        stacklevel=2,
    )
    return "cpu"
