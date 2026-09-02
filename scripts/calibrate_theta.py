#!/usr/bin/env python3
"""Calibrate template-similarity threshold θ on held-out benign and injection samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from turnshift.overrides.template_override import TemplateOverrideProvider


def _stable_fraction(key: str, salt: str, frac: float) -> bool:
    digest = hashlib.md5(f"{salt}:{key}".encode(), usedforsecurity=False).hexdigest()
    bucket = int(digest, 16) % 10_000
    return bucket < int(frac * 10_000)


def _load_personachat_train_messages(dataset_path: Path) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    train_user_ids = set(data["splits"]["train"]["user_ids"])
    return [m for m in data["messages"] if m["user_id"] in train_user_ids]


def _split_calibration_sets(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hold out 10% benign and 20% injected messages from the train split."""
    benign_cal: list[dict] = []
    injection_cal: list[dict] = []
    for msg in messages:
        msg_key = msg.get("message_id") or f"{msg['user_id']}:{msg['message_text'][:40]}"
        if msg.get("should_flag", False):
            if _stable_fraction(msg_key, "inj_cal", 0.20):
                injection_cal.append(msg)
        elif _stable_fraction(msg_key, "benign_cal", 0.10):
            benign_cal.append(msg)
    return benign_cal, injection_cal


def _max_similarities(provider: TemplateOverrideProvider, messages: list[dict]) -> np.ndarray:
    if not messages:
        return np.array([], dtype=np.float64)
    texts = [m["message_text"] for m in messages]
    embeddings = provider.model.encode(texts, normalize_embeddings=True)
    return np.array([provider.max_similarity(emb) for emb in embeddings], dtype=np.float64)


def _rates_at_theta(scores: np.ndarray, theta: float) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores >= theta))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate template override θ")
    parser.add_argument(
        "--template-path",
        default=str(ROOT / "data" / "jailbreak_templates.json"),
        help="JSON file with [{text, category}] templates",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(ROOT / "datasets" / "personachat_processed.json"),
        help="PersonaChat processed dataset (train split used for calibration)",
    )
    parser.add_argument(
        "--output-figure",
        default=str(ROOT / "figures" / "theta_calibration.png"),
        help="Path for similarity distribution plot",
    )
    args = parser.parse_args()

    train_messages = _load_personachat_train_messages(Path(args.dataset_path))
    benign_msgs, injection_msgs = _split_calibration_sets(train_messages)
    print(f"Calibration benign messages: {len(benign_msgs)}")
    print(f"Calibration injection messages: {len(injection_msgs)}")

    provider = TemplateOverrideProvider(args.template_path, theta=0.0)
    benign_scores = _max_similarities(provider, benign_msgs)
    injection_scores = _max_similarities(provider, injection_msgs)

    fig_path = Path(args.output_figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if benign_scores.size:
        ax.hist(benign_scores, bins=30, alpha=0.6, label="benign (train holdout)", density=True)
    if injection_scores.size:
        ax.hist(injection_scores, bins=30, alpha=0.6, label="injection (train holdout)", density=True)
    ax.set_xlabel("Max template cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Template similarity calibration (PersonaChat train holdout)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved distribution plot to {fig_path}")

    thetas = [round(v, 2) for v in np.arange(0.70, 0.9501, 0.02)]
    print("\nθ | TPR | FPR")
    print("-" * 28)
    recommended_theta = None
    for theta in thetas:
        tpr = _rates_at_theta(injection_scores, theta)
        fpr = _rates_at_theta(benign_scores, theta)
        marker = ""
        if recommended_theta is None and fpr <= 0.01:
            recommended_theta = theta
            marker = "  <= FPR target"
        print(f"{theta:.2f} | {tpr:.4f} | {fpr:.4f}{marker}")

    if recommended_theta is not None:
        print(f"\nRecommended θ (benign FPR <= 0.01): {recommended_theta:.2f}")
    else:
        print("\nNo θ in sweep achieved benign FPR <= 0.01")


if __name__ == "__main__":
    main()
