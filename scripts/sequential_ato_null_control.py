#!/usr/bin/env python3
"""
Placebo (null) control for the sequential ATO study's cusum_embed detector.

Concern: episode streams contain 5-10 more statistic draws than benign streams,
so some detections could be length artifacts. Control: for each episode stream,
continue the CUSUM from its true pre-episode state, but feed k residuals
bootstrap-sampled from the user's OWN standardized train residuals (same
author, same length) instead of the attacker's. The crossing rate at the
FA=1/1000 threshold estimates the length-induced null detection rate.

Appends {"null_control_cusum_embed": ...} to results/sequential_ato_study.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.sequential_ato_study as study  # noqa: E402
from scripts.sequential_ato_study import (  # noqa: E402
    CUSUM_KAPPA,
    LAMBDA_DECAY,
    SEED,
    cosine_distance,
    unflatten,
)

N_SIM = 500


def main() -> None:
    import argparse

    from turnshift.embedding_config import load_sentence_transformer

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(study.DATASET_PATHS), default="personachat")
    args = parser.parse_args()
    study.set_dataset(args.dataset)
    DATASET, SCORE_CACHE, OUT_PATH = study.DATASET, study.SCORE_CACHE, study.OUT_PATH

    report = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    ops = report["detectors"]["cusum_embed"]["operating_points"]
    thresholds = {op["target_fa_per_1000"]: op["threshold"] for op in ops}

    cache = np.load(SCORE_CACHE, allow_pickle=True)
    lengths = cache["lengths"].astype(int)
    episode_start = cache["episode_start"].astype(int)
    episode_len = cache["episode_len"].astype(int)
    user_ids = cache["user_ids"]
    trajs = unflatten(cache["cusum_embed"].astype(float), lengths)

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    train_by_user = {st["user_id"]: [m["message_text"] for m in st["train"]] for st in data["streams"]}

    ep_idx = [i for i in range(len(lengths)) if episode_start[i] >= 0]
    texts: list[str] = []
    seen: dict[str, int] = {}
    for i in ep_idx:
        for t in train_by_user[str(user_ids[i])]:
            if t not in seen:
                seen[t] = len(texts)
                texts.append(t)
    print(f"embedding {len(texts)} train texts for {len(ep_idx)} episode users...", flush=True)
    model = load_sentence_transformer()
    emb = model.encode(texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False)

    def E(t: str) -> np.ndarray:
        return emb[seen[t]]

    rng = np.random.default_rng(SEED)
    null_rates = {k: [] for k in thresholds}
    for i in ep_idx:
        uid = str(user_ids[i])
        train_texts = train_by_user[uid]
        # Replicate ProfileManager EMA centroid (mu_i = 0.5*mu + 0.5*e, normalized)
        mu = E(train_texts[0]).astype(np.float64).copy()
        for t in train_texts[1:]:
            mu = LAMBDA_DECAY * mu + (1.0 - LAMBDA_DECAY) * E(t)
        norm = np.linalg.norm(mu)
        centroid = mu / norm if norm > 0 else mu

        tr_res = np.array([cosine_distance(E(t), centroid) for t in train_texts])
        mu_e, sd_e = tr_res.mean(), max(tr_res.std(), 0.02)
        std_res = (tr_res - mu_e) / sd_e

        es, el = episode_start[i], episode_len[i]
        s0 = trajs[i][es - 1] if es > 0 else 0.0

        draws = rng.choice(std_res, size=(N_SIM, el), replace=True)
        # vectorized CUSUM over simulations
        s = np.full(N_SIM, s0)
        crossed = {k: np.zeros(N_SIM, dtype=bool) for k in thresholds}
        for j in range(el):
            s = np.maximum(0.0, s + draws[:, j] - CUSUM_KAPPA)
            for k, h in thresholds.items():
                crossed[k] |= s > h
        for k in thresholds:
            null_rates[k].append(crossed[k].mean())

    out = {
        "method": (
            "per-episode CUSUM continuation from true pre-episode state with k residuals "
            f"bootstrap-sampled from the user's own standardized train residuals (n_sim={N_SIM}, seed={SEED})"
        ),
        "null_detection_rate_at_fa": {
            str(k): round(float(np.mean(v)), 4) for k, v in null_rates.items()
        },
        "observed_detection_rate_at_fa": {
            str(op["target_fa_per_1000"]): op["detection_rate"] for op in ops
        },
    }
    report["null_control_cusum_embed"] = out
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Appended to {OUT_PATH}")


if __name__ == "__main__":
    main()
