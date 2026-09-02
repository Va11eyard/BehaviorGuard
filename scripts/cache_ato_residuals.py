#!/usr/bin/env python3
"""
Cache standardized embedding/stylometric residuals for sequential ATO streams.

Faster than a full --recompute of sequential_ato_study.py because it skips the
per-message BehaviorGuard evaluator and verifier LR. Used by kappa / distance-
metric ablations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.sequential_ato_study as study  # noqa: E402
from behaviorguard.embedding_config import load_sentence_transformer  # noqa: E402
from scripts.sequential_ato_study import (  # noqa: E402
    LAMBDA_DECAY,
    SEED,
    cosine_distance,
    cusum,
    stylo_features,
)


def _centroid_from_train(embs: list[np.ndarray]) -> np.ndarray:
    mu = embs[0].astype(np.float64).copy()
    for e in embs[1:]:
        mu = LAMBDA_DECAY * mu + (1.0 - LAMBDA_DECAY) * e
    n = np.linalg.norm(mu)
    return mu / n if n > 0 else mu


def cache_residuals(dataset: str) -> Path:
    study.set_dataset(dataset)
    data = json.loads(study.DATASET.read_text(encoding="utf-8"))
    streams = data["streams"]

    all_texts: list[str] = []
    seen: dict[str, int] = {}
    for st in streams:
        for m in st["train"] + st["stream"]:
            t = m["message_text"]
            if t not in seen:
                seen[t] = len(all_texts)
                all_texts.append(t)
    print(f"embedding {len(all_texts)} texts...", flush=True)
    model = load_sentence_transformer()
    emb = model.encode(all_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False).astype(
        np.float32
    )

    def E(t: str) -> np.ndarray:
        return emb[seen[t]]

    feat_cache = {t: stylo_features(t) for t in all_texts}
    global_std = np.std(np.stack(list(feat_cache.values())), axis=0)
    std_floor = np.maximum(0.05 * global_std, 1e-3)

    lengths = []
    episode_start = []
    episode_len = []
    user_ids = []
    res_emb_all, res_sty_all, res_comb_all = [], [], []
    cusum_emb_all = []
    raw_cos_all = []
    ep_shift_deltas = []

    t0 = time.perf_counter()
    for si, st in enumerate(streams):
        train_texts = [m["message_text"] for m in st["train"]]
        stream_texts = [m["message_text"] for m in st["stream"]]
        train_embs = [E(t) for t in train_texts]
        centroid = _centroid_from_train(train_embs)

        F_train = np.stack([feat_cache[t] for t in train_texts])
        f_mean = F_train.mean(axis=0)
        f_std = np.maximum(F_train.std(axis=0), std_floor)
        tr_res_sty = np.abs((F_train - f_mean) / f_std).mean(axis=1)
        mu_sty, sd_sty = tr_res_sty.mean(), max(tr_res_sty.std(), 0.1)

        tr_cos = np.array([cosine_distance(e, centroid) for e in train_embs])
        mu_emb, sd_emb = tr_cos.mean(), max(tr_cos.std(), 0.02)

        F_stream = np.stack([feat_cache[t] for t in stream_texts])
        stream_cos = np.array([cosine_distance(E(t), centroid) for t in stream_texts])
        res_sty = (np.abs((F_stream - f_mean) / f_std).mean(axis=1) - mu_sty) / sd_sty
        res_emb = (stream_cos - mu_emb) / sd_emb
        res_comb = 0.5 * (res_sty + res_emb)

        es = st["episode"]["start_idx"] if st["episode"] else -1
        el = st["episode"]["length"] if st["episode"] else 0
        if es >= 0 and el > 0:
            pre = res_emb[:es] if es > 0 else np.array([0.0])
            post = res_emb[es : es + el]
            ep_shift_deltas.append(float(post.mean() - pre.mean()))

        lengths.append(len(stream_texts))
        episode_start.append(es)
        episode_len.append(el)
        user_ids.append(st["user_id"])
        res_emb_all.append(res_emb)
        res_sty_all.append(res_sty)
        res_comb_all.append(res_comb)
        cusum_emb_all.append(cusum(res_emb))
        raw_cos_all.append(stream_cos)

        if (si + 1) % 1000 == 0:
            print(f"  ... {si + 1}/{len(streams)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    out = study.SCORE_CACHE.parent / f"sequential_ato_residuals_{dataset}.npz"
    np.savez(
        out,
        lengths=np.array(lengths),
        episode_start=np.array(episode_start),
        episode_len=np.array(episode_len),
        user_ids=np.array(user_ids, dtype=object),
        res_emb=np.concatenate(res_emb_all),
        res_sty=np.concatenate(res_sty_all),
        res_comb=np.concatenate(res_comb_all),
        cusum_embed=np.concatenate(cusum_emb_all),
        raw_cosine=np.concatenate(raw_cos_all),
        episode_mean_shift=np.array(ep_shift_deltas),
        delta_hat=float(np.mean(ep_shift_deltas)) if ep_shift_deltas else 0.0,
        seed=SEED,
        lambda_decay=LAMBDA_DECAY,
    )
    print(f"Saved {out}  delta_hat={np.mean(ep_shift_deltas) if ep_shift_deltas else 0:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(study.DATASET_PATHS), default="personachat")
    args = parser.parse_args()
    cache_residuals(args.dataset)


if __name__ == "__main__":
    main()
