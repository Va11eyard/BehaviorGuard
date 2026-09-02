#!/usr/bin/env python3
"""
Online-profile and dual-lambda CUSUM ablations for sequential ATO detection.

1) Online update: after each stream message the EMA centroid absorbs that message
   (ProfileManager.update_profile semantics). Detects how much signal is lost when
   an attacker poisons the profile in real time.
2) Dual-lambda: maintain fast (lambda=0.50) and slow (lambda=0.99) centroids;
   CUSUM on the divergence |cos(fast, msg) - cos(slow, msg)|, which lags under
   gradual poisoning.
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
from turnshift.embedding_config import load_sentence_transformer  # noqa: E402
from scripts.sequential_ato_study import (  # noqa: E402
    CUSUM_KAPPA,
    SEED,
    cosine_distance,
    cusum,
    evaluate_detector,
    unflatten,
)

FAST_LAMBDA = 0.50
SLOW_LAMBDA = 0.99


def _ema_update(mu: np.ndarray, e: np.ndarray, decay: float) -> np.ndarray:
    out = decay * mu + (1.0 - decay) * e
    n = np.linalg.norm(out)
    return out / n if n > 0 else out


def run_ablations(dataset: str) -> dict:
    study.set_dataset(dataset)
    data = json.loads(study.DATASET.read_text(encoding="utf-8"))
    streams = data["streams"]

    texts: list[str] = []
    seen: dict[str, int] = {}
    for st in streams:
        for m in st["train"] + st["stream"]:
            t = m["message_text"]
            if t not in seen:
                seen[t] = len(texts)
                texts.append(t)
    print(f"embedding {len(texts)} texts...", flush=True)
    model = load_sentence_transformer()
    emb = model.encode(texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False).astype(
        np.float32
    )

    def E(t: str) -> np.ndarray:
        return emb[seen[t]]

    lengths, es, el, uids = [], [], [], []
    frozen_res, online_res, dual_res = [], [], []
    t0 = time.perf_counter()

    for si, st in enumerate(streams):
        train_texts = [m["message_text"] for m in st["train"]]
        stream_texts = [m["message_text"] for m in st["stream"]]

        mu_fast = E(train_texts[0]).astype(np.float64).copy()
        mu_slow = mu_fast.copy()
        for t in train_texts[1:]:
            e = E(t)
            mu_fast = _ema_update(mu_fast, e, FAST_LAMBDA)
            mu_slow = _ema_update(mu_slow, e, SLOW_LAMBDA)

        # train residual standardizers (frozen centroid = fast after train)
        tr_cos = np.array([cosine_distance(E(t), mu_fast) for t in train_texts])
        mu_e, sd_e = tr_cos.mean(), max(tr_cos.std(), 0.02)
        tr_dual = np.array(
            [abs(cosine_distance(E(t), mu_fast) - cosine_distance(E(t), mu_slow)) for t in train_texts]
        )
        mu_d, sd_d = tr_dual.mean(), max(tr_dual.std(), 0.01)

        # frozen residual on stream (centroid held fixed)
        fr = (np.array([cosine_distance(E(t), mu_fast) for t in stream_texts]) - mu_e) / sd_e

        # online: update after each message
        mu_on = mu_fast.copy()
        on = []
        for t in stream_texts:
            r = (cosine_distance(E(t), mu_on) - mu_e) / sd_e
            on.append(r)
            mu_on = _ema_update(mu_on, E(t), FAST_LAMBDA)

        # dual-lambda divergence (both update online after scoring)
        mu_f, mu_s = mu_fast.copy(), mu_slow.copy()
        du = []
        for t in stream_texts:
            e = E(t)
            div = abs(cosine_distance(e, mu_f) - cosine_distance(e, mu_s))
            du.append((div - mu_d) / sd_d)
            mu_f = _ema_update(mu_f, e, FAST_LAMBDA)
            mu_s = _ema_update(mu_s, e, SLOW_LAMBDA)

        lengths.append(len(stream_texts))
        es.append(st["episode"]["start_idx"] if st["episode"] else -1)
        el.append(st["episode"]["length"] if st["episode"] else 0)
        uids.append(st["user_id"])
        frozen_res.append(fr)
        online_res.append(np.array(on))
        dual_res.append(np.array(du))
        if (si + 1) % 1000 == 0:
            print(f"  ... {si + 1}/{len(streams)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    lengths_a = np.array(lengths)
    es_a = np.array(es)
    el_a = np.array(el)
    report = {"dataset": dataset, "seed": SEED, "detectors": {}}
    for name, residuals in (
        ("cusum_embed_frozen", frozen_res),
        ("cusum_embed_online", online_res),
        ("cusum_dual_lambda", dual_res),
    ):
        trajs = [cusum(r, kappa=CUSUM_KAPPA) for r in residuals]
        m = evaluate_detector(trajs, es_a, el_a)
        op1 = next(o for o in m["operating_points"] if o["target_fa_per_1000"] == 1.0)
        report["detectors"][name] = {
            "auc": m["episode_auc"],
            "auc_ci95": m["episode_auc_ci95"],
            "det_rate_fa1": op1["detection_rate"],
            "det_rate_fa1_ci95": op1["detection_rate_ci95"],
            "median_delay": op1["median_delay_msgs"],
        }

    frozen = report["detectors"]["cusum_embed_frozen"]["det_rate_fa1"]
    online = report["detectors"]["cusum_embed_online"]["det_rate_fa1"]
    report["online_detection_loss_points"] = round(100 * (frozen - online), 2)
    report["notes"] = {
        "frozen": "canonical study protocol (profile frozen after train)",
        "online": "EMA absorbs each stream message after scoring (production realism / poisoning)",
        "dual_lambda": f"CUSUM on |d_fast - d_slow| with lambdas {FAST_LAMBDA}/{SLOW_LAMBDA}",
    }
    out = ROOT / "results" / f"sequential_ato_online_dual_{dataset}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="personachat")
    args = parser.parse_args()
    run_ablations(args.dataset)


if __name__ == "__main__":
    main()
