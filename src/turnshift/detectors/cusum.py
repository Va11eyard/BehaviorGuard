"""CUSUM over embedding residuals and episode-level evaluation.

Logic is copied from the sequential ATO study scripts without changing
thresholds, delay indexing, or bootstrap seeds.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 42
CUSUM_KAPPA = 0.5
TARGET_FA_PER_1000 = [0.5, 1.0, 2.0, 5.0, 10.0]
N_BOOTSTRAP = 2000


def cosine_distance(e: np.ndarray, c: np.ndarray) -> float:
    ne, nc = np.linalg.norm(e), np.linalg.norm(c)
    if ne == 0 or nc == 0:
        return 1.0
    return float(1.0 - np.dot(e / ne, c / nc))


def cusum(residuals: np.ndarray, kappa: float = CUSUM_KAPPA) -> np.ndarray:
    out = np.empty_like(residuals)
    s = 0.0
    for i, r in enumerate(residuals):
        s = max(0.0, s + r - kappa)
        out[i] = s
    return out


def unflatten(flat: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    out, pos = [], 0
    for n in lengths:
        out.append(flat[pos : pos + n])
        pos += n
    return out


def evaluate_detector(
    trajs: list[np.ndarray],
    episode_start: np.ndarray,
    episode_len: np.ndarray,
    *,
    seed: int = SEED,
    n_bootstrap: int = N_BOOTSTRAP,
    target_fa_per_1000: list[float] | None = None,
) -> dict:
    if target_fa_per_1000 is None:
        target_fa_per_1000 = TARGET_FA_PER_1000
    is_episode = episode_start >= 0
    benign_maxima = []          # candidate false alarms (one per stream / pre-episode segment)
    n_benign_msgs = 0
    for tr, es in zip(trajs, episode_start):
        if es < 0:
            benign_maxima.append(tr.max())
            n_benign_msgs += len(tr)
        else:
            if es > 0:
                benign_maxima.append(tr[:es].max())
            n_benign_msgs += int(es)
    n_benign_msgs = int(n_benign_msgs)
    benign_maxima = np.array(benign_maxima)

    # Episode-level AUC over whole-stream max statistic
    stream_max = np.array([tr.max() for tr in trajs])
    auc = float(roc_auc_score(is_episode.astype(int), stream_max))
    rng = np.random.default_rng(seed)
    n = len(stream_max)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(is_episode[idx])) < 2:
            continue
        boot.append(roc_auc_score(is_episode[idx].astype(int), stream_max[idx]))
    auc_ci = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]

    operating_points = []
    for target in target_fa_per_1000:
        # threshold from benign maxima so that expected FAs ~= target rate
        n_fa_allowed = target * n_benign_msgs / 1000.0
        q = 1.0 - n_fa_allowed / max(len(benign_maxima), 1)
        q = min(max(q, 0.0), 1.0)
        h = float(np.quantile(benign_maxima, q))

        n_fa = int((benign_maxima > h).sum())
        detected_delays = []
        n_pre_alarm = 0
        for tr, es, el in zip(trajs, episode_start, episode_len):
            if es < 0:
                continue
            if es > 0 and tr[:es].max() > h:
                n_pre_alarm += 1
                continue
            ep = tr[es : es + el]
            hits = np.nonzero(ep > h)[0]
            if len(hits):
                detected_delays.append(int(hits[0]) + 1)
        n_episodes = int(is_episode.sum())
        det_rate = len(detected_delays) / n_episodes
        # bootstrap CI on detection rate over episodes
        det_flags = np.array(
            [1] * len(detected_delays) + [0] * (n_episodes - len(detected_delays))
        )
        rng2 = np.random.default_rng(seed + 1)
        dr_boot = [
            det_flags[rng2.choice(n_episodes, n_episodes, replace=True)].mean()
            for _ in range(n_bootstrap)
        ]
        operating_points.append(
            {
                "target_fa_per_1000": target,
                "threshold": round(h, 4),
                "actual_fa_per_1000": round(1000.0 * n_fa / n_benign_msgs, 3),
                "n_false_alarms": n_fa,
                "detection_rate": round(det_rate, 4),
                "detection_rate_ci95": [
                    round(float(np.percentile(dr_boot, 2.5)), 4),
                    round(float(np.percentile(dr_boot, 97.5)), 4),
                ],
                "n_detected": len(detected_delays),
                "n_pre_episode_alarms": n_pre_alarm,
                "median_delay_msgs": float(np.median(detected_delays)) if detected_delays else None,
                "mean_delay_msgs": round(float(np.mean(detected_delays)), 2) if detected_delays else None,
            }
        )

    return {
        "episode_auc": round(auc, 4),
        "episode_auc_ci95": auc_ci,
        "n_benign_messages": n_benign_msgs,
        "operating_points": operating_points,
    }


def placebo_continuation(
    std_train_residuals: np.ndarray,
    s0: float,
    episode_len: int,
    thresholds: dict,
    *,
    kappa: float = CUSUM_KAPPA,
    n_sim: int = 500,
    rng: np.random.Generator,
) -> dict:
    """Vectorized CUSUM continuation with bootstrap-resampled same-author residuals."""
    draws = rng.choice(std_train_residuals, size=(n_sim, episode_len), replace=True)
    s = np.full(n_sim, s0)
    crossed = {k: np.zeros(n_sim, dtype=bool) for k in thresholds}
    for j in range(episode_len):
        s = np.maximum(0.0, s + draws[:, j] - kappa)
        for k, h in thresholds.items():
            crossed[k] |= s > h
    return {k: float(crossed[k].mean()) for k in thresholds}
