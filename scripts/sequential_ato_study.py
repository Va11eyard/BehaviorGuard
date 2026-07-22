#!/usr/bin/env python3
"""
Sequential ATO detection study on the episode-injected PersonaChat dataset.

Question: after a mid-conversation author substitution, how many messages does
it take to detect, at a fixed false-alarm budget?

Detectors (all share the same alarm-time framework; statistic trajectory per stream):
  cusum_stylo     CUSUM over standardized stylometric residuals (primary)
  cusum_embed     CUSUM over standardized embedding-distance residuals (ablation)
  cusum_combined  CUSUM over the mean of both standardized residuals
  permsg_combined same combined residual, no accumulation (isolates CUSUM's value)
  permsg_bg       per-message BehaviorGuard composite (ling-excluded cosine,
                  overrides off - the best-supported per-message config)
  window_embed    CQA-style cumulative context: normalized mean embedding of the
                  last W=5 messages, distance to centroid, standardized
  verifier_lr     per-user authorship verifier (logistic regression,
                  stylometric + embedding-distance features, impostor negatives)

Metrics:
  - Detection rate within episode vs false alarms per 1,000 benign messages
    (operating points chosen from benign-maxima quantiles; no fixed tau)
  - Detection delay (messages from episode start to first alarm)
  - Episode-level ROC-AUC over stream max-statistics, bootstrap 95% CI

Usage:
    set HF_HUB_OFFLINE=1 & set TRANSFORMERS_OFFLINE=1
    python scripts/sequential_ato_study.py [--recompute]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DATASET_PATHS = {
    "personachat": {
        "dataset": ROOT / "datasets" / "personachat_ato_episodes.json",
        "cache": ROOT / "results" / "sequential_ato_scores.npz",
        "out": ROOT / "results" / "sequential_ato_study.json",
    },
    "bst": {
        "dataset": ROOT / "datasets" / "blended_skill_talk_ato_episodes.json",
        "cache": ROOT / "results" / "sequential_ato_scores_bst.npz",
        "out": ROOT / "results" / "sequential_ato_study_bst.json",
    },
}

DATASET = DATASET_PATHS["personachat"]["dataset"]
SCORE_CACHE = DATASET_PATHS["personachat"]["cache"]
OUT_PATH = DATASET_PATHS["personachat"]["out"]


def set_dataset(name: str) -> None:
    """Point the module-level paths at a dataset (default: personachat)."""
    global DATASET, SCORE_CACHE, OUT_PATH
    DATASET = DATASET_PATHS[name]["dataset"]
    SCORE_CACHE = DATASET_PATHS[name]["cache"]
    OUT_PATH = DATASET_PATHS[name]["out"]

SEED = 42
LAMBDA_DECAY = 0.5
CUSUM_KAPPA = 0.5
WINDOW_W = 5
VERIFIER_NEG_RATIO = 5
TARGET_FA_PER_1000 = [0.5, 1.0, 2.0, 5.0, 10.0]
N_BOOTSTRAP = 2000

STOPWORDS = frozenset(
    "a an the i me my we you your he she it they this that is are was were be been "
    "do does did have has had will would can could to of in on at for with and or "
    "but not no so if then than as from by about just really very".split()
)
FIRST_PERSON = frozenset(("i", "me", "my", "mine", "im", "i'm", "we", "our"))
POLITE = ("please", "thank", "sorry", "could you", "would you")


# ---------------------------------------------------------------- features

def stylo_features(text: str) -> np.ndarray:
    """10 content-light stylometric features."""
    words = text.split()
    n = max(len(words), 1)
    lower_words = [w.lower().strip(".,!?;:'\"") for w in words]
    chars = max(len(text), 1)
    letters = [c for c in text if c.isalpha()]
    n_letters = max(len(letters), 1)
    return np.array(
        [
            float(len(words)),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            len(set(lower_words)) / n,
            sum(1 for c in text if c in ".,!?;:'\"-") / chars,
            text.count("?") / n,
            float("!" in text),
            sum(1 for c in letters if c.isupper()) / n_letters,
            sum(1 for w in lower_words if w in STOPWORDS) / n,
            sum(1 for w in lower_words if w in FIRST_PERSON) / n,
            float(any(p in text.lower() for p in POLITE)),
        ],
        dtype=np.float64,
    )


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


# ---------------------------------------------------------------- scoring

def compute_scores() -> dict:
    """Heavy phase: embeddings, PM profiles, per-stream statistic trajectories."""
    import evaluation as ev  # noqa: PLC0415 (heavy import: loads datasets + models)
    from behaviorguard.models import EvaluationInput, SystemConfig  # noqa: PLC0415

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    streams = data["streams"]

    # ---- batch-embed every unique text once
    all_texts: list[str] = []
    seen: dict[str, int] = {}
    for st in streams:
        for m in st["train"] + st["stream"]:
            t = m["message_text"]
            if t not in seen:
                seen[t] = len(all_texts)
                all_texts.append(t)
    print(f"  embedding {len(all_texts)} unique texts...", flush=True)
    t0 = time.perf_counter()
    emb = ev.semantic_analyzer.model.encode(
        all_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False
    ).astype(np.float32)
    print(f"  ... done in {time.perf_counter() - t0:.0f}s", flush=True)

    def E(text: str) -> np.ndarray:
        return emb[seen[text]]

    # ---- global feature scale for std floors
    print("  computing stylometric features...", flush=True)
    feat_cache = {t: stylo_features(t) for t in all_texts}
    global_std = np.std(np.stack(list(feat_cache.values())), axis=0)
    std_floor = np.maximum(0.05 * global_std, 1e-3)

    # ---- frozen per-user baselines (ProfileManager) + residual standardizers
    print("  building ProfileManager profiles (lambda=0.5)...", flush=True)
    builder = ev._build_profile_with_pm(LAMBDA_DECAY)
    bg_config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,
        enable_linguistic_scoring=False,
        linguistic_component_enabled=False,
        enable_semantic_scoring=True,
        enable_temporal_scoring=True,
        semantic_scoring_mode="cosine",
    )

    rng = np.random.default_rng(SEED)
    # Global pool of train messages for verifier impostor negatives
    train_pool_uids: list[str] = []
    train_pool_texts: list[str] = []
    for st in streams:
        for m in st["train"]:
            train_pool_uids.append(st["user_id"])
            train_pool_texts.append(m["message_text"])
    train_pool_uids = np.array(train_pool_uids, dtype=object)
    pool_idx_all = np.arange(len(train_pool_texts))

    det_names = [
        "cusum_stylo",
        "cusum_embed",
        "cusum_combined",
        "permsg_combined",
        "permsg_bg",
        "window_embed",
        "verifier_lr",
    ]
    trajectories: dict[str, list[np.ndarray]] = {d: [] for d in det_names}
    meta = []
    n_skipped = 0
    t0 = time.perf_counter()

    for si, st in enumerate(streams):
        uid = st["user_id"]
        train_msgs = st["train"]
        stream_msgs = st["stream"]
        train_texts = [m["message_text"] for m in train_msgs]
        stream_texts = [m["message_text"] for m in stream_msgs]

        profile = builder(
            {"user_id": uid, "account_age_days": 100},
            [
                {
                    "message_text": m["message_text"],
                    "timestamp": m["timestamp"],
                    "session_id": m["session_id"],
                }
                for m in train_msgs
            ],
        )
        if profile is None:
            n_skipped += 1
            continue
        centroid = np.array(profile.semantic_profile.embedding_centroid, dtype=np.float64)

        # -- stylometric baseline + standardizer (train, in-sample)
        F_train = np.stack([feat_cache[t] for t in train_texts])
        f_mean = F_train.mean(axis=0)
        f_std = np.maximum(F_train.std(axis=0), std_floor)
        tr_res_sty = np.abs((F_train - f_mean) / f_std).mean(axis=1)
        mu_sty, sd_sty = tr_res_sty.mean(), max(tr_res_sty.std(), 0.1)

        # -- embedding baseline standardizer
        tr_res_emb = np.array([cosine_distance(E(t), centroid) for t in train_texts])
        mu_emb, sd_emb = tr_res_emb.mean(), max(tr_res_emb.std(), 0.02)

        # -- window (CQA-style) standardizer: mean-embedding of last <=W train msgs
        tr_win = []
        for i in range(len(train_texts)):
            w = np.stack([E(t) for t in train_texts[max(0, i - WINDOW_W + 1) : i + 1]]).mean(axis=0)
            tr_win.append(cosine_distance(w, centroid))
        tr_win = np.array(tr_win)
        mu_win, sd_win = tr_win.mean(), max(tr_win.std(), 0.02)

        # -- stream residuals
        F_stream = np.stack([feat_cache[t] for t in stream_texts])
        res_sty = (np.abs((F_stream - f_mean) / f_std).mean(axis=1) - mu_sty) / sd_sty
        res_emb = (
            np.array([cosine_distance(E(t), centroid) for t in stream_texts]) - mu_emb
        ) / sd_emb
        res_comb = 0.5 * (res_sty + res_emb)

        win_stat = []
        for i in range(len(stream_texts)):
            w = np.stack([E(t) for t in stream_texts[max(0, i - WINDOW_W + 1) : i + 1]]).mean(axis=0)
            win_stat.append((cosine_distance(w, centroid) - mu_win) / sd_win)

        # -- BehaviorGuard per-message composite
        bg_stat = []
        for i, m in enumerate(stream_msgs):
            prev = stream_msgs[i - 1] if i > 0 and stream_msgs[i - 1]["session_id"] == m["session_id"] else None
            cur = ev.message_to_current_message(
                {"message_text": m["message_text"], "timestamp": m["timestamp"], "session_id": m["session_id"]},
                prev,
                user_profile=profile,
            )
            r = ev.evaluator.evaluate(
                EvaluationInput(user_profile=profile, current_message=cur, system_config=bg_config)
            )
            bg_stat.append(float(r.anomaly_score))

        # -- per-user authorship verifier
        pos = np.hstack([F_train, tr_res_emb.reshape(-1, 1)])
        donor = st["episode"]["donor_id"] if st["episode"] else None
        neg_mask = (train_pool_uids != uid) & (train_pool_uids != donor)
        neg_idx = rng.choice(pool_idx_all[neg_mask], size=min(VERIFIER_NEG_RATIO * len(pos), neg_mask.sum()), replace=False)
        neg = np.stack(
            [
                np.append(feat_cache[train_pool_texts[j]], cosine_distance(E(train_pool_texts[j]), centroid))
                for j in neg_idx
            ]
        )
        X = np.vstack([pos, neg])
        y = np.array([0] * len(pos) + [1] * len(neg))
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(scaler.transform(X), y)
        stream_X = np.hstack(
            [F_stream, np.array([cosine_distance(E(t), centroid) for t in stream_texts]).reshape(-1, 1)]
        )
        ver_stat = clf.predict_proba(scaler.transform(stream_X))[:, 1]

        trajectories["cusum_stylo"].append(cusum(res_sty))
        trajectories["cusum_embed"].append(cusum(res_emb))
        trajectories["cusum_combined"].append(cusum(res_comb))
        trajectories["permsg_combined"].append(res_comb)
        trajectories["permsg_bg"].append(np.array(bg_stat))
        trajectories["window_embed"].append(np.array(win_stat))
        trajectories["verifier_lr"].append(ver_stat)
        meta.append(
            {
                "user_id": uid,
                "episode_start": st["episode"]["start_idx"] if st["episode"] else -1,
                "episode_len": st["episode"]["length"] if st["episode"] else 0,
                "stream_len": len(stream_msgs),
            }
        )
        if (si + 1) % 500 == 0:
            print(f"  ... {si + 1}/{len(streams)} streams ({time.perf_counter() - t0:.0f}s)", flush=True)

    print(f"  scored {len(meta)} streams; skipped {n_skipped} (no profile)", flush=True)

    # Flatten trajectories to a cacheable form
    flat: dict[str, np.ndarray] = {}
    lengths = np.array([m["stream_len"] for m in meta])
    for d in det_names:
        flat[d] = np.concatenate(trajectories[d])
    cache = {
        "lengths": lengths,
        "episode_start": np.array([m["episode_start"] for m in meta]),
        "episode_len": np.array([m["episode_len"] for m in meta]),
        "user_ids": np.array([m["user_id"] for m in meta], dtype=object),
        **flat,
    }
    SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SCORE_CACHE, **cache)
    print(f"  cached statistic trajectories to {SCORE_CACHE.name}", flush=True)
    return cache


def recompute_window_embed(cache: dict) -> dict:
    """
    Rebuild the CQA-style baseline faithfully: embed the CONCATENATION of the
    last <=W messages (like Cumulative Query Auditing re-evaluates the joined
    history), take cosine distance to the frozen centroid, standardized by
    train-window residuals. Replicates ProfileManager's EMA centroid
    (mu_0 = e_0; mu_i = 0.5*mu_{i-1} + 0.5*e_i; L2-normalized) so no profile
    rebuild is needed.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    streams = data["streams"]
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def windows(texts: list[str]) -> list[str]:
        return [" ".join(texts[max(0, i - WINDOW_W + 1) : i + 1]) for i in range(len(texts))]

    all_texts: list[str] = []
    seen: dict[str, int] = {}

    def register(ts: list[str]) -> None:
        for t in ts:
            if t not in seen:
                seen[t] = len(all_texts)
                all_texts.append(t)

    per_stream = []
    for st in streams:
        train_texts = [m["message_text"] for m in st["train"]]
        stream_texts = [m["message_text"] for m in st["stream"]]
        tw, sw = windows(train_texts), windows(stream_texts)
        register(train_texts)
        register(tw)
        register(sw)
        per_stream.append((train_texts, tw, sw))

    print(f"  [window patch] embedding {len(all_texts)} unique texts...", flush=True)
    emb = model.encode(all_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False)

    def E(t: str) -> np.ndarray:
        return emb[seen[t]]

    new_flat: list[np.ndarray] = []
    for train_texts, tw, sw in per_stream:
        mu = E(train_texts[0]).astype(np.float64).copy()
        for t in train_texts[1:]:
            mu = LAMBDA_DECAY * mu + (1.0 - LAMBDA_DECAY) * E(t)
        norm = np.linalg.norm(mu)
        centroid = mu / norm if norm > 0 else mu

        # Per-window-size standardization: window residual magnitude depends on
        # how many messages the window spans, so pooling sizes penalizes short
        # (early) windows and confounds benign-stream maxima.
        tr_res = np.array([cosine_distance(E(w), centroid) for w in tw])
        tr_sizes = np.array([min(i + 1, WINDOW_W) for i in range(len(tw))])
        pooled_mu, pooled_sd = tr_res.mean(), max(tr_res.std(), 0.05)
        by_size: dict[int, tuple[float, float]] = {}
        for s in np.unique(tr_sizes):
            vals = tr_res[tr_sizes == s]
            mu_s = float(vals.mean())
            sd_s = max(float(vals.std()), 0.05) if len(vals) >= 2 else pooled_sd
            by_size[int(s)] = (mu_s, sd_s)

        stream_res = np.array([cosine_distance(E(w), centroid) for w in sw])
        stream_sizes = [min(i + 1, WINDOW_W) for i in range(len(sw))]
        new_flat.append(
            np.array(
                [
                    (r - by_size.get(s, (pooled_mu, pooled_sd))[0])
                    / by_size.get(s, (pooled_mu, pooled_sd))[1]
                    for r, s in zip(stream_res, stream_sizes)
                ]
            )
        )

    cache = dict(cache)
    cache["window_embed"] = np.concatenate(new_flat)
    np.savez(SCORE_CACHE, **cache)
    print("  [window patch] cache updated", flush=True)
    return cache


def load_scores(recompute: bool, patch_window: bool = False) -> dict:
    if SCORE_CACHE.exists() and not recompute:
        print(f"  loading cached trajectories from {SCORE_CACHE.name}")
        d = np.load(SCORE_CACHE, allow_pickle=True)
        cache = {k: d[k] for k in d.files}
        if patch_window:
            cache = recompute_window_embed(cache)
        return cache
    return compute_scores()


# ---------------------------------------------------------------- evaluation

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
) -> dict:
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
    rng = np.random.default_rng(SEED)
    n = len(stream_max)
    boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(is_episode[idx])) < 2:
            continue
        boot.append(roc_auc_score(is_episode[idx].astype(int), stream_max[idx]))
    auc_ci = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]

    operating_points = []
    for target in TARGET_FA_PER_1000:
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
        rng2 = np.random.default_rng(SEED + 1)
        dr_boot = [
            det_flags[rng2.choice(n_episodes, n_episodes, replace=True)].mean()
            for _ in range(N_BOOTSTRAP)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="personachat")
    parser.add_argument(
        "--patch-window",
        action="store_true",
        help="Recompute only the window_embed (CQA-style) baseline in the cache",
    )
    args = parser.parse_args()
    set_dataset(args.dataset)

    print(f"Sequential ATO detection study ({args.dataset})", flush=True)
    cache = load_scores(args.recompute, patch_window=args.patch_window)
    lengths = cache["lengths"].astype(int)
    episode_start = cache["episode_start"].astype(int)
    episode_len = cache["episode_len"].astype(int)

    det_names = [
        "cusum_stylo",
        "cusum_embed",
        "cusum_combined",
        "permsg_combined",
        "permsg_bg",
        "window_embed",
        "verifier_lr",
    ]
    report: dict = {
        "protocol": {
            "dataset": DATASET.name,
            "seed": SEED,
            "cusum_kappa": CUSUM_KAPPA,
            "window_w": WINDOW_W,
            "lambda_decay": LAMBDA_DECAY,
            "n_streams": int(len(lengths)),
            "n_episodes": int((episode_start >= 0).sum()),
            "verifier_negatives_ratio": VERIFIER_NEG_RATIO,
            "fa_definition": "alarm on a benign stream or before episode start; rate per 1000 benign messages",
            "delay_definition": "1-based index of first alarm within the episode window",
        },
        "detectors": {},
    }
    for d in det_names:
        trajs = unflatten(cache[d].astype(float), lengths)
        report["detectors"][d] = evaluate_detector(trajs, episode_start, episode_len)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'detector':<16} {'AUC':>7} {'CI95':>18} | at FA=1/1000: det-rate (CI)    med-delay")
    for d in det_names:
        r = report["detectors"][d]
        op = next(o for o in r["operating_points"] if o["target_fa_per_1000"] == 1.0)
        ci = r["episode_auc_ci95"]
        dci = op["detection_rate_ci95"]
        med = op["median_delay_msgs"]
        print(
            f"{d:<16} {r['episode_auc']:>7.4f} [{ci[0]:.3f},{ci[1]:.3f}]   | "
            f"{op['detection_rate']:>6.1%} [{dci[0]:.1%},{dci[1]:.1%}]  "
            f"{med if med is not None else '-':>5}"
        )
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
