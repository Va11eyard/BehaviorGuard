#!/usr/bin/env python3
"""
Compare linguistic configurations by AUC-PR on the full corrected holdout:
  - linguistic off (cosine+temporal only)
  - legacy 4-feature linguistic
  - stylometric 10-feature linguistic

Writes results/stylometric_linguistic_eval.json and recommends the canonical default.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from behaviorguard.analyzers.linguistic_ml import LinguisticAnalyzerML  # noqa: E402
from behaviorguard.models import EvaluationInput, SystemConfig  # noqa: E402

DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"
SEED = 42


def score_config(label: str, ling_mode: str) -> dict:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    builder = ev._build_profile_with_pm(0.50)
    by = defaultdict(list)
    for m in data["messages"]:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    users = {u["user_id"]: u for u in data["users"]}

    enable = ling_mode != "off"
    analyzer = None
    if ling_mode == "stylometric":
        analyzer = LinguisticAnalyzerML(use_stylometric=True)
    elif ling_mode == "legacy4":
        analyzer = LinguisticAnalyzerML(use_stylometric=False)

    config = SystemConfig(
        sensitivity_level="medium",
        deployment_context="enterprise",
        overrides_enabled=False,
        enable_linguistic_scoring=enable,
        linguistic_component_enabled=enable,
        enable_semantic_scoring=True,
        enable_temporal_scoring=True,
    )

    y, scores = [], []
    for uid, msgs in sorted(by.items()):
        split = int(len(msgs) * 0.8)
        profile = builder(users[uid], msgs[:split])
        if profile is None:
            continue
        test = msgs[split:]
        for i, msg in enumerate(test):
            prev = (
                test[i - 1]
                if i > 0 and test[i - 1].get("session_id") == msg.get("session_id")
                else None
            )
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            if enable and analyzer is not None:
                # Force analyzer mode by temporarily swapping
                old = ev.evaluator.linguistic_analyzer
                ev.evaluator.linguistic_analyzer = analyzer
                try:
                    r = ev.evaluator.evaluate(
                        EvaluationInput(
                            user_profile=profile, current_message=cur, system_config=config
                        )
                    )
                finally:
                    ev.evaluator.linguistic_analyzer = old
            else:
                r = ev.evaluator.evaluate(
                    EvaluationInput(
                        user_profile=profile, current_message=cur, system_config=config
                    )
                )
            y.append(int(bool(msg.get("should_flag") or msg.get("is_anomaly"))))
            scores.append(float(r.anomaly_score))

    yt, ys = np.asarray(y), np.asarray(scores)
    return {
        "label": label,
        "ling_mode": ling_mode,
        "n": int(len(yt)),
        "n_pos": int(yt.sum()),
        "auc_pr": float(average_precision_score(yt, ys)),
        "auc_roc": float(roc_auc_score(yt, ys)),
    }


def main() -> None:
    rows = [
        score_config("linguistic_off", "off"),
        score_config("legacy_4feature", "legacy4"),
        score_config("stylometric_10feature", "stylometric"),
    ]
    best = max(rows, key=lambda r: r["auc_pr"])
    # Canonical rule: enable linguistic only if it beats off by AUC-PR
    off_ap = next(r["auc_pr"] for r in rows if r["ling_mode"] == "off")
    enable = best["ling_mode"] != "off" and best["auc_pr"] > off_ap + 0.01
    report = {
        "seed": SEED,
        "configs": rows,
        "best_by_auc_pr": best,
        "recommended_linguistic_component_enabled": enable,
        "recommended_mode": best["ling_mode"] if enable else "off",
        "decision_rule": "enable linguistic iff best mode beats linguistic-off AUC-PR by >0.01",
    }
    out = ROOT / "results" / "stylometric_linguistic_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
