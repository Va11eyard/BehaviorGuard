#!/usr/bin/env python3
"""
1-week diagnostic gate: Mahalanobis semantic + s_ling fix, proper holdout eval, AE artifact check.

Usage:
    set HF_HUB_OFFLINE=1; set TRANSFORMERS_OFFLINE=1
    python scripts/diagnostic_gate_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import evaluation as ev  # noqa: E402
from behaviorguard import ProfileManager, MessageRecord  # noqa: E402
from behaviorguard.baselines.autoencoder_baseline import AutoencoderBaseline  # noqa: E402
from behaviorguard.models import SystemConfig  # noqa: E402
import scripts.corrected_proper_generalization_eval as pg  # noqa: E402

CORRECTED_PC = ROOT / "datasets/personachat_processed_corrected.json"
INDEPENDENT_MSGS = ROOT / "data/independent_anomaly_messages.json"
OUT = ROOT / "results/diagnostic_gate_eval.json"
BASELINE_PROPER_PATH = ROOT / "results/corrected_proper_generalization_eval.json"

DIAG_BG_CONFIG = SystemConfig(
    sensitivity_level="medium",
    deployment_context="enterprise",
    overrides_enabled=False,
    semantic_scoring_mode="mahalanobis",
    mahalanobis_shrinkage=0.1,
)

BASELINE_F1_REFERENCE = {
    "personachat": {"bg": 0.3487, "ae": 0.7644, "if": 0.3701},
    "blended_skill_talk": {"bg": 0.4688, "ae": 0.4874, "if": 0.1772},
    "anthropic_hh": {"bg": 0.3495, "ae": 0.1750, "if": 0.1027},
}


def load_baseline_proper() -> dict[str, Any]:
    if not BASELINE_PROPER_PATH.exists():
        return {}
    data = _load(BASELINE_PROPER_PATH)
    out: dict[str, Any] = {}
    for dk, ds in data.get("datasets", {}).items():
        proper = ds.get("proper_generalization", {})
        if "behaviorguard_test_aggregate" in proper:
            agg = proper
        else:
            agg = {"result": proper.get("result", {})}
        out[dk] = {
            "behaviorguard": agg.get("behaviorguard_test_aggregate")
            or agg.get("result", {}).get("behaviorguard"),
            "isolation_forest": agg.get("isolation_forest_test_aggregate")
            or agg.get("result", {}).get("isolation_forest"),
            "autoencoder": agg.get("autoencoder_test_aggregate")
            or agg.get("result", {}).get("autoencoder"),
        }
    return out


EXIT_PC_F1_TARGET = 0.51
EXIT_GAP_CLOSE = 0.40


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_covariance(test_data: dict, n_users: int = 5) -> dict[str, Any]:
    """Confirm ProfileManager computes and stores embedding covariance."""
    pm = ProfileManager(decay=0.5)
    by_user = pg._messages_by_user(test_data)
    users = pg._user_lookup(test_data)
    anomaly_users = [
        uid for uid, msgs in by_user.items() if any(m.get("should_flag") for m in msgs)
    ][:n_users]
    rows = []
    for uid in anomaly_users:
        msgs = by_user[uid]
        split = int(len(msgs) * 0.8)
        records = [
            MessageRecord(
                text=m["message_text"],
                timestamp=m["timestamp"],
                session_id=m.get("session_id", "s0"),
                is_anomaly=m.get("is_anomaly", False),
            )
            for m in msgs[:split]
            if not m.get("is_anomaly")
        ]
        profile = pm.build_from_history(uid, records)
        sp = profile.semantic_profile
        has_cov = sp.embedding_covariance is not None and len(sp.embedding_covariance) > 0
        d = len(sp.embedding_mean or []) if sp.embedding_mean else 0
        cov_ok = has_cov and len(sp.embedding_covariance) == d * d
        rows.append(
            {
                "user_id": uid,
                "n_train_organic": len(records),
                "embedding_sample_count": sp.embedding_sample_count,
                "has_mean": sp.embedding_mean is not None,
                "has_covariance": has_cov,
                "covariance_shape_ok": cov_ok,
                "covariance_dim": d,
            }
        )
    return {
        "algorithm1_covariance_implemented": all(r["covariance_shape_ok"] for r in rows),
        "note": "Paper references update_second_moment(Σ_u); now computed at build_from_history",
        "sample_users": rows,
    }


def ling_subfeature_audit(test_data: dict) -> dict[str, Any]:
    """Sub-feature stats organic vs anomalous on PersonaChat tune split."""
    anomaly_ids, benign_ids = pg.build_eval_pool(test_data, use_test_split_only=False)
    all_users = anomaly_ids + benign_ids
    tune, _, _ = pg.split_holdout(all_users)
    tune_set = set(tune)
    by_user = pg._messages_by_user(test_data)
    users = pg._user_lookup(test_data)

    organic_feats: dict[str, list[float]] = defaultdict(list)
    anom_feats: dict[str, list[float]] = defaultdict(list)

    for uid in tune_set:
        msgs = by_user[uid]
        split = int(len(msgs) * 0.8)
        profile = pg._build_profile(users[uid], msgs[:split], 0.5)
        if profile is None:
            continue
        for i, msg in enumerate(msgs[split:]):
            prev = pg._prev_in_session(msgs[split:], i)
            cur = ev.message_to_current_message(msg, prev, user_profile=profile)
            lf = cur.linguistic_features
            bucket = anom_feats if msg.get("should_flag") else organic_feats
            bucket["message_length_tokens"].append(float(lf.message_length_tokens))
            bucket["message_length_chars"].append(float(lf.message_length_chars))
            bucket["lexical_diversity"].append(float(lf.lexical_diversity))
            bucket["formality_score"].append(float(lf.formality_score))
            bucket["politeness_score"].append(float(lf.politeness_score))

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "n": len(vals),
            "mean": round(mean(vals), 4),
            "std": round(pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    features = [
        "message_length_tokens",
        "message_length_chars",
        "lexical_diversity",
        "formality_score",
        "politeness_score",
    ]
    report = {}
    for f in features:
        o, a = organic_feats[f], anom_feats[f]
        o_s, a_s = _stats(o), _stats(a)
        o_range = (o_s["max"] or 0) - (o_s["min"] or 0) if o_s["n"] else 0
        a_range = (a_s["max"] or 0) - (a_s["min"] or 0) if a_s["n"] else 0
        mean_diff = abs((a_s["mean"] or 0) - (o_s["mean"] or 0))
        report[f] = {
            "organic": o_s,
            "anomalous": a_s,
            "mean_abs_diff": round(mean_diff, 4),
            "organic_range": round(o_range, 4),
            "anomalous_range": round(a_range, 4),
            "likely_dead_or_saturated": o_range < 0.05 and a_range < 0.05,
        }
    report["fixes_applied"] = [
        "formality/politeness now computed from text proxies in message_to_current_message",
        "message_length std uses avg_message_length_tokens_std from ProfileManager",
        "logistic mapping steepness adjusted (k=0.8, d0=2.5)",
    ]
    return report


def ae_artifact_check(test_data: dict) -> dict[str, Any]:
    """Compare AE recall on template vs independently-authored anomalies."""
    anomaly_ids, benign_ids = pg.build_eval_pool(test_data, use_test_split_only=False)
    all_users = anomaly_ids + benign_ids
    tune, val, test = pg.split_holdout(all_users)
    train_f = pg._baseline_train_features(test_data, tune)
    val_f, val_y = pg._baseline_eval_rows(test_data, val)

    ae = AutoencoderBaseline(input_dim=train_f.shape[1], random_seed=ev.SEED)
    ae.fit(train_f, verbose=False)
    val_scores = ae.predict(val_f)["anomaly_scores"]
    high = max(float(np.max(val_scores)), 0.99)
    val_t, _ = pg._f1_max_threshold(val_y, val_scores, 0.01, high, 0.01)

    # Template positives from corrected test window
    by_user = pg._messages_by_user(test_data)
    users = pg._user_lookup(test_data)
    template_scores, template_labels = [], []
    indep_scores, indep_labels = [], []

    for uid in test:
        msgs = by_user[uid]
        split = int(len(msgs) * 0.8)
        profile = ev.build_user_profile(users[uid], msgs[:split])
        if profile is None:
            continue
        for msg in msgs[split:]:
            feat = ev.extract_features_for_baselines(msg, profile)
            sc = ae.predict(feat.reshape(1, -1))["anomaly_scores"][0]
            template_scores.append(sc)
            template_labels.append(bool(msg.get("should_flag")))

    indep_data = _load(INDEPENDENT_MSGS)
    ref_uid = test[0]
    ref_msgs = by_user[ref_uid]
    ref_profile = ev.build_user_profile(
        users[ref_uid], ref_msgs[: int(len(ref_msgs) * 0.8)]
    )
    for cat, texts in indep_data["categories"].items():
        for text in texts:
            fake_msg = {"message_text": text, "timestamp": ref_msgs[-1]["timestamp"]}
            feat = ev.extract_features_for_baselines(fake_msg, ref_profile)
            sc = ae.predict(feat.reshape(1, -1))["anomaly_scores"][0]
            indep_scores.append(sc)
            indep_labels.append(True)

    def _recall_at(scores: list[float], labels: list[bool], t: float) -> float:
        y = np.array(labels, dtype=bool)
        s = np.array(scores, dtype=float)
        if y.sum() == 0:
            return 0.0
        return float((s[y] > t).sum() / y.sum())

    tmpl_rec = _recall_at(template_scores, template_labels, val_t)
    pos_tmpl = sum(template_labels)
    pos_indep = len(indep_labels)
    indep_rec = _recall_at(indep_scores, indep_labels, val_t)

    return {
        "val_threshold": round(float(val_t), 4),
        "template_positives_in_test_window": pos_tmpl,
        "template_recall_at_val_threshold": round(tmpl_rec, 4),
        "independent_messages_per_category": {
            k: len(v) for k, v in indep_data["categories"].items()
        },
        "independent_recall_at_val_threshold": round(indep_rec, 4),
        "recall_ratio_indep_over_template": round(
            indep_rec / tmpl_rec if tmpl_rec > 0 else 0.0, 4
        ),
        "recall_holds_steady": indep_rec >= 0.75 * tmpl_rec if tmpl_rec > 0 else False,
    }


def _extract_bg_f1(ds_result: dict) -> float | None:
    proper = ds_result.get("proper_generalization", {})
    if "behaviorguard_test_aggregate" in proper:
        return proper["behaviorguard_test_aggregate"]["f1"]["mean"]
    return proper.get("result", {}).get("behaviorguard", {}).get("f1")


def _write_output(out: dict) -> None:
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT}", flush=True)


def build_output(
    cov_audit: dict,
    ling_audit: dict,
    diagnostic_results: dict,
    ae_check: dict,
    baseline_proper: dict,
) -> dict:
    pc_f1 = _extract_bg_f1(diagnostic_results.get("personachat", {}))
    baseline_pc_f1 = BASELINE_F1_REFERENCE["personachat"]["bg"]
    ae_baseline_pc = BASELINE_F1_REFERENCE["personachat"]["ae"]
    gap_before = ae_baseline_pc - baseline_pc_f1
    gap_after = ae_baseline_pc - (pc_f1 or 0)
    gap_closed = (gap_before - gap_after) / gap_before if gap_before > 0 else 0.0

    exit_criteria = {
        "pc_f1_target": EXIT_PC_F1_TARGET,
        "pc_f1_diagnostic": pc_f1,
        "pc_f1_pass": (pc_f1 or 0) >= EXIT_PC_F1_TARGET,
        "gap_close_fraction_required": EXIT_GAP_CLOSE,
        "gap_close_fraction_achieved": round(gap_closed, 4),
        "gap_close_pass": gap_closed >= EXIT_GAP_CLOSE,
        "ae_recall_holds_steady": ae_check.get("recall_holds_steady", False),
        "proceed_to_full_sprint": (
            (pc_f1 or 0) >= EXIT_PC_F1_TARGET
            and ae_check.get("recall_holds_steady", False)
        ),
    }

    comparison = {}
    for dk, diag in diagnostic_results.items():
        bg_f1 = _extract_bg_f1(diag)
        ref = BASELINE_F1_REFERENCE.get(dk, {})
        bl = baseline_proper.get(dk, {})
        comparison[dk] = {
            "behaviorguard_f1_diagnostic": bg_f1,
            "behaviorguard_f1_baseline_proper": ref.get("bg"),
            "behaviorguard_delta": round((bg_f1 or 0) - ref.get("bg", 0), 4)
            if bg_f1 is not None
            else None,
            "isolation_forest_f1_baseline_proper": ref.get("if"),
            "autoencoder_f1_baseline_proper": ref.get("ae"),
            "baseline_if_ae_from_file": {
                "if": bl.get("isolation_forest"),
                "ae": bl.get("autoencoder"),
            },
        }

    return {
        "task1_covariance_audit": cov_audit,
        "task2_ling_subfeature_audit": ling_audit,
        "task3_proper_holdout_diagnostic": diagnostic_results,
        "task3_vs_baseline_comparison": comparison,
        "task3_baseline_f1_reference": BASELINE_F1_REFERENCE,
        "task4_ae_artifact_check": ae_check,
        "exit_criteria": exit_criteria,
        "diagnostic_config": {
            "semantic_scoring_mode": "mahalanobis",
            "mahalanobis_shrinkage": 0.1,
            "s_ling_fixes": ling_audit.get("fixes_applied"),
        },
    }


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description="Diagnostic gate evaluation")
    parser.add_argument(
        "--datasets",
        default="personachat,blended_skill_talk,anthropic_hh",
        help="Comma-separated dataset keys to evaluate",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip covariance and ling audits (reuse from partial file if present)",
    )
    parser.add_argument(
        "--task4-only",
        action="store_true",
        help="Run only AE artifact check",
    )
    parser.add_argument(
        "--partial",
        type=Path,
        default=OUT,
        help="Partial/final output JSON path",
    )
    args = parser.parse_args()
    OUT = args.partial

    baseline_proper = load_baseline_proper()
    pc_data = _load(CORRECTED_PC)

    prior: dict = {}
    if OUT.exists():
        prior = _load(OUT)

    if args.task4_only:
        print("Task 4: AE independent-source artifact check...", flush=True)
        ae_check = ae_artifact_check(pc_data)
        out = prior or {}
        out["task4_ae_artifact_check"] = ae_check
        if "exit_criteria" in out:
            pc_f1 = out.get("exit_criteria", {}).get("pc_f1_diagnostic")
            out["exit_criteria"]["ae_recall_holds_steady"] = ae_check["recall_holds_steady"]
            out["exit_criteria"]["proceed_to_full_sprint"] = (
                (pc_f1 or 0) >= EXIT_PC_F1_TARGET and ae_check["recall_holds_steady"]
            )
        _write_output(out)
        return

    if args.skip_audit and prior:
        cov_audit = prior.get("task1_covariance_audit", {})
        ling_audit = prior.get("task2_ling_subfeature_audit", {})
        diagnostic_results = prior.get("task3_proper_holdout_diagnostic", {})
    else:
        print("Task 0: Covariance audit...", flush=True)
        cov_audit = audit_covariance(pc_data)
        print(
            f"  covariance implemented: {cov_audit['algorithm1_covariance_implemented']}",
            flush=True,
        )
        print("Task 2: Linguistic sub-feature audit (PC tune split)...", flush=True)
        ling_audit = ling_subfeature_audit(pc_data)
        diagnostic_results = prior.get("task3_proper_holdout_diagnostic", {})

    selected = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(
        f"Task 3: Proper holdout eval with Mahalanobis + s_ling fix (BG only): {selected}",
        flush=True,
    )
    for dk in selected:
        if dk not in pg.CORRECTED_PATHS:
            print(f"  skip unknown dataset: {dk}", flush=True)
            continue
        print(f"  Dataset: {dk}", flush=True)
        td = _load(pg.CORRECTED_PATHS[dk])
        diagnostic_results[dk] = pg.run_dataset(
            dk, td, system_config=DIAG_BG_CONFIG, skip_baselines=True
        )
        bg_f1 = _extract_bg_f1(diagnostic_results[dk])
        print(f"    BG test F1={bg_f1}", flush=True)
        partial = build_output(
            cov_audit,
            ling_audit,
            diagnostic_results,
            prior.get("task4_ae_artifact_check", {}),
            baseline_proper,
        )
        _write_output(partial)

    print("Task 4: AE independent-source artifact check...", flush=True)
    ae_check = ae_artifact_check(pc_data)

    out = build_output(
        cov_audit, ling_audit, diagnostic_results, ae_check, baseline_proper
    )
    _write_output(out)

    pc_f1 = out["exit_criteria"]["pc_f1_diagnostic"]
    baseline_pc_f1 = BASELINE_F1_REFERENCE["personachat"]["bg"]
    print(f"PC BG F1 diagnostic: {pc_f1:.4f} (baseline proper: {baseline_pc_f1})", flush=True)
    print(
        f"Gap closed: {out['exit_criteria']['gap_close_fraction_achieved']:.1%} "
        f"(need {EXIT_GAP_CLOSE:.0%})",
        flush=True,
    )
    print(f"AE recall holds: {ae_check['recall_holds_steady']}", flush=True)
    print(f"PROCEED: {out['exit_criteria']['proceed_to_full_sprint']}", flush=True)


if __name__ == "__main__":
    main()
