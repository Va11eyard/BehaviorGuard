#!/usr/bin/env python3
"""
BehaviorGuard evaluation CLI.

Usage:
  python evaluate.py                    # Full evaluation (overrides on)
  python evaluate.py --overrides off   # Override ablation: run both on/off, produce comparison table
  python evaluate.py --datasets personachat  # Single dataset
  python evaluate.py --datasets all    # All datasets (default)
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def main():
    parser = argparse.ArgumentParser(
        description="BehaviorGuard evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--overrides",
        choices=["on", "off", "both"],
        default="on",
        help="Override mode: 'on'=normal, 'off'=override ablation (run both, compare), 'both'=same as off",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset names or 'all' for personachat,blended_skill_talk,anthropic_hh",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap iterations for 95%% CI (0=disabled)",
    )
    parser.add_argument(
        "--output",
        default="full_evaluation_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    if args.overrides == "both":
        args.overrides = "off"  # both = run ablation

    dataset_names = (
        ["personachat", "blended_skill_talk", "anthropic_hh"]
        if args.datasets == "all"
        else [d.strip() for d in args.datasets.split(",")]
    )

    print("=" * 80)
    print("BEHAVIORGUARD EVALUATION")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overrides: {args.overrides}, Datasets: {dataset_names}")
    print(f"Seed: {SEED}")

    # Import evaluation components
    from evaluation import (
        compute_bootstrap_ci,
        datasets,
        evaluate_method,
        run_evaluation,
        results,
        convert_to_json_serializable,
    )

    # Run full evaluation pipeline
    dataset_filter = set(dataset_names) if dataset_names != ["all"] else None
    run_evaluation(dataset_filter=dataset_filter)

    run_datasets = {k: v for k, v in datasets.items() if k in dataset_names}

    # Override ablation: run BehaviorGuard with overrides ON and OFF, produce comparison
    if args.overrides == "off":
        print("\n[OVERRIDE ABLATION] Running BehaviorGuard with overrides ON and OFF...")
        results["override_ablation"] = {"overrides_on": {}, "overrides_off": {}}

        for ds_name in run_datasets:
            m_on, _ = evaluate_method(
                "behaviorguard",
                ds_name,
                datasets[ds_name],
                overrides_enabled=True,
            )
            m_off, _ = evaluate_method(
                "behaviorguard",
                ds_name,
                datasets[ds_name],
                overrides_enabled=False,
            )
            results["override_ablation"]["overrides_on"][ds_name] = m_on
            results["override_ablation"]["overrides_off"][ds_name] = m_off

        # Print override ablation table
        print("\n" + "=" * 80)
        print("OVERRIDE ABLATION TABLE (overrides=on vs overrides=off)")
        print("=" * 80)
        for ds_name in run_datasets:
            on = results["override_ablation"]["overrides_on"][ds_name]
            off = results["override_ablation"]["overrides_off"][ds_name]
            print(f"\n{ds_name}:")
            print(
                f"  {'Metric':<12} {'Overrides ON':<14} {'Overrides OFF':<14}"
            )
            print(
                f"  {'Precision':<12} {on['precision']:<14.4f} {off['precision']:<14.4f}"
            )
            print(
                f"  {'Recall':<12} {on['recall']:<14.4f} {off['recall']:<14.4f}"
            )
            print(f"  {'F1':<12} {on['f1']:<14.4f} {off['f1']:<14.4f}")
            print(f"  {'FPR':<12} {on['fpr']:<14.4f} {off['fpr']:<14.4f}")
            print(f"  {'AUC':<12} {on['roc_auc']:<14.4f} {off['roc_auc']:<14.4f}")

    # Bootstrap confidence intervals (when requested)
    if args.bootstrap > 0:
        print("\n[BOOTSTRAP] Computing 95% CI (stratified by user)...")
        for method in ["behaviorguard"]:
            for ds_name in run_datasets:
                data = results.get("methods", {}).get(method, {}).get(ds_name)
                if not data or "predictions" not in data:
                    continue
                preds = data["predictions"]
                if not preds or "user_id" not in preds[0]:
                    continue
                cis = compute_bootstrap_ci(preds, n_bootstrap=args.bootstrap, seed=SEED)
                m = data["metrics"]
                print(f"\n  {method} on {ds_name}:")
                print(f"    Precision = {m['precision']:.3f} [{cis['precision'][0]:.3f}, {cis['precision'][1]:.3f}]")
                print(f"    Recall    = {m['recall']:.3f} [{cis['recall'][0]:.3f}, {cis['recall'][1]:.3f}]")
                print(f"    F1        = {m['f1']:.3f} [{cis['f1'][0]:.3f}, {cis['f1'][1]:.3f}]")
                print(f"    FPR       = {m['fpr']:.3f} [{cis['fpr'][0]:.3f}, {cis['fpr'][1]:.3f}]")
                results.setdefault("bootstrap_ci", {})[f"{method}_{ds_name}"] = {
                    k: list(v) for k, v in cis.items()
                }

    # Save results
    results["metadata"]["evaluate_cli"] = {
        "overrides": args.overrides,
        "datasets": list(run_datasets.keys()),
        "bootstrap": args.bootstrap,
    }
    out = convert_to_json_serializable(results)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
