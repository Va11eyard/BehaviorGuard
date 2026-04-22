#!/usr/bin/env python3
"""
Single-command reproduction of BehaviorGuard paper results.

Runs: profile building → scoring → baselines → ablations → statistical tests
Outputs: results/tables.csv, results/figures/ (if generated)
Prints: version hash, dataset checksums, seed
"""

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42


def _codebase_hash() -> str:
    """Compute hash of relevant Python source files for version tracking."""
    patterns = ["*.py"]
    paths = []
    for p in patterns:
        paths.extend(PROJECT_ROOT.rglob(p))
    paths = [p for p in paths if "venv" not in str(p) and "__pycache__" not in str(p)]
    paths.sort()
    h = hashlib.sha256()
    for p in paths:
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _dataset_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    if not filepath.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    print("=" * 80)
    print("BEHAVIORGUARD REPRODUCTION")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {SEED}")
    print(f"Codebase hash: {_codebase_hash()}")

    dataset_files = {
        "personachat": PROJECT_ROOT / "datasets" / "personachat_processed.json",
        "blended_skill_talk": PROJECT_ROOT / "datasets" / "blended_skill_talk_processed.json",
        "anthropic_hh": PROJECT_ROOT / "datasets" / "anthropic_hh_processed.json",
    }
    print("Dataset checksums:")
    for name, p in dataset_files.items():
        print(f"  {name}: {_dataset_checksum(p)}")

    # Ensure results directory exists
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "figures").mkdir(exist_ok=True)

    # Run full evaluation
    from evaluation import run_evaluation, results, convert_to_json_serializable

    run_evaluation()

    # Save JSON
    out_path = results_dir / "full_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")

    # Write tables.csv (main results table)
    table_path = results_dir / "tables.csv"
    rows = []
    for ds_name, data in results.get("methods", {}).get("behaviorguard", {}).items():
        m = data["metrics"]
        rows.append(
            f"behaviorguard,{ds_name},{m['precision']:.4f},{m['recall']:.4f},"
            f"{m['f1']:.4f},{m['fpr']:.4f},{m['roc_auc']:.4f}"
        )
    for method in ["rule_based", "isolation_forest", "autoencoder", "content_safety"]:
        for ds_name, data in results.get("methods", {}).get(method, {}).items():
            m = data["metrics"]
            rows.append(
                f"{method},{ds_name},{m['precision']:.4f},{m['recall']:.4f},"
                f"{m['f1']:.4f},{m['fpr']:.4f},{m['roc_auc']:.4f}"
            )
    with open(table_path, "w") as f:
        f.write("method,dataset,precision,recall,f1,fpr,roc_auc\n")
        f.write("\n".join(rows))
    print(f"[OK] Table saved to {table_path}")

    print("\n" + "=" * 80)
    print("REPRODUCTION COMPLETE")
    print("=" * 80)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
