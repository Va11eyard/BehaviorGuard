"""Compare a sequential-study JSON to the committed primary artifact.

Used by CI after `scripts/sequential_ato_study.py --dataset personachat --out ...`
so the smoke run does not overwrite committed files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "results" / "primary" / "sequential_ato_study.json"
TOL = 1e-3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("smoke_json", type=Path)
    args = parser.parse_args()
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    smoke = json.loads(args.smoke_json.read_text(encoding="utf-8"))
    g = gold["detectors"]["cusum_embed"]
    s = smoke["detectors"]["cusum_embed"]
    gop = next(o for o in g["operating_points"] if o["target_fa_per_1000"] == 1.0)
    sop = next(o for o in s["operating_points"] if o["target_fa_per_1000"] == 1.0)
    auc_ok = abs(g["episode_auc"] - s["episode_auc"]) <= TOL
    det_ok = abs(gop["detection_rate"] - sop["detection_rate"]) <= TOL
    print(
        f"gold auc={g['episode_auc']} det={gop['detection_rate']} | "
        f"smoke auc={s['episode_auc']} det={sop['detection_rate']}"
    )
    if not auc_ok or not det_ok:
        print("SMOKE MISMATCH", file=sys.stderr)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
