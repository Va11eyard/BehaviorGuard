#!/usr/bin/env python3
"""
Acquire externally-authored attack prompts and rebuild injected datasets.

Unlike tools/anomaly_templates.py (author-written), this uses published corpora:
  - TrustAIRLab/in-the-wild-jailbreak-prompts  -> prompt_injection
  - LibrAI/do-not-answer                      -> social_engineering
  - AdvBench harmful_behaviors.csv (GitHub)   -> account_takeover

Injection positions and per-user anomaly counts match the corrected protocol;
only the message text source changes. Outputs:
  datasets/{name}_processed_external.json
  data/external_attack_templates.json
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.rebuild_injected_datasets import (  # noqa: E402
    INJECTION_SEED,
    _assign_sessions_and_gaps,
    _build_injected_message,
    _injection_plan,
    _pick_positions,
    _strip_leaky_metadata,
    _timestamp_for_insert,
    _user_rng,
)

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)

DATASET_CONFIG: dict[str, dict[str, str]] = {
    "personachat": {
        "input": "datasets/personachat_processed.json",
        "output": "datasets/personachat_processed_external.json",
        "dataset_source": "personachat",
    },
    "blended_skill_talk": {
        "input": "datasets/blended_skill_talk_processed.json",
        "output": "datasets/blended_skill_talk_processed_external.json",
        "dataset_source": "blended_skill_talk",
    },
    "anthropic_hh": {
        "input": "datasets/anthropic_hh_processed.json",
        "output": "datasets/anthropic_hh_processed_external.json",
        "dataset_source": "anthropic_hh",
    },
}


def _clean(text: str, max_chars: int = 500) -> str:
    t = " ".join(str(text).strip().split())
    if len(t) > max_chars:
        t = t[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return t


def fetch_external_templates(seed: int = 42) -> dict[str, list[str]]:
    """Download published attack prompts and map them to BG anomaly families."""
    from datasets import load_dataset

    rng = random.Random(seed)
    templates: dict[str, list[str]] = {
        "account_takeover": [],
        "social_engineering": [],
        "prompt_injection": [],
    }

    print("Loading TrustAIRLab/in-the-wild-jailbreak-prompts...", flush=True)
    jb = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")
    jb_prompts = [_clean(r["prompt"]) for r in jb["train"] if r.get("prompt")]
    jb_prompts = [p for p in jb_prompts if 20 <= len(p) <= 500]
    rng.shuffle(jb_prompts)
    templates["prompt_injection"] = jb_prompts[:400]
    print(f"  jailbreak prompts kept: {len(templates['prompt_injection'])}")

    print("Loading LibrAI/do-not-answer...", flush=True)
    dna = load_dataset("LibrAI/do-not-answer", split="train")
    dna_prompts = [_clean(r["question"]) for r in dna if r.get("question")]
    dna_prompts = [p for p in dna_prompts if 10 <= len(p) <= 500]
    rng.shuffle(dna_prompts)
    templates["social_engineering"] = dna_prompts[:400]
    print(f"  do-not-answer prompts kept: {len(templates['social_engineering'])}")

    print(f"Downloading AdvBench harmful_behaviors.csv...", flush=True)
    with urllib.request.urlopen(ADVBENCH_URL, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    adv = [_clean(row["goal"]) for row in reader if row.get("goal")]
    adv = [p for p in adv if 10 <= len(p) <= 500]
    rng.shuffle(adv)
    templates["account_takeover"] = adv[:400]
    print(f"  advbench goals kept: {len(templates['account_takeover'])}")

    # Benign-surface ATO positives still need off-topic organic text; reuse a
    # fixed set of published lifestyle questions from Do-Not-Answer's safer
    # categories when available, else fall back to short AdvBench-adjacent
    # off-topic fillers already present in DNA adult/off-topic mix filtered out.
    # For benign_surface we sample from social_engineering pool that does NOT
    # look like ATO (already non-credential text).
    templates["benign_surface"] = list(templates["social_engineering"][:80])

    for k, v in templates.items():
        if k != "benign_surface" and len(v) < 50:
            raise RuntimeError(f"Insufficient external templates for {k}: {len(v)}")
    return templates


def pick_external(
    templates: dict[str, list[str]],
    category: str,
    surface: str,
    rng: random.Random,
) -> str:
    if category == "account_takeover" and surface == "benign_surface":
        pool = templates["benign_surface"] or templates["social_engineering"]
    else:
        pool = templates.get(category) or templates["prompt_injection"]
    return rng.choice(pool)


def rebuild_user_timeline_external(
    user_id: str,
    all_msgs: list[dict],
    dataset_source: str,
    templates: dict[str, list[str]],
    seed: int,
) -> list[dict]:
    organic = sorted(
        [_strip_leaky_metadata(m) for m in all_msgs if not m.get("is_anomaly")],
        key=lambda m: m["timestamp"],
    )
    orig_anomalies = [m for m in all_msgs if m.get("is_anomaly")]
    plan = _injection_plan(orig_anomalies)
    if not plan:
        _assign_sessions_and_gaps(user_id, organic)
        return organic

    rng = _user_rng(user_id, seed)
    positions = _pick_positions(len(organic), len(plan), rng)
    used_texts: set[str] = set()
    timeline = list(organic)
    for inj_num, (pos, (category, surface)) in enumerate(
        sorted(zip(positions, plan), key=lambda x: x[0], reverse=True)
    ):
        for attempt in range(40):
            text = pick_external(templates, category, surface, rng)
            if text not in used_texts or attempt == 39:
                used_texts.add(text)
                break
        ts = _timestamp_for_insert(timeline, pos, rng)
        new_msg = _build_injected_message(
            user_id, text, category, ts, dataset_source, inj_num
        )
        new_msg["injection_source"] = "external_published"
        timeline.insert(pos, new_msg)

    timeline.sort(key=lambda m: m["timestamp"])
    _assign_sessions_and_gaps(user_id, timeline)
    return timeline


def rebuild_dataset_external(
    data: dict,
    dataset_source: str,
    templates: dict[str, list[str]],
    seed: int = INJECTION_SEED,
) -> dict:
    by_user: dict[str, list[dict]] = defaultdict(list)
    for msg in data["messages"]:
        by_user[msg["user_id"]].append(msg)

    new_messages: list[dict] = []
    for user_id in sorted(by_user.keys()):
        new_messages.extend(
            rebuild_user_timeline_external(
                user_id, by_user[user_id], dataset_source, templates, seed
            )
        )

    out = copy.deepcopy(data)
    out["messages"] = new_messages
    meta = dict(out.get("metadata") or {})
    meta["injection_protocol"] = "external_published_v1"
    meta["injection_sources"] = {
        "account_takeover": "AdvBench harmful_behaviors.csv (llm-attacks GitHub)",
        "social_engineering": "LibrAI/do-not-answer",
        "prompt_injection": "TrustAIRLab/in-the-wild-jailbreak-prompts",
        "benign_surface": "sampled from do-not-answer (non-credential)",
    }
    meta["injection_seed"] = seed
    out["metadata"] = meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="personachat,blended_skill_talk,anthropic_hh",
        help="Comma-separated dataset keys",
    )
    parser.add_argument("--seed", type=int, default=INJECTION_SEED)
    args = parser.parse_args()

    templates = fetch_external_templates(seed=args.seed)
    tpl_path = ROOT / "data" / "external_attack_templates.json"
    tpl_path.parent.mkdir(parents=True, exist_ok=True)
    tpl_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "counts": {k: len(v) for k, v in templates.items()},
                "templates": templates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved template pool to {tpl_path}")

    for key in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        cfg = DATASET_CONFIG[key]
        inp = ROOT / cfg["input"]
        outp = ROOT / cfg["output"]
        if not inp.exists():
            print(f"[SKIP] missing {inp}")
            continue
        print(f"\nRebuilding {key} with external positives...", flush=True)
        data = json.loads(inp.read_text(encoding="utf-8"))
        rebuilt = rebuild_dataset_external(data, cfg["dataset_source"], templates, args.seed)
        outp.write_text(json.dumps(rebuilt), encoding="utf-8")
        n_pos = sum(1 for m in rebuilt["messages"] if m.get("is_anomaly"))
        print(f"  wrote {outp.name}: {len(rebuilt['messages'])} msgs, {n_pos} positives")


if __name__ == "__main__":
    main()
