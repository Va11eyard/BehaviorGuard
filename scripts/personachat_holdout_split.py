#!/usr/bin/env python3
"""
PersonaChat holdout partitioning for threshold-sweep / reporting protocols.

Keeps the production 80/20 per-user chronological split for profile building.
Only the held-out 20% message tail is subdivided into validation-half vs
final-test-half (50/50 by stable user hash). Does NOT assign whole users to
train vs val vs test — every scored user retains train messages for profiling.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "datasets" / "personachat_processed_corrected.json"
VALIDATION_HALF_SALT = "personachat_holdout_validation_half"
CV_FOLD_SALT = "personachat_holdout_cv_fold"
TRAIN_FRACTION = 0.8
N_CV_FOLDS = 5
POSITIVE_CV_SEED = 42
CV_VALIDATION_POSITIVE_THRESHOLD = 12


def user_in_validation_half(user_id: str, salt: str = VALIDATION_HALF_SALT) -> bool:
    """Deterministic 50/50 assignment for held-out users (same salt => same half)."""
    digest = hashlib.md5(f"{salt}:{user_id}".encode(), usedforsecurity=False).hexdigest()
    return (int(digest, 16) % 2) == 0


def user_cv_fold_bucket(
    user_id: str,
    n_folds: int = N_CV_FOLDS,
    salt: str = CV_FOLD_SALT,
) -> int:
    """Deterministic fold bucket 0..n_folds-1 for organic-only holdout users."""
    digest = hashlib.md5(f"{salt}:{user_id}".encode(), usedforsecurity=False).hexdigest()
    return int(digest, 16) % n_folds


def _messages_by_user(test_data: dict) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for m in test_data["messages"]:
        by[m["user_id"]].append(m)
    for uid in by:
        by[uid].sort(key=lambda x: x["timestamp"])
    return by


def _is_positive(m: dict) -> bool:
    return bool(m.get("should_flag", False))


@dataclass
class UserHoldoutRecord:
    """Per-user 80/20 split with full holdout tail (for CV or single-split protocols)."""

    user_id: str
    train_msgs: list[dict]
    holdout_msgs: list[dict]
    has_test_positive: bool
    positive_user_fold: int | None = None
    organic_fold_bucket: int | None = None


@dataclass
class UserHoldoutPartition:
    user_id: str
    train_msgs: list[dict]
    validation_msgs: list[dict]
    final_test_msgs: list[dict]
    has_test_positive: bool
    in_validation_half: bool


@dataclass
class HoldoutSplitResult:
    users: list[UserHoldoutPartition]
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class CvFoldAssignment:
    fold_index: int
    test_user_ids: set[str]
    validation_user_ids: set[str]
    test_positives: int
    validation_positives: int
    test_messages: int
    validation_messages: int


@dataclass
class CvFoldPlan:
    positive_user_folds: list[list[str]]
    folds: list[CvFoldAssignment]
    audit: dict[str, Any] = field(default_factory=dict)


def build_holdout_records(
    test_data: dict,
    *,
    train_fraction: float = TRAIN_FRACTION,
) -> list[UserHoldoutRecord]:
    """80/20 per-user records with unified holdout tail (profile train unchanged)."""
    by_user = _messages_by_user(test_data)
    records: list[UserHoldoutRecord] = []
    for uid in sorted(by_user.keys()):
        msgs = by_user[uid]
        if not msgs:
            continue
        split_idx = int(len(msgs) * train_fraction)
        holdout_msgs = msgs[split_idx:]
        if not holdout_msgs:
            continue
        has_test_positive = any(_is_positive(m) for m in holdout_msgs)
        records.append(
            UserHoldoutRecord(
                user_id=uid,
                train_msgs=msgs[:split_idx],
                holdout_msgs=holdout_msgs,
                has_test_positive=has_test_positive,
            )
        )
    return records


def assign_positive_user_folds(
    positive_user_ids: list[str],
    *,
    n_folds: int = N_CV_FOLDS,
    seed: int = POSITIVE_CV_SEED,
) -> list[list[str]]:
    import numpy as np

    ids = list(positive_user_ids)
    if len(ids) != n_folds * (len(ids) // n_folds):
        raise ValueError(f"Expected {n_folds} equal folds; got {len(ids)} users")
    rng = np.random.default_rng(seed)
    arr = np.array(ids)
    rng.shuffle(arr)
    per_fold = len(arr) // n_folds
    return [arr[i * per_fold : (i + 1) * per_fold].tolist() for i in range(n_folds)]


def build_cv_fold_plan(
    records: list[UserHoldoutRecord],
    *,
    n_folds: int = N_CV_FOLDS,
    seed: int = POSITIVE_CV_SEED,
) -> CvFoldPlan:
    """
    5-fold user-level CV on holdout tail.

    - 25 holdout-positive users: shuffled (seed), 5 per fold.
    - Organic-only holdout users: hash-bucketed into the same 5 folds.
    - For fold k: test = fold-k positives + organic bucket k; val = other folds.
    """
    positive_ids = sorted(r.user_id for r in records if r.has_test_positive)
    if len(positive_ids) != 25:
        raise ValueError(f"Expected 25 holdout-positive users, got {len(positive_ids)}")

    positive_folds = assign_positive_user_folds(positive_ids, n_folds=n_folds, seed=seed)
    organic_by_bucket: dict[int, list[str]] = {i: [] for i in range(n_folds)}
    for r in records:
        if r.has_test_positive:
            continue
        bucket = user_cv_fold_bucket(r.user_id, n_folds=n_folds)
        r.organic_fold_bucket = bucket
        organic_by_bucket[bucket].append(r.user_id)

    for fold_idx, fold_users in enumerate(positive_folds):
        for uid in fold_users:
            for r in records:
                if r.user_id == uid:
                    r.positive_user_fold = fold_idx
                    break

    holdout_by_user = {r.user_id: r.holdout_msgs for r in records}
    folds: list[CvFoldAssignment] = []
    for k in range(n_folds):
        test_pos = set(positive_folds[k])
        val_pos = set(uid for i, f in enumerate(positive_folds) if i != k for uid in f)
        test_org = set(organic_by_bucket[k])
        val_org = set(uid for b, uids in organic_by_bucket.items() if b != k for uid in uids)
        test_users = test_pos | test_org
        val_users = val_pos | val_org
        test_msgs = sum(len(holdout_by_user[u]) for u in test_users)
        val_msgs = sum(len(holdout_by_user[u]) for u in val_users)
        test_pos_n = sum(
            1 for u in test_users for m in holdout_by_user[u] if _is_positive(m)
        )
        val_pos_n = sum(
            1 for u in val_users for m in holdout_by_user[u] if _is_positive(m)
        )
        folds.append(
            CvFoldAssignment(
                fold_index=k,
                test_user_ids=test_users,
                validation_user_ids=val_users,
                test_positives=test_pos_n,
                validation_positives=val_pos_n,
                test_messages=test_msgs,
                validation_messages=val_msgs,
            )
        )

    audit = {
        "n_folds": n_folds,
        "positive_cv_seed": seed,
        "n_holdout_positive_users": len(positive_ids),
        "positive_users_per_fold": [len(f) for f in positive_folds],
        "organic_users_per_bucket": [len(organic_by_bucket[i]) for i in range(n_folds)],
        "validation_positive_counts_per_fold": [f.validation_positives for f in folds],
        "test_positive_counts_per_fold": [f.test_positives for f in folds],
        "use_cv_primary": True,
        "single_split_validation_positives": 11,
        "cv_trigger_threshold": CV_VALIDATION_POSITIVE_THRESHOLD,
        "cv_selected_because": (
            "validation_half positives (11) < 12; 5-fold user CV is primary"
        ),
        "positive_user_folds": positive_folds,
    }
    return CvFoldPlan(positive_user_folds=positive_folds, folds=folds, audit=audit)


def partition_holdout(
    test_data: dict,
    *,
    train_fraction: float = TRAIN_FRACTION,
    validation_salt: str = VALIDATION_HALF_SALT,
) -> HoldoutSplitResult:
    """
    Step 1 (corrected): 80/20 profile split unchanged; subdivide only the 20% tail.

    Users with >=1 positive in the held-out tail are split 50/50 by hash across
    validation vs final-test (entire user tail goes to one half). Organic-only
    held-out users use the same hash rule.
    """
    by_user = _messages_by_user(test_data)
    users_out: list[UserHoldoutPartition] = []

    n_holdout_msgs = 0
    n_val_msgs = 0
    n_final_msgs = 0
    n_val_pos = 0
    n_final_pos = 0
    users_with_test_positive = 0
    users_test_positive_val = 0
    users_test_positive_final = 0
    users_organic_holdout = 0
    users_skipped_no_holdout = 0
    users_train_only_positive = 0

    for uid in sorted(by_user.keys()):
        msgs = by_user[uid]
        if not msgs:
            continue
        split_idx = int(len(msgs) * train_fraction)
        train_msgs = msgs[:split_idx]
        holdout_msgs = msgs[split_idx:]
        if not holdout_msgs:
            users_skipped_no_holdout += 1
            continue

        has_any_positive = any(_is_positive(m) for m in msgs)
        has_test_positive = any(_is_positive(m) for m in holdout_msgs)
        has_train_only_positive = any(_is_positive(m) for m in train_msgs) and not has_test_positive

        if has_train_only_positive:
            users_train_only_positive += 1

        to_val = user_in_validation_half(uid, validation_salt)
        if has_test_positive:
            users_with_test_positive += 1
            if to_val:
                users_test_positive_val += 1
            else:
                users_test_positive_final += 1
        else:
            users_organic_holdout += 1

        val_msgs = holdout_msgs if to_val else []
        final_msgs = [] if to_val else holdout_msgs

        n_holdout_msgs += len(holdout_msgs)
        n_val_msgs += len(val_msgs)
        n_final_msgs += len(final_msgs)
        n_val_pos += sum(1 for m in val_msgs if _is_positive(m))
        n_final_pos += sum(1 for m in final_msgs if _is_positive(m))

        users_out.append(
            UserHoldoutPartition(
                user_id=uid,
                train_msgs=train_msgs,
                validation_msgs=val_msgs,
                final_test_msgs=final_msgs,
                has_test_positive=has_test_positive,
                in_validation_half=to_val,
            )
        )

    audit = {
        "train_fraction": train_fraction,
        "validation_half_salt": validation_salt,
        "n_users_total": len(by_user),
        "n_users_with_holdout": len(users_out),
        "n_users_skipped_no_holdout": users_skipped_no_holdout,
        "n_users_any_positive_full_timeline": sum(
            1 for ms in by_user.values() if any(_is_positive(m) for m in ms)
        ),
        "n_users_positive_in_train_only": users_train_only_positive,
        "n_users_positive_in_holdout_tail": users_with_test_positive,
        "n_users_organic_only_in_holdout": users_organic_holdout,
        "n_users_test_positive_assigned_validation": users_test_positive_val,
        "n_users_test_positive_assigned_final_test": users_test_positive_final,
        "holdout_messages_total": n_holdout_msgs,
        "validation_half_messages": n_val_msgs,
        "final_test_half_messages": n_final_msgs,
        "validation_half_positives": n_val_pos,
        "final_test_half_positives": n_final_pos,
        "validation_half_positives_plus_final": n_val_pos + n_final_pos,
    }
    return HoldoutSplitResult(users=users_out, audit=audit)


def load_dataset(path: Path | None = None) -> dict:
    path = path or DEFAULT_DATASET
    return json.loads(path.read_text(encoding="utf-8"))


def print_split_audit(audit: dict[str, Any]) -> None:
    print("=== CORRECTED HOLDOUT PARTITION AUDIT ===")
    for key, value in audit.items():
        print(f"  {key}: {value}")


def print_cv_fold_audit(plan: CvFoldPlan) -> None:
    print("=== 5-FOLD CV FOLD PLAN ===")
    for key, value in plan.audit.items():
        if key == "positive_user_folds":
            continue
        print(f"  {key}: {value}")
    for i, fold in enumerate(plan.folds):
        print(
            f"  fold {i}: test_users={len(fold.test_user_ids)} "
            f"(+{fold.test_positives} pos, {fold.test_messages} msgs) | "
            f"val_users={len(fold.validation_user_ids)} "
            f"(+{fold.validation_positives} pos, {fold.validation_messages} msgs)"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", action="store_true", help="Print 5-fold CV plan audit")
    args = parser.parse_args()
    data = load_dataset()
    if args.cv_folds:
        records = build_holdout_records(data)
        plan = build_cv_fold_plan(records)
        print_cv_fold_audit(plan)
        return
    result = partition_holdout(data)
    print_split_audit(result.audit)


if __name__ == "__main__":
    main()
