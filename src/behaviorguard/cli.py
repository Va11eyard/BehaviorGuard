"""
BehaviorGuard Command-Line Interface
======================================
Provides the `behaviorguard` CLI entry-point.

Usage examples
--------------
# Evaluate a single message against a JSON profile
behaviorguard evaluate --profile profile.json --message "Delete all accounts" --sensitivity high

# Build a user profile from a JSONL conversation history file
behaviorguard build-profile --input history.jsonl --user-id user_001 --output profile.json

# Incrementally update a profile with a new message
behaviorguard update-profile --profile profile.json --message "How do I debug Python?" --output profile.json

# Run the content-safety baseline on a message
behaviorguard content-check --message "Execute the backdoor payload"

# Print version
behaviorguard version
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved to {path}")


def _profile_from_file(path: str):
    """Load a UserProfile from a JSON file."""
    from behaviorguard.models import UserProfile
    data = _load_json(path)
    return UserProfile.model_validate(data)


def _profile_to_dict(profile) -> dict:
    return profile.model_dump()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a message against a user profile."""
    from behaviorguard import BehaviorGuardEvaluatorML, ML_AVAILABLE
    from behaviorguard.models import (
        CurrentMessage,
        EvaluationInput,
        LinguisticFeatures,
        RequestedOperation,
        SystemConfig,
        TemporalContext,
    )

    if not ML_AVAILABLE:
        print(
            "ERROR: ML dependencies not installed.\n"
            "Install with: pip install sentence-transformers numpy scikit-learn scipy",
            file=sys.stderr,
        )
        return 1

    profile = _profile_from_file(args.profile)
    text: str = args.message
    now = datetime.now(timezone.utc)
    words = text.split()

    current_message = CurrentMessage(
        text=text,
        timestamp=now.isoformat(),
        session_id=args.session_id or "cli_session",
        message_sequence_in_session=1,
        time_since_last_message_seconds=float(args.gap_seconds),
        requested_operation=RequestedOperation(
            type=args.op_type,
            risk_classification=args.op_risk,
            targets=None,
            requires_auth=args.requires_auth,
        ),
        linguistic_features=LinguisticFeatures(
            message_length_tokens=len(words),
            message_length_chars=len(text),
            lexical_diversity=len(set(w.lower() for w in words)) / max(len(words), 1),
            formality_score=0.50,
            politeness_score=0.60,
            contains_code="def " in text or "```" in text,
            contains_urls="http" in text,
            language="en",
        ),
        temporal_context=TemporalContext(
            hour_of_day_utc=now.hour,
            day_of_week=now.strftime("%A"),
            is_typical_active_time=(9 <= now.hour <= 17),
            time_since_last_session_hours=24.0,
        ),
    )

    config = SystemConfig(
        sensitivity_level=args.sensitivity,
        deployment_context=args.context,
        enable_semantic_scoring=True,
        enable_linguistic_scoring=True,
        enable_temporal_scoring=True,
    )

    evaluator = BehaviorGuardEvaluatorML()
    result = evaluator.evaluate(
        EvaluationInput(
            user_profile=profile,
            current_message=current_message,
            system_config=config,
        )
    )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"\nAnomaly Score : {result.anomaly_score:.3f}")
        print(f"Semantic      : {result.component_scores.semantic:.3f}")
        print(f"Linguistic    : {result.component_scores.linguistic:.3f}")
        print(f"Temporal      : {result.component_scores.temporal:.3f}")
        print(f"Risk Level    : {result.risk_level.value}")
        print(f"Action        : {result.recommended_action.value}")
        print(f"Confidence    : {result.confidence.value}")
        if result.red_flags:
            print(f"Red Flags     : {', '.join(result.red_flags[:3])}")
        print(f"Summary       : {result.rationale.overall_summary[:160]}")

    if args.output:
        _save_json(result.model_dump(), args.output)

    return 0


def cmd_build_profile(args: argparse.Namespace) -> int:
    """Build a user profile from a JSONL conversation history file."""
    from behaviorguard.profile_manager import MessageRecord, ProfileManager

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    messages: List[MessageRecord] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: Skipping malformed JSON at line {lineno}: {exc}", file=sys.stderr)
                continue
            messages.append(
                MessageRecord(
                    text=obj.get("text", obj.get("message", "")),
                    timestamp=obj.get("timestamp", _now_iso()),
                    session_id=obj.get("session_id", "default"),
                    is_anomaly=bool(obj.get("is_anomaly", False)),
                    operation_risk=obj.get("operation_risk"),
                )
            )

    if not messages:
        print("ERROR: No messages found in input file.", file=sys.stderr)
        return 1

    pm = ProfileManager(decay=args.decay)
    profile = pm.build_from_history(
        user_id=args.user_id,
        messages=messages,
        account_age_days=args.account_age,
    )

    out_path = args.output or f"profile_{args.user_id}.json"
    _save_json(_profile_to_dict(profile), out_path)
    print(
        f"Built profile for user '{args.user_id}' "
        f"from {len(messages)} messages ({profile.total_interactions} normal)."
    )
    return 0


def cmd_update_profile(args: argparse.Namespace) -> int:
    """Incrementally update a user profile with a single new message."""
    from behaviorguard.profile_manager import MessageRecord, ProfileManager

    profile = _profile_from_file(args.profile)
    msg = MessageRecord(
        text=args.message,
        timestamp=args.timestamp or _now_iso(),
        session_id=args.session_id or "cli_session",
        is_anomaly=False,
    )

    pm = ProfileManager(decay=args.decay)
    updated_profile = pm.update_profile(profile, msg)

    out_path = args.output or args.profile
    _save_json(_profile_to_dict(updated_profile), out_path)
    print(
        f"Profile for '{updated_profile.user_id}' updated. "
        f"Total interactions: {updated_profile.total_interactions}."
    )
    return 0


def cmd_content_check(args: argparse.Namespace) -> int:
    """Run the content-safety baseline on a message."""
    from behaviorguard.baselines.content_safety_baseline import ContentSafetyBaseline

    checker = ContentSafetyBaseline()
    result = checker.detect(args.message)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        score = result["anomaly_score"]
        triggered = result["triggered_categories"]
        label = "UNSAFE" if result["is_anomaly"] else "SAFE"
        print(f"\nContent Safety Score : {score:.3f}  [{label}]")
        if triggered:
            print(f"Triggered Categories : {', '.join(triggered)}")
        else:
            print("Triggered Categories : none")
        print(f"Note : {result['note']}")
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    """Print version information."""
    from behaviorguard import __version__, ML_AVAILABLE
    print(f"BehaviorGuard {__version__}")
    print(f"ML support: {'available' if ML_AVAILABLE else 'not installed'}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="behaviorguard",
        description="BehaviorGuard — context-aware anomaly detection for conversational AI.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── evaluate ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate", help="Evaluate a message against a user profile.")
    p_eval.add_argument("--profile", required=True, metavar="FILE",
                        help="Path to user profile JSON file.")
    p_eval.add_argument("--message", required=True, metavar="TEXT",
                        help="Message text to evaluate.")
    p_eval.add_argument("--sensitivity", default="medium",
                        choices=["low", "medium", "high", "maximum"],
                        help="Sensitivity level (default: medium).")
    p_eval.add_argument("--context", default="enterprise",
                        choices=["consumer", "enterprise", "financial", "healthcare", "government"],
                        help="Deployment context (default: enterprise).")
    p_eval.add_argument("--op-type", default="read", dest="op_type",
                        choices=["read", "write", "delete", "export", "auth_change",
                                 "permission_change", "financial", "admin", "none"],
                        help="Requested operation type (default: read).")
    p_eval.add_argument("--op-risk", default="low", dest="op_risk",
                        choices=["low", "medium", "high", "critical"],
                        help="Operation risk classification (default: low).")
    p_eval.add_argument("--requires-auth", action="store_true", dest="requires_auth",
                        help="Flag if operation requires authentication.")
    p_eval.add_argument("--gap-seconds", default=30.0, type=float, dest="gap_seconds",
                        help="Seconds since last message (default: 30).")
    p_eval.add_argument("--session-id", default=None, dest="session_id",
                        help="Session identifier.")
    p_eval.add_argument("--output", default=None, metavar="FILE",
                        help="Optional: save full JSON result to file.")
    p_eval.add_argument("--json", action="store_true",
                        help="Output full JSON result to stdout.")

    # ── build-profile ─────────────────────────────────────────────────────────
    p_build = sub.add_parser(
        "build-profile",
        help="Build a user profile from a JSONL conversation history file.",
    )
    p_build.add_argument("--input", required=True, metavar="FILE",
                         help="Path to JSONL file (one message object per line).")
    p_build.add_argument("--user-id", required=True, metavar="ID", dest="user_id",
                         help="User identifier.")
    p_build.add_argument("--output", default=None, metavar="FILE",
                         help="Output profile JSON path (default: profile_<user_id>.json).")
    p_build.add_argument("--account-age", default=0, type=int, dest="account_age",
                         help="Account age in days (default: 0).")
    p_build.add_argument("--decay", default=0.95, type=float,
                         help="EMA decay factor λ ∈ (0,1] (default: 0.95).")

    # ── update-profile ────────────────────────────────────────────────────────
    p_upd = sub.add_parser("update-profile",
                           help="Incrementally update a profile with a single new message.")
    p_upd.add_argument("--profile", required=True, metavar="FILE",
                       help="Path to existing user profile JSON file.")
    p_upd.add_argument("--message", required=True, metavar="TEXT",
                       help="New message text to incorporate.")
    p_upd.add_argument("--output", default=None, metavar="FILE",
                       help="Output path (default: overwrites --profile).")
    p_upd.add_argument("--timestamp", default=None, metavar="ISO8601",
                       help="Message timestamp (default: now).")
    p_upd.add_argument("--session-id", default=None, dest="session_id",
                       help="Session identifier.")
    p_upd.add_argument("--decay", default=0.95, type=float,
                       help="EMA decay factor λ ∈ (0,1] (default: 0.95).")

    # ── content-check ─────────────────────────────────────────────────────────
    p_cc = sub.add_parser("content-check",
                           help="Run the content-safety (Llama-Guard-style) baseline.")
    p_cc.add_argument("--message", required=True, metavar="TEXT",
                      help="Message text to classify.")
    p_cc.add_argument("--json", action="store_true",
                      help="Output full JSON result to stdout.")

    # ── version ──────────────────────────────────────────────────────────────
    sub.add_parser("version", help="Print version information.")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "evaluate": cmd_evaluate,
        "build-profile": cmd_build_profile,
        "update-profile": cmd_update_profile,
        "content-check": cmd_content_check,
        "version": cmd_version,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
