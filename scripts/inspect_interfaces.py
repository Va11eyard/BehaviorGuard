#!/usr/bin/env python3
"""Discover public analyzer interfaces for diagnostic_harness wiring."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TARGETS = [
    ("SemanticAnalyzerML", "turnshift.analyzers.semantic_ml", "SemanticAnalyzerML"),
    ("LinguisticAnalyzerML", "turnshift.analyzers.linguistic_ml", "LinguisticAnalyzerML"),
    ("TemporalAnalyzerML", "turnshift.analyzers.temporal_ml", "TemporalAnalyzerML"),
    ("TurnShiftEvaluatorML", "turnshift.evaluator_ml", "TurnShiftEvaluatorML"),
    ("ProfileManager", "turnshift.profile_manager", "ProfileManager"),
    ("CurrentMessage", "turnshift.models", "CurrentMessage"),
    ("SystemConfig", "turnshift.models", "SystemConfig"),
]


def _fmt_sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(signature unavailable)"


def main() -> None:
    print("TurnShift interface discovery")
    print(f"repo_root={ROOT}")
    print("=" * 72)

    for label, module_path, class_name in TARGETS:
        print(f"\n[{label}] module={module_path}")
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
        except Exception as exc:  # noqa: BLE001 — report import failures verbatim
            print(f"[SKIP] import failed: {type(exc).__name__}: {exc}")
            continue

        print(f"  class: {cls.__module__}.{cls.__qualname__}")
        if inspect.isclass(cls):
            init = getattr(cls, "__init__", None)
            if init:
                print(f"  __init__{_fmt_sig(init)}")
            for name, member in sorted(inspect.getmembers(cls)):
                if name.startswith("_"):
                    continue
                if inspect.isfunction(member) or inspect.ismethoddescriptor(member):
                    print(f"  {name}{_fmt_sig(member)}")
        else:
            print(f"  (not a class — type={type(cls).__name__})")


if __name__ == "__main__":
    main()
