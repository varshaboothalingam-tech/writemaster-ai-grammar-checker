"""pipeline.ai_core — public entry of the SINGLE master pipeline.

Delegates to ``pipeline.master.check_master`` so every caller (app.py,
api/index.py, tests) shares one authoritative flow:

    TEXT
      → rule detector + v4 hints (candidates)
      → Gemini ANALYSIS  (full-context, strict JSON)
      → consensus merge (AGREED / AI_ONLY / LOCAL_ONLY, Phase 5–8)
      → Gemini VERIFICATION (accept / reject / uncertain)
      → confidence gates + meaning guard
      → final errors (standard object + legacy aliases) + corrected_text

Principles:
    * prefer "do not change a sentence" over "make an uncertain correction";
    * never fabricate corrections when Gemini is unavailable (offline =
      conservative rule-only path);
    * errors list and corrected_text are guaranteed consistent (offsets are
      recomputed locally, never trusted from the model).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .master import check_master, detect_rules, aggregate
from .rule_detector import detect_rules as _detect_rules  # re-export safety

# Backwards-compatible aliases for existing callers/tests.
def check_ai_text(text: str,
                  use_ai: bool = True,
                  raw_call: Optional[Callable[[str], str]] = None,
                  include_v4_hints: bool = True,
                  v4_result: Optional[Dict] = None) -> Dict:
    """Full master-pipeline check (public API — shape unchanged)."""
    return check_master(text, use_ai=use_ai, raw_call=raw_call,
                        include_v4_hints=include_v4_hints, v4_result=v4_result)


__all__ = ["check_ai_text", "detect_rules", "aggregate", "check_master"]


__all__ = ["check_ai_text", "detect_rules", "aggregate"]