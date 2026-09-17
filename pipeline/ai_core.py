"""pipeline.ai_core — AI-first orchestration.

    TEXT
      → rule detector (candidates)   [+ optional v4 spaCy candidates as hints]
      → Gemini ANALYSIS  (full-context, strict JSON)
      → Gemini VERIFICATION          (accept / reject / uncertain)
      → confidence filter
      → final errors + corrected_text

Principles:
    * prefer "do not change a sentence" over "make an uncertain correction";
    * never fabricate corrections when Gemini is unavailable (offline =
      conservative rule-only path);
    * errors list and corrected_text are guaranteed consistent (offsets are
      recomputed locally, never trusted from the model).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from .rule_detector import detect_rules
from .aggregator import aggregate, apply_corrections, relocate_candidates
from . import ai_analyzer, ai_verifier

# Offline fallback floor — only confident rule corrections are applied
# without AI confirmation.
_OFFLINE_MIN = 0.75
# Above this confidence a correction may be auto-applied.
_AUTO_APPLY_MIN = 0.79


def _v4_candidates(text: str) -> List[Dict]:
    """Best-effort v4 spaCy candidates. Never raises."""
    try:
        from new_pipeline import check_v4
        res = check_v4(text, use_ai=False)
        return res.get("errors") or []
    except Exception:
        return []


def _blend_with_rules(ai_errors: List[Dict], rules: List[Dict]) -> List[Dict]:
    rule_index = {
        (r["start"], r["end"], r["correct"].lower()): r["confidence"]
        for r in rules if r.get("correct")
    }
    out = []
    for e in ai_errors:
        conf = float(e.get("confidence", 0.5))
        key = (e.get("start", -1), e.get("end", -1), (e.get("correct") or "").lower())
        if key in rule_index:
            conf = max(conf, rule_index[key])
        e["confidence"] = min(0.99, conf)
        out.append(e)
    return out


def check_ai_text(text: str,
                  use_ai: bool = True,
                  raw_call: Optional[Callable[[str], str]] = None,
                  include_v4_hints: bool = True,
                  v4_result: Optional[Dict] = None) -> Dict:
    """Full AI-first check.

    ``v4_result`` lets a caller reuse an already-computed ``check_v4`` result
    as the hint source instead of running spaCy twice.

    Returns:
        {success, original_text, corrected_text, errors, grammar_status,
         meta, processing_time_ms}
    """
    started = time.time()
    text = (text or "").strip()
    if not text:
        return {
            "success": True, "original_text": "", "corrected_text": "",
            "errors": [], "grammar_status": "correct",
            "meta": {"pipeline": "ai", "ai_used": False, "word_count": 0},
            "processing_time_ms": 0,
        }

    rules = detect_rules(text)
    hints: List[Dict] = []
    if include_v4_hints:
        hints = (v4_result or {}).get("errors") if v4_result is not None else _v4_candidates(text)
    candidates = aggregate(rules, hints)
    candidates = relocate_candidates(candidates, text)

    errors: List[Dict] = []
    ai_used = False
    verification = {"decision": "uncertain", "confidence": 0.5, "reason": "skipped"}

    if use_ai:
        analysis = ai_analyzer.analyze(text, candidates, raw_call=raw_call)
        if analysis is not None and analysis.get("errors") is not None:
            # Gemini produced a verdict (even "no errors") — it wins over rules.
            ai_used = True
            merged = _blend_with_rules(analysis["errors"], rules)
            merged = relocate_candidates(merged, text)
            corrected = apply_corrections(text, merged)
            changes = corrected.get("changes") or []
            if changes:
                verification = ai_verifier.verify(text, corrected.get("corrected_text", text),
                                                  changes, raw_call=raw_call)
                decision = verification.get("decision", "uncertain")
                if decision == "accept":
                    errors = merged
                elif decision == "uncertain":
                    # keep as suggestions only, never auto-apply
                    for e in merged:
                        e["confidence"] = min(0.6, e["confidence"])
                    errors = merged
                else:  # reject
                    errors = []
            else:
                errors = merged
    elif ai_analyzer.ai_key_configured():
        # AI configured but explicitly disabled by the caller
        pass

    if not ai_used:
        # offline / AI-unavailable / AI-rejected → conservative rule path
        errors = [e for e in candidates if float(e.get("confidence", 0)) >= _OFFLINE_MIN]

    errors.sort(key=lambda e: (-e["confidence"], e["start"]))
    app_result = apply_corrections(text, errors, min_confidence=_AUTO_APPLY_MIN)
    corrected_text = app_result["corrected_text"]

    for e in errors:
        e.setdefault("message", e.get("explanation", ""))
        e.setdefault("source", "rule")
        e["confidence"] = round(float(e["confidence"]), 2)

    return {
        "success": True,
        "original_text": text,
        "corrected_text": corrected_text,
        "errors": errors,
        "grammar_status": "errors_found" if errors else "correct",
        "meaning_preserved": verification.get("decision") != "reject",
        "meta": {
            "pipeline": "ai",
            "ai_used": ai_used,
            "verification": verification,
            "candidate_count": len(candidates),
            "rule_count": len(rules),
            "v4_hint_count": len(hints),
            "word_count": len(text.split()),
        },
        "processing_time_ms": int((time.time() - started) * 1000),
    }


__all__ = ["check_ai_text", "detect_rules", "aggregate"]