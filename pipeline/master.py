"""pipeline.master — the SINGLE master pipeline (Phase 2).

One authoritative flow for /api/check, /api/check-v4, /api/fix-all and the
Vercel entry. Everything else is a detector, an evidence source, or a UI.

Steps (the 29-step flow, consolidated):

  1.  clean / strip input
  2.  sentence split → sentence_id map
  3.  rule detection        (pipeline.rule_detector)      -> LOCAL candidates
  4.  high-confidence rules (high_confidence_rules)       -> LOCAL candidates
  5.  v4 spaCy hints        (optional, local-only)        -> LOCAL candidates
  6.  aggregate + dedup + overlap resolution + relocate
  7.  AI analysis           (Gemini full-context, strict JSON)
  8.  relocate AI errors (offsets never trusted from model)
  9.  consensus merge: per-error discovery tag
        AGREED (rule+AI) / AI_ONLY (AI found, rules missed)
        LOCAL_ONLY (rules only, AI unavailable/rejected)
  10. confidence gate + severity
  11. secondary AI verification (accept / reject / uncertain)
  12. correction validation (meaning preservation, span safety)
  13. apply corrections (right-to-left, no overlap, cap-preserving)
  14. standard error objects emitted (schema.make_error + legacy aliases)
  15. meta + quality summary

Offline (no Gemini key / AI disabled): step 7-9 degraded to the conservative
rule path (>={offline floor} confidence only), text otherwise untouched.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from . import ai_analyzer, ai_verifier
from .rule_detector import detect_rules
from .aggregator import (
    aggregate, apply_corrections, relocate_candidates, resolve_overlaps,
    multi_source_bonus)
from .schema import (
    make_error, to_legacy, from_candidate, normalize_category,
    consensus_tag, NEVER_AUTO_APPLY_CATEGORIES,
)

# Offline floor — only confident rule corrections applied without AI.
_OFFLINE_MIN = 0.75
# Above this confidence a correction may be auto-applied.
_AUTO_APPLY_MIN = 0.79
# Local candidates that match an AI correction stay; local candidates the AI
# explicitly rejected are dropped at this confidence only if AI produced a
# better/wrong-slot correction for the same span.
_CONFLICT_HANDLING = True

_OFFLINE_SOURCES = {"rule", "high_conf", "v4", "fast", "nlp", "data"}


def _v4_candidates(text: str) -> List[Dict]:
    """Best-effort v4 spaCy candidates hint source. Never raises."""
    try:
        from new_pipeline import check_v4
        res = check_v4(text, use_ai=False)
        return res.get("errors") or []
    except Exception:
        return []


def _sentence_boundaries(text: str) -> List[Dict]:
    """Split into sentences and return [{'id', 'start', 'end', 'text'}]."""
    import re
    out, start = [], 0
    for m in re.finditer(r'[.!?]+[\"\'\u201d\u2019]?(?=\s+|$)', text):
        end = m.end()
        out.append({"id": len(out), "start": start, "end": end,
                    "text": text[start:end]})
        start = end
    if start < len(text):
        out.append({"id": len(out), "start": start, "end": len(text),
                    "text": text[start:]})
    return out


def _sentence_id_of(sents: List[Dict], pos: int) -> int:
    for s in sents:
        if s["start"] <= pos < s["end"]:
            return s["id"]
    return max(0, len(sents) - 1) if sents else 0


def _blend_rules_into_ai(ai_errors: List[Dict], rules: List[Dict]) -> List[Dict]:
    """Raise confidence when rule and AI agree on (span, correction)."""
    rule_index = {
        (r["start"], r["end"], r["correct"].lower()): r["confidence"]
        for r in rules if r.get("correct") and r.get("start") is not None
    }
    for e in ai_errors:
        key = (e.get("start", -1), e.get("end", -1),
               (e.get("correct") or e.get("correction") or "").lower())
        if key in rule_index:
            conf = max(float(e.get("confidence", 0.5)), rule_index[key])
            e["confidence"] = min(0.99, conf)
            src = set(e.get("source") or ["ai"])
            src.add("rule")
            e["source"] = sorted(src)
    return ai_errors


def _local_only_candidates(candidates: List[Dict], ai_spans) -> List[Dict]:
    """Candidates with no covering AI error -> LOCAL_ONLY suggestions, reported
    only above the report floor and never auto-applied."""
    covered = {(s, e) for s, e in ai_spans}
    out = []
    for c in candidates:
        if (c.get("start"), c.get("end")) in covered:
            continue
        c = dict(c)
        c["source"] = sorted(set(c.get("source") or [c.get("source_module", "rule")]))
        c["_discovery"] = "LOCAL_ONLY"
        out.append(c)
    return out


def _meaning_preserved_ok(original: str, corrected: str, errors: List[Dict]) -> bool:
    """Cheap local meaning guard: a correction that deletes most of the text
    or replaces most characters is suspicious. Token-based guards misfire on
    word-splits (everyday -> every day), so use character similarity."""
    if original == corrected:
        return True
    if not corrected.strip():
        return False
    import re as _re
    a = _re.sub(r"\s+", "", original.lower())
    b = _re.sub(r"\s+", "", corrected.lower())
    if not a or not b:
        return False
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, a, b).ratio()
    # corrections that keep >=60% character identity are meaning-preserving;
    # a verified AI verdict above that is trusted.
    return ratio >= 0.60


def check_master(text: str,
                 use_ai: bool = True,
                 raw_call: Optional[Callable[[str], str]] = None,
                 include_v4_hints: bool = True,
                 v4_result: Optional[Dict] = None,
                 min_auto_apply: float = _AUTO_APPLY_MIN,
                 auto_correct_only: bool = True,
                 report_local_only: bool = False) -> Dict:
    """Run the single master pipeline.

    ``report_local_only`` (Phase 5 discovery mode): when a rule candidate with
    high confidence is not covered by the AI verdict it is ADDED to the result
    as a ``LOCAL_ONLY`` suggestion (never auto-applied). Default False keeps
    the AI-authoritative contract (AI errors win outright).

    Returns (backwards compatible with the previous ``check_ai_text`` shape,
    plus ``schema``, ``quality``, ``consensus`` fields):
        {success, original_text, corrected_text, errors, grammar_status,
         meta, processing_time_ms, quality, consensus}
    ``errors`` are standard error objects WITH legacy aliases (wrong/correct/
    type) so existing tests and frontends keep working unchanged.
    """
    started = time.time()
    text = (text or "").strip()
    empty = {
        "success": True, "original_text": "", "corrected_text": "",
        "errors": [], "grammar_status": "correct",
        "meta": {"pipeline": "master", "ai_used": False, "word_count": 0},
        "processing_time_ms": 0,
        "quality": {"confidence": 1.0, "ai_validated": False},
        "consensus": {"agreed": 0, "ai_only": 0, "local_only": 0},
    }
    if not text:
        return empty

    sents = _sentence_boundaries(text)

    # 3-6 local candidate generation + merge + relocation
    rules = detect_rules(text)
    hints: List[Dict] = []
    if include_v4_hints:
        hints = ((v4_result or {}).get("errors") if v4_result is not None
                 else _v4_candidates(text))
    candidates = aggregate(rules, hints)
    candidates = relocate_candidates(candidates, text)

    errors_map = {}   # key: (start,end) -> error
    error_list: List[Dict] = []
    ai_used = False
    verification = {"decision": "uncertain", "confidence": 0.5, "reason": "skipped"}
    offline = True

    ai_result = None
    if use_ai:
        ai_result = ai_analyzer.analyze(text, candidates, raw_call=raw_call)
        if ai_result is not None and ai_result.get("errors") is not None:
            ai_used = True
            offline = False

    if ai_used:
        # 7-9 AI produced a verdict (even "no errors") — it is authoritative.
        ai_errs = _blend_rules_into_ai(ai_result["errors"], rules)
        ai_errs = relocate_candidates(ai_errs, text)
        # (span, correction) -> canonical object, global index
        for idx, e in enumerate(ai_errs):
            key = (e.get("start"), e.get("end"))
            if key in errors_map:
                # overlapping AI errors: keep higher confidence
                existing = errors_map[key]
                if float(e.get("confidence", 0.5)) > existing["confidence"]:
                    existing["confidence"] = float(e.get("confidence", 0.5))
                    existing["explanation"] = e.get("explanation", existing["explanation"])
                continue
            sources = set(e.get("source") or ["ai"])
            if not sources:
                sources = {"ai"}
            src = list(dict.fromkeys(
                list(sources if "rule" in sources else ["ai"] + list(sources))))
            canon = make_error(
                original=e.get("wrong", e.get("original", "")),
                correction=e.get("correct", e.get("correction", "")),
                category=e.get("type", e.get("category", "grammar")),
                start=e.get("start", 0),
                end=e.get("end", 0),
                sentence_id=_sentence_id_of(sents, e.get("start", 0)),
                confidence=float(e.get("confidence", 0.8)),
                subcategory=e.get("rule_id", e.get("type", "")),
                explanation=e.get("explanation", ""),
                source=src,
                evidence=[e] if isinstance(e, dict) else [],
                message=e.get("message", e.get("explanation", "")),
                index=idx,
            )
            canon["_discovery"] = consensus_tag(canon, True)
            canon["_ai_raw"] = e
            errors_map[key] = canon

        # LOCAL_ONLY: rule candidates the AI did not touch become low-priority
        # suggestions (improves recall without forcing wrong auto-fixes).
        # Opt-in (Phase 5 discovery mode) to keep the AI-authoritative contract
        # by default.
        ai_spans = list(errors_map.keys())
        for e in (_local_only_candidates(candidates, ai_spans) if report_local_only else []):
            conf = float(e.get("confidence", 0.6))
            if conf < 0.6:
                continue
            key = (e.get("start"), e.get("end"))
            if key in errors_map:
                continue
            canon = from_candidate(e, sentence_id=_sentence_id_of(sents, e.get("start", 0)),
                                   index=len(error_list))
            canon["_discovery"] = "LOCAL_ONLY"
            canon["source"] = list(dict.fromkeys(canon.get("source") or ["rule"]))
            errors_map[key] = canon

        error_list = sorted(errors_map.values(),
                            key=lambda e: (-e["confidence"], e["start"]))

        # 11 secondary verification on the aggregate change set
        legacy_for_apply = [{
            "wrong": e["original"], "correct": e["correction"],
            "start": e["start"], "end": e["end"],
            "confidence": float(e.get("confidence", 0)),
        } for e in error_list]
        corrected_pre = apply_corrections(text, legacy_for_apply, min_confidence=0.0)
        if corrected_pre.get("changes"):
            verification = ai_verifier.verify(
                text, corrected_pre.get("corrected_text", text),
                corrected_pre.get("changes") or [], raw_call=raw_call)
            decision = verification.get("decision", "uncertain")
            if decision == "reject":
                # the verifier says the whole change set is wrong — drop all
                # (conservative "do not fabricate a correction" rule).
                error_list = []
            # accept → keep; uncertain → cap confidence at 0.6
            if decision == "uncertain":
                for e in error_list:
                    e["confidence"] = min(0.6, float(e["confidence"]))
    else:
        # 13 offline / AI unavailable → conservative rule path
        if ai_analyzer.ai_key_configured() and use_ai:
            # AI configured but analysis failed or returned None → offline floor
            pass
        error_list = [from_candidate(c) for c in candidates
                      if float(c.get("confidence", 0)) >= _OFFLINE_MIN]
        for e in error_list:
            e["_discovery"] = "LOCAL_ONLY"

    # 14 sort + strip internal keys (keep _discovery as the consensus tag)
    error_list.sort(key=lambda e: (-e["confidence"], e["start"]))
    clean = []
    for e in error_list:
        d = to_legacy(dict(e))
        d.pop("_ai_raw", None)
        tag = d.pop("_discovery", None)
        if tag:
            d["consensus"] = tag
        clean.append(d)

    # 12/13 apply corrections (auto-apply gate + meaning guard)
    corrected_text = text
    if clean:
        gate = min_auto_apply
        auto = []
        for e in error_list:
            cat = e.get("category", "grammar")
            if cat in NEVER_AUTO_APPLY_CATEGORIES:
                continue
            if float(e.get("confidence", 0)) >= gate:
                auto.append({
                    "wrong": e["original"], "correct": e["correction"],
                    "start": e["start"], "end": e["end"],
                    "confidence": float(e.get("confidence", 0)),
                })
        app = apply_corrections(text, auto, min_confidence=0.0)
        if _meaning_preserved_ok(text, app["corrected_text"], auto):
            corrected_text = app["corrected_text"]
        else:
            corrected_text = text

    agreed = sum(1 for e in error_list if e.get("_discovery", "") == "AGREED")
    ai_only_n = sum(1 for e in error_list if e.get("_discovery", "") == "AI_ONLY")
    local_n = sum(1 for e in error_list if e.get("_discovery", "") == "LOCAL_ONLY")

    avg_conf = float(sum((e.get("confidence", 0) for e in error_list), 0.0)) / len(error_list) if error_list else 1.0

    return {
        "success": True,
        "original_text": text,
        "corrected_text": corrected_text,
        "errors": clean,
        "grammar_status": "errors_found" if clean else "correct",
        "meaning_preserved": verification.get("decision") != "reject",
        "schema": "v2_error_object",
        "quality": {
            "confidence": round(avg_conf, 2),
            "ai_validated": ai_used and verification.get("decision") in ("accept", "uncertain"),
        },
        "consensus": {"agreed": agreed, "ai_only": ai_only_n,
                      "local_only": local_n, "offline": offline},
        "meta": {
            "pipeline": "master",
            "ai_used": ai_used,
            "offline": offline,
            "verification": verification,
            "candidate_count": len(candidates),
            "rule_count": len(rules),
            "v4_hint_count": len(hints),
            "word_count": len(text.split()),
        },
        "processing_time_ms": int((time.time() - started) * 1000),
    }


def check_text(text: str, use_ai: bool = True) -> Dict:
    """Alias used by the Flask app / serverless entry with defaults."""
    return check_master(text, use_ai=use_ai)


__all__ = ["check_master", "check_text", "detect_rules", "aggregate"]