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
from .consensus import merge as merge_consensus
from .consensus import statistics as build_statistics
from .consensus import groups_of as build_groups

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
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    # corrections that keep >=60% character identity are meaning-preserving;
    # a verified AI verdict above that is trusted.
    return ratio >= 0.60


def check_master(text: str,
                 use_ai: bool = True,
                 raw_call: Optional[Callable[[str], str]] = None,
                 include_v4_hints: bool = False,
                 v4_result: Optional[Dict] = None,
                 min_auto_apply: float = _AUTO_APPLY_MIN,
                 auto_correct_only: bool = True,
                 report_local_only: bool = True,
                 max_passes: int = 1) -> Dict:
    """Run the single master pipeline.

    ``report_local_only`` (Phase 5 discovery mode): when a rule candidate with
    confidence at or above the offline floor (0.75) is not covered by the AI
    verdict it is ADDED to the result as a ``LOCAL_ONLY`` suggestion and — when
    above the auto-apply gate (default 0.79) — applied to the corrected text.
    This keeps the high-precision rule layer active even when Gemini's short
    verdict misses a low-frequency error (e.g. ``feeded -> fed``,
    ``slepping -> sleeping``). Legacy behaviour (default False) drops every
    uncovered local candidate below 0.95.

    ``max_passes`` (second full check, §15/§16): when > 1 in AI mode, the
    corrected text is re-checked up to ``max_passes`` (spec cap 3) to catch
    errors the first pass missed. Loop prevention uses normalized text hashes.

    Returns (backwards compatible with the previous ``check_ai_text`` shape,
    plus ``schema``, ``quality``, ``consensus``, ``statistics`` fields):
        {success, original_text, corrected_text, errors, grammar_status,
         meta, processing_time_ms, quality, consensus, statistics}
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
    candidates = aggregate(rules, hints, text)
    candidates = relocate_candidates(candidates, text)

    errors_map = {}   # key: (start,end) -> error
    error_list: List[Dict] = []
    ai_used = False
    verification = {"decision": "offline", "confidence": 1.0, "reason": "AI unavailable"}
    offline = True

    ai_result = None
    if use_ai:
        ai_result = ai_analyzer.analyze(text, candidates, raw_call=raw_call)
        if ai_result is not None and ai_result.get("errors") is not None:
            ai_used = True
            offline = False

    if ai_used:
        # 7-9 consensus engine (pipeline.consensus, Case A-D):
        #   local candidates + Gemini complete-text verdict -> one change set.
        ai_errs = _blend_rules_into_ai(ai_result["errors"], rules)
        ai_errs = relocate_candidates(ai_errs, text)
        ai_candidates = [{
            "wrong": e.get("wrong", e.get("original", "")),
            "correct": e.get("correct", e.get("correction", "")),
            "type": e.get("type", e.get("category", "grammar")),
            "start": e.get("start", 0),
            "end": e.get("end", 0),
            "confidence": float(e.get("confidence", 0.8)),
            "_ai_raw": e,
        } for e in ai_errs]

        merged = merge_consensus(candidates, ai_candidates,
                                 report_local_only=report_local_only,
                                 reference="gemini",
                                 agree_sources={"rule"})

        for i, rec in enumerate(merged["records"]):
            sid = _sentence_id_of(sents, rec.get("start", 0))
            canon = from_candidate(rec, sentence_id=sid, index=i)
            canon["_discovery"] = rec.get("_discovery") or consensus_tag(canon, True)
            canon["_group"] = rec.get("_group", "")
            if rec.get("_ai_raw"):
                canon["_ai_raw"] = rec["_ai_raw"]
            errors_map[(rec.get("start", 0), rec.get("end", 0))] = canon

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
        d.pop("_group", None)
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
    conflict_n = sum(1 for e in error_list if e.get("_discovery", "") == "CONFLICT")

    # 15/16 second full check (max 3 passes, loop prevention via text hashes)
    recheck_info = {"passes_used": 1, "loop_detected": False,
                    "extra_errors": [], "residual_filtered": 0,
                    "missed_logged": 0}
    # 18/19 missed-error corpus: log AI_ONLY (Gemini-caught, local-missed)
    # errors from the FIRST pass before any recheck folding.
    missed_logged = 0
    if ai_used and ai_only_n:
        from .recheck import log_missed_from_result
        result_for_logging = {
            "original_text": text,
            "errors": clean,
            "meta": {"ai_used": ai_used,
                     "sentences": [s["text"] for s in sents]},
        }
        missed_logged += log_missed_from_result(result_for_logging, pass_no=1)
    if use_ai and ai_used and max_passes > 1 and corrected_text != text:
        from .recheck import recheck as run_recheck

        def _make_pass(current: str) -> Dict:
            return check_master(current, use_ai=use_ai, raw_call=raw_call,
                                include_v4_hints=include_v4_hints,
                                v4_result=None,
                                min_auto_apply=min_auto_apply,
                                auto_correct_only=auto_correct_only,
                                report_local_only=report_local_only,
                                max_passes=1)

        rc = run_recheck(
            text,
            {"corrected_text": corrected_text, "errors": clean,
             "meta": {"ai_used": ai_used}},
            _make_pass,
            max_passes=max_passes,
            log_missed=True,
            raw_call=raw_call)
        recheck_info = rc
        if rc.get("corrected_text") and _meaning_preserved_ok(
                text, rc["corrected_text"], clean):
            corrected_text = rc["corrected_text"]
        extra = rc.get("extra_errors") or []
        if extra:
            error_list.extend(extra)
            error_list.sort(key=lambda e: (-e["confidence"], e["start"]))
            clean = []
            for e in error_list:
                d = to_legacy(dict(e))
                d.pop("_ai_raw", None)
                d.pop("_group", None)
                tag = d.pop("_discovery", None)
                if tag:
                    d["consensus"] = tag
                clean.append(d)
            # a change set that was rejected after recheck never happens here
            # (recheck only folds in errors that survived their own pass).

    avg_conf = float(sum((e.get("confidence", 0) for e in error_list), 0.0)) / len(error_list) if error_list else 1.0

    # 17 auditable statistics block
    gemini_candidates = len(ai_candidates) if ai_used else 0
    rejected_n = len(merged["rejected"]) if ai_used else 0
    conflict_total = len(merged["conflicts"]) if ai_used else 0
    missed_logged = missed_logged + recheck_info.get("missed_logged", 0)
    stats = build_statistics(
        records=[{"_group": (e.get("_discovery") or "")} for e in error_list],
        local_count=len(candidates),
        gemini_count=gemini_candidates,
        rejected_count=rejected_n,
        conflict_count=conflict_total,
        passes=recheck_info.get("passes_used", 1),
        missed_logged=missed_logged,
        uncertain=int(verification.get("decision") == "uncertain") if ai_used else 0,
    )

    # §8 groups: verified_local / rejected_local / gemini_only
    groups = build_groups([
        {"_group": (e.get("_discovery") or ""),
         "wrong": e.get("wrong") or e.get("original"),
         "correct": e.get("correct") or e.get("correction"),
         "type": e.get("category") or e.get("type") or "grammar",
         "confidence": float(e.get("confidence", 0))}
        for e in error_list])

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
                      "local_only": local_n, "conflicts": conflict_n,
                      "offline": offline},
        "statistics": stats,
        "meta": {
            "pipeline": "master",
            "ai_used": ai_used,
            "offline": offline,
            "verification": verification,
            "candidate_count": len(candidates),
            "rule_count": len(rules),
            "v4_hint_count": len(hints),
            "word_count": len(text.split()),
            "sentences": [s["text"] for s in sents],
            "groups": groups,
            "recheck": {k: v for k, v in recheck_info.items() if k != "extra_errors"},
        },
        "processing_time_ms": int((time.time() - started) * 1000),
    }


def check_text(text: str, use_ai: bool = True) -> Dict:
    """Alias used by the Flask app / serverless entry with defaults."""
    return check_master(text, use_ai=use_ai)


__all__ = ["check_master", "check_text", "detect_rules", "aggregate"]