"""
new_pipeline.py — v4 unified pipeline (evidence-merged + AI-validated).

Detection (all local, deterministic):
    FastDetector + NLPDetector + DataDrivenDetector + HighConfidenceDetector
Merge -> EvidenceStore (multi-source candidates)
Filter -> ContextEngine -> FalsePositiveFilter -> CorrectionValidator
Rank  -> final confidence (multi-source bonus)
Judge -> AIValidator (optional; strict-JSON conservative; offline = pass-through)

This replaces the ad-hoc endpoint zoo with ONE pipeline used by /api/check.
"""

import json
import os
import re
from typing import Dict, List, Optional

from unified_pipeline import (
    ErrorCandidate, get_pipeline, get_nlp,
    ContextEngine, ConfidenceEngine, FalsePositiveFilter,
    CorrectionValidator,
)
from high_confidence_rules import HighConfidenceDetector
from evidence import EvidenceStore, evidence_confidence, severity_for_confidence
from ai_validator import get_validator, AIValidator

# Display floor. Below this a candidate is not reported.
REPORT_MIN_CONFIDENCE = float(os.environ.get("GC_REPORT_MIN", "0.70"))

_AI_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "ai_decisions.jsonl")
_AI_LOG_ENABLED = os.environ.get("GC_AI_LOG", "").lower() in ("1", "true", "yes")


def _log_ai_decision(record: Dict) -> None:
    """Dev/debug log of every AI verdict (spec §11). Never shipped to the UI."""
    if not _AI_LOG_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(_AI_LOG_PATH), exist_ok=True)
        with open(_AI_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


_DIALECT_PRONS = {"she", "he", "it"}
_AINT_NEGATORS = {"nobody", "nothing", "nowhere", "no", "none"}


def _dialect_filter(items, text: str, doc):
    """v4-scoped protection for dialectal / non-standard variants (AAVE, etc.).

    These are legitimate in their dialect; we refuse to auto-correct them.
    """
    lower = text.lower()
    out = []
    for item in items:
        rule = getattr(item, "rule_id", "") or ""
        orig = (getattr(item, "original", "") or "").lower()

        # "Ain't nobody got time." / "Ain't nothing..." — double-negative dialect
        if rule == "DOUBLE_NEGATIVE" and re.search(r"\b(ain'?t)\b", lower):
            continue

        # spaCy splits "ain't" -> "Ai" + "n't"; suppress bogus "Ai -> is"
        if rule in ("SVA", "AGREEMENT") and orig == "ai":
            continue

        # 3rd-person don't — AAVE/general dialect ("She don't like broccoli.")
        if rule == "DO_SUPPORT" and orig == "don't":
            subj = _subject_of(doc, item.start)
            if subj in _DIALECT_PRONS:
                continue
            # dependency fallback: aux attaches to main verb, so grep the surface
            if re.search(r"(^|[.!?;]\s+)[`'\"]?(she|he|it)\s+don'?t\b", lower, re.I):
                continue

        out.append(item)
    return out


def _subject_of(doc, char_start: int) -> str:
    if doc is None:
        return ""
    for tok in doc:
        if tok.idx <= char_start < tok.idx + len(tok.text):
            for child in tok.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    return child.lower_
            break
    return ""


def _merge_and_filter(text: str, doc, store_owner, threshold: float = 0.70):
    """Shared detect -> merge -> context -> fp_filter -> validate stage."""
    pipe = get_pipeline()

    fast = pipe.fast_detector.detect(text)
    nlp_c = pipe.nlp_detector.detect(text, doc)
    data = pipe.data_detector.detect(text, doc)
    hc = store_owner.detect(text, doc)
    all_c = fast + nlp_c + data + hc

    store = EvidenceStore(all_c)

    # Context suppression
    survivors = []
    for item in store.merged():
        suppressed, reason = pipe.context_engine.should_suppress(item, text, doc)
        if suppressed:
            continue
        conf = evidence_confidence(item)
        if conf < threshold:
            continue
        item.final_confidence = conf
        survivors.append(item)

    # False-positive filter (entity, colloquial, dialect strategies)
    survivors = pipe.fp_filter.filter(survivors, text, doc)
    survivors = _dialect_filter(survivors, text, doc)

    # Final dedup: when conflicting replacements survive at the same span
    # (e.g., fuzzy spelling vs irregular-keep only the higher-confidence one.
    from unified_pipeline import deduplicate
    survivors = deduplicate(survivors)

    return survivors


def check_v4(text: str, use_ai: Optional[bool] = None,
             log_ai: Optional[bool] = None) -> Dict:
    """Full v4 check. Returns dict in the shape used by /api/check.

    Returns:
        {errors: [...], meta: {sources_used, ai_used, ai_available}}
    """
    do_log = _AI_LOG_ENABLED if log_ai is None else bool(log_ai)
    errors: List[Dict] = []
    if not text or not text.strip():
        return {"errors": errors,
                "meta": {"sources_used": [], "ai_used": False, "ai_available": False}}

    pipe = get_pipeline()
    nlp = get_nlp()
    clean_text, _protected = pipe.preprocessor.protect(text)

    doc = nlp(clean_text)
    hc = HighConfidenceDetector()
    survivors = _merge_and_filter(clean_text, doc, hc)

    # Correction validation
    valid_survivors = []
    for item in survivors:
        ok, _reason = pipe.validator.validate(item, clean_text)
        if ok:
            item.final_confidence = max(0.0, min(0.99, item.final_confidence))
            valid_survivors.append(item)

    # AI final validation (only for surviving candidates; one call per candidate)
    ai = get_validator()
    ai_used = False
    if use_ai is not False and ai.is_enabled():
        ai_used = True
        sents = [s for s in doc.sents]
        payload = []
        for item in valid_survivors:
            idx = next((i for i, s in enumerate(sents)
                        if s.start_char <= item.start < s.end_char), None)
            if idx is None:
                sentence, prev_s, next_s = clean_text, "", ""
            else:
                sentence = sents[idx].text
                prev_s = sents[idx - 1].text if idx > 0 else ""
                next_s = sents[idx + 1].text if idx + 1 < len(sents) else ""
            payload.append({
                "original": item.original,
                "original_text": item.original,
                "replacement": item.replacement,
                "sentence": sentence.strip(),
                "previous_sentence": prev_s.strip(),
                "next_sentence": next_s.strip(),
                "category": getattr(item.category, "value", str(item.category)) if hasattr(item.category, "value") else str(item.category),
                "rule_name": item.rule_id,
                "confidence": item.final_confidence,
            })
        verdicts = ai.validate(payload)
        ai_validated = len(valid_survivors)
        ai_approved = 0
        keep = []
        for item, v, cand in zip(valid_survivors, verdicts, payload):
            approved = v.is_error and v.correction_is_valid
            if approved:
                ai_approved += 1
            if do_log:
                _log_ai_decision({
                    "text": clean_text,
                    "sentence": cand.get("sentence", ""),
                    "detector": item.detector,
                    "rule": item.rule_id,
                    "category": getattr(item.category, "value", str(item.category))
                    if hasattr(item.category, "value") else str(item.category),
                    "original": item.original,
                    "replacement": item.replacement,
                    "detector_confidence": round(item.final_confidence, 4),
                    "gemini_is_error": v.is_error,
                    "gemini_correction_valid": v.correction_is_valid,
                    "gemini_confidence": round(v.confidence, 4) if v.confidence is not None else None,
                    "gemini_reason": v.reason,
                    "gemini_replacement": v.replacement,
                    "final_decision": "APPROVED" if approved else "REJECTED",
                })
            if approved:
                item.final_confidence = max(0.5, min(0.99, v.confidence or item.final_confidence))
                item._ai_reason = v.reason
                item._ai_category = v.category
                keep.append(item)
        valid_survivors = keep

    # Sort by confidence and map to output
    valid_survivors.sort(key=lambda i: i.final_confidence, reverse=True)

    for item in valid_survivors:
        if item.final_confidence < REPORT_MIN_CONFIDENCE:
            continue
        category = item.category.value if hasattr(item.category, "value") else str(item.category)
        source_ids = list(dict.fromkeys(item.sources))
        errors.append({
            "id": f"{item.start}:{item.end}:{item.rule_id}",
            "start": item.start,
            "end": item.end,
            "original": item.original,
            "replacement": item.replacement,
            "category": category.upper(),
            "rule_id": item.rule_id,
            "severity": severity_for_confidence(item.final_confidence),
            "confidence": round(item.final_confidence, 2),
            "explanation": getattr(item, "_ai_reason", "") or item.message,
            "message": item.message,
            "sources": source_ids,
            "detector": item.detector,
            "context": clean_text,
        })

    return {
        "errors": errors,
        "meta": {
            "sources_used": ["fast_detector", "nlp_detector",
                             "data_detector", "high_conf_rules"],
            "ai_used": ai_used,
            "ai_available": ai.is_enabled() and ai.available(),
            "ai_validated": ai_validated if ai_used else None,
            "ai_approved": ai_approved if ai_used else None,
            "ai_rejected": (ai_validated - ai_approved) if ai_used else None,
            "word_count": len(clean_text.split()),
        },
    }