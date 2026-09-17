"""pipeline.schema — canonical error object + category taxonomy.

Every detector, the aggregator, Gemini analysis/verification, and the final
API response speak ONE schema:

    {
      "id":            str,  "S{sent}:{start}-{end}:{cat}:{n}"
      "sentence_id":   int,
      "start":         int,  // char offset in ORIGINAL text
      "end":           int,
      "original":      str,  // what is wrong (IN the text, exact casing)
      "correction":    str,  // suggested replacement
      "category":      str,  // one of CATEGORIES
      "subcategory":   str,  // granular rule id (e.g. "sva_3sg")
      "severity":      str,  // error | warning | info
      "confidence":    0..1 float,
      "explanation":   str,
      "source":        [str],       // detectors that agree (rule, ai, v4, ...)
      "evidence":      [dict],      // per-source evidence entries
      "alternatives":  [str],       // other acceptable corrections
      "message":       str          // short human string for the UI
    }

Backward compatibility: the legacy field names ``wrong``, ``correct``,
``type`` are kept as ALIASES on every object emitted by
``to_legacy()`` so the existing frontend and tests never change.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Phase 4 category taxonomy (canonical).
CATEGORIES = {
    "spelling", "grammar", "tense", "verb_form", "punctuation",
    "capitalization", "articles", "pronouns", "prepositions",
    "sentence_structure", "style", "semantic",
}

# Aliases accepted from legacy detections -> canonical category.
_CATEGORY_MAP = {
    "spelling": "spelling", "typo": "spelling", "misspelling": "spelling",
    "grammar": "grammar", "subject_verb": "grammar",
    "subject_verb_agreement": "grammar", "sva": "grammar",
    "agreement": "grammar", "plural": "grammar",
    "tense": "tense", "verb_form": "verb_form", "verbform": "verb_form",
    "word_form": "verb_form", "punctuation": "punctuation",
    "capitalization": "capitalization", "capitalisation": "capitalization",
    "article": "articles", "articles": "articles",
    "pronoun": "pronouns", "pronouns": "pronouns", "pronoun_reference": "pronouns",
    "preposition": "prepositions", "prepositions": "prepositions",
    "word_order": "sentence_structure", "sentence_structure": "sentence_structure",
    "structure": "sentence_structure", "missing_word": "sentence_structure",
    "extra_word": "sentence_structure", "redundancy": "sentence_structure",
    "word_usage": "semantic", "word_choice": "semantic", "context": "semantic",
    "confusable": "semantic", "collocation": "semantic",
    "style": "style", "style_suggestion": "style",
    "repeated_word": "style", "passive_voice": "style",
}

# Which categories are automatic fixes vs suggestions only.
# Only pure style suggestions are never auto-applied; "semantic"/"word_choice"
# (homophones, confusables, collocations) are REAL errors and must auto-fix.
AUTO_APPLY_ALWAYS_CATEGORIES = {
    "spelling", "grammar", "tense", "verb_form", "punctuation",
    "capitalization", "articles", "pronouns", "prepositions",
    "semantic",
}
# sentence_structure and style are suggestions (may be high-confidence though)
# e.g. word_order IS a real error, so it keeps auto-apply below; only pure
# style hints (filler words, passive voice) are excluded.
NEVER_AUTO_APPLY_CATEGORIES = {"style"}


def normalize_category(raw: str) -> str:
    """Map any legacy/granular category string to a canonical CATEGORIES value."""
    v = (raw or "").strip().lower().replace("_", " ")
    v = " ".join(v.split())
    if v in CATEGORIES:
        return v
    joined = v.replace(" ", "_")
    if joined in _CATEGORY_MAP:
        return _CATEGORY_MAP[joined]
    if "verb" in v and "form" in v:
        return "verb_form"
    if "subject" in v or "agreement" in v or "plural" in v:
        return "grammar"
    if "capital" in v:
        return "capitalization"
    if "punct" in v:
        return "punctuation"
    if "prep" in v:
        return "prepositions"
    if "article" in v:
        return "articles"
    if "pronoun" in v:
        return "pronouns"
    if "estructur" in v or "order" in v:
        return "sentence_structure"
    if "style" in v or "redund" in v:
        return "style"
    if "choice" in v or "usage" in v or "context" in v or "semant" in v:
        return "semantic"
    if "tense" in v:
        return "tense"
    if "spell" in v:
        return "spelling"
    return "grammar"


def severity_for(confidence: float, category: str = "grammar") -> str:
    """Map confidence to a UI severity bucket."""
    cat = normalize_category(category)
    if cat == "spelling":
        return "error" if confidence >= 0.7 else "warning"
    if confidence >= 0.79:
        return "error"
    if confidence >= 0.6:
        return "warning"
    return "info"


def make_error(
    original: str,
    correction: str,
    category: str,
    start: int,
    end: int,
    sentence_id: int = 0,
    confidence: float = 0.8,
    subcategory: str = "",
    explanation: str = "",
    source: Optional[List[str]] = None,
    evidence: Optional[List[Dict]] = None,
    alternatives: Optional[List[str]] = None,
    message: str = "",
    index: int = 0,
) -> Dict:
    """Build a canonical error object. Alignment-safe: ``start``/``end`` must
    cover ``original`` in the source text (callers relocate first)."""
    cat = normalize_category(category)
    start, end = int(start), int(end)
    conf = max(0.0, min(0.99, float(confidence)))
    src = source or ["rule"]
    ev = evidence or []
    alt = alternatives or []
    if correction and correction.lower() != original.lower() and correction not in alt:
        alt = [correction] + alt
    eid = f"S{sentence_id}:{start}-{end}:{cat}:{index}"
    msg = message or explanation or (
        f"'{original}' -> '{correction}'" if correction else f"'{original}' may need attention")
    return {
        "id": eid,
        "sentence_id": sentence_id,
        "start": start,
        "end": end,
        "original": original,
        "correction": correction,
        "category": cat,
        "subcategory": subcategory or "",
        "severity": severity_for(conf, cat),
        "confidence": conf,
        "explanation": explanation,
        "source": list(dict.fromkeys(src)),
        "evidence": ev,
        "alternatives": list(dict.fromkeys(alt)),
        "message": msg,
    }


def to_legacy(err: Dict) -> Dict:
    """Add the legacy alias fields used by the v1/v4 frontend and tests:
    wrong/correct/type, rule_id, replacement. The canonical fields
    (category lowercase, original, correction) are left untouched."""
    out = dict(err)
    out["wrong"] = out.get("original", "")
    out["correct"] = out.get("correction", "")
    out["type"] = out.get("category", "grammar")
    out["rule_id"] = out.get("subcategory") or out.get("category", "grammar").upper()
    out.setdefault("replacement", out["correct"])
    return out


def from_candidate(c: Dict, sentence_id: int = 0, index: int = 0) -> Dict:
    """Upgrade any detector/legacy candidate dict into the canonical object."""
    original = str(c.get("original", c.get("wrong", ""))).strip()
    correction = str(c.get("correction", c.get("correct", c.get("replacement", "")))).strip()
    start = int(c.get("start", c.get("start_position", 0)))
    end = int(c.get("end", c.get("end_position", start + len(original))))
    category = normalize_category(str(c.get("category", c.get("type", "grammar"))))
    sources = c.get("sources") or ([c.get("source", "rule")] if c.get("source") else ["rule"])
    evidence = c.get("evidence") or []
    alternatives = list(c.get("alternatives") or [])
    explanation = str(c.get("explanation", c.get("message", ""))).strip()
    return make_error(
        original=original,
        correction=correction,
        category=category,
        start=start,
        end=end,
        sentence_id=sentence_id,
        confidence=float(c.get("confidence", 0.8)),
        subcategory=str(c.get("rule_id", c.get("subcategory", ""))),
        explanation=explanation,
        source=sources,
        evidence=evidence,
        alternatives=alternatives,
        message=str(c.get("message", c.get("explanation", ""))).strip(),
        index=index,
    )


def consensus_tag(err: Dict, ai_found: bool) -> str:
    """Label how an error was discovered (Phase 5 taxonomy)."""
    sources = set(err.get("source") or [])
    has_ai = "ai" in sources or ai_found
    has_local = bool(sources - {"ai"})
    if has_ai and has_local:
        return "AGREED"
    if has_ai:
        return "AI_ONLY"
    if has_local:
        return "LOCAL_ONLY"
    return "UNKNOWN"


__all__ = [
    "CATEGORIES", "normalize_category", "severity_for", "make_error",
    "to_legacy", "from_candidate", "consensus_tag",
    "AUTO_APPLY_ALWAYS_CATEGORIES", "NEVER_AUTO_APPLY_CATEGORIES",
]