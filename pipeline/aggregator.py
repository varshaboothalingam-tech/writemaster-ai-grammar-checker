"""pipeline.aggregator — merge candidates from every source.

Sources:
    * rule detector          (dependency-light, always available)
    * v4 spaCy pipeline      (optional; requires spaCy + en_core_web_sm)

Responsibilities:
    1. Normalize all sources into one dict schema.
    2. De-duplicate identical candidates.
    3. Resolve overlapping / conflicting spans (subsumption rule).
    4. Multi-source bonus (detectors that agree raise confidence).
    5. Sort by confidence.
    6. ``apply_corrections`` — build the corrected text from approved candidates.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def _normalize(item: Dict) -> Dict:
    """Bring any candidate/error dict into the canonical schema."""
    return {
        "wrong": str(item.get("wrong", item.get("original", ""))),
        "correct": str(item.get("correct", item.get("replacement", ""))),
        "type": _to_type(str(item.get("type", item.get("category", "grammar")))),
        "start": int(item.get("start", item.get("start_position", 0))),
        "end": int(item.get("end", item.get("end_position", 0))),
        "confidence": float(item.get("confidence", 0.8)),
        "source": item.get("source", item.get("detector", "unknown")),
        "rule_id": str(item.get("rule_id", "")),
        "message": str(item.get("message", item.get("explanation", ""))),
    }


_TYPE_MAP = {
    "spelling": "spelling", "grammar": "grammar", "punctuation": "punctuation",
    "word_usage": "word_choice", "contextual_word_usage": "word_choice",
    "word_choice": "word_choice", "verb_form": "verb_form", "tense": "tense",
    "subject_verb_agreement": "subject_verb", "subject_verb": "subject_verb",
    "article": "article", "pronoun": "pronoun", "preposition": "preposition",
    "sentence_structure": "structure", "structure": "structure",
    "capitalization": "capitalization", "redundancy": "redundancy",
    "repeated_word": "redundancy", "typo": "spelling",
    "plural": "plural", "missing_word": "missing_word",
    "extra_word": "extra_word", "word_order": "word_order",
    "style": "style", "style_suggestion": "style",
}


def _to_type(raw: str) -> str:
    v = (raw or "").lower().strip()
    return _TYPE_MAP.get(v, v or "grammar")


def _overlap(a: Dict, b: Dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def _is_capitalized(s: str) -> bool:
    return bool(s) and s[0].isupper()


def _contained(a: Dict, b: Dict) -> bool:
    """Is candidate a's span fully inside candidate b's span?"""
    return b["start"] <= a["start"] and a["end"] <= b["end"]


def normalize_candidates(items) -> List[Dict]:
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            try:
                it = vars(it)
            except TypeError:
                continue
        if it.get("start", it.get("start_position")) is None:
            continue
        out.append(_normalize(it))
    return _dedup(out)


def _dedup(items: List[Dict]) -> List[Dict]:
    """Drop exact (span, correction) duplicates — but MERGE their sources and
    keep the max confidence so the multi-source bonus can still fire."""
    index: Dict[tuple, int] = {}
    out: List[Dict] = []
    for it in items:
        if not it["correct"]:
            continue
        key = (it["start"], it["end"], it["correct"].lower())
        if key in index:
            existing = out[index[key]]
            source = it.get("source", "unknown")
            if source not in existing.setdefault("sources", []):
                existing["sources"].append(source)
            existing["confidence"] = max(existing["confidence"], it["confidence"])
            continue
        index[key] = len(out)
        if "sources" not in it:
            it["sources"] = [it.get("source", "unknown")]
        out.append(it)
    return out


def _tail(correct: str) -> str:
    return (correct or "").strip().rsplit(maxsplit=1)[-1].lower()


def _remove_doubling(candidates: List[Dict]) -> List[Dict]:
    """Collapse adjacent fixes where one correction ends with the next one's
    correction or target word (e.g. AI "There"->"There were" next to rule
    "was"->"were" would render "There were were many").  Keep the
    smaller-span member of the pair (the precise fix wins over the rewrite).
    """
    out: List[Dict] = list(candidates)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(out):
            if changed:
                break
            for j in range(i + 1, len(out)):
                b = out[j]
                lo, hi = sorted((a, b), key=lambda x: x["start"])
                if _overlap(lo, hi):
                    continue
                if not (lo["end"] <= hi["start"]):
                    continue
                tails = (_tail(lo["correct"]), _tail(hi["correct"]))
                targets = ((hi.get("wrong") or "").strip().lower(),
                           (hi["correct"] or "").strip().lower(),
                           (lo.get("wrong") or "").strip().lower(),
                           (lo["correct"] or "").strip().lower())
                if tails[0] in targets[0:2] or tails[1] in targets[2:4]:
                    wide = a if (a["end"] - a["start"]) >= (b["end"] - b["start"]) else b
                    out.remove(wide)
                    changed = True
                    break
    return out


def resolve_overlaps(candidates: List[Dict]) -> List[Dict]:
    """Drop candidates fully contained in an earlier (wider/higher-conf) span.

    Also resolves same-span conflicts by keeping the highest-confidence one.
    """
    ordered = sorted(candidates, key=lambda c: (c["start"], -(c["end"] - c["start"]),
                                                -c["confidence"]))
    kept: List[Dict] = []
    for c in ordered:
        dropped = False
        for k in kept:
            if not _overlap(c, k):
                continue
            # identical span, different correction → keep higher confidence
            if c["start"] == k["start"] and c["end"] == k["end"]:
                if c["confidence"] <= k["confidence"]:
                    dropped = True
                    break
                continue
            # one span inside the other → the wider correction wins UNLESS the
            # narrower candidate is materially more confident (gap >= 0.05).
            # Without this, a low-confidence wide candidate like "goes to the
            # college" (0.72) silently swallows the precise "goes" -> "go" (0.9).
            if _contained(c, k):
                if c["confidence"] - k["confidence"] >= 0.05:
                    kept.remove(k)
                    continue
                dropped = True
                break
            if _contained(k, c):
                if k["confidence"] <= c["confidence"]:
                    kept.remove(k)
                    continue
                dropped = True
                break
        if not dropped:
            kept.append(c)
    return kept


def multi_source_bonus(candidates: List[Dict]) -> List[Dict]:
    """+0.04 confidence when two independent sources agree on a span."""
    sources: Dict[tuple, set] = {}
    for c in candidates:
        key = (c["start"], c["end"], c["correct"].lower())
        for s in (c.get("sources") or [c.get("source", "unknown")]):
            sources.setdefault(key, set()).add(s)
    for c in candidates:
        key = (c["start"], c["end"], c["correct"].lower())
        vals = sources.get(key, set())
        if len(vals) >= 2:
            c["confidence"] = min(0.99, c["confidence"] + 0.04)
            c["sources"] = sorted(vals)
        else:
            c["sources"] = list(vals) or [c.get("source", "unknown")]
    return candidates


def _relocate_one(text: str, wrong: str) -> Optional[tuple]:
    """Return (start, end) of the first case-insensitive, word-boundary match
    of ``wrong`` in ``text``, or None when not found. Never matches inside a
    word (so a candidate like 'to' cannot land inside 'tomorrow')."""
    needle = (wrong or "").strip()
    if not needle:
        return None
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])",
        re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return None
    return (m.start(), m.end())


def _relocate_nearest(text: str, wrong: str, preferred: int = 0) -> Optional[tuple]:
    """Locate ``wrong`` and pick the occurrence NEAREST the ``preferred``
    offset. Prevents a repeated word (e.g. ``were``) from always snapping to
    the FIRST occurrence while the detector meant a later one (§11)."""
    needle = (wrong or "").strip()
    if not needle:
        return None
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])",
        re.IGNORECASE)
    best = None
    for m in pattern.finditer(text):
        if best is None or abs(m.start() - preferred) < abs(best[0] - preferred):
            best = (m.start(), m.end())
    return best


def relocate_candidates(candidates: List[Dict], text: str) -> List[Dict]:
    """Guarantee every candidate's (start, end) span matches its ``wrong``
    string in the original text. Candidates whose ``wrong`` cannot be located
    are dropped (they would corrupt the corrected_text)."""
    out: List[Dict] = []
    for c in candidates or []:
        wrong = (c.get("wrong") or "").strip()
        if not wrong:
            out.append(c)
            continue
        s = c.get("start")
        e = c.get("end")
        valid_span = isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text)
        if valid_span and text[s:e].strip().lower() == wrong.lower():
            out.append(c)
            continue
        # span missing/stale → relocate to the occurrence NEAREST the declared
        # position (repeated-word safe), else first occurrence.
        located = _relocate_nearest(text, wrong, s if isinstance(s, int) else 0)
        if located is None:
            continue
        c = dict(c)
        c["start"], c["end"] = located
        out.append(c)
    return out


def aggregate(rule_candidates: List[Dict],
              extra_candidates: Optional[List[Dict]] = None) -> List[Dict]:
    """Merge, dedup, resolve overlaps and boost multi-source confidence."""
    combined = list(rule_candidates or [])
    if extra_candidates:
        combined.extend(normalize_candidates(extra_candidates))
    combined = _dedup(normalize_candidates(combined))
    combined = resolve_overlaps(combined)
    combined = _remove_doubling(combined)
    combined = multi_source_bonus(combined)
    combined.sort(key=lambda c: (-c["confidence"], c["start"]))
    return combined


def relocate_candidates_guard(text: str, candidates: List[Dict]) -> List[Dict]:
    """Alias kept for callers that prefer an explicit guard step."""
    return relocate_candidates(candidates, text)


def apply_corrections(text: str, candidates: List[Dict],
                      min_confidence: float = 0.0) -> Dict:
    """Apply the highest-confidence, non-overlapping corrections.

    Returns ``{corrected_text, changes}``. Changes = applied candidate list,
    each annotated with its computed span. Overlapping edits are skipped
    (the wider/first one wins).
    """
    if not candidates:
        return {"corrected_text": text, "changes": []}
    ordered = sorted(
        [c for c in candidates if c["correct"] and c["confidence"] >= min_confidence],
        key=lambda c: (-c["confidence"], c["start"], c["end"] - c["start"]))

    applied: List[Dict] = []
    used: List[tuple] = []
    for c in ordered:
        s, e = c["start"], c["end"]
        wrong = (c.get("wrong") or "").strip()
        # defense: a stray span that does not cover its own replacement target
        # would corrupt the text — relocate it, or refuse to apply.
        if wrong and isinstance(s, int) and isinstance(e, int):
            if 0 <= s < e <= len(text) and text[s:e].strip().lower() == wrong.lower():
                pass
            else:
                located = _relocate_nearest(text, wrong, s)
                if located is None:
                    continue
                s, e = located
                c = dict(c)
                c["start"], c["end"] = s, e
        if e <= s or s < 0 or e > len(text):
            continue
        if any(_overlap(c, {"start": u[0], "end": u[1]}) for u in used):
            continue
        used.append((s, e))
        applied.append(dict(c))

    # apply right-to-left so offsets stay valid
    buf = text
    for c in sorted(applied, key=lambda c: -c["start"]):
        s, e = c["start"], c["end"]
        repl = c["correct"]
        # keep sentence-start capitalization when the original was capitalized
        at_start = (s == 0 or (buf[s - 1] in ".!?" and buf[s - 2:s - 1].isspace()))
        if at_start and _is_capitalized(text[s:e]) and repl[:1].islower():
            repl = repl[:1].upper() + repl[1:]
        buf = buf[:s] + repl + buf[e:]

    return {"corrected_text": buf, "changes": applied}