"""
evidence.py — Evidence aggregation and multi-source confidence scoring.

Multiple detection engines may flag the same span. Instead of dropping the
losers (old dedup behavior), we MERGE overlapping candidates into one with a
list of sources and a confidence boosted by the number of independent sources
and agreement of the corrections.
"""

from typing import Dict, List, Optional, Tuple


def overlap(a_start: int, a_end: int, b_start: int, b_end: int,
            loosen: bool = False) -> bool:
    """True when the two spans intersect (or nearly touch when loosen)."""
    if loosen:
        a_end = max(a_start, a_end) + 1
        b_end = max(b_start, b_end) + 1
    return a_start < b_end and b_start < a_end


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


class EvidenceCandidate:
    """A merged candidate carrying one or more sources."""

    def __init__(self, cand):
        self.start = cand.start
        self.end = cand.end
        self.original = cand.original
        self.replacement = cand.replacement
        self.category = cand.category
        self.rule_id = cand.rule_id
        self.message = cand.message
        self.detector = cand.detector
        self.raw_confidence = getattr(cand, "raw_confidence", 0.8)
        self.metadata = dict(getattr(cand, "metadata", {}) or {})
        self.sources: List[str] = [cand.detector]

    def add(self, cand) -> bool:
        """Merge another candidate if it overlaps AND suggests the same fix."""
        if not overlap(self.start, self.end, cand.start, cand.end):
            return False
        if _norm(cand.replacement) and _norm(cand.replacement) != _norm(self.replacement):
            return False
        if cand.detector != self.detector and cand.detector not in self.sources:
            self.sources.append(cand.detector)
        self.raw_confidence = max(self.raw_confidence,
                                  getattr(cand, "raw_confidence", 0.8))
        if getattr(cand, "message", "") and not self.message:
            self.message = cand.message
        return True


class EvidenceStore:
    """Merge candidates into unique, source-tagged errors."""

    def __init__(self, candidates: List, loosen: bool = True):
        self.items: List[EvidenceCandidate] = []
        for c in candidates or []:
            self.add(c, loosen=loosen)

    def add(self, cand, loosen: bool = True) -> None:
        for item in self.items:
            if item.add(cand):
                return
        self.items.append(EvidenceCandidate(cand))

    def merged(self) -> List[EvidenceCandidate]:
        return self.items


# ── Confidence aggregation weights (tuned on validation set, conservative) ──
_CONF_BASE = {"spelling": 0.0, "grammar": 0.0, "punctuation": 0.0, "style": 0.0}
_SOURCE_BONUS = 0.04          # per additional independent source
_MAX_SOURCE_BONUS = 0.10


def evidence_confidence(item: EvidenceCandidate) -> float:
    """Compute final confidence from detector confidence + source consensus."""
    conf = item.raw_confidence

    # Independent-source consensus bonus
    n_sources = len(item.sources)
    if n_sources >= 2:
        conf = min(0.99, conf + min((n_sources - 1) * _SOURCE_BONUS, _MAX_SOURCE_BONUS))

    # Corrections that produce a fully identical string when applied are safer
    return conf


def severity_for_confidence(conf: float) -> str:
    if conf >= 0.90:
        return "error"
    if conf >= 0.80:
        return "warning"
    return "info"


def does_not_introduce_new_error(new_text: str, text_with_error: str) -> bool:
    """Cheap heuristic: the corrected text must not contain known double forms."""
    import re
    if re.search(r"\bmore\s+\w+er\b", new_text, re.I):
        return False
    if re.search(r"\bmost\s+\w+est\b", new_text, re.I):
        return False
    if re.search(r"\b(could|should|would|may|might|must)\s+of\b", new_text, re.I):
        return False
    if re.search(r"\b(has|have|had)\s+went\b", new_text, re.I):
        return False
    return True