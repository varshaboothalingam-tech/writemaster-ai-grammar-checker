"""pipeline.consensus — the consensus engine (Case A–D).

Merges the LOCAL NLP candidates and the GEMINI verdicts into one verified,
deduplicated change set. Every surviving record is tagged with its discovery
group (the pipeline's ``consensus`` field), and a ``statistics`` block is
emitted so callers can audit how the final suggestion list was built.

Cases:

    Case A  AGREED      local NLP and Gemini propose the same fix
                        (confidence = max of the two, both sources kept).
    Case B  LOCAL_ONLY  only local NLP proposed it (Gemini did not support
                        it). By default (AI-authoritative contract) these are
                        NOT auto-applied; they are reported as low-priority
                        suggestions only in discovery mode.
    Case C  AI_ONLY     only Gemini found it (local NLP missed it). These are
                        the "missed errors" that are logged to
                        datasets/missed_errors.jsonl.
    Case D  CONFLICT    local and Gemini proposed different fixes for the same
                        span. The Gemini verdict is the bias-free reference
                        (§6); the higher-confidence / Gemini side wins and the
                        loser is counted as rejected.

The module intentionally has NO knowledge of HTTP, the UI or the transport;
it is a pure function over canonical candidate dicts. ``merge`` never raises.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .schema import normalize_category

AGREED = "AGREED"
LOCAL_ONLY = "LOCAL_ONLY"
AI_ONLY = "AI_ONLY"
CONFLICT = "CONFLICT"

# Groups that reach the final suggestion list.
_REPORTED = {AGREED, AI_ONLY, LOCAL_ONLY, CONFLICT}


def _norm(v: str) -> str:
    return " ".join((v or "").lower().split())


def _same_fix(a: Dict, b: Dict) -> bool:
    return _norm(a.get("correct") or a.get("correction")) == _norm(
        b.get("correct") or b.get("correction"))


def _ai_index(ai_errors: List[Dict]) -> Dict[tuple, Dict]:
    """Index Gemini errors by exact (start, end) — keep the highest-conf one."""
    index: Dict[tuple, Dict] = {}
    for e in ai_errors or []:
        s, en = e.get("start"), e.get("end")
        if not isinstance(s, int) or not isinstance(en, int) or en <= s:
            continue
        key = (s, en)
        if key in index:
            cur = index[key]
            if float(e.get("confidence", 0.5)) > float(cur.get("confidence", 0.5)):
                index[key] = e
        else:
            index[key] = e
    return index


def _record(c: Dict, group: str, ref: str):
    """Return a copy of a candidate dict tagged with its discovery group."""
    out = dict(c)
    sources = list(dict.fromkeys(out.get("sources") or [out.get("source", "rule")]))
    out["sources"] = sources
    out["_group"] = group
    out["_discovery"] = group
    out["_reference"] = ref
    return out


def merge(local_candidates: List[Dict],
          ai_errors: List[Dict],
          report_local_only: bool = True,
          reference: str = "gemini",
          agree_sources: Optional[set] = None) -> Dict:
    """Combine local + Gemini candidates into one consensus record set.

    Arguments:
        local_candidates: candidates from the local NLP layer (rule detector
            + v4 hints), already relocated to the ORIGINAL text.
        ai_errors: Gemini error dicts with ``wrong/correct/type/confidence``
            already located on the ORIGINAL text (offsets never trusted raw
            from the model).
        report_local_only: when False (AI-authoritative default), Case B
            candidates are counted but not returned. When True (discovery
            mode) they are reported as low-priority suggestions.
        reference: which side decides Case D ties ("gemini" default).
        agree_sources: source names that count as "Case A agreement" between
            local NLP and Gemini. ``None`` means any local source counts;
            pass e.g. ``{"rule"}`` to treat only rule-detector candidates as
            agreement (v4 hints then stay AI_ONLY even when they match).

    Returns:
        {
          "records":    final suggestion list (dicts tagged _group/_discovery),
          "rejected":   losing records (conflict losers + dropped Case B),
          "conflicts":  [{span, local, ai, winner}] for traceability,
          "statistics": the audit block (local_count, gemini_count, agreed,
                        ai_only, local_only, conflicts, rejected, ...),
        }
    """
    locals_by_span: Dict[tuple, Dict] = {}
    for c in local_candidates or []:
        s, en = c.get("start"), c.get("end")
        if not isinstance(s, int) or not isinstance(en, int) or en <= s:
            continue
        key = (s, en)
        if key not in locals_by_span:
            locals_by_span[key] = c
        elif float(c.get("confidence", 0)) > float(locals_by_span[key].get("confidence", 0)):
            locals_by_span[key] = c

    ai_index = _ai_index(ai_errors)
    records: List[Dict] = []
    rejected: List[Dict] = []
    conflicts: List[Dict] = []
    seen_spans = set()  # spans already claimed by an AI record

    # Case C first: every Gemini error is authoritative (bias-free reference).
    def _agrees(local: Dict) -> bool:
        if agree_sources is None:
            return True
        for s in (local.get("sources") or [local.get("source", "")]):
            if s in agree_sources:
                return True
        return False

    for key, ai in ai_index.items():
        s, en = key
        local = locals_by_span.get(key)
        if local is not None and _agrees(local) and _same_fix(local, ai):
            # Case A — local NLP independently found the same fix.
            conf = max(float(local.get("confidence", 0)),
                       float(ai.get("confidence", 0.5)))
            rec = _record(ai, AGREED, reference)
            rec["confidence"] = min(0.99, conf)
            rec["sources"] = list(dict.fromkeys(
                ["ai"] + (local.get("sources") or [local.get("source", "rule")])))
            rec["wrong"] = local.get("wrong", ai.get("wrong"))
            records.append(rec)
            seen_spans.add(key)
            continue
        if local is not None and not _same_fix(local, ai):
            # Case D — both sides claim this span but disagree on the fix.
            # Deferred to the Case B/D loop (which has the ai record at hand).
            continue
        # Case C — Gemini alone found it (or a same-fix local that is outside
        # the agreement sources, e.g. a v4 hint): AI_ONLY, Gemini-authoritative.
        rec = _record(ai, AI_ONLY, reference)
        rec["sources"] = ["ai"]
        records.append(rec)
        seen_spans.add(key)

    # Case B + Case D for the local side.
    for key, local in locals_by_span.items():
        if key in seen_spans:
            continue
        # Case D: same span, different fix — Gemini is the reference.
        ai = ai_index.get(key)
        if ai is not None:
            winner, loser = _resolve_conflict(local, ai, reference)
            conflicts.append({
                "span": list(key),
                "local": {"wrong": local.get("wrong"), "correct": local.get("correct"),
                          "confidence": local.get("confidence")},
                "ai": {"wrong": ai.get("wrong"), "correct": ai.get("correct"),
                       "confidence": ai.get("confidence")},
                "winner": winner.get("_group"),
            })
            seen_spans.add(key)
            records.append(winner)
            rejected.append(loser)
            continue
        # Case B — local only.
        if report_local_only:
            records.append(_record(local, LOCAL_ONLY, reference))
        elif float(local.get("confidence", 0)) < 0.95:
            rejected.append(_record(local, LOCAL_ONLY, reference))

    records = _dedup_records(records)
    records.sort(key=lambda r: (-float(r.get("confidence", 0)),
                                int(r.get("start", 0))))

    stats = statistics(records, len(local_candidates), len(ai_index),
                       len(rejected), len(conflicts))
    return {"records": records, "rejected": rejected,
            "conflicts": conflicts, "statistics": stats}


def _resolve_conflict(local: Dict, ai: Dict, reference: str) -> tuple:
    """Case D: the Gemini verdict is the bias-free reference (§6), so it wins
    the same-span conflict outright (unless a non-gemini reference was chosen).
    The loser is counted in ``rejected``.
    """
    if reference != "gemini":
        lc = float(local.get("confidence", 0))
        ac = float(ai.get("confidence", 0.5))
        if lc > ac or not (ac >= lc or ac >= 0.6):
            rec = _record(local, CONFLICT, "local")
            return rec, _record(ai, CONFLICT, "gemini")
    rec = _record(ai, CONFLICT, "gemini")
    rec["sources"] = ["ai"]
    return rec, _record(local, CONFLICT, "local")


def _dedup_records(records: List[Dict]) -> List[Dict]:
    """Drop duplicate (span, correction) records — keep the richer one."""
    out: List[Dict] = []
    seen: set = set()
    for r in records:
        key = (int(r.get("start", 0)), int(r.get("end", 0)),
               _norm(r.get("correct") or r.get("correction")))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def statistics(records: List[Dict],
               local_count: int,
               gemini_count: int,
               rejected_count: int = 0,
               conflict_count: int = 0,
               passes: int = 1,
               missed_logged: int = 0,
               uncertain: int = 0) -> Dict:
    """: Build the auditable statistics block (§17)."""
    counts = {g: 0 for g in _REPORTED}
    for r in records:
        counts[r.get("_group", "UNKNOWN")] = counts.get(r.get("_group", "UNKNOWN"), 0) + 1
    return {
        "local_candidates": local_count,
        "gemini_candidates": gemini_count,
        "agreed": counts[AGREED],
        "ai_only": counts[AI_ONLY],
        "local_only": counts[LOCAL_ONLY],
        "conflicts": conflict_count,
        "rejected": rejected_count,
        "uncertain": uncertain,
        "final_errors": len(records),
        "passes": passes,
        "missed_logged": missed_logged,
    }


def groups_of(records: List[Dict]) -> Dict:
    """Split records into the §8 response groups.

    Returns {verified_local, rejected_local, gemini_only} — each a list of
    {wrong, correct, category, confidence, group}.
    """
    verified_local, rejected_local, gemini_only = [], [], []
    for r in records:
        item = {
            "wrong": r.get("wrong") or r.get("original"),
            "correct": r.get("correct") or r.get("correction"),
            "category": normalize_category(r.get("type") or r.get("category") or "grammar"),
            "confidence": float(r.get("confidence", 0)),
            "group": r.get("_group", ""),
        }
        g = r.get("_group", "")
        if g in (AGREED, AI_ONLY, LOCAL_ONLY):
            gemini_only.append(item) if g == AI_ONLY else verified_local.append(item)
        else:
            rejected_local.append(item)
    return {"verified_local": verified_local,
            "rejected_local": rejected_local,
            "gemini_only": gemini_only}


__all__ = ["merge", "statistics", "groups_of",
           "AGREED", "LOCAL_ONLY", "AI_ONLY", "CONFLICT"]