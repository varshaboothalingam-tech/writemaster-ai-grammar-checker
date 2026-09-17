"""pipeline.ai_verifier — Gemini second-pass verification.

A single Gemini analysis is not blindly trusted. This module sends the
original text, the proposed correction and the list of changes to Gemini a
second time and asks for a verdict:

    accept      → correction is safe, fixes real errors, meaning preserved
    reject      → correction is wrong / changes meaning / original was fine
    uncertain   → ambiguous; report as a suggestion, never force it

Contract (strict JSON):
    {"decision": "accept|reject|uncertain",
     "confidence": 0.0,
     "reason": "one-line explanation"}
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

from .ai_analyzer import _provider_callable, _extract_json

_VERIFY_SYSTEM = (
    "You verify English grammar corrections. Judge ONLY whether the proposed "
    "correction is safe:\n"
    "1. It must fix a real error (grammar, spelling, punctuation, tense, verb form, articles, "
    "pronouns, prepositions, capitalization) without being merely a style preference.\n"
    "2. It must preserve the writer's meaning.\n"
    "3. If the original sentence was already correct, the correct decision is 'reject'.\n"
    "4. If the correction is plausible for a learner but context is ambiguous, answer "
    "'uncertain' instead of inventing certainty.\n"
    "5. 'She is a boy.' is NOT an error; do not accept a rewrite of it.\n\n"
)


def build_verify_prompt(original: str, corrected: str,
                        changes: Optional[List[Dict]] = None) -> str:
    change_list = []
    for c in changes or []:
        change_list.append({"wrong": c.get("wrong") or c.get("original"),
                            "correct": c.get("correct") or c.get("replacement")})
    body = {"original_text": original, "corrected_text": corrected, "changes": change_list}
    return (
        _VERIFY_SYSTEM
        + json.dumps(body, ensure_ascii=False)
        + '\n\nReturn ONLY strict JSON: {"decision": "accept|reject|uncertain", '
          '"confidence": 0.0, "reason": "short reason"}\n'
    )


_DECISIONS = {"accept", "reject", "uncertain"}


def _build_changes_prompt(original: str, changes: List[Dict]) -> str:
    change_list = []
    for c in changes or []:
        change_list.append({"wrong": c.get("wrong") or c.get("original"),
                            "correct": c.get("correct") or c.get("replacement")})
    return (
        _VERIFY_SYSTEM
        + "Judge EVERY proposed change one by one.\n"
        + json.dumps({"original_text": original, "changes": change_list}, ensure_ascii=False)
        + '\n\nReturn ONLY strict JSON: {"verdicts": ['
        '{"wrong": "", "correct": "", "decision": "accept|reject|uncertain", '
        '"reason": "short reason"}]}\n'
        'One verdict entry per proposed change. "accept" = fixes a real error and preserves '
        'meaning; "reject" = not a real error / changes meaning; "uncertain" = ambiguous.\n'
    )


def verify_changes(original: str,
                   changes: Optional[List[Dict]] = None,
                   raw_call: Optional[Callable[[str], str]] = None) -> Dict:
    """Per-change verification. Returns the §8 groups:

        {verified_local: [accepted change dicts],
         rejected_local: [rejected change dicts],
         uncertain:      [ambiguous change dicts],
         decisions:      {wrong->correct: verdict}}

    Falls back to the whole-set ``verify`` verdict when the model does not
    return per-change entries (so callers always get a usable answer).
    Never raises.
    """
    groups = {"verified_local": [], "rejected_local": [], "uncertain": []}
    decisions: Dict[str, str] = {}
    if not original or not changes:
        return {**groups, "decisions": decisions, "per_change": False}
    call = raw_call if raw_call is not None else _provider_callable()
    if call is None:
        return {**groups, "decisions": decisions, "per_change": False}
    try:
        raw = call(_build_changes_prompt(original, changes))
    except Exception:
        raw = None
    data = _extract_json(raw) if raw else None
    verdicts = (data or {}).get("verdicts") if isinstance(data, dict) else None
    if not isinstance(verdicts, list) or not verdicts:
        # fall back to the whole-set verdict
        whole = verify(original, original, changes, raw_call=raw_call)
        decision = whole.get("decision", "uncertain")
        for c in changes:
            mark = "verified_local" if decision == "accept" else (
                "rejected_local" if decision == "reject" else "uncertain")
            groups[mark].append(dict(c))
            decisions[f"{c.get('wrong')}->{c.get('correct')}"] = decision
        return {**groups, "decisions": decisions, "per_change": False}

    by_key = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        w = str(v.get("wrong") or "").strip()
        c = str(v.get("correct") or "").strip()
        if not w:
            continue
        d = str(v.get("decision") or "uncertain").lower()
        d = d if d in _DECISIONS else "uncertain"
        by_key[(w.lower(), c.lower())] = d
    for c in changes:
        w = str(c.get("wrong") or "").strip()
        cc = str(c.get("correct") or "").strip()
        d = by_key.get((w.lower(), cc.lower()), "uncertain")
        decisions[f"{w}->{cc}"] = d
        mark = "verified_local" if d == "accept" else (
            "rejected_local" if d == "reject" else "uncertain")
        groups[mark].append(dict(c))
    return {**groups, "decisions": decisions, "per_change": True}


__all__ = ["verify", "verify_changes", "build_verify_prompt"]


def verify(original: str, corrected: str,
           changes: Optional[List[Dict]] = None,
           raw_call: Optional[Callable[[str], str]] = None) -> Dict:
    """Second-pass verification. Never raises on transport failures."""
    verdict = {"decision": "uncertain", "confidence": 0.5,
               "reason": "verification unavailable"}
    call = raw_call if raw_call is not None else _provider_callable()
    if call is None or not original or not corrected:
        return verdict
    prompt = build_verify_prompt(original, corrected, changes)
    try:
        raw = call(prompt)
    except Exception:
        return verdict
    data = _extract_json(raw) if raw else None
    if not data or not isinstance(data, dict):
        return verdict
    decision = str(data.get("decision") or "uncertain").lower()
    if decision not in _DECISIONS:
        decision = "uncertain"
    try:
        conf = float(data.get("confidence") or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "decision": decision,
        "confidence": max(0.0, min(0.99, conf)),
        "reason": str(data.get("reason") or ""),
    }