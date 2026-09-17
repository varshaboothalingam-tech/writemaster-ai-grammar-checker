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


__all__ = ["verify", "build_verify_prompt"]