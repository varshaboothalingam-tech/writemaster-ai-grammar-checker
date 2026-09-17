"""pipeline.ai_analyzer — Gemini full-context analysis.

The rule/spaCy layer produces candidates; THIS module asks the Gemini
neural network to read the COMPLETE text and independently decide what is
wrong and what the safest correction is.

Transport: ``ai_validator.AIValidator._call(prompt)`` by default, or any
callable supplied as ``raw_call`` (used by tests and mocks).

Contract (strict JSON):
    {
      "original_text": "...",
      "corrected_text": "...",
      "errors": [
        {"wrong", "correct", "type",
         "explanation", "confidence"}
      ],
      "grammar_status": "errors_found | correct",
      "meaning_preserved": true
    }

Offsets are NOT trusted from the model. They are recomputed on our side by
locating ``wrong`` in the original text, so the API contract
(start/end ints) is always reliable.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

SYSTEM_PROMPT = (
    "You are a professional English writing assistant for non-native learners. "
    "Read the COMPLETE text and understand its full context before deciding anything.\n\n"
    "HARD RULES:\n"
    "1. Report ONLY real errors. Never change a correct sentence.\n"
    "2. Separate grammar errors from style preferences. A nicer phrasing is NOT an error.\n"
    "3. 'She is a boy.' is NOT a grammar error; unusual content is not bad grammar.\n"
    "4. 'I went to school yesterday.' / 'She goes to school every day.' are correct.\n"
    "5. Use sentence context to resolve ambiguous words. Example: in the sentence\n"
    "   'She goes to school evry day.' the word 'evry' is a misspelling of 'every',\n"
    "   not a misspelling of 'very'.\n"
    "6. Find ALL errors. Do not stop after the first one. Check every word.\n"
    "7. Repeated letters indicate a typing mistake: 'homeeee' -> 'home',\n"
    "   'schoollll' -> 'school', 'happyyy' -> 'happy'.\n"
    "8. A correction must preserve the writer's meaning and be natural English.\n"
    "9. For each error give one word-safe 'wrong' and one 'correct' string.\n"
    "10. Repeated words are common. ALWAYS include the `context` field: a short\n"
    "    literal snippet of the text IMMEDIATELY around the exact occurrence you mean\n"
    "    (e.g. 'the food were', 'While we were'). If the same word appears twice,\n"
    "    your context decides WHICH occurrence we correct — get it right.\n\n"
)


def build_prompt(text: str, candidates: Optional[List[Dict]] = None) -> str:
    cand_block = "none"
    if candidates:
        cand_block = json.dumps(
            [{"wrong": c.get("wrong"), "correct": c.get("correct"),
              "type": c.get("type"), "confidence": round(float(c.get("confidence", 0.6)), 2)}
             for c in candidates], ensure_ascii=False)
    return (
        SYSTEM_PROMPT
        + "Candidate hints from a local detector (they MAY contain false positives AND may be "
        "located at the WRONG occurrence of a repeated word — verify each against the real "
        "context, keep only the correct ones, and always include the `context` field so the "
        "exact occurrence is unambiguous):\n"
        + cand_block + "\n\n"
        + "Text to analyze:\n"
        + text + "\n\n"
        + 'Return ONLY strict JSON with this exact shape:\n'
        + '{"original_text": "", "corrected_text": "", "errors": ['
        '{"wrong": "", "correct": "", "type": "spelling|grammar|subject_verb|tense|verb_form|'
        'article|pronoun|preposition|plural|word_order|missing_word|extra_word|punctuation|'
        'capitalization|word_choice|redundancy|context", "explanation": "", "confidence": 0.9, '
        '"context": "exact short snippet around the occurrence"}], '
        '"grammar_status": "errors_found|correct", "meaning_preserved": true}\n'
        'If the text is already correct, return errors: [] and grammar_status "correct".'
    )


def _provider_callable() -> Optional[Callable[[str], str]]:
    """Return the default Gemini transport, or None when unavailable."""
    try:
        from ai_validator import get_validator
        v = get_validator()
        if not v.is_enabled():
            return None
        return v._call
    except Exception:
        return None


def _extract_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    # last-resort: slice between first '{' and last '}'
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except ValueError:
            return None
    return None


_TYPE_VALUES = {
    "spelling", "grammar", "subject_verb", "tense", "verb_form", "article",
    "pronoun", "preposition", "plural", "word_order", "missing_word",
    "extra_word", "punctuation", "capitalization", "word_choice",
    "redundancy", "context", "structure", "style",
}


def _validate_error(e: Dict) -> Optional[Dict]:
    wrong = str(e.get("wrong") or "").strip()
    correct = str(e.get("correct") or "").strip()
    if not wrong or not correct or wrong.lower() == correct.lower():
        return None
    etype = str(e.get("type") or "grammar").lower().replace(" ", "_")
    if etype not in _TYPE_VALUES:
        etype = "grammar"
    try:
        conf = float(e.get("confidence") or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(0.99, conf))
    return {
        "wrong": wrong,
        "correct": correct,
        "type": etype,
        "explanation": str(e.get("explanation") or "").strip(),
        "confidence": conf,
        "context": str(e.get("context") or "").strip(),
    }


def locate_span(text: str, wrong: str, cursor: int = 0) -> Optional[int]:
    """Case-insensitive start offset of ``wrong`` in ``text`` at/after ``cursor``."""
    low_t, low_w = text.lower(), wrong.lower()
    pos = low_t.find(low_w, cursor)
    if pos == -1 and wrong.strip():
        pos = low_t.find(low_w.strip(), cursor)
    return pos if pos != -1 else None


def _occurrences(text: str, wrong: str) -> List[int]:
    """All word-boundary start offsets of ``wrong`` (repeated-word aware)."""
    import re
    needle = (wrong or "").strip()
    if not needle:
        return []
    return [m.start() for m in re.finditer(
        r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])",
        text, re.IGNORECASE)]


def _context_score(text: str, start: int, end: int, context: str) -> int:
    """How many normalized context tokens appear in a window around (start,end)."""
    ctx = " ".join((context or "").lower().split())
    if not ctx:
        return -1
    window = " ".join(text[max(0, start - 60):end + 60].lower().split())
    tokens = ctx.split()
    return sum(1 for t in tokens if t in window)


def _locate_with_context(text: str, fixed: Dict, cursor: int) -> Optional[int]:
    """Locate the EXACT occurrence of a repeated word using the model's
    ``context`` snippet (§7/§11). Falls back to the honest ambiguity handling:
    the occurrence at/after ``cursor``, unchanged from the previous contract."""
    starts = _occurrences(text, fixed["wrong"])
    if not starts:
        return None
    if len(starts) == 1:
        return starts[0]
    ctx = fixed.get("context") or ""
    if ctx:
        best, best_score, best_first = None, -1, None
        for s in starts:
            score = _context_score(text, s, s + len(fixed["wrong"]), ctx)
            if score > best_score:
                best, best_score, best_first = s, score, (s if best_first is None else best_first)
        if best is not None and best_score > 0:
            return best
    for s in starts:
        if s >= cursor:
            return s
    return starts[0]


def _locate_errors(text: str, raw_errors: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    cursor = 0
    for e in raw_errors:
        fixed = _validate_error(e)
        if not fixed:
            continue
        pos = _locate_with_context(text, fixed, cursor)
        multiword = len(fixed["wrong"].split()) > 1
        if pos is None and multiword:
            # phrase not contiguous → highlight only its first token so the
            # correction object is still span-safe
            first = fixed["wrong"].split()[0]
            pos = locate_span(text, first, cursor)
            if pos is None:
                continue
            fixed["wrong"] = text[pos:pos + len(first)]
            fixed["end"] = pos + len(first)
        elif pos is None:
            # single word that cannot be found as a whole token → refuse to
            # guess (otherwise 'go' would be highlighted inside 'goes')
            continue
        else:
            match = text[pos:pos + len(fixed["wrong"])]
            if match.strip().lower() != fixed["wrong"].strip().lower():
                # anchored locator drifted (model "wrong" differs in case/space) →
                # keep the exact model string but realign end onto the text
                fixed["wrong"] = match.rstrip()
            fixed["end"] = pos + len(fixed["wrong"])
        fixed["start"] = pos
        cursor = pos + max(1, len(fixed["wrong"]))
        out.append(fixed)
    return out


def analyze(text: str,
            candidates: Optional[List[Dict]] = None,
            raw_call: Optional[Callable[[str], str]] = None) -> Optional[Dict]:
    """Run the Gemini analysis pass. Returns the parsed result or None."""
    if not text or not text.strip():
        return None
    call = raw_call if raw_call is not None else _provider_callable()
    if call is None:
        return None
    prompt = build_prompt(text, candidates)
    try:
        raw = call(prompt)
    except Exception:
        return None
    if not raw:
        return None
    data = _extract_json(raw)
    if not data or not isinstance(data, dict):
        return None
    raw_errors = data.get("errors") or []
    errors = _locate_errors(text, raw_errors) if isinstance(raw_errors, list) else []
    status = str(data.get("grammar_status") or ("errors_found" if errors else "correct"))
    return {
        "original_text": str(data.get("original_text") or text),
        "corrected_text": str(data.get("corrected_text") or text),
        "errors": errors,
        "grammar_status": status,
        "meaning_preserved": bool(data.get("meaning_preserved", True)),
        "raw": raw,
    }


def ai_key_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("AI_API_KEY"))