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
    "9. For each error give one word-safe 'wrong' and one 'correct' string.\n\n"
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
        + "Candidate hints from a local detector (they MAY contain false positives — verify each "
        "against the real context, keep only the correct ones):\n"
        + cand_block + "\n\n"
        + "Text to analyze:\n"
        + text + "\n\n"
        + 'Return ONLY strict JSON with this exact shape:\n'
        + '{"original_text": "", "corrected_text": "", "errors": ['
        '{"wrong": "", "correct": "", "type": "spelling|grammar|subject_verb|tense|verb_form|'
        'article|pronoun|preposition|plural|word_order|missing_word|extra_word|punctuation|'
        'capitalization|word_choice|redundancy|context", "explanation": "", "confidence": 0.9}], '
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
    }


def locate_span(text: str, wrong: str, cursor: int = 0) -> Optional[int]:
    """Case-insensitive start offset of ``wrong`` in ``text`` at/after ``cursor``."""
    low_t, low_w = text.lower(), wrong.lower()
    pos = low_t.find(low_w, cursor)
    if pos == -1 and wrong.strip():
        pos = low_t.find(low_w.strip(), cursor)
    return pos if pos != -1 else None


def _locate_errors(text: str, raw_errors: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    cursor = 0
    for e in raw_errors:
        fixed = _validate_error(e)
        if not fixed:
            continue
        pos = locate_span(text, fixed["wrong"], cursor)
        if pos is None:
            # highlight only the first token if the phrase is not contiguous
            first = fixed["wrong"].split()[0]
            pos = locate_span(text, first, cursor)
            if pos is None:
                continue
            fixed["wrong"] = text[pos:pos + len(first)]
            fixed["end"] = pos + len(first)
        else:
            fixed["wrong"] = text[pos:pos + len(fixed["wrong"])]
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