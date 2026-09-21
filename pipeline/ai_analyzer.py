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
    "You are a full-context English grammar analyzer. Read the ENTIRE text below "
    "and find every grammar, spelling, punctuation, tense, and agreement error. "
    "Reason from general English grammar rules — never rely on a fixed word list.\n\n"
    "ERROR TYPES TO ACTIVELY CHECK:\n\n"
    "- TENSE CONSISTENCY in a past-tense narrative: every main verb should be "
    "past tense unless there's a clear reason otherwise. "
    'e.g. "We leaves" -> "We left", "my father want" -> "wanted", '
    '"fall down" -> "fell down", "we opens" -> "we opened", '
    '"decides" -> "decided", "One of them invite" -> "invited"\n'
    '- EXISTENTIAL "THERE" + SVA: "there was/were" must agree with the noun '
    "that follows it, including plural nouns. "
    'e.g. "There was sandwiches" -> "There were sandwiches"\n'
    '- RELATIVE CLAUSE SVA: the verb inside a relative clause ("who", "which", '
    '"that") must agree with its own subject, not the main clause subject. '
    'e.g. "children who was building" -> "children who were building"\n'
    '- POSSESSIVE APOSTROPHE MISUSE creating a false contraction: a possessive '
    "'s before a verb like \"don't/doesn't\" is usually wrong — the writer "
    'meant a plain subject, not "belonging to." '
    'e.g. "my brother\'s don\'t want" -> "my brother didn\'t want" '
    "(note: also fix the tense - didn't, not doesn't, if context is past)\n"
    '- ARTICLE before "few/little": "few" alone means "not many" (often '
    'negative); "a few" means "some." In most narrative contexts "a few" '
    'is intended. e.g. "After few minutes" -> "After a few minutes"\n'
    "- SPELLING: any misspelled word, not just common ones. "
    'e.g. "droped" -> "dropped"\n'
    '- VERB FORM after "to": bare infinitive required. '
    'e.g. "to spent the day" -> "to spend the day"\n'
    "- MID-TEXT CAPITALIZATION after a period: the first word of a new "
    "sentence must be capitalized even if it's a common lowercase word "
    'like "everyone". e.g. ". everyone was hungry" -> ". Everyone was hungry"\n\n'
    "CRITICAL — DO NOT FLAG CORRECT ENGLISH:\n"
    "If a native speaker would write or accept a sentence as-is, do not flag "
    "it, even if an alternative phrasing exists. Do not flag \"sat on the "
    "chair\"-type sentences with multiple valid prepositions/phrasings. Do not "
    "flag \"We had enjoyed the trip\" as a hard error — past perfect is "
    "grammatically valid there even though simple past is more natural; if you "
    "flag it, mark it as low-confidence style (\"subcategory\": "
    "\"style_suggestion\", confidence below 0.60), never as a definite grammar error.\n\n"
    "OUTPUT RULES:\n"
    "- start/end are LOCAL character offsets into the exact text given, 0-indexed.\n"
    '- text[start:end] must exactly equal "original".\n'
    "- No overlapping spans. Minimal fix only — don't rewrite beyond the error.\n"
    "- confidence: reserve >0.90 for errors a native speaker would certainly fix.\n"
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
        "context, keep only the correct ones, and give exact start/end offsets for each):\n"
        + cand_block + "\n\n"
        + "Text to analyze:\n"
        + '"""\n' + text + '\n"""\n\n'
        + "Return ONLY valid JSON, no markdown fences, no prose, in this exact shape:\n"
        + '{"errors": [{"start": <int>, "end": <int>, "original": "<exact substring of '
        'text[start:end]>", "replacement": "<corrected text>", '
        '"category": "grammar" | "spelling" | "punctuation" | "capitalization" | '
        '"word_choice", "subcategory": "<specific type, e.g. subject_verb_agreement, '
        'tense_consistency, possessive>", "confidence": <float>, "explanation": '
        '"<one short sentence>"}]}\n'
        'If there are no errors, return {"errors": []}.\n'
    )


def _provider_callable() -> Optional[Callable[[str], str]]:
    """Return the default AI transport (provider chain / ensemble), or None."""
    try:
        from ai_validator import get_validator, AIValidator
        v = get_validator()
        if v.is_enabled():
            return v.call_any
        # The singleton may have been built before Ollama came up or with a
        # different provider set — re-resolve the current env fresh.
        fresh = AIValidator()
        if fresh.is_enabled():
            return fresh.call_any
        return None
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
    # last-resort: slice between first '{' and last '}' if the whole string is
    # one clean object (with fenced/backtick noise stripped above).
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except ValueError:
            pass
    # multi-array repair: the model sometimes emits errors as separate arrays,
    # e.g. {"errors": [{...}], [{...}], [{...}]} — splice the brace objects
    # together into one errors list. Run BEFORE the truncation repair so a
    # partial one-error prefix does not win over the full set.
    try:
        errs = [_ for _ in _iter_brace_objects(s)
                if ("original" in _ or "wrong" in _)]
        if errs:
            return {"errors": errs}
    except Exception:
        pass
    # truncated-JSON repair: the model sometimes gets cut off mid-list; the
    # outer structure is always {..., "errors": [...]}, so repair by closing
    # the unclosed brackets and walking closing braces backwards.
    for suffix in ("", "]}"):
        end = len(s)
        for _ in range(300):
            end = s.rfind("}", 0, end)
            if end == -1:
                break
            try:
                obj = json.loads(s[a:end + 1] + suffix)
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                continue
    return None


def _iter_brace_objects(s: str):
    """Yield every balanced {...} object found anywhere in the string."""
    import re as _re
    # walk delimiters so nested braces are handled; strings containing { } are
    # skipped by tracking quote state
    stack = []
    infl = False
    esc = False
    for i, ch in enumerate(s):
        if infl:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                infl = False
            continue
        if ch == '"':
            infl = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            chunk = s[start:i + 1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict) and obj:
                    yield obj
            except (ValueError, TypeError):
                pass
            continue


_TYPE_VALUES = {
    "spelling", "grammar", "subject_verb", "tense", "verb_form", "article",
    "pronoun", "preposition", "plural", "word_order", "missing_word",
    "extra_word", "punctuation", "capitalization", "word_choice",
    "redundancy", "context", "structure", "style",
}


def _validate_error(e: Dict) -> Optional[Dict]:
    wrong = str(e.get("original") or e.get("wrong") or "").strip()
    correct = str(e.get("replacement") or e.get("correct") or "").strip()
    if not wrong or not correct or wrong.lower() == correct.lower():
        return None
    etype = str(e.get("category") or e.get("type") or "grammar").lower().replace(" ", "_")
    if etype not in _TYPE_VALUES:
        etype = "grammar"
    sub = str(e.get("subcategory") or "").strip().lower().replace(" ", "_")
    try:
        conf = float(e.get("confidence") or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(0.99, conf))
    return {
        "wrong": wrong,
        "correct": correct,
        "type": etype,
        "subcategory": sub,
        "explanation": str(e.get("explanation") or "").strip(),
        "confidence": conf,
        "context": str(e.get("context") or "").strip(),
        "start": e.get("start"),
        "end": e.get("end"),
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
        # trust a model-provided span ONLY when it exactly matches the wrong
        # string; otherwise re-locate (model offsets are frequently off).
        s, en = fixed.get("start"), fixed.get("end")
        if (isinstance(s, int) and isinstance(en, int) and 0 <= s < en <= len(text)
                and text[s:en].strip().lower() == fixed["wrong"].lower()):
            fixed["start"], fixed["end"] = s, en
            cursor = en + max(1, len(fixed["wrong"]))
            out.append(fixed)
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
    if "corrected_text" not in data or not data.get("corrected_text"):
        from .aggregator import apply_corrections
        data["corrected_text"] = apply_corrections(
            text, [dict(e) for e in errors], min_confidence=0.0).get("corrected_text", text)
    return {
        "original_text": str(data.get("original_text") or text),
        "corrected_text": data["corrected_text"],
        "errors": errors,
        "grammar_status": status,
        "meaning_preserved": bool(data.get("meaning_preserved", True)),
        "raw": raw,
    }


def ai_key_configured() -> bool:
    """True when any AI provider is usable for the full-text analysis pass.

    A remote provider counts only when its API key is present; local Ollama
    counts when a provider config selects it (or it is reachable by default).
    AI_PROVIDER=none / off / disabled always returns False and wins over any
    configured keys.
    """
    if (os.environ.get("AI_PROVIDER") or "").lower() in ("none", "off", "disabled"):
        return False
    if any(os.environ.get(k) for k in (
            "GEMINI_API_KEY", "AI_API_KEY", "OPENAI_API_KEY",
            "GROQ_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")):
        return True
    try:
        from ai_validator import _enabled_providers
        return "ollama" in _enabled_providers()
    except Exception:
        return False