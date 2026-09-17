"""pipeline.recheck — second full check (§15/§16) + missed-error corpus (§19).

After a first correction pass, the corrected text is run through the pipeline
again (up to ``max_passes`` to the nearest specification cap of 3). Each later
pass finds NEW errors that the previous pass missed:

    * errors are de-duplicated against the already-reported set,
    * residual re-detections (the previous pass's own fixes) are filtered out,
    * passes stop as soon as the corrected text stops changing (loop guard: a
      set of normalized text hashes prevents go -> goes -> go cycles),
    * every AI_ONLY error (Gemini caught it, the local NLP missed it) that
      survives verification is appended to ``datasets/missed_errors.jsonl``
      so the local layer can be improved from real misses.

All helpers are defensive: logging never raises and never breaks the pipeline.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Callable, Dict, List, Optional

from . import ai_analyzer

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "missed_errors.jsonl")


def dataset_path() -> str:
    return os.environ.get("MISSED_ERRORS_PATH") or _DEFAULT_PATH


def _collections_enabled() -> bool:
    return os.environ.get("COLLECT_MISSED_ERRORS", "1").strip() != "0"


def log_missed_error(wrong: str, correct: str, category: str,
                     confidence: float, text: str, start: int, end: int,
                     sentence: str = "", pass_no: int = 1,
                     discovery: str = "AI_ONLY", reason: str = "") -> bool:
    """Append one AI_ONLY (missed by local NLP) error to the corpus file.

    Never raises. Returns True when the record was written.
    """
    if not _collections_enabled():
        return False
    if not wrong or not correct or wrong.lower() == correct.lower():
        return False
    path = dataset_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        return False
    if category in ("style",):
        return False
    try:
        seed = f"{text}|{start}|{end}|{wrong}|{correct}|{sentence}"
        rid = hashlib.sha1(seed.encode("utf-8", "replace")).hexdigest()[:16]
        record = {
            "id": rid,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "pass": pass_no,
            "discovery": discovery,
            "sentence": sentence,
            "start": start,
            "end": end,
            "wrong": wrong,
            "correct": correct,
            "category": category,
            "confidence": round(float(confidence), 3),
            "reason": reason,
            "original_text": text,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def log_missed_from_result(result: Dict, pass_no: int = 1) -> int:
    """Scan a final result for AI_ONLY errors and write each to the corpus."""
    if not result or not result.get("meta", {}).get("ai_used", False):
        return 0
    text = result.get("original_text") or ""
    sents = (result.get("meta") or {}).get("sentences") or []
    logged = 0
    for e in result.get("errors") or []:
        if e.get("consensus") != "AI_ONLY":
            continue
        sid = int(e.get("sentence_id") or 0)
        sentence = ""
        if 0 <= sid < len(sents):
            sentence = sents[sid].get("text", "") if isinstance(sents[sid], dict) else sents[sid]
        if not sentence:
            sentence = text[max(0, int(e.get("start") or 0) - 40):int(e.get("end") or 0) + 40]
        if log_missed_error(
                wrong=str(e.get("wrong") or ""),
                correct=str(e.get("correct") or ""),
                category=str(e.get("category") or "grammar"),
                confidence=float(e.get("confidence") or 0.0),
                text=text,
                start=int(e.get("start") or 0),
                end=int(e.get("end") or 0),
                sentence=sentence,
                pass_no=pass_no,
                discovery="AI_ONLY",
                reason="gemini found, local NLP missed"):
            logged += 1
    return logged


def _norm_key(s: str) -> str:
    return " ".join((s or "").lower().split())


def _hash(text: str) -> str:
    return hashlib.sha1(_norm_key(text).encode("utf-8")).hexdigest()


def recheck(original: str,
            first: Dict,
            make_pass: Callable[[str], Dict],
            max_passes: int = 3,
            log_missed: bool = True,
            raw_call: Optional[Callable[[str], str]] = None) -> Dict:
    """Run the second full check.

    ``make_pass`` runs ONE full pipeline pass (no re-check) on a given text and
    returns a result dict shaped like ``check_master``'s output. The loop:

        1. continues while new, non-residual, located errors are produced,
        2. stops at ``max_passes`` (spec cap = 3),
        3. stops the moment the corrected text repeats a previously-seen
           normalized hash (loop prevention, §16).

    Returns:
        {
          corrected_text, extra_errors (original-relative canonical objects),
          passes_used, loop_detected, residual_filtered, missed_logged,
        }
    """
    if max_passes <= 1:
        return {"corrected_text": first.get("corrected_text", original),
                "extra_errors": [], "passes_used": 1, "loop_detected": False,
                "residual_filtered": 0, "missed_logged": 0}
    if not first.get("meta", {}).get("ai_used", False):
        return {"corrected_text": first.get("corrected_text", original),
                "extra_errors": [], "passes_used": 1, "loop_detected": False,
                "residual_filtered": 0, "missed_logged": 0}

    reported = set()
    for e in first.get("errors") or []:
        reported.add((_norm_key(e.get("wrong")), _norm_key(e.get("correct"))))

    current = first.get("corrected_text", original)
    seen = {_hash(original), _hash(current)}
    extra: List[Dict] = []
    passes = 1
    loop_detected = False
    residual_filtered = 0
    missed_logged = 0
    text = original

    while passes < max_passes:
        passes += 1
        res = make_pass(current)
        if not res.get("meta", {}).get("ai_used", False):
            break
        next_text = res.get("corrected_text", current)
        h = _hash(next_text)
        if h in seen:
            loop_detected = True
            current = next_text
            break
        seen.add(h)

        new_errors: List[Dict] = []
        for e in res.get("errors") or []:
            w = str(e.get("wrong") or "")
            c = str(e.get("correct") or "")
            key = (_norm_key(w), _norm_key(c))
            if key in reported:
                residual_filtered += 1
                continue
            pos = ai_analyzer.locate_span(text, w)
            if pos is None:
                continue
            # rebuild an original-relative canonical error
            ex = dict(e)
            ex["start"] = pos
            ex["end"] = pos + len(w)
            if ex.get("sentence_id") is not None:
                ex["sentence_id"] = int(ex.get("sentence_id") or 0) + 10000
            new_errors.append(ex)
            reported.add(key)
            if log_missed and ex.get("consensus") == "AI_ONLY":
                if log_missed_error(
                        w, c, str(ex.get("category") or ex.get("type") or "grammar"),
                        float(ex.get("confidence") or 0.0), text, pos, pos + len(w),
                        sentence=text[max(0, pos - 40):pos + len(w) + 40],
                        pass_no=passes, discovery="AI_ONLY",
                        reason="gemini found, local NLP missed"):
                    missed_logged += 1

        if not new_errors:
            break
        extra.extend(new_errors)
        current = next_text

    return {
        "corrected_text": current,
        "extra_errors": extra,
        "passes_used": passes,
        "loop_detected": loop_detected,
        "residual_filtered": residual_filtered,
        "missed_logged": missed_logged,
    }


__all__ = ["recheck", "log_missed_error", "log_missed_from_result",
           "dataset_path"]