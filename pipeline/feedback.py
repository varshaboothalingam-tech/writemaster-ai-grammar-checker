"""pipeline.feedback — user feedback capture with a quality gate.

When a learner accepts (or rejects) a correction we record it in
``data/feedback/feedback.jsonl``. Only records that pass a quality gate
(required fields, sane values, not a duplicate) are written. The feedback
dataset is NEVER used to retrain a model automatically; it is a clean,
verified future-training source.

Record schema:
    original_text, corrected_text, wrong, correct, error_type, source,
    accepted (0/1), reason (optional), explanation, timestamp
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feedback")
_REQUIRED = {"original_text", "corrected_text", "wrong", "correct", "error_type"}
_MAX_ENTRIES = 100_000


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def quality_gate(record: Dict) -> Optional[str]:
    """Return an error message when the record must be rejected, else None."""
    missing = _REQUIRED - set(record)
    if missing:
        return f"missing fields: {sorted(missing)}"
    original = str(record.get("original_text") or "").strip()
    corrected = str(record.get("corrected_text") or "").strip()
    wrong = str(record.get("wrong") or "").strip()
    correct = str(record.get("correct") or "").strip()
    if not original or not corrected:
        return "empty text"
    if not wrong or not correct or wrong.lower() == correct.lower():
        return "empty or identical wrong/correct"
    if len(original) > 10_000 or len(corrected) > 10_000:
        return "text too long"
    return None


def _dup_fingerprint(record: Dict) -> str:
    h = hashlib.sha256()
    h.update((str(record.get("original_text") or "")).encode("utf-8"))
    h.update("\x00".encode())
    h.update((str(record.get("corrected_text") or "")).encode("utf-8"))
    h.update("\x00".encode())
    h.update((str(record.get("wrong") or "")).lower().encode("utf-8"))
    h.update("\x00".encode())
    h.update((str(record.get("correct") or "")).lower().encode("utf-8"))
    return h.hexdigest()


class FeedbackStore:
    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.dir = base_dir or _BASE
        self._fp: Optional[set] = None

    def _index(self) -> set:
        if self._fp is None:
            self._fp = set()
            if os.path.exists(self.records_path()):
                with open(self.records_path(), "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        self._fp.add(_dup_fingerprint(rec))
        return self._fp

    def records_path(self) -> str:
        return os.path.join(self.dir, "feedback.jsonl")

    def count(self) -> int:
        idx = self._index()
        return len(idx) if self._fp is not None else len(idx)

    def add(self, original_text: str, corrected_text: str, wrong: str, correct: str,
            error_type: str, accepted: bool, source: str = "ai",
            explanation: str = "", reason: str = "") -> Dict:
        """Validate + write one feedback record. Returns a report dict."""
        if not self._index():
            pass
        if self.count() >= _MAX_ENTRIES:
            return {"success": False, "rejected": "feedback dataset full"}
        record = {
            "original_text": original_text,
            "corrected_text": corrected_text,
            "wrong": wrong,
            "correct": correct,
            "error_type": error_type,
            "source": source,
            "accepted": 1 if accepted else 0,
            "explanation": explanation,
            "reason": reason,
            "timestamp": _now(),
        }
        problem = quality_gate(record)
        if problem:
            return {"success": False, "rejected": problem}
        fp = _dup_fingerprint(record)
        if fp in self._index():
            return {"success": False, "rejected": "duplicate"}
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(self.records_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            return {"success": False, "rejected": f"write error: {exc}"}
        self._fp.add(fp)
        return {"success": True, "id": fp}

    def recent(self, limit: int = 20) -> List[Dict]:
        out: List[Dict] = []
        if not os.path.exists(self.records_path()):
            return out
        with open(self.records_path(), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out[-limit:]


_default_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    global _default_store
    if _default_store is None:
        _default_store = FeedbackStore()
    return _default_store


def record_feedback(original_text: str, corrected_text: str, wrong: str, correct: str,
                    error_type: str, accepted: bool, source: str = "ai",
                    explanation: str = "", reason: str = "") -> Dict:
    return get_feedback_store().add(original_text, corrected_text, wrong, correct,
                                    error_type, accepted, source, explanation, reason)


__all__ = ["FeedbackStore", "record_feedback", "get_feedback_store", "quality_gate"]