"""
Suggestion Validator — the gatekeeper between rule detection and user display.

Every candidate error from any checker must pass through this module before
being shown to the user. Its job is to minimize false positives and ensure
every suggestion is correct, minimal, and meaningful.

Pipeline:
  rule detection → candidate → context analysis → correction validation
  → confidence scoring → false-positive filter → final suggestion
"""

import re
import os
import json
from typing import List, Dict, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Canonical candidate schema
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = (
    "rule_id", "category", "original_text", "replacement",
    "explanation", "confidence", "severity",
    "start_position", "end_position",
)

# Confidence bands
BAND_SHOW = 85       # 85-100: always show
BAND_SECONDARY = 75  # 75-84: show if any supporting signal exists (same or different category)
# Below 75: suppressed (logged for review)

# Style-only categories that must NOT appear in the grammar-error layer
STYLE_ONLY_CATEGORIES = {
    "passive_voice", "sentence_length", "informal_register",
    "filler_words", "wordiness", "weak_words",
}

# Words / patterns that are never grammar errors
_NEVER_FLAG_PATTERNS: List[re.Pattern] = []

# Known false-positive corrections: (original, suggestion) → always suppress
_FALSE_POSITIVE_CORRECTIONS = {
    ("as", "ere"),
    ("as", "ear"),
    ("as", "os"),
    ("is", "it"),
    ("an", "a"),
    ("a", "an"),
    ("to", "too"),
    ("too", "to"),
    ("in", "on"),
    ("on", "in"),
}

# Proper-noun / technical-name cache (loaded once)
_TECHNICAL_NAMES: Set[str] = set()

# User dictionary (words the user explicitly accepted)
_USER_DICT: Set[str] = set()

_USER_DICT_PATH: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_candidates(
    candidates: List[Dict],
    text: str,
    supporting_signals: Optional[List[Dict]] = None,
    user_dict_path: Optional[str] = None,
) -> List[Dict]:
    """Validate, filter, and return only high-confidence suggestions.

    Args:
        candidates: Raw errors from all checkers.
        text: The original text being checked.
        supporting_signals: Optional extra candidates from a *different*
            checker used to elevate borderline (80-89) suggestions.
        user_dict_path: Path to user_dictionary.json.

    Returns:
        Filtered, deduplicated, sorted list of validated suggestions.
    """
    global _USER_DICT, _USER_DICT_PATH
    if user_dict_path:
        _USER_DICT_PATH = user_dict_path
        _load_user_dict(user_dict_path)

    validated: List[Dict] = []
    for c in candidates:
        # 1. Ensure required fields
        _ensure_fields(c)

        # 2. Skip style-only issues in grammar layer
        if c.get("category", "").upper() in STYLE_ONLY_CATEGORIES:
            continue

        # 2b. Skip known false-positive corrections
        pair = (c.get("original_text", "").strip().lower(),
                c.get("replacement", "").strip().lower())
        if pair in _FALSE_POSITIVE_CORRECTIONS:
            continue

        # 3. Skip if word is in user dictionary
        if _is_user_dict_word(c, text):
            continue

        # 4. Skip URLs, emails, code-like content
        if _is_unflaggable_span(c, text):
            continue

        # 5. Skip if replacement is empty or identical to original
        if not c.get("replacement") or not c["replacement"].strip():
            continue
        if c["replacement"].strip().lower() == c["original_text"].strip().lower():
            continue

        # 6. Validate correction is reasonable
        if not _is_correction_valid(c, text):
            continue

        # 7. Compute final confidence with context boost / penalty
        final_conf = _compute_final_confidence(c, text, supporting_signals)
        c["confidence"] = round(final_conf, 2)

        # 8. Apply confidence band filter
        band = _get_band(final_conf)
        if band == "suppress":
            _log_suppressed(c)
            continue
        if band == "secondary":
            # Only show if a second independent checker agrees on this span
            if not _has_independent_agreement(c, supporting_signals):
                _log_suppressed(c)
                continue

        # 9. Enforce minimal span
        _enforce_minimal_span(c, text)

        validated.append(c)

    # 10. Final dedup + conflict resolution
    validated = _deduplicate_and_resolve(validated)
    validated.sort(key=lambda e: (-_severity_order(e["severity"]), -e.get("confidence", 0)))
    return validated


# ---------------------------------------------------------------------------
# 1. Field normalization
# ---------------------------------------------------------------------------

def _ensure_fields(c: Dict):
    """Fill missing fields with safe defaults."""
    c.setdefault("rule_id", "unknown")
    c.setdefault("category", "GRAMMAR")
    c.setdefault("original_text", "")
    c.setdefault("replacement", "")
    c.setdefault("explanation", "")
    c.setdefault("confidence", 0.5)
    c.setdefault("severity", "MEDIUM")
    c.setdefault("start_position", 0)
    c.setdefault("end_position", 0)
    c.setdefault("error_type", c["category"].lower())
    c.setdefault("message", c["explanation"])
    c.setdefault("context", "")
    # Normalize confidence to 0-1 scale if it looks like 0-100
    if isinstance(c["confidence"], (int, float)):
        if c["confidence"] > 1.0:
            c["confidence"] = c["confidence"] / 100.0


# ---------------------------------------------------------------------------
# 2-3. User dictionary & unflaggable spans
# ---------------------------------------------------------------------------

def _load_user_dict(path: str):
    global _USER_DICT
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            words = data.get("words", [])
            _USER_DICT = {w.lower() for w in words if isinstance(w, str)}
        except Exception:
            _USER_DICT = set()
    else:
        _USER_DICT = set()


def add_to_user_dict(word: str, path: Optional[str] = None):
    """Add a word to the user dictionary and persist it."""
    global _USER_DICT
    p = path or _USER_DICT_PATH
    if not p:
        return
    _USER_DICT.add(word.lower())
    data = {"words": sorted(_USER_DICT)}
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _is_user_dict_word(c: Dict, text: str) -> bool:
    word = c.get("original_text", "").strip().lower()
    return word in _USER_DICT


_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
_CODE_LIKE_RE = re.compile(r'`[^`]+`|<[a-z/][^>]*>|\{[^}]+\}')


def _is_unflaggable_span(c: Dict, text: str) -> bool:
    start = c.get("start_position", 0)
    end = c.get("end_position", 0)
    span = text[max(0, start):min(len(text), end)]
    if _URL_RE.search(span):
        return True
    if _EMAIL_RE.search(span):
        return True
    if _CODE_LIKE_RE.search(span):
        return True
    return False


# ---------------------------------------------------------------------------
# 5-6. Correction validation
# ---------------------------------------------------------------------------

def _is_correction_valid(c: Dict, text: str) -> bool:
    replacement = c.get("replacement", "").strip()
    original = c.get("original_text", "").strip()
    if not replacement:
        return False
    # Replacement must not be wildly longer (more than 3x) unless it's a multi-word fix
    if len(replacement) > len(original) * 4 and len(original) > 3:
        return False
    # Replacement must contain at least some alphabetic content
    if not re.search(r'[a-zA-Z]', replacement):
        return False
    # Category-specific validation
    cat = c.get("category", "").upper()
    if cat == "SPELLING":
        if not _is_spelling_correction_valid(original, replacement):
            return False
    if cat == "GRAMMAR":
        if not _is_grammar_correction_valid(c, text):
            return False
    return True


def _is_spelling_correction_valid(original: str, replacement: str) -> bool:
    orig_lower = original.lower().strip()
    repl_lower = replacement.lower().strip()
    # Same word = not a correction
    if orig_lower == repl_lower:
        return False
    # Replacement is absurdly short (single char for a multi-char word)
    if len(repl_lower) == 1 and len(orig_lower) > 3:
        return False
    return True


def _is_grammar_correction_valid(c: Dict, text: str) -> bool:
    replacement = c.get("replacement", "").strip()
    original = c.get("original_text", "").strip()
    # Basic: replacement should not be completely empty after stripping
    if not replacement:
        return False
    # Check if the replacement preserves case intent (first letter)
    if original and replacement:
        if original[0].isupper() and replacement[0].islower():
            # Don't reject outright, but we might want to preserve case
            pass  # Allow it — sentence position matters
    return True


# ---------------------------------------------------------------------------
# 7. Confidence computation
# ---------------------------------------------------------------------------

def _compute_final_confidence(
    c: Dict,
    text: str,
    supporting_signals: Optional[List[Dict]] = None,
) -> float:
    """Compute final confidence with contextual adjustments."""
    base = float(c.get("confidence", 0.5))
    cat = c.get("category", "").upper()
    start = c.get("start_position", 0)
    end = c.get("end_position", 0)

    # Boost for high-signal categories
    if cat == "SPELLING":
        base = min(base + 0.05, 1.0)
    elif cat == "CONTEXTUAL_WORD_USAGE":
        base = min(base + 0.03, 1.0)

    # Penalty for very short spans (1-2 chars) — more likely false positives
    span_len = end - start
    if span_len <= 2:
        base *= 0.9

    # Penalty if the "error" word appears multiple times in the text
    # (might be intentional repetition or proper noun)
    word = c.get("original_text", "").strip().lower()
    if word and len(word) > 2:
        count = text.lower().count(word)
        if count > 3:
            base *= 0.92

    # Penalty for sentence-initial position (capital letter confusion)
    if start == 0 or (start > 0 and text[start - 1:start] in '.!?\n'):
        cat = c.get("category", "").upper()
        if cat in ("CAPITALIZATION", "SPELLING"):
            base *= 0.95

    # Boost if there's independent agreement from another checker
    if supporting_signals:
        for sig in supporting_signals:
            sig_start = sig.get("start_position", 0)
            sig_end = sig.get("end_position", 0)
            if abs(sig_start - start) <= 2 and abs(sig_end - end) <= 2:
                if sig.get("category", "") != cat:
                    base = min(base + 0.05, 1.0)
                    break

    return max(0.0, min(1.0, base))


# ---------------------------------------------------------------------------
# 8. Confidence band filtering
# ---------------------------------------------------------------------------

def _get_band(confidence: float) -> str:
    pct = confidence * 100
    if pct >= BAND_SHOW:
        return "show"
    elif pct >= BAND_SECONDARY:
        return "secondary"
    else:
        return "suppress"


def _has_independent_agreement(
    c: Dict,
    supporting_signals: Optional[List[Dict]] = None,
) -> bool:
    if not supporting_signals:
        return False
    start = c.get("start_position", 0)
    end = c.get("end_position", 0)
    cat = c.get("category", "").upper()
    rule_id = c.get("rule_id", "")
    for sig in supporting_signals:
        sig_start = sig.get("start_position", 0)
        sig_end = sig.get("end_position", 0)
        sig_cat = sig.get("category", "").upper()
        sig_rule = sig.get("rule_id", "")
        # Overlapping span from a different rule OR different category
        if sig_start < end and sig_end > start:
            if sig_cat != cat or sig_rule != rule_id:
                return True
    return False


# ---------------------------------------------------------------------------
# 9. Minimal span enforcement
# ---------------------------------------------------------------------------

def _enforce_minimal_span(c: Dict, text: str):
    """Shrink the span to the smallest range that contains the actual change.
    Only applies to SPELLING corrections where the replacement is a simple
    character-level fix. For grammar/contextual corrections, leave the span as-is
    since the replacement is often a completely different word."""
    cat = c.get("category", "").upper()
    if cat != "SPELLING":
        return

    original = c.get("original_text", "")
    replacement = c.get("replacement", "")
    start = c.get("start_position", 0)
    end = c.get("end_position", 0)

    orig_lower = original.lower()
    repl_lower = replacement.lower()

    # Don't shrink multi-word replacements
    if " " in replacement.strip() or " " in original.strip():
        return

    # Find common prefix
    prefix_len = 0
    for i in range(min(len(orig_lower), len(repl_lower))):
        if orig_lower[i] == repl_lower[i]:
            prefix_len += 1
        else:
            break

    # Find common suffix
    suffix_len = 0
    for i in range(1, min(len(orig_lower), len(repl_lower)) - prefix_len + 1):
        if orig_lower[-i] == repl_lower[-i]:
            suffix_len += 1
        else:
            break

    # Only shrink if we actually reduce the span meaningfully
    new_orig = original[prefix_len:len(original) - suffix_len if suffix_len else None]
    new_repl = replacement[prefix_len:len(replacement) - suffix_len if suffix_len else None]

    if new_orig and new_repl and len(new_orig) < len(original):
        c["original_text"] = new_orig
        c["replacement"] = new_repl
        c["start_position"] = start + prefix_len
        c["end_position"] = end - suffix_len


# ---------------------------------------------------------------------------
# 10. Deduplication & conflict resolution
# ---------------------------------------------------------------------------

CATEGORY_PRIORITY = {
    "CONTEXTUAL_WORD_USAGE": 0, "GRAMMAR": 1, "WORD_USAGE": 2,
    "SPELLING": 3, "PUNCTUATION": 4, "CAPITALIZATION": 5, "STYLE": 6,
}


def _deduplicate_and_resolve(candidates: List[Dict]) -> List[Dict]:
    """Deduplicate and resolve conflicts between candidates."""
    if not candidates:
        return []

    # Group by span overlap
    groups: List[List[Dict]] = []
    for c in candidates:
        placed = False
        for group in groups:
            if _overlaps_any(c, group):
                group.append(c)
                placed = True
                break
        if not placed:
            groups.append([c])

    resolved: List[Dict] = []
    for group in groups:
        if len(group) == 1:
            resolved.append(group[0])
        else:
            # Keep complementary corrections that fix different words
            kept = _pick_complementary(group)
            resolved.extend(kept)

    return resolved


def _pick_complementary(group: List[Dict]) -> List[Dict]:
    """From a group of overlapping corrections, keep complementary ones.
    Two corrections are complementary if they fix different original words.
    Two corrections are contradictory if they fix the same word differently."""
    # Sort by category priority, span length, confidence
    def sort_key(c):
        cat_pri = CATEGORY_PRIORITY.get(c.get("category", "").upper(), 10)
        span = c.get("end_position", 0) - c.get("start_position", 0)
        conf = c.get("confidence", 0)
        return (cat_pri, -span, -conf)

    sorted_group = sorted(group, key=sort_key)

    kept = []
    seen_originals = {}  # original_text.lower() -> replacement.lower()

    for c in sorted_group:
        orig = c.get("original_text", "").lower().strip()
        repl = c.get("replacement", "").lower().strip()

        if orig in seen_originals:
            prev_repl = seen_originals[orig]
            if prev_repl == repl:
                # Same correction, skip duplicate
                continue
            else:
                # Truly contradictory: same word, different fix → keep higher confidence only
                # Since sorted by confidence desc, skip the lower-confidence one
                continue
        else:
            seen_originals[orig] = repl
            kept.append(c)

    return kept


def _overlaps_any(c: Dict, group: List[Dict]) -> bool:
    start = c.get("start_position", 0)
    end = c.get("end_position", 0)
    cat = c.get("category", "").upper()
    for g in group:
        g_start = g.get("start_position", 0)
        g_end = g.get("end_position", 0)
        g_cat = g.get("category", "").upper()
        # Overlapping spans from same category: exact duplicate
        if g_start == start and g_end == end and g_cat == cat:
            return True
        # Overlapping spans from different categories
        if g_start < end and g_end > start:
            return True
    return False


def _pick_best(group: List[Dict]) -> Optional[Dict]:
    """Pick the best candidate from a conflict group.
    Only suppress if two candidates propose different corrections for the
    SAME original text (truly contradictory). Complementary corrections that
    fix different words in overlapping spans should all be kept."""
    # Sort by: category priority (lower = better), span length (longer = more complete), confidence
    def sort_key(c):
        cat_pri = CATEGORY_PRIORITY.get(c.get("category", "").upper(), 10)
        span = c.get("end_position", 0) - c.get("start_position", 0)
        conf = c.get("confidence", 0)
        return (cat_pri, -span, -conf)

    group.sort(key=sort_key)

    # Only suppress if there are truly contradictory corrections
    # (same original_text but different replacement)
    originals_seen = {}
    for c in group:
        orig = c.get("original_text", "").lower().strip()
        repl = c.get("replacement", "").lower().strip()
        if orig in originals_seen:
            prev_repl = originals_seen[orig]
            if prev_repl != repl:
                # Truly contradictory: same word, different fix → keep higher confidence only
                # (this is fine, we already sorted by confidence)
                pass  # The sort already puts the best first
        else:
            originals_seen[orig] = repl

    return group[0] if group else None


def _log_suppressed(c: Dict):
    """Log suppressed candidates for false-negative review."""
    # In production, this could write to a file or metrics system.
    # For now, a no-op placeholder.
    pass


# ---------------------------------------------------------------------------
# Severity ordering helper
# ---------------------------------------------------------------------------

def _severity_order(sev: str) -> int:
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(sev, 1)
