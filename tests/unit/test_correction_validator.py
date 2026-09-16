"""Unit tests for the hardened local CorrectionValidator (no-op + meaning
preservation) used before the AI gate."""
from unified_pipeline import (CorrectionValidator, ErrorCandidate,
                              ErrorCategory)

VAL = CorrectionValidator()


def _cand(text, original, replacement):
    start = text.index(original)
    return ErrorCandidate(start=start, end=start + len(original),
                          original=original, replacement=replacement,
                          category=ErrorCategory.GRAMMAR, rule_id="TEST",
                          message="t", detector="test", raw_confidence=0.9)


def test_valid_fix_accepted():
    ok, reason = VAL.validate(
        _cand("Some of them was missing.", "was", "were"), "Some of them was missing.")
    assert ok is True


def test_none_replacement_rejected():
    ok, reason = VAL.validate(
        _cand("She was here.", "was", None), "She was here.")
    assert ok is False
    assert "No replacement" in reason


def test_noop_rejected():
    ok, reason = VAL.validate(_cand("She was here.", "was", "was"), "She was here.")
    assert ok is False
    assert "no-op" in reason


def test_case_only_change_is_not_noop():
    ok, reason = VAL.validate(_cand("i like dogs.", "i", "I"), "i like dogs.")
    assert ok is True


def test_capitalization_fix_accepted():
    ok, reason = VAL.validate(_cand("she went home.", "s", "S"), "she went home.")
    assert ok is True


def test_multiword_verb_fix_accepted():
    ok, reason = VAL.validate(
        _cand("She answered polite.", "answered polite", "answered politely"),
        "She answered polite.")
    assert ok is True


def test_wh_do_support_accepted():
    ok, reason = VAL.validate(
        _cand("Why he left?", "Why he left", "Why did he leave"),
        "Why he left?")
    assert ok is True


def test_meaning_loss_cannot_slip_outside_span():
    # A content word immediately OUTSIDE the corrected span must be preserved.
    # Replacing "finish" drops nothing around it -> accepted.
    ok, _ = VAL.validate(
        _cand("The team finished the project.", "finished", "completed"),
        "The team finished the project.")
    assert ok is True


def test_content_overlap_mechanism():
    from unified_pipeline import get_nlp
    nlp = get_nlp()
    r = VAL._content_overlap_ratio(
        nlp("The quick brown fox jumps."),
        nlp("Birds fly away."), 0, 0)
    assert r < 0.5