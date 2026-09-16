"""Spec §8 regression tests — the 12 context-aware test cases.

Each error sentence must be flagged with exactly the expected replacement and
each clean sentence must produce no error. All rules are general patterns in
unified_pipeline (SVA, TENSE_CONSISTENCY, TENSE_PAST_MARKER, MODAL_WRONG_FORM,
DUPLICATE_VERB); no sentence-specific hacks.

Exercised through the full check_v4 pipeline (matching the "Yesterday, our
sales team has attended..." spec requirement).
"""
import pytest

from new_pipeline import check_v4


def _issues(text):
    return check_v4(text)["errors"]


def _apply_fixes(text, issues):
    out = text
    for iss in sorted(issues, key=lambda i: (i.get("start", 0) or 0), reverse=True):
        repl = iss.get("replacement")
        s, e = iss.get("start", 0), iss.get("end", 0)
        if not repl or e <= s:
            continue
        out = out[:s] + repl + out[e:]
    return out


def _expect_fix(text, want, rule_ids=None):
    issues = _issues(text)
    assert issues, f"expected an error for {text!r}"
    fixed = _apply_fixes(text, issues)
    assert fixed == want, f"got {fixed!r}, want {want!r}"
    if rule_ids is not None:
        got_ids = {i["rule_id"] for i in issues}
        assert got_ids & set(rule_ids), f"expected rules {rule_ids}, got {got_ids}"


def _expect_clean(text):
    issues = _issues(text)
    assert not issues, f"expected NO ERROR for {text!r}, got {issues}"


# ─── Error cases (must be flagged + corrected) ──────────────────────────────

def test_yesterday_present_perfect():
    _expect_fix(
        "Yesterday, our sales team has attended an important meeting.",
        "Yesterday, our sales team attended an important meeting.",
        rule_ids={"TENSE_CONSISTENCY"},
    )

def test_meeting_were():
    _expect_fix(
        "The meeting were scheduled for Monday.",
        "The meeting was scheduled for Monday.",
        rule_ids={"SVA"},
    )

def test_arrive_yesterday():
    _expect_fix(
        "Several employees arrive late yesterday.",
        "Several employees arrived late yesterday.",
    )

def test_modal_past_form():
    _expect_fix(
        "She could completed the presentation.",
        "She could complete the presentation.",
        rule_ids={"MODAL_WRONG_FORM"},
    )

def test_sections_was():
    _expect_fix(
        "Two sections was repeated.",
        "Two sections were repeated.",
        rule_ids={"SVA"},
    )

def test_duplicate_verb():
    _expect_fix(
        "We discussed discuss the remaining problems.",
        "We discussed the remaining problems.",
        rule_ids={"DUPLICATE_VERB"},
    )

def test_each_member_need():
    _expect_fix(
        "Each team member need to be careful.",
        "Each team member needs to be careful.",
        rule_ids={"SVA"},
    )

def test_will_sent():
    _expect_fix(
        "She will sent the documents.",
        "She will send the documents.",
        rule_ids={"MODAL_WRONG_FORM"},
    )


# ─── Clean cases (must NOT be flagged) ──────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Some information was incorrect.",
    "I have already checked the document.",
    "He will send the missing files.",
    "Several important details were missing.",
])
def test_spec_clean_cases(text):
    _expect_clean(text)