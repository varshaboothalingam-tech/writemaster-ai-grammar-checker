"""Unit tests for the 8 new error patterns and their FP-safety guards.

All rules live in the NLPDetector (unified_pipeline) and are exercised through
the check_v4 pipeline, mirroring the DIDNT_PAST_FORM / DOUBLE_NEGATIVE tests.
"""
import pytest

from new_pipeline import check_v4
from unified_pipeline import NLPDetector


def _v4_has(text, rule_id):
    ids = [e["rule_id"] for e in check_v4(text)["errors"]]
    return rule_id in ids


# ─── 1. COLLECTIVE SVA, AmE `have/had` (team have → team has) ───────────────

@pytest.mark.parametrize("text", [
    "Our team have attended the meeting.",
    "The team have made a decision.",
    "The family have moved to Boston.",
    "The committee have approved the proposal.",
])
def test_collective_have_sva_pos(text):
    assert _v4_has(text, "SVA")

@pytest.mark.parametrize("text", [
    "Our team has attended the meeting.",
    "The team has made a decision.",
    "The committee has approved the proposal.",
])
def test_collective_have_sva_neg(text):
    assert not _v4_has(text, "SVA")

@pytest.mark.parametrize("text", [
    "The staff have attended the ceremony.",
    "The police have made a decision.",
    "The government have considered the proposal.",
    "The staff are divided on the issue.",
    "The government are considering new laws.",
])
def test_collective_bre_tolerant_neg(text):
    assert not _v4_has(text, "SVA")


# ─── 2. RELATIVE SVA (manager who were → who was) ───────────────────────────

def test_relative_sva_pos():
    assert _v4_has("I spoke with the manager who were responsible for the delay.", "SVA")

def test_relative_sva_neg():
    assert not _v4_has("The people who were responsible apologized.", "SVA")


# ─── 3. MODAL + VBN / VB-ed (could noticed → could notice) ──────────────────

@pytest.mark.parametrize("text", [
    "He could noticed the mistake.",
    "She should focused on the task.",
    "They could stopped the car.",
])
def test_modal_wrong_form_pos(text):
    assert _v4_has(text, "MODAL_WRONG_FORM")

@pytest.mark.parametrize("text", [
    "The door must be locked before leaving.",
    "He could be noticed immediately.",
    "Must the file be deleted?",
    "She must have completed the form.",
    "We should have focused on the test.",
    "You can feed the animals.",
    "He could decide to leave.",
    "He could of done it better.",   # "of" guard → WOULD_OF_* domain, not modal
])
def test_modal_wrong_form_neg(text):
    assert not _v4_has(text, "MODAL_WRONG_FORM")


# ─── 4. DUPLICATE VERB (discussed discuss → discussed) ──────────────────────

@pytest.mark.parametrize("text", [
    "We discussed discuss the next steps.",
    "She explained explain the process.",
])
def test_duplicate_verb_pos(text):
    assert _v4_has(text, "DUPLICATE_VERB")

def test_duplicate_verb_case_b_detector_level():
    # "have have" is flagged DUPLICATE_VERB by the nlp detector; the merged
    # pipeline may promote it to REPEATED_WORD (higher confidence) instead.
    import spacy
    nlp = spacy.load("en_core_web_sm")
    text = "They have have already submitted the report."
    ids = [c.rule_id for c in NLPDetector().detect(text, nlp(text))]
    assert "DUPLICATE_VERB" in ids

def test_duplicate_verb_case_b_pipeline():
    # Accept whichever naming the merged pipeline keeps.
    ids = [e["rule_id"] for e in check_v4("They have have already submitted the report.")["errors"]]
    assert "DUPLICATE_VERB" in ids or "REPEATED_WORD" in ids

@pytest.mark.parametrize("text", [
    "I went to go to the office.",
    "They had had enough time.",
])
def test_duplicate_verb_neg(text):
    assert not _v4_has(text, "DUPLICATE_VERB")
    assert not _v4_has(text, "REPEATED_WORD")


# ─── 5. PREPOSITION SCHEDULED ON (scheduled on Monday → for) ────────────────

@pytest.mark.parametrize("text", [
    "The meeting is scheduled on Monday morning.",
    "The event was scheduled on Friday.",
    "The interview is scheduled on Wednesday afternoon.",
])
def test_scheduled_on_pos(text):
    assert _v4_has(text, "PREPOSITION_SCHEDULED_ON")

@pytest.mark.parametrize("text", [
    "The meeting is scheduled on the calendar.",
    "The appointment is scheduled in the system.",
    "The meeting is scheduled for Tuesday.",
    "The call is scheduled for Monday morning.",
])
def test_scheduled_on_neg(text):
    assert not _v4_has(text, "PREPOSITION_SCHEDULED_ON")


# ─── 6. TENSE BACKSHIFT (past narrative → past subordinate) ─────────────────

def test_backshift_pos():
    assert _v4_has("Last week, I went to the office and noticed that the slide contains incorrect information.", "TENSE_BACKSHIFT")

def test_backshift_pos_clausal():
    assert _v4_has("Yesterday, she realized that the app includes a serious bug.", "TENSE_BACKSHIFT")

@pytest.mark.parametrize("text", [
    "The slide contains incorrect information.",
    "She told me she is tired.",
    "Last week, the teacher said the earth is round.",   # BE-form copula guard
    "Today, I noticed that the app works fine.",         # "today" is not a past marker
])
def test_backshift_neg(text):
    assert not _v4_has(text, "TENSE_BACKSHIFT")


# ─── 7. ANYBODY/ANYONE DECLARATIVE (Anybody knew → Nobody knew) ─────────────

@pytest.mark.parametrize("text", [
    "Anybody knew the answer.",
    "Anyone saw the old building.",
])
def test_anybody_declarative_pos(text):
    assert _v4_has(text, "ANYBODY_DECLARATIVE")

@pytest.mark.parametrize("text", [
    "Anybody can learn to code.",
    "Anybody who knows the answer should speak up.",
    "Anyone with experience is welcome.",
    "Did anybody know the answer?",
])
def test_anybody_declarative_neg(text):
    assert not _v4_has(text, "ANYBODY_DECLARATIVE")


# ─── 8. EDGE: pipeline must not crash on trivial inputs ─────────────────────

@pytest.mark.parametrize("text", ["", "   ", "OK.", "React", "pH"])
def test_edge_no_crash(text):
    check_v4(text)   # must not raise