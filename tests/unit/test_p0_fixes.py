"""Regression tests for the Phase-4 audit P0 fixes (rule_detector.py).

Each fix closed a precision bug that produced fake corrections or false
positives, or a recall gap that let a real learner error slip through:

    * NARRATIVE_REGULAR no longer fabricates corrections for "properly",
      "finally", "carefully", "again" (adverb / invariant overlaps).
    * e-drop recovery in DIDNT_PAST_FORM / MODAL_PAST_BASE handles silent-e
      (liked->like), y->i (studied->study) and doubled consonants
      (stopped->stop, petted->pet), never producing a mangled base.
    * WHO_WAS skips known-plural heads ("the children who were").
    * REFLEXIVE_SUBJECT only fires in subject position.
    * PLURAL_NOUN_WAS / PLURAL_WAS only fire on known-plural nouns
      ("the students was" flagged, "the house was" untouched, "glass" safe).
    * EVERYDAY_DAY flags adverbial "everyday" but not "everyday life".
    * PERSON_NOUN_BE flags "My brother are" / "The child were".

Tests assert local rule output ONLY (rule_detector) so the fixes stay
hermetic and independent of the AI layer.
"""
import pytest

from pipeline.rule_detector import detect_rules


def _rows(text):
    return [(c["wrong"], c["correct"], c["rule_id"]) for c in detect_rules(text)]


def _ids(text):
    return {c["rule_id"] for c in detect_rules(text)}


# ─── NARRATIVE_REGULAR: never fabricate a correction ────────────────────────

@pytest.mark.parametrize("text", [
    "He properly explains.",
    "She finally ran.",
    "He carefully washed the car.",
    "She again told him.",
    "He properly planned it.",
    "She really enjoys it.",
])
def test_narrative_regular_no_fake_corrections(text):
    assert _rows(text) == []


def test_narrative_regular_still_flags_real_past_needed():
    assert ("walk", "walked", "NARRATIVE_REGULAR") in _rows(
        "Yesterday he walk to the park.")


# ─── DIDNT_PAST_FORM / MODAL_PAST_BASE: e-drop + doubled-consonant recovery ─

@pytest.mark.parametrize("text", [
    "He has liked the movie.",
    "She had loved him.",
    "They have studied hard.",
    "He has moved to London.",
    "They have stopped the car.",
    "She had planned the trip.",
])
def test_aux_past_forms_not_reflagged(text):
    # auxiliary + past participle is correct; base recovery must not fire
    assert _rows(text) == []


def test_did_liked_recovers_silent_e():
    assert ("liked", "like", "DIDNT_PAST_FORM") in _rows("He did liked it.")


def test_modal_liked_recovers_silent_e():
    assert ("liked", "like", "MODAL_PAST_BASE") in _rows("He could liked it.")


def test_did_studied_recovers_y_to_i():
    assert ("studied", "study", "DIDNT_PAST_FORM") in _rows("She did studied hard.")


def test_modal_stopped_recovers_doubled_consonant():
    assert ("stopped", "stop", "MODAL_PAST_BASE") in _rows("He could stopped the car.")


def test_did_rolled_keeps_rolled_base():
    assert ("rolled", "roll", "DIDNT_PAST_FORM") in _rows("He did rolled the ball.")


def test_did_planned_recovers_doubled_consonant():
    assert ("planned", "plan", "DIDNT_PAST_FORM") in _rows("She did planned the trip.")


def test_did_petted_recovers_base_pet():
    assert ("petted", "pet", "DIDNT_PAST_FORM") in _rows("They did petted the dog.")


# ─── WHO_WAS: plural heads are exempt ───────────────────────────────────────

@pytest.mark.parametrize("text", [
    "The children who were playing.",
    "The people who were there.",
    "The women who were singing.",
])
def test_who_were_plural_heads_not_flagged(text):
    assert _rows(text) == []


def test_who_were_singular_head_flagged():
    assert ("were", "was", "WHO_WAS") in _rows("The boy who were playing.")


# ─── REFLEXIVE_SUBJECT: subject position only ───────────────────────────────

@pytest.mark.parametrize("text", [
    "I did it myself.",
    "She cooked dinner herself.",
    "He made it himself.",
    "They enjoyed themselves.",
])
def test_reflexive_object_position_not_flagged(text):
    assert _rows(text) == []


def test_reflexive_subject_position_flagged():
    assert any(r[2] == "REFLEXIVE_SUBJECT" for r in _rows("Myself and John went."))


# ─── PLURAL_NOUN_WAS / PLURAL_WAS: known plurals only ───────────────────────

@pytest.mark.parametrize("text", [
    "The house was big.",
    "The glass was empty.",
    "The mouse was scared.",
    "The news is good.",
])
def test_singular_was_not_flagged(text):
    assert _rows(text) == []


@pytest.mark.parametrize("text", [
    "The students was late.",
    "The birds was singing.",
    "The people was happy.",
    "The children was happy.",
])
def test_known_plural_was_flagged(text):
    assert any(c[2] in ("PLURAL_NOUN_WAS", "PLURAL_WAS") for c in _rows(text))


# ─── EVERYDAY_DAY: adverbial vs attributive ─────────────────────────────────

def test_everyday_adverbial_flagged():
    assert ("everyday", "every day", "EVERYDAY_DAY") in _rows("I go there everyday.")


@pytest.mark.parametrize("text", [
    "An everyday life.",
    "Everyday tasks are easy.",
    "It is an everyday routine.",
])
def test_everyday_attributive_not_flagged(text):
    assert _rows(text) == []


# ─── PERSON_NOUN_BE: 3rd-person singular person-noun + be ───────────────────

def test_person_noun_are_flagged():
    assert ("are", "is", "PERSON_NOUN_BE") in _rows("My brother are waiting.")


def test_person_noun_were_flagged():
    assert ("were", "was", "PERSON_NOUN_BE") in _rows("The child were playing.")


@pytest.mark.parametrize("text", [
    "My parents are here.",
    "The boys are waiting.",
    "I am ready.",
])
def test_plural_or_first_person_not_flagged(text):
    assert _rows(text) == []


# ─── helper: base-form recovery never mangles words ─────────────────────────

def test_no_mangled_bases_smoke():
    # a bag of safe sentences must not generate a candidate whose correction
    # is a non-word or a duplicate of the wrong text.
    safe = [
        "He has liked it.",
        "She had loved it.",
        "They have stopped.",
        "We have planned it.",
        "I moved here.",
        "The children who were happy.",
        "They were playing.",
    ]
    for text in safe:
        for wrong, correct, rule in _rows(text):
            assert correct != wrong
            assert correct.strip()


# ────────────────────────────────────────────────────────────────────────
# TENSE_PAST_MARKER negative-habitual guard
def test_tense_past_marker_keeps_base_after_negative_auxiliary():
    rows = _rows("He don't usually get sick.")
    assert ("get", "got", "TENSE_PAST_MARKER") not in rows
    assert ("get", "got", "TENSE_PAST_MARKER") not in _rows("She doesn't often come here.")


def test_tense_past_marker_still_fixes_real_past_errors():
    rows = _rows("Yesterday I go to the market.")
    assert ("go", "went", "TENSE_PAST_MARKER_AFTER") in rows


# WORD_FORM_EMOTION_ADJ (was very worry -> was very worried)
def test_word_form_emotion_adjective():
    assert ("worry", "worried", "WORD_FORM_EMOTION_ADJ") in _rows("I was very worry.")
    assert ("worry", "worried", "WORD_FORM_EMOTION_ADJ") in _rows("She is so worry about it.")
    assert ("surprise", "surprised", "WORD_FORM_EMOTION_ADJ") in _rows("They were quite surprise.")
    # nominal use of the noun must NOT be touched
    assert ("worry", "worried", "WORD_FORM_EMOTION_ADJ") not in _rows("Worry is a feeling.")
    assert ("tire", "tired", "WORD_FORM_EMOTION_ADJ") in _rows("He feels very tire.")


# MODAL_GERUND_BASE (would going -> would go)
def test_modal_gerund_base():
    assert ("going", "go", "MODAL_GERUND_BASE") in _rows("She told me she would going home early.")
    assert ("eating", "eat", "MODAL_GERUND_BASE") in _rows("We will eating soon.")
    assert ("running", "run", "MODAL_GERUND_BASE") in _rows("They could running faster.")


# NARRATIVE_PAST nested-clause habitual guard (my friend say that she has
# never seen -> say uses PRESENT habitual, so it must be… wait, gold expects
# 'said'; the nested 'never' must not rescue 'say')
def test_narrative_past_nested_clause_habitual():
    rows = _rows(
        "While we were walking, I saw an elephant and my friend say that she "
        "has never seen one before.")
    assert ("say", "said", "NARRATIVE_PAST") in rows