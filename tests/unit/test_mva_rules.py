"""Regression tests for MVA round-2 rule additions (rule_detector.py)."""
import pytest
from pipeline.rule_detector import detect_rules


def _rows(text):
    return [(c["wrong"], c["correct"], c["rule_id"]) for c in detect_rules(text)]


def test_for_to_infinitive():
    rows = _rows("we went to the beach for enjoy our holiday.")
    assert ("for", "to", "FOR_TO_INFINITIVE") in rows


def test_since_for_duration():
    rows = _rows("we haven't visited there since many months.")
    assert ("since many months", "for many months", "SINCE_FOR") in rows


def test_there_was_plural():
    rows = _rows("There was many people swimming.")
    assert ("was", "were", "THERE_WAS_PLURAL") in rows


def test_one_of_were():
    rows = _rows("One of my friends were taking photos.")
    assert ("were", "was", "ONE_OF_WAS") in rows


def test_others_was():
    rows = _rows("the others was eating snacks.")
    assert ("was", "were", "PLURAL_WAS") in rows


def test_sometime_split():
    rows = _rows("After sometime, we decided to buy ice cream.")
    assert ("sometime", "some time", "SOMETIME_SPLIT") in rows


def test_dont_past_form():
    rows = _rows("he don't answered my phone.")
    assert ("answered", "answer", "DONT_PAST_FORM") in rows
    assert ("don't", "didn't", "SVA_DOESNT_DIDNT") in rows
def test_compound_subject_sva():
    rows = _rows("me and my friends goes to the beach.")
    assert ("goes", "go", "SVA_COMPOUND") in rows
    rows = _rows("my brother and I plays football.")
    assert ("plays", "play", "SVA_COMPOUND") in rows
from pipeline.aggregator import _remove_doubling


def test_remove_doubling_collapses_adjacent_rewrite():
    cands = [
        {"start": 0, "end": 5, "wrong": "There", "correct": "There were",
         "confidence": 0.9},
        {"start": 6, "end": 9, "wrong": "was", "correct": "were",
         "confidence": 0.95},
    ]
    out = _remove_doubling(cands)
    assert len(out) == 1
    assert out[0]["wrong"] == "was" and out[0]["correct"] == "were"
def test_lets_contraction_imperative():
    rows = _rows("Lets go to the park.")
    assert ("Lets", "Let's", "LETS_CONTRACTION") in rows
    # "go" must NOT be pasted inside the imperative
    assert not any(r[0] == "go" for r in rows)


def test_narrative_past_respects_imperative():
    from pipeline.master import check_master
    r = check_master("Yesterday it rained. Lets go to the park.", use_ai=False)
    assert "lets went" not in r["corrected_text"].lower()


def test_contractions_no_fp_on_3sg_lets():
    rows = _rows("She lets the dog out every morning.")
    assert not any(r[2] == "LETS_CONTRACTION" for r in rows)


def test_possessive_apostrophe_kinship():
    rows = _rows("my uncle house is big and my father car is red.")
    assert ("uncle", "uncle's", "POSSESSIVE_APOSTROPHE") in rows
    assert ("father", "father's", "POSSESSIVE_APOSTROPHE") in rows


def test_possessive_apostrophe_no_fp_compounds():
    rows = _rows("His mother tongue is Hindi and she told her friend group about it.")
    assert not any(r[2] == "POSSESSIVE_APOSTROPHE" for r in rows)


def test_possessive_apostrophe_no_fp_verb_head():
    rows = _rows("I saw my brother play cricket.")
    assert not any(r[2] == "POSSESSIVE_APOSTROPHE" for r in rows)


def test_present_perfect_duration_no_fp():
    from pipeline.master import check_master
    for t in ("I have worked here since last year.",
              "We have been friends since childhood.",
              "I lived in Chennai for two years."):
        r = check_master(t, use_ai=False)
        assert r["corrected_text"] == t, f"FP on {t!r}: {r['corrected_text']}"


def test_tense_past_marker_backward_not_duration():
    rows = _rows("I have known him until last year and I go to school. my uncle house is big")
    assert not any(r[2] == "TENSE_PAST_MARKER" for r in rows)
    assert ("uncle", "uncle's", "POSSESSIVE_APOSTROPHE") in rows


def test_tense_consistency_marker_after_verb():
    rows = _rows("She has bought it yesterday.")
    assert ("has bought", "bought", "TENSE_CONSISTENCY") in rows


def test_possessive_no_fp_on_conjunction():
    rows = _rows("He called his brother and asked him to search the room.")
    assert not any(r[2] == "POSSESSIVE_APOSTROPHE" for r in rows)


def test_there_was_plural_known_noun():
    rows = _rows("There was sandwiches on the table. There was problems with it.")
    assert rows.count(("was", "were", "THERE_WAS_PLURAL")) >= 2


def test_there_was_plural_no_fp_singulars():
    rows = _rows("There was a bus waiting. There was glass on the floor. "
                 "There was news on TV.")
    assert not any(r[2] == "THERE_WAS_PLURAL" for r in rows)


def test_there_was_plural_lots_of():
    rows = _rows("There was lots of people at the market.")
    assert ("was", "were", "THERE_WAS_PLURAL") in rows


def test_dont_agreement_3sg():
    rows = _rows("my brother don't want to come and he don't like it.")
    assert rows.count(("don't", "doesn't", "DONT_AGREEMENT")) == 2


def test_dont_agreement_no_fp_plural():
    rows = _rows("My friends don't want to come.")
    assert not any(r[2] == "DONT_AGREEMENT" for r in rows)
