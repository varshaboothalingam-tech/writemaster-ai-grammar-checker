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
