"""Unit tests for high-confidence rule detectors (positive, negative, edge).

Covers the new QUANT_PRON_SVA and PLURAL_DOESNT rules as well as a
representative sample of other rule families to confirm pos/neg/edge
behaviour. Two rules live in unified_pipeline (DIDNT_PAST_FORM,
DOUBLE_NEGATIVE) and are tested at the check_v4 pipeline level.
"""
import pytest

from high_confidence_rules import HighConfidenceDetector
from new_pipeline import check_v4

HC = HighConfidenceDetector()


def _ids(text, doc=None):
    return [c.rule_id for c in HC.detect(text, doc)]


def _has(text, rule_prefix, doc=None):
    return any(r.startswith(rule_prefix) for r in _ids(text, doc))


def _v4_has(text, rule_id):
    ids = [e["rule_id"] for e in check_v4(text)["errors"]]
    return rule_id in ids


# ─── QUANTIFIER PRONOUN SVA ─────────────────────────────────────────────────

def test_quant_pron_sva_pos():
    assert _has("Some of them was missing.", "DATA_RULE_QUANT_PRON_SVA")

@pytest.mark.parametrize("text", [
    "Many of us are ready.",
    "Each of them was prepared.",
])
def test_quant_pron_sva_neg(text):
    assert not _has(text, "DATA_RULE_QUANT_PRON_SVA")


# ─── PLURAL DOESN'T ─────────────────────────────────────────────────────────

def test_plural_doesnt_pos():
    assert _has("They doesn't care.", "DATA_RULE_PLURAL_DOESNT")

def test_plural_doesnt_edge():
    assert _has("We doesn't care.", "DATA_RULE_PLURAL_DOESNT")

@pytest.mark.parametrize("text", [
    "She doesn't care.",
    "He doesn't care.",
])
def test_plural_doesnt_neg(text):
    assert not _has(text, "DATA_RULE_PLURAL_DOESNT")


# ─── CAPITALIZATION ─────────────────────────────────────────────────────────

def test_capital_i_pos():
    assert _has("i like dogs.", "DATA_RULE_CAPITAL_I")

def test_capital_i_neg():
    assert not _has("I like dogs.", "DATA_RULE_CAPITAL_I")

def test_sent_start_pos():
    assert _has("she went home.", "DATA_RULE_SENT_START")

def test_sent_start_neg():
    assert not _has("She went home.", "DATA_RULE_SENT_START")


# ─── VERB FORM (high-confidence rules) ──────────────────────────────────────

def test_modal_of_pos():
    assert _has("He could of helped.", "DATA_RULE_MODAL_OF_HAVE")

def test_modal_of_neg():
    assert not _has("He could have helped.", "DATA_RULE_MODAL_OF_HAVE")


def test_didnt_past_v4_pos():
    assert _v4_has("He did not went to school.", "DIDNT_PAST_FORM")

def test_didnt_past_v4_neg():
    assert not _v4_has("He did not go to school.", "DIDNT_PAST_FORM")


def test_double_negative_v4_pos():
    assert _v4_has("I didn't see nothing.", "DOUBLE_NEGATIVE")


# ─── COLLECTIVE / GROUP SVA ────────────────────────────────────────────────

def test_collective_sva_pos():
    assert _has("The team are working hard.", "DATA_RULE_COLLECTIVE_SVA")

def test_group_of_sva_pos():
    assert _has("The group of students were absent.", "DATA_RULE_GROUP_OF_SVA")

def test_group_of_sva_neg():
    assert not _has("The group of students was absent.", "DATA_RULE_GROUP_OF_SVA")


# ─── MAKE/DO, ADJ/ADVERB, WH DO-SUPPORT ────────────────────────────────────

def test_make_do_pos():
    assert _has("She did a decision.", "DATA_RULE_MAKE_DO")

def test_adj_as_adverb_pos():
    assert _has("She answered polite.", "DATA_RULE_ADJ_AS_ADVERB")

def test_wh_do_support_pos():
    assert _has("Why he left so early?", "DATA_RULE_WH_DO_SUPPORT")

def test_cant_hardly_pos():
    assert _has("He can't hardly wait.", "DATA_RULE_CANT_HARDLY")


# ─── EDGE: empty / whitespace / acronyms ───────────────────────────────────

@pytest.mark.parametrize("text", ["", "   ", "OK.", "React", "pH"])
def test_edge_no_crash(text):
    HC.detect(text)   # must not raise