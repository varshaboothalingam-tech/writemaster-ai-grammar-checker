"""Regression: every error in the user-reported library paragraph must be
caught offline (no AI), and the previously-broken rewrites must stay fixed
("a short break" must not become "a short broke", "calls" must not become
"callsed")."""
import pytest

from pipeline.master import check_master

TEXT = (
    "Yesterday, my classmates and I goes to the college library to prepare for our final exam. "
    "We was planning to study for three hours, but one of my friends forgetted to bring his notes. "
    "He don't knew where he kept them, so he calls his brother and ask him to search the room. "
    "While we was waiting, another student come into the library and starts talking very loudly. "
    "The librarian tell him to keep quiet because several students was trying to concentrate. "
    "After some time, my friend finded his notes inside a old bag. "
    "We started studying, but I couldn't understood some of the difficult topics. "
    "My friend explain them to me and give me a useful example. "
    "Later, we takes a short break and went to the cafeteria. "
    "I ordered a sandwich and an orange juice, but there wasn't many tables available. "
    "We finally found a empty table near the window and sat there. "
    "One of my friends said that he have never seen such a crowded cafeteria before. "
    "After lunch, we returned to the library and continued our preparation. "
    "By the evening, everyone were tired, but we was satisfied because we had completed most of our revision. "
    "Before leaving, we promised that we will meet again tomorrow and finish the remaining chapters."
)

# rule_id -> the (wrong -> correct) fixes that must be present.
EXPECTED = {
    # past-narrative "goes" may be rewritten to either the SVA base ("go") or
    # the past tense ("went"); both are correct for the target sentence.
    "goes": {"go", "went"},
    "calls": {"called"},
    "ask": {"asked"},
    "come": {"came"},
    "starts": {"started"},
    "was": {"were"},
    "understood": {"understand"},
    "explain": {"explained"},
    "give": {"gave"},
    "wasn't": {"weren't"},
    "will": {"would"},
    "forgetted": {"forgot"},
    "finded": {"found"},
    "don't": {"doesn't"},
    "knew": {"know"},
}

FORBIDDEN = {
    # narrative rules must never rewrite the noun "break" / the 3sg "calls".
    "break": {"broke"},
    "calls": {"callsed"},
}


def _fixes_by_wrong(res):
    by_word = {}
    for e in res["errors"]:
        by_word.setdefault(e["wrong"], set()).add(e["correct"])
    return by_word


def test_repro_paragraph_finds_all_fixes():
    res = check_master(TEXT, use_ai=False)
    by_word = _fixes_by_wrong(res)
    for wrong, corrects in EXPECTED.items():
        assert wrong in by_word, f"missing fix for '{wrong}'"
        assert by_word[wrong] & corrects, \
            f"'{wrong}' got {sorted(by_word[wrong])}, wanted one of {sorted(corrects)}"


def test_repro_paragraph_no_bad_rewrites():
    res = check_master(TEXT, use_ai=False)
    by_word = _fixes_by_wrong(res)
    for wrong, bad in FORBIDDEN.items():
        if wrong in by_word:
            assert not (by_word[wrong] & bad), \
                f"'{wrong}' must not become one of {sorted(bad)}"


@pytest.mark.parametrize("text", [
    "I could not understand the lesson because it was difficult.",
    "I could not sleep well last night because of the noise.",
    "She said she would come and help us tomorrow.",
    "We will meet again tomorrow and finish the project.",
    "Every morning I go to school and play with my friends.",
    "He called his brother and asked him to search the room.",
])
def test_no_false_positives(text):
    res = check_master(text, use_ai=False)
    assert res["errors"] == [], f"unexpected fixes: {res['errors']}"