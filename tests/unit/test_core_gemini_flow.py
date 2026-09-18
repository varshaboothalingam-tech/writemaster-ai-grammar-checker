"""Core Gemini + local-NLP flow tests (the final-phase build-out).

Covers the requirements checklist:

    * consensus engine Cases A-D          (pipeline/consensus.py)
    * spec §21 test sentences              (offline local-NLP path)
    * repeated-word mis-location §7/§11    (context disambiguation + nearest
      relocation) — the "While we were waiting / food were" mixed paragraph
    * the §8 change-set groups            (verified_local / rejected_local /
      gemini_only)
    * statistics block §17
    * second full check + loop guard §15/§16   (pipeline/recheck.py)
    * missed-error corpus §18/§19         (datasets/missed_errors.jsonl)
"""
import json
import os

import pytest

from pipeline.master import check_master
from pipeline import ai_analyzer, aggregator, recheck as recheck_mod
from pipeline.ai_verifier import verify_changes
from pipeline.consensus import merge, AGREED, AI_ONLY, LOCAL_ONLY, CONFLICT


def _fake_gemini(analysis, verify='{"decision": "accept", "confidence": 0.98, "reason": "safe"}'):
    def raw_call(prompt):
        if "Text to analyze" in prompt:
            return "```json\n" + json.dumps(analysis) + "\n```"
        return verify
    return raw_call


def _noop_candidate(wrong, correct, start, end, source="rule", conf=0.9, category="grammar"):
    return {"wrong": wrong, "correct": correct, "type": category,
            "start": start, "end": end, "confidence": conf, "source": source,
            "sources": [source]}


class TestConsensusCases:
    """pipeline/consensus: Case A (AGREED), B (LOCAL_ONLY), C (AI_ONLY), D (CONFLICT)."""

    def test_case_a_agreed(self):
        local = [_noop_candidate("go", "goes", 4, 6, conf=0.9)]
        ai = [{"wrong": "go", "correct": "goes", "type": "subject_verb",
               "start": 4, "end": 6, "confidence": 0.97}]
        res = merge(local, ai, report_local_only=True)
        rec = res["records"][0]
        assert rec["_group"] == AGREED
        assert "ai" in rec["sources"] and "rule" in rec["sources"]
        assert rec["confidence"] == pytest.approx(0.97)
        assert res["statistics"]["agreed"] == 1

    def test_case_c_ai_only(self):
        local = [_noop_candidate("go", "goes", 4, 6, conf=0.9)]
        ai = [{"wrong": "everyday", "correct": "every day", "type": "spelling",
               "start": 13, "end": 21, "confidence": 0.9}]
        res = merge(local, ai, report_local_only=False)
        groups = {r["_group"] for r in res["records"]}
        assert groups == {AI_ONLY}
        assert res["statistics"]["ai_only"] == 1

    def test_case_b_local_only_discovery_mode(self):
        local = [_noop_candidate("go", "goes", 4, 6, conf=0.9)]
        report = merge(local, [], report_local_only=True)
        assert report["records"][0]["_group"] == LOCAL_ONLY
        strict = merge(local, [], report_local_only=False)
        assert strict["records"] == []
        assert strict["statistics"]["local_only"] == 0
        assert len(strict["rejected"]) == 1

    def test_case_d_conflict_gemini_wins(self):
        local = [_noop_candidate("go", "goes", 4, 6, conf=0.8)]
        ai = [{"wrong": "go", "correct": "went", "type": "tense",
               "start": 4, "end": 6, "confidence": 0.85}]
        res = merge(local, ai, report_local_only=True)
        winner = [r for r in res["records"] if r["_group"] == CONFLICT]
        assert len(winner) == 1
        assert winner[0]["correct"] == "went"
        assert res["statistics"]["conflicts"] == 1
        assert len(res["rejected"]) == 1

    def test_agree_sources_restricts_case_a(self):
        # v4-sourced local candidate agrees with AI -> still AI_ONLY, not AGREED.
        local = [_noop_candidate("go", "went", 13, 15,
                                 source="nlp_detector", conf=0.92)]
        ai = [{"wrong": "go", "correct": "went", "type": "tense",
               "start": 13, "end": 15, "confidence": 0.92}]
        res = merge(local, ai, report_local_only=True, agree_sources={"rule"})
        assert res["records"][0]["_group"] == AI_ONLY
        assert res["statistics"]["agreed"] == 0


class TestSpecSentences:
    """§21 — the eight test sentences must produce the expected corrections."""

    @pytest.mark.parametrize("bad,good", [
        ("She go to school every day.", "She goes to school every day."),
        ("She goes to school evry day.", "She goes to school every day."),
        ("They was playing cricket.", "They were playing cricket."),
        ("He has went there.", "He has gone there."),
        ("They have ate dinner.", "They have eaten dinner."),
        ("I go to the market yesterday.", "I went to the market yesterday."),
        ("I am go to school.", "I am going to school."),
    ])
    def test_offline_correction(self, bad, good):
        res = check_master(bad, use_ai=False)
        assert res["corrected_text"] == good, res["errors"]

    def test_never_change_a_correct_sentence(self):
        res = check_master("She is a boy.", use_ai=False)
        assert res["errors"] == []
        assert res["corrected_text"] == "She is a boy."
        assert res["grammar_status"] == "correct"


class TestRepeatedWords:
    """§7/§11 — repeated words must not snap to the first occurrence."""

    TEXT = ("Last Saturday, my family decide to visit a new restaurant that we had heard many good things about. "
            "We leaves our house around seven in the evening, but the traffic was heavier than we expected. "
            "When we arrived at the restaurant, there was only few tables available and the waiter ask us to wait for fifteen minutes. "
            "While we were waiting, my younger brother accidentally drop his phone on the floor and the screen get broken. "
            "After we finally got a table, we looked at the menu and ordered some dishes that we never tried before. "
            "The food were delicious, but one of the dishes was too spicy for my mother, so she don't eat it. "
            "The waiter apologized and bring her another dish without charging extra money. "
            "After dinner, we wanted to take some photographs, but my brother said that he has forgotten his phone at the table. "
            "We quickly went back and fortunately it was still there. "
            "On our way home, we talked about the restaurant and agreed that we will visit it again because everyone had enjoyed the evening very much. "
            "It was one of the most enjoyable evenings we have ever spend together.")

    def test_context_disambiguation_in_locate_errors(self):
        # Two 'were' occurrences; context must pin 'the food were', NOT
        # 'While we were waiting'.
        errs = ai_analyzer._locate_errors(self.TEXT, [
            {"wrong": "were", "correct": "was", "type": "grammar",
             "confidence": 0.9, "context": "The food were delicious"},
        ])
        assert len(errs) == 1
        found = self.TEXT[errs[0]["start"]:errs[0]["end"]]
        assert found == "were"
        assert "food were" in self.TEXT[errs[0]["start"] - 8:errs[0]["start"] + 4]
        assert errs[0]["start"] != self.TEXT.find("were")  # not the FIRST one

    def test_context_fallback_keeps_cursor_order(self):
        # Without context the old cursor-based (first-at-or-after) order holds.
        errs = ai_analyzer._locate_errors(self.TEXT, [
            {"wrong": "were", "correct": "was", "type": "grammar", "confidence": 0.9},
        ])
        assert errs[0]["start"] == self.TEXT.find("were")

    def test_nearest_occurrence_relocation(self):
        cands = aggregator.relocate_candidates(
            [{"wrong": "were", "correct": "was", "start": 500, "end": 504,
              "confidence": 0.9, "source": "x"}],
            "There were cats. The food were good.")
        assert cands[0]["start"] == 26  # nearest to the declared 500-504 -> 2nd

    def test_mixed_paragraph_ai_mode_no_new_errors(self):
        analysis = {
            "original_text": self.TEXT,
            "corrected_text": "...",
            "errors": [
                {"wrong": "decide", "correct": "decides", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.9, "context": "family decide"},
                {"wrong": "leaves", "correct": "leave", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.9, "context": "We leaves"},
                {"wrong": "was", "correct": "were", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.93, "context": "there was only"},
                {"wrong": "ask", "correct": "asks", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.9, "context": "waiter ask"},
                {"wrong": "were", "correct": "was", "type": "subject_verb_agreement",
                 "explanation": "mass-noun SVA", "confidence": 0.9, "context": "The food were"},
                {"wrong": "drop", "correct": "dropped", "type": "tense",
                 "explanation": "past narrative", "confidence": 0.85, "context": "brother accidentally drop"},
                {"wrong": "get", "correct": "got", "type": "tense",
                 "explanation": "past narrative", "confidence": 0.85, "context": "screen get"},
                {"wrong": "don't", "correct": "doesn't", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.94, "context": "she don't"},
                {"wrong": "bring", "correct": "brought", "type": "tense",
                 "explanation": "past narrative", "confidence": 0.88, "context": "apologized and bring"},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        res = check_master(self.TEXT, use_ai=True, raw_call=_fake_gemini(analysis),
                           max_passes=1)
        c = res["corrected_text"]
        # repeated-word fix: 'While we were waiting' must NOT be corrupted
        assert "While we were waiting" in c
        assert "While we was waiting" not in c
        # and the intended occurrence IS corrected
        assert "The food was delicious" in c
        assert "she doesn't eat it" in c
        assert "the screen got broken" in c
        # the AI_ONLY were->was is reported at the food occurrence
        were_fix = [e for e in res["errors"]
                    if e["wrong"] == "were" and e["correct"] == "was"]
        assert were_fix and "food were" in self.TEXT[
            were_fix[0]["start"] - 8:were_fix[0]["start"] + 4]


class TestStatistics:
    def test_statistics_block_present_and_consistent(self):
        analysis = {
            "original_text": "She go to school. I very like music.",
            "corrected_text": "She goes to school. I really like music.",
            "errors": [
                {"wrong": "go", "correct": "goes", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.97},
                {"wrong": "very", "correct": "really", "type": "word_choice",
                 "explanation": "use 'really' before a verb", "confidence": 0.9},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        res = check_master("She go to school. I very like music.", use_ai=True,
                           raw_call=_fake_gemini(analysis))
        s = res["statistics"]
        assert s["final_errors"] == len(res["errors"]) == 2
        assert s["agreed"] >= 1
        assert s["ai_only"] >= 1
        assert s["local_candidates"] >= 1
        assert s["gemini_candidates"] == 2
        assert s["passes"] == 1
        g = res["meta"]["groups"]
        assert any(i["wrong"] == "go" for i in g["verified_local"])
        assert any(i["wrong"] == "very" for i in g["gemini_only"])


class TestSecondPass:
    def test_offline_ignores_max_passes(self):
        res = check_master("She go to school evry day.", use_ai=False, max_passes=3)
        assert res["meta"]["recheck"]["passes_used"] == 1

    def test_second_pass_catches_new_error(self):
        calls = {"n": 0}
        analysis_a = {
            "original_text": "She go to school evry day.",
            "corrected_text": "She goes to school evry day.",
            "errors": [{"wrong": "go", "correct": "goes", "type": "subject_verb",
                        "explanation": "SVA", "confidence": 0.97}],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        analysis_ab = {
            "original_text": "She goes to school evry day.",
            "corrected_text": "She goes to school every day.",
            "errors": [
                {"wrong": "go", "correct": "goes", "type": "subject_verb",
                 "explanation": "SVA", "confidence": 0.97},
                {"wrong": "evry", "correct": "every", "type": "spelling",
                 "explanation": "typo", "confidence": 0.92},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }

        def raw_call(prompt):
            if "Text to analyze" in prompt:
                calls["n"] += 1
                return "```json\n" + json.dumps(analysis_a if calls["n"] == 1 else analysis_ab) + "\n```"
            return '{"decision": "accept", "confidence": 0.98, "reason": "safe"}'

        res = check_master("She go to school evry day.", use_ai=True,
                           raw_call=raw_call, max_passes=3)
        assert res["corrected_text"] == "She goes to school every day."
        assert res["meta"]["recheck"]["passes_used"] >= 2
        assert any(e["wrong"] == "evry" and e["correct"] == "every"
                   for e in res["errors"])
        assert res["statistics"]["passes"] >= 2

    def test_recheck_loop_prevention(self):
        # corrected text ping-pongs A->B->A: the pipeline must STOP by the
        # time the hash repeats and never exceed the cap of 3.
        def make_pass(t):
            return {"meta": {"ai_used": True},
                    "corrected_text": "go to marketX" if t == "go to market" else "go to market",
                    "errors": [{"wrong": "go", "correct": "went",
                                "consensus": "AI_ONLY", "category": "tense",
                                "confidence": 0.9}]}

        first = {"corrected_text": "go to market", "errors": [],
                 "meta": {"ai_used": True}}
        rc = recheck_mod.recheck("go to market", first, make_pass, max_passes=3)
        assert rc["loop_detected"] is True  # hash repeated -> stopped
        assert rc["passes_used"] == 3       # but never exceeded the cap


class TestVerifyChangesGroups:
    def test_per_change_verdicts(self):
        def raw_call(prompt):
            return json.dumps({"verdicts": [
                {"wrong": "go", "correct": "goes", "decision": "accept"},
                {"wrong": "evry", "correct": "every", "decision": "reject"},
                {"wrong": "x", "correct": "y", "decision": "uncertain"},
            ]})

        changes = [{"wrong": "go", "correct": "goes"},
                   {"wrong": "evry", "correct": "every"},
                   {"wrong": "x", "correct": "y"}]
        out = verify_changes("She go evry.", changes, raw_call=raw_call)
        assert out["per_change"] is True
        assert [c["wrong"] for c in out["verified_local"]] == ["go"]
        assert [c["wrong"] for c in out["rejected_local"]] == ["evry"]
        assert [c["wrong"] for c in out["uncertain"]] == ["x"]

    def test_whole_set_fallback(self):
        def raw_call(prompt):
            return '{"decision": "reject", "confidence": 0.9, "reason": "fine"}'

        out = verify_changes("I like apple.", [{"wrong": "apple", "correct": "apples"}],
                             raw_call=raw_call)
        assert out["per_change"] is False
        assert out["rejected_local"][0]["wrong"] == "apple"


class TestMissedErrorsCorpus:
    def _ai_only_analysis(self):
        return {
            "original_text": "She go to school. I very like music.",
            "corrected_text": "She goes to school. I really like music.",
            "errors": [{"wrong": "very", "correct": "really",
                        "type": "word_choice",
                        "explanation": "use 'really' before a verb", "confidence": 0.9}],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }

    def test_ai_only_logged_to_dataset(self, tmp_path, monkeypatch):
        path = tmp_path / "missed_errors.jsonl"
        monkeypatch.setenv("MISSED_ERRORS_PATH", str(path))
        monkeypatch.setenv("COLLECT_MISSED_ERRORS", "1")
        res = check_master("She go to school. I very like music.", use_ai=True,
                           raw_call=_fake_gemini(self._ai_only_analysis()))
        assert res["statistics"]["missed_logged"] == 1
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 1
        rec = lines[0]
        assert rec["wrong"] == "very"
        assert rec["correct"] == "really"
        assert rec["category"] == "semantic"
        assert rec["original_text"] == "She go to school. I very like music."

    def test_logging_disabled_by_default(self, tmp_path, monkeypatch):
        path = tmp_path / "missed_errors.jsonl"
        monkeypatch.setenv("MISSED_ERRORS_PATH", str(path))
        monkeypatch.setenv("COLLECT_MISSED_ERRORS", "0")
        check_master("She go to school. I very like music.", use_ai=True,
                     raw_call=_fake_gemini(self._ai_only_analysis()))
        assert not path.exists()


class TestLongParagraph:
    """Long-paragraph guard (§7 UX): corrected_text must NOT be discarded for
    texts >= 200 chars. Regression for difflib.SequenceMatcher autojunk=True
    collapsing the meaning-guard character-identity ratio on long inputs."""

    _LONG = (
        "She go to school everyday and she dont like it. The boys was playing football. "
        "I have many informations about the project. The team are winning every match. "
        "The childrens are playing in the garden. We seen the movie last night. "
        "There is three things to remember. We should of taken the earlier train. "
        "The data shows that more peoples are using online shopping. Every one of the "
        "boxes need to be checked now. He recommended me to apply for the job. "
        "Her and me are going to the market. My friend he always arrive late on mondays. "
        "The weather is more colder today. The teacher give us a homework. "
        "She have been working since three years."
    )

    def test_long_paragraph_returns_corrected_text_offline(self):
        assert len(self._LONG) >= 200
        res = check_master(self._LONG, use_ai=False, include_v4_hints=False)
        assert res["success"]
        assert len(res["errors"]) > 0
        assert res["corrected_text"] != res["original_text"]
        assert all(w not in res["corrected_text"]
                   for w in ("She go to", "many informations", "more colder"))

    def test_long_paragraph_offline_with_v4_hints(self):
        res = check_master(self._LONG, use_ai=False, include_v4_hints=True)
        assert res["success"]
        assert len(res["errors"]) > 0
        assert res["corrected_text"] != res["original_text"]

    def test_meaning_guard_ratio_stable_on_long_text(self):
        from pipeline.master import _meaning_preserved_ok
        res = check_master(self._LONG, use_ai=False, include_v4_hints=False)
        ratio_ok = _meaning_preserved_ok(
            res["original_text"], res["corrected_text"], res["errors"])
        assert ratio_ok is True