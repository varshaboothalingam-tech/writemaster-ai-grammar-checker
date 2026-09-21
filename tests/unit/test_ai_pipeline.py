"""AI-first pipeline tests.

Hermetic: Gemini is replaced by a fake ``raw_call``; everything runs offline
(AI_PROVIDER=none, no network). spaCy/v4 may or may not be installed — the
rule detector must stand alone.
"""
import json

import pytest

from pipeline.ai_core import check_ai_text, detect_rules
from pipeline.aggregator import (
    aggregate, apply_corrections, relocate_candidates, _relocate_one)
from pipeline import ai_analyzer, ai_verifier
from pipeline.feedback import FeedbackStore


# --------------------------------------------------------------------------
# Offline rule path (always available, most conservative)
# --------------------------------------------------------------------------

class TestOfflineCorrections:
    @pytest.mark.parametrize("text,expected", [
        ("She go to school evry day.", "She goes to school every day."),
        ("They is went homeeee.", "They went home."),
        ("He don't like coffee.", "He doesn't like coffee."),
        ("Their going to the store tommorow.", "They're going to the store tomorrow."),
        ("She have two brothers.", "She has two brothers."),
        ("I go to the market yesterday.", "I went to the market yesterday."),
        ("He has went there.", "He has gone there."),
        ("They have ate dinner.", "They have eaten dinner."),
        ("Me and him went to the park.", "He and I went to the park."),
    ])
    def test_corrects(self, text, expected):
        res = check_ai_text(text, use_ai=False)
        assert res["grammar_status"] == "errors_found"
        assert res["corrected_text"] == expected

    @pytest.mark.parametrize("text", [
        "I like apples.",
        "She goes to school every day.",
        "The cat sat on the mat and purred loudly.",
        "We are going to the park tomorrow.",
        "Some information was incorrect.",
    ])
    def test_clean_text_yields_no_errors(self, text):
        res = check_ai_text(text, use_ai=False)
        assert res["errors"] == []
        assert res["corrected_text"] == text
        assert res["grammar_status"] == "correct"

    def test_empty_text(self):
        res = check_ai_text("", use_ai=False)
        assert res["errors"] == []
        assert res["corrected_text"] == ""
        assert res["meta"]["word_count"] == 0

    def test_span_consistency(self):
        text = "Their going to the store tommorow."
        res = check_ai_text(text, use_ai=False)
        for e in res["errors"]:
            segment = text[e["start"]:e["end"]]
            assert segment.lower() == e["wrong"].lower()
            assert e["start"] >= 0 and e["end"] <= len(text)

    def test_evry_typo_corrects_to_every_not_very(self):
        res = check_ai_text("She go to school evry day.", use_ai=False)
        spellings = [e for e in res["errors"] if e["type"] == "spelling"
                     and e["wrong"] == "evry"]
        assert spellings and spellings[0]["correct"] == "every"

    def test_corrected_text_matches_errors(self):
        text = "They is went homeeee."
        res = check_ai_text(text, use_ai=False)
        rebuilt = text
        for e in sorted(res["errors"], key=lambda x: -x["start"]):
            rebuilt = rebuilt[:e["start"]] + e["correct"] + rebuilt[e["end"]:]
        assert rebuilt.lower() == res["corrected_text"].lower()


# --------------------------------------------------------------------------
# Aggregator: relocation guards (the v4 imported-span bug)
# --------------------------------------------------------------------------

class TestRelocation:
    def test_stray_v4_span_is_relocated(self):
        text = "Their going to the store tommorow."
        cands = [{
            "wrong": "Their going", "correct": "they're going",
            "type": "word_choice", "start": 0, "end": 17,
            "confidence": 0.91, "source": "v4",
        }]
        fixed = relocate_candidates(cands, text)
        assert fixed[0]["start"] == 0
        assert fixed[0]["end"] == 11
        app = apply_corrections(text, fixed)
        assert app["corrected_text"] == "They're going to the store tommorow."
        for c in app["changes"]:
            assert text[c["start"]:c["end"]].lower() == "their going"

    def test_unlocatable_candidate_is_dropped(self):
        text = "It works fine."
        cands = [{
            "wrong": "zzz missing", "correct": "replacement",
            "start": 10, "end": 20, "confidence": 0.9, "source": "x",
        }]
        assert relocate_candidates(cands, text) == []

    def test_relocate_one_no_partial_word_hit(self):
        text = "We go to the store tomorrow."
        assert _relocate_one(text, "to") == (6, 8)  # "to" (not inside "tomorrow")
        assert _relocate_one(text, "tomorrow") == (19, 27)

    def test_apply_skips_bad_span_instead_of_corrupting(self):
        text = "Their going to the store tommorow."
        cands = [{
            "wrong": "going", "correct": "went",
            "start": 5, "end": 10, "confidence": 0.95, "source": "x",
        }]
        # span (5,10) does NOT cover "going" (6,11) in original casing? it does —
        # force a deliberately wrong span and confirm relocation fixes it.
        cands[0]["start"], cands[0]["end"] = 0, 10
        app = apply_corrections(text, cands)
        # wrong="going" relocates to (6,11), so "went" replaces "going"
        assert app["corrected_text"] == "Their went to the store tommorow."


# --------------------------------------------------------------------------
# Mock-Gemini analysis + verification
# --------------------------------------------------------------------------

def _fake_gemini(analysis, verify='{"decision": "accept", "confidence": 0.98, "reason": "safe"}'):
    emitted = {}

    def raw_call(prompt):
        if "Text to analyze" in prompt:
            emitted["last_analysis"] = prompt
            return "```json\n" + json.dumps(analysis) + "\n```"
        emitted["last_verify"] = prompt
        return verify
    return raw_call, emitted


class TestAiPath:
    def test_analysis_errors_accepted_and_applied(self):
        analysis = {
            "original_text": "She go to school evry day.",
            "corrected_text": "She goes to school every day.",
            "errors": [
                {"wrong": "go", "correct": "goes", "type": "subject_verb",
                 "explanation": "SVA", "confidence": 0.95},
                {"wrong": "evry", "correct": "every", "type": "spelling",
                 "explanation": "typo", "confidence": 0.98},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        raw_call, _ = _fake_gemini(analysis)
        res = check_ai_text("She go to school evry day.", use_ai=True, raw_call=raw_call)
        assert res["meta"]["ai_used"] is True
        assert res["corrected_text"] == "She goes to school every day."
        assert res["grammar_status"] == "errors_found"
        assert res["meta"]["verification"]["decision"] == "accept"
        for e in res["errors"]:
            seg = "She go to school evry day."[e["start"]:e["end"]]
            assert seg.lower() == e["wrong"].lower()

    def test_uncertain_verification_caps_confidence(self):
        analysis = {
            "original_text": "He don't like coffee.",
            "corrected_text": "He doesn't like coffee.",
            "errors": [
                {"wrong": "don't", "correct": "doesn't", "type": "subject_verb",
                 "explanation": "SVA", "confidence": 0.9},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        verify = '{"decision": "uncertain", "confidence": 0.5, "reason": "ambiguous"}'
        raw_call, _ = _fake_gemini(analysis, verify=verify)
        res = check_ai_text("He don't like coffee.", use_ai=True, raw_call=raw_call)
        assert res["meta"]["verification"]["decision"] == "uncertain"
        assert all(e["confidence"] <= 0.6 for e in res["errors"])

    def test_reject_verification_drops_errors(self):
        analysis = {
            "original_text": "I go to the market yesterday.",
            "corrected_text": "I went to the market yesterday.",
            "errors": [
                {"wrong": "go", "correct": "goes", "type": "subject_verb",
                 "explanation": "", "confidence": 0.95},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        verify = '{"decision": "reject", "confidence": 0.9, "reason": "original correct"}'
        raw_call, _ = _fake_gemini(analysis, verify=verify)
        res = check_ai_text("I go to the market yesterday.", use_ai=True, raw_call=raw_call)
        assert res["errors"] == []
        assert res["corrected_text"] == "I go to the market yesterday."

    def test_clean_text_skips_verify_call(self):
        analysis = {
            "original_text": "I like apples.",
            "corrected_text": "I like apples.",
            "errors": [],
            "grammar_status": "correct",
            "meaning_preserved": True,
        }
        raw_call, emitted = _fake_gemini(analysis)
        res = check_ai_text("I like apples.", use_ai=True, raw_call=raw_call)
        assert res["errors"] == []
        assert res["meta"]["ai_used"] is True
        assert "last_verify" not in emitted

    def test_model_offsets_never_trusted(self):
        # Model reports bogus offsets; they must be recomputed locally.
        analysis = {
            "original_text": "She go to school evry day.",
            "corrected_text": "She goes to school every day.",
            "errors": [
                {"wrong": "go", "correct": "goes", "type": "subject_verb",
                 "start": 999, "end": 1001, "confidence": 0.95, "explanation": ""},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        raw_call, _ = _fake_gemini(analysis)
        res = check_ai_text("She go to school evry day.", use_ai=True, raw_call=raw_call)
        go_errors = [e for e in res["errors"]
                     if e["wrong"] == "go" and e["correct"] == "goes"]
        assert go_errors, res["errors"]
        e = go_errors[0]
        assert e["start"] == 4 and e["end"] == 6
        # the model only reported 'go' (bogus offsets); local rules (spelling)
        # may still surface other high-confidence fixes such as evry -> every.
        assert any(ev["wrong"] == "evry" for ev in res["errors"])


# --------------------------------------------------------------------------
# Feedback quality gate
# --------------------------------------------------------------------------

class TestFeedback:
    def test_valid_record_written(self, tmp_path):
        store = FeedbackStore(base_dir=str(tmp_path))
        report = store.add("She go to school", "She goes to school",
                           "go", "goes", "subject_verb", True)
        assert report["success"] is True
        assert store.count() == 1
        rec = store.recent(1)[0]
        assert rec["original_text"] == "She go to school"
        assert rec["accepted"] == 1
        assert "timestamp" in rec

    def test_duplicate_rejected(self, tmp_path):
        store = FeedbackStore(base_dir=str(tmp_path))
        store.add("A", "B", "aa", "bb", "spelling", True)
        dup = store.add("A", "B", "aa", "bb", "spelling", True)
        assert dup["success"] is False
        assert "duplicate" in dup["rejected"]

    def test_missing_fields_rejected(self, tmp_path):
        store = FeedbackStore(base_dir=str(tmp_path))
        report = store.add("", "", "", "", "grammar", True)
        assert report["success"] is False
        assert "empty" in report["rejected"].lower() or "missing" in report["rejected"].lower()

    def test_identical_wrong_correct_rejected(self, tmp_path):
        store = FeedbackStore(base_dir=str(tmp_path))
        report = store.add("x", "y", "same", "same", "spelling", True)
        assert report["success"] is False


# --------------------------------------------------------------------------
# Module-level import sanity (independent of spaCy availability)
# --------------------------------------------------------------------------

class TestImports:
    def test_rule_detector_standalone(self):
        rules = detect_rules("She go to school.")
        assert any(r["wrong"] == "go" and r["correct"] == "goes" for r in rules)

    def test_aggregate_dedups_sources(self):
        base = {"start": 4, "end": 6, "correct": "goes", "confidence": 0.9,
                "type": "subject_verb"}
        merged = aggregate(
            [dict(base, wrong="go", source="rule")],
            [dict(base, wrong="go", source="fast", id="x")])
        assert len(merged) == 1
        assert merged[0]["sources"] == ["fast", "rule"]


# --------------------------------------------------------------------------
# Local Ollama provider awareness (no API key needed)
# --------------------------------------------------------------------------

class TestOllamaAwareConfig:
    def _no_ai_env(self, monkeypatch):
        for k in ("AI_PROVIDER", "GC_AI_PROVIDERS", "AI_API_KEY",
                  "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
                  "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AI_MODEL",
                  "OLLAMA_MODEL"):
            monkeypatch.delenv(k, raising=False)
        # Never let an enabled singleton leak into the rest of the suite
        # (check_v4 uses get_validator() and would hit real Ollama).
        monkeypatch.setattr("ai_validator._instance", None)

    def test_ai_configured_autodetect_ollama(self, monkeypatch):
        self._no_ai_env(monkeypatch)
        monkeypatch.setattr("ai_validator._ollama_available", lambda: True)
        assert ai_analyzer.ai_key_configured() is True
        assert ai_analyzer._provider_callable() is not None

    def test_ai_disabled_with_explicit_none_even_if_ollama_up(self, monkeypatch):
        self._no_ai_env(monkeypatch)
        monkeypatch.setenv("AI_PROVIDER", "none")
        monkeypatch.setattr("ai_validator._ollama_available", lambda: True)
        assert ai_analyzer.ai_key_configured() is False

    def test_ai_configured_with_explicit_ollama_provider(self, monkeypatch):
        self._no_ai_env(monkeypatch)
        monkeypatch.setenv("AI_PROVIDER", "ollama")
        assert ai_analyzer.ai_key_configured() is True

    def test_ai_configured_with_remote_key(self, monkeypatch):
        self._no_ai_env(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        assert ai_analyzer.ai_key_configured() is True

    def test_ai_yield_from_local_ollama_via_pipeline(self, monkeypatch):
        """End-to-end: ollama auto-detected -> full AI judge path is reachable."""
        self._no_ai_env(monkeypatch)
        monkeypatch.setattr("ai_validator._ollama_available", lambda: True)
        from ai_validator import get_validator
        v = get_validator()
        assert v.is_enabled()
        call = ai_analyzer._provider_callable()
        assert call is not None
        # No network calls happen here; we only assert the transport is wired.
        assert any(p == "ollama" for p in v.providers)

def test_multi_array_json_repair():
    from pipeline.ai_analyzer import _extract_json
    raw = ('{"errors": [{"start": 0, "end": 4, "original": "goes", "replacement": "went", '
           '"category": "grammar", "subcategory": "tense", "confidence": 0.99, '
           '"explanation": "x"}], [{"start": 5, "end": 9, "original": "goes", '
           '"replacement": "went", "category": "grammar", "subcategory": "tense", '
           '"confidence": 0.99, "explanation": "y"}]}')
    d = _extract_json(raw)
    assert d and len(d.get("errors", [])) == 2
