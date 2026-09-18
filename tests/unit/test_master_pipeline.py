"""Master pipeline + consensus engine tests (Phases 2-8).

Covers the canonical schema, the AGREED / AI_ONLY / LOCAL_ONLY taxonomy and
the AI-authoritative vs discovery-mode contracts.
"""
import json

import pytest

from pipeline.master import check_master
from pipeline.schema import (
    normalize_category, severity_for, consensus_tag, from_candidate,
    AUTO_APPLY_ALWAYS_CATEGORIES, NEVER_AUTO_APPLY_CATEGORIES,
)


def _fake_gemini(analysis, verify='{"decision": "accept", "confidence": 0.98, "reason": "safe"}'):
    def raw_call(prompt):
        if "Text to analyze" in prompt:
            return "```json\n" + json.dumps(analysis) + "\n```"
        return verify
    return raw_call


class TestSchema:
    def test_normalize_category_variants(self):
        assert normalize_category("SUBJECT_VERB_AGREEMENT") == "grammar"
        assert normalize_category("subject_verb") == "grammar"
        assert normalize_category("sva") == "grammar"
        assert normalize_category("SPELLING") == "spelling"
        assert normalize_category("word-choice") == "semantic"
        assert normalize_category("word choice") == "semantic"
        assert normalize_category("time-tense") == "tense"
        assert normalize_category("unknown-thing") == "grammar"

    def test_severity_rules(self):
        assert severity_for(0.5, "grammar") == "info"
        assert severity_for(0.6, "grammar") == "warning"
        assert severity_for(0.8, "grammar") == "error"
        assert severity_for(0.97, "spelling") == "error"
        assert severity_for(0.5, "spelling") == "warning"
        assert severity_for(0.9, "style") == "error"
        assert severity_for(0.9, "semantic") == "error"

    def test_from_candidate_upgrades_legacy(self):
        c = from_candidate(
            {"wrong": "go", "correct": "goes", "type": "subject_verb",
             "start": 4, "end": 6, "confidence": 0.9, "rule_id": "SVA_3SG"},
            sentence_id=0, index=0)
        assert c["original"] == "go"
        assert c["correction"] == "goes"
        assert c["category"] == "grammar"
        assert c["subcategory"] == "SVA_3SG"
        assert c["severity"] == "error"
        assert c["source"] == ["rule"]
        assert c["evidence"] == []

    def test_consensus_tag_taxonomy(self):
        # source list drives the label ("ai" present -> AI side, others -> local)
        assert consensus_tag({"start": 0, "end": 2, "confidence": 0.9,
                              "source": ["ai"]}, True) == "AI_ONLY"
        assert consensus_tag({"start": 0, "end": 2, "confidence": 0.9,
                              "source": ["rule"]}, True) == "AGREED"
        assert consensus_tag({"start": 0, "end": 2, "confidence": 0.9,
                              "source": ["rule"]}, False) == "LOCAL_ONLY"
        assert consensus_tag({"start": 0, "end": 2, "confidence": 0.9,
                              "source": ["ai", "rule"]}, False) == "AGREED"
        assert consensus_tag({"start": 0, "end": 2, "confidence": 0.9,
                              "source": []}, False) == "UNKNOWN"

    def test_never_auto_apply_excludes_real_errors(self):
        assert "style" in NEVER_AUTO_APPLY_CATEGORIES
        assert "semantic" not in NEVER_AUTO_APPLY_CATEGORIES
        assert "spelling" in AUTO_APPLY_ALWAYS_CATEGORIES


class TestDiscoveryMode:
    def test_agreed_and_ai_only_in_ai_mode(self):
        analysis = {
            "original_text": "She go to school everyday.",
            "corrected_text": "She goes to school every day.",
            "errors": [
                {"wrong": "go", "correct": "goes", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.97},
                {"wrong": "everyday", "correct": "every day", "type": "spelling",
                 "explanation": "typo", "confidence": 0.9},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        raw = _fake_gemini(analysis)
        res = check_master("She go to school everyday.", use_ai=True, raw_call=raw,
                           report_local_only=True)
        tags = {e["consensus"] for e in res["errors"]}
        assert "AGREED" in tags   # go -> goes is both AI and rule
        assert "AI_ONLY" in tags  # everyday -> every day is AI-only
        assert res["corrected_text"] == "She goes to school every day."
        assert res["consensus"]["agreed"] >= 1
        assert res["consensus"]["ai_only"] >= 1

    def test_ai_authoritative_by_default_no_local_only(self):
        analysis = {
            "original_text": "She go to school everyday.",
            "corrected_text": "She goes to school every day.",
            "errors": [
                {"wrong": "go", "correct": "goes", "type": "subject_verb_agreement",
                 "explanation": "SVA", "confidence": 0.97},
            ],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        raw = _fake_gemini(analysis)
        res = check_master("She go to school everyday.", use_ai=True, raw_call=raw)
        assert all(e["consensus"] != "LOCAL_ONLY" for e in res["errors"])
        assert res["errors"][0]["consensus"] == "AGREED"

    def test_reject_drops_everything_in_discovery_mode_too(self):
        analysis = {
            "original_text": "I like apple.",
            "corrected_text": "I like apples.",
            "errors": [{"wrong": "apple", "correct": "apples", "type": "number",
                        "explanation": "plural", "confidence": 0.9}],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        verify = '{"decision": "reject", "confidence": 0.9, "reason": "original correct"}'
        raw = _fake_gemini(analysis, verify=verify)
        res = check_master("I like apple.", use_ai=True, raw_call=raw,
                           report_local_only=True)
        assert res["errors"] == []
        assert res["corrected_text"] == "I like apple."
        assert res["meta"]["verification"]["decision"] == "reject"

    def test_missing_past_tense_ai_only(self):
        # "Yesterday, I go" — either the local marker-first rule (AGREED) or the
        # AI alone (AI_ONLY) must surface the past-tense fix.
        analysis = {
            "original_text": "Yesterday, I go to the market.",
            "corrected_text": "Yesterday, I went to the market.",
            "errors": [{"wrong": "go", "correct": "went", "type": "tense",
                        "explanation": "past marker", "confidence": 0.92}],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        }
        raw = _fake_gemini(analysis)
        res = check_master("Yesterday, I go to the market.", use_ai=True, raw_call=raw)
        assert any(e["correct"] == "went"
                   and e["consensus"] in ("AI_ONLY", "AGREED")
                   for e in res["errors"])

    def test_quality_block(self):
        raw = _fake_gemini({
            "original_text": "She go to school.",
            "corrected_text": "She goes to school.",
            "errors": [{"wrong": "go", "correct": "goes", "type": "subject_verb",
                        "explanation": "SVA", "confidence": 0.97}],
            "grammar_status": "errors_found",
            "meaning_preserved": True,
        })
        res = check_master("She go to school.", use_ai=True, raw_call=raw)
        assert res["quality"]["ai_validated"] is True
        assert res["schema"] == "v2_error_object"
        assert res["success"] is True