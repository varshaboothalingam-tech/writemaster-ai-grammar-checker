"""Unit tests for the AI final-validation gate (ai_validator.py).

Covers the fail-safe policy matrix from the master prompt:
  valid accept / uncertain / wrong-correction / invalid category /
  invalid correction / malformed JSON / missing verdict / unreachable provider /
  permissive fallback / disabled pass-through.
"""
import json
import os

import pytest

from ai_validator import (ALLOWED_CATEGORIES, AIValidator, ValidationResult,
                          _local_category, _threshold)


class FakeAI(AIValidator):
    """AI subclass with a scripted, reachable response."""

    verdicts_json = ""

    def __init__(self, verdicts):
        super().__init__(providers=["openai"])
        os.environ["AI_API_KEY"] = "test"
        self._available = True
        self.verdicts_json = verdicts

    def available(self):
        return True

    def is_enabled(self):
        return True

    def _call(self, prompt):
        return self.verdicts_json


def _candidate(index=0, conf=0.95):
    return {"original": "was", "replacement": "were",
            "sentence": "Some of them was missing.", "confidence": conf}


def _verdict(index=0, is_error=True, corr_valid=True, category="grammar",
             confidence=0.98, replacement="were"):
    return {"index": index, "is_error": is_error,
            "correction_is_valid": corr_valid,
            "replacement": replacement, "category": category,
            "subcategory": "sva", "confidence": confidence,
            "reason": "ok"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setenv("GC_AI_THRESHOLD", "0.90")
    monkeypatch.setenv("GC_AI_STRICT", "1")


def test_valid_accept():
    ai = FakeAI(json.dumps([_verdict()]))
    out = ai.validate([_candidate()])
    assert out[0].is_error is True
    assert out[0].correction_is_valid is True
    assert out[0].validated is True
    assert out[0].ai_used is True


def test_low_confidence_reject():
    ai = FakeAI(json.dumps([_verdict(confidence=0.60)]))
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "threshold" in out[0].reason


def test_uncertain_reject():
    ai = FakeAI(json.dumps([_verdict(is_error=False)]))
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "valid" in out[0].reason


def test_wrong_correction_reject():
    ai = FakeAI(json.dumps([_verdict(corr_valid=False)]))
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "not valid" in out[0].reason


def test_invalid_category_reject():
    # Non-whitelisted category fails schema validation -> strict reject.
    ai = FakeAI(json.dumps([_verdict(category="mediocre")]))
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "rejected" in out[0].reason


def test_malformed_json_reject_strict():
    ai = FakeAI("this is not json at all")
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "malformed" in out[0].reason or "missing" in out[0].reason


def test_malformed_json_permissive_fallback(monkeypatch):
    monkeypatch.setenv("GC_AI_STRICT", "0")
    ai = FakeAI("this is not json at all")
    out = ai.validate([_candidate(conf=0.8)])
    assert out[0].is_error is True
    assert out[0].replacement == "were"
    assert out[0].validated is False


def test_missing_verdict_index_strict():
    # Response has verdict for index 1 only; candidate 0 gets none.
    ai = FakeAI(json.dumps([_verdict(index=1)]))
    out = ai.validate([_candidate(), _candidate()])
    assert out[0].is_error is False
    assert out[1].is_error is True


def test_unreachable_strict_rejects(monkeypatch):
    os.environ["AI_API_KEY"] = "test"
    ai = AIValidator(providers=["openai"])
    monkeypatch.setattr(ai, "available", lambda: False)
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "unreachable" in out[0].reason
    assert out[0].ai_used is False


def test_unreachable_permissive_falls_back(monkeypatch):
    monkeypatch.setenv("GC_AI_STRICT", "0")
    os.environ["AI_API_KEY"] = "test"
    ai = AIValidator(providers=["openai"])
    monkeypatch.setattr(ai, "available", lambda: False)
    out = ai.validate([_candidate()])
    assert out[0].is_error is True
    assert out[0].replacement == "were"
    assert out[0].validated is False


def test_disabled_passthrough_no_ai(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "none")
    os.environ.pop("AI_API_KEY", None)
    ai = AIValidator()
    assert ai.is_enabled() is False
    out = ai.validate([_candidate()])
    assert out[0].is_error is True
    assert out[0].replacement is not None
    assert out[0].validated is False
    assert out[0].ai_used is False


def test_threshold_env(monkeypatch):
    monkeypatch.setenv("GC_AI_THRESHOLD", "0.80")
    assert _threshold() == 0.80
    monkeypatch.setenv("GC_AI_THRESHOLD", "garbage")
    assert _threshold() == 0.90


def test_category_whitelist_content():
    assert ALLOWED_CATEGORIES == {"grammar", "spelling", "punctuation",
                                  "capitalization", "word_choice", "style"}


def test_local_category_maps_to_whitelist():
    assert _local_category({"category": "SPELLING"}) == "spelling"
    assert _local_category({"category": "weird"}) == "grammar"
    assert _local_category({}) == "grammar"


# --------------------------------------------------------------------------
# FINAL GRAMMAR VALIDATOR protection chain (per-candidate gate)
# --------------------------------------------------------------------------

def _obj(is_error=True, corr_valid=True, category="grammar", confidence=0.98,
         replacement="were", reason="ok"):
    return {"is_error": is_error, "correction_is_valid": corr_valid,
            "replacement": replacement, "category": category,
            "subcategory": "sva", "confidence": confidence, "reason": reason}


class FakeSingleAI(FakeAI):
    """FakeAI variant returning a single JSON object (per-candidate format)."""

    def __init__(self, obj):
        super().__init__(json.dumps(obj))


def test_single_object_verdict_accept():
    ai = FakeSingleAI(_obj())
    out = ai.validate([_candidate()])
    assert out[0].is_error is True
    assert out[0].correction_is_valid is True
    assert out[0].validated is True
    assert out[0].ai_used is True


def test_fenced_markdown_json_still_parses():
    obj = _obj()
    ai = FakeAI("```json\n" + json.dumps(obj) + "\n```")
    out = ai.validate([_candidate()])
    assert out[0].is_error is True


def test_replacement_mismatch_rejects():
    # AI approves but returns a replacement that is not the proposed span.
    ai = FakeSingleAI(_obj(replacement="was"))
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "replacement" in out[0].reason


def test_contradictory_verdict_rejects():
    # AI approves an error but the "replacement" is identical to the original.
    c = {"original": "was", "replacement": "was",
         "sentence": "Some of them was missing.", "confidence": 0.9}
    ai = FakeSingleAI(_obj(replacement="was"))
    out = ai.validate([c])
    assert out[0].is_error is False
    assert "contradictory" in out[0].reason


def test_per_candidate_call_count_is_one_call_per_candidate():
    calls = []

    class CountingAI(FakeAI):
        def _call(self, prompt):
            calls.append(prompt)
            return super()._call(prompt)

    ai = CountingAI(json.dumps([_verdict(), _verdict(index=1)]))
    ai.validate([_candidate(), _candidate()])
    assert len(calls) == 2


def test_prompt_contains_candidate_fields():
    ai = FakeAI(json.dumps([_verdict()]))
    seen = {}

    class SpyAI(FakeAI):
        def _call(self, prompt):
            seen["prompt"] = prompt
            return super()._call(prompt)

    ai = SpyAI(json.dumps([_verdict()]))
    ai.validate([{"original": "contains", "original_text": "contains",
                  "replacement": "contain", "sentence": "One of the tables contains outdated information.",
                  "previous_sentence": "Check the rows.",
                  "next_sentence": "Update it now.",
                  "category": "GRAMMAR", "rule_name": "SVA",
                  "confidence": 0.8}])
    p = seen["prompt"]
    assert "contains" in p
    assert "contain" in p
    assert "SVA" in p
    assert "One of the tables contains outdated information." in p
    assert "Check the rows." in p
    assert "Update it now." in p
    assert "FINAL GRAMMAR VALIDATOR" in p
    assert '"is_error": true' in p
    assert '"is_error": false' in p


def test_rejection_verdict_with_markdown_trailing_prose():
    raw = (json.dumps(_obj(is_error=False)) + "\nHope this helps!")
    ai = FakeAI(raw)
    out = ai.validate([_candidate()])
    assert out[0].is_error is False
    assert "valid" in out[0].reason


# ─── GEMINI_API_KEY / .env support ──────────────────────────────────────────

def test_gemini_key_prefers_gemini_api_key(monkeypatch):
    from ai_validator import _api_key
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.setenv("AI_API_KEY", "ak")
    assert _api_key("gemini") == "gk"
    assert _api_key("openai") == "ak"


def test_gemini_key_falls_back_to_ai_api_key(monkeypatch):
    from ai_validator import _api_key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("AI_API_KEY", "ak")
    assert _api_key("gemini") == "ak"


def test_gemini_provider_configured_with_gemini_api_key(monkeypatch):
    from ai_validator import _provider_configured
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.delenv("AI_API_KEY", raising=False)
    assert _provider_configured("gemini") is True


def test_env_parser_ignores_comments_and_quotes(tmp_path):
    from ai_validator import _parse_env_lines
    cfg = _parse_env_lines([
        "# a comment",
        "GEMINI_API_KEY=sec ret",
        'AI_MODEL="gemini-2.0-flash"',
        "NO_EQUALS_LINE",
        "EMPTYVAL=",
        "WITH_SPACES = spaced-value ",
    ])
    assert cfg["GEMINI_API_KEY"] == "sec ret"
    assert cfg["AI_MODEL"] == "gemini-2.0-flash"
    assert "NO_EQUALS_LINE" not in cfg
    assert cfg["EMPTYVAL"] == ""
    assert cfg["WITH_SPACES"] == "spaced-value"


# ─── provider fallback (call_any) ──────────────────────────────────────────

class ScriptedFallback(AIValidator):
    """Multi-provider stub: dispatch per active provider via a script dict."""

    def __init__(self, script):
        os.environ["GEMINI_API_KEY"] = "gtest"
        os.environ["GROQ_API_KEY"] = "qtest"
        os.environ["OPENROUTER_API_KEY"] = "otest"
        super().__init__(providers=["gemini", "groq", "openrouter"])
        self.script = script

    def _call(self, prompt):
        name = getattr(self, "_active_provider", None) or self.provider
        fn = self.script.get(name)
        if fn is None:
            raise RuntimeError(f"{name} not scripted")
        return fn(prompt)


def test_call_any_uses_primary_when_healthy():
    ai = ScriptedFallback({
        "gemini": lambda p: "gemini-ok",
        "groq": lambda p: "groq-ok",
    })
    assert ai.call_any("hi") == "gemini-ok"


def test_call_any_falls_back_when_primary_fails():
    def boom(prompt):
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    ai = ScriptedFallback({
        "gemini": boom,
        "groq": lambda p: '{"verdict": 1}',
    })
    assert ai.call_any("hi") == '{"verdict": 1}'


def test_call_any_skips_down_providers_and_raises_last_error():
    def boom(prompt):
        raise RuntimeError("down")

    ai = ScriptedFallback({"gemini": boom, "groq": boom, "openrouter": boom})
    with pytest.raises(RuntimeError, match="down"):
        ai.call_any("hi")


def test_call_any_restores_active_provider_after_call():
    ai = ScriptedFallback({
        "gemini": lambda p: "a",
        "groq": lambda p: "b",
    })
    assert ai.call_any("hi") == "a"
    assert ai._active_provider is None
    assert ai.provider == "gemini"


# ─── local Ollama auto-detection + provider selection ─────────────────────

def _clear_provider_env(monkeypatch):
    for k in ("AI_PROVIDER", "GC_AI_PROVIDERS", "AI_API_KEY",
              "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
              "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_MODEL",
              "AI_MODEL", "AI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)


def test_enabled_providers_autodetect_local_ollama(monkeypatch):
    from ai_validator import _enabled_providers
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr("ai_validator._ollama_available", lambda: True)
    assert _enabled_providers() == ["ollama"]


def test_enabled_providers_disabled_never_autodetects(monkeypatch):
    from ai_validator import _enabled_providers
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "none")
    monkeypatch.setattr("ai_validator._ollama_available", lambda: True)
    assert _enabled_providers() == ["none"]


def test_enabled_providers_ollama_explicit(monkeypatch):
    from ai_validator import _enabled_providers
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    assert _enabled_providers() == ["ollama"]


def test_enabled_providers_ensemble_ollama_gemini(monkeypatch):
    from ai_validator import _enabled_providers
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GC_AI_PROVIDERS", "ollama,gemini,groq")
    monkeypatch.setenv("GEMINI_API_KEY", "gkey")
    monkeypatch.setenv("GROQ_API_KEY", "qkey")
    monkeypatch.setattr("ai_validator._ollama_available", lambda: False)
    assert _enabled_providers() == ["ollama", "gemini", "groq"]


def test_enabled_providers_remote_requires_key(monkeypatch):
    from ai_validator import _enabled_providers
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GC_AI_PROVIDERS", "gemini,groq")
    monkeypatch.setenv("GEMINI_API_KEY", "gkey")
    monkeypatch.setattr("ai_validator._ollama_available", lambda: False)
    assert _enabled_providers() == ["gemini"]


def test_ollama_model_uses_ai_model_env_for_init(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("AI_MODEL", "phi3:mini")
    v = AIValidator(providers=["ollama"])
    assert v.provider == "ollama"
    assert v.model == "phi3:mini"


def test_ollama_model_per_provider_env_via_call_any(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    captured = {}

    class M(AIValidator):
        def __init__(self):
            super().__init__(providers=["ollama"])

        def _call(self, prompt):
            captured["model"] = self.model
            raise RuntimeError("injected")

    ai = M()
    with pytest.raises(RuntimeError, match="injected"):
        ai.call_any("hi")
    assert captured["model"] == "qwen2.5:3b"


def test_call_any_uses_ollama_then_falls_back_to_gemini(monkeypatch):
    class Dispatch(AIValidator):
        def __init__(self):
            super().__init__(providers=["ollama", "gemini"])
            self.calls = []

        def _call(self, prompt):
            self.calls.append(self._active_provider)
            if self._active_provider == "ollama":
                raise RuntimeError("connection refused")
            return "gemini-result"

    monkeypatch.setattr("ai_validator._ollama_available", lambda: False)
    ai = Dispatch()
    assert ai.call_any("hi") == "gemini-result"
    assert ai.calls == ["ollama", "gemini"]