"""
ai_validator.py — AI FINAL GRAMMAR VALIDATOR (judge, not engine).

This module is the LAST gate before an error reaches the UI. The local engines
(FastDetector / NLPDetector / DataDrivenDetector / HighConfidenceDetector)
produce candidate errors. THIS module NEVER looks for new errors. For every
candidate it sends ONE model call that validates THAT SPECIFIC candidate span:

    candidate -> prompt -> Gemini -> JSON -> schema -> gates -> APPROVE/REJECT

Per-candidate flow (mirrors the "FINAL GRAMMAR VALIDATOR" master prompt):
    Gemini response
          ↓
    JSON parsing
          ↓
    Schema validation
          ↓
    is_error == true?
          ↓
    correction_is_valid == true?
          ↓
    confidence >= 0.90?
          ↓
    replacement matches expected span?
          ↓
    No contradictory result?
          ↓
    APPROVE          (any step fails -> reject the candidate)

Fail-safe semantics:
  - Provider disabled (AI_PROVIDER=none / "")   -> local-only pass-through
    (checker works without AI, clearly marked validated=False).
  - Provider enabled but UNREACHABLE            -> strict: reject all (nothing
    leaks unvalidated); permissive: local fallback.
  - Provider enabled + reachable but a verdict
    for a candidate is malformed/missing        -> strict: reject that candidate;
    permissive: local fallback.

Policy (configurable via environment):
    AI_PROVIDER      ollama | gemini | openai | anthropic | none/off(disabled)
    AI_MODEL         model name (or <PROVIDER>_MODEL for a specific provider)
    AI_API_KEY       key (NOT required for local ollama)
    GEMINI_API_KEY   priority key for the gemini provider (falls back to AI_API_KEY)
    AI_BASE_URL      optional override (defaults per provider)
    GC_AI_PROVIDERS  comma-separated ensemble (e.g. "gemini,ollama"). With a
                     single provider you can set AI_PROVIDER instead. Providers
                     without a key (ollama) are always candidates; remote ones
                     are included only when their key is present.
    GC_AI_THRESHOLD  minimum AI confidence to ACCEPT a correction (default 0.90)
    GC_AI_STRICT     "1" (default) reject when the AI verdict is missing/malformed
                     while the provider is enabled + reachable; "0" falls back to
                     local confidence in that case.
    GC_AI_PERMISSIVE alias kept for compatibility with GC_AI_STRICT=0.

Auto-detection: when AI_PROVIDER is unset and no API keys are configured but a
local Ollama server is reachable on AI_BASE_URL (default http://localhost:11434),
Ollama is used automatically so the checker gets a real AI judge with zero setup.
A bare AI_PROVIDER=none / off / disabled always disables AI outright.

Keys are read from the process environment OR a .env file in the project root
(loaded lazily on first use; python-dotenv is optional). No provider = feature
disabled by default. Never commit keys or .env files.
"""

import json
import os
import re
import threading
from typing import List, Dict, Optional

try:  # python-dotenv is optional; we degrade to a tiny stdlib parser when absent
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - fallback parser
    load_dotenv = None


def _parse_env_lines(lines) -> Dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, # comments, quoted values."""
    out: Dict[str, str] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def _load_env() -> None:
    """Load .env from the project root *before* any _cfg() read."""
    if os.environ.get("_GC_ENV_LOADED"):
        return
    os.environ["_GC_ENV_LOADED"] = "1"
    root = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(root, ".env")
    if not os.path.isfile(env_file):
        return
    try:
        if load_dotenv is not None:
            load_dotenv(env_file, override=False)
        else:  # stdlib fallback parser
            with open(env_file, encoding="utf-8") as f:
                for k, v in _parse_env_lines(f).items():
                    if k not in os.environ:
                        os.environ[k] = v
    except OSError:
        pass

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Dict] = {}

# The only categories the AI is allowed to return for a confirmed error.
ALLOWED_CATEGORIES = {
    "grammar", "spelling", "punctuation", "capitalization",
    "word_choice", "style",
}

# Final grammar validator prompt — ONE candidate per call.
_FINAL_VALIDATOR_PROMPT = """You are the FINAL GRAMMAR VALIDATOR in a high-precision English grammar checking system.


Your job is NOT to rewrite the sentence.
Your job is to VALIDATE ONE CANDIDATE ERROR detected by another grammar engine.


The previous grammar engines may produce false positives.
Therefore, NEVER trust the candidate automatically.


You must analyze:
1. The complete sentence.
2. The surrounding context.
3. The exact candidate span.
4. The proposed correction.
5. Grammar, syntax, meaning, tense, agreement, word usage, and sentence structure.
6. Whether the original wording could be grammatically valid in context.


IMPORTANT:
- Do not flag something merely because another wording sounds better.
- Do not flag style preferences as grammar errors.
- Do not rewrite the whole sentence.
- Do not invent additional errors.
- Do not change correct English.
- Do not be overly conservative.
- Approve the candidate when it is a GENUINE, OBJECTIVE error: subject-verb agreement, tense, pronoun case, double negatives, article/determiner misuse, misspellings, punctuation, capitalization, or word-choice that is clearly wrong in context.
- Reject ONLY when the original wording is clearly grammatically acceptable, when the correction changes the intended meaning, or when the candidate is pure style preference.
- When in doubt about a REAL, well-known error type, approve; when in doubt about a stylistic alternate wording, reject.
- Consider standard modern English.
- Accept legitimate variations such as collective nouns when grammatically valid.
- Consider the full sentence before making a decision.


CANDIDATE:


Original sentence:
"{sentence}"


Candidate text:
"{original_text}"


Proposed correction:
"{replacement}"


Detected category:
"{category}"


Detected rule:
"{rule_name}"


CONTEXT:


Previous sentence:
"{previous_sentence}"


Next sentence:
"{next_sentence}"


TASK:


Determine whether the candidate is a genuine English grammar/spelling/punctuation/word-choice error.


Return ONLY valid JSON.


Required JSON schema:


{{
  "is_error": true,
  "correction_is_valid": true,
  "replacement": "{replacement}",
  "category": "grammar",
  "subcategory": "specific_error_type",
  "confidence": 0.99,
  "reason": "Short explanation of why the original is incorrect."
}}


If the candidate is NOT a genuine error:


{{
  "is_error": false,
  "correction_is_valid": false,
  "replacement": null,
  "category": null,
  "subcategory": null,
  "confidence": 0.99,
  "reason": "Short explanation of why the original is acceptable."
}}


VALIDATION RULES:


1. is_error MUST be false if the original sentence is grammatically acceptable.
2. is_error MUST be false if the proposed correction is unnecessary.
3. is_error MUST be false if the issue is only stylistic.
4. is_error MUST be false if the meaning depends on missing context.
5. is_error MUST be false if the candidate is a legitimate English construction.
6. correction_is_valid MUST be true only when the proposed replacement actually fixes the identified error.
7. Never return a correction that changes the intended meaning.
8. Never correct an entire sentence when only one candidate was supplied.
9. Confidence must represent your confidence in the VALIDATION decision, not how confident the original detector was.
10. Use confidence from 0.00 to 1.00.
11. Return valid JSON only. No Markdown. No additional text.


FINAL DECISION:


Approve the candidate ONLY when:
- is_error = true
- correction_is_valid = true
- confidence >= 0.90


Otherwise reject the candidate completely."""


def _cfg(key: str, default: str = "") -> str:
    _load_env()
    return os.environ.get(key, default).strip()


_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _api_key(provider: str) -> str:
    """Resolve the API key. Provider-specific key wins, then AI_API_KEY.

    Gemini -> GEMINI_API_KEY, Groq -> GROQ_API_KEY,
    OpenRouter -> OPENROUTER_API_KEY; all fall back to AI_API_KEY.
    """
    _load_env()
    env_name = _KEY_ENV.get(provider)
    if env_name:
        return _cfg(env_name) or _cfg("AI_API_KEY")
    return _cfg("AI_API_KEY")


def _provider_configured(provider: str) -> bool:
    if provider == "ollama":
        return True  # local, always try; will be marked unavailable on connect error
    return bool(_api_key(provider))


def _threshold() -> float:
    _load_env()
    try:
        return float(os.environ.get("GC_AI_THRESHOLD", "") or "0.90")
    except ValueError:
        return 0.90


def _strict_mode() -> bool:
    _load_env()
    val = (os.environ.get("GC_AI_STRICT", "") or
           os.environ.get("GC_AI_PERMISSIVE", "") or "").lower()
    return val not in ("0", "false", "permissive")


def _detect_provider() -> str:
    p = _cfg("AI_PROVIDER").lower()
    if p in ("", "none", "off", "disabled"):
        return "none"
    if p in ("ollama", "gemini", "openai", "anthropic", "groq", "openrouter"):
        return p
    return "gemini"  # default remote judge = Gemini


def _enabled_providers() -> List[str]:
    """Providers enabled for the ensemble.

    AI_PROVIDER=none/off/disabled always disables AI completely. Otherwise:
    GC_AI_PROVIDERS (comma separated) selects the ensemble; a plain
    AI_PROVIDER value selects a single provider. Only configured providers
    (key present, or local ollama) are returned.
    """
    p = _cfg("AI_PROVIDER").lower()
    if p in ("", "none", "off", "disabled"):
        if p:
            return ["none"]
        raw = _cfg("GC_AI_PROVIDERS")
        select_from = raw
    else:
        raw = _cfg("GC_AI_PROVIDERS")
        select_from = raw or p
    names = []
    for name in (x.strip().lower() for x in select_from.split(",")):
        if not name or name in ("none", "off", "disabled"):
            continue
        if name not in ("ollama", "gemini", "openai", "anthropic", "groq", "openrouter"):
            continue
        if _provider_configured(name):
            names.append(name)
    if not names:
        # Nothing explicitly configured: fall back to the default remote judge
        # (gemini when a key exists) or auto-detect a reachable local Ollama.
        base = _detect_provider()
        if base in ("ollama",):
            return ["ollama"]
        if _api_key(base):
            return [base]
        if not any(_api_key(k) for k in ("gemini", "openai", "anthropic",
                                         "groq", "openrouter")):
            try:
                if _ollama_available():
                    return ["ollama"]
            except Exception:
                pass
        return [base]
    return names


def _min_approvals() -> int:
    _load_env()
    try:
        return max(1, int(os.environ.get("GC_AI_MIN_APPROVALS", "") or "2"))
    except ValueError:
        return 2


def _ollama_available() -> bool:
    """Lightweight liveness probe to avoid blocking app startup on every check."""
    try:
        import urllib.request
        base = _cfg("AI_BASE_URL", "http://localhost:11434")
        req = urllib.request.Request(base + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


class ValidationResult:
    __slots__ = ("is_error", "correction_is_valid", "replacement", "category",
                 "subcategory", "confidence", "reason", "validated", "ai_used")

    def __init__(self, is_error, correction_is_valid=False, replacement=None,
                 category=None, subcategory=None, confidence=0.0, reason="",
                 validated=True, ai_used=False):
        self.is_error = is_error
        self.correction_is_valid = correction_is_valid
        self.replacement = replacement
        self.category = category
        self.subcategory = subcategory
        self.confidence = confidence
        self.reason = reason
        self.validated = validated
        self.ai_used = ai_used


class AIValidator:
    """Conservative final judge over candidate errors.

    Validates ONE candidate per model call using the FINAL GRAMMAR VALIDATOR
    prompt. A candidate is APPROVED only when the whole protection chain passes:

        valid JSON -> schema ok -> is_error -> correction_is_valid ->
        category allowed -> confidence >= threshold ->
        replacement matches the proposed span -> not self-contradictory.
    """

    def __init__(self, providers: Optional[List[str]] = None):
        self.provider = None
        self.model = ""
        self.providers: List[str] = []
        self._active_provider: Optional[str] = None
        self._available = None  # None = unknown, True/False cached
        for name in (providers or _enabled_providers()):
            if name == "none":
                continue
            if not _provider_configured(name):
                continue
            self.providers.append(name)
            if self.provider is None:
                self.provider = name
                self.model = _cfg("AI_MODEL", self._default_model(name))
        if not self.providers:
            self._available = False

    @staticmethod
    def _default_model(provider: str) -> str:
        return {
            "ollama": "llama3.2:3b-instruct-q4_K_M",
            "gemini": "gemini-3.5-flash-lite",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "groq": "llama-3.3-70b-versatile",
            "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
        }.get(provider, "")

    # ---------------------------------------------------------------- public
    def available(self) -> bool:
        if self._available is None:
            if self.provider is None:
                self._available = False
            elif self.provider == "ollama":
                self._available = _ollama_available()
            else:
                self._available = bool(_api_key(self.provider))
        return self._available

    def is_enabled(self) -> bool:
        return self.provider is not None

    def call_any(self, prompt: str) -> str:
        """Call configured providers in order; fall back to the next on any
        failure (rate limit, model retired, network error, ...).

        Returns the first successful response. If every configured provider
        fails, the last exception is re-raised.
        """
        last: Optional[Exception] = None
        for name in self.providers:
            prev_active = self._active_provider
            prev_model = self.model
            try:
                self._active_provider = name
                if name == "gemini":
                    self.model = _cfg("AI_MODEL", self._default_model(name))
                else:
                    self.model = _cfg(f"{name.upper()}_MODEL", self._default_model(name))
                return self._call(prompt)
            except Exception as exc:  # try the next provider
                last = exc
            finally:
                self._active_provider = prev_active
                self.model = prev_model
        if last is not None:
            raise last
        raise RuntimeError("no AI provider available")

    def validate(self, candidates: List[Dict]) -> List[ValidationResult]:
        """Validate each candidate individually (one model call per candidate).

        Each candidate needs {original, replacement, sentence} plus optional
        {previous_sentence, next_sentence, category, rule_name}. Returns one
        ValidationResult per candidate. The candidate is ACCEPTED only when the
        whole protection chain in the module docstring passes; any failed step
        rejects it.

        When more than one provider is configured (GC_AI_PROVIDERS), each
        provider votes per candidate and the candidate is accepted only when
        at least GC_AI_MIN_APPROVALS (capped at the number of providers that
        actually answered) agree. This ensemble veto raises precision above a
        single judge while majority voting avoids over-conservative rejection.
        """
        results: List[ValidationResult] = []
        if not candidates:
            return results

        enabled = self.is_enabled()
        reachable = self.available() if enabled else False

        # ---- AI fully disabled -> local-only pass-through ------------------
        if not enabled:
            for c in candidates:
                results.append(ValidationResult(
                    is_error=True, correction_is_valid=True,
                    replacement=c.get("replacement"),
                    category=_local_category(c), confidence=float(c.get("confidence", 0.8)),
                    reason="Local detection only (AI disabled).", validated=False,
                    ai_used=False,
                ))
            return results

        # ---- AI enabled but unreachable -> strict reject / permissive pass --
        if not reachable:
            if _strict_mode():
                for c in candidates:
                    results.append(ValidationResult(
                        is_error=False, correction_is_valid=False,
                        replacement=None, confidence=0.0, validated=False,
                        reason="AI provider unreachable; candidate not confirmed (strict mode).",
                        ai_used=False,
                    ))
            else:
                for c in candidates:
                    results.append(ValidationResult(
                        is_error=True, correction_is_valid=True,
                        replacement=c.get("replacement"),
                        category=_local_category(c),
                        confidence=float(c.get("confidence", 0.8)), validated=False,
                        reason="AI provider unreachable; local fallback (permissive).",
                        ai_used=False,
                    ))
            return results

        th = _threshold()
        if len(self.providers) > 1:
            return self._validate_ensemble(candidates, th)
        return self._validate_single(candidates, th)

    # ------------------------------------------------------- single provider
    def _validate_single(self, candidates: List[Dict], th: float) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for i, c in enumerate(candidates):
            verdict = None
            try:
                self._active_provider = self.provider
                raw = self._call(self._build_prompt(c))
                verdict = self._parse_verdict(raw, expected_index=i)
            except Exception:
                verdict = None

            if verdict is None or not self._verdict_ok(verdict):
                # Missing, unparseable, or schema-invalid verdict.
                if _strict_mode():
                    results.append(ValidationResult(
                        is_error=False, correction_is_valid=False,
                        replacement=None, confidence=0.0, validated=True,
                        reason="AI verdict missing/malformed; rejected (strict mode).",
                        ai_used=True,
                    ))
                else:
                    results.append(ValidationResult(
                        is_error=True, correction_is_valid=True,
                        replacement=c.get("replacement"),
                        category=_local_category(c),
                        confidence=float(c.get("confidence", 0.8)), validated=False,
                        reason="AI verdict missing/malformed; local fallback (permissive).",
                        ai_used=True,
                    ))
                continue

            is_error = bool(verdict.get("is_error"))
            corr_ok = bool(verdict.get("correction_is_valid", is_error))
            category = (verdict.get("category") or "").lower()
            subcategory = verdict.get("subcategory") or ""
            conf = _clamp_conf(verdict.get("confidence"))

            if (not (is_error and corr_ok)
                    or category not in ALLOWED_CATEGORIES
                    or conf < th):
                results.append(ValidationResult(
                    is_error=False, correction_is_valid=False,
                    replacement=None, category=category, subcategory=subcategory,
                    confidence=min(conf, th - 0.01), reason=_reject_reason(
                        is_error, corr_ok, category, conf, th),
                    validated=True, ai_used=True,
                ))
                continue

            # Protection: the AI's replacement must match the proposed span.
            if not _replacement_matches(c.get("replacement"), verdict.get("replacement")):
                results.append(ValidationResult(
                    is_error=False, correction_is_valid=False,
                    replacement=None, category=category, subcategory=subcategory,
                    confidence=min(conf, th - 0.01),
                    reason="AI: replacement does not match the proposed correction; rejected.",
                    validated=True, ai_used=True,
                ))
                continue

            # Protection: reject self-contradictory verdicts.
            if _contradicts(c, verdict):
                results.append(ValidationResult(
                    is_error=False, correction_is_valid=False,
                    replacement=None, category=category, subcategory=subcategory,
                    confidence=min(conf, th - 0.01),
                    reason="AI: contradictory verdict; rejected.",
                    validated=True, ai_used=True,
                ))
                continue

            results.append(ValidationResult(
                is_error=True, correction_is_valid=True,
                replacement=c.get("replacement"),
                category=category, subcategory=subcategory,
                confidence=conf, reason=verdict.get("reason", ""),
                validated=True, ai_used=True,
            ))
        return results

    # ---------------------------------------------------------- ensemble vote
    def _validate_ensemble(self, candidates: List[Dict], th: float) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        active = [p for p in self.providers
                  if p != "ollama" or _ollama_available()]
        need = max(1, _min_approvals())
        for i, c in enumerate(candidates):
            prompt = self._build_prompt(c)
            responses = []  # [(provider, verdict)] for providers that answered
            for p in active:
                try:
                    self._active_provider = p
                    raw = self._call(prompt)
                    verdict = self._parse_verdict(raw, expected_index=i)
                    if verdict is not None and self._verdict_ok(verdict):
                        responses.append((p, verdict))
                except Exception:
                    continue

            total = len(responses)
            if total == 0:
                # Every provider failed/malformed its answer.
                if _strict_mode():
                    results.append(ValidationResult(
                        is_error=False, correction_is_valid=False,
                        replacement=None, confidence=0.0, validated=True,
                        reason="AI ensemble verdict missing/malformed; rejected (strict mode).",
                        ai_used=True,
                    ))
                else:
                    results.append(ValidationResult(
                        is_error=True, correction_is_valid=True,
                        replacement=c.get("replacement"),
                        category=_local_category(c),
                        confidence=float(c.get("confidence", 0.8)), validated=False,
                        reason="AI ensemble verdict missing/malformed; local fallback (permissive).",
                        ai_used=True,
                    ))
                continue

            approved_by = {p for p, v in responses
                           if self._approval_ok(c, v, th)}
            total_need = min(need, total)
            approved = len(approved_by) >= total_need and len(approved_by) > 0
            votes = ", ".join(
                f"{p}:{'approve' if p in approved_by else 'reject'}"
                for p, v in responses)
            if approved:
                best = max((v for p, v in responses if p in approved_by),
                           key=lambda v: _clamp_conf(v.get("confidence")))
                results.append(ValidationResult(
                    is_error=True, correction_is_valid=True,
                    replacement=c.get("replacement"),
                    category=(best.get("category") or "grammar").lower(),
                    subcategory=best.get("subcategory") or "",
                    confidence=_clamp_conf(best.get("confidence")),
                    reason=f"AI ensemble approved ({len(approved_by)}/{total}): {votes}.",
                    validated=True, ai_used=True,
                ))
            else:
                results.append(ValidationResult(
                    is_error=False, correction_is_valid=False,
                    replacement=None,
                    confidence=0.0,
                    reason=f"AI ensemble rejected ({len(approved_by)}/{total}): {votes}.",
                    validated=True, ai_used=True,
                ))
        return results

    def _approval_ok(self, c: Dict, verdict: Dict, th: float) -> bool:
        """Full approval gate for a single provider verdict (ensemble voter)."""
        is_error = bool(verdict.get("is_error"))
        corr_ok = bool(verdict.get("correction_is_valid", is_error))
        category = (verdict.get("category") or "").lower()
        conf = _clamp_conf(verdict.get("confidence"))
        if (not (is_error and corr_ok)
                or category not in ALLOWED_CATEGORIES
                or conf < th):
            return False
        if not _replacement_matches(c.get("replacement"), verdict.get("replacement")):
            return False
        if _contradicts(c, verdict):
            return False
        return True

    # -------------------------------------------------------------- internals
    def _call(self, prompt: str) -> str:
        provider = getattr(self, "_active_provider", None) or self.provider
        if provider == "ollama":
            return self._call_ollama(prompt)
        if provider == "gemini":
            return self._call_gemini(prompt)
        if provider == "anthropic":
            return self._call_anthropic(prompt)
        if provider == "groq":
            return self._call_groq(self._model_list("groq"), prompt)
        if provider == "openrouter":
            return self._call_openrouter(self._model_list("openrouter"), prompt)
        return self._call_openai(prompt)

    def _model_list(self, provider: str) -> List[str]:
        cfg_model = _cfg(f"{provider.upper()}_MODEL") or _cfg("AI_MODEL")
        defaults = {
            "groq": ["llama-3.3-70b-versatile",
                     "llama-3.1-8b-instant", "openai/gpt-oss-20b"],
            "openrouter": ["meta-llama/llama-3.3-70b-instruct:free",
                           "google/gemini-2.0-flash-exp:free",
                           "deepseek/deepseek-chat-v3-0324:free"],
        }.get(provider, [])
        if cfg_model:
            return [cfg_model] + [m for m in defaults if m != cfg_model]
        return defaults or [self.model or self._default_model(provider)]

    @staticmethod
    def _openai_compatible(base: str, models: List[str], prompt: str,
                           key: str) -> str:
        """POST chat.completions to an OpenAI-compatible endpoint, trying each
        model in order until one responds (catalog churn friendly)."""
        import urllib.request
        last_err = None
        for model in models:
            try:
                body = json.dumps({
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 512,
                }).encode()
                req = urllib.request.Request(
                    base + "/chat/completions", data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer " + key},
                    method="POST")
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode())
                    return data["choices"][0]["message"]["content"]
            except Exception as e:  # try next model
                last_err = e
        raise last_err if last_err else RuntimeError("no models to try")

    def _call_groq(self, models: List[str], prompt: str) -> str:
        return self._openai_compatible(
            "https://api.groq.com/openai/v1", models, prompt, _api_key("groq"))

    def _call_openrouter(self, models: List[str], prompt: str) -> str:
        return self._openai_compatible(
            "https://openrouter.ai/api/v1", models, prompt, _api_key("openrouter"))
    def _build_prompt(self, c: Dict) -> str:
        sentence = (c.get("sentence") or "").strip()
        original_text = (c.get("original_text") or c.get("original") or "").strip()
        replacement = (c.get("replacement") or "").strip()
        category = (c.get("category") or "grammar").strip() or "grammar"
        rule_name = (c.get("rule_name") or c.get("rule_id") or "unknown").strip()
        prev = (c.get("previous_sentence") or "").strip()
        next_ = (c.get("next_sentence") or "").strip()
        return _FINAL_VALIDATOR_PROMPT.format(
            sentence=sentence, original_text=original_text, replacement=replacement,
            category=category, rule_name=rule_name,
            previous_sentence=prev, next_sentence=next_,
        )

    def _call_ollama(self, prompt: str) -> str:
        import time as _time
        import urllib.error
        import urllib.request
        base = _cfg("AI_BASE_URL", "http://localhost:11434")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 4096},
        }).encode()
        last_err: Optional[Exception] = None
        for attempt in range(3):
            req = urllib.request.Request(
                base + "/api/chat", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode())
                return data.get("message", {}).get("content", "")
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (404, 429, 500, 502, 503, 504):
                    _time.sleep(2.0 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, OSError) as e:
                # unreachable (or refused) -> do not retry; fail fast so the
                # fallback provider chain (gemini/groq/...) takes over.
                raise e
        raise last_err if last_err else RuntimeError("ollama call failed")

    def _call_openai(self, prompt: str) -> str:
        import urllib.request
        base = _cfg("AI_BASE_URL", "https://api.openai.com/v1")
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2048,
        }).encode()
        req = urllib.request.Request(
            base + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + _cfg("AI_API_KEY")},
            method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str) -> str:
        import re
        import time as _time
        import urllib.error
        import urllib.request
        model = self.model or "gemini-3.6-flash"
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
               f":generateContent?key={_api_key('gemini')}")
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
        }).encode()
        last_err: Optional[Exception] = None
        deadline = _time.monotonic() + 45.0
        attempt = 0
        while _time.monotonic() < deadline:
            attempt += 1
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode())
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 429:
                    delay = None
                    if getattr(e, "headers", None):
                        try:
                            delay = float(e.headers.get("Retry-After"))
                        except (TypeError, ValueError):
                            delay = None
                    if delay is None:
                        try:
                            hint = e.read().decode()[:500]
                            m = re.search(r"retry in ([\d.]+)s", hint, re.IGNORECASE)
                            delay = float(m.group(1)) if m else None
                        except Exception:
                            delay = None
                    if delay is None:
                        delay = 5.0
                    delay = min(delay, 15.0)
                    if _time.monotonic() + delay < deadline:
                        _time.sleep(delay)
                    continue
                if e.code in (500, 502, 503, 504):
                    delay = min(4.0 * attempt, 15.0)
                    if _time.monotonic() + delay < deadline:
                        _time.sleep(delay)
                    continue
                raise
            except (urllib.error.URLError, OSError) as e:
                last_err = e
                delay = min(4.0 * attempt, 15.0)
                if _time.monotonic() + delay < deadline:
                    _time.sleep(delay)
        raise last_err if last_err else RuntimeError("gemini call failed")

    def _call_anthropic(self, prompt: str) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model or "claude-sonnet-4-20250514",
            "max_tokens": 512,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"Content-Type": "application/json",
                     "x-api-key": _cfg("AI_API_KEY"),
                     "anthropic-version": "2023-06-01"},
            method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["content"][0]["text"]

    @staticmethod
    def _parse_verdict(raw: str, expected_index: Optional[int] = None) -> Optional[Dict]:
        """Parse ONE candidate verdict from the model response.

        Accepts a single JSON object (normal per-candidate case) or a legacy
        batch array (matched by "index"). Strips markdown fences/prose. Returns
        None on malformed output — the caller then applies strict/permissive
        policy.
        """
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw).strip()
        if not raw:
            return None

        # Single object (per-candidate responses).
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "is_error" in obj:
                return obj
        except Exception:
            pass

        # Legacy batch array: pick the object whose "index" matches.
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group(0))
            except Exception:
                arr = None
            if isinstance(arr, list):
                for item in arr:
                    if not isinstance(item, dict) or "index" not in item:
                        continue
                    try:
                        idx = int(item["index"])
                    except (TypeError, ValueError):
                        continue
                    if expected_index is None or idx == expected_index:
                        return item

        # Object embedded in extra prose (only for non-batch responses).
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
            if isinstance(obj, dict) and "is_error" in obj:
                if "index" not in obj:
                    return obj
                try:
                    idx = int(obj["index"])
                except (TypeError, ValueError):
                    idx = None
                if expected_index is not None and idx == expected_index:
                    return obj
        return None

    def _verdict_ok(self, v: Dict) -> bool:
        """Schema check on a single verdict (per-candidate JSON schema)."""
        if not isinstance(v, dict):
            return False
        if not isinstance(v.get("is_error"), bool):
            return False
        if not isinstance(v.get("correction_is_valid"), bool):
            return False
        if not isinstance(v.get("reason"), str) or not v.get("reason"):
            return False
        cat = (v.get("category") or "").lower()
        if v.get("is_error"):
            # Approving verdicts must carry an allowed category + a concrete fix.
            if cat not in ALLOWED_CATEGORIES:
                return False
            repl = v.get("replacement")
            if not isinstance(repl, str) or not repl.strip():
                return False
        conf = _clamp_conf(v.get("confidence"))
        return conf > 0.0


_instance = None


def _local_category(c: Dict) -> str:
    cat = (c.get("category") or "").lower()
    return cat if cat in ALLOWED_CATEGORIES else "grammar"


def _clamp_conf(x) -> float:
    try:
        c = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, c))


def _norm_span(s) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s.strip(" .,!?;:'\"()").lower()


def _replacement_matches(expected, actual) -> bool:
    """The AI's replacement must reproduce the proposed span correction.

    Exact (normalized) equality passes; a slightly larger corrected phrase that
    contains the exact span passes. A narrower/partial or unrelated replacement
    fails -> the candidate is rejected (never rewrite a whole sentence).
    """
    if not expected or not actual:
        return False
    e = _norm_span(str(expected))
    a = _norm_span(str(actual))
    if not e or not a:
        return False
    return e == a or e in a


def _contradicts(candidate: Dict, verdict: Dict) -> bool:
    """Reject verdicts that contradict themselves.

    E.g. approving an error while the supplied "replacement" leaves the
    candidate span unchanged, or reporting a fix that equals the original.
    """
    orig = (candidate.get("original_text")
            or candidate.get("original") or "").strip()
    repl = verdict.get("replacement") or ""
    if orig and repl and _norm_span(orig) == _norm_span(repl):
        return True
    return False


def _reject_reason(is_error, corr_ok, category, conf, th) -> str:
    if not corr_ok:
        return "AI: correction is not valid; rejected."
    if not is_error:
        return "AI: original wording is valid; rejected."
    if category not in ALLOWED_CATEGORIES:
        return f"AI: invalid category {category!r}; rejected."
    return f"AI: confidence {conf:.2f} below threshold {th:.2f}; rejected."


def get_validator() -> AIValidator:
    global _instance
    if _instance is None:
        _instance = AIValidator()
    return _instance


def active_providers() -> List[str]:
    """Configured, key-bearing providers in the active ensemble (live config)."""
    return list(get_validator().providers)


def provider_status() -> Dict[str, Dict]:
    """Per-provider config status for /api/health (no network probes)."""
    v = get_validator()
    status: Dict[str, Dict] = {}
    for p in ("gemini", "groq", "openrouter", "ollama", "openai", "anthropic"):
        if p not in v.providers:
            continue
        if p == "ollama":
            status[p] = {"configured": True,
                         "available": _ollama_available(),
                         "model": v._default_model(p)}
        else:
            status[p] = {"configured": True,
                         "available": bool(_api_key(p)),
                         "model": _cfg(f"{p.upper()}_MODEL") or v._default_model(p)}
    return status