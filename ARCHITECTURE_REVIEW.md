# ARCHITECTURE_REVIEW.md

High-precision hybrid grammar checker with AI final validation.
Review date: 2026-09-15 · Project root: `grammar_checker`

---

## 1. Executive summary

The project is 70% aligned with the master prompt already. It has a working
multi-detector evidence pipeline, a strict-JSON AI validator with provider
abstraction, a full-page editor UI, and an honest 2000-case benchmark
(1000 error + 1000 clean). Current verified local-only benchmark:

| Split | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Train+Val | 1.000 | 1.000 | 1.000 | 0.000 |
| Held-out | 1.000 | 1.000 | 1.000 | 0.000 |

**The four critical gaps before this can be called production-ready:**

1. **The AI final gate is too weak** — `new_pipeline.py` accepts AI verdicts at
   `confidence >= 0.40` (0.90 required), does not validate `correction_is_valid`,
   and on AI failure falls back to *pass-through* instead of *reject*.
2. **The production UI is wired to the wrong endpoint** — `static/js/script.js`
   calls `/api/check-v3` (v3 unified pipeline, **no AI gate, no high-conf rules**).
   The AI-validated v4 path exists (`/api/check-v4`) but the editor never calls it.
3. **No measurable correction accuracy** — the benchmark counts only
   *flagged vs clean* (TP/FP/FN/TN). It never checks whether each
   `replacement` actually equals the intended correction.
4. **Missing detectors from the spec** — Harper, LanguageTool, Hunspell/SymSpell
   are not integrated at all (only spaCy + custom rules + a custom speller).
   They must be optional, gracefully-degrading integrations.

Secondary gaps: no per-category metrics, no BENCHMARK_REPORT.md /
ERROR_ANALYSIS.md, only 200 hand-written test cases (no pytest unit tests for
the AI validator fail-safe branches, no detector pos/neg/edge tests).

---

## 2. What already exists (mapped to the master prompt)

| Master prompt | Existing implementation | Status |
|---|---|---|
| Text normalization | `preprocessor.protect()` | ✅ |
| Sentence segmentation / tokenization | spaCy `doc = nlp(clean_text)` | ✅ |
| Spell detection | `FastDetector` + `spelling_checker` (custom dict + fuzzy) | ✅ |
| Custom grammar rules | `HighConfidenceDetector` (58 rules) + `high_confidence_rules.py` | ✅ |
| Harper / LanguageTool / Hunspell | none (absent) | ❌ |
| spaCy NLP / POS / dependency | `NLPDetector`, `pos_tagger`, `spacy_detectors` | ✅ |
| Context analysis | `ContextEngine.should_suppress` (entity/context traps) | ✅ |
| Candidate detection | 4 detectors merged in `new_pipeline._merge_and_filter` | ✅ |
| Candidate deduplication | `EvidenceStore` (overlap + same-fix merge, source tags) | ✅ |
| Correction generation | every detector produces `replacement` | ✅ |
| Correction validation | `CorrectionValidator.validate` | ✅ |
| AI final validation | `AIValidator` (owner: ai_validator.py) | ⚠️ weak threshold |
| Final confidence score | `evidence_confidence` + validation clamp | ✅ |
| Strict filter | `REPORT_MIN_CONFIDENCE = 0.70` | ✅ |
| Full-page UI | `templates/index.html` + editor, highlights, popovers, sidebar | ⚠️ wired to v3 |

---

## 3. Gap analysis, ranked by impact

### G1 — AI gate threshold and fail-safe (CRITICAL, master §5/§14/§32)
`new_pipeline.py:149-154`:

```python
if v.is_error and v.confidence >= 0.40:
    keep.append(item)
```

Problems:
- Accepts verdicts as low as 0.40 confidence. Master requires >= 0.90 default.
- Ignores `correction_is_valid` (schema field the master defines in §8).
- If `_parse_verdicts` returns `[]` (malformed JSON) the loop uses
  `v = verdicts[i] if i < len(...) else None`; on `None` it calls
  `ValidationResult(... confidence=..., reason="AI verdict unparsed; deterministic
  fallback.")` and **keeps the candidate**. That violates "if AI fails → REJECT".
- Fail-safe differentiates "unavailable" (pass-through OK, configured fallback)
  from "available but produced garbage" (must reject). Not implemented.
- Category from AI is not whitelisted (`grammar|spelling|punctuation|
  capitalization|word_choice|style`), so a model hallucinating `category=null`
  or `"mediocre"` is accepted.

**Fix:** add env `GC_AI_THRESHOLD` (default 0.90), require `correction_is_valid`,
whitelist categories, and add a strict mode: if AI is enabled + available but a
verdict is malformed/missing → reject candidate (unless `GC_AI_PERMISSIVE=1`,
then fall back to local confidence). Keep `GC_AI_PROVIDER=none` behavior
(disabled) identical so the deterministic benchmark stays valid.

### G2 — UI not using the v4 (AI-validated) endpoint (CRITICAL, master §23/§24)
`static/js/script.js`: `fetch('/api/check-v3', ...)`.
- v3 = `unified_check` (older detector set, **no** `HighConfidenceDetector`,
  **no** EvidenceStore source-tagging, **no** AI gate).
- The editor therefore never shows AI-validated results, and its highlight
  positions come from a different span model.
- `/api/check-v4` exists and maps to the full v4 pipeline + optional AI.

**Fix:** switch `script.js` to `/api/check-v4`, send `use_ai` per settings,
and keep the existing (already correct) render-from-`issues` behavior. Verify
highlight offsets still line up with v4 spans (v4 returns `start`/`end`).

### G3 — Correction accuracy not measured (HIGH, master §20)
`run_benchmark.py` compares only `len(issues) > 0`. It never asserts
`issue.replacement == expected_good_text` on the span. F1=1.000 can hide wrong
corrections. Need a per-case `correction_ok` check: the flagged span's
`replacement` must equal / produce the `good` text, else that TP becomes a
"wrong-correction" (counted separately, reported as a distinct metric).

### G4 — Missing external detectors (MEDIUM, master §11)
Harper, LanguageTool (`language_tool_python`), Hunspell/SymSpell absent.
`importlib.util.find_spec()` confirms none installed. Adding them as mandatory
deps breaks portability. **Fix:** create `grammar_engine/integrations/` with
`harper.py`, `languagetool.py`, `hunspell.py` that import specs, expose
`detect(text) -> List[Candidate]`, and immediately return `[]` when the lib is
unavailable. Wire them into `_merge_and_filter` as *optional additional sources*
(sources feed the EvidenceStore consensus bonus). Using an external detector must
never crash the pipeline.

### G5 — Docs and reports missing (LOW)
Master §29/§33 requires `BENCHMARK_REPORT.md`, `ERROR_ANALYSIS.md`,
`ARCHITECTURE_REVIEW.md` (this file). Generate after the fixes.

### G6 — Tests incomplete (MEDIUM)
Only 200 hand-written HTTP cases in `tests/grammar_accuracy/test_cases.json`.
No focused unit tests for:
- AI validator: valid / invalid / uncertain / wrong-correction /
  meaning-changing / malformed-JSON / timeout branches.
- Detector rules: positive, negative, edge examples.
Master §30 requires these.

---

## 4. Recommended implementation order (tied to the todo list)

1. **Phase 1** — Harden `ai_validator.py` + `new_pipeline.py` AI gate
   (threshold, correction_is_valid, category whitelist, strict fail-safe).
2. **Phase 2** — Local validation hardening: keep `CorrectionValidator`, make
   sure meaning-preservation checks exist; add per-candidate local gate.
3. **Phase 3** — Re-wire UI JS + index.html to `/api/check-v4`, add AI-validated
   badge, verify inline offsets.
4. **Phase 4** — Unit tests (pytest) for AI validator branches + detector
   pos/neg/edge; run the 200-case suite.
5. **Phase 5** — Extend benchmark with correction accuracy + per-category
   metrics; run 1000+error / 1000+clean; write BENCHMARK_REPORT.md and
   ERROR_ANALYSIS.md.

---

## 5. Acceptance checklist (from master §33)

- [ ] AI gate: confidence >= 0.90 (configurable), correction_is_valid enforced,
      category whitelist, fail-safe = reject on malformed AI output.
- [ ] UI calls the v4 endpoint; editorial UI is the only place issues render.
- [ ] CORRECTION ACCURACY metric present in benchmark output.
- [ ] Optional Harper / LanguageTool / SymSpell integrations (graceful absence).
- [ ] 1000+ error + 1000+ clean benchmark re-run, honest numbers.
- [ ] BENCHMARK_REPORT.md + ERROR_ANALYSIS.md written.
- [ ] Sample acceptance paragraph from master §31 → correct issues only,
      no over-flagging.