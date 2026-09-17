# PHASE 1 AUDIT — WriteMaster AI

> Dependency map, duplication, dead-code triage, and file disposition
> (keep / merge / rename / remove). See `ARCHITECTURE_REVIEW.md` and
> `ARCHITECTURE_REPORT.txt` for earlier analysis this document supersedes.
>
> **Status: 5a EXECUTED; Phases 2–8 in progress** — dead cluster removed (git
> history preserves all 96 archive scripts + dead modules). 169 unit tests +
> serverless import verified green after removal. `results/` + stale benchmark
> output JSONs removed; `benchmark/{clean,error}_cases.json`,
> `test_cases.json`, `grammar_error_dataset_1000.csv`, `data/jfleg_test.json`
> KEPT as input datasets (Phases 12–14).
>
> Master pipeline built: `pipeline/master.py` (check_master, single entry;
> `ai_core.check_ai_text` now delegates to it), `pipeline/schema.py` (canonical
> v2 error object + taxonomy), consensus engine AGREED/AI_ONLY/LOCAL_ONLY with
> AI-authoritative default and opt-in `report_local_only` discovery mode;
> character-similarity meaning guard (handles word splits). 179 unit tests
> green, serverless import green. Commit `36140e7`.

Audit method: AST import-graph reachability from the four real entry points,
then a targeted grep cross-check on every "dead" candidate (no dynamic
`__import__` / `importlib` loading found anywhere in the repo).

---

## 1. Entry points (what actually runs)

| Entry | Where it runs | Loads |
|---|---|---|
| `api/index.py` | **Vercel (production)** | `pipeline.ai_core`, `pipeline.ai_analyzer`, `ai_validator`, `pipeline.rule_detector`, `pipeline.aggregator` |
| `app.py` (Flask) | Local / legacy | `grammar_engine`, `ai_engine`, `writing_engine`, `scoring_engine`, `pipeline.*`, `unified_pipeline`, `new_pipeline`, `suggestion_aggregator`, `ai_validator` |
| `tests/unit/*.py` | pytest (CI) | pipeline, ai_validator, unified/new pipeline |
| `tests/grammar_accuracy/*`, `benchmark/*` | Manual harnesses | pipeline, new_pipeline, unified_pipeline |

**No module uses `importlib`, `__import__`, or exec-based dynamic loading.**
The static reachability map below is therefore authoritative.

---

## 2. Live module graph (reachable from one of the entries above)

### 2a. Serverless production surface (what Vercel actually imports)
```
api/index.py
  └─ pipeline.ai_core.check_ai_text
       ├─ pipeline.rule_detector  → high_confidence_rules, data/*.json
       ├─ pipeline.aggregator
       ├─ pipeline.ai_analyzer    → ai_validator (Gemini transport)
       └─ pipeline.ai_verifier    → ai_validator
ai_validator.py (provider envelope: Gemini/OpenAI/Anthropic/Ollama)
```

This is the **only** surface deployed to production.
`unified_pipeline.py` (spaCy) and `new_pipeline.py` (v4) are simply **not
importable** in the serverless bundle (no spaCy), so they are hint sources
*only when running locally* inside `pipeline/ai_core._v4_candidates()`.

### 2b. Legacy Flask local surface (`app.py`)
```
app.py
 ├─ pipeline.ai_core / pipeline.aggregator / pipeline.feedback   (AI-first, shared)
 ├─ unified_pipeline    (ReadabilityEngine, ToneEngine, StyleEngine, check_text — v3)
 ├─ new_pipeline        (check_v4 — v4 evidence+AI-validate)
 ├─ grammar_engine      (v1)  → preprocessor, pos_tagger, spelling_checker,
 │                             grammar_analyzer, correction_engine, error_manager
 │       grammar_analyzer → article_checker, subject_verb_checker, tense_checker,
 │                          pronoun_checker, sentence_structure_checker,
 │                          context_checker, semantic_checker
 │       correction_engine → intelligent_engine, nlp_engine
 ├─ ai_engine           (rephrase/tone/enhance/summarize — heuristic, no AI call)
 ├─ writing_engine      (v2)  → contextual_engine, punctuation_checker,
 │                              suggestion_validator, readability
 ├─ scoring_engine      (compute_scores)
 └─ suggestion_aggregator
```

---

## 3. Duplication (implemented 2–5×)

| Rule area | Implementations | Consolidation target |
|---|---|---|
| Spelling / typo | `pipeline/rule_detector` (`data/spelling_dictionary.json`), `unified_pipeline.FastDetector`, `spelling_checker.py`, `new_pipeline` (wraps unified) | `pipeline/rule_detector` (single source of truth) |
| Subject–verb agreement | `pipeline/rule_detector._subject_verb_agreement`, `high_confidence_rules`, `unified_pipeline.NLPDetector`, `subject_verb_checker.py` | one consensus engine |
| Tense / verb form | `pipeline/rule_detector` (aux+participle, be+past, did+past, marker), `high_confidence_rules`, `unified_pipeline`, `tense_checker.py`, `intelligent_engine` | one canonical detector per category |
| Articles (a/an, the) | `pipeline/rule_detector._articles`, `high_confidence_rules`, `unified_pipeline`, `article_checker.py` | one canonical detector |
| Pronouns | `high_confidence_rules`, `unified_pipeline`, `pronoun_checker.py`, `pipeline/rule_detector._reflexive_subject` | one canonical detector |
| Prepositions | `pipeline/rule_detector._prepositions`, `high_confidence_rules`, `unified_pipeline` | one canonical detector |
| Punctuation | `pipeline/rule_detector._punctuation`, `unified_pipeline`, `punctuation_checker.py` | one canonical detector |
| Homophones | `pipeline/rule_detector._homophones` (`data/confusing_words.json`), `unified_pipeline`, `high_confidence_rules` | one canonical detector |
| Capitalization | `pipeline/rule_detector._capitalization`, `unified_pipeline`, `spelling_checker` | one canonical detector |

**Root cause:** `unified_pipeline.py` (7,389 lines) re-implements everything the
stdlib-only layers already cover. It exists to serve the *legacy local* Flask
frontend and the v4 hint path, but it can never run on Vercel.

---

## 4. Dead code — PROOFED by import-graph + grep cross-check

Nothing below is imported (transitively) from `api/index.py`, `app.py`, any
`tests/unit` suite, `tests/grammar_accuracy`, or `benchmark`. Verified
individually:

| File | Importers | Verdict |
|---|---|---|
| `analyze_fn.py`, `analyze_fn2.py`, `analyze_fn_deep.py`, `analyze_fn_detail.py`, `analyze_fn_round3.py`, `analyze_fns.py` | nobody | **REMOVE** (scratch analysis scripts) |
| `gen_tests.py` (79 KB test generator) | nobody | **REMOVE** |
| `correction_generator.py` | nobody | **REMOVE** |
| `correction_validator.py` | only dead `correction_generator`, `new_pipeline_runner` | **REMOVE** (cluster) |
| `extended_detectors.py` | only dead `new_pipeline_runner` | **REMOVE** |
| `fp_filter.py` | only dead `new_pipeline_runner`, `spacy_context` | **REMOVE** |
| `grammar_pipeline.py` | only dead files (correction_validator, extended_detectors, ...) | **REMOVE** |
| `new_pipeline_runner.py` | nobody | **REMOVE** (dead cluster root) |
| `linguistic_analyzer.py` | nobody | **REMOVE** |
| `nlp_analyzer.py` | only dead `linguistic_analyzer` | **REMOVE** |
| `spacy_detectors.py` | only dead `new_pipeline_runner` | **REMOVE** |
| `spacy_context.py` | only dead files | **REMOVE** |
| `archive/test_*.py` (96 debug scripts) | standalone, never imported | **REMOVE** (git history preserves) |
| `stderr.txt` (leftover dev-server log) | — | **REMOVE** |
| `results/*.json` | data artifacts of old evals | **REMOVE** (superseded by new datasets) |
| `benchmark/*.json` + `tests/grammar_accuracy/*.json` | outputs, not code | archive or remove; keep the *generators* |

**CAUTION — do NOT delete (look like dead, actually live):**
- `pipeline/__init__.py` — package marker (imported on `import pipeline`)
- `pipeline/ai_verifier.py` — live via relative `from . import ai_verifier`
- `tests/unit/conftest.py` — pytest fixture, auto-loaded
- `intelligent_engine.py`, `nlp_engine.py` — live via v1 `correction_engine` chain
- `contextual_engine.py`, `punctuation_checker.py`, `suggestion_validator.py` —
  live via `writing_engine`
- everything under `pipeline/`, `api/`, `data/*.json`

**Characterize the dead cluster**: `new_pipeline_runner`, `grammar_pipeline`,
`spacy_detectors`, `spacy_context`, `extended_detectors`, `fp_filter`,
`correction_validator`, `correction_generator`, `linguistic_analyzer`,
`nlp_analyzer`, `gen_tests`, `analyze_fn*` — a **second abandoned pipeline**
(the "v5 rugby" attempt). It imports nothing live and nothing live imports it.

---

## 5. File disposition plan

### 5a. REMOVE (verified dead, 12 root modules + 6 scratch + 96 archive scripts)
```
analyze_fn.py            analyze_fn2.py          analyze_fn_deep.py
analyze_fn_detail.py     analyze_fn_round3.py    analyze_fns.py
gen_tests.py             correction_generator.py  correction_validator.py
extended_detectors.py    fp_filter.py             grammar_pipeline.py
linguistic_analyzer.py   new_pipeline_runner.py   nlp_analyzer.py
spacy_context.py         spacy_detectors.py       stderr.txt
archive/ (96 test_*.py debug scripts)
results/ (3 json eval outputs) — superseded
```
→ use `git rm`; nothing is lost (history), and `archive/` content is also fully
retrievable from commits.

### 5b. RENAME (clarity)
| Now | Sugg. | Reason |
|---|---|---|
| `new_pipeline.py` (v4) | `pipeline_v4.py` | it is **not** "new", it is the v4 legacy evidence pipeline |
| `grammar_pipeline.py` | (removed, see 5a) | — |
| `pipeline/ai_core.py` → `pipeline/core.py` | optional | reduces "ai_" noise in the pipeline package; keep if you value `ai_core` name |

### 5c. MERGE (dedupe into ONE canonical pipeline)
1. `pipeline/rule_detector.py` becomes the **single** deterministic detector
   (absorb what it lacks from `high_confidence_rules` — which is already
   stdlib-only and can be kept as a pure function module).
2. `unified_pipeline.py` shrinks to: `ReadabilityEngine`, `ToneEngine`,
   `StyleEngine` (UI features) **only**; its spaCy detectors are folded into
   a single optional "spaCy hint source" behind `pipeline/ai_core` (local-only).
3. `new_pipeline.py` / `evidence.py` — fold the evidence/consensus logic into
   `pipeline/aggregator.py` (one consensus engine).
4. v1 checker chain (`grammar_engine.py` + 8 checkers + pos_tagger +
   spelling_checker + correction_engine + error_manager + intelligent_engine +
   nlp_engine) — keep only what `app.py` truly still calls (`/api/correct`,
   `/api/check-v2` fallback), then delete the rest.

### 5d. KEEP (production + tests + config)
```
api/index.py   app.py   vercel.json   requirements.txt   .github/workflows/test.yml
pipeline/ (core, rule_detector, aggregator, ai_analyzer, ai_verifier, feedback, __init__)
ai_validator.py   high_confidence_rules.py   evidence.py (until merged)
data/*.json   docs/   static/   templates/   README.md   .env.example
new_pipeline.py (until merged)   unified_pipeline.py (until trimmed)
scoring_engine.py   writing_engine.py   ai_engine.py
tests/unit/   tests/grammar_accuracy/*.py + test_cases.json   benchmark/generate_benchmark.py
```

---

## 6. Recommended execution order (drives Phases 2+)

1. **Delete the dead cluster (5a)** — zero behavioural risk; run the unit suite
   before and after to prove no import broke.
2. **`pipeline_v4.py` rename** (`new_pipeline.py`), update `app.py`, `test_api_v4.py`,
   `tests/grammar_accuracy/run_benchmark_v4.py`, `docs/*`.
3. Build the **single master pipeline** (`pipeline/master.py`, 15-step flow) with
   the standard error object; keep `/api/check` response shape backwards
   compatible (**DONE** — Phases 2–4).
4. Add **consensus engine** (rule + AI + optional spaCy-v4 evidence) and
   **offline local detectors as evidence providers**, not separate endpoints
   (**DONE** — Phases 5–8; `report_local_only` discovery mode backend-complete).
5. Build **datasets / gold test set**, then measure precision / recall / F1
   (Phases 12–14).
6. Remove verified `test_*.py`+dead modules at the **end**, after gold tests
   prove equal-or-better behaviour (Phase 16).

---

## 7. Metrics today (baseline, run before any merge)

- `pipeline/ai_core` unit suite: 169 tests, hermetic (`AI_PROVIDER=none`).
- Legacy benchmark (ARCHITECTURE_REDESIGN.md, v2): Precision 100%, FPR 0.0%,
  **Recall 23.4%**, F1 0.379 — the recall problem the AI-first pass exists to fix.
- Live `/api/check` (Vercel, Gemini): verified correct on
  "She go to school everyday and she dont like it" (4 errors all fixed, ai_used).

*(Precision/recall/F1 will be re-measured on the new gold test set in Phase 13.)*