# WriteMaster AI — Architecture

> Version: 2.0 (AI-first re-architecture, 2026)

This document records the **current (legacy) architecture**, the **audit findings**,
the **proposed AI-first architecture**, and the **migration plan**.

---

## 1. Current Architecture (as shipped v1)

The original project was a Flask web app with **four overlapping pipelines**, many
of them experimental. Only one was actually wired into the primary endpoints.

### 1.1 Pipeline zoo discovered in the audit

| Pipeline | Module | Status |
|---|---|---|
| v1 legacy | `grammar_engine.py` + `grammar_analyzer.py` + 8 checkers + `pos_tagger.py` | Active (`/api/correct`) |
| v2 writing engine | `writing_engine.py` | Legacy fallback only |
| v3 unified | `unified_pipeline.py` (~7,389 lines) | Active (`/api/check-v3`, `/api/fix-all`) |
| v4 evidence+AI | `new_pipeline.py` + `ai_validator.py` | **Primary** (`/api/check`, `/api/check-v4`, `/api/validate`) |
| new-runner (dead) | `new_pipeline_runner.py` + `grammar_pipeline.py` + `spacy_detectors.py` + `extended_detectors.py` | Dead — no production imports |
| AI validator | `ai_validator.py` | Active — Gemini/OpenAI/Anthropic judge for v4 |

### 1.2 The v4 data flow (current primary)

```
USER TEXT
  → preprocessor.protect()
  → spaCy doc (en_core_web_sm)
  → FastDetector + NLPDetector + DataDrivenDetector + HighConfidenceDetector
  → EvidenceStore merge (evidence.py)
  → ContextEngine suppression
  → FalsePositiveFilter + dialect filter
  → deduplicate
  → CorrectionValidator
  → AIValidator (Gemini per-candidate judge)   [optional]
  → confidence sort  →  /api/check response
```

### 1.3 Module inventory

**Production core (keep):**

| Module | Purpose |
|---|---|
| `app.py` | Flask routes + response shaping |
| `unified_pipeline.py` | spaCy detection engines (v3), readability/tone/style |
| `new_pipeline.py` | v4 orchestration (`check_v4`) |
| `ai_validator.py` | Multi-provider AI judge; Gemini/OpenAI/Anthropic/Ollama; strict JSON parsing; `.env` loader |
| `high_confidence_rules.py` | stdlib-only high-confidence rules (irregular past, homophones, comparatives) |
| `evidence.py` | multi-source evidence merging + confidence |
| `scoring_engine.py` | correctness/clarity/engagement/delivery scores |
| `writing_engine.py` | readability, custom dictionary endpoints |
| `ai_engine.py` | rephrase / tone / enhance / synonyms (heuristic) |
| `grammar_engine.py` | `/api/correct` auto-correct |
| `data/*.json` | spelling/homophone/grammar/vocab dictionaries |

**Legacy checkers (used only through v1 chain):** `pos_tagger.py`, `preprocessor.py`,
`spelling_checker.py`, `subject_verb_checker.py`, `tense_checker.py`, `article_checker.py`,
`pronoun_checker.py`, `sentence_structure_checker.py`, `semantic_checker.py`,
`punctuation_checker.py`, `context_checker.py`, `correction_engine.py`,
`error_manager.py`, `grammar_analyzer.py`.

**Dead / experimental (no production imports):**
`grammar_pipeline.py`, `new_pipeline_runner.py`, `spacy_detectors.py`,
`extended_detectors.py`, `spacy_context.py`, `fp_filter.py`,
`correction_validator.py`, `correction_generator.py`, `linguistic_analyzer.py`,
`nlp_analyzer.py`, `intelligent_engine.py`, `nlp_engine.py`, `contextual_engine.py`,
`suggestion_validator.py`, `suggestion_aggregator.py`, plus ~97 root `test_*.py`
debug scripts and `tests/grammar_accuracy/*` standalone benchmark scripts.

### 1.4 Known defects found during audit

1. `"evry" → "very"` — the fuzzy spellchecker produces a **wrong** correction for a
   common mistake (should be `"every"`). Only Gemini's full-context analysis can fix this.
2. `requirements.txt` was missing `pyspellchecker`, `python-dotenv`, `pytest`,
   `requests` → `import unified_pipeline` failed on a fresh checkout.
3. NLTK corpora (`words`, `wordnet`) were required but not downloaded.
4. Readability response was missing `syllables`, `complex_word_pct`, `explanation`
   that the frontend reads (UI showed `undefined`).
5. `/api/check` returned 503 whenever the v4 import failed → no graceful fallback.
6. Gemini validated **one candidate at a time** and never analyzed the complete
   sentence as a whole.

---

## 2. Proposed Architecture (AI-first, neural-network brain)

The rule + spaCy layer becomes a **candidate generator**. The **Gemini neural
network** becomes the primary reader/decider. All corrections go through a
**verification pass**. Nothing is fabricated when AI is unavailable.

```
USER TEXT
  → TEXT PREPROCESSOR              (normalize, protect tokens)
  → RULE + SPA CY DETECTORS        (candidate generation, dependency-light rules)
  → CANDIDATE AGGREGATOR           (dedup, merge, span alignment)
  → GEMINI AI ANALYSIS             (full-context read → strict JSON)
       - is there an error?  which words?  what is safe correction?
  → GEMINI VERIFICATION            (2nd pass: accept / reject / uncertain)
  → CONFIDENCE / QUALITY FILTER    (HIGH/MEDIUM/LOW → only HIGH auto-corrects)
  → FINAL CORRECTIONS              (grammar vs spelling vs style separated)
  → USER FEEDBACK                  (accept/reject)
  → FEEDBACK QUALITY GATE          (dedup + format validation)
  → data/feedback/                 (verified, frozen — never used to retrain live)
```

### 2.1 New modules

| Path | Responsibility |
|---|---|
| `pipeline/rule_detector.py` | Dependency-light rule candidates (stdlib + `data/` only; runs on Vercel) |
| `pipeline/aggregator.py` | Merge rule + v4/spaCy candidates, align spans, dedup, score confidence |
| `pipeline/ai_analyzer.py` | Gemini full-text analysis → strict JSON contract |
| `pipeline/ai_verifier.py` | Gemini second-pass verification (accept/reject/uncertain) |
| `pipeline/ai_core.py` | Orchestrator: preprocess → candidates → analyze → verify → filter |
| `pipeline/feedback.py` | Feedback store + quality gate → `data/feedback/feedback.jsonl` |
| `api/index.py` | Vercel serverless entry (light path: rule detector + Gemini only) |

### 2.2 Gemini JSON contract (analysis)

```json
{
  "original_text": "She go to school evry day.",
  "corrected_text": "She goes to school every day.",
  "errors": [
    {"wrong": "go", "correct": "goes", "type": "subject_verb",
     "explanation": "The singular subject 'She' requires 'goes'.", "confidence": 0.99},
    {"wrong": "evry", "correct": "every", "type": "spelling",
     "explanation": "'evry' is a misspelling of 'every'.", "confidence": 0.99}
  ],
  "grammar_status": "errors_found",
  "meaning_preserved": true
}
```

For a correct sentence: `"errors": []`, `"grammar_status": "correct"`.

### 2.3 Verification contract

Single-field verdict per check (not per candidate):

```json
{"decision": "accept", "confidence": 0.99, "reason": "Correction is safe and preserves meaning."}
```
`decision` ∈ `accept | reject | uncertain`. The system **never** forces an uncertain correction.

### 2.4 Confidence rules

| Level | Meaning |
|---|---|
| HIGH | rule + Gemini agree, verification accepted, meaning preserved |
| MEDIUM | Gemini verified but single source / lower model confidence |
| LOW | detectors disagree or context ambiguous → **reported but never auto-applied** |

### 2.5 Fallback behavior (no AI key / offline)

- Rule generator alone decides; conservative thresholds; only HIGH rule confidence is reported.
- `/api/check` falls back to the deterministic v4 pipeline (never a 503).
- Vercel path degrades to rule-detector + readability (no spaCy bundled).

---

## 3. Migration Plan

| Phase | Work |
|---|---|
| 1 (done) | Environment: install pyspellchecker/pytest/requests/dotenv, download NLTK corpora, verify `check_v4` runs |
| 2 | Build `pipeline/` package with rule/aggregate/AI analyze/AI verify/feedback |
| 3 | Repoint `/api/check` to `ai_core` (AI-first, v4 fallback); add `/api/feedback` |
| 4 | Offline test suite with mocked Gemini |
| 5 | Vercel light path, GitHub Actions, docs, README |
| 6 | Archive obsolete root `test_*.py` debug scripts → `archive/` |
| 7 | Validate, commit, push |

### 3.1 File retention policy

- **Never delete** anything still imported by `app.py`, `new_pipeline.py`, or
  `unified_pipeline.py`.
- Move the 97 root debug scripts and stale benchmark scripts to `archive/` (git
  history preserves them; nothing is lost).
- Keep `data/` JSON dictionaries (spelling/homophones/grammar rules).

---

## 4. Data Flow for `They is went to school evry day`

1. Preprocessor: protect nothing special, sentence-split.
2. Candidates:
   - Rule: `is went → went` (be + past-participle + an intransitive main verb → delete `is`).
   - Rule/spelling + context: `evry → every` (spelling list + the AI resolves ambiguity vs `very`).
3. Aggregator: aligns spans, 2 candidates, both HIGH.
4. Gemini analysis (full sentence read): outputs the two errors above.
5. Gemini verification: accepted, meaning preserved.
6. Final:
   - `"is went" → "went"` (type grammar/tense)
   - `"evry" → "every"` (type spelling)
7. User accepts/rejects → feedback record → quality gate → `data/feedback/`.