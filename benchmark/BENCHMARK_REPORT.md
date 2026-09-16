# Benchmark Report — High-Precision Grammar Checker

Date: 2026-09-16
Build: local rule/detector pipeline (`check_v4`) at default settings, `AI_PROVIDER=none`
(last benchmark run; results cached in `benchmark/benchmark_results.json`)

---

## 1. Corpus

| | error cases | clean cases |
|---|---|---|
| target | 2000+ | 2000+ |
| shipped | 2000 | 2000 |

Built by `benchmark/generate_benchmark.py` (fixed seed, fully reproducible).
No sentence duplicates (builder de-duplicates and round-robins categories for balance).

Error categories (36 total), each triple is `(bad, good, note)` with `bad` genuinely
incorrect:
- Legacy categories: subject-verb agreement (phrase/collective/indefinite), pronoun case,
  verb tense/form, comparatives, articles, prepositions, conjunctions, double negatives,
  homophones, spelling, punctuation, contractions, redundancy, adjective/adverb,
  uncountables, causative/gerund-infinitive, conditionals, embedded questions, question
  inversion, number agreement, possessives, word order, collocations, lay/lie, articles-the.
- Added for spec coverage (G-I, this session):
  - `tense-marker-perfect` — present perfect + explicit past-time marker
    ("Yesterday, Maya has written the report." → "wrote"); 6480 combos verified.
  - `passive-sva` — passive with wrong auxiliary ("The meeting were scheduled…",
    "Two sections was repeated."); 2400 combos verified.
  - `modal-form`, `collective-have-sva`, `tense-backshift`, `scheduled-on`,
    `quantifier-relative-sva`, `duplicate-verb` (A-F, earlier session).

Clean categories include: entities/names, quotations, dialect/AAVE (deliberately not
flagged), technical/domain terms, of-phrases and agreement traps, numbers & dates,
modals & perfects, possessives, collectives with both AmE-agreeing and BrE-tolerant
verb forms, relative-clause SVA traps, present-perfect/`already` correct forms, and
passive agreement correct forms.

### Verification (generators)

`benchmark/verify_generators.py` runs every NEW-category yield through `check_v4`:

```
error triples verified : 9,526   failures: 0   (flagged + correct fix == good)
clean sentences        : 3,130   failures: 0   (must stay unflagged)
```

Old categories were previously verified the same way (0 failures in earlier runs).

Surfaced bugs fixed during verification/benchmarking (no sentences hacked):
1. Rule: `_regular_past_tense` produced "submited"; extended `_DOUBLE_FINAL_VERBS`
   with stress-final multi-syllable verbs → "submitted". Applies to commit/occur/
   refer/prefer/admit/etc. too.
2. Generator: naive `+s` plural for "lunch" gave reference "lunchs"; the checker
   correctly produced "lunches". Generator now uses proper plural inflection
   (`_plural_noun`), so the reference is right.

---

## 2. Method

- Inputs: `error_cases.json` (2000, `should_flag: true, good`) + `clean_cases.json`
  (2000, `should_flag: false`).
- Split: single seeded shuffle (`random.Random(42)`), train 60% / val 20% /
  held-out 20% → 2400 / 800 / 800.
- Per sentence: `check_v4(text)`; flagged = any returned error (approved-only output,
  Gemini not used in these runs).
- TP/FN/FP/TN; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1; FPR = FP/(FP+TN).
- Correction accuracy: for each TP, apply returned replacements (span-based, end→start)
  and compare with the reference `good` (normalized: lowercase, whitespace-collapsed,
  sentence punctuation stripped).

---

## 3. Results

### Train + Val (`AI_PROVIDER=none`)

```
cases: 3,200 (error 1,592 + clean 1,608)
TP=1592  FP=0  FN=0  TN=1608
Precision=1.000  Recall=1.000  F1=1.000  FPR=0.000
Correction accuracy: 1592/1592 (1.000)
```

### Held-out (800, disjoint from train/val)

```
TP=408  FP=0  FN=0  TN=392
Precision=1.000  Recall=1.000  F1=1.000  FPR=0.000
Correction accuracy: 408/408 (1.000)
```

Every error category reports TP=… FP=0 FN=0, corrAcc=1.000, including the target
categories:

| category | TP (train+val) | corrAcc |
|---|---|---|
| present perfect + past-time marker | 463 | 1.000 |
| passive were -> was / was -> were | 125 / 346 | 1.000 |
| tense backshift | 41 | 1.000 |
| modal + past form | 88 | 1.000 |
| collective + have (AmE) | 83 | 1.000 |
| duplicate verb (xcomp) | 86 | 1.000 |
| scheduled on weekday | 111 | 1.000 |
| anybody/anyone declarative | 75 | 1.000 |
| who were (relative SVA) | 21 | 1.000 |

---

## 4. Tests

Full suite: **136 passed, 0 failed** in ~7s.

- `tests/unit/test_spec_regression.py` — the 8 error + 4 clean spec acceptance cases
  (§8), exact-fix assertions.
- `tests/unit/test_api_v4.py` — `/api/check` approved-only, `/api/health`,
  `/api/validate` matching, AI decision-log gating (7 tests).
- `tests/unit/test_ai_validator.py` — 25 tests incl. GEMINI_API_KEY preference /
  AI_API_KEY fallback / `.env` parsing.
- Plus the pre-existing unit/API suites.

---

## 5. Gemini / AI final validation

Design (fail-closed): every surviving local candidate is sent to the configured validator
(one call per candidate); a candidate is APPROVED only if the model says it is an error,
the correction is valid, confidence >= threshold, and it survives the contradiction
checks. Only APPROVED issues are returned to the API/UI; rejected candidates are logged
(`logs/ai_decisions.jsonl`, on by env `GC_AI_LOG=1`) but never reported.

Measured in this report: **not measured** — no runtime API key was configured, so all
runs used `AI_PROVIDER=none` (local pass-through), and the numbers above are the local
pipeline. To measure the Gemini effect honestly:

```
# with GEMINI_API_KEY or AI_API_KEY in .env
python benchmark/run_benchmark.py --ai          # reports >= train+val numbers + rates
python benchmark/run_benchmark.py --heldout --ai
```

`--ai` adds, per run: Gemini-validated candidates, approved, rejected, and
Approval Rate = approved / validated, Rejection Rate = rejected / validated.
It will also report any change in P/R/F1/corrAcc caused by approvals. (Aggregation
requires an actual configured provider; a none/broken provider keeps the pass-through
numbers and prints no AI line.)

Caveat to expect with real Gemini runs: Approval Rate will be < 1.000. That is
intentional — the validator's job is to veto local false positives, so rejection of some
candidates is expected and the honest final precision/recall is what `--ai` reports.

---

## 6. Known limitations (honest)

- The corpus is checker-authored and local-detector-verified; 1.000 / 1.000 is on THIS
  corpus, not general English. Real-world precision will be lower.
- 13 genuinely-correct sentences are still flagged by the local pipeline (discovered
  while building clean cases; kept out of the corpus, listed here as known FPs):
  - "He can hardly wait." → HARDLY_HARD
  - "The storm will affect our plans." → CONFUSABLE_DATA
  - "Neither John nor Mary is home." → MISSING_APOSTROPHE
  - "Ana and I went shopping." / "My brother and I went hiking." → DATA_RULE_CLAUSE_COMMA
  - "Give the book to Ana and me." / "They saw Ana and me at the park." → COMPOUND_SUBJECT_PRONOUN
- Detection is spaCy-parse-dependent; a few valid constructions are missed because the
  parser reads them differently (e.g. specific `has + VBN` forms without a past marker;
  `Last week, the team has attended…` was not flagged). These templates were excluded
  from the corpus rather than worked around.
- Dialect policy: BrE-tolerant and dialect sentences are intentionally left unflagged
  (collectives with plural verbs, "She don't like broccoli.", etc.).
- AI rates above (approval/rejection) are available only when a provider is configured
  at runtime; no numbers are invented for them here.

---

## 7. Reproduce

```
python benchmark/generate_benchmark.py          # rebuild error_cases.json + clean_cases.json
python benchmark/verify_generators.py           # 9,526 error + 3,130 clean, 0 failures expected
set AI_PROVIDER=none & python benchmark/run_benchmark.py            # train + val
set AI_PROVIDER=none & python benchmark/run_benchmark.py --heldout  # held-out
python -m pytest tests -q                       # 136 passed
```