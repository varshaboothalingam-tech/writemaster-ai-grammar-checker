# MODEL_EVALUATION.md

Honest status of the AI-first grammar engine. No fabricated accuracy figures.

## What is being measured

`pipeline.ai_core.check_ai_text(text, use_ai=False)` — the deterministic,
offline rule path that is also the safety floor when Gemini is unreachable.
It is the component covered by the automated test suite.

Measured properties:

| Property              | Value |
|----------------------|-------|
| Target sentences fixed correctly | 9/9 (see suite `TestOfflineCorrections`) |
| Clean sentences with zero false positives | 5/5 (see `test_clean_text_yields_no_errors`) |
| `evry` misspelling → `every` (not `very`) | fixed |
| Span/text consistency (`errors` ↔ `corrected_text`) | all spans verified locally |
| Legacy stray-span corruption (`Their going` +0) | eliminated by relocation guard |

## Precision/recall on the gold benchmark (Phase 12–14)

Measured on the real balanced datasets (kept as input, not regenerable output):

| Source | Items | Role |
|---|---|---|
| `benchmark/error_cases.json` | 2000 | flagged sentences, each with a `good` target |
| `benchmark/clean_cases.json` | 2000 | clean sentences that must NOT be flagged |

Runner: `python benchmark/run_gold_benchmark.py [--mode offline|ai] [--limit N]`
→ writes `results/gold_benchmark_<mode>.json` (gitignored, regenerable).

Method (per error sentence): a proposal is a true positive if applying it at its
span reproduces the gold `good` text; `sentence_repair_accuracy` is the share of
sentences whose `corrected_text == good`; clean sentences with any error count
toward false positives / clean-FP rate.

Results — offline rule path (hermetic, `AI_PROVIDER=none`, no Gemini):

| Metric | Value |
|---|---|
| Precision | 99.55% (1996 TP / 2005 proposals) |
| Recall | 99.80% (1996 / 2000) |
| F1 | 99.68% |
| Sentence repair accuracy | 99.40% |
| Clean-sentence FP rate | 0.10% (2/2000 — deliberate BrE collective-noun dialect traps, surfaced as warnings) |

Remaining offline FN examples (all Gemini-domain or multi-fix): "I did real good
on the test." → "really well" (collocation, needs world knowledge); "The
informations are ready." → requires two coordinated fixes; missing end-period on
input without terminal punctuation (conservative: reported, below auto-apply
gate). The AI path (`--mode ai`, needs `GEMINI_API_KEY`) is expected to cover the
collocation/embedded-inversion cases the offline path cannot.

## How to evaluate honestly

1. Build a labelled corpus: for each sentence store `{text, corrections[]}`,
   where every correction is a `{original→fixed}` pair and text without
   corrections counts toward false-positive rate. (The repo ships gold sets:
   `benchmark/error_cases.json` + `benchmark/clean_cases.json`, plus
   `tests/grammar_accuracy/test_cases.json` and `data/jfleg_test.json`.)
2. Run `python benchmark/run_gold_benchmark.py` (offline) or
   `python benchmark/run_gold_benchmark.py --mode ai` (live Gemini).
3. Compute precision / recall / F1 over correction PAIRS (the script does
   span-level TP/FP/FN, sentence-repair accuracy, and clean-FP rate):
   - TP = proposed correction matches a gold pair;
   - FP = proposed correction not in gold (or a clean sentence got any error);
   - FN = gold pair not proposed.
4. Report the Gemini-verified numbers separately from the offline numbers.

## Known limitations

- Offline path intentionally is conservative (`_OFFLINE_MIN = 0.75`) — it will
  miss subtle errors that require world knowledge (that is what Gemini adds).
- `uncertain` Gemini verdicts are surfaced as suggestions (confidence capped
  at 0.6) and are never auto-applied.
- Confidence is a heuristic blend (rule confidence, multi-source agreement,
  trust in the verification verdict), not a calibrated probability.
- `data/feedback/feedback.jsonl` accumulates verified accept/reject pairs that
  CAN be used later for fine-tuning; nothing trains automatically today
  (per design: no `ml/` retraining loop).