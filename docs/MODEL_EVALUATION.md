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

## Precise/recall on the AutoCorrect benchmarks

Not published. The repo contains benchmark harnesses under
`tests/grammar_accuracy/` that were run against earlier heuristic pipelines
with widely varying (and noisy) generators; they do not reflect the new AI
pipeline, so re-publishing those numbers would be misleading. Before quoting
precision/recall/F1 for the new engine, run a controlled, hand-labelled
corpus — see "How to evaluate" below.

## How to evaluate honestly

1. Build a labelled corpus: for each sentence store `{text, corrections[]}`,
   where every correction is a `{original→fixed}` pair and text without
   corrections counts toward false-positive rate.
2. Run: `from pipeline.ai_core import check_ai_text; res = check_ai_text(text, use_ai=False)`
   (offline) or with `use_ai=True` + a live `GEMINI_API_KEY`.
3. Compute precision / recall / F1 over correction PAIRS:
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