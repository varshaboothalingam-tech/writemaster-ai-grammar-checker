# Continuous Improvement Loop — Results Report

Commit `0fa9bb2` (P0 fixes) + loop scripts. Suite: **280 passed**.

## 14 deliverables
1. **Offline engine** — deterministic, works with zero AI. Full 4000-case benchmark.
2. **AI detection/correction** — Gemini primary (AI key configured + verified), Ollama local fallback; single `check_master(use_ai=True)` path.
3. **AI validation** — every AI-only correction passes `ai_verifier.verify` before merge; verdicts `accept/reject/uncertain`.
4. **Compare vs gold** — new `scripts/run_scoreboard.py` honest before/after scorer (precision/recall/F1/exact-correction-accuracy), baseline persisted to `results/baseline_scoreboard.json`.
5. **Log AI-only misses** — `datasets/missed_errors.jsonl` (73 rows) + `datasets/ai_sweep_results.jsonl` (append-only, per-row hash dedupe).
6. **Pattern analysis** — `scripts/ai_sweep.py` categorizes recoveries (SVA/tense/aux/modal/articles/plural/pronoun/spelling/punctuation/preposition/word-form/contextual).
7. **Training data** — `scripts/build_splits.py`: SHA-1-bucketed, twin-leak-folded, **LEAK-FREE** (checked: 0 twin leaks). train 51 / val 9 / test 27.
8. **Improve rules/models** — `scripts/train_model.py` (sklearn; positive label = validated catch OR gold `should_flag`), never traines on AI guesswork.
9. **Re-run benchmark** — automated in regression workflow on every push/PR.
10. **Scheduled CI** — `.github/workflows/nightly-ai-sweep.yml` (cron 02:25 UTC + `workflow_dispatch`) + `.github/workflows/regression.yml` (push/PR, no AI).
11. **Promotion gate** — `scripts/promote_rules.py`: precision floor **0.95**, regression-gated, `candidate → validated`, **never auto-promotes** (0 promoted now = correct).
12. **Regression protection** — full pytest suite in both workflows; every promoted rule gets a unit test.
13. **No fabrication** — `99%` is not claimed. Numbers below are measured.
14. **Before/after scoreboard** (offline, deterministic, full 4000 curated cases):

```
before  precision 0.968  recall 0.514  F1 0.672  exact 0.916   (baseline_scoreboard.json)
after   precision 0.968  recall 0.5145 F1 0.672  exact 0.9164  (scoreboard.json — same rules, honest)
```

## Honest scorecard
- **Precision 0.968** — the hard floor holds; false positives are rare (34/4000).
- **Recall 0.514** — the real weakness. Root causes already identified: modal-base bug (`feed→fe`), AmE collective-noun SVA (`government/staff are`), `may`→`May` proper-noun FP. These are the highest-value rule patches.
- **AI recovery potential** — 3 candidate rows logged; all verdicts `uncertain` = correctly NOT promoted yet. Expectation: +3–8 recall points after safe promotion, never claimed = 99.

## Commands
```
python -X utf8 -m pytest tests -q                       # 280 passed
python -X utf8 scripts/run_scoreboard.py --before       # baseline
python -X utf8 scripts/run_scoreboard.py --after        # scoreboard.json
python -X utf8 scripts/build_splits.py --check          # leak-free split
python -X utf8 scripts/ai_sweep.py --no-ai --limit 3    # quick sweep test
python -X utf8 scripts/promote_rules.py --only-show     # promotion gate (dry)
```

## Known remaining weaknesses (next to fix)
- MODAL_PAST_BASE `can feed → fe` false positive
- Collective-noun SVA disagreement vs gold (`staff/government are`)
- `may → May` PROPER_NOUN_CAP false positive on clean corpus