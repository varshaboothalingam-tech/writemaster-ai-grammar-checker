# FILE_STRUCTURE.md

Active production surface. Everything not listed is either legacy (see
`docs/ARCHITECTURE.md` "Module inventory") or a one-off debug script and may
be moved to `archive/` later.

```
.
├── app.py                        # Flask app — routes 1..~700
│                                  #   /api/check, /api/check-v4  → AI-first pipeline
│                                  #   /api/feedback              → quality-gated store
│                                  #   legacy: /api/validate, /api/add-to-dictionary,
│                                  #   /api/ignore-all, /api/fix-all, /api/tone,
│                                  #   /api/readability, /api/rephrase, /api/enhance,
│                                  #   /api/synonyms, /api/summarize, /api/correct,
│                                  #   /api/health, /
├── pipeline/                     # ★ NEW AI-first pipeline (stdlib-only, serverless-safe)
│   ├── __init__.py               #   package marker (module list only)
│   ├── rule_detector.py          #   rule-based CANDIDATES: spelling dict (373), repeated
│   │                              #   letters/words, high-confidence rules, SVA, be-agreement,
│   │                              #   aux+participle, be+past, did+past, tense marker, a/an,
│   │                              #   prepositions, homophones (contextual), plurals,
│   │                              #   missing-aux, reflexive subject, capitalization, punctuation
│   ├── aggregator.py             #   dedup (merges sources), overlap resolution, multi-source
│   │                              #   bonus (+0.04), span relocation guard, apply_corrections
│   ├── ai_analyzer.py            #   Gemini ANALYSIS prompt + strict-JSON parse + local offsets
│   ├── ai_verifier.py            #   Gemini VERIFICATION (accept / reject / uncertain)
│   └── ai_core.py                #   check_ai_text() orchestration + confidence gates
│   └── feedback.py               #   FeedbackStore → data/feedback/feedback.jsonl
├── ai_validator.py               # legacy Gemini transport reused by pipeline (provider, .env)
├── high_confidence_rules.py      # deterministic stdlib-only rule source (shared)
├── new_pipeline.py               # v4 legacy pipeline (check_v4) — hint source for pipeline
├── unified_pipeline.py           # v3 legacy pipeline (check_text, ReadabilityEngine, ToneEngine)
├── grammar_engine.py             # legacy: GrammarChecker, auto_correct
├── ai_engine.py                  # legacy: AIEngine (tone/rephrase/enhance/summary)
├── writing_engine.py             # legacy: WritingAnalyzer (spelling, readability)
├── scoring_engine.py             # compute_scores()
├── data/
│   ├── spelling_dictionary.json  #   373 misspellings (incl. evry→every)
│   ├── confusing_words.json      #   homophones (their/there/they're, ...)
│   ├── feedback/                 #   NOT tracked — quality-gated records (feedback.jsonl)
│   └── ...                       #   other dictionaries used by legacy engines
├── api/index.py                  # ★ Vercel serverless entry (stdlib-only + Gemini)
├── vercel.json                   # @vercel/python build + /api/* route
├── .github/workflows/test.yml    # CI: install deps, download model/corpora, pytest
├── examples/                     # sample texts used by docs
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FILE_STRUCTURE.md
│   └── MODEL_EVALUATION.md
├── tests/
│   ├── unit/                     # pytest (AI_PROVIDER=none → hermetic)
│   │   ├── conftest.py
│   │   ├── test_api_v4.py              # legacy API contract
│   │   ├── test_ai_pipeline.py         # ★ AI pipeline + feedback + relocation
│   │   └── ...
│   └── grammar_accuracy/         # benchmark harnesses (run_tests.py etc.)
├── static/                       # frontend assets (js/css) — unchanged contract
├── templates/index.html          # frontend — unchanged contract
├── requirements.txt
├── .gitignore                    # .env, data/feedback/, __pycache__, ...
└── README.md
```