# ARCHITECTURE_REDESIGN.md — WriteMaster AI v3

## Current State (v2)
- **Precision**: 100% (1,000-sentence benchmark)
- **FPR**: 0.0%
- **Recall**: 23.4%
- **F1**: 0.379
- **Latency**: ~1.5s per sentence (too slow)

## Problems
1. **Massive duplication**: SVA implemented 4x, tense 4x, pronouns 3x, articles 4x
2. **Two parallel pipelines** that don't share results
3. **No ML models** — everything is rule-based or spaCy parse-based
4. **Low recall** — misses 76.6% of actual errors
5. **Multiple spaCy loads** (4 separate nlp = spacy.load() calls)
6. **No text preprocessor** — URLs, code, names checked as normal English
7. **No semantic engine** — can't distinguish "I wanted to" from "I want to"
8. **No correction generator** — corrections are hardcoded per-detector
9. **No style/tone separation** — style suggestions mixed with grammar errors

## Proposed Architecture (v3)

```
USER TEXT
  ↓
TEXT PREPROCESSOR (protect URLs, code, names, etc.)
  ↓
TOKENIZER + SENTENCE SEGMENTER (single spaCy load)
  ↓
LINGUISTIC PARSER (POS, NER, deps, morphology)
  ↓
TIER 1: FAST CANDIDATE DETECTION (spelling, typos, repeated words)
  ↓
TIER 2: NLP-BASED DETECTION (SVA, tense, articles, pronouns)
  ↓
TIER 3: CONTEXT ENGINE (is this actually wrong in THIS context?)
  ↓
TIER 4: SEMANTIC ENGINE (meaning preservation check)
  ↓
ERROR CLASSIFICATION + CONFIDENCE SCORING
  ↓
FALSE-POSITIVE FILTER (15+ strategies)
  ↓
CORRECTION GENERATION (minimal edits)
  ↓
CORRECTION VALIDATION (re-parse corrected text)
  ↓
DEDUPLICATION + SUGGESTION RANKING
  ↓
STYLE/CLARITY/TONE/READABILITY (separate system)
  ↓
FINAL USER SUGGESTION
```

## Key Design Principles
1. **Single spaCy load** — shared across all components
2. **Detect ≠ Correct** — separate detection from correction generation
3. **Grammar ≠ Style** — never show style as grammar error
4. **Confidence-gated** — suppress anything below threshold
5. **Regression-safe** — every FP becomes a permanent test

## Migration Plan
- Phase 1-2: Architecture redesign + benchmark expansion
- Phase 3-4: Preprocessor + unified pipeline core
- Phase 5-6: ML detection + context engine
- Phase 7-8: Correction validation + real-time optimization
- Phase 9-10: Final benchmark + cleanup
