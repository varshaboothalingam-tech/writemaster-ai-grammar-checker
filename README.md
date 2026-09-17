# WriteMaster AI — Grammar Checker & Writing Assistant

AI-first English grammar checker. A Gemini neural-network **brain** reads the
full text, proposes corrections, verifies them in a second pass, and only then
commits them — with a deterministic rule engine (spaCy + spelling + grammar
rules) as the fallback and candidate feeder so the app **never breaks when the
AI is offline**. Built with Flask.

## How it works

```
TEXT
  → pipeline/rule_detector + v4 hints        (candidate generators)
  → pipeline/aggregator                      (dedup, overlap resolution,
                                              span-relocation guard,
                                              multi-source confidence bonus)
  → Gemini ANALYSIS   (full-context, strict JSON, local offsets only)
  → Gemini VERIFICATION (accept / reject / uncertain)
  → confidence filter → final errors + corrected_text
  → user accept/reject → data/feedback/feedback.jsonl   (quality-gated)
```

- **Safety first:** offsets are recomputed locally; a stray model/legacy span
  can never corrupt the sentence. `uncertain` corrections are suggestions only
  (never auto-applied).
- **Offline floor:** rule candidates with confidence ≥ 0.75 are applied
  without AI; text is otherwise left untouched.
- **No retraining loop:** feedback is accumulated for a future clean
  fine-tuning dataset (`docs/MODEL_EVALUATION.md`).

## Features

- **Spelling** — 373 misspellings + homophone disambiguation (their/there,
  your/you're, its/it's...)
- **Grammar rules** — subject-verb agreement, double negatives, a/an, verb
  form (is went → went), aux + past participle (has went → has gone),
  capitalization, repeated words, punctuation
- **Tense & verb-form checking**, article, pronoun, preposition, word-order
- **Style suggestions** — passive voice, filler words, weak vocabulary
- **AI rephrasing, tone analysis, summarize, synonym finder**
- **Flesch-Kincaid readability** score and grade level
- **One-click Auto-Correct**, per-issue fixes, custom-dictionary, ignore
- **User feedback API** (`POST /api/feedback`) with a quality gate
- **Clean responsive UI** — Bootstrap 5, Chart.js doughnut, expandable cards,
  check history (localStorage)

## Tech Stack

- **Backend:** Python, Flask 3.x, Pipeline: stdlib-only rules + optional Gemini API; legacy deterministic engines on spaCy + NLTK
- **Frontend:** HTML5, Bootstrap 5.3.2, Font Awesome, Chart.js, vanilla JS

## Getting Started

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5001 in your browser.

Set `GEMINI_API_KEY` in `.env` (see `.env.example` layout) to enable the
Gemini brain; without it, the app runs in conservative offline mode.

Command return shape (clean AI contract on `/api/check`):

```json
{
  "success": true,
  "original_text": "She go to school evry day.",
  "corrected_text": "She goes to school every day.",
  "errors": [{"wrong": "go", "correct": "goes", "type": "subject_verb",
              "start": 4, "end": 6, "confidence": 0.9,
              "explanation": "..."}],
  "issues": [...], "readability": {...}, "scores": {...},
  "meta": {...}, "grammar_status": "errors_found",
  "processing_time_ms": 48
}
```

## API Endpoints

| Endpoint                  | Description                                  |
|---------------------------|----------------------------------------------|
| `POST /api/check`         | AI-first full grammar/spelling/style analysis|
| `POST /api/check-v4`      | Frontend alias (same AI engine + legacy shape) |
| `POST /api/fix-all`       | Apply all high-confidence corrections        |
| `POST /api/feedback`      | Record accept/reject for a correction        |
| `POST /api/tone`          | Tone detection                               |
| `POST /api/rephrase`      | Sentence rephrasing                         |
| `POST /api/synonyms`      | Synonym lookup                               |
| `POST /api/readability`   | Readability + grade-level scores             |
| `GET  /api/health`        | Health + capability probe                    |
| `POST /api/correct`       | Legacy auto-correct                          |

## Serverless (Vercel)

`api/index.py` + `vercel.json` deploy the stdlib-only pipeline (rules +
Gemini) as a serverless function:

```bash
pip install vercel
vercel --prod
```

Set `GEMINI_API_KEY` in the Vercel project environment.

## Project Structure

See `docs/FILE_STRUCTURE.md` for the active production surface.

```
├── app.py                  # Flask routes
├── pipeline/               # AI-first pipeline (stdlib-only)
│   ├── rule_detector.py    #   rule candidates
│   ├── aggregator.py       #   merge / dedup / overlap / relocation
│   ├── ai_analyzer.py      #   Gemini analysis
│   ├── ai_verifier.py      #   Gemini verification
│   ├── ai_core.py          #   check_ai_text()
│   └── feedback.py         #   quality-gated feedback store
├── api/index.py            # Vercel serverless entry
├── data/                   # dictionaries + feedback/ (gitignored)
├── docs/                   # ARCHITECTURE, FILE_STRUCTURE, MODEL_EVALUATION
├── tests/                  # pytest unit suite + grammar_accuracy harness
├── static/ + templates/    # frontend (contract unchanged)
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/unit -q        # 169 tests, hermetic (no network)
```

## Sample Text to Try

> "Their going to the store tommorow. I definately need to get their early."

## License

College mini project — free to use for educational purposes.