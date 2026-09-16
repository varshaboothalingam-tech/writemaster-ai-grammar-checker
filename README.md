# WriteMaster AI — Grammar Checker & Writing Assistant

AI-powered English grammar checker that detects **spelling, grammar, punctuation, tense, verb-form, and sentence-structure** errors and provides corrected suggestions with full explanations. Built with Flask and rule-based NLP — no external APIs required.

## Features

- **Spelling detection** — 100+ common misspellings plus homophone disambiguation (their/there/they're, your/you're, its/it's)
- **Grammar rules** — subject-verb agreement, double negatives, a/an usage, capitalization, repeated words, comma splices, who/whom
- **Tense & verb-form checking**, subject-verb relationship analysis
- **Articles, pronouns, prepositions, and word-order checks**
- **Punctuation and capitalization checks**
- **Sentence structure and context analysis**
- **Style suggestions** — passive voice, filler words, weak vocabulary, repeated phrases
- **Tone analysis** across 7 categories (formal, informal, confident, tentative, negative, positive, academic)
- **Flesch-Kincaid readability score** and grade-level calculation
- **One-click Auto-Correct**, sentence Rephrasing, extractive Summarization, and a Synonym Finder
- **Clean responsive UI** — Bootstrap 5, Chart.js issue-breakdown doughnut, expandable issue cards, check history (localStorage)

## Tech Stack

- **Backend:** Python 3.14, Flask 3.x, regex-based NLP, spaCy, NLTK
- **Frontend:** HTML5, CSS3, Bootstrap 5.3.2, Font Awesome, Chart.js, vanilla JS
- **Data:** JSON dictionaries (spelling, homophones, grammar rules, vocabulary)

## Getting Started

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5001 in your browser.

## API Endpoints

| Endpoint            | Description                                  |
|---------------------|----------------------------------------------|
| `POST /api/check`   | Full grammar, spelling, and style analysis   |
| `POST /api/auto_correct` | Automatic text correction               |
| `POST /api/synonyms`| Synonym lookup                               |
| `POST /api/rephrase`| Sentence rephrasing                         |
| `POST /api/summarize`| Extractive text summarization              |
| `POST /api/readability`| Readability and grade-level scores        |

## Project Structure

```
├── app.py                       # Flask routes and API endpoints
├── grammar_engine.py            # Spelling, grammar rules, context, style
├── ai_engine.py                 # Passive voice, fillers, tone, summary, rephrase
├── unified_pipeline.py          # Aggregated v3 pipeline (spaCy + rules)
├── new_pipeline.py              # v4 evidence-merged, AI-validated pipeline
├── subject_verb_checker.py      # Subject–verb agreement
├── tense_checker.py             # Tense and verb-form analysis
├── article_checker.py           # a/an/the usage
├── pronoun_checker.py           # Pronoun checks
├── preprocessor.py              # Text normalization/tokenization
├── pos_tagger.py                # Part-of-speech tagging
├── data/                        # JSON dictionaries and datasets
├── static/                      # CSS + JS assets
├── templates/                   # HTML templates
└── tests/                       # Unit tests
```

## Sample Text to Try

> "Their going to the store tommorow. I definately need to get their early."

## License

College mini project — free to use for educational purposes.