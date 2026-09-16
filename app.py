import os
import json
from flask import Flask, render_template, request, jsonify
from grammar_engine import GrammarChecker
from ai_engine import AIEngine
from writing_engine import WritingAnalyzer
from scoring_engine import compute_scores

# New pipeline (removed — unified pipeline replaces it)
new_pipeline_available = False

# Unified pipeline (v3)
try:
    from unified_pipeline import check_text as unified_check, ReadabilityEngine, ToneEngine, StyleEngine, get_nlp
    unified_pipeline_available = True
except Exception:
    unified_pipeline_available = False

# v4 pipeline (evidence-merged + AI-validated)
try:
    from new_pipeline import check_v4
    v4_pipeline_available = True
except Exception:
    v4_pipeline_available = False

app = Flask(__name__)
grammar_checker = GrammarChecker()
ai_engine = AIEngine()

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
writing_analyzer = WritingAnalyzer(DATA_DIR)

VOCAB_DATA = None

# Central aggregation layer
try:
    from suggestion_aggregator import SuggestionAggregator
    aggregator = SuggestionAggregator(
        writing_analyzer, grammar_checker, ai_engine, DATA_DIR
    )
except ImportError:
    aggregator = None


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/rules')
def rules():
    grammar_data = load_json('grammar_errors.json')
    return render_template('rules.html', rules=grammar_data)


@app.route('/api/check', methods=['POST'])
def api_check():
    """Spec §10: /api/check runs the COMPLETE pipeline and returns ONLY
    approved errors (detection -> filtering -> dedup -> correction validation
    -> Gemini final validation -> final results)."""
    data = request.get_json() or {}
    text = data.get('text', '')
    if not v4_pipeline_available:
        return jsonify({'error': 'v4 pipeline not available', 'issues': []}), 503
    try:
        return _v4_check_response(text, use_ai=data.get('use_ai'))
    except Exception as e:
        return jsonify({'error': str(e), 'pipeline': 'v4', 'issues': [], 'meta': {}}), 500


@app.route('/api/health', methods=['GET'])
def api_health():
    """Spec §10: health + capability probe."""
    ai = None
    ai_state = {'enabled': False, 'provider': 'none', 'model': '', 'available': False}
    try:
        from ai_validator import get_validator, active_providers, provider_status, _min_approvals
        ai = get_validator()
        ai_state = {
            'enabled': ai.is_enabled(),
            'provider': ai.provider or 'none',
            'model': ai.model or '',
            'available': bool(ai.available()) if ai.is_enabled() else False,
            'providers': active_providers(),
            'providers_detail': provider_status(),
            'min_approvals': _min_approvals(),
            'ensemble': len(ai.providers) > 1,
        }
    except Exception:
        ai_state = {'enabled': False, 'provider': 'none', 'model': '', 'available': False,
                    'error': 'ai_validator import failed'}
    return jsonify({
        'status': 'ok',
        'pipeline': 'v4',
        'v4_pipeline_available': v4_pipeline_available,
        'unified_pipeline_available': unified_pipeline_available,
        'ai': ai_state,
    })


@app.route('/api/validate', methods=['POST'])
def api_validate():
    """Spec §10: validate a text (or a specific candidate) through the full
    pipeline. Response contains ONLY approved errors; rejected candidates are
    written to the AI decision log (GC_AI_LOG=1) for development.
    """
    data = request.get_json() or {}
    text = data.get('text', '')
    if not v4_pipeline_available:
        return jsonify({'error': 'v4 pipeline not available', 'issues': []}), 503
    try:
        result = check_v4(text, use_ai=data.get('use_ai'), log_ai=True)
        approved = result['errors']
        matched = None
        if data.get('original') or data.get('replacement'):
            orig = data.get('original', '')
            repl = data.get('replacement', '')
            matched = next((
                e for e in approved
                if (not orig or e.get('original') == orig)
                and (not repl or e.get('replacement') == repl)
            ), None)
        return jsonify({
            'validated': True,
            'approved_errors': approved,
            'matched_candidate': matched,
            'meta': result['meta'],
            'pipeline': 'v4',
        })
    except Exception as e:
        return jsonify({'error': str(e), 'validated': False, 'pipeline': 'v4'}), 500


def _v4_check_response(text, use_ai=None):
    """Run v4 and shape the approved-only result for the frontend."""
    result = check_v4(text, use_ai=use_ai)
    readability = ReadabilityEngine().calculate(text) if text.strip() else {}
    word_count = readability.get('words', 0)
    sentence_count = readability.get('sentences', 0)

    score_errors = [{
        "category": iss.get("category", "grammar").lower(),
        "error_type": iss.get("rule_id", "grammar"),
        "severity": iss.get("severity", "warning").upper(),
        "confidence": iss.get("confidence", 0.8),
    } for iss in result['errors']]

    issues = _v4_frontend_issues(result['errors'])

    return jsonify({
        'issues': issues,
        'readability': readability,
        'scores': compute_scores(score_errors, readability, word_count, sentence_count),
        'meta': result['meta'],
        'pipeline': 'v4',
    })


def _frontend_type(category: str, rule_id: str) -> str:
    """Map a v4 category/rule to the frontend issue 'type' bucket."""
    cat = (category or '').upper()
    rule = (rule_id or '').upper()
    if ('SPELL' in cat or 'SPELL' in rule or 'MISSPELL' in cat
            or 'TYPO' in cat or 'CAPITAL' in cat):
        return 'spelling'
    if ('PUNCT' in cat or 'COMMA' in rule or 'CAPITALIZATION' in rule
            or 'APOSTROPHE' in rule or 'QUOTE' in rule):
        return 'punctuation'
    if ('WORD_CHOICE' in cat or 'WORD_USAGE' in cat or 'CONTEXT' in cat
            or 'COLLOCATION' in rule or 'REDUNDANT' in rule or 'CONFUSED' in rule
            or 'CHOICE' in rule or 'STYLE' in cat):
        return 'style'
    return 'grammar'


def _v4_frontend_issues(errors):
    issues = []
    for iss in errors:
        item = dict(iss)
        item["word"] = item.get("original", "")
        item["suggestions"] = [item["replacement"]] if item.get("replacement") else []
        item["position"] = item.get("start", 0)
        item["start_position"] = item.get("start", 0)
        item["end_position"] = item.get("end", 0)
        item["original_text"] = item.get("original", "")
        item["type"] = _frontend_type(item.get("category", ""), item.get("rule_id", ""))
        issues.append(item)
    return issues


@app.route('/api/check-v2', methods=['POST'])
def api_check_v2():
    """New pipeline: candidate detection → context analysis → FP filter → validation."""
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'issues': [], 'readability': writing_analyzer.readability.calculate(''), 'scores': compute_scores([], {}, 0, 0)})

    if new_pipeline_available:
        issues = check_text_new(text)
    else:
        # Fallback to old pipeline
        if aggregator:
            result = aggregator.aggregate(text)
            issues = [_to_frontend_issue(e) for e in result['issues']]
        else:
            result = writing_analyzer.analyze(text)
            issues = [_to_frontend_issue(e) for e in result['errors']]

    readability = writing_analyzer.readability.calculate(text)
    word_count = readability.get('words', 0)
    sentence_count = readability.get('sentences', 0)

    # Compute scores from the issues
    score_errors = []
    for iss in issues:
        score_errors.append({
            "category": iss.get("category", "grammar"),
            "error_type": iss.get("error_type", "grammar"),
            "severity": iss.get("severity", "MEDIUM"),
            "confidence": iss.get("confidence", 0.8),
        })
    scores = compute_scores(score_errors, readability, word_count, sentence_count)

    return jsonify({
        'issues': issues,
        'readability': readability,
        'scores': scores,
        'pipeline': 'v2' if new_pipeline_available else 'v1',
    })


@app.route('/api/check-v3', methods=['POST'])
def api_check_v3():
    """Unified pipeline v3: preprocessor → NLP → context → validation → ranking."""
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'issues': [], 'readability': {}, 'scores': compute_scores([], {}, 0, 0), 'pipeline': 'v3'})

    if unified_pipeline_available:
        try:
            result = unified_check(text)
            readability = ReadabilityEngine().calculate(text)
            tone = ToneEngine().analyze(text)
            style = StyleEngine().analyze(text)

            word_count = readability.get('words', 0)
            sentence_count = readability.get('sentences', 0)

            score_errors = []
            for iss in result:
                score_errors.append({
                    "category": iss.get("category", "grammar"),
                    "error_type": iss.get("error_type", "grammar"),
                    "severity": iss.get("severity", "MEDIUM"),
                    "confidence": iss.get("confidence", 0.8),
                })
            scores = compute_scores(score_errors, readability, word_count, sentence_count)

            for iss in result:
                iss["word"] = iss.get("original", "")
                iss["suggestions"] = [iss["replacement"]] if iss.get("replacement") else []
                iss["position"] = iss.get("start", 0)
                iss["start_position"] = iss.get("start", 0)
                iss["end_position"] = iss.get("end", 0)
                iss["original_text"] = iss.get("original", "")

            return jsonify({
                'issues': result,
                'readability': readability,
                'scores': scores,
                'tone': tone,
                'style': style,
                'pipeline': 'v3',
            })
        except Exception as e:
            return jsonify({'error': str(e), 'pipeline': 'v3', 'issues': []}), 500
    else:
        return jsonify({'error': 'Unified pipeline not available', 'pipeline': 'v3', 'issues': []}), 503


@app.route('/api/check-v4', methods=['POST'])
def api_check_v4():
    """v4: evidence-merged detection + optional AI final validation.
    Alias of /api/check — returns ONLY approved errors."""
    data = request.get_json() or {}
    text = data.get('text', '')

    if not v4_pipeline_available:
        return jsonify({'error': 'v4 pipeline not available', 'issues': []}), 503

    try:
        return _v4_check_response(text, use_ai=data.get('use_ai'))
    except Exception as e:
        return jsonify({'error': str(e), 'pipeline': 'v4', 'issues': [], 'meta': {}}), 500


def _to_frontend_issue(err):
    suggestions = []
    if err.get('replacement'):
        suggestions.append(err['replacement'])
    elif err.get('suggestions'):
        suggestions = err['suggestions']

    severity_map = {
        'HIGH': 'error',
        'MEDIUM': 'warning',
        'LOW': 'info',
    }
    return {
        'word': err.get('original_text', ''),
        'type': err.get('error_type', 'grammar'),
        'severity': severity_map.get(err.get('severity', 'MEDIUM'), 'warning'),
        'rule': err.get('category', 'GRAMMAR'),
        'message': err.get('message', ''),
        'suggestions': suggestions,
        'corrected_text': err.get('replacement', ''),
        'position': err.get('start_position', 0),
        'start_position': err.get('start_position', 0),
        'end_position': err.get('end_position', 0),
        'sentence': err.get('context', ''),
        'confidence': err.get('confidence', 0.8),
        'explanation': err.get('explanation', ''),
        'can_add_to_dict': err.get('category', '') in ('SPELLING', 'CONTEXTUAL_WORD_USAGE'),
    }


@app.route('/api/add-to-dictionary', methods=['POST'])
def api_add_to_dictionary():
    data = request.get_json()
    word = data.get('word', '').strip().lower()
    if not word:
        return jsonify({'success': False, 'message': 'No word provided.'})
    writing_analyzer.spelling.add_custom_word(word)
    return jsonify({'success': True, 'message': f'"{word}" added to dictionary.'})


@app.route('/api/ignore-all', methods=['POST'])
def api_ignore_all():
    data = request.get_json()
    word = data.get('word', '').strip().lower()
    if not word:
        return jsonify({'success': False, 'message': 'No word provided.'})
    writing_analyzer.spelling.add_custom_word(word)
    return jsonify({'success': True, 'message': f'All instances of "{word}" will be ignored.'})


def _map_legacy_type(etype, rule):
    if etype == 'spelling':
        return 'SPELLING'
    if 'capital' in rule.lower():
        return 'CAPITALIZATION'
    if 'punctuation' in rule.lower():
        return 'PUNCTUATION'
    if 'context' in etype.lower():
        return 'WORD_USAGE'
    return 'GRAMMAR'


@app.route('/api/correct', methods=['POST'])
def api_correct():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'corrected': '', 'changes': []})
    corrected = grammar_checker.auto_correct(text)
    changes = []
    original_words = text.split()
    corrected_words = corrected.split()
    max_len = max(len(original_words), len(corrected_words))
    for i in range(max_len):
        orig = original_words[i] if i < len(original_words) else ''
        corr = corrected_words[i] if i < len(corrected_words) else ''
        if orig != corr:
            changes.append({'original': orig, 'corrected': corr})
    return jsonify({'corrected': corrected, 'changes': changes})


@app.route('/api/rephrase', methods=['POST'])
def api_rephrase():
    data = request.get_json()
    text = data.get('text', '')
    tone = data.get('tone', 'professional')
    if not text.strip():
        return jsonify({'rephrased': '', 'tone': tone})
    rephrased = ai_engine.rephrase(text, tone)
    return jsonify({'rephrased': rephrased, 'tone': tone})


@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'enhanced_text': '', 'suggestions': [], 'stats': {}})
    result = ai_engine.enhance_writing(text)
    return jsonify(result)


@app.route('/api/tone', methods=['POST'])
def api_tone():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'dominant_tone': 'neutral', 'confidence': 0, 'tones': {}, 'analysis': {}})
    result = ai_engine.detect_tone(text)
    return jsonify(result)


@app.route('/api/readability', methods=['POST'])
def api_readability():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'score': 0, 'level': 'N/A', 'grade': 0, 'sentences': 0, 'words': 0, 'syllables': 0, 'avg_words_sentence': 0, 'reading_time': 0, 'speaking_time': 0, 'complex_word_pct': 0, 'explanation': 'No text to analyze.'})
    result = writing_analyzer.readability.calculate(text)
    return jsonify(result)


@app.route('/api/fix-all', methods=['POST'])
def api_fix_all():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'corrected': '', 'issues_fixed': 0, 'changes': []})

    if unified_pipeline_available:
        try:
            errors = unified_check(text)
        except Exception:
            errors = []
    else:
        result = writing_analyzer.analyze(text)
        errors = result['errors']

    high_conf = [e for e in errors if e.get('confidence', 0) >= 0.70
                 and e.get('replacement') and e['replacement'].strip()]
    high_conf.sort(key=lambda e: -(e.get('start', e.get('start_position', 0))))

    corrected = text
    changes = []
    applied_ranges = []
    for e in high_conf:
        start = e.get('start', e.get('start_position', 0))
        end = e.get('end', e.get('end_position', 0))
        replacement = e['replacement']
        original = e.get('original', e.get('original_text', ''))
        overlap = False
        for a_start, a_end in applied_ranges:
            if start < a_end and end > a_start:
                overlap = True
                break
            if abs(start - a_end) <= 1 or abs(end - a_start) <= 1:
                if replacement.lower() in corrected[a_start:a_end].lower():
                    overlap = True
                    break
        if overlap:
            continue
        if 0 <= start < end <= len(corrected):
            segment = corrected[start:end]
            if segment.lower() == original.lower():
                corrected = corrected[:start] + replacement + corrected[end:]
                changes.append({'original': original, 'corrected': replacement})
                applied_ranges.append((start, start + len(replacement)))

    issues_before = len(errors)
    if unified_pipeline_available:
        try:
            issues_after = len(unified_check(corrected))
        except Exception:
            issues_after = 0
    else:
        issues_after = len(writing_analyzer.analyze(corrected)['errors'])

    return jsonify({
        'corrected': corrected,
        'issues_fixed': max(0, issues_before - issues_after),
        'changes': changes,
    })


@app.route('/api/synonyms', methods=['POST'])
def api_synonyms():
    global VOCAB_DATA
    data = request.get_json()
    word = data.get('word', '')
    if not word.strip():
        return jsonify({'word': '', 'synonyms': [], 'filler_words': [], 'weak_words': {}})
    if VOCAB_DATA is None:
        VOCAB_DATA = load_json('vocabulary.json')
    synonyms_map = VOCAB_DATA.get('synonyms', {})
    lower = word.lower().strip()
    synonyms = synonyms_map.get(lower, [])
    filler_words = VOCAB_DATA.get('filler_words', [])
    weak_words = VOCAB_DATA.get('weak_words', {})
    return jsonify({'word': word, 'synonyms': synonyms, 'filler_words': filler_words, 'weak_words': weak_words})


@app.route('/api/ai-rewrite', methods=['POST'])
def api_ai_rewrite():
    """Optional AI-powered rewrite using Anthropic Claude.
    Requires ANTHROPIC_API_KEY env var. Clearly labeled as AI-generated."""
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return jsonify({
            'success': False,
            'message': 'AI rewrite requires ANTHROPIC_API_KEY environment variable.',
            'rewritten': '',
        })

    data = request.get_json()
    text = data.get('text', '')
    tone = data.get('tone', 'professional')
    if not text.strip():
        return jsonify({'success': False, 'message': 'No text provided.', 'rewritten': ''})

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f'Rewrite the following text in a {tone} tone. '
            f'Fix grammar, spelling, and punctuation errors. '
            f'Improve clarity and readability. '
            f'Return ONLY the rewritten text, no explanation.\n\n'
            f'Original:\n{text}'
        )
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=2048,
            messages=[{'role': 'user', 'content': prompt}],
        )
        rewritten = response.content[0].text
        return jsonify({
            'success': True,
            'rewritten': rewritten,
            'tone': tone,
            'model': 'claude-sonnet-4-20250514',
            'disclaimer': 'This text was rewritten by AI (Claude). Please review before using.',
        })
    except ImportError:
        return jsonify({
            'success': False,
            'message': 'anthropic package not installed. Run: pip install anthropic',
            'rewritten': '',
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'AI rewrite failed: {str(e)}',
            'rewritten': '',
        })


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')
    app.run(debug=debug_mode, host='127.0.0.1', port=5001)
