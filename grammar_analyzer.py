from intelligent_engine import IntelligentGrammarEngine

_intelligent_engine = IntelligentGrammarEngine()

_intelligent_error_positions = set()


def run_all_checks(sentence):
    errors = []
    errors.extend(_run_legacy_checks(sentence))
    return errors


def run_intelligent_checks(text):
    global _intelligent_error_positions
    errors = _intelligent_engine.analyze(text)
    _intelligent_error_positions = set()
    for e in errors:
        key = (e.get('category', ''), e.get('incorrect', ''), e.get('position', 0))
        _intelligent_error_positions.add(key)
    return errors


def get_intelligent_positions():
    return _intelligent_error_positions


def _run_legacy_checks(sentence):
    errors = []
    try:
        from subject_verb_checker import check_subject_verb_agreement
        errors.extend(check_subject_verb_agreement(sentence))
    except Exception:
        pass
    try:
        from tense_checker import check_verb_tense
        errors.extend(check_verb_tense(sentence))
    except Exception:
        pass
    try:
        from pronoun_checker import check_pronouns
        errors.extend(check_pronouns(sentence))
    except Exception:
        pass
    try:
        from article_checker import check_articles, check_countable_uncountable
        errors.extend(check_articles(sentence))
        errors.extend(check_countable_uncountable(sentence))
    except Exception:
        pass
    try:
        from sentence_structure_checker import (check_double_negatives, check_comparatives,
                                                check_sentence_structure, check_prepositions,
                                                check_conjunctions, check_word_order,
                                                check_fragment, check_dangling_modifier,
                                                check_parallelism)
        errors.extend(check_double_negatives(sentence))
        errors.extend(check_comparatives(sentence))
        errors.extend(check_sentence_structure(sentence))
        errors.extend(check_word_order(sentence))
        errors.extend(check_fragment(sentence))
        errors.extend(check_dangling_modifier(sentence))
        errors.extend(check_parallelism(sentence))
    except Exception:
        pass
    try:
        from context_checker import check_context
        errors.extend(check_context(sentence))
    except Exception:
        pass
    try:
        from semantic_checker import check_semantic
        errors.extend(check_semantic(sentence))
    except Exception:
        pass
    return errors
