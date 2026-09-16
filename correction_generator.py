import re


def generate_correction(sentence_text, error):
    corrected = sentence_text
    incorrect = error.get('incorrect', '')
    correction = error.get('correction', '')
    if not incorrect or not correction:
        return corrected

    if error['category'] == 'Capitalization' and error.get('pass') == 'structure':
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]
        return corrected

    if error['category'] == 'Punctuation':
        corrected = corrected.rstrip()
        if correction in '.!?':
            corrected = corrected + correction
        return corrected

    if error['category'] == 'Spelling':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Subject-Verb Agreement':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Modal Verb':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Verb Tense':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Verb Form':
        parts = incorrect.split(' ', 1)
        if len(parts) == 2:
            corrected = corrected.replace(incorrect, correction, 1)
        else:
            corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Pronoun Case':
        corrected = corrected.replace(incorrect, correction, 1)
        return corrected

    if error['category'] == 'Article Usage':
        if correction == '(remove)':
            corrected = re.sub(r'\b' + re.escape(incorrect) + r'\s+', '', corrected, count=1, flags=re.IGNORECASE)
        else:
            corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Comparative':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Double Negative':
        return corrected

    if error['category'] == 'Repetition':
        corrected = corrected.replace(incorrect, correction, 1)
        return corrected

    if error['category'] == 'Confusing Words':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction.split('/')[0], corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Semantic Consistency':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Countable/Uncountable':
        corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
        return corrected

    if error['category'] == 'Word Order':
        corrected = corrected.replace(incorrect, correction, 1)
        return corrected

    corrected = re.sub(r'\b' + re.escape(incorrect) + r'\b', correction, corrected, count=1, flags=re.IGNORECASE)
    return corrected


def apply_corrections(text, errors, grammar_checker_func, max_iterations=3):
    from correction_validator import validate_and_retry
    return validate_and_retry(text, errors, generate_correction, grammar_checker_func, max_iterations)
