def deduplicate(errors):
    seen = set()
    unique = []
    for e in errors:
        key = (
            e.get('category', ''),
            e.get('incorrect', '').lower().strip(),
            e.get('correction', '').lower().strip(),
            e.get('position', 0),
        )
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


def rank_by_confidence(errors):
    return sorted(errors, key=lambda e: e.get('confidence', 0), reverse=True)


def filter_by_confidence(errors, min_confidence=0):
    return [e for e in errors if e.get('confidence', 0) >= min_confidence]


def merge_errors(error_lists):
    all_errors = []
    for el in error_lists:
        all_errors.extend(el)
    all_errors = deduplicate(all_errors)
    all_errors = rank_by_confidence(all_errors)
    return all_errors


def group_by_sentence(errors):
    groups = {}
    for e in errors:
        sent = e.get('sentence', '')
        if sent not in groups:
            groups[sent] = []
        groups[sent].append(e)
    return groups


def normalize_for_frontend(errors):
    normalized = []
    for e in errors:
        entry = dict(e)
        entry['word'] = e.get('incorrect', '')
        correction = e.get('correction', '')
        if correction:
            entry['suggestions'] = [correction]
        else:
            entry['suggestions'] = []
        entry['rule'] = e.get('category', e.get('type', ''))
        sentence = e.get('sentence', '')
        incorrect = e.get('incorrect', '')
        word_start = e.get('word_start', -1)
        word_end = e.get('word_end', -1)
        if sentence and incorrect and correction:
            if correction == '(remove)':
                if word_start >= 0 and word_end > word_start:
                    corrected_sentence = sentence[:word_start] + sentence[word_end:]
                else:
                    idx = sentence.lower().find(' ' + incorrect.lower() + ' ')
                    if idx != -1:
                        idx += 1
                        corrected_sentence = sentence[:idx] + sentence[idx + len(incorrect):]
                    else:
                        corrected_sentence = sentence
                corrected_sentence = ' '.join(corrected_sentence.split())
                entry['corrected_text'] = corrected_sentence
            elif correction not in ('(consider rephrasing)',):
                if word_start >= 0 and word_end > word_start:
                    corrected_sentence = sentence[:word_start] + correction + sentence[word_end:]
                else:
                    idx = sentence.lower().find(incorrect.lower())
                    if idx != -1:
                        corrected_sentence = sentence[:idx] + correction + sentence[idx + len(incorrect):]
                    else:
                        corrected_sentence = ''
                entry['corrected_text'] = corrected_sentence
            else:
                entry['corrected_text'] = ''
        normalized.append(entry)
    return normalized
