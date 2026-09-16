from pos_tagger import (IRREGULAR_VERBS, PAST_TO_BASE, AUXILIARIES, MODALS,
                        BE_VERBS, HAVE_VERBS, DO_VERBS, PRONOUNS_SUBJECT,
                        COMMON_WORDS, UNCOUNTABLE_NOUNS, VERB_BASE_FORMS,
                        ALL_PAST_FORMS, ALL_PAST_PART_FORMS, COMMON_ADJECTIVES,
                        PLURAL_ONLY_NOUNS)

SINGULAR_PRONOUNS = {'he', 'she', 'it'}
PLURAL_PRONOUNS = {'we', 'they'}
FIRST_PERSON = {'i', 'we'}
SECOND_PERSON = {'you'}


def _noun_is_plural(word):
    w = word.lower
    if w in PLURAL_ONLY_NOUNS:
        return True
    if w.endswith('s') and not w.endswith('ss') and not w.endswith('us') and len(w) > 3:
        return True
    return False


def _find_subject_verb_pairs(sentence):
    pairs = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if token.pos == 'PRP' and w in PRONOUNS_SUBJECT:
            for j in range(i + 1, min(i + 12, n)):
                next_token = words[j]
                if next_token.pos in {'VBZ', 'VBP', 'VBD', 'VB', 'VBN', 'VBG', 'MD'}:
                    if not any(words[k].lower in {'who', 'which', 'that', 'whom', 'whose'}
                              for k in range(i + 1, j) if words[k].is_word):
                        pairs.append((i, j, token, next_token))
                        break
                if next_token.pos in {'NN', 'NNS', 'NNP'} and j > i + 1:
                    break
                if next_token.is_punct and next_token.text in {',', '.', '!', '?', ';', ':'}:
                    break

        elif token.pos in {'NN', 'NNS', 'NNP'}:
            for j in range(i + 1, min(i + 12, n)):
                next_token = words[j]
                if next_token.pos in {'VBZ', 'VBP', 'VBD', 'VB', 'VBN', 'VBG', 'MD'}:
                    if not any(words[k].lower in {'who', 'which', 'that', 'whom', 'whose'}
                              for k in range(i + 1, j) if words[k].is_word):
                        pairs.append((i, j, token, next_token))
                        break
                if next_token.pos in {'NN', 'NNS', 'NNP'} and j > i + 1:
                    break
                if next_token.is_punct and next_token.text in {',', '.', '!', '?', ';', ':'}:
                    break

    return pairs


def _get_number(person):
    if person in {'he', 'she', 'it'}:
        return 'singular'
    if person in {'i', 'you'}:
        return 'singular'
    if person in {'we', 'they'}:
        return 'plural'
    return 'singular'


def _make_error(subj_token, verb_token, correction, sentence, message, confidence):
    return {
        'type': 'grammar',
        'severity': 'error',
        'category': 'Subject-Verb Agreement',
        'incorrect': verb_token.text,
        'correction': correction,
        'position': verb_token.start,
        'end_position': verb_token.end,
        'sentence': sentence.text,
        'message': message,
        'confidence': confidence,
        'pass': 'grammar',
    }


def check_subject_verb_agreement(sentence):
    errors = []
    words = sentence.words
    pairs = _find_subject_verb_pairs(sentence)

    for subj_idx, verb_idx, subj, verb in pairs:
        sw = subj.lower
        vw = verb.lower

        if vw in {'am', 'is', 'are', 'was', 'were'}:
            if sw in {'i'}:
                if vw == 'are':
                    errors.append(_make_error(subj, verb, 'is', sentence,
                        'Use "is" with "I".', 95))
                elif vw == 'were':
                    if not any(words[k].lower == 'if' for k in range(max(0, verb_idx - 5), verb_idx) if words[k].is_word):
                        errors.append(_make_error(subj, verb, 'was', sentence,
                            'Use "was" with "I" in past tense.', 90))
            elif sw in SINGULAR_PRONOUNS | {'he', 'she', 'it'}:
                if vw == 'are':
                    errors.append(_make_error(subj, verb, 'is', sentence,
                        f'The subject "{sw}" is singular and requires "is", not "are".', 95))
                elif vw == 'were':
                    if not any(words[k].lower == 'if' for k in range(max(0, verb_idx - 5), verb_idx) if words[k].is_word):
                        errors.append(_make_error(subj, verb, 'was', sentence,
                            f'The subject "{sw}" is singular and requires "was", not "were".', 90))
            elif sw in PLURAL_PRONOUNS:
                if vw == 'is':
                    errors.append(_make_error(subj, verb, 'are', sentence,
                        f'The subject "{sw}" is plural and requires "are", not "is".', 95))
                elif vw == 'was':
                    errors.append(_make_error(subj, verb, 'were', sentence,
                        f'The subject "{sw}" is plural and requires "were", not "was".', 90))
            elif sw in {'you'}:
                if vw == 'is':
                    errors.append(_make_error(subj, verb, 'are', sentence,
                        'Use "are" with "you".', 95))
                elif vw == 'was':
                    errors.append(_make_error(subj, verb, 'were', sentence,
                        'Use "were" with "you".', 90))
            else:
                if subj.pos in {'NN', 'NNP', 'NNS'}:
                    is_plural = _noun_is_plural(subj) or subj.pos == 'NNS'
                    if is_plural and vw == 'is':
                        errors.append(_make_error(subj, verb, 'are', sentence,
                            f'The subject "{sw}" appears to be plural and requires "are", not "is".', 80))
                    elif is_plural and vw == 'was':
                        errors.append(_make_error(subj, verb, 'were', sentence,
                            f'The subject "{sw}" appears to be plural and requires "were", not "was".', 75))
                    elif not is_plural and vw == 'are':
                        errors.append(_make_error(subj, verb, 'is', sentence,
                            f'The subject "{sw}" appears to be singular and requires "is", not "are".', 80))
                    elif not is_plural and vw == 'were':
                        errors.append(_make_error(subj, verb, 'was', sentence,
                            f'The subject "{sw}" appears to be singular and requires "was", not "were".', 75))

        elif vw in {'has', 'have', 'had'}:
            if sw in {'i', 'you', 'we', 'they'}:
                if vw == 'has':
                    errors.append(_make_error(subj, verb, 'have', sentence,
                        f'The subject "{sw}" requires "have", not "has".', 95))
            elif sw in SINGULAR_PRONOUNS:
                if vw == 'have':
                    errors.append(_make_error(subj, verb, 'has', sentence,
                        f'The subject "{sw}" requires "has", not "have".', 95))

        elif vw in {'does', 'do', 'did'}:
            if sw in {'i', 'you', 'we', 'they'}:
                if vw == 'does':
                    errors.append(_make_error(subj, verb, 'do', sentence,
                        f'The subject "{sw}" requires "do", not "does".', 95))
            elif sw in SINGULAR_PRONOUNS:
                if vw == 'do':
                    errors.append(_make_error(subj, verb, 'does', sentence,
                        f'The subject "{sw}" requires "does", not "do".', 95))

        elif vw not in MODALS and vw not in BE_VERBS | HAVE_VERBS | DO_VERBS:
            if sw in SINGULAR_PRONOUNS and vw not in {'was', 'has', 'does', 'is'}:
                base = vw
                if vw.endswith('s') and not vw.endswith('ss'):
                    if sw in {'he', 'she', 'it'}:
                        pass
                    else:
                        errors.append(_make_error(subj, verb, base, sentence,
                            f'The subject "{sw}" is singular and requires the base form "{base}".', 85))
                else:
                    expected = base + 's'
                    if base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
                        expected = base + 'es'
                    elif base.endswith('y') and base[-2] not in 'aeiou':
                        expected = base[:-1] + 'ies'
                    if expected != vw and base not in BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS:
                        errors.append(_make_error(subj, verb, expected, sentence,
                            f'The subject "{sw}" is third-person singular and requires "{expected}", not "{vw}".', 90))

            elif sw in PLURAL_PRONOUNS:
                if vw.endswith('s') and not vw.endswith('ss') and not vw.endswith('us'):
                    base = vw
                    if vw.endswith('es'):
                        base = vw[:-2]
                    elif vw.endswith('s'):
                        base = vw[:-1]
                    if base and (base in COMMON_WORDS or base in VERB_BASE_FORMS):
                        errors.append(_make_error(subj, verb, base, sentence,
                            f'The subject "{sw}" is plural and requires the base form "{base}", not "{vw}".', 85))

    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if w in {'there', "there's"} and i + 1 < n:
            next_token = words[i + 1]
            if next_token.is_word and next_token.lower in {'is', 'was'}:
                quantifiers = {'many', 'several', 'few', 'a', 'lot', 'some', 'these', 'those',
                               'numerous', 'multiple', 'various', 'both', 'all', 'couple'}
                for j in range(i + 2, min(i + 6, n)):
                    if words[j].is_word:
                        nw = words[j].lower
                        if nw in quantifiers or words[j].pos in {'NNS'}:
                            be_verb = next_token.lower
                            expected = 'are' if be_verb == 'is' else 'were'
                            errors.append(_make_error(token, next_token, expected, sentence,
                                f'After "there" with a plural quantity, use "{expected}", not "{be_verb}".', 75))
                        break

    return errors
