from pos_tagger import (PRONOUNS_SUBJECT, PRONOUNS_OBJECT, PRONOUNS_POSSESSIVE,
                        PRONOUNS_REFLEXIVE)

MALE_PRONOUNS = {'he', 'him', 'his', 'himself'}
FEMALE_PRONOUNS = {'she', 'her', 'hers', 'herself'}
NEUTRAL_PRONOUNS = {'it', 'its', 'itself'}
PLURAL_PRONOUNS_SET = {'they', 'them', 'their', 'theirs', 'themselves'}
FIRST_PERSON = {'i', 'me', 'my', 'mine', 'myself'}
SECOND_PERSON = {'you', 'your', 'yours', 'yourself', 'yourselves'}
FIRST_PERSON_PLURAL = {'we', 'us', 'our', 'ours', 'ourselves'}

SUBJECT_TO_OBJECT = {
    'i': 'me', 'he': 'him', 'she': 'her', 'we': 'us', 'they': 'them',
}
OBJECT_TO_SUBJECT = {v: k for k, v in SUBJECT_TO_OBJECT.items()}
SUBJECT_TO_POSSESSIVE = {
    'i': 'my', 'he': 'his', 'she': 'her', 'it': 'its',
    'we': 'our', 'they': 'their', 'you': 'your',
}


def check_pronouns(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if w == 'i':
            if token.text != 'I':
                errors.append({
                    'type': 'grammar', 'severity': 'error',
                    'category': 'Capitalization',
                    'incorrect': token.text, 'correction': 'I',
                    'position': token.start, 'end_position': token.end,
                    'sentence': sentence.text,
                    'message': 'The pronoun "I" should always be capitalized.',
                    'confidence': 99, 'pass': 'grammar',
                })

    for i, token in enumerate(words):
        w = token.lower
        if w in {'me'}:
            if i == 0 or (i > 0 and words[i - 1].lower in {'and', 'or'}):
                if i + 1 < n and words[i + 1].lower in {'and', 'or'}:
                    errors.append({
                        'type': 'grammar', 'severity': 'error',
                        'category': 'Pronoun Case',
                        'incorrect': token.text, 'correction': 'I',
                        'position': token.start, 'end_position': token.end,
                        'sentence': sentence.text,
                        'message': 'Use "I" instead of "me" when it is the subject of a sentence.',
                        'confidence': 90, 'pass': 'grammar',
                    })
                elif i == 0:
                    for j in range(i + 1, min(i + 4, n)):
                        if words[j].lower in {'and', 'or'}:
                            continue
                        if words[j].lower == 'and' and j + 1 < n:
                            next_after_and = words[j + 1].lower
                            if next_after_and in PRONOUNS_SUBJECT | {'my', 'your', 'his', 'her', 'our', 'their'}:
                                errors.append({
                                    'type': 'grammar', 'severity': 'error',
                                    'category': 'Pronoun Case',
                                    'incorrect': 'Me and',
                                    'correction': next_after_and.capitalize() + ' and I',
                                    'position': token.start, 'end_position': words[j].end,
                                    'sentence': sentence.text,
                                    'message': 'Use "I" instead of "me" as a subject. Conventionally, put yourself last.',
                                    'confidence': 85, 'pass': 'grammar',
                                })
                            break
                        break

    for i, token in enumerate(words):
        w = token.lower
        if w in {'him', 'her', 'us', 'them'}:
            if i == 0:
                for j in range(i + 1, min(i + 5, n)):
                    if words[j].lower in {'and', 'or'}:
                        if j + 1 < n and words[j + 1].is_word:
                            next_after = words[j + 1].lower
                            subj_form = OBJECT_TO_SUBJECT.get(w, w.capitalize())
                            if next_after in PRONOUNS_SUBJECT | {'my', 'your', 'his', 'our', 'their'}:
                                errors.append({
                                    'type': 'grammar', 'severity': 'error',
                                    'category': 'Pronoun Case',
                                    'incorrect': token.text, 'correction': subj_form,
                                    'position': token.start, 'end_position': token.end,
                                    'sentence': sentence.text,
                                    'message': f'Use the subject form "{subj_form}" instead of the object form "{w}" at the start of a sentence.',
                                    'confidence': 85, 'pass': 'grammar',
                                })
                            elif next_after in PRONOUNS_OBJECT:
                                errors.append({
                                    'type': 'grammar', 'severity': 'error',
                                    'category': 'Pronoun Case',
                                    'incorrect': f'{token.text} and {words[j+1].text}',
                                    'correction': f'{subj_form} and {OBJECT_TO_SUBJECT.get(next_after, next_after.capitalize())}',
                                    'position': token.start, 'end_position': words[j + 1].end,
                                    'sentence': sentence.text,
                                    'message': f'Use subject forms "{subj_form}" and "{OBJECT_TO_SUBJECT.get(next_after, next_after.capitalize())}" at the start of a sentence.',
                                    'confidence': 85, 'pass': 'grammar',
                                })
                        break

    for i, token in enumerate(words):
        w = token.lower
        if w in {"don't", 'dont'} and i > 0:
            prev = words[i - 1].lower if i > 0 else ''
            if prev in {'he', 'she', 'it'}:
                errors.append({
                    'type': 'grammar', 'severity': 'error',
                    'category': 'Subject-Verb Agreement',
                    'incorrect': "don't", 'correction': "doesn't",
                    'position': token.start, 'end_position': token.end,
                    'sentence': sentence.text,
                    'message': f'The subject "{prev}" is third-person singular and requires "doesn\'t", not "don\'t".',
                    'confidence': 95, 'pass': 'grammar',
                })
        if w in {"doesn't", 'doesnt'} and i > 0:
            prev = words[i - 1].lower if i > 0 else ''
            if prev in {'we', 'they'} | {'you'}:
                errors.append({
                    'type': 'grammar', 'severity': 'error',
                    'category': 'Subject-Verb Agreement',
                    'incorrect': "doesn't", 'correction': "don't",
                    'position': token.start, 'end_position': token.end,
                    'sentence': sentence.text,
                    'message': f'The subject "{prev}" is plural and requires "don\'t", not "doesn\'t".',
                    'confidence': 95, 'pass': 'grammar',
                })

    return errors
