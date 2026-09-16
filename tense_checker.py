from pos_tagger import (IRREGULAR_VERBS, PAST_TO_BASE, AUXILIARIES, MODALS,
                        BE_VERBS, HAVE_VERBS, DO_VERBS, ALL_PAST_FORMS,
                        ALL_PAST_PART_FORMS, VERB_BASE_FORMS, COMMON_WORDS)


def check_verb_tense(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower

        if w in {'didn', "didn't", 'didnt'} or (w == 'did' and i + 1 < n and words[i + 1].lower == "n't"):
            for j in range(i + 1, min(i + 4, n)):
                next_w = words[j].lower
                if next_w in {'not', "n't"}:
                    continue
                if next_w in PAST_TO_BASE:
                    base = PAST_TO_BASE[next_w]
                    errors.append({
                        'type': 'grammar', 'severity': 'error',
                        'category': 'Verb Tense',
                        'incorrect': next_w, 'correction': base,
                        'position': words[j].start, 'end_position': words[j].end,
                        'sentence': sentence.text,
                        'message': f'After "didn\'t", use the base form "{base}", not "{next_w}".',
                        'confidence': 95, 'pass': 'grammar',
                    })
                    break
                if next_w in AUXILIARIES or next_w in MODALS:
                    break
                if next_w.endswith('ed') and len(next_w) > 3:
                    base = next_w
                    if next_w.endswith('ied'):
                        base = next_w[:-3] + 'y'
                    elif next_w.endswith('ed') and len(next_w) > 4:
                        base = next_w[:-2]
                        if not is_real_base_form(base):
                            base = next_w[:-1]
                    errors.append({
                        'type': 'grammar', 'severity': 'error',
                        'category': 'Verb Tense',
                        'incorrect': next_w, 'correction': base,
                        'position': words[j].start, 'end_position': words[j].end,
                        'sentence': sentence.text,
                        'message': f'After "didn\'t", use the base form "{base}", not "{next_w}".',
                        'confidence': 90, 'pass': 'grammar',
                    })
                    break
                break

        if w in MODALS:
            for j in range(i + 1, min(i + 3, n)):
                next_token = words[j]
                if next_token.is_word:
                    nw = next_token.lower
                    if nw in AUXILIARIES:
                        break
                    if nw in {'not', "n't"}:
                        continue
                    if nw.endswith('s') and not nw.endswith('ss') and not nw.endswith('us'):
                        base = nw
                        if nw.endswith('es'):
                            base = nw[:-2]
                        elif nw.endswith('s'):
                            base = nw[:-1]
                        if base and len(base) > 1:
                            errors.append({
                                'type': 'grammar', 'severity': 'error',
                                'category': 'Modal Verb',
                                'incorrect': nw, 'correction': base,
                                'position': next_token.start, 'end_position': next_token.end,
                                'sentence': sentence.text,
                                'message': f'After "{w}", use the base form "{base}", not "{nw}".',
                                'confidence': 90, 'pass': 'grammar',
                            })
                    break

    for i, token in enumerate(words):
        w = token.lower
        if w in {'is', 'are', 'am', 'was', 'were'}:
            for j in range(i + 1, min(i + 3, n)):
                next_token = words[j]
                if next_token.is_word:
                    nw = next_token.lower
                    if nw in {'not', "n't"}:
                        continue
                    if nw in AUXILIARIES:
                        break
                    if nw.endswith('ing'):
                        break
                    if nw in IRREGULAR_VERBS or nw.endswith('ed') or (nw in VERB_BASE_FORMS and nw not in BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS | COMMON_WORDS):
                        if w in {'is', 'are', 'am'}:
                            expected = nw + 'ing'
                            if nw.endswith('e'):
                                expected = nw[:-1] + 'ing'
                            errors.append({
                                'type': 'grammar', 'severity': 'error',
                                'category': 'Verb Form',
                                'incorrect': f'{w} {nw}', 'correction': f'{w} {expected}',
                                'position': token.start, 'end_position': next_token.end,
                                'sentence': sentence.text,
                                'message': f'After "{w}", use the present participle "{expected}", not "{nw}".',
                                'confidence': 85, 'pass': 'grammar',
                            })
                    break

    for i, token in enumerate(words):
        w = token.lower
        if w in {'has', 'have', 'had'}:
            for j in range(i + 1, min(i + 4, n)):
                next_token = words[j]
                if next_token.is_word:
                    nw = next_token.lower
                    if nw in {'not', "n't"}:
                        continue
                    if nw in AUXILIARIES or nw in MODALS:
                        break
                    if nw.endswith('ing'):
                        break
                    if w in {'has', 'have'}:
                        if nw in ALL_PAST_FORMS and nw not in ALL_PAST_PART_FORMS:
                            base = PAST_TO_BASE.get(nw, nw)
                            expected = IRREGULAR_VERBS.get(base, {}).get('past_part', nw)
                            errors.append({
                                'type': 'grammar', 'severity': 'error',
                                'category': 'Verb Tense',
                                'incorrect': nw, 'correction': expected,
                                'position': next_token.start, 'end_position': next_token.end,
                                'sentence': sentence.text,
                                'message': f'After "{w}", use the past participle "{expected}", not the simple past "{nw}".',
                                'confidence': 90, 'pass': 'grammar',
                            })
                    break

    return errors


def is_real_base_form(word):
    if word in VERB_BASE_FORMS:
        return True
    if word in IRREGULAR_VERBS:
        return True
    return False
