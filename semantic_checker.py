MALE_NOUNS = {'man', 'boy', 'guy', 'male', 'father', 'brother', 'son', 'husband',
              'uncle', 'grandfather', 'grandpa', 'nephew', 'prince', 'king',
              'lord', 'sir', 'gentleman', 'daddy', 'papa', 'dad'}
FEMALE_NOUNS = {'woman', 'girl', 'lady', 'female', 'mother', 'sister', 'daughter',
                'wife', 'aunt', 'grandmother', 'grandma', 'niece', 'princess',
                'queen', 'madam', 'ma\'am', 'gentlewoman', 'mommy', 'mama', 'mom'}
NEUTRAL_NOUNS = {'person', 'baby', 'child', 'children', 'friend', 'teacher',
                 'doctor', 'student', 'neighbor', 'stranger', 'someone', 'anyone',
                 'everyone', 'nobody', 'somebody', 'anybody', 'everybody'}

MALE_PRONOUNS = {'he', 'him', 'his', 'himself'}
FEMALE_PRONOUNS = {'she', 'her', 'hers', 'herself'}


def _find_predicate_noun(words, start_idx, n):
    for j in range(start_idx, min(start_idx + 5, n)):
        token = words[j]
        if not token.is_word:
            continue
        w = token.lower
        if w in {'a', 'an', 'the', 'this', 'that', 'my', 'your', 'his', 'her',
                 'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each',
                 'myself', 'yourself', 'himself', 'herself', 'itself'}:
            continue
        if token.pos in {'NN', 'NNS', 'NNP'}:
            return j, token
        if w in MALE_NOUNS | FEMALE_NOUNS | NEUTRAL_NOUNS:
            return j, token
        if token.pos in {'VBZ', 'VBP', 'VBD', 'VB', 'MD', 'IN', 'CC'}:
            return None, None
        break
    return None, None


def check_semantic(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if w in {'he', 'she'} and i + 1 < n:
            next_token = words[i + 1]
            if next_token.is_word:
                nw = next_token.lower
                if nw in {'is', 'was', 'are', 'were'} and i + 2 < n:
                    pred_idx, pred_token = _find_predicate_noun(words, i + 2, n)
                    if pred_token:
                        pn = pred_token.lower
                        if w == 'he' and pn in FEMALE_NOUNS:
                            errors.append({
                                'type': 'semantic', 'severity': 'warning',
                                'category': 'Semantic Consistency',
                                'incorrect': f'{w} {nw} {pn}',
                                'correction': f'she {nw} {pn}',
                                'position': token.start, 'end_position': pred_token.end,
                                'sentence': sentence.text,
                                'message': f'The pronoun "{w}" typically refers to a male, but "{pn}" refers to a female. Consider using "she".',
                                'confidence': 85, 'pass': 'semantic',
                            })
                        if w == 'she' and pn in MALE_NOUNS:
                            errors.append({
                                'type': 'semantic', 'severity': 'warning',
                                'category': 'Semantic Consistency',
                                'incorrect': f'{w} {nw} {pn}',
                                'correction': f'he {nw} {pn}',
                                'position': token.start, 'end_position': pred_token.end,
                                'sentence': sentence.text,
                                'message': f'The pronoun "{w}" typically refers to a female, but "{pn}" refers to a male. Consider using "he".',
                                'confidence': 85, 'pass': 'semantic',
                            })

    for i, token in enumerate(words):
        w = token.lower
        if w in {'it'} and i + 1 < n:
            next_token = words[i + 1]
            if next_token.is_word:
                nw = next_token.lower
                if nw in {'is', 'was'} and i + 2 < n:
                    pred_idx, pred_token = _find_predicate_noun(words, i + 2, n)
                    if pred_token:
                        pn = pred_token.lower
                        if pn in MALE_NOUNS | FEMALE_NOUNS:
                            gender = 'male' if pn in MALE_NOUNS else 'female'
                            suggestion = 'he' if gender == 'male' else 'she'
                            errors.append({
                                'type': 'semantic', 'severity': 'warning',
                                'category': 'Semantic Consistency',
                                'incorrect': f'{w} {nw} {pn}',
                                'correction': f'{suggestion} {nw} {pn}',
                                'position': token.start, 'end_position': pred_token.end,
                                'sentence': sentence.text,
                                'message': f'"{pn}" refers to a {gender} person. Consider using "{suggestion}" instead of "it".',
                                'confidence': 75, 'pass': 'semantic',
                            })

    return errors
