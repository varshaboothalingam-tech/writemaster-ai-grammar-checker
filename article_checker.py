from pos_tagger import UNCOUNTABLE_NOUNS, PLURAL_ONLY_NOUNS, SINGULAR_ONLY_NOUNS


def check_articles(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if w in {'a', 'an'} and i + 1 < n:
            next_token = words[i + 1]
            if next_token.is_word:
                nw = next_token.lower
                vowel_sounds = set('aeiou')
                starts_with_vowel = nw[0] in vowel_sounds
                h_words = {'hour', 'honest', 'honor', 'honour', 'heir', 'herb'}
                silent_h = nw in h_words
                u_words = {'university', 'uniform', 'unique', 'unit', 'united', 'universal',
                           'universe', 'union', 'usage', 'usual', 'usually', 'utilize',
                           'utility', 'utensil', 'unicorn', 'ukulele', 'uranium'}
                y_sound = nw in u_words or nw.startswith('eu') or nw.startswith('one')
                if w == 'a' and (starts_with_vowel or silent_h) and not y_sound:
                    errors.append({
                        'type': 'grammar', 'severity': 'error',
                        'category': 'Article Usage',
                        'incorrect': 'a', 'correction': 'an',
                        'position': token.start, 'end_position': token.end,
                        'sentence': sentence.text,
                        'message': f'Use "an" before words starting with a vowel sound. "{nw}" starts with a vowel sound.',
                        'confidence': 90, 'pass': 'grammar',
                    })
                elif w == 'an' and not starts_with_vowel and not silent_h:
                    errors.append({
                        'type': 'grammar', 'severity': 'error',
                        'category': 'Article Usage',
                        'incorrect': 'an', 'correction': 'a',
                        'position': token.start, 'end_position': token.end,
                        'sentence': sentence.text,
                        'message': f'Use "a" before words starting with a consonant sound. "{nw}" starts with a consonant sound.',
                        'confidence': 90, 'pass': 'grammar',
                    })

    return errors


def check_countable_uncountable(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if w == 'much' and i + 1 < n:
            next_token = words[i + 1]
            if next_token.is_word:
                nw = next_token.lower
                if nw in UNCOUNTABLE_NOUNS:
                    continue
                if nw not in UNCOUNTABLE_NOUNS:
                    errors.append({
                        'type': 'grammar', 'severity': 'warning',
                        'category': 'Countable/Uncountable',
                        'incorrect': f'much {nw}', 'correction': f'many {nw}',
                        'position': token.start, 'end_position': next_token.end,
                        'sentence': sentence.text,
                        'message': f'"{nw}" appears to be countable. Use "many" instead of "much" with countable nouns.',
                        'confidence': 65, 'pass': 'grammar',
                    })

        if w == 'many' and i + 1 < n:
            next_token = words[i + 1]
            if next_token.is_word:
                nw = next_token.lower
                if nw in UNCOUNTABLE_NOUNS:
                    errors.append({
                        'type': 'grammar', 'severity': 'warning',
                        'category': 'Countable/Uncountable',
                        'incorrect': f'many {nw}', 'correction': f'much {nw}',
                        'position': token.start, 'end_position': next_token.end,
                        'sentence': sentence.text,
                        'message': f'"{nw}" is uncountable. Use "much" instead of "many" with uncountable nouns.',
                        'confidence': 65, 'pass': 'grammar',
                    })

    return errors
