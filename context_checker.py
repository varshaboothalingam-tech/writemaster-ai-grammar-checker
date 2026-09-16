def check_context(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    context_rules = [
        {
            'word': 'your',
            'check': lambda w, i, words, n: (
                i + 1 < n and words[i + 1].is_word and
                (words[i + 1].pos in {'VBZ', 'VBP', 'VBD', 'VB', 'VBG', 'VBN', 'MD'} or
                 words[i + 1].lower in {'welcome', 'right', 'wrong', 'correct',
                                          'late', 'early', 'ready', 'sure', 'free'})
            ),
            'correction': "you're",
            'message': '"your" is possessive. Use "you\'re" for "you are".',
            'confidence': 85,
        },
        {
            'word': "you're",
            'check': lambda w, i, words, n: (
                i + 1 < n and words[i + 1].is_word and
                words[i + 1].pos in {'NN', 'NNS', 'NNP', 'DT', 'JJ', 'PRP$'} and
                words[i + 1].lower not in {'going', 'coming', 'doing', 'being', 'getting'}
            ),
            'correction': 'your',
            'message': '"you\'re" means "you are". Use "your" for possession.',
            'confidence': 75,
        },
        {
            'word': 'their',
            'check': lambda w, i, words, n: (
                i + 1 < n and words[i + 1].is_word and
                words[i + 1].pos in {'VBZ', 'VBP', 'VBD', 'VB', 'VBG', 'VBN', 'MD'}
            ),
            'correction': "they're",
            'message': '"their" is possessive. Use "they\'re" for "they are".',
            'confidence': 85,
        },
        {
            'word': "they're",
            'check': lambda w, i, words, n: (
                i + 1 < n and words[i + 1].is_word and
                words[i + 1].pos in {'NN', 'NNS', 'NNP', 'DT', 'JJ', 'PRP$'} and
                words[i + 1].lower not in {'going', 'coming', 'doing', 'being', 'getting'}
            ),
            'correction': 'their',
            'message': '"they\'re" means "they are". Use "their" for possession.',
            'confidence': 75,
        },
        {
            'word': "it's",
            'check': lambda w, i, words, n: (
                i + 1 < n and words[i + 1].is_word and
                words[i + 1].pos in {'NN', 'NNS', 'NNP', 'DT', 'JJ', 'PRP$'}
            ),
            'correction': 'its',
            'message': '"it\'s" means "it is". Use "its" for possession.',
            'confidence': 75,
        },
        {
            'word': 'its',
            'check': lambda w, i, words, n: (
                i + 1 < n and words[i + 1].is_word and
                words[i + 1].pos in {'VBZ', 'VBP', 'VBD', 'VB', 'VBG', 'VBN', 'MD'}
            ),
            'correction': "it's",
            'message': '"its" is possessive. Use "it\'s" for "it is".',
            'confidence': 80,
        },
        {
            'word': 'then',
            'check': lambda w, i, words, n: (
                i > 0 and words[i - 1].is_word and
                (words[i - 1].lower.endswith('er') or
                 words[i - 1].lower in {'more', 'less', 'better', 'worse', 'faster',
                                         'slower', 'bigger', 'smaller', 'taller',
                                         'shorter', 'longer', 'older', 'younger',
                                         'greater', 'higher', 'lower', 'stronger',
                                         'weaker', 'richer', 'poorer', 'easier',
                                         'harder', 'simpler'})
            ),
            'correction': 'than',
            'message': 'Use "than" for comparisons, not "then".',
            'confidence': 95,
        },
    ]

    for rule in context_rules:
        target = rule['word']
        for i, token in enumerate(words):
            w = token.lower
            if w == target:
                try:
                    if rule['check'](w, i, words, n):
                        errors.append({
                            'type': 'context', 'severity': 'warning',
                            'category': 'Confusing Words',
                            'incorrect': token.text, 'correction': rule['correction'],
                            'position': token.start, 'end_position': token.end,
                            'sentence': sentence.text,
                            'message': f'"{token.text}" may be incorrect in this context. {rule["message"]}',
                            'confidence': rule['confidence'], 'pass': 'context',
                        })
                except (IndexError, KeyError):
                    pass

    return errors
