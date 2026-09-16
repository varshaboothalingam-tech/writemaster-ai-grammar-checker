from pos_tagger import (PRONOUNS_SUBJECT, PRONOUNS_OBJECT, PRONOUNS_POSSESSIVE,
                        PRONOUNS_REFLEXIVE, PREPOSITIONS, CONJUNCTIONS, COMMON_ADVERBS)

SUBJECT_IRIS = {
    "i", "you", "he", "she", "it", "we", "they",
    "who", "whom", "which", "that",
}

INFINITIVE_MARKERS = {"to"}

PHRASE_STARTERS = {
    "although", "because", "since", "while", "if", "when",
    "before", "after", "until", "unless", "whereas", "though",
    "by", "in", "on", "at", "for", "with", "without", "through",
    "during", "after", "before", "about", "above", "below", "under",
}

# Common false starts / fragments that lack a main verb
FRAGMENT_TRIGGERS = {
    "running", "walking", "having", "being", "doing", "making",
    "going", "coming", "taking", "getting", "giving", "finding",
    "thinking", "knowing", "seeing", "feeling", "looking",
}

PREPOSITION_STARTERS = {
    "in", "on", "at", "by", "for", "with", "without", "through",
    "during", "before", "after", "about", "above", "below", "under",
    "between", "among", "against", "along", "around", "behind",
    "beside", "beyond", "inside", "outside", "near", "toward",
    "towards", "upon", "within", "across", "beneath", "despite",
    "except", "like", "past", "per", "plus", "sans", "save",
    "than", "via", "vs", "versus",
}

COMMON_PREPOSITION_ERRORS = {
    'at': {'school': True, 'home': True, 'work': True, 'church': True, 'hospital': True},
    'to': {'school': True, 'home': True, 'work': True, 'church': True, 'hospital': True},
}

VERB_PREPOSITION_PAIRS = {
    'depend': 'on', 'listen': 'to', 'look': 'at', 'wait': 'for',
    'apologize': 'for', 'apply': 'for', 'ask': 'for', 'belong': 'to',
    'believe': 'in', 'care': 'about', 'consist': 'of', 'dream': 'about',
    'focus': 'on', 'insist': 'on', 'laugh': 'at', 'participate': 'in',
    'pay': 'for', 'point': 'at', 'refer': 'to', 'rely': 'on',
    'respond': 'to', 'smile': 'at', 'succeed': 'in', 'suffer': 'from',
    'think': 'about', 'worry': 'about',
}

WORD_ORDER_PATTERNS = [
    (['go', 'went', 'goes'], {'home', 'there', 'here'}),
]


def check_prepositions(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if w in VERB_PREPOSITION_PAIRS and i > 0:
            verb_token = None
            for j in range(i - 1, max(0, i - 4), -1):
                if words[j].is_word and words[j].pos in {'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN'}:
                    verb_token = words[j]
                    break
                if words[j].is_word and words[j].pos not in {'RB', 'JJ'}:
                    break
            if verb_token:
                expected_prep = VERB_PREPOSITION_PAIRS[verb_token.lower]
                if i + 1 < n and words[i + 1].is_word:
                    next_w = words[i + 1].lower
                    if w != expected_prep and next_w in PREPOSITIONS:
                        pass

    return errors


def check_conjunctions(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower

        if w in {'because', 'although', 'though', 'while', 'whereas', 'if', 'unless',
                 'until', 'since', 'when', 'whenever', 'where', 'wherever', 'whether'}:
            pass

    return errors


def check_word_order(sentence):
    errors = []
    words = sentence.words
    n = len(words)

    for i, token in enumerate(words):
        w = token.lower
        if w in {'never', 'always', 'often', 'usually', 'sometimes', 'rarely', 'seldom',
                 'frequently', 'occasionally', 'generally', 'normally', 'typically'}:
            if i + 1 < n and words[i + 1].is_word:
                next_pos = words[i + 1].pos
                if next_pos in {'VBZ', 'VBP', 'VBD'}:
                    if i > 0 and words[i - 1].is_word and words[i - 1].pos in {'VBZ', 'VBP', 'VBD'}:
                        errors.append({
                            'type': 'grammar', 'severity': 'warning',
                            'category': 'Word Order',
                            'incorrect': f'{words[i-1].text} {token.text}',
                            'correction': f'{token.text} {words[i-1].text}',
                            'position': words[i - 1].start,
                            'end_position': token.end,
                            'sentence': sentence.text,
                            'message': f'Adverb "{token.text}" should typically come before the verb.',
                            'confidence': 65, 'pass': 'structure',
                        })

    return errors


def check_double_negatives(sentence):
    errors = []
    words = sentence.words
    negations = {'not', "n't", 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere',
                 'nobody', 'no one', 'none', 'hardly', 'scarcely', 'barely'}
    contracted_negations = {"don't", "doesn't", "didn't", "won't", "wouldn't",
                            "couldn't", "shouldn't", "can't", "isn't", "aren't",
                            "wasn't", "weren't", "hasn't", "haven't", "hadn't"}
    plain_negations = {"dont", "doesnt", "didnt", "wont", "wouldnt",
                       "couldnt", "shouldnt", "cant", "isnt", "arent",
                       "wasnt", "werent", "hasnt", "havent", "hadnt"}
    neg_count = 0
    neg_positions = []
    for i, token in enumerate(words):
        w = token.lower
        is_neg = w in negations or w in contracted_negations or w in plain_negations
        if is_neg:
            neg_count += 1
            neg_positions.append((i, token))
        if w in {'no', 'nothing', 'nobody', 'nowhere', 'none', 'neither'}:
            if neg_count > 1 and neg_positions:
                prev_idx, prev_tok = neg_positions[-2]
                if i - prev_idx <= 5:
                    errors.append({
                        'type': 'grammar', 'severity': 'error',
                        'category': 'Double Negative',
                        'incorrect': f'{prev_tok.text} ... {token.text}',
                        'correction': 'Remove one negative',
                        'position': prev_tok.start, 'end_position': token.end,
                        'sentence': sentence.text,
                        'message': f'Double negative detected: "{prev_tok.text}" and "{token.text}". Use only one negative.',
                        'confidence': 90, 'pass': 'grammar',
                    })
    return errors


def check_comparatives(sentence):
    errors = []
    words = sentence.words
    n = len(words)
    for i, token in enumerate(words):
        w = token.lower
        if w == 'then' and i > 0:
            prev = words[i - 1].lower
            comparatives = {'more', 'less', 'better', 'worse', 'faster', 'slower', 'bigger',
                            'smaller', 'taller', 'shorter', 'longer', 'older', 'younger',
                            'greater', 'higher', 'lower', 'stronger', 'weaker', 'richer',
                            'poorer', 'easier', 'harder', 'simpler'}
            if prev.endswith('er') or prev in comparatives or prev == 'more':
                errors.append({
                    'type': 'grammar', 'severity': 'error',
                    'category': 'Comparative',
                    'incorrect': 'then', 'correction': 'than',
                    'position': token.start, 'end_position': token.end,
                    'sentence': sentence.text,
                    'message': 'Use "than" for comparisons, not "then".',
                    'confidence': 95, 'pass': 'grammar',
                })
    return errors


def check_sentence_structure(sentence):
    errors = []
    words = sentence.words
    n = len(words)
    if n == 0:
        return errors

    first_word = words[0]
    if first_word.is_word and first_word.text[0].islower() and first_word.lower != 'i':
        errors.append({
            'type': 'grammar', 'severity': 'error',
            'category': 'Capitalization',
            'incorrect': first_word.text,
            'correction': first_word.text[0].upper() + first_word.text[1:],
            'position': first_word.start, 'end_position': first_word.end,
            'sentence': sentence.text,
            'message': 'Sentences should begin with a capital letter.',
            'confidence': 95, 'pass': 'structure',
        })

    if not sentence.text.strip():
        return errors
    last_char = sentence.text.strip()[-1]
    if last_char not in '.!?':
        errors.append({
            'type': 'punctuation', 'severity': 'warning',
            'category': 'Punctuation',
            'incorrect': '', 'correction': '.',
            'position': len(sentence.text.strip()),
            'end_position': len(sentence.text.strip()),
            'sentence': sentence.text,
            'message': 'Sentences should end with proper punctuation (., !, or ?).',
            'confidence': 90, 'pass': 'structure',
        })

    for i in range(n - 1):
        if words[i].is_word and words[i + 1].is_word and words[i].lower == words[i + 1].lower:
            if words[i].lower not in {'that', 'this'}:
                errors.append({
                    'type': 'grammar', 'severity': 'warning',
                    'category': 'Repetition',
                    'incorrect': f'{words[i].text} {words[i+1].text}',
                    'correction': words[i].text,
                    'position': words[i].start, 'end_position': words[i + 1].end,
                    'sentence': sentence.text,
                    'message': f'Word "{words[i].text}" is repeated.',
                    'confidence': 95, 'pass': 'structure',
                })

    return errors


def check_fragment(sentence):
    """Detect sentence fragments (missing main verb or subject)."""
    errors = []
    words = sentence.words
    n = len(words)
    if n < 2:
        return errors

    has_verb = False
    has_subject = False

    for i, token in enumerate(words):
        if token.pos in ("VB", "VBP", "VBZ", "VBD", "VBN", "VBG"):
            has_verb = True
        if token.pos in ("NN", "NNS", "NNP", "NNPS", "PRP"):
            if token.lower not in PREPOSITION_STARTERS and token.lower not in PHRASE_STARTERS:
                has_subject = True

    if not has_verb:
        return errors

    if not has_subject:
        first_word = words[0].lower
        if first_word in PREPOSITION_STARTERS or first_word in FRAGMENT_TRIGGERS:
            errors.append({
                "type": "grammar",
                "severity": "warning",
                "category": "Fragment",
                "incorrect": sentence.text,
                "correction": "",
                "position": words[0].start,
                "end_position": words[-1].end,
                "sentence": sentence.text,
                "message": 'Possible sentence fragment. This may be missing a subject (e.g., add "I" or "we" at the beginning).',
                "confidence": 55,
                "pass": "structure",
            })

    return errors


def check_dangling_modifier(sentence):
    """Detect dangling modifiers (participial phrases not followed by the subject they modify)."""
    errors = []
    words = sentence.words
    n = len(words)
    if n < 4:
        return errors

    if words[0].pos == "VBG" or (words[0].lower in FRAGMENT_TRIGGERS and len(words) > 3):
        comma_index = -1
        for i, token in enumerate(words):
            if token.text == "," and i <= 5:
                comma_index = i
                break

        if comma_index > 0:
            after_comma = words[comma_index + 1:] if comma_index + 1 < n else []
            if after_comma:
                after_subject = after_comma[0]
                if after_subject.pos in ("NN", "NNS", "NNP", "NNPS"):
                    modifier_text = " ".join(w.text for w in words[:comma_index])
                    subject_text = after_subject.text
                    inanimate_subjects = {
                        "bus", "car", "train", "plane", "book", "report",
                        "essay", "paper", "document", "file", "system",
                        "program", "computer", "phone", "building", "house",
                        "door", "window", "road", "street", "table", "chair",
                        "weather", "rain", "snow", "wind", "storm",
                    }
                    if subject_text.lower() in inanimate_subjects:
                        errors.append({
                            "type": "grammar",
                            "severity": "warning",
                            "category": "Dangling Modifier",
                            "incorrect": sentence.text,
                            "correction": "",
                            "position": words[0].start,
                            "end_position": comma_index,
                            "sentence": sentence.text,
                            "message": f'Possibly dangling modifier: "{modifier_text}" does not logically modify "{subject_text}".',
                            "confidence": 60,
                            "pass": "structure",
                        })

    return errors


def check_parallelism(sentence):
    """Check for lack of parallel structure in lists."""
    errors = []
    words = sentence.words
    n = len(words)
    if n < 4:
        return errors

    items = []
    item_start = 0

    for i, token in enumerate(words):
        if token.lower in ("and", "or") and i > 0 and i < n - 1:
            item_text = " ".join(w.text for w in words[item_start:i])
            items.append(item_text)
            item_start = i + 1

    if len(items) < 2:
        return errors

    last_item_text = " ".join(w.text for w in words[item_start:])
    items.append(last_item_text)

    def get_item_start_pos(item_text):
        item_words = item_text.strip().split()
        if not item_words:
            return None
        first = item_words[0].lower
        verb_starters = {"to", "running", "walking", "having", "being", "doing",
                         "making", "going", "coming", "taking", "getting"}
        if first in verb_starters:
            return "infinitive" if first == "to" else "gerund"
        if first.endswith("ed"):
            return "past"
        return "other"

    start_types = [get_item_start_pos(item) for item in items]
    non_none = [t for t in start_types if t is not None]

    if len(non_none) >= 2:
        if len(set(non_none)) > 1:
            errors.append({
                "type": "grammar",
                "severity": "warning",
                "category": "Parallelism",
                "incorrect": " | ".join(items),
                "correction": "",
                "position": words[0].start,
                "end_position": words[-1].end,
                "sentence": sentence.text,
                "message": "Items in a list should use parallel structure (all start with the same grammatical form).",
                "confidence": 55,
                "pass": "structure",
            })

    return errors
