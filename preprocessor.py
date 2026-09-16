import re
import unicodedata

SENTENCE_TERMINATORS = {'.', '!', '?'}
ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'ave', 'blvd',
    'dept', 'est', 'etc', 'inc', 'ltd', 'corp', 'vs', 'viz', 'al',
    'approx', 'appt', 'apt', 'dept', 'est', 'govt', 'jr', 'mt',
    'oct', 'pvt', 'rd', 'rep', 'rev', 'sen', 'sgt', 'sr', 'st',
    'tel', 'temp', 'vol', 'vs',
}
CONTRACTIONS = {
    "n't", "'re", "'ve", "'ll", "'d", "'m", "'s",
    "can't", "cannot", "couldn't", "shouldn't", "wouldn't", "won't",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "mustn't", "needn't",
    "i'm", "you're", "we're", "they're", "he's", "she's", "it's",
    "that's", "what's", "who's", "there's", "here's",
    "i've", "you've", "we've", "they've",
    "i'll", "you'll", "he'll", "she'll", "it'll", "we'll", "they'll",
    "i'd", "you'd", "he'd", "she'd", "it'd", "we'd", "they'd",
    "let's", "how's",
}

CONTRACTION_MAP = {
    "n't": {"base": "", "expansion": "not"},
    "'re": {"base": "", "expansion": "are"},
    "'ve": {"base": "", "expansion": "have"},
    "'ll": {"base": "", "expansion": "will"},
    "'d": {"base": "", "expansion": "would"},
    "'m": {"base": "", "expansion": "am"},
    "'s": {"base": "", "expansion": "'s"},
}


class Token:
    def __init__(self, text, start, end, is_word=False, is_punct=False, is_space=False):
        self.text = text
        self.start = start
        self.end = end
        self.is_word = is_word
        self.is_punct = is_punct
        self.is_space = is_space
        self.lower = text.lower()
        self.pos = None
        self.dep = None
        self.lemma = text.lower()

    def __repr__(self):
        return f"Token({self.text!r}, pos={self.pos})"


class Sentence:
    def __init__(self, text, tokens, start_idx=0):
        self.text = text
        self.tokens = tokens
        self.start_idx = start_idx
        self.words = [t for t in tokens if t.is_word]

    def __repr__(self):
        return f"Sentence({self.text[:50]!r}, words={len(self.words)})"


class Document:
    def __init__(self, text, sentences):
        self.text = text
        self.sentences = sentences
        self.all_tokens = []
        for s in sentences:
            self.all_tokens.extend(s.tokens)

    def __repr__(self):
        return f"Document(sentences={len(self.sentences)})"


def normalize_text(text):
    text = text.replace('\u200b', '').replace('\ufeff', '')
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"['']", "'", text)
    text = re.sub(r'…', '...', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'—', '-', text)
    return text


def tokenize(text):
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            start = i
            while i < len(text) and text[i].isspace():
                i += 1
            tokens.append(Token(text[start:i], start, i, is_space=True))
        elif text[i].isalpha() or (text[i] == "'" and i + 1 < len(text) and text[i + 1].isalpha()):
            start = i
            while i < len(text) and (text[i].isalpha() or text[i] == "'"):
                i += 1
            tokens.append(Token(text[start:i], start, i, is_word=True))
        elif text[i].isdigit():
            start = i
            while i < len(text) and (text[i].isdigit() or text[i] in ',.'):
                i += 1
            tokens.append(Token(text[start:i], start, i, is_word=True))
        else:
            start = i
            i += 1
            tokens.append(Token(text[start:i], start, i, is_punct=True))
    return tokens


def split_sentences(text, tokens):
    sentences = []
    current_tokens = []
    current_start = 0

    for token in tokens:
        current_tokens.append(token)
        if token.is_punct and token.text in SENTENCE_TERMINATORS:
            word_tokens = [t for t in current_tokens if t.is_word]
            if word_tokens:
                last_word = word_tokens[-1].lower
                if last_word not in ABBREVIATIONS:
                    sent_text = ''.join(t.text for t in current_tokens).strip()
                    sentences.append(Sentence(sent_text, list(current_tokens), current_start))
                    current_start = token.end
                    current_tokens = []

    if current_tokens:
        sent_text = ''.join(t.text for t in current_tokens).strip()
        if sent_text:
            sentences.append(Sentence(sent_text, current_tokens, current_start))

    if not sentences and text.strip():
        all_tokens = tokenize(text)
        sentences.append(Sentence(text.strip(), all_tokens, 0))

    return sentences


def preprocess(text):
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    sentences = split_sentences(normalized, tokens)
    return Document(normalized, sentences)


def expand_contractions(text):
    result = text
    replacements = [
        (r"\bcan't\b", "cannot"), (r"\bcannot\b", "cannot"),
        (r"\bwon't\b", "will not"), (r"\bwill\s+not\b", "will not"),
        (r"\bcan't\b", "cannot"), (r"\bshouldn't\b", "should not"),
        (r"\bwouldn't\b", "would not"), (r"\bcouldn't\b", "could not"),
        (r"\bdon't\b", "do not"), (r"\bdoesn't\b", "does not"),
        (r"\bdidn't\b", "did not"), (r"\bisn't\b", "is not"),
        (r"\baren't\b", "are not"), (r"\baren't\b", "are not"),
        (r"\bwasn't\b", "was not"), (r"\bweren't\b", "were not"),
        (r"\bhasn't\b", "has not"), (r"\bhaven't\b", "have not"),
        (r"\bhadn't\b", "had not"), (r"\bmustn't\b", "must not"),
        (r"\bneedn't\b", "need not"),
        (r"\bi'm\b", "I am"), (r"\bim\b", "I am"),
        (r"\bi've\b", "I have"), (r"\bive\b", "I have"),
        (r"\bi'll\b", "I will"), (r"\bi'd\b", "I would"),
        (r"\byou're\b", "you are"), (r"\byou've\b", "you have"),
        (r"\byou'll\b", "you will"), (r"\byou'd\b", "you would"),
        (r"\bwe're\b", "we are"), (r"\bwe've\b", "we have"),
        (r"\bwe'll\b", "we will"), (r"\bwe'd\b", "we would"),
        (r"\bthey're\b", "they are"), (r"\bthey've\b", "they have"),
        (r"\bthey'll\b", "they will"), (r"\bthey'd\b", "they would"),
        (r"\bhe's\b", "he is"), (r"\bshe's\b", "she is"),
        (r"\bit's\b", "it is"), (r"\bthat's\b", "that is"),
        (r"\bwhat's\b", "what is"), (r"\bwho's\b", "who is"),
        (r"\bthere's\b", "there is"), (r"\bhere's\b", "here is"),
        (r"\blet's\b", "let us"),
    ]
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result
