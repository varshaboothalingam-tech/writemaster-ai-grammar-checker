import spacy
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


@dataclass
class WordAnalysis:
    text: str
    index: int
    pos: str
    tag: str
    dep: str
    head_text: str
    head_index: int
    children: List[str] = field(default_factory=list)
    lemma: str = ""
    is_alpha: bool = True
    whitespace_after: str = " "

    def __repr__(self):
        return f"WordAnalysis({self.text!r}, pos={self.pos}, dep={self.dep})"


@dataclass
class SentenceAnalysis:
    text: str
    words: List[WordAnalysis]
    root: Optional[WordAnalysis] = None
    subjects: List[WordAnalysis] = field(default_factory=list)
    objects: List[WordAnalysis] = field(default_factory=list)
    verbs: List[WordAnalysis] = field(default_factory=list)
    auxiliaries: List[WordAnalysis] = field(default_factory=list)
    modifiers: List[WordAnalysis] = field(default_factory=list)
    prepositional_phrases: List[Tuple[WordAnalysis, WordAnalysis]] = field(default_factory=list)
    time_expressions: List[WordAnalysis] = field(default_factory=list)
    negations: List[WordAnalysis] = field(default_factory=list)
    noun_chunks: List[Tuple[int, int, str]] = field(default_factory=list)
    verb_chunks: List[Tuple[int, int, str]] = field(default_factory=list)


TIME_WORDS = {
    'yesterday', 'today', 'tomorrow', 'now', 'then', 'soon', 'recently',
    'currently', 'previously', 'always', 'never', 'often', 'sometimes',
    'usually', 'already', 'still', 'yet', 'just', 'ever', 'before',
    'after', 'during', 'since', 'until', 'while', 'ago', 'later',
    'earlier', 'morning', 'afternoon', 'evening', 'night', 'day',
    'week', 'month', 'year', 'decade', 'century', 'moment',
    'last', 'next', 'this', 'every', 'each'
}

PAST_TIME_INDICATORS = {
    'yesterday', 'ago', 'last', 'before', 'previously', 'earlier',
    'in 2020', 'in 2019', 'in 2018', 'in 2017', 'in 2016',
    'last week', 'last month', 'last year', 'last night',
    'last monday', 'last tuesday', 'last wednesday', 'last thursday',
    'last friday', 'last saturday', 'last sunday',
    'the other day', 'a while ago', 'some time ago',
}

FUTURE_TIME_INDICATORS = {
    'tomorrow', 'next', 'soon', 'later', 'in the future',
    'next week', 'next month', 'next year',
    'next monday', 'next tuesday', 'next wednesday', 'next thursday',
    'next friday', 'next saturday', 'next sunday',
}

GENDERED_NOUNS = {
    'boy': 'male', 'girl': 'female', 'man': 'male', 'woman': 'female',
    'husband': 'male', 'wife': 'female', 'father': 'male', 'mother': 'female',
    'son': 'male', 'daughter': 'female', 'brother': 'male', 'sister': 'female',
    'uncle': 'male', 'aunt': 'female', 'nephew': 'male', 'niece': 'female',
    'grandfather': 'male', 'grandmother': 'female', 'grandpa': 'male', 'grandma': 'female',
    'prince': 'male', 'princess': 'female', 'king': 'male', 'queen': 'female',
    'hero': 'male', 'heroine': 'female', 'actor': 'male', 'actress': 'female',
    'husband': 'male', 'wife': 'female', 'gentleman': 'male', 'lady': 'female',
    'sir': 'male', 'madam': 'female', 'mr': 'male', 'mrs': 'female', 'ms': 'female',
}

MALE_PRONOUNS = {'he', 'him', 'his', 'himself'}
FEMALE_PRONOUNS = {'she', 'her', 'hers', 'herself'}


def analyze_sentence(text: str) -> SentenceAnalysis:
    doc = nlp(text)
    words = []
    subjects = []
    objects = []
    verbs = []
    auxiliaries = []
    modifiers = []
    time_expressions = []
    negations = []
    noun_chunks = []
    verb_chunks = []
    root = None

    for token in doc:
        word = WordAnalysis(
            text=token.text,
            index=token.i,
            pos=token.pos_,
            tag=token.tag_,
            dep=token.dep_,
            head_text=token.head.text,
            head_index=token.head.i,
            children=[child.text for child in token.children],
            lemma=token.lemma_,
            is_alpha=token.is_alpha,
            whitespace_after=token.whitespace_,
        )
        words.append(word)

        if token.dep_ == 'ROOT':
            root = word
        if token.dep_ in ('nsubj', 'nsubjpass', 'csubj'):
            subjects.append(word)
        if token.dep_ in ('dobj', 'pobj', 'attr', 'oprd'):
            objects.append(word)
        if token.pos_ == 'VERB' or token.pos_ == 'AUX':
            verbs.append(word)
        if token.pos_ == 'AUX':
            auxiliaries.append(word)
        if token.pos_ in ('ADV', 'ADJ') or token.dep_ in ('advmod', 'amod'):
            modifiers.append(word)
        if token.lower_ in TIME_WORDS or token.dep_ == 'npadvmod':
            time_expressions.append(word)
        if token.dep_ == 'neg' or token.lower_ in ('not', "n't", 'no', 'never', 'neither', 'nor'):
            negations.append(word)

    for chunk in doc.noun_chunks:
        noun_chunks.append((chunk.start, chunk.end, chunk.text))

    verb_chunks_list = []
    for token in doc:
        if token.pos_ in ('VERB', 'AUX'):
            children_texts = [child.text for child in token.children]
            verb_chunks_list.append((token.i, token.i + 1, f"{token.text} {' '.join(children_texts)}"))
    verb_chunks = verb_chunks_list

    return SentenceAnalysis(
        text=text,
        words=words,
        root=root,
        subjects=subjects,
        objects=objects,
        verbs=verbs,
        auxiliaries=auxiliaries,
        modifiers=modifiers,
        time_expressions=time_expressions,
        negations=negations,
        noun_chunks=noun_chunks,
        verb_chunks=verb_chunks,
    )


def analyze_document(text: str) -> List[SentenceAnalysis]:
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        analysis = analyze_sentence(sent.text)
        sentences.append(analysis)
    return sentences


def get_word_by_index(words: List[WordAnalysis], index: int) -> Optional[WordAnalysis]:
    for w in words:
        if w.index == index:
            return w
    return None


def get_verb_chain(words: List[WordAnalysis], verb: WordAnalysis) -> List[WordAnalysis]:
    chain = [verb]
    for w in words:
        if w.head_index == verb.index and w.pos == 'AUX':
            chain.append(w)
    return sorted(chain, key=lambda w: w.index)


def get_subject_of_verb(words: List[WordAnalysis], verb: WordAnalysis) -> Optional[WordAnalysis]:
    for w in words:
        if w.head_index == verb.index and w.dep in ('nsubj', 'nsubjpass', 'csubj'):
            return w
    return None


def get_object_of_verb(words: List[WordAnalysis], verb: WordAnalysis) -> Optional[WordAnalysis]:
    for w in words:
        if w.head_index == verb.index and w.dep in ('dobj', 'attr', 'oprd'):
            return w
    return None


def is_third_person_singular(subject_text: str) -> bool:
    lower = subject_text.lower()
    if lower in ('he', 'she', 'it'):
        return True
    if lower in ('i', 'you', 'we', 'they'):
        return False
    if lower.endswith('s') and not lower.endswith('ss'):
        return True
    if lower in ('everyone', 'someone', 'anyone', 'nobody', 'somebody', 'anybody',
                  'everything', 'something', 'anything', 'nothing', 'each', 'every',
                  'neither', 'either'):
        return True
    return False


def is_plural_subject(subject_text: str) -> bool:
    lower = subject_text.lower()
    if lower in ('i', 'you', 'we', 'they'):
        return True
    if lower in ('he', 'she', 'it'):
        return False
    if lower.endswith('s') and not lower.endswith('ss'):
        return True
    if lower in ('people', 'children', 'men', 'women', 'police', 'cattle'):
        return True
    return False


def detect_past_time_context(words: List[WordAnalysis], sentence_text: str) -> bool:
    lower_text = sentence_text.lower()
    for indicator in PAST_TIME_INDICATORS:
        if indicator in lower_text:
            return True
    for w in words:
        if w.text.lower() in ('yesterday', 'ago', 'previously', 'earlier'):
            return True
        if w.dep_ if hasattr(w, 'dep_') else w.dep == 'npadvmod':
            if w.text.lower() in ('last', 'before'):
                return True
    return False


def detect_future_time_context(words: List[WordAnalysis], sentence_text: str) -> bool:
    lower_text = sentence_text.lower()
    for indicator in FUTURE_TIME_INDICATORS:
        if indicator in lower_text:
            return True
    for w in words:
        if w.text.lower() in ('tomorrow', 'soon', 'later'):
            return True
    return False
