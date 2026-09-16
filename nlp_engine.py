import spacy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load('en_core_web_sm')
    return _nlp


@dataclass
class NlpToken:
    text: str
    idx: int
    idx_end: int
    pos: str
    tag: str
    dep: str
    head_idx: int
    lemma: str
    is_alpha: bool
    is_stop: bool
    morph: Dict[str, str] = field(default_factory=dict)

    @property
    def lower(self):
        return self.text.lower()


@dataclass
class NlpSentence:
    text: str
    start_char: int
    end_char: int
    tokens: List[NlpToken] = field(default_factory=list)
    root_token: Optional[NlpToken] = None


@dataclass
class NlpDocument:
    text: str
    sentences: List[NlpSentence] = field(default_factory=list)


def _extract_morph(token) -> Dict[str, str]:
    morph = {}
    morph_str = str(token.morph)
    if morph_str:
        for pair in morph_str.split('|'):
            if '=' in pair:
                k, v = pair.split('=', 1)
                morph[k] = v
    return morph


def analyze_text(text: str) -> NlpDocument:
    nlp = get_nlp()
    doc = nlp(text)
    result = NlpDocument(text=text)

    for sent in doc.sents:
        nlp_sent = NlpSentence(
            text=sent.text,
            start_char=sent.start_char,
            end_char=sent.end_char,
        )
        for token in sent:
            nlp_token = NlpToken(
                text=token.text,
                idx=token.idx,
                idx_end=token.idx + len(token.text),
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                head_idx=token.head.idx,
                lemma=token.lemma_,
                is_alpha=token.is_alpha,
                is_stop=token.is_stop,
                morph=_extract_morph(token),
            )
            nlp_sent.tokens.append(nlp_token)
            if token.dep_ == 'ROOT':
                nlp_sent.root_token = nlp_token
        result.sentences.append(nlp_sent)

    return result


def get_children(sent: NlpSentence, token_idx: int) -> List[NlpToken]:
    return [t for t in sent.tokens if t.head_idx == token_idx and t.idx != token_idx]


def get_dependents(sent: NlpSentence, token_idx: int, recursive=True) -> List[NlpToken]:
    if not recursive:
        return get_children(sent, token_idx)
    result = []
    visited = set()
    stack = list(get_children(sent, token_idx))
    while stack:
        child = stack.pop()
        if child.idx in visited:
            continue
        visited.add(child.idx)
        result.append(child)
        stack.extend(get_children(sent, child.idx))
    return result


def get_subtree_tokens(sent: NlpSentence, token_idx: int, _visited=None) -> List[NlpToken]:
    if _visited is None:
        _visited = set()
    if token_idx in _visited:
        return []
    _visited.add(token_idx)
    children = get_children(sent, token_idx)
    tokens = [token_idx]
    for child in children:
        tokens.extend(get_subtree_tokens(sent, child.idx, _visited))
    tokens.sort(key=lambda idx: next(t.idx for t in sent.tokens if t.idx == idx))
    return [t for t in sent.tokens if t.idx in set(tokens)]


def find_subject(sent: NlpSentence, verb_token: NlpToken, _visited=None) -> Optional[NlpToken]:
    if _visited is None:
        _visited = set()
    if verb_token.idx in _visited:
        return None
    _visited.add(verb_token.idx)
    children = get_children(sent, verb_token.idx)
    for child in children:
        if child.dep in ('nsubj', 'nsubjpass'):
            return child
    head_token = next((t for t in sent.tokens if t.idx == verb_token.head_idx), None)
    if head_token and head_token.pos in ('VERB', 'AUX'):
        return find_subject(sent, head_token, _visited)
    return None


def find_object(sent: NlpSentence, verb_token: NlpToken) -> Optional[NlpToken]:
    children = get_children(sent, verb_token.idx)
    for child in children:
        if child.dep in ('dobj', 'attr', 'oprd'):
            return child
    return None


def get_verb_chain(sent: NlpSentence, verb_token: NlpToken) -> List[NlpToken]:
    chain = [verb_token]
    children = get_children(sent, verb_token.idx)
    for child in children:
        if child.dep in ('aux', 'auxpass', 'neg') and child.pos == 'AUX':
            chain.insert(0, child)
        elif child.dep == 'xcomp':
            chain.append(child)
    chain.sort(key=lambda t: t.idx)
    return chain


def is_compound_subject(sent: NlpSentence, token: NlpToken) -> bool:
    if token.dep in ('nsubj', 'nsubjpass'):
        children = get_children(sent, token.idx)
        for child in children:
            if child.dep == 'conj':
                return True
    return False


def get_compound_subjects(sent: NlpSentence, token: NlpToken) -> List[NlpToken]:
    subjects = [token]
    children = get_children(sent, token.idx)
    for child in children:
        if child.dep == 'conj' and child.pos in ('NOUN', 'PROPN', 'PRON'):
            subjects.append(child)
    return subjects
