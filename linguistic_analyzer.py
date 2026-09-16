import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from nlp_analyzer import (
    SentenceAnalysis,
    WordAnalysis,
    analyze_sentence,
    analyze_document,
    get_verb_chain,
    get_subject_of_verb,
    get_object_of_verb,
    is_third_person_singular,
    is_plural_subject,
    detect_past_time_context,
    detect_future_time_context,
    GENDERED_NOUNS,
    MALE_PRONOUNS,
    FEMALE_PRONOUNS,
)


@dataclass
class GrammarIssue:
    sentence: str
    incorrect: str
    correction: str
    category: str
    message: str
    confidence: float
    position: int
    severity: str
    word_start: int
    word_end: int

    def to_dict(self):
        return {
            'sentence': self.sentence,
            'incorrect': self.incorrect,
            'correction': self.correction,
            'category': self.category,
            'type': self.category,
            'message': self.message,
            'confidence': self.confidence,
            'position': self.position,
            'severity': self.severity,
            'word_start': self.word_start,
            'word_end': self.word_end,
        }


SUBJECT_PRONOUNS = {"i", "he", "she", "we", "they"}
OBJECT_PRONOUNS = {"me", "him", "her", "us", "them"}
PRONOUN_CASE_MAP = {
    "i": "me", "me": "i",
    "he": "him", "him": "he",
    "she": "her", "her": "she",
    "we": "us", "us": "we",
    "they": "them", "them": "they",
}
SUBJECT_TO_OBJECT = {
    "i": "me", "he": "him", "she": "her", "we": "us", "they": "them",
}
OBJECT_TO_SUBJECT = {
    "me": "i", "him": "he", "her": "she", "us": "we", "them": "they",
}

BE_FORMS = {"is", "am", "are", "was", "were", "be", "been", "being"}
HAVE_FORMS = {"have", "has", "had"}
DO_FORMS = {"do", "does", "did"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}
NEGATIVE_WORDS = {"no", "not", "never", "neither", "nobody", "nothing",
                  "nowhere", "nor", "cannot", "can't", "won't", "don't",
                  "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                  "hasn't", "haven't", "hadn't", "couldn't", "shouldn't",
                  "wouldn't", "mustn't"}
CONTRACTION_NEGATIVE = {"n't", "not"}

BASE_FORM_MAP = {
    "goes": "go", "does": "do", "has": "have",
    "plays": "play", "runs": "run", "walks": "walk",
    "eats": "eat", "drinks": "drink", "reads": "read",
    "writes": "write", "sings": "sing", "dances": "dance",
    "makes": "make", "takes": "take", "gives": "give",
    "comes": "come", "knows": "know", "thinks": "think",
    "says": "say", "sees": "see", "gets": "get",
    "puts": "put", "means": "mean", "sets": "set",
    "keeps": "keep", "lets": "let", "begins": "begin",
    "seems": "seem", "helps": "help", "shows": "show",
    "hears": "hear", "plays": "play", "moves": "move",
    "lives": "live", "believes": "believe", "holds": "hold",
    "brings": "bring", "happens": "happen", "writes": "write",
    "provides": "provide", "sits": "sit", "stands": "stand",
    "loses": "lose", "pays": "pay", "meets": "meet",
    "includes": "include", "continues": "continue",
    "follows": "follow", "learns": "learn", "changes": "change",
    "leads": "lead", "understands": "understand",
    "watches": "watch",
}

PAST_TO_BASE = {
    "went": "go", "did": "do", "had": "have", "said": "say",
    "got": "get", "made": "make", "knew": "know", "thought": "think",
    "came": "come", "saw": "see", "took": "take", "gave": "give",
    "found": "find", "told": "tell", "became": "become",
    "left": "leave", "felt": "feel", "brought": "bring",
    "began": "begin", "kept": "keep", "held": "hold",
    "wrote": "write", "stood": "stand", "heard": "hear",
    "let": "let", "meant": "mean", "ran": "run",
    "sat": "sit", "led": "lead", "lost": "lose",
    "paid": "pay", "met": "meet", "read": "read",
    "grew": "grow", "fell": "fall", "sent": "send",
    "built": "build", "spent": "spend", "won": "win",
    "taught": "teach", "caught": "catch", "drove": "drive",
    "chose": "choose", "spoke": "speak", "wore": "wear",
    "broke": "break", "forgot": "forget", "drew": "draw",
    "threw": "throw", "flew": "fly", "blew": "blow",
    "woke": "wake", "rode": "ride", "shook": "shake",
    "rose": "rise", "hid": "hide", "bit": "bite",
    "tore": "tear", "bore": "bear", "stole": "steal",
    "swore": "swear", "froze": "freeze", "sang": "sing",
    "drank": "drink", "rang": "ring", "swam": "swim",
    "sank": "sink", "sprang": "spring", "shrank": "shrink",
    "stank": "stink", "ate": "eat",
    "was": "be", "were": "be", "been": "be",
    "did": "do", "done": "do",
    "gone": "go", "came": "come",
    "sworn": "swear", "frozen": "freeze",
    "rung": "ring", "sung": "sing", "shrunk": "shrink",
    "sunk": "sink", "driven": "drive", "chosen": "choose",
    "spoken": "speak", "broken": "break", "forgotten": "forget",
    "drawn": "draw", "thrown": "throw", "flown": "fly",
    "blown": "blow", "woken": "wake", "risen": "rise",
    "hidden": "hide", "bitten": "bite", "eaten": "eat",
    "given": "give", "taken": "take", "seen": "see",
    "known": "know", "grown": "grow", "fallen": "fall",
    "begun": "begin", "swum": "swim", "drunk": "drink",
    "written": "write", "ridden": "ride", "shaken": "shake",
    "mistaken": "mistake", "forgiven": "forgive", "arisen": "arise",
    "borne": "bear", "worn": "wear", "torn": "tear",
}

IRREGULAR_VERBS = {
    'be': {'past': 'was', 'past_part': 'been'},
    'have': {'past': 'had', 'past_part': 'had'},
    'do': {'past': 'did', 'past_part': 'done'},
    'go': {'past': 'went', 'past_part': 'gone'},
    'eat': {'past': 'ate', 'past_part': 'eaten'},
    'drink': {'past': 'drank', 'past_part': 'drunk'},
    'sing': {'past': 'sang', 'past_part': 'sung'},
    'swim': {'past': 'swam', 'past_part': 'swum'},
    'ring': {'past': 'rang', 'past_part': 'rung'},
    'write': {'past': 'wrote', 'past_part': 'written'},
    'drive': {'past': 'drove', 'past_part': 'driven'},
    'speak': {'past': 'spoke', 'past_part': 'spoken'},
    'break': {'past': 'broke', 'past_part': 'broken'},
    'choose': {'past': 'chose', 'past_part': 'chosen'},
    'fly': {'past': 'flew', 'past_part': 'flown'},
    'grow': {'past': 'grew', 'past_part': 'grown'},
    'fall': {'past': 'fell', 'past_part': 'fallen'},
    'begin': {'past': 'began', 'past_part': 'begun'},
    'blow': {'past': 'blew', 'past_part': 'blown'},
    'throw': {'past': 'threw', 'past_part': 'thrown'},
    'draw': {'past': 'drew', 'past_part': 'drawn'},
    'freeze': {'past': 'froze', 'past_part': 'frozen'},
    'shake': {'past': 'shook', 'past_part': 'shaken'},
    'ride': {'past': 'rode', 'past_part': 'ridden'},
    'wake': {'past': 'woke', 'past_part': 'woken'},
    'rise': {'past': 'rose', 'past_part': 'risen'},
    'hide': {'past': 'hid', 'past_part': 'hidden'},
    'bite': {'past': 'bit', 'past_part': 'bitten'},
    'tear': {'past': 'tore', 'past_part': 'torn'},
    'wear': {'past': 'wore', 'past_part': 'worn'},
    'bear': {'past': 'bore', 'past_part': 'borne'},
    'steal': {'past': 'stole', 'past_part': 'stolen'},
    'swear': {'past': 'swore', 'past_part': 'sworn'},
    'forget': {'past': 'forgot', 'past_part': 'forgotten'},
    'forgive': {'past': 'forgave', 'past_part': 'forgiven'},
    'give': {'past': 'gave', 'past_part': 'given'},
    'take': {'past': 'took', 'past_part': 'taken'},
    'see': {'past': 'saw', 'past_part': 'seen'},
    'know': {'past': 'knew', 'past_part': 'known'},
    'say': {'past': 'said', 'past_part': 'said'},
    'get': {'past': 'got', 'past_part': 'got'},
    'make': {'past': 'made', 'past_part': 'made'},
    'think': {'past': 'thought', 'past_part': 'thought'},
    'come': {'past': 'came', 'past_part': 'come'},
    'find': {'past': 'found', 'past_part': 'found'},
    'tell': {'past': 'told', 'past_part': 'told'},
    'feel': {'past': 'felt', 'past_part': 'felt'},
    'leave': {'past': 'left', 'past_part': 'left'},
    'keep': {'past': 'kept', 'past_part': 'kept'},
    'hold': {'past': 'held', 'past_part': 'held'},
    'bring': {'past': 'brought', 'past_part': 'brought'},
    'catch': {'past': 'caught', 'past_part': 'caught'},
    'teach': {'past': 'taught', 'past_part': 'taught'},
    'build': {'past': 'built', 'past_part': 'built'},
    'spend': {'past': 'spent', 'past_part': 'spent'},
    'send': {'past': 'sent', 'past_part': 'sent'},
    'sell': {'past': 'sold', 'past_part': 'sold'},
    'win': {'past': 'won', 'past_part': 'won'},
    'pay': {'past': 'paid', 'past_part': 'paid'},
    'meet': {'past': 'met', 'past_part': 'met'},
    'lose': {'past': 'lost', 'past_part': 'lost'},
    'sit': {'past': 'sat', 'past_part': 'sat'},
    'stand': {'past': 'stood', 'past_part': 'stood'},
    'lead': {'past': 'led', 'past_part': 'led'},
    'read': {'past': 'read', 'past_part': 'read'},
    'feed': {'past': 'fed', 'past_part': 'fed'},
    'show': {'past': 'showed', 'past_part': 'shown'},
    'hear': {'past': 'heard', 'past_part': 'heard'},
    'run': {'past': 'ran', 'past_part': 'run'},
    'cut': {'past': 'cut', 'past_part': 'cut'},
    'put': {'past': 'put', 'past_part': 'put'},
    'set': {'past': 'set', 'past_part': 'set'},
    'let': {'past': 'let', 'past_part': 'let'},
    'hit': {'past': 'hit', 'past_part': 'hit'},
    'cost': {'past': 'cost', 'past_part': 'cost'},
}

BASE_TO_PAST = {}
for _base, _forms in IRREGULAR_VERBS.items():
    past = _forms.get('past', '')
    if past and past not in BASE_TO_PAST:
        BASE_TO_PAST[_base] = past
for k, v in PAST_TO_BASE.items():
    if v not in BASE_TO_PAST:
        BASE_TO_PAST[v] = k


def _get_past_tense(verb_lower: str) -> str:
    if verb_lower in BASE_TO_PAST:
        return BASE_TO_PAST[verb_lower]
    if verb_lower.endswith('e'):
        return verb_lower + 'd'
    if verb_lower.endswith('y') and len(verb_lower) > 1 and verb_lower[-2] not in 'aeiou':
        return verb_lower[:-1] + 'ied'
    if (verb_lower.endswith(('s', 'sh', 'ch', 'x', 'z'))
            or (verb_lower.endswith('o') and verb_lower[-1] == 'o')):
        return verb_lower + 'es'
    return verb_lower + 'ed'


def _get_past_participle(verb_lower: str) -> str:
    for base_v, forms in IRREGULAR_VERBS.items():
        if forms.get('past') == verb_lower:
            return forms.get('past_part', verb_lower)
    for base_v, forms in IRREGULAR_VERBS.items():
        if base_v == verb_lower:
            return forms.get('past_part', verb_lower)
    if verb_lower.endswith('e'):
        return verb_lower + 'd'
    if verb_lower.endswith('y') and len(verb_lower) > 1 and verb_lower[-2] not in 'aeiou':
        return verb_lower[:-1] + 'ied'
    if (verb_lower.endswith(('s', 'sh', 'ch', 'x', 'z'))
            or (verb_lower.endswith('o') and verb_lower[-1] == 'o')):
        return verb_lower + 'es'
    return verb_lower + 'ed'

UNCOUNTABLE_NOUNS = {
    "information", "advice", "furniture", "luggage", "equipment",
    "knowledge", "music", "water", "food", "money", "traffic",
    "weather", "news", "homework", "evidence", "research",
    "progress", "happiness", "sadness", "anger", "love",
    "hate", "courage", "patience", "experience", "software",
    "hardware", "paper", "glass", "wood", "metal", "plastic",
    "electricity", "energy", "time", "space", "air", "earth",
    "fire", "gold", "silver", "iron", "copper", "rice",
    "wheat", "corn", "milk", "tea", "coffee", "bread",
    "sugar", "salt", "flour", "oil", "gas", "coal",
    "stone", "dust", "dirt", "mud", "sand", "snow",
    "ice", "fog", "rain", "sunshine", "moonlight",
}

INDEFINITE_ARTICLES = {"a", "an", "the", "some", "any", "no", "every", "each",
                       "my", "your", "his", "her", "its", "our", "their",
                       "this", "that", "these", "those", "one", "two",
                       "three", "four", "five", "many", "few", "several",
                       "much", "all", "most", "more", "less", "enough"}

PLURAL_DETERMINERS = {"my", "your", "his", "her", "its", "our", "their",
                      "these", "those", "many", "few", "several", "all",
                      "most", "some", "two", "three", "four", "five"}

GIRL_NOUNS = {"girl", "woman", "lady", "daughter", "mother", "mom",
              "mother", "sister", "grandmother", "grandma", "wife",
              "aunt", "niece", "girlfriend", "bride", "heroine",
              "actress", "princess", "queen", "empress", "goddess"}

BOY_NOUNS = {"boy", "man", "gentleman", "son", "father", "dad",
             "brother", "grandfather", "grandpa", "husband",
             "uncle", "nephew", "boyfriend", "groom", "hero",
             "actor", "prince", "king", "emperor", "god"}


def _find_word_in_analysis(analysis: SentenceAnalysis, word_text: str) -> Optional[WordAnalysis]:
    target = word_text.lower()
    for word in analysis.words:
        if word.text.lower() == target or word.lemma.lower() == target:
            return word
    return None


def _get_position_in_sentence(sentence: str, word: str, offset: int = 0) -> Tuple[int, int]:
    lower_sentence = sentence.lower()
    lower_word = word.lower()
    start = lower_sentence.find(lower_word, offset)
    if start == -1:
        return (-1, -1)
    return (start, start + len(word))


def _get_token_position(sentence: str, token_index: int, words: List[WordAnalysis]) -> Tuple[int, int]:
    pos = 0
    for i, w in enumerate(words):
        if i == token_index:
            return (pos, pos + len(w.text))
        pos += len(w.text) + 1
    return (-1, -1)


def _is_negative_token(word: WordAnalysis) -> bool:
    text = word.text.lower()
    if text in NEGATIVE_WORDS:
        return True
    if text.endswith("n't"):
        return True
    if word.lemma == "not":
        return True
    return False


def _check_subject_verb_agreement(analysis: SentenceAnalysis) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    sentence = analysis.text

    for word in analysis.words:
        if word.dep not in ("ROOT",) and not word.dep.startswith("aux"):
            continue
        if word.pos not in ("VERB", "AUX"):
            continue

        subject = get_subject_of_verb(analysis.words, word)
        if subject is None:
            continue

        subject_lower = subject.text.lower()
        verb_lower = word.text.lower()

        if subject_lower in ("i", "you", "we", "they"):
            if verb_lower in BE_FORMS:
                if subject_lower == "i" and verb_lower == "is":
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction="am",
                        category="subject-verb-agreement",
                        message=f"The subject 'I' requires 'am', not 'is'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
                elif subject_lower == "i" and verb_lower == "are":
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction="am",
                        category="subject-verb-agreement",
                        message=f"The subject 'I' requires 'am', not 'are'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
                elif subject_lower in ("we", "they") and verb_lower == "is":
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction="are",
                        category="subject-verb-agreement",
                        message=f"The subject '{subject.text}' is plural and requires 'are', not 'is'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
                elif subject_lower in ("we", "they") and verb_lower == "was":
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction="were",
                        category="subject-verb-agreement",
                        message=f"The subject '{subject.text}' is plural and requires 'were', not 'was'.",
                        confidence=0.90,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
            elif verb_lower in ("does", "has") and subject_lower in ("i", "you", "we", "they"):
                base = "do" if verb_lower == "does" else "have"
                pos_tuple = _get_position_in_sentence(sentence, word.text)
                issues.append(GrammarIssue(
                    sentence=sentence,
                    incorrect=word.text,
                    correction=base,
                    category="subject-verb-agreement",
                    message=f"The subject '{subject.text}' requires '{base}', not '{verb_lower}'.",
                    confidence=0.95,
                    position=pos_tuple[0],
                    severity="error",
                    word_start=pos_tuple[0],
                    word_end=pos_tuple[1],
                ))
            elif verb_lower == "goes":
                pos_tuple = _get_position_in_sentence(sentence, word.text)
                issues.append(GrammarIssue(
                    sentence=sentence,
                    incorrect=word.text,
                    correction="go",
                    category="subject-verb-agreement",
                    message=f"The subject '{subject.text}' requires 'go', not 'goes'.",
                    confidence=0.90,
                    position=pos_tuple[0],
                    severity="error",
                    word_start=pos_tuple[0],
                    word_end=pos_tuple[1],
                ))
            else:
                if verb_lower in BASE_FORM_MAP:
                    base = BASE_FORM_MAP[verb_lower]
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction=base,
                        category="subject-verb-agreement",
                        message=f"The subject '{subject.text}' requires the base form '{base}', not '{verb_lower}'.",
                        confidence=0.80,
                        position=pos_tuple[0],
                        severity="warning",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))

        elif subject_lower in ("he", "she", "it") or is_third_person_singular(subject.text):
            if verb_lower in BE_FORMS:
                if subject_lower in ("he", "she", "it") and verb_lower == "are":
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction="is",
                        category="subject-verb-agreement",
                        message=f"The subject '{subject.text}' is third person singular and requires 'is', not 'are'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
                elif subject_lower in ("he", "she", "it") and verb_lower == "were":
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction="was",
                        category="subject-verb-agreement",
                        message=f"The subject '{subject.text}' is third person singular and requires 'was', not 'were'.",
                        confidence=0.90,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
                elif subject_lower == "i" and verb_lower in ("is", "are", "was", "were"):
                    expected = "am" if verb_lower in ("is", "are") else "was"
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction=expected,
                        category="subject-verb-agreement",
                        message=f"The subject 'I' requires '{expected}', not '{verb_lower}'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
            elif verb_lower in ("do", "have") and subject_lower in ("he", "she", "it"):
                expected = "does" if verb_lower == "do" else "has"
                pos_tuple = _get_position_in_sentence(sentence, word.text)
                issues.append(GrammarIssue(
                    sentence=sentence,
                    incorrect=word.text,
                    correction=expected,
                    category="subject-verb-agreement",
                    message=f"The subject '{subject.text}' requires '{expected}', not '{verb_lower}'.",
                    confidence=0.95,
                    position=pos_tuple[0],
                    severity="error",
                    word_start=pos_tuple[0],
                    word_end=pos_tuple[1],
                ))
            elif verb_lower == "go" and subject_lower in ("he", "she", "it"):
                pos_tuple = _get_position_in_sentence(sentence, word.text)
                issues.append(GrammarIssue(
                    sentence=sentence,
                    incorrect=word.text,
                    correction="goes",
                    category="subject-verb-agreement",
                    message=f"The subject '{subject.text}' requires 'goes', not 'go'.",
                    confidence=0.90,
                    position=pos_tuple[0],
                    severity="error",
                    word_start=pos_tuple[0],
                    word_end=pos_tuple[1],
                ))

        if subject_lower == "there":
            for child in analysis.words:
                if child.head_text == word.text and child.dep == "attr":
                    if child.pos in ("NOUN", "PROPN") and is_plural_subject(child.text):
                        if verb_lower == "is":
                            pos_tuple = _get_position_in_sentence(sentence, word.text)
                            issues.append(GrammarIssue(
                                sentence=sentence,
                                incorrect=word.text,
                                correction="are",
                                category="subject-verb-agreement",
                                message=f"'There is' with plural subject '{child.text}' requires 'are'.",
                                confidence=0.90,
                                position=pos_tuple[0],
                                severity="error",
                                    word_start=pos_tuple[0],
                                    word_end=pos_tuple[1],
                                ))

    for word in analysis.words:
        if word.pos == "PRON" and word.dep == "conj":
            head = None
            for w in analysis.words:
                if w.text == word.head_text:
                    head = w
                    break
            if head and head.dep in ("nsubj", "nsubjpass"):
                word_lower = word.text.lower()
                if word_lower in OBJECT_PRONOUNS:
                    correction = OBJECT_TO_SUBJECT.get(word_lower)
                    if correction:
                        pos_tuple = _get_position_in_sentence(sentence, word.text)
                        issues.append(GrammarIssue(
                            sentence=sentence, incorrect=word.text, correction=correction,
                            category="pronoun-case",
                            message=f"In compound subject, use '{correction}' instead of '{word.text}'.",
                            confidence=0.90, position=pos_tuple[0], severity="error",
                            word_start=pos_tuple[0], word_end=pos_tuple[1],
                        ))

    return issues


def _check_tense_consistency(analysis: SentenceAnalysis) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    sentence = analysis.text

    past_context = False
    future_context = False
    for w in analysis.words:
        if w.dep in ("npadvmod", "advmod", "tmod", "amod") or w.pos in ("NOUN", "PROPN", "ADV", "ADJ"):
            lower = w.text.lower()
            if lower in ('yesterday', 'ago', 'previously', 'earlier'):
                past_context = True
            if lower == 'last' or lower == 'before':
                next_words = [x for x in analysis.words if x.index > w.index and x.index <= w.index + 2]
                if any(n.text.lower() in ('week', 'month', 'year', 'night', 'time', 'monday', 'tuesday',
                                    'wednesday', 'thursday', 'friday', 'saturday', 'sunday') for n in next_words):
                    past_context = True
            if lower in ('tomorrow', 'soon', 'later'):
                future_context = True
            if lower == 'next':
                next_words = [x for x in analysis.words if x.index > w.index and x.index <= w.index + 2]
                if any(n.text.lower() in ('week', 'month', 'year', 'night', 'time', 'monday', 'tuesday',
                                    'wednesday', 'thursday', 'friday', 'saturday', 'sunday') for n in next_words):
                    future_context = True
    lower_text = sentence.lower()
    if ' yesterday' in lower_text or lower_text.startswith('yesterday'):
        past_context = True
    if ' ago' in lower_text:
        past_context = True

    root_verbs = [w for w in analysis.words if w.dep == "ROOT"]

    for verb in root_verbs:
        verb_lower = verb.text.lower()

        if verb.pos == "AUX" and verb_lower in ("is", "am", "are", "was", "were"):
            if past_context and verb_lower in ("is", "am", "are"):
                correction = "was" if verb_lower != "are" else "were"
                pos_tuple = _get_position_in_sentence(sentence, verb.text)
                issues.append(GrammarIssue(
                    sentence=sentence, incorrect=verb.text, correction=correction,
                    category="tense-consistency",
                    message=f"Past time context detected but verb is in present tense. Use '{correction}'.",
                    confidence=0.85, position=pos_tuple[0], severity="error",
                    word_start=pos_tuple[0], word_end=pos_tuple[1],
                ))

        if verb.pos == "VERB":
            has_aux = any(w.head_text == verb.text and w.dep in ("aux", "aux:pass") for w in analysis.words)
            aux_words = [w for w in analysis.words if w.head_text == verb.text and w.dep in ("aux", "aux:pass")]
            aux_lower = [a.text.lower() for a in aux_words]

            if past_context and not has_aux:
                if verb_lower in ("is", "am", "are"):
                    correction = "was" if verb_lower != "are" else "were"
                    pos_tuple = _get_position_in_sentence(sentence, verb.text)
                    issues.append(GrammarIssue(
                        sentence=sentence, incorrect=verb.text, correction=correction,
                        category="tense-consistency",
                        message=f"Past time context detected but verb is in present tense. Use '{correction}'.",
                        confidence=0.85, position=pos_tuple[0], severity="error",
                        word_start=pos_tuple[0], word_end=pos_tuple[1],
                    ))
                elif verb_lower in ("do", "does"):
                    correction = "did"
                    pos_tuple = _get_position_in_sentence(sentence, verb.text)
                    issues.append(GrammarIssue(
                        sentence=sentence, incorrect=verb.text, correction=correction,
                        category="tense-consistency",
                        message=f"Past time context detected but verb is in present tense. Use '{correction}'.",
                        confidence=0.85, position=pos_tuple[0], severity="error",
                        word_start=pos_tuple[0], word_end=pos_tuple[1],
                    ))
                elif verb.tag in ("VBP",) and verb_lower not in BE_FORMS and verb_lower not in DO_FORMS:
                    past_form = _get_past_tense(verb_lower)
                    if past_form and past_form != verb_lower:
                        pos_tuple = _get_position_in_sentence(sentence, verb.text)
                        issues.append(GrammarIssue(
                            sentence=sentence, incorrect=verb.text, correction=past_form,
                            category="tense-consistency",
                            message=f"Past time context detected but verb is in present tense. Use '{past_form}'.",
                            confidence=0.85, position=pos_tuple[0], severity="error",
                            word_start=pos_tuple[0], word_end=pos_tuple[1],
                        ))
                    elif verb_lower.endswith('s') and len(verb_lower) > 3:
                        base = verb_lower
                        if base.endswith('es') and len(base) > 4:
                            base = base[:-2]
                        elif base.endswith('ies'):
                            base = base[:-3] + 'y'
                        elif base.endswith('s'):
                            base = base[:-1]
                        past_form = _get_past_tense(base)
                        if past_form and past_form != base:
                            pos_tuple = _get_position_in_sentence(sentence, verb.text)
                            issues.append(GrammarIssue(
                                sentence=sentence, incorrect=verb.text, correction=past_form,
                                category="tense-consistency",
                                message=f"Past time context detected but verb is in present tense. Use '{past_form}'.",
                                confidence=0.80, position=pos_tuple[0], severity="warning",
                                word_start=pos_tuple[0], word_end=pos_tuple[1],
                            ))

            if 'will' in aux_lower or 'would' in aux_lower or 'shall' in aux_lower or 'should' in aux_lower:
                if verb_lower in PAST_TO_BASE:
                    base = PAST_TO_BASE[verb_lower]
                    pos_tuple = _get_position_in_sentence(sentence, verb.text)
                    modal_word = next(a for a in aux_lower if a in ('will','would','shall','should'))
                    issues.append(GrammarIssue(
                        sentence=sentence, incorrect=verb.text, correction=base,
                        category="tense-consistency",
                        message=f"After '{modal_word}', use base form '{base}' instead of '{verb_lower}'.",
                        confidence=0.95, position=pos_tuple[0], severity="error",
                        word_start=pos_tuple[0], word_end=pos_tuple[1],
                    ))

            if any(a in ('have', 'has', 'had') for a in aux_lower):
                if verb_lower in PAST_TO_BASE:
                    past_part = None
                    for base_v, forms in IRREGULAR_VERBS.items():
                        if forms['past'] == verb_lower:
                            past_part = forms['past_part']
                            break
                    if not past_part:
                        if verb_lower.endswith('ed') and len(verb_lower) > 3:
                            past_part = verb_lower
                        elif verb_lower.endswith('e'):
                            past_part = verb_lower + 'd'
                        else:
                            past_part = verb_lower + 'ed'
                    pos_tuple = _get_position_in_sentence(sentence, verb.text)
                    issues.append(GrammarIssue(
                        sentence=sentence, incorrect=verb.text, correction=past_part,
                        category="tense-consistency",
                        message=f"After 'have/has/had', use past participle '{past_part}' instead of '{verb_lower}'.",
                        confidence=0.90, position=pos_tuple[0], severity="error",
                        word_start=pos_tuple[0], word_end=pos_tuple[1],
                    ))

    for word in analysis.words:
        word_lower = word.text.lower()

        if word_lower in ("do", "does") and word.dep == "aux":
            root = [w for w in analysis.words if w.dep == "ROOT" and w.head_text == w.text]
            if not root:
                root = [w for w in analysis.words if w.dep == "ROOT"]
            if root:
                main_verb = root[0]
                main_lower = main_verb.text.lower()
                if word_lower == "does" and main_verb.text.lower() not in ('go', 'do'):
                    if main_verb.tag in ("VBP", "VB") and main_lower not in PAST_TO_BASE:
                        pass

        if word_lower in ("didn't", "did not", "don't", "doesn't"):
            for child in analysis.words:
                if child.head_text == word.text and child.dep in ("ROOT", "xcomp", "ccomp"):
                    child_lower = child.text.lower()
                    if child_lower in PAST_TO_BASE:
                        base = PAST_TO_BASE[child_lower]
                        pos_tuple = _get_position_in_sentence(sentence, child.text)
                        issues.append(GrammarIssue(
                            sentence=sentence, incorrect=child.text, correction=base,
                            category="tense-consistency",
                            message=f"After '{word_lower}', use base form '{base}' instead of past tense '{child_lower}'.",
                            confidence=0.95, position=pos_tuple[0], severity="error",
                            word_start=pos_tuple[0], word_end=pos_tuple[1],
                        ))
                    break

        if word_lower in HAVE_FORMS and word.dep == "aux":
            for child in analysis.words:
                if child.dep == "ROOT" and child.text.lower() != word_lower:
                    child_lower = child.text.lower()
                    if child_lower in PAST_TO_BASE:
                        past_part = None
                        for base_v, forms in IRREGULAR_VERBS.items():
                            if forms['past'] == child_lower:
                                past_part = forms['past_part']
                                break
                        if not past_part:
                            if child_lower.endswith('ed'):
                                past_part = child_lower
                            elif child_lower.endswith('e'):
                                past_part = child_lower + 'd'
                            else:
                                past_part = child_lower + 'ed'
                        pos_tuple = _get_position_in_sentence(sentence, child.text)
                        issues.append(GrammarIssue(
                            sentence=sentence, incorrect=child.text, correction=past_part,
                            category="tense-consistency",
                            message=f"After 'have/has/had', use past participle '{past_part}' instead of past tense '{child_lower}'.",
                            confidence=0.90, position=pos_tuple[0], severity="error",
                            word_start=pos_tuple[0], word_end=pos_tuple[1],
                        ))
                    break

        if word_lower in MODALS and word.dep == "aux":
            for child in analysis.words:
                if child.dep == "ROOT" and child.text.lower() != word_lower:
                    child_lower = child.text.lower()
                    if child_lower in PAST_TO_BASE:
                        base = PAST_TO_BASE[child_lower]
                        pos_tuple = _get_position_in_sentence(sentence, child.text)
                        issues.append(GrammarIssue(
                            sentence=sentence, incorrect=child.text, correction=base,
                            category="tense-consistency",
                            message=f"After modal '{word_lower}', use base form '{base}' instead of '{child_lower}'.",
                            confidence=0.95, position=pos_tuple[0], severity="error",
                            word_start=pos_tuple[0], word_end=pos_tuple[1],
                        ))
                    break

    return issues


def _check_pronoun_case(analysis: SentenceAnalysis) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    sentence = analysis.text

    for word in analysis.words:
        if word.pos != "PRON":
            continue

        word_lower = word.text.lower()

        if word.dep in ("nsubj", "nsubjpass"):
            if word_lower in OBJECT_PRONOUNS and word_lower not in ("her",):
                correction = OBJECT_TO_SUBJECT.get(word_lower)
                if correction:
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction=correction,
                        category="pronoun-case",
                        message=f"Subject position requires '{correction}', not '{word.text}'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
            elif word_lower == "her" and word.dep in ("nsubj", "nsubjpass"):
                pos_tuple = _get_position_in_sentence(sentence, word.text)
                issues.append(GrammarIssue(
                    sentence=sentence,
                    incorrect=word.text,
                    correction="she",
                    category="pronoun-case",
                    message="Subject position requires 'she', not 'her'.",
                    confidence=0.95,
                    position=pos_tuple[0],
                    severity="error",
                    word_start=pos_tuple[0],
                    word_end=pos_tuple[1],
                ))

        if word.dep in ("dobj", "pobj", "iobj", "attr"):
            if word_lower in SUBJECT_PRONOUNS and word_lower not in ("you",):
                correction = SUBJECT_TO_OBJECT.get(word_lower)
                if correction:
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction=correction,
                        category="pronoun-case",
                        message=f"Object position requires '{correction}', not '{word.text}'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))

        if word.dep == "poss":
            if word_lower in OBJECT_PRONOUNS and word_lower not in ("her",):
                correction = OBJECT_TO_SUBJECT.get(word_lower)
                if correction:
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction=correction,
                        category="pronoun-case",
                        message=f"Possessive position requires '{correction}' (e.g., '{correction} book'), not '{word.text}'.",
                        confidence=0.85,
                        position=pos_tuple[0],
                        severity="warning",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))

    for word in analysis.words:
        if word.text.lower() in ("and", "&"):
            head_word = None
            for w in analysis.words:
                if w.text == word.head_text:
                    head_word = w
                    break
            if head_word and head_word.dep in ("nsubj", "nsubjpass"):
                for child in analysis.words:
                    if child.head_text == head_word.text and child.dep == "conj" and child.pos == "PRON":
                        child_lower = child.text.lower()
                        if child_lower in OBJECT_PRONOUNS:
                            correction = OBJECT_TO_SUBJECT.get(child_lower)
                            if correction:
                                pos_tuple = _get_position_in_sentence(sentence, child.text)
                                issues.append(GrammarIssue(
                                    sentence=sentence,
                                    incorrect=child.text,
                                    correction=correction,
                                    category="pronoun-case",
                                    message=f"In compound subject, use '{correction}' instead of '{child.text}'.",
                                    confidence=0.90,
                                    position=pos_tuple[0],
                                    severity="error",
                                    word_start=pos_tuple[0],
                                    word_end=pos_tuple[1],
                                ))

    return issues


def _check_double_negation(analysis: SentenceAnalysis) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    sentence = analysis.text
    tokens_lower = [w.text.lower().replace("n't", "") for w in analysis.words]
    tokens_lower_full = [w.text.lower() for w in analysis.words]

    negative_count = 0
    negative_positions: List[Tuple[int, str]] = []

    for i, word in enumerate(analysis.words):
        if _is_negative_token(word):
            negative_count += 1
            negative_positions.append((i, word.text))

    if negative_count >= 2:
        for i, word in enumerate(analysis.words):
            word_lower = word.text.lower()
            if word_lower in ("no", "nothing", "nobody", "nowhere", "neither", "nor"):
                if i > 0:
                    prev_word = analysis.words[i - 1]
                    if _is_negative_token(prev_word) or prev_word.text.lower().replace("n't", "") in ("do", "does", "did", "is", "are", "was", "were", "have", "has", "had", "will", "would", "can", "could"):
                        pos_tuple = _get_position_in_sentence(sentence, word.text)
                        correction = "any" if word_lower == "no" else (
                            "anyone" if word_lower == "nobody" else (
                                "anything" if word_lower == "nothing" else (
                                    "anywhere" if word_lower == "nowhere" else "any"
                                )
                            )
                        )
                        issues.append(GrammarIssue(
                            sentence=sentence,
                            incorrect=word.text,
                            correction=correction,
                            category="double-negation",
                            message=f"Double negation detected. Use '{correction}' instead of '{word.text}'.",
                            confidence=0.90,
                            position=pos_tuple[0],
                            severity="error",
                            word_start=pos_tuple[0],
                            word_end=pos_tuple[1],
                        ))

    return issues


def _check_article_usage(analysis: SentenceAnalysis) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    sentence = analysis.text

    for i, word in enumerate(analysis.words):
        if word.pos != "NOUN":
            continue
        if word.text.lower() in UNCOUNTABLE_NOUNS:
            if i > 0:
                prev = analysis.words[i - 1]
                if prev.text.lower() in ("a", "an"):
                    pos_tuple = _get_token_position(sentence, i - 1, analysis.words)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=prev.text,
                        correction="(remove)",
                        category="article-usage",
                        message=f"'{word.text}' is uncountable and should not take 'a/an'.",
                        confidence=0.90,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))
            continue

        has_determiner = False
        for other in analysis.words:
            if other.head_text == word.text and other.dep in ("det", "poss"):
                has_determiner = True
                break
        if word.dep in ("attr",) and analysis.words:
            root = [w for w in analysis.words if w.dep == "ROOT"]
            if root and root[0].dep == "attr":
                has_determiner = False

        if not has_determiner:
            if word.dep in ("dobj", "pobj", "attr", "nsubj", "nsubjpass", "conj"):
                prev_idx = None
                for j, w in enumerate(analysis.words):
                    if w.text == word.text:
                        prev_idx = j
                        break
                if prev_idx is not None and prev_idx > 0:
                    prev = analysis.words[prev_idx - 1]
                    if prev.text.lower() in INDEFINITE_ARTICLES or prev.pos in ("DET",):
                        continue

                needs_article = True
                for other in analysis.words:
                    if other.head_text == word.text:
                        if other.dep == "nummod":
                            needs_article = False
                            break
                        if other.dep == "quantmod":
                            needs_article = False
                            break

                COMMON_NO_ARTICLE_NOUNS = {
                    'dinner', 'lunch', 'breakfast', 'school', 'home', 'bed', 'work',
                    'church', 'prison', 'hospital', 'college', 'university', 'town',
                    'class', 'court', 'army', 'navy', 'air', 'water', 'food',
                    'music', 'art', 'love', 'hate', 'life', 'death', 'nature',
                    'history', 'science', 'math', 'english', 'football', 'soccer',
                    'tennis', 'basketball', 'baseball', 'golf', 'cricket',
                    'travel', 'business', 'health', 'freedom', 'peace', 'war',
                    'god', 'religion', 'philosophy', 'beauty', 'truth',
                }

                if needs_article and word.text.lower() not in COMMON_NO_ARTICLE_NOUNS:
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=word.text,
                        correction=f"a {word.text}",
                        category="article-usage",
                        message=f"Count noun '{word.text}' may need a determiner or article.",
                        confidence=0.50,
                        position=pos_tuple[0],
                        severity="warning",
                        word_start=pos_tuple[0],
                        word_end=pos_tuple[1],
                    ))

    return issues


def _check_negation_patterns(analysis: SentenceAnalysis) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    sentence = analysis.text

    has_neg = any(w.dep == "neg" or w.text.lower() in ("n't", "not") for w in analysis.words)
    do_aux = None
    if has_neg:
        for word in analysis.words:
            if word.text.lower() == "do" and word.pos == "AUX":
                neg_child = None
                for w in analysis.words:
                    if w.head_text == word.text and w.dep == "neg":
                        neg_child = w
                        break
                if neg_child:
                    do_aux = word
                    break
        if not do_aux:
            for word in analysis.words:
                if word.text.lower() == "do" and word.pos == "AUX":
                    root_verb = None
                    for w in analysis.words:
                        if w.dep == "ROOT" and w.pos in ("VERB",):
                            root_verb = w
                            break
                    if root_verb:
                        for w in analysis.words:
                            if w.head_text == root_verb.text and w.dep == "neg":
                                do_aux = word
                                break
                    if do_aux:
                        break

    if do_aux:
        main_verb = None
        for w in analysis.words:
            if w.dep == "ROOT" and w.pos in ("VERB",):
                main_verb = w
                break
        if main_verb:
            subject = get_subject_of_verb(analysis.words, main_verb)
            if subject:
                subj_lower = subject.text.lower()
                if subj_lower in ("he", "she", "it") or is_third_person_singular(subject.text):
                    pos_tuple = _get_position_in_sentence(sentence, do_aux.text)
                    neg_token = None
                    for w in analysis.words:
                        if w.dep == "neg" and w.head_text == main_verb.text:
                            neg_token = w
                            break
                    if neg_token:
                        neg_pos = _get_position_in_sentence(sentence, neg_token.text)
                        word_end = neg_pos[1]
                    else:
                        word_end = pos_tuple[1]
                    combined_incorrect = sentence[pos_tuple[0]:word_end]
                    issues.append(GrammarIssue(
                        sentence=sentence,
                        incorrect=combined_incorrect,
                        correction="doesn't",
                        category="negation-pattern",
                        message=f"The subject '{subject.text}' is third person singular. Use 'doesn't', not 'don't'.",
                        confidence=0.95,
                        position=pos_tuple[0],
                        severity="error",
                        word_start=pos_tuple[0],
                        word_end=word_end,
                    ))

    for word in analysis.words:
        word_lower = word.text.lower()

        if word_lower in ("don't", "do not"):
            subject = get_subject_of_verb(analysis.words, word)
            if subject:
                subj_lower = subject.text.lower()
                if subj_lower in ("he", "she", "it") or is_third_person_singular(subject.text):
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence, incorrect=word.text, correction="doesn't",
                        category="negation-pattern",
                        message=f"The subject '{subject.text}' is third person singular. Use 'doesn't'.",
                        confidence=0.95, position=pos_tuple[0], severity="error",
                        word_start=pos_tuple[0], word_end=pos_tuple[1],
                    ))

        if word_lower in ("doesn't", "does not"):
            subject = get_subject_of_verb(analysis.words, word)
            if subject:
                subj_lower = subject.text.lower()
                if subj_lower in ("i", "you", "we", "they"):
                    pos_tuple = _get_position_in_sentence(sentence, word.text)
                    issues.append(GrammarIssue(
                        sentence=sentence, incorrect=word.text, correction="don't",
                        category="negation-pattern",
                        message=f"The subject '{subject.text}' requires 'don't'.",
                        confidence=0.95, position=pos_tuple[0], severity="error",
                        word_start=pos_tuple[0], word_end=pos_tuple[1],
                    ))

        if word_lower in ("didn't", "did not"):
            for child in analysis.words:
                if child.head_text == word.text and child.dep in ("ROOT", "xcomp", "ccomp"):
                    child_lower = child.text.lower()
                    if child_lower in PAST_TO_BASE:
                        base = PAST_TO_BASE[child_lower]
                        pos_tuple = _get_position_in_sentence(sentence, child.text)
                        issues.append(GrammarIssue(
                            sentence=sentence, incorrect=child.text, correction=base,
                            category="negation-pattern",
                            message=f"After 'didn't', use base form '{base}' instead of past tense '{child_lower}'.",
                            confidence=0.95, position=pos_tuple[0], severity="error",
                            word_start=pos_tuple[0], word_end=pos_tuple[1],
                        ))
                    break

    return issues


def _check_semantic_context(analysis: SentenceAnalysis) -> List[GrammarIssue]:
    issues: List[GrammarIssue] = []
    sentence = analysis.text

    for word in analysis.words:
        if word.pos != "PRON":
            continue
        word_lower = word.text.lower()

        if word_lower in MALE_PRONOUNS or word_lower in FEMALE_PRONOUNS:
            verb_head = None
            for w in analysis.words:
                if w.head_text == word.text and w.pos in ("VERB", "AUX"):
                    verb_head = w
                    break
            if verb_head is None:
                for w in analysis.words:
                    if w.dep in ("nsubj", "nsubjpass") and w.head_text in [v.text for v in analysis.words if v.dep == "ROOT"]:
                        if w.text.lower() == word_lower:
                            verb_head = [v for v in analysis.words if v.dep == "ROOT"][0] if [v for v in analysis.words if v.dep == "ROOT"] else None
                            break

            if verb_head:
                for child in analysis.words:
                    if child.head_text == verb_head.text and child.dep in ("attr", "appos"):
                        child_lower = child.text.lower()
                        if word_lower in MALE_PRONOUNS:
                            if child_lower in GIRL_NOUNS or child_lower in FEMALE_PRONOUNS:
                                pos_tuple = _get_position_in_sentence(sentence, word.text)
                                issues.append(GrammarIssue(
                                    sentence=sentence, incorrect=word.text, correction="she",
                                    category="semantic-mismatch",
                                    message=f"Gendered noun '{child.text}' does not match masculine pronoun '{word.text}'. Use 'she'.",
                                    confidence=0.85, position=pos_tuple[0], severity="warning",
                                    word_start=pos_tuple[0], word_end=pos_tuple[1],
                                ))
                        elif word_lower in FEMALE_PRONOUNS:
                            if child_lower in BOY_NOUNS or child_lower in MALE_PRONOUNS:
                                pos_tuple = _get_position_in_sentence(sentence, word.text)
                                issues.append(GrammarIssue(
                                    sentence=sentence, incorrect=word.text, correction="he",
                                    category="semantic-mismatch",
                                    message=f"Gendered noun '{child.text}' does not match feminine pronoun '{word.text}'. Use 'he'.",
                                    confidence=0.85, position=pos_tuple[0], severity="warning",
                                    word_start=pos_tuple[0], word_end=pos_tuple[1],
                                ))

    for word in analysis.words:
        word_lower = word.text.lower()
        if word_lower == "there" and word.dep in ("expl", "advmod"):
            subject = get_subject_of_verb(analysis.words, word)
            if subject is None:
                root = [w for w in analysis.words if w.dep == "ROOT"]
                if root:
                    for child in analysis.words:
                        if child.head_text == root[0].text and child.dep in ("attr", "nsubj"):
                            subject = child
                            break
            if subject and subject.pos in ("NOUN", "PROPN"):
                verb = None
                for w in analysis.words:
                    if w.dep == "ROOT" and w.pos in ("VERB", "AUX"):
                        verb = w
                        break
                if verb:
                    verb_lower = verb.text.lower()
                    if is_plural_subject(subject.text) and verb_lower == "is":
                        pos_tuple = _get_position_in_sentence(sentence, verb.text)
                        issues.append(GrammarIssue(
                            sentence=sentence,
                            incorrect=verb.text,
                            correction="are",
                            category="agreement",
                            message=f"'There is' with plural subject '{subject.text}' requires 'are'.",
                            confidence=0.90,
                            position=pos_tuple[0],
                            severity="error",
                            word_start=pos_tuple[0],
                            word_end=pos_tuple[1],
                        ))
                    elif not is_plural_subject(subject.text) and verb_lower == "are":
                        pos_tuple = _get_position_in_sentence(sentence, verb.text)
                        issues.append(GrammarIssue(
                            sentence=sentence,
                            incorrect=verb.text,
                            correction="is",
                            category="agreement",
                            message=f"'There are' with singular subject '{subject.text}' requires 'is'.",
                            confidence=0.90,
                            position=pos_tuple[0],
                            severity="error",
                            word_start=pos_tuple[0],
                            word_end=pos_tuple[1],
                        ))

    root_words = [w for w in analysis.words if w.dep == "ROOT"]
    for root in root_words:
        for child in analysis.words:
            if child.head_text == root.text and child.dep == "nsubj":
                if child.pos == "PRON" and child.text.lower() not in ("there",):
                    for other in analysis.words:
                        if other.head_text == root.text and other.dep == "nsubj" and other != child:
                            if other.pos in ("NOUN", "PROPN"):
                                pos_tuple = _get_position_in_sentence(sentence, child.text)
                                issues.append(GrammarIssue(
                                    sentence=sentence,
                                    incorrect=child.text,
                                    correction="(remove)",
                                    category="double-subject",
                                    message=f"Double subject detected: '{other.text}' and '{child.text}'. Remove the pronoun.",
                                    confidence=0.80,
                                    position=pos_tuple[0],
                                    severity="warning",
                                    word_start=pos_tuple[0],
                                    word_end=pos_tuple[1],
                                ))

    return issues


def analyze_sentence_grammar(text: str) -> List[GrammarIssue]:
    analysis = analyze_sentence(text)
    all_issues: List[GrammarIssue] = []

    all_issues.extend(_check_subject_verb_agreement(analysis))
    all_issues.extend(_check_tense_consistency(analysis))
    all_issues.extend(_check_pronoun_case(analysis))
    all_issues.extend(_check_double_negation(analysis))
    all_issues.extend(_check_article_usage(analysis))
    all_issues.extend(_check_negation_patterns(analysis))
    all_issues.extend(_check_semantic_context(analysis))

    seen = set()
    unique_issues: List[GrammarIssue] = []
    for issue in all_issues:
        key = (issue.incorrect, issue.correction, issue.category)
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    return unique_issues


def analyze_document_grammar(text: str) -> List[GrammarIssue]:
    all_issues: List[GrammarIssue] = []
    sentence_analyses = analyze_document(text)

    for sentence_analysis in sentence_analyses:
        issues = analyze_sentence_grammar(sentence_analysis.text)
        all_issues.extend(issues)

    return all_issues
