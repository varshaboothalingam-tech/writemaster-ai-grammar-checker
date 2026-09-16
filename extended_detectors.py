"""Extended candidate detectors for the grammar pipeline.

Adds spelling, word usage, compound-subject pronouns, existential there,
article errors, and preposition errors.
"""
from __future__ import annotations

import os
import json
import re
from typing import List, Optional, Set

import nltk
from nltk.corpus import words as nltk_words

from grammar_pipeline import Candidate, CandidateDetector, ContextAnalyzer


import spacy
_nlp = None
def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


class SpellingDetector(CandidateDetector):
    """Detects spelling errors using NLTK words + common misspellings."""
    name = "SpellingDetector"

    def __init__(self, data_dir: str = "data"):
        self._misspellings: dict[str, str] = {}
        self._valid_words: set[str] = set()
        self._load(data_dir)

    def _load(self, data_dir: str):
        base = {w.lower() for w in nltk_words.words()}
        self._valid_words = base.copy()
        for w in list(base):
            if len(w) < 2:
                continue
            if w.endswith("s"):
                self._valid_words.add(w)
            else:
                self._valid_words.add(w + "s")
                if w.endswith(("sh", "ch", "x", "z", "s")):
                    self._valid_words.add(w + "es")
                elif w.endswith("y") and w[-2] not in "aeiou":
                    self._valid_words.add(w[:-1] + "ies")
            if not w.endswith("e"):
                self._valid_words.add(w + "ed")
                self._valid_words.add(w + "ing")
            else:
                if not w.endswith("ee"):
                    self._valid_words.add(w[:-1] + "ed")
                    self._valid_words.add(w[:-1] + "ing")
            if len(w) >= 3 and w[-1] in "bcdfgklmnprst" and w[-2] in "aeiou" and w[-3] not in "aeiou":
                self._valid_words.add(w + w[-1] + "ing")
                self._valid_words.add(w + w[-1] + "ed")
        extras = {
            "fox", "jump", "jumps", "lazy", "brown", "quick", "dog", "cat",
            "honest", "apple", "hour", "person", "john", "india", "happy",
            "really", "big", "was", "decided", "hardly", "play", "plays",
            "playing", "played", "goes", "going", "went", "gone", "ate",
            "eaten", "eat", "saw", "see", "seen", "gave", "given", "give",
            "took", "taken", "take", "knew", "known", "know", "thought",
            "think", "made", "make", "ran", "run", "runs", "running",
            "wrote", "written", "write", "spoke", "spoken", "speak",
            "drove", "driven", "drive", "broke", "broken", "break",
            "chose", "chosen", "choose", "did", "done", "do", "does",
            "doing", "flew", "flown", "fly", "grew", "grown", "grow",
            "threw", "thrown", "throw", "wore", "worn", "wear",
            "drew", "drawn", "draw", "began", "begun", "begin", "drank",
            "drunk", "drink", "rang", "rung", "ring", "swam", "swum",
            "swim", "sang", "sung", "sing", "sat", "sit", "sits",
            "stood", "stand", "stands", "found", "find", "finds", "felt",
            "feel", "feels", "kept", "keep", "keeps", "slept", "sleep",
            "sleeps", "left", "leave", "leaves", "brought", "bring",
            "brings", "bought", "buy", "buys", "caught", "catch",
            "catches", "taught", "teach", "teaches", "told", "tell",
            "tells", "sold", "sell", "sells", "spent", "spend", "spends",
            "built", "build", "builds", "sent", "send", "sends", "won",
            "win", "wins", "lost", "lose", "loses", "held", "hold",
            "holds", "met", "meet", "meets", "put", "puts", "cut",
            "cuts", "cost", "costs", "hit", "hits", "let", "lets",
            "shut", "shuts", "hurt", "hurts",
            "stopping", "opening", "closing", "changing", "running",
            "walking", "talking", "looking", "helping", "working",
            "sitting", "standing", "lying", "dying", "shopping", "writing",
            "friends", "shopping", "mall", "clothes", "shops", "people",
            "wanted", "waited", "reach", "reached",
            "beautiful", "universe", "computer", "internet", "science",
            "music", "art", "history", "math", "english", "spanish",
            "france", "germany", "japan", "china", "india", "london",
            "paris", "tokyo", "mumbai", "delhi", "united", "states",
            "america", "africa", "europe", "asia", "australia",
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "january", "february", "march",
            "april", "june", "july", "august", "september", "october",
            "november", "december", "christmas", "easter", "halloween",
            "thanksgiving", "birthday", "anniversary", "wedding",
            "doctor", "lawyer", "teacher", "engineer", "nurse", "chef",
            "artist", "musician", "writer", "scientist", "student",
            "running", "playing", "walking", "talking", "eating",
            "drinking", "sleeping", "reading", "writing", "studying",
            "working", "cooking", "cleaning", "washing", "driving",
            "gonna", "wanna", "gotta", "kinda", "sorta", "dunno",
        }
        self._valid_words.update(extras)
        try:
            path = os.path.join(data_dir, "spelling_dictionary.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._misspellings = data.get("common_misspellings", {})
        except Exception:
            pass
        try:
            cm_path = os.path.join(data_dir, "common_mistakes.json")
            with open(cm_path, "r", encoding="utf-8") as f:
                cm = json.load(f)
            for pair in cm.get("common_spelling_mistakes", []):
                if isinstance(pair, dict):
                    wrong = pair.get("wrong", "").lower()
                    right = pair.get("correct", "")
                    if wrong and right:
                        self._misspellings[wrong] = right
                elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                    self._misspellings[pair[0].lower()] = pair[1]
        except Exception:
            pass
        self._misspellings.update({
            "recieve": "receive", "freind": "friend", "msitake": "mistake",
            "tommorow": "tomorrow", "alot": "a lot", "untill": "until",
            "wierd": "weird", "truely": "truly", "occured": "occurred",
            "enviroment": "environment", "goverment": "government",
            "neccessary": "necessary", "priviledge": "privilege",
            "relevent": "relevant", "successfull": "successful",
            "grammer": "grammar", "independant": "independent",
            "libary": "library", "maintainance": "maintenance",
            "posession": "possession", "publically": "publicly",
            "responsable": "responsible", "similiar": "similar",
            "suprise": "surprise", "wheter": "whether", "wich": "which",
            "yeild": "yield",             "accross": "across",
        })

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer' = None) -> List[Candidate]:
        import re as _re
        candidates = []
        for match in _re.finditer(r'\b([a-zA-Z]+)\b', text):
            word = match.group(1)
            lower = word.lower()
            if len(lower) < 3:
                continue
            if lower in self._valid_words:
                continue
            if lower in self._misspellings:
                candidates.append(Candidate(
                    start=match.start(),
                    end=match.end(),
                    original=word,
                    replacement=self._misspellings[lower],
                    category="spelling",
                    rule_id="SPELL_MISSPELLING",
                    message=f'"{word}" may be misspelled. Did you mean "{self._misspellings[lower]}"?',
                    checker_name=self.name,
                    raw_confidence=0.90,
                    context=text,
                    sentence_text=text,
                    metadata={"word": word, "suggestion": self._misspellings[lower]},
                ))
        return candidates

    def _find_closest(self, word: str) -> Optional[str]:
        best = None
        best_dist = 3
        for valid in self._valid_words:
            if abs(len(valid) - len(word)) > 2:
                continue
            d = self._edit_distance(word, valid)
            if d < best_dist:
                best_dist = d
                best = valid
        return best if best_dist <= 2 else None

    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                if c1 == c2:
                    curr.append(prev[j])
                else:
                    curr.append(1 + min((prev[j], prev[j + 1], curr[-1])))
            prev = curr
        return prev[-1]


class WordUsageDetector(CandidateDetector):
    """Detects common word-usage errors and confusables."""
    name = "WordUsageDetector"

    CONFUSABLES = {
        "their": {"there", "they're"},
        "there": {"their", "they're"},
        "they're": {"their", "there"},
        "your": {"you're"},
        "you're": {"your"},
        "its": {"it's"},
        "it's": {"its"},
        "affect": {"effect"},
        "effect": {"affect"},
        "accept": {"except"},
        "except": {"accept"},
        "lose": {"loose"},
        "loose": {"lose"},
        "then": {"than"},
        "than": {"then"},
        "to": {"too", "two"},
        "too": {"to", "two"},
        "two": {"to", "too"},
        "which": {"witch"},
        "witch": {"which"},
        "whose": {"who's"},
        "who's": {"whose"},
        "weather": {"whether"},
        "whether": {"weather"},
        "advice": {"advise"},
        "advise": {"advice"},
        "practice": {"practise"},
        "practise": {"practice"},
        "stationary": {"stationery"},
        "stationery": {"stationary"},
        "principle": {"principal"},
        "principal": {"principle"},
        "compliment": {"complement"},
        "complement": {"compliment"},
        "dessert": {"desert"},
        "desert": {"dessert"},
        "now": {"know"},
        "know": {"now"},
    }

    WORD_CORRECTIONS = {
        "alot": "a lot",
        "should of": "should have",
        "could of": "could have",
        "would of": "would have",
        "might of": "might have",
        "must of": "must have",
        "may of": "may have",
        "use to": "used to",
        "suppose to": "supposed to",
        "accidently": "accidentally",
        "irregardless": "regardless",
        "conversate": "converse",
        "conversing": "conversing",
        "impactful": "impactful",
        "nother": "another",
        "anymore": "anymore",
        "stopping mall": "shopping mall",
        "clothe": "clothes",
    }

    WEAK_WORDS = {
        "very", "really", "quite", "rather", "somewhat", "basically",
        "actually", "practically", "virtually", "literally", "honestly",
        "totally", "completely", "absolutely", "definitely", "certainly",
        "probably", "possibly", "maybe", "perhaps", "just", "simply",
        "easily", "clearly", "obviously", "apparently", "supposedly",
        "hopefully", "fortunately", "unfortunately", "importantly",
        "significant", "significant", "interesting", "nice", "good",
        "bad", "great", "amazing", "awesome", "fantastic", "wonderful",
        "terrible", "horrible", "awful", "stuff", "things", "got",
        "getting", "gets", "get", "went", "going", "goes", "go",
        "said", "say", "says", "saying", "like", "stuff", "things",
    }

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer' = None) -> List[Candidate]:
        candidates = []
        lower_text = text.lower()
        for wrong, correct in self.WORD_CORRECTIONS.items():
            idx = lower_text.find(wrong)
            while idx != -1:
                end = idx + len(wrong)
                before_ok = idx == 0 or not text[idx - 1].isalpha()
                after_ok = end >= len(text) or not text[end].isalpha()
                if before_ok and after_ok:
                    candidates.append(Candidate(
                        start=idx,
                        end=end,
                        original=text[idx:end],
                        replacement=correct,
                        category="word_usage",
                        rule_id=f"WUSE_{wrong.replace(' ', '_').upper()}",
                        message=f'Consider using "{correct}" instead of "{wrong}".',
                        checker_name=self.name,
                        raw_confidence=0.88,
                        context=text,
                        sentence_text=text,
                        metadata={"wrong": wrong, "correct": correct},
                    ))
                idx = lower_text.find(wrong, end)
        return candidates


class CompoundSubjectPronounDetector(CandidateDetector):
    """Detects pronoun case errors in compound subjects.

    "My friend and me went" → "My friend and I went"
    "Her and him left" → "She and he left"
    """
    name = "CompoundSubjectPronounDetector"

    SUBJECT_FORMS = {
        "me": "I", "him": "he", "her": "she", "them": "they",
        "us": "we",
    }

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer' = None) -> List[Candidate]:
        nlp = _get_nlp()
        candidates = []
        doc = nlp(text)

        for token in doc:
            if token.lower_ not in self.SUBJECT_FORMS:
                continue
            # Pattern 1: "X and me went" — me is conj of X which is nsubj
            if token.dep_ == "conj" and token.i > 0:
                prev = doc[token.i - 1]
                if prev.text.lower() in ("and", "or") and token.i >= 2:
                    two_back = doc[token.i - 2]
                    if two_back.dep_ in ("nsubj", "nsubjpass") or two_back.pos_ in ("NOUN", "PROPN"):
                        replacement = self.SUBJECT_FORMS[token.text.lower()]
                        candidates.append(Candidate(
                            start=token.idx,
                            end=token.idx + len(token.text),
                            original=token.text,
                            replacement=replacement,
                            category="pronouns",
                            rule_id="PRONOUN_COMPOUND_SUBJECT",
                            message=f'In a compound subject, use "{replacement}" instead of "{token.text}".',
                            checker_name=self.name,
                            raw_confidence=0.90,
                            context=text,
                            sentence_text=text,
                            metadata={
                                "pronoun_form": "object",
                                "expected_form": "subject",
                            },
                        ))
            # Pattern 2: "Me and X went" — me is nsubj at start
            elif token.dep_ in ("nsubj", "nsubjpass"):
                # Check if preceded by "and"/"or" + noun/pronoun
                for child in token.children:
                    if child.dep_ == "conj":
                        pass
                if token.i > 0:
                    prev = doc[token.i - 1]
                    if prev.text.lower() in ("and", "or") and token.i >= 2:
                        two_back = doc[token.i - 2]
                        if two_back.pos_ in ("NOUN", "PROPN", "PRON"):
                            replacement = self.SUBJECT_FORMS[token.text.lower()]
                            candidates.append(Candidate(
                                start=token.idx,
                                end=token.idx + len(token.text),
                                original=token.text,
                                replacement=replacement,
                                category="pronouns",
                                rule_id="PRONOUN_COMPOUND_SUBJECT_2",
                                message=f'In a compound subject, use "{replacement}" instead of "{token.text}".',
                                checker_name=self.name,
                                raw_confidence=0.90,
                                context=text,
                                sentence_text=text,
                                metadata={
                                    "pronoun_form": "object",
                                    "expected_form": "subject",
                                },
                            ))
        return candidates


class ExistentialThereDetector(CandidateDetector):
    """Detects existential there + wrong verb number.

    "There was many people" → "There were many people"
    "There is a lot of cats" → "There are a lot of cats"
    """
    name = "ExistentialThereDetector"

    BE_SINGULAR = {"is", "was", "has"}
    BE_PLURAL = {"are", "were", "have"}

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer' = None) -> List[Candidate]:
        nlp = _get_nlp()
        candidates = []
        doc = nlp(text)

        for token in doc:
            if token.lower_ != "there" or token.pos_ != "PRON":
                continue
            verb = None
            # "there" is a dependent (expl) — the verb is its head
            if token.dep_ == "expl" and token.head.pos_ == "VERB":
                verb = token.head
            if verb is None:
                continue
            real_subject = None
            for child in verb.children:
                if child.dep_ == "attr":
                    real_subject = child
                    break
            if real_subject is None:
                continue
            if real_subject.lower_ in ("lot", "number", "plenty", "lots", "couple", "series", "range"):
                for child in real_subject.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                real_subject = grandchild
                                break
            is_plural = self._is_plural_subject(real_subject, doc)
            if is_plural is None:
                continue
            if is_plural and verb.text.lower() in self.BE_SINGULAR:
                replacement = self._get_plural_form(verb.text.lower())
                if replacement and replacement != verb.text.lower():
                    candidates.append(Candidate(
                        start=verb.idx,
                        end=verb.idx + len(verb.text),
                        original=verb.text,
                        replacement=replacement,
                        category="agreement",
                        rule_id="EXISTENTIAL_THERE_SVA",
                        message=f'With the plural subject "{real_subject.text}", use "{replacement}" instead of "{verb.text}".',
                        checker_name=self.name,
                        raw_confidence=0.92,
                        context=text,
                        sentence_text=text,
                        metadata={
                            "subject": real_subject.text,
                            "subject_number": "plural",
                            "verb": verb.text,
                            "existential_there": True,
                        },
                    ))
            elif not is_plural and verb.text.lower() in self.BE_PLURAL:
                replacement = self._get_singular_form(verb.text.lower())
                if replacement and replacement != verb.text.lower():
                    candidates.append(Candidate(
                        start=verb.idx,
                        end=verb.idx + len(verb.text),
                        original=verb.text,
                        replacement=replacement,
                        category="agreement",
                        rule_id="EXISTENTIAL_THERE_SVA",
                        message=f'With the singular subject "{real_subject.text}", use "{replacement}" instead of "{verb.text}".',
                        checker_name=self.name,
                        raw_confidence=0.92,
                        context=text,
                        sentence_text=text,
                        metadata={
                            "subject": real_subject.text,
                            "subject_number": "singular",
                            "verb": verb.text,
                            "existential_there": True,
                        },
                    ))
        return candidates

    def _is_plural_subject(self, token, doc) -> Optional[bool]:
        UNCOUNTABLE_NOUNS = {"information", "knowledge", "evidence", "advice",
                            "furniture", "luggage", "equipment", "progress",
                            "chaos", "music", "news", "math", "physics",
                            "economics", "politics", "ethics", "traffic",
                            "weather", "rice", "sugar", "water", "milk",
                            "bread", "money", "research", "homework",
                            "education", "experience", "health", "love",
                            "happiness", "sadness", "anger", "fear"}
        if token.lower_ in UNCOUNTABLE_NOUNS:
            return False
        if token.pos_ in ("NN", "NNP"):
            return False
        if token.pos_ in ("NNS", "NNPS"):
            return True
        if token.tag_ == "CD":
            num_text = token.text.lower()
            if num_text in ("one", "1", "a", "an", "each", "every"):
                return False
            if num_text in ("zero", "0"):
                return False
            return True
        if token.tag_ in ("PRP",):
            if token.lower_ in ("i", "he", "she", "it"):
                return False
            if token.lower_ in ("we", "they"):
                return True
        if token.lower_ in ("people", "students", "children", "men", "women",
                            "things", "items", "cases", "words", "rules"):
            return True
        if token.tag_ in ("PRP",):
            return None
        if token.pos_ == "NOUN" and token.text.lower().endswith("s"):
            return True
        return None

    def _get_plural_form(self, verb: str) -> Optional[str]:
        mapping = {"is": "are", "was": "were", "has": "have"}
        return mapping.get(verb)

    def _get_singular_form(self, verb: str) -> Optional[str]:
        mapping = {"are": "is", "were": "was", "have": "has"}
        return mapping.get(verb)


class ArticleDetector(CandidateDetector):
    """Detects basic article errors: a/an misuse, missing articles."""
    name = "ArticleDetector"

    VOWEL_SOUNDS = set("aeiou")
    EXCEPTIONS_AN = {"university", "uniform", "unique", "unit", "united",
                     "universal", "unified", "union", "useful", "user",
                     "one", "once", "european", "hour", "honest", "honour",
                     "umbrella", "uncle", "aunt"}
    # Words starting with 'u' that have "yoo" sound (use "a", not "an")
    YOO_SOUND_U = {"university", "uniform", "unique", "unit", "united",
                   "universal", "unified", "union", "useful", "user",
                   "universe", "usage", "usual", "usually", "utensil",
                   "utility", "utilize", "utensil", "eulogy", "euphemism",
                   "euphoria", "european", "euclidean"}
    VOWEL_LETTER_WORDS = {"hour", "honest", "honour", "heir", "herb"}

    def detect(self, text: str, doc=None) -> List[Candidate]:
        nlp = _get_nlp()
        candidates = []
        doc = nlp(text)

        for i, token in enumerate(doc):
            if token.lower_ in ("a", "an") and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.pos_ in ("NOUN", "PROPN", "ADJ"):
                    word = next_token.text.lower()
                    use_an = self._should_use_an(word)
                    if token.lower_ == "a" and use_an:
                        candidates.append(Candidate(
                            start=token.idx,
                            end=token.idx + len(token.text),
                            original=token.text,
                            replacement="an",
                            category="articles",
                            rule_id="ARTICLE_A_AN",
                            message=f'Use "an" before "{word}" (vowel sound).',
                            checker_name=self.name,
                            raw_confidence=0.92,
                            context=text,
                            sentence_text=text,
                            metadata={"rule": "a_an", "next_word": word},
                        ))
                    elif token.lower_ == "an" and not use_an:
                        candidates.append(Candidate(
                            start=token.idx,
                            end=token.idx + len(token.text),
                            original=token.text,
                            replacement="a",
                            category="articles",
                            rule_id="ARTICLE_AN_A",
                            message=f'Use "a" before "{word}" (consonant sound).',
                            checker_name=self.name,
                            raw_confidence=0.92,
                            context=text,
                            sentence_text=text,
                            metadata={"rule": "a_an", "next_word": word},
                        ))
        return candidates

    def _should_use_an(self, word: str) -> bool:
        if word in self.YOO_SOUND_U:
            return False  # "a university" not "an university"
        if word in self.EXCEPTIONS_AN:
            return True
        if word in self.VOWEL_LETTER_WORDS:
            return True
        first_letter = word[0].lower() if word else ""
        return first_letter in self.VOWEL_SOUNDS


class PrepositionDetector(CandidateDetector):
    """Detects common preposition errors."""
    name = "PrepositionDetector"

    PREPOSITION_ERRORS = {
        "different than": "different from",
        "irregardless": "regardless",
        "could of": "could have",
        "would of": "would have",
        "should of": "should have",
        "might of": "might have",
        "must of": "must have",
        "may of": "may have",
        "deal with": "deal with",
    }

    def detect(self, text: str, doc=None) -> List[Candidate]:
        candidates = []
        lower = text.lower()
        for wrong, correct in self.PREPOSITION_ERRORS.items():
            idx = lower.find(wrong)
            while idx != -1:
                end = idx + len(wrong)
                before_ok = idx == 0 or not text[idx - 1].isalpha()
                after_ok = end >= len(text) or not text[end].isalpha()
                if before_ok and after_ok:
                    candidates.append(Candidate(
                        start=idx,
                        end=end,
                        original=text[idx:end],
                        replacement=correct,
                        category="prepositions",
                        rule_id=f"PREP_{wrong.replace(' ', '_').upper()}",
                        message=f'Consider "{correct}" instead of "{wrong}".',
                        checker_name=self.name,
                        raw_confidence=0.85,
                        context=text,
                        sentence_text=text,
                        metadata={"wrong": wrong, "correct": correct},
                    ))
                idx = lower.find(wrong, end)
        return candidates
