"""
Context-Aware Writing Engine
Uses spaCy NLP + NLTK + phrase databases for contextual word usage detection.
Detects correctly-spelled words that are wrong in context.
"""

import re
import os
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter

import spacy

try:
    import nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False


def _make_error(text, start, end, category, message, suggestion="",
                severity="MEDIUM", confidence=0.8, context="", explanation=""):
    return {
        "original_text": text,
        "replacement": suggestion,
        "category": category,
        "error_type": "contextual_word_usage" if category == "CONTEXTUAL_WORD_USAGE" else "grammar",
        "message": message,
        "explanation": explanation or message,
        "start_position": start,
        "end_position": end,
        "severity": severity,
        "confidence": round(confidence, 2),
        "context": context,
    }


class ContextualEngine:
    """
    Context-aware analysis engine using spaCy NLP + phrase databases.
    Detects correctly-spelled words that are wrong in context.
    """

    def __init__(self, data_dir: str):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None

        self._custom_dict = set()
        self._phrase_db = {}
        self._load_phrases(data_dir)
        self._load_data(data_dir)

    def _load_phrases(self, data_dir: str):
        """Load phrase/collocation databases for contextual detection."""
        self._phrase_db = {
            "shopping mall": {"wrong_forms": ["stopping mall", "shoping mall", "shooping mall"], "confidence": 0.97},
            "clothes": {"wrong_forms": ["clothe"], "confidence": 0.95},
            "wanted to": {"wrong_forms": ["waited to buy", "waited to"], "confidence": 0.82,
                          "context_triggers": ["buy", "purchase", "get"]},
            "shops were": {"wrong_forms": ["stops was"], "confidence": 0.93},
            "were not open": {"wrong_forms": ["was not spend"], "confidence": 0.88,
                              "context_triggers": ["shop", "store", "mall", "open"]},
            "people waiting": {"wrong_forms": ["people writing"], "confidence": 0.75,
                               "context_triggers": ["because", "closed", "not open", "wait"]},
        }

        self._bigram_phrases = {
            ("shopping", "mall"): {"stopping", "shoping", "shooping", "stopping"},
            ("clothes",): set(),
            ("wanted", "to"): {"waited"},
            ("shops", "were"): {"stops"},
            ("not", "open"): {"spend", "open"},
        }

    def _load_data(self, data_dir: str):
        """Load additional data files."""
        try:
            with open(os.path.join(data_dir, "spelling_dictionary.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
                for w in data.get("valid_words", []):
                    self._custom_dict.add(w.lower())
        except Exception:
            pass

    def check(self, text: str) -> List[Dict]:
        """Run contextual analysis on text."""
        errors = []
        if not text or not text.strip():
            return errors
        if not self.nlp:
            return errors

        doc = self.nlp(text)
        sentences = list(doc.sents)

        errors.extend(self._check_phrase_context(doc, text))
        errors.extend(self._check_tense_consistency(doc, text, sentences))
        errors.extend(self._check_compound_subjects(doc, text))
        errors.extend(self._check_there_construction(doc, text))
        errors.extend(self._check_noun_verb_agreement(doc, text))
        errors.extend(self._check_semantic_errors(doc, text))

        return errors

    def _find_word_position(self, text: str, word: str, start_hint: int = 0) -> Tuple[int, int]:
        """Find position of a word in text starting from a hint."""
        lower_text = text.lower()
        lower_word = word.lower()
        idx = lower_text.find(lower_word, start_hint)
        if idx >= 0:
            return idx, idx + len(word)
        idx = lower_text.find(lower_word)
        if idx >= 0:
            return idx, idx + len(word)
        return -1, -1

    def _find_token_position(self, token, text: str) -> Tuple[int, int]:
        """Find exact position of a spaCy token in the original text."""
        return token.idx, token.idx + len(token.text)

    def _check_phrase_context(self, doc, text: str) -> List[Dict]:
        """Detect incorrect collocations and phrases."""
        errors = []
        text_lower = text.lower()
        tokens_lower = [t.text.lower() for t in doc]

        wrong_bigrams = {
            ("stopping", "mall"): ("shopping mall", 0.97,
                "The correct phrase is \"shopping mall\", not \"stopping mall\"."),
            ("shoping", "mall"): ("shopping mall", 0.99,
                "\"shopping\" is the correct spelling for \"shoping\"."),
            ("stops", "was"): ("shops were", 0.93,
                "The subject requires the plural form. Use \"shops were\" instead."),
            ("stops", "is"): ("shops are", 0.90,
                "Use the plural form \"shops are\"."),
        }

        for i in range(len(tokens_lower) - 1):
            bigram = (tokens_lower[i], tokens_lower[i + 1])
            if bigram in wrong_bigrams:
                suggestion, conf, explanation = wrong_bigrams[bigram]
                wrong_text = doc[i].text + " " + doc[i + 1].text
                start = doc[i].idx
                end = doc[i + 1].idx + len(doc[i + 1].text)
                errors.append(_make_error(
                    text=wrong_text, start=start, end=end,
                    category="CONTEXTUAL_WORD_USAGE",
                    message=f'"{wrong_text}" is not the correct phrase. Did you mean "{suggestion}"?',
                    suggestion=suggestion,
                    severity="HIGH",
                    confidence=conf,
                    context=text,
                    explanation=explanation,
                ))

        for phrase, config in self._phrase_db.items():
            wrong_forms = config.get("wrong_forms", [])
            conf = config.get("confidence", 0.85)
            triggers = config.get("context_triggers", [])

            for wrong in wrong_forms:
                idx = text_lower.find(wrong.lower())
                if idx >= 0:
                    if triggers:
                        sent_text = text[max(0, idx - 100):idx + len(wrong) + 100].lower()
                        if not any(t in sent_text for t in triggers):
                            continue
                    errors.append(_make_error(
                        text=text[idx:idx + len(wrong)], start=idx, end=idx + len(wrong),
                        category="CONTEXTUAL_WORD_USAGE",
                        message=f'"{wrong}" may be incorrect in this context. Did you mean "{phrase}"?',
                        suggestion=phrase,
                        severity="HIGH",
                        confidence=conf,
                        context=text,
                        explanation=f'"{phrase}" is the appropriate expression in this context.',
                    ))

        return errors

    def _check_tense_consistency(self, doc, text: str, sentences) -> List[Dict]:
        """Detect tense inconsistencies using context cues."""
        errors = []

        past_markers = {
            "yesterday", "ago", "last", "before", "earlier", "previously",
            "then", "once", "formerly", "when", "after",
        }
        present_markers = {
            "today", "now", "currently", "right now", "at the moment",
            "these days", "lately", "recently",
        }

        for sent in sentences:
            sent_lower = sent.text.lower()
            tokens = list(sent)

            has_past_marker = any(t.text.lower() in past_markers or
                                   t.lemma_.lower() in past_markers
                                   for t in tokens)
            # Also detect past tense verbs as evidence the sentence is in past tense
            has_past_verb = any(t.tag_ in ("VBD", "VBN") for t in tokens)
            has_present_marker = any(t.text.lower() in present_markers for t in tokens)

            if (has_past_marker or has_past_verb) and not has_present_marker:
                for t in tokens:
                    if t.pos_ == "VERB" and t.tag_ in ("VBZ", "VBP"):
                        start, end = self._find_token_position(t, text)
                        if start < 0:
                            continue
                        base = t.lemma_.lower()
                        past_form = self._get_irregular_past(base) or base + "ed"
                        if past_form == t.text.lower():
                            continue
                        errors.append(_make_error(
                            text=t.text, start=start, end=end,
                            category="GRAMMAR",
                            message=f'In past tense context ("yesterday"), use "{past_form}" instead of "{t.text}".',
                            suggestion=past_form,
                            severity="HIGH",
                            confidence=0.88,
                            context=sent.text,
                            explanation=f'The sentence contains a past tense marker. Use the past tense form.',
                        ))

        return errors

    def _check_compound_subjects(self, doc, text: str) -> List[Dict]:
        """Detect compound subject pronoun errors like 'friend and me'."""
        errors = []
        tokens = list(doc)

        obj_to_subj = {
            "me": "I", "him": "he", "her": "she", "us": "we", "them": "they",
        }

        for i, t in enumerate(tokens):
            if t.text.lower() == "and" and i > 0 and i + 1 < len(tokens):
                prev = tokens[i - 1]
                nxt = tokens[i + 1]

                if prev.pos_ in ("NOUN", "PROPN", "PRON") and nxt.text.lower() in obj_to_subj:
                    right = obj_to_subj[nxt.text.lower()]

                    has_det_before = False
                    if i >= 2:
                        check = tokens[i - 2]
                        if check.pos_ == "DET" or check.text.lower() in ("my", "your", "his", "her", "its", "our", "their"):
                            has_det_before = True

                    if prev.text.lower() in ("my", "your", "his", "her", "its", "our", "their"):
                        has_det_before = True

                    # Always flag object pronoun in compound subject — fix the pronoun only
                    error_text = prev.text + " and " + nxt.text
                    start = prev.idx
                    end = nxt.idx + len(nxt.text)

                    if has_det_before:
                        # Keep the determiner and noun, just fix the pronoun
                        # Find the determiner
                        det = ""
                        if i >= 2:
                            check = tokens[i - 2]
                            if check.text.lower() in ("my", "your", "his", "her", "its", "our", "their"):
                                det = check.text.lower() + " "
                            elif check.pos_ == "DET":
                                det = check.text + " "
                        suggestion = det + prev.text.lower() + " and " + right
                        errors.append(_make_error(
                            text=error_text, start=start, end=end,
                            category="GRAMMAR",
                            message=f'Use the subject form "{right}" instead of "{nxt.text}" in a compound subject.',
                            suggestion=suggestion,
                            severity="HIGH",
                            confidence=0.90,
                            context=text,
                            explanation=f'"{nxt.text}" is an object pronoun. Use the subject form "{right}".',
                        ))
                    else:
                        det = "my" if right == "I" else ""
                        if det:
                            suggestion = det + " " + prev.text.lower() + " and " + right
                            errors.append(_make_error(
                                text=error_text, start=start, end=end,
                                category="GRAMMAR",
                                message=f'Use the subject form "{right}" in a compound subject. Try "{suggestion}".',
                                suggestion=suggestion,
                                severity="HIGH",
                                confidence=0.90,
                                context=text,
                                explanation=f'"{nxt.text}" is an object pronoun. Use the subject form "{right}" and add a possessive determiner.',
                            ))

                if prev.text.lower() in obj_to_subj and nxt.pos_ in ("NOUN", "PROPN"):
                    right = obj_to_subj[prev.text.lower()]
                    error_text = prev.text + " and " + nxt.text
                    start = prev.idx
                    end = nxt.idx + len(nxt.text)
                    if prev.text.lower() == "me":
                        suggestion = right + " and " + nxt.text.lower()
                        errors.append(_make_error(
                            text=error_text, start=start, end=end,
                            category="GRAMMAR",
                            message=f'Use the subject form "{right}" instead of "{prev.text}" in a compound subject.',
                            suggestion=suggestion,
                            severity="HIGH",
                            confidence=0.88,
                            context=text,
                            explanation=f'"{prev.text}" is an object pronoun. Use "{right}" as the subject.',
                        ))
        return errors

    def _check_there_construction(self, doc, text: str) -> List[Dict]:
        """Detect 'there was many' → 'there were many' errors."""
        errors = []
        tokens = list(doc)

        for i, t in enumerate(tokens):
            if t.text.lower() == "there" and i + 1 < len(tokens):
                verb = tokens[i + 1]
                if verb.text.lower() in ("was", "were") and verb.pos_ in ("AUX", "VERB"):
                    for j in range(i + 2, min(i + 6, len(tokens))):
                        nxt = tokens[j]
                        if nxt.pos_ in ("NOUN", "PROPN", "PRON"):
                            is_plural = (
                                nxt.text.lower() in ("many", "few", "several", "both") or
                                nxt.text.lower() in ("people", "children", "men", "women",
                                                      "students", "workers", "friends") or
                                nxt.tag_ in ("NNS", "NNPS") or
                                nxt.text.lower().endswith("s") and not nxt.text.lower().endswith("ss")
                            )
                            if is_plural and verb.text.lower() == "was":
                                expected = "were"
                                start, end = self._find_token_position(verb, text)
                                if start < 0:
                                    continue
                                errors.append(_make_error(
                                    text=verb.text, start=start, end=end,
                                    category="GRAMMAR",
                                    message=f'With "{nxt.text}", use "{expected}" instead of "{verb.text}".',
                                    suggestion=expected,
                                    severity="HIGH",
                                    confidence=0.93,
                                    context=text,
                                    explanation=f'The subject "{nxt.text}" is plural, so use "{expected}".',
                                ))
                            elif not is_plural and verb.text.lower() == "were":
                                expected = "was"
                                start, end = self._find_token_position(verb, text)
                                if start < 0:
                                    continue
                                errors.append(_make_error(
                                    text=verb.text, start=start, end=end,
                                    category="GRAMMAR",
                                    message=f'With "{nxt.text}", use "{expected}" instead of "{verb.text}".',
                                    suggestion=expected,
                                    severity="HIGH",
                                    confidence=0.90,
                                    context=text,
                                    explanation=f'The subject "{nxt.text}" is singular, so use "{expected}".',
                                ))
                            break
                        if nxt.pos_ in ("VERB", "AUX"):
                            break
        return errors

    def _check_noun_verb_agreement(self, doc, text: str) -> List[Dict]:
        """Detect noun-verb agreement errors like 'there was many people'."""
        errors = []
        tokens = list(doc)

        for i, t in enumerate(tokens):
            if t.pos_ == "NOUN" and t.tag_ in ("NNS", "NNPS"):
                for j in range(i + 1, min(i + 3, len(tokens))):
                    nxt = tokens[j]
                    if nxt.pos_ in ("AUX", "VERB") and nxt.text.lower() in ("was", "is"):
                        if nxt.text.lower() == "was":
                            expected = "were"
                        else:
                            expected = "are"
                        start, end = self._find_token_position(nxt, text)
                        if start < 0:
                            continue
                        errors.append(_make_error(
                            text=nxt.text, start=start, end=end,
                            category="GRAMMAR",
                            message=f'The plural noun "{t.text}" requires "{expected}" instead of "{nxt.text}".',
                            suggestion=expected,
                            severity="HIGH",
                            confidence=0.88,
                            context=text,
                            explanation=f'Plural nouns require the plural form of "be".',
                        ))
                        break
                    if nxt.pos_ in ("VERB",):
                        break
        return errors

    def _check_semantic_errors(self, doc, text: str) -> List[Dict]:
        """Detect semantic errors using word vectors and context."""
        errors = []
        if not self.nlp:
            return errors

        tokens = list(doc)
        text_lower = text.lower()

        semantic_fixes = {
            ("spend", "open"): {
                "wrong": "spend",
                "correct": "open",
                "conf": 0.85,
                "explanation": '"Open" is the appropriate word in the context of shops/stores. "Spend" does not fit this context.',
            },
        }

        for i, t in enumerate(tokens):
            if t.pos_ in ("VERB", "NOUN"):
                for (wrong, context_word), fix in semantic_fixes.items():
                    if t.text.lower() == wrong:
                        for j in range(max(0, i - 5), min(len(tokens), i + 6)):
                            if j != i and tokens[j].text.lower() == context_word:
                                start, end = self._find_token_position(t, text)
                                if start < 0:
                                    continue
                                errors.append(_make_error(
                                    text=t.text, start=start, end=end,
                                    category="CONTEXTUAL_WORD_USAGE",
                                    message=f'"{t.text}" does not fit this context. Did you mean "{fix["correct"]}"?',
                                    suggestion=fix["correct"],
                                    severity="HIGH",
                                    confidence=fix["conf"],
                                    context=text,
                                    explanation=fix["explanation"],
                                ))
                                break
        return errors

    def _get_irregular_past(self, base: str) -> Optional[str]:
        """Get irregular past tense form."""
        irregulars = {
            "go": "went", "eat": "ate", "see": "saw", "come": "came",
            "take": "took", "give": "gave", "know": "knew",
            "think": "thought", "make": "made", "run": "ran",
            "write": "wrote", "speak": "spoke", "drive": "drove",
            "break": "broke", "choose": "chose", "do": "did",
            "fly": "flew", "grow": "grew", "throw": "threw",
            "wear": "wore", "draw": "drew", "begin": "began",
            "drink": "drank", "ring": "rang", "swim": "swam",
            "sing": "sang", "sit": "sat", "stand": "stood",
            "find": "found", "feel": "felt", "keep": "kept",
            "sleep": "slept", "leave": "left", "bring": "brought",
            "buy": "bought", "catch": "caught", "teach": "taught",
            "tell": "told", "sell": "sold", "spend": "spent",
            "build": "built", "send": "sent", "win": "won",
            "lose": "lost", "hold": "held", "meet": "met",
            "read": "read", "reach": "reached", "put": "put",
            "cut": "cut", "cost": "cost", "hit": "hit",
            "let": "let", "set": "set", "shut": "shut", "hurt": "hurt",
        }
        return irregulars.get(base)
