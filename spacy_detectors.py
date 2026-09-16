"""
spacy_detectors.py — spaCy-based candidate error detectors.
Each detector produces Candidate objects, NOT final errors.
"""

import re
from typing import List, Dict, Optional, Tuple
from grammar_pipeline import CandidateDetector, Candidate, ContextAnalyzer
from spacy_context import (
    nlp, IRREGULAR_VERBS, BE_FORMS, HAVE_FORMS, MODALS,
    COLLECTIVE_NOUNS, UNCOUNTABLE_NOUNS, SINGULAR_PRONOUNS, PLURAL_PRONOUNS
)


# ── Subject-Verb Agreement Detector ───────────────────────────────────

class SubjectVerbAgreementDetector(CandidateDetector):
    """
    Detects subject-verb agreement errors using spaCy dependency parsing.
    
    Uses actual NLP to find:
    - Subject tokens (nsubj, nsubjpass)
    - Their head verbs
    - Number mismatches
    """
    name = "sva_detector"

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer') -> List[Candidate]:
        candidates = []
        doc = nlp(text)

        for sent in doc.sents:
            sent_start = sent.start_char
            tokens = list(sent)

            # Find subject-verb pairs
            for token in tokens:
                if token.dep_ not in ("nsubj", "nsubjpass"):
                    continue

                subject = token
                verb = token.head

                # Skip if verb is not a finite verb
                if verb.pos_ not in ("VERB", "AUX"):
                    continue

                # Skip non-finite verbs (past participles, gerunds in passive chains)
                if verb.tag_ in ("VBN", "VBG"):
                    # Check if this is part of a passive/continuous chain
                    # If verb has an auxiliary (be/have), skip it — it's not the finite verb
                    has_aux = any(t.dep_ in ("aux", "auxpass") for t in verb.children)
                    if has_aux:
                        continue
                    # Also check if parent is a passive construction
                    if verb.dep_ in ("advcl", "relcl", "ccomp", "xcomp"):
                        continue

                # Skip if subject is in a subordinate clause
                if self._is_in_subordinate_clause(subject):
                    continue

                # Skip passive voice subjects
                if token.dep_ == "nsubjpass":
                    continue

                # Skip if there's an auxpass anywhere in the verb chain
                if any(t.dep_ == "auxpass" for t in verb.children):
                    continue
                # Also check parent verb for passive
                parent_verb = verb.head
                if parent_verb != verb and any(t.dep_ == "auxpass" for t in parent_verb.children):
                    continue

                # Skip if verb is after "did/does/do" (question/inversion form)
                if self._has_do_auxiliary(verb, tokens):
                    continue

                # Skip if verb is in a subordinate clause (that/what/whether/if/where/when)
                if self._is_in_that_clause(verb, tokens):
                    continue

                # Skip compound quantity expressions (five dollars is, two thirds is)
                if self._is_quantity_expression(subject, tokens):
                    continue

                # Skip if verb is part of a contraction (he'll, she's, etc.)
                if self._is_in_contraction(verb, tokens):
                    continue

                # Get subject number
                subj_number = self._get_subject_number(subject, tokens)
                if subj_number == "unknown":
                    continue

                # Get the expected verb form
                verb_form = self._get_verb_form(verb, subj_number, tokens)
                if verb_form is None:
                    continue

                # Check if current verb matches expected
                current_verb = verb.text.lower()
                if current_verb == verb_form:
                    continue  # Correct

                # Special cases: collective nouns, uncountable nouns
                if subject.lower_ in COLLECTIVE_NOUNS:
                    # Collective nouns can take singular or plural in English
                    # Only flag if it's clearly inconsistent within the same sentence
                    continue

                if subject.lower_ in UNCOUNTABLE_NOUNS:
                    # Uncountable nouns take singular verbs
                    # But "data" can be treated as plural in formal contexts
                    if current_verb in ("are", "were", "have"):
                        # Only flag if clearly wrong
                        if subject.lower_ != "data":  # "data" can be plural
                            continue

                # Generate candidate
                replacement = verb_form
                if verb.lemma_ in IRREGULAR_VERBS:
                    irregular = IRREGULAR_VERBS[verb.lemma_]
                    if subj_number == "singular" and verb.tag_ == "VBD":
                        replacement = irregular.get("past_sg", verb_form)
                    elif subj_number == "plural" and verb.tag_ == "VBD":
                        replacement = irregular.get("past_pl", verb_form)

                # Calculate confidence based on evidence strength
                confidence = self._calculate_confidence(subject, verb, subj_number, tokens)

                candidates.append(Candidate(
                    start=sent_start + verb.idx,
                    end=sent_start + verb.idx + len(verb.text),
                    original=verb.text,
                    replacement=replacement,
                    category="agreement",
                    rule_id=f"SVA_{subj_number.upper()}_VERB",
                    message=f'The subject "{subject.text}" is {subj_number}, but the verb "{verb.text}" does not agree.',
                    checker_name=self.name,
                    raw_confidence=confidence,
                    context=sent.text,
                    sentence_text=sent.text,
                    metadata={
                        "subject": subject.text,
                        "subject_number": subj_number,
                        "verb": verb.text,
                        "verb_tag": verb.tag_,
                    }
                ))

        return candidates

    def _get_subject_number(self, subject, tokens) -> str:
        """Determine the number of a subject token."""
        if subject.lower_ in SINGULAR_PRONOUNS:
            return "singular"
        if subject.lower_ in PLURAL_PRONOUNS:
            return "plural"
        if subject.tag_ in ("NN", "NNP"):
            return "singular"
        if subject.tag_ in ("NNS", "NNPS"):
            return "plural"

        # Check for compound subjects (X and Y)
        for token in tokens:
            if token.dep_ == "conj" and token.head == subject:
                if token.text.lower() == "and":
                    return "plural"
                # "X or Y" — match the closest to verb
                if token.text.lower() in ("or", "nor"):
                    return self._get_nearest_number(token, tokens)

        return "unknown"

    def _get_nearest_number(self, token, tokens) -> str:
        """For 'X or Y', get the number of Y (closest to verb)."""
        for t in tokens:
            if t.i > token.i and t.dep_ in ("nsubj", "nsubjpass"):
                return self._get_subject_number(t, tokens)
        return "unknown"

    def _is_in_subordinate_clause(self, token) -> bool:
        """Check if token is in a subordinate clause."""
        current = token
        while current.head != current:
            if current.dep_ in ("mark", "advcl", "relcl", "csubj"):
                return True
            current = current.head
        return False

    def _has_do_auxiliary(self, verb, tokens) -> bool:
        """Check if verb has do/does/did as auxiliary (question/inversion)."""
        for t in tokens:
            if t.dep_ == "aux" and t.text.lower() in ("do", "does", "did"):
                return True
        return False

    def _is_in_that_clause(self, verb, tokens) -> bool:
        """Check if verb is in a that/what/whether/if/where/when clause."""
        # Check if verb's head is a reporting verb or cognitive verb
        head = verb.head
        if head.pos_ == "VERB":
            # Check for complementizer
            for t in tokens:
                if t.dep_ == "mark" and t.i < verb.i and t.head == verb:
                    if t.lower_ in ("that", "what", "whether", "if", "where", "when"):
                        return True
        # Also check if the verb is in a subordinate clause after a reporting verb
        # "They discussed what they should do" — "discussed" is reporting verb
        # "She told me how much it cost" — "told" is reporting verb
        for t in tokens:
            if t.dep_ == "ccomp" and t.head.pos_ == "VERB":
                if verb.i > t.head.i and verb.i < t.i + 10:
                    return True
        return False

    def _is_quantity_expression(self, subject, tokens) -> bool:
        """Check if subject is part of a quantity expression (five dollars is)."""
        # "Five dollars is" — quantity as a unit
        quantity_words = {"five", "ten", "twenty", "fifty", "hundred", "thousand",
                         "million", "billion", "dozen", "couple", "pair", "two",
                         "three", "four", "six", "seven", "eight", "nine",
                         "thirds", "quarters", "half", "percent", "majority",
                         "minority", "number", "total", "amount", "lot"}
        for t in tokens:
            if t.i < subject.i and t.lower_ in quantity_words:
                return True
        # Also check if subject itself is a quantity word
        if subject.lower_ in quantity_words:
            return True
        return False

    def _is_in_contraction(self, verb, tokens) -> bool:
        """Check if verb is part of a contraction (he'll, she's, etc.)."""
        for t in tokens:
            if "'" in t.text and t.i <= verb.i:
                # Check if this contraction includes the verb
                if t.text.lower().endswith("ll") or t.text.lower().endswith("n't"):
                    return True
        return False

    def _get_verb_form(self, verb, subj_number: str, tokens) -> Optional[str]:
        """Get the expected verb form for the given subject number."""
        # Check for modals — only if the modal is a direct auxiliary of this verb
        for token in tokens:
            if token.dep_ == "aux" and token.text.lower() in MODALS:
                if token.head == verb:
                    return verb.lemma_  # Modal + base form

        # Check for auxiliaries — only if the aux is a direct auxiliary of this verb
        for token in tokens:
            if token.dep_ == "aux" and token.text.lower() in BE_FORMS:
                if token.head == verb:
                    if subj_number == "singular":
                        return "is" if verb.tag_ == "VBG" else verb.lemma_
                    return "are" if verb.tag_ == "VBG" else verb.lemma_

        # Check for semi-modals (dare, need, ought) — only if direct aux of this verb
        for token in tokens:
            if token.dep_ == "aux" and token.text.lower() in ("dare", "need", "ought"):
                if token.head == verb:
                    return verb.lemma_  # Semi-modal + base form

        # Regular verb: add -s for singular third person present
        if verb.tag_ == "VBZ":
            if subj_number == "plural":
                return verb.lemma_  # Base form for plural
            return None  # Already singular

        if verb.tag_ in ("VBP", "VB"):
            if subj_number == "singular":
                # Check if it's 3rd person
                if self._is_third_person(verb, tokens):
                    return verb.lemma_ + "s"  # Add -s
            return None

        return None

    def _is_third_person(self, verb, tokens) -> bool:
        """Check if the verb is in third person."""
        for token in tokens:
            if token.dep_ == "nsubj" and token.head == verb:
                if token.lower_ in ("i", "we", "you"):
                    return False
                return True
        return True  # Default to third person

    def _calculate_confidence(self, subject, verb, subj_number, tokens) -> float:
        """Calculate confidence based on evidence strength."""
        conf = 0.80  # Base

        # Stronger if subject is a clear pronoun
        if subject.tag_ in ("PRP",):
            conf += 0.10

        # Stronger if verb is clearly wrong form
        if verb.tag_ == "VBZ" and subj_number == "plural":
            conf += 0.05
        if verb.tag_ in ("VBP", "VB") and subj_number == "singular" and self._is_third_person(verb, tokens):
            conf += 0.05

        # Weaker if there are intervening words
        distance = abs(verb.i - subject.i)
        if distance > 5:
            conf -= 0.10

        # Weaker if there's a compound subject
        has_conj = any(t.dep_ == "conj" for t in subject.children)
        if has_conj:
            conf -= 0.05

        return min(max(conf, 0.5), 0.99)


# ── Tense Consistency Detector ────────────────────────────────────────

class TenseConsistencyDetector(CandidateDetector):
    """
    Detects tense consistency errors within sentences.
    
    Checks that verbs in the same sentence don't mix tenses inappropriately.
    """
    name = "tense_detector"

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer') -> List[Candidate]:
        candidates = []
        doc = nlp(text)

        for sent in doc.sents:
            sent_start = sent.start_char
            tokens = list(sent)

            # Collect all finite verbs with their tenses
            verb_tenses = []
            for token in tokens:
                if token.pos_ in ("VERB", "AUX") and token.dep_ != "aux":
                    tense = self._get_verb_tense(token)
                    if tense:
                        verb_tenses.append((token, tense))

            if len(verb_tenses) < 2:
                continue

            # Check for tense consistency
            # Main clause verb determines the expected tense
            main_verb = None
            for token, tense in verb_tenses:
                if token.dep_ == "ROOT":
                    main_verb = (token, tense)
                    break

            if not main_verb:
                continue

            main_token, main_tense = main_verb

            # Check each non-main verb
            for token, tense in verb_tenses:
                if token == main_token:
                    continue

                # Skip if verb is a past participle in passive chain
                if token.tag_ == "VBN" and any(t.dep_ == "auxpass" for t in token.head.children):
                    continue

                # Skip if in subordinate clause with its own tense marker
                if self._has_own_tense_marker(token):
                    continue

                # Skip if verb is after a modal
                if self._has_modal_before(token, tokens):
                    continue

                # Skip non-finite verbs (gerunds, infinitives)
                if token.tag_ in ("VBG", "VB"):
                    continue

                # Skip if verb has auxiliaries (part of a verb chain)
                if any(t.dep_ in ("aux", "auxpass") for t in token.children):
                    continue

                # Check for specific patterns
                if main_tense == "past" and tense == "present":
                    # "Yesterday I go to the store" → should be past
                    if not self._is_gerund_or_participle(token):
                        replacement = self._get_past_form(token)
                        if replacement and replacement != token.text.lower():
                            candidates.append(Candidate(
                                start=sent_start + token.idx,
                                end=sent_start + token.idx + len(token.text),
                                original=token.text,
                                replacement=replacement,
                                category="tense",
                                rule_id="TENSE_PAST_CONTEXT",
                                message=f'In past tense context, use "{replacement}" instead of "{token.text}".',
                                checker_name=self.name,
                                raw_confidence=0.75,
                                context=sent.text,
                                sentence_text=sent.text,
                                metadata={"main_tense": main_tense, "token_tense": tense}
                            ))

        return candidates

    def _get_verb_tense(self, token) -> Optional[str]:
        """Determine the tense of a verb token."""
        tag = token.tag_
        if tag == "VBD":
            return "past"
        if tag in ("VBZ", "VBP"):
            return "present"
        if tag == "VB":
            return "base"
        if tag == "VBG":
            return "gerund"
        if tag == "VBN":
            return "past_participle"
        return None

    def _has_own_tense_marker(self, token) -> bool:
        """Check if a verb has its own tense marker (e.g., 'that' clause)."""
        current = token
        while current.head != current:
            if current.dep_ in ("ccomp", "xcomp", "advcl"):
                return True
            current = current.head
        return False

    def _has_modal_before(self, token, tokens) -> bool:
        """Check if there's a modal verb before this verb."""
        for t in tokens:
            if t.i < token.i and t.dep_ == "aux" and t.text.lower() in MODALS:
                return True
        return False

    def _is_gerund_or_participle(self, token) -> bool:
        """Check if the verb is a gerund or participle (not a finite verb)."""
        return token.tag_ in ("VBG", "VBN")

    def _get_past_form(self, token) -> Optional[str]:
        """Get the past tense form of a verb."""
        lemma = token.lemma_.lower()
        if lemma in IRREGULAR_VERBS:
            return IRREGULAR_VERBS[lemma].get("past", lemma + "ed")
        # Regular verb: add -ed
        if lemma.endswith("e"):
            return lemma + "d"
        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ied"
        return lemma + "ed"


# ── Pronoun Case Detector ─────────────────────────────────────────────

class PronounCaseDetector(CandidateDetector):
    """
    Detects pronoun case errors.
    
    Examples:
    - "Me and him went" → "He and I went"
    - "Between you and I" → "between you and me"
    - "Her went home" → "She went home"
    """
    name = "pronoun_detector"

    SUBJECT_FORMS = {"i": "I", "me": "I", "he": "he", "him": "he",
                     "she": "she", "her": "she", "we": "we", "us": "we",
                     "they": "they", "them": "they"}
    OBJECT_FORMS = {"i": "me", "me": "me", "he": "him", "him": "him",
                    "she": "her", "her": "her", "we": "us", "us": "us",
                    "they": "them", "them": "them"}

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer') -> List[Candidate]:
        candidates = []
        doc = nlp(text)

        for sent in doc.sents:
            sent_start = sent.start_char
            tokens = list(sent)

            for token in tokens:
                if token.pos_ != "PRON":
                    continue

                # Subject position → should be subject form
                if token.dep_ in ("nsubj", "nsubjpass"):
                    if token.lower_ in self.SUBJECT_FORMS:
                        correct = self.SUBJECT_FORMS[token.lower_]
                        if token.text != correct:
                            # Special case: "me" in compound subject
                            if token.lower_ == "me":
                                confidence = 0.85
                                # Check if it's in a compound subject
                                for t in tokens:
                                    if t.dep_ == "conj" and t.head == token.head:
                                        if t.lower_ in ("and", "or", "nor"):
                                            confidence = 0.90
                                            break

                            candidates.append(Candidate(
                                start=sent_start + token.idx,
                                end=sent_start + token.idx + len(token.text),
                                original=token.text,
                                replacement=correct,
                                category="pronouns",
                                rule_id="PRONOUN_SUBJECT_CASE",
                                message=f'Use the subject form "{correct}" instead of "{token.text}" in subject position.',
                                checker_name=self.name,
                                raw_confidence=confidence if token.lower_ == "me" else 0.85,
                                context=sent.text,
                                sentence_text=sent.text,
                                metadata={"position": "subject"}
                            ))

                # Object position → should be object form
                if token.dep_ in ("dobj", "pobj", "iobj"):
                    if token.lower_ in self.OBJECT_FORMS:
                        correct = self.OBJECT_FORMS[token.lower_]
                        if token.text.lower() != correct and token.lower_ != correct:
                            candidates.append(Candidate(
                                start=sent_start + token.idx,
                                end=sent_start + token.idx + len(token.text),
                                original=token.text,
                                replacement=correct,
                                category="pronouns",
                                rule_id="PRONOUN_OBJECT_CASE",
                                message=f'Use the object form "{correct}" instead of "{token.text}" in object position.',
                                checker_name=self.name,
                                raw_confidence=0.80,
                                context=sent.text,
                                sentence_text=sent.text,
                                metadata={"position": "object"}
                            ))

        return candidates


# ── Possessive Detector ───────────────────────────────────────────────

class PossessiveDetector(CandidateDetector):
    """
    Detects its/it's, your/you're, their/they're errors.
    """
    name = "possessive_detector"

    PAIRS = {
        "its": {"correct": "its", "wrong": "it's", "meaning": "belonging to it"},
        "it's": {"correct": "it's", "wrong": "its", "meaning": "it is / it has"},
        "your": {"correct": "your", "wrong": "you're", "meaning": "belonging to you"},
        "you're": {"correct": "you're", "wrong": "your", "meaning": "you are"},
        "their": {"correct": "their", "wrong": "they're", "meaning": "belonging to them"},
        "they're": {"correct": "they're", "wrong": "their", "meaning": "they are"},
    }

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer') -> List[Candidate]:
        candidates = []
        doc = nlp(text)

        for sent in doc.sents:
            sent_start = sent.start_char

            for token in sent:
                lower = token.text.lower()

                # its vs it's
                if lower == "its":
                    # Check if followed by a noun (possessive = correct)
                    # or by adjective/verb (contraction = should be it's)
                    next_token = token.nbor(1) if token.i + 1 < len(doc) else None
                    if next_token:
                        if next_token.pos_ in ("ADJ", "ADV", "VERB", "AUX"):
                            # "its very good" → "it's"
                            candidates.append(Candidate(
                                start=sent_start + token.idx,
                                end=sent_start + token.idx + len(token.text),
                                original=token.text,
                                replacement="it's",
                                category="word_usage",
                                rule_id="POSSESSIVE_ITS_VS_ITS",
                                message='"its" should be "it\'s" (it is) in this context.',
                                checker_name=self.name,
                                raw_confidence=0.78,
                                context=sent.text,
                                sentence_text=sent.text,
                            ))

                elif lower == "it's":
                    # Check if followed by a noun (should be its)
                    next_token = token.nbor(1) if token.i + 1 < len(doc) else None
                    if next_token:
                        if next_token.pos_ in ("NOUN", "PROPN"):
                            # "it's tail" → "its"
                            candidates.append(Candidate(
                                start=sent_start + token.idx,
                                end=sent_start + token.idx + len(token.text),
                                original=token.text,
                                replacement="its",
                                category="word_usage",
                                rule_id="POSSESSIVE_ITS_CONTRACTION",
                                message='"it\'s" should be "its" (possessive) before a noun.',
                                checker_name=self.name,
                                raw_confidence=0.80,
                                context=sent.text,
                                sentence_text=sent.text,
                            ))

                # your vs you're
                elif lower == "your":
                    next_token = token.nbor(1) if token.i + 1 < len(doc) else None
                    if next_token and next_token.lower_ in ("welcome", "right", "wrong", "self", "own"):
                        candidates.append(Candidate(
                            start=sent_start + token.idx,
                            end=sent_start + token.idx + len(token.text),
                            original=token.text,
                            replacement="you're",
                            category="word_usage",
                            rule_id="POSSESSIVE_YOUR_VS_YOURE",
                            message='"your" should be "you\'re" (you are) in this context.',
                            checker_name=self.name,
                            raw_confidence=0.80,
                            context=sent.text,
                            sentence_text=sent.text,
                        ))

                # their vs they're
                elif lower == "their":
                    next_token = token.nbor(1) if token.i + 1 < len(doc) else None
                    if next_token and next_token.pos_ in ("ADV", "ADJ"):
                        if next_token.lower_ in ("very", "really", "quite", "so", "too", "not", "always", "never", "just", "also"):
                            candidates.append(Candidate(
                                start=sent_start + token.idx,
                                end=sent_start + token.idx + len(token.text),
                                original=token.text,
                                replacement="they're",
                                category="word_usage",
                                rule_id="POSSESSIVE_THEIR_VS_THEYRE",
                                message='"their" should be "they\'re" (they are) in this context.',
                                checker_name=self.name,
                                raw_confidence=0.75,
                                context=sent.text,
                                sentence_text=sent.text,
                            ))

        return candidates


# ── Modal Verb Detector ───────────────────────────────────────────────

class ModalVerbDetector(CandidateDetector):
    """
    Detects modal verb errors.
    
    Examples:
    - "She can sings" → "can sing"
    - "He must to go" → "must go"
    - "They should went" → "should go"
    """
    name = "modal_detector"

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer') -> List[Candidate]:
        candidates = []
        doc = nlp(text)

        for sent in doc.sents:
            sent_start = sent.start_char
            tokens = list(sent)

            for token in tokens:
                if token.text.lower() in MODALS and token.dep_ == "aux":
                    # Find the main verb
                    main_verb = None
                    for t in tokens:
                        if t.head == token and t.pos_ in ("VERB", "AUX"):
                            main_verb = t
                            break

                    if not main_verb:
                        continue

                    # Modal + to → should be modal + base
                    if main_verb.text.lower() == "to":
                        # Find the actual verb after "to"
                        for t in tokens:
                            if t.i > main_verb.i and t.pos_ in ("VERB",):
                                replacement = t.lemma_
                                candidates.append(Candidate(
                                    start=sent_start + token.idx,
                                    end=sent_start + t.idx + len(t.text),
                                    original=f"{token.text} to {t.text}",
                                    replacement=f"{token.text} {replacement}",
                                    category="grammar",
                                    rule_id="MODAL_NO_TO",
                                    message=f'After "{token.text}", use the base verb form, not "to + verb".',
                                    checker_name=self.name,
                                    raw_confidence=0.90,
                                    context=sent.text,
                                    sentence_text=sent.text,
                                    metadata={"modal": token.text, "verb": t.text}
                                ))
                                break

                    # Modal + conjugated verb → should be modal + base
                    elif main_verb.tag_ in ("VBZ", "VBP") and main_verb.text.lower() != main_verb.lemma_:
                        replacement = main_verb.lemma_
                        candidates.append(Candidate(
                            start=sent_start + token.idx,
                            end=sent_start + main_verb.idx + len(main_verb.text),
                            original=f"{token.text} {main_verb.text}",
                            replacement=f"{token.text} {replacement}",
                            category="grammar",
                            rule_id="MODAL_BASE_FORM",
                            message=f'After "{token.text}", use the base verb form "{replacement}" not "{main_verb.text}".',
                            checker_name=self.name,
                            raw_confidence=0.90,
                            context=sent.text,
                            sentence_text=sent.text,
                            metadata={"modal": token.text, "verb": main_verb.text}
                        ))

                    # Modal + past tense → should be modal + base
                    elif main_verb.tag_ == "VBD" and main_verb.text.lower() != main_verb.lemma_:
                        replacement = main_verb.lemma_
                        candidates.append(Candidate(
                            start=sent_start + token.idx,
                            end=sent_start + main_verb.idx + len(main_verb.text),
                            original=f"{token.text} {main_verb.text}",
                            replacement=f"{token.text} {replacement}",
                            category="grammar",
                            rule_id="MODAL_NO_PAST",
                            message=f'After "{token.text}", use the base verb form, not the past tense.',
                            checker_name=self.name,
                            raw_confidence=0.88,
                            context=sent.text,
                            sentence_text=sent.text,
                            metadata={"modal": token.text, "verb": main_verb.text}
                        ))

        return candidates


# ── "Could have" Detector ─────────────────────────────────────────────

class CouldHaveDetector(CandidateDetector):
    """
    Detects "could of" → "could have" / "could've".
    """
    name = "could_have_detector"

    CONTRACTIONS = {
        "could of": "could have",
        "should of": "should have",
        "would of": "would have",
        "might of": "might have",
        "must of": "must have",
        "may of": "may have",
        "shall of": "shall have",
        "will of": "will have",
        "can of": "can have",
    }

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer') -> List[Candidate]:
        candidates = []
        text_lower = text.lower()

        for wrong, correct in self.CONTRACTIONS.items():
            idx = 0
            while True:
                pos = text_lower.find(wrong, idx)
                if pos == -1:
                    break
                candidates.append(Candidate(
                    start=pos,
                    end=pos + len(wrong),
                    original=text[pos:pos + len(wrong)],
                    replacement=correct,
                    category="word_usage",
                    rule_id="COULD_HAVE_NOT_OF",
                    message=f'"{wrong}" should be "{correct}".',
                    checker_name=self.name,
                    raw_confidence=0.95,
                    context=text[max(0, pos - 30):pos + len(wrong) + 30],
                    sentence_text=text,
                ))
                idx = pos + 1

        return candidates


# ── "Used to" Detector ────────────────────────────────────────────────

class UsedToDetector(CandidateDetector):
    """
    Detects "use to" → "used to".
    """
    name = "used_to_detector"

    def detect(self, text: str, context_analyzer: 'ContextAnalyzer') -> List[Candidate]:
        candidates = []
        # Match "use to" but not "used to"
        pattern = r'\buse to\b'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            candidates.append(Candidate(
                start=m.start(),
                end=m.end(),
                original=m.group(),
                replacement="used to",
                category="grammar",
                rule_id="USED_TO",
                message='"use to" should be "used to" when expressing past habit.',
                checker_name=self.name,
                raw_confidence=0.85,
                context=text[max(0, m.start() - 30):m.end() + 30],
                sentence_text=text,
            ))
        return candidates
