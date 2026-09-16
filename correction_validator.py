"""
correction_validator.py — Validates that proposed corrections actually fix the error
without introducing new errors.
"""

import spacy
from typing import Tuple, List
from grammar_pipeline import Candidate

nlp = spacy.load("en_core_web_sm")


class SpaCyCorrectionValidator:
    """
    Validates corrections by:
    1. Applying the correction to a copy of the sentence
    2. Re-parsing the corrected sentence
    3. Checking if the original error is resolved
    4. Checking if new errors are introduced
    """

    name = "spacy_correction_validator"

    def validate(self, candidate: Candidate, full_text: str) -> Tuple[bool, str]:
        """
        Validate a correction candidate.
        Returns (is_valid, reason).
        """
        if not candidate.replacement:
            return False, "No replacement provided"

        # Apply correction
        corrected_text = (
            full_text[:candidate.start] +
            candidate.replacement +
            full_text[candidate.end:]
        )

        # Parse both versions
        original_doc = nlp(full_text)
        corrected_doc = nlp(corrected_text)

        # Check 1: Does the correction resolve the original error?
        if not self._error_resolved(candidate, original_doc, corrected_doc):
            return False, "Correction does not resolve the original error"

        # Check 2: Does the correction introduce new errors?
        new_errors = self._detect_new_errors(candidate, original_doc, corrected_doc)
        if new_errors:
            return False, f"Correction introduces new error: {new_errors[0]}"

        # Check 3: Is the correction grammatically valid?
        if not self._is_grammatically_valid(corrected_doc):
            return False, "Corrected text is not grammatically valid"

        return True, "Valid correction"

    def _error_resolved(self, candidate: Candidate, original_doc, corrected_doc) -> bool:
        """Check if the original error is resolved after correction."""
        # Simple heuristic: if the correction changes the problematic token,
        # it's likely resolved
        for token in corrected_doc:
            if token.idx >= candidate.start and token.idx < candidate.end:
                # Check if the token at this position changed
                for orig_token in original_doc:
                    if orig_token.idx == token.idx:
                        if orig_token.text.lower() != token.text.lower():
                            return True
        return False

    def _detect_new_errors(self, candidate: Candidate, original_doc, corrected_doc) -> List[str]:
        """Detect if the correction introduces new errors."""
        errors = []

        # Check for new subject-verb agreement issues
        for sent in corrected_doc.sents:
            for token in sent:
                if token.dep_ == "nsubj" and token.pos_ in ("NOUN", "PRON"):
                    verb = token.head
                    if verb.pos_ in ("VERB", "AUX"):
                        # Simple check: does the verb agree with the subject?
                        if not self._check_basic_agreement(token, verb):
                            errors.append(f"New SVA error: {token.text} {verb.text}")

        # Check for new modal errors
        for token in corrected_doc:
            if token.text.lower() in ("can", "could", "will", "would", "shall", "should",
                                       "may", "might", "must") and token.dep_ == "aux":
                # Find the main verb
                for t in corrected_doc:
                    if t.head == token and t.pos_ == "VERB":
                        # Check if verb is in correct form after modal
                        if t.tag_ not in ("VB", "VBP"):
                            if t.text.lower() != t.lemma_:
                                errors.append(f"New modal error: {token.text} {t.text}")
                        break

        return errors

    def _check_basic_agreement(self, subject, verb) -> bool:
        """Basic subject-verb agreement check."""
        subj_number = "singular"
        if subject.tag_ in ("NNS", "NNPS"):
            subj_number = "plural"
        if subject.lower_ in ("i", "we", "you", "they"):
            subj_number = "plural"

        if subj_number == "plural":
            if verb.tag_ == "VBZ":
                return False
        elif subj_number == "singular":
            if verb.tag_ in ("VBP",) and verb.lower_ not in ("are",):
                # "are" is OK for "you"
                if subject.lower_ != "you":
                    return False

        return True

    def _is_grammatically_valid(self, doc) -> bool:
        """Check if the corrected text is grammatically valid."""
        # Basic sanity checks
        for sent in doc.sents:
            tokens = list(sent)
            if not tokens:
                continue

            # Check for obvious issues
            for i, token in enumerate(tokens):
                # Double subject
                if token.dep_ == "nsubj" and i > 0:
                    prev = tokens[i - 1]
                    if prev.dep_ == "nsubj" and prev.head == token.head:
                        return False

        return True
