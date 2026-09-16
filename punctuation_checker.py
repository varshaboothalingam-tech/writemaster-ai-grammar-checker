"""
Extended Punctuation Checker — rules beyond the basic PunctuationEngine.

Rules:
  - Missing comma before coordinating conjunction in compound sentences
  - Missing/misplaced comma in lists
  - Apostrophe misuse: possessive vs contraction (its/it's, your/you're, their/they're)
  - Unpaired/mismatched quotation marks
  - Semicolon misuse
"""

import re
from typing import List, Dict


def _make_error(text, start, end, category, message, suggestion="",
                severity="MEDIUM", confidence=0.8, context="", explanation=""):
    return {
        "original_text": text,
        "replacement": suggestion,
        "category": category,
        "error_type": "punctuation",
        "message": message,
        "explanation": explanation or message,
        "start_position": start,
        "end_position": end,
        "severity": severity,
        "confidence": round(confidence, 2),
        "context": context,
        "rule_id": "punctuation_extended",
    }


class PunctuationChecker:
    """Extended punctuation rules beyond the basic engine."""

    COORDINATING_CONJUNCTIONS = {"for", "and", "nor", "but", "or", "yet", "so"}

    POSSESSIVE_CONTRACTION_PAIRS = {
        "its": {"possessive": True, "contraction": "it's", "meaning": "it is/it has"},
        "it's": {"possessive": False, "contraction": True, "meaning": "it is/it has"},
        "your": {"possessive": True, "contraction": None, "meaning": "belonging to you"},
        "you're": {"possessive": False, "contraction": "you are", "meaning": "you are"},
        "their": {"possessive": True, "contraction": None, "meaning": "belonging to them"},
        "they're": {"possessive": False, "contraction": "they are", "meaning": "they are"},
        "there": {"possessive": False, "contraction": None, "meaning": "location"},
        "whose": {"possessive": True, "contraction": None, "meaning": "belonging to whom"},
        "who's": {"possessive": False, "contraction": "who is", "meaning": "who is"},
    }

    def check(self, text: str) -> List[Dict]:
        if not text or not text.strip():
            return []
        errors = []
        errors.extend(self._check_comma_before_conjunction(text))
        errors.extend(self._check_list_commas(text))
        errors.extend(self._check_possessive_contraction(text))
        errors.extend(self._check_unpaired_quotes(text))
        errors.extend(self._check_semicolon_misuse(text))
        return errors

    def _check_comma_before_conjunction(self, text: str) -> List[Dict]:
        """Missing comma before FANBOYS in compound sentences."""
        errors = []
        pattern = re.compile(
            r'\b(\w+)\s+(and|but|or|nor|yet|so|for)\s+(\w+)',
            re.IGNORECASE
        )
        for m in pattern.finditer(text):
            before_start = max(0, m.start() - 1)
            char_before = text[before_start] if before_start < m.start() else ""

            # Check if there's a comma before the conjunction
            # Look back from the conjunction position
            conj_start = m.start() + len(m.group(1)) + 1
            preceding = text[before_start:conj_start].rstrip()

            if preceding and not preceding.endswith(','):
                # Check if both sides look like independent clauses
                # (have a verb — simplified heuristic)
                left = m.group(1).lower()
                right = m.group(3).lower()
                conjunction = m.group(2).lower()

                # Skip if it's a simple list pattern like "cats and dogs"
                if left in ("a", "an", "the", "my", "your", "his", "her", "its", "our", "their"):
                    continue
                if right in ("the", "a", "an"):
                    continue

                comma_pos = conj_start - 1
                errors.append(_make_error(
                    text=",", start=comma_pos, end=comma_pos,
                    category="PUNCTUATION",
                    message=f'Consider adding a comma before "{conjunction}" in this compound sentence.',
                    suggestion=",",
                    severity="LOW",
                    confidence=0.60,
                    context=text[max(0, m.start() - 30):m.end() + 30],
                    explanation=f'A comma before a coordinating conjunction ("{conjunction}") joining two independent clauses is recommended.',
                ))
        return errors

    def _check_list_commas(self, text: str) -> List[Dict]:
        """Detect missing commas in lists of 3+ items."""
        errors = []
        # Pattern: word word and/or word (without commas separating them)
        # e.g., "I like cats dogs and birds" should be "I like cats, dogs, and birds"
        list_pattern = re.compile(
            r'\b(\w+(?:\s+\w+)*)\s+(and|or)\s+(\w+(?:\s+\w+)*)\b',
            re.IGNORECASE
        )
        for m in list_pattern.finditer(text):
            preceding_char = text[m.start() - 1:m.start()] if m.start() > 0 else ""
            if preceding_char in ",;:":
                continue

            left_words = m.group(1).split()
            right_words = m.group(3).split()

            # If both sides have multiple words, it might be a list
            if len(left_words) >= 2 and len(right_words) >= 1:
                # Check if there's no comma anywhere in the left portion
                left_text = m.group(1)
                if "," not in left_text:
                    errors.append(_make_error(
                        text=m.group(0), start=m.start(), end=m.end(),
                        category="PUNCTUATION",
                        message='Missing commas in a list of items.',
                        suggestion=m.group(0),
                        severity="LOW",
                        confidence=0.55,
                        context=text[max(0, m.start() - 30):m.end() + 30],
                        explanation='Use commas to separate items in a list for clarity.',
                    ))
        return errors

    def _check_possessive_contraction(self, text: str) -> List[Dict]:
        """Detect its/it's, your/you're, their/they're misuse."""
        errors = []

        # its vs it's
        for m in re.finditer(r"\bits\b", text, re.IGNORECASE):
            word = m.group()
            start, end = m.start(), m.end()
            after = text[end:end + 20].lstrip()
            before = text[max(0, start - 20):start].rstrip()

            # "it's" followed by a verb/adjective → likely contraction
            if re.match(r"(n't|s\b|ll\b|re\b|ve\b)", after):
                continue  # Already correct contraction

            # "its" followed by a noun → possessive (correct)
            if re.match(r"\s+\w+(s?\b)", after) and not re.match(r"\s+(is|was|has|had|will|would|could|should|might|must|can)\b", after):
                continue  # Possessive usage

            # "its" followed by adjective/adverb → likely should be "it's"
            if re.match(r"\s+(very|really|quite|so|too|not|always|never|just|also)\b", after):
                errors.append(_make_error(
                    text=word, start=start, end=end,
                    category="PUNCTUATION",
                    message=f'"{word}" should be "it\'s" (it is) in this context.',
                    suggestion="it's",
                    severity="MEDIUM",
                    confidence=0.78,
                    context=text[max(0, start - 30):end + 30],
                    explanation='"it\'s" is the contraction of "it is" or "it has".',
                ))

        # your vs you're
        for m in re.finditer(r"\byour\b", text, re.IGNORECASE):
            word = m.group()
            start, end = m.start(), m.end()
            after = text[end:end + 20].lstrip()

            if re.match(r"\s+(very|really|quite|so|too|not|always|never|just|also|welcome|right|wrong|self|own)\b", after):
                # "your welcome" → "you're welcome"
                if after.lower().startswith("welcome"):
                    errors.append(_make_error(
                        text=word, start=start, end=end,
                        category="PUNCTUATION",
                        message=f'"{word}" should be "you\'re" (you are) in this context.',
                        suggestion="you're",
                        severity="MEDIUM",
                        confidence=0.80,
                        context=text[max(0, start - 30):end + 30],
                        explanation='"you\'re" is the contraction of "you are".',
                    ))

        # their vs they're
        for m in re.finditer(r"\btheir\b", text, re.IGNORECASE):
            word = m.group()
            start, end = m.start(), m.end()
            after = text[end:end + 20].lstrip()

            if re.match(r"\s+(very|really|quite|so|too|not|always|never|just|also)\b", after):
                errors.append(_make_error(
                    text=word, start=start, end=end,
                    category="PUNCTUATION",
                    message=f'"{word}" should be "they\'re" (they are) in this context.',
                    suggestion="they're",
                    severity="MEDIUM",
                    confidence=0.75,
                    context=text[max(0, start - 30):end + 30],
                    explanation='"they\'re" is the contraction of "they are".',
                ))

        return errors

    def _check_unpaired_quotes(self, text: str) -> List[Dict]:
        """Detect unpaired/mismatched quotation marks."""
        errors = []
        for quote_char in ('"', "'"):
            if quote_char == "'" and re.search(r"\w'\w", text):
                # Skip apostrophes in contractions
                positions = []
                for m in re.finditer(re.escape(quote_char), text):
                    pos = m.start()
                    # Check if this is likely an apostrophe (between letters)
                    if pos > 0 and pos < len(text) - 1:
                        if text[pos - 1].isalpha() and text[pos + 1].isalpha():
                            continue
                    positions.append(pos)
            else:
                positions = [m.start() for m in re.finditer(re.escape(quote_char), text)]

            if len(positions) % 2 != 0:
                # Find the unmatched one
                last_pos = positions[-1]
                errors.append(_make_error(
                    text=quote_char, start=last_pos, end=last_pos + 1,
                    category="PUNCTUATION",
                    message=f'Unpaired quotation mark: "{quote_char}".',
                    suggestion='',
                    severity="MEDIUM",
                    confidence=0.70,
                    context=text[max(0, last_pos - 30):last_pos + 30],
                    explanation='Every opening quotation mark needs a matching closing mark.',
                ))
        return errors

    def _check_semicolon_misuse(self, text: str) -> List[Dict]:
        """Detect semicolons used incorrectly."""
        errors = []
        # Semicolon followed by a conjunction (should be comma or nothing)
        for m in re.finditer(r';\s*(and|but|or|nor|yet|so|for)\b', text, re.IGNORECASE):
            errors.append(_make_error(
                text=m.group(), start=m.start(), end=m.end(),
                category="PUNCTUATION",
                message=f'Semicolon before "{m.group(1)}" is unnecessary. Use a comma or remove the semicolon.',
                suggestion=m.group(1),
                severity="LOW",
                confidence=0.65,
                context=text[max(0, m.start() - 30):m.end() + 30],
                explanation='A semicolon before a coordinating conjunction is usually redundant.',
            ))

        # Semicolon after a subordinating conjunction
        for m in re.finditer(r'\b(although|because|since|while|if|when|after|before|unless|until|whereas|though)\b\s+[^;]*;', text, re.IGNORECASE):
            errors.append(_make_error(
                text=";", start=m.end() - 1, end=m.end(),
                category="PUNCTUATION",
                message='Use a comma instead of a semicolon after a dependent clause.',
                suggestion=",",
                severity="LOW",
                confidence=0.60,
                context=text[max(0, m.start() - 30):m.end() + 30],
                explanation='A dependent clause followed by an independent clause uses a comma, not a semicolon.',
            ))

        return errors
