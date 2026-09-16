"""
fp_filter.py — False-positive filter for the grammar pipeline.

Asks: Is the original actually incorrect? Could it be valid?
If uncertain, SUPPRESS the suggestion.
"""

from typing import List, Dict, Set, Tuple
from grammar_pipeline import Candidate, FalsePositiveFilter
from spacy_context import (
    nlp, COLLECTIVE_NOUNS, UNCOUNTABLE_NOUNS,
    IRREGULAR_VERBS, MODALS, BE_FORMS, HAVE_FORMS
)


# ── Protected terms (never flag) ──────────────────────────────────────

PROPER_NOUNS: Set[str] = set()  # Will be populated by NER

TECHNICAL_TERMS: Set[str] = {
    "javascript", "python", "react", "node", "api", "url", "html", "css",
    "sql", "json", "xml", "http", "https", "ftp", "ssh", "dns", "tcp",
    "udp", "ip", "vpn", "ssl", "tls", "aws", "azure", "gcp",
    "kubernetes", "docker", "git", "github", "gitlab", "bitbucket",
    "npm", "pip", "gem", "cargo", "maven", "gradle",
    "linux", "unix", "windows", "macos", "ios", "android",
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    "kafka", "rabbitmq", "nginx", "apache",
}

# Words that are correct in certain dialects/informal English
DIALECTAL_ACCEPTABLE = {
    "they was", "we was", "you was", "he don't", "she don't",
    "they don't", "we don't", "i seen", "he seen", "she seen",
    "we was", "they was", "he be", "she be", "we be", "they be",
    "i be", "it be",
}


class SpaCyFalsePositiveFilter(FalsePositiveFilter):
    """
    Filters false positives using multiple strategies:
    1. Proper noun protection (NER)
    2. Technical term protection
    3. Dialectal/informal acceptability
    4. Context-aware suppression
    5. Collective noun handling
    6. Uncountable noun handling
    7. Quoted text protection
    8. Irregular verb edge cases
    """

    name = "spacy_fp_filter"

    def __init__(self):
        self.nlp = nlp

    def filter(self, candidates: List[Candidate], text: str) -> List[Candidate]:
        """Filter candidates that are likely false positives."""
        if not candidates:
            return []

        # Get NER entities for proper noun protection
        doc = self.nlp(text)
        proper_nouns = set()
        for ent in doc.ents:
            proper_nouns.add(ent.text.lower())

        filtered = []
        for c in candidates:
            if self._is_false_positive(c, text, doc, proper_nouns):
                continue
            filtered.append(c)

        return filtered

    def _is_false_positive(self, c: Candidate, text: str, doc, proper_nouns: Set[str]) -> bool:
        """Check if a candidate is likely a false positive."""

        # Strategy 1: Proper noun protection
        if self._is_proper_noun_context(c, text, proper_nouns):
            return True

        # Strategy 2: Technical term protection
        if self._is_technical_term(c, text):
            return True

        # Strategy 3: Dialectal acceptability
        if self._is_dialectal_acceptable(c, text):
            return True

        # Strategy 4: Collective noun handling
        if self._is_collective_noun_ok(c, text):
            return True

        # Strategy 5: Uncountable noun handling
        if self._is_uncountable_noun_ok(c, text):
            return True

        # Strategy 5b: Singular they (everyone, someone, nobody)
        if self._is_singular_they_ok(c, text):
            return True

        # Strategy 6: Quoted text protection
        if self._is_in_quotes(c, text):
            return True

        # Strategy 7: Irregular verb edge cases
        if self._is_irregular_verb_ok(c, text):
            return True

        # Strategy 8: "There is/are" constructions
        if self._is_there_construction_ok(c, text):
            return True

        # Strategy 9: Passive voice (don't flag as grammar error)
        if self._is_passive_context(c, text):
            return True

        # Strategy 10: Modal + base form edge cases
        if self._is_modal_edge_case(c, text):
            return True

        return False

    def _is_proper_noun_context(self, c: Candidate, text: str, proper_nouns: Set[str]) -> bool:
        """Check if the candidate is in a proper noun context."""
        # Check if the word at this position is a named entity
        doc = self.nlp(text)
        for token in doc:
            if token.idx >= c.start and token.idx < c.end:
                if token.ent_type_:
                    return True
            if token.idx >= c.end:
                break

        # Check if the original word is a known proper noun
        if c.original.lower() in proper_nouns:
            return True

        return False

    def _is_technical_term(self, c: Candidate, text: str) -> bool:
        """Check if the candidate is a technical term."""
        if c.original.lower() in TECHNICAL_TERMS:
            return True
        # Check surrounding context
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in TECHNICAL_TERMS:
                # Check if candidate is near this technical term
                if abs(i - words.index(c.original)) <= 2:
                    return True
        return False

    def _is_dialectal_acceptable(self, c: Candidate, text: str) -> bool:
        """Check if the usage is acceptable in some dialects."""
        context = text[max(0, c.start - 20):c.end + 20].lower().strip()
        for pattern in DIALECTAL_ACCEPTABLE:
            if pattern in context:
                return True
        return False

    def _is_collective_noun_ok(self, c: Candidate, text: str) -> bool:
        """Handle collective nouns — they can be singular or plural."""
        if c.category != "agreement":
            return False

        # Find the subject in context
        doc = self.nlp(text)
        for token in doc:
            if token.lower_ in COLLECTIVE_NOUNS:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    # Collective nouns can take either verb form
                    # Only flag if it's clearly wrong
                    return True  # Suppress all collective noun agreement issues
        return False

    def _is_uncountable_noun_ok(self, c: Candidate, text: str) -> bool:
        """Handle uncountable nouns — they take singular verbs."""
        if c.category != "agreement":
            return False

        doc = self.nlp(text)
        for token in doc:
            if token.lower_ in UNCOUNTABLE_NOUNS:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    return True  # Suppress uncountable noun agreement issues
        return False

    def _is_singular_they_ok(self, c: Candidate, text: str) -> bool:
        """Handle singular they (everyone, someone, nobody + have/are)."""
        if c.category != "agreement":
            return False
        singular_they_subjects = {"everyone", "everybody", "someone", "somebody",
                                 "nobody", "anyone", "anybody", "each", "every",
                                 "either", "neither", "one"}
        subject = c.metadata.get("subject", "").lower()
        if subject in singular_they_subjects:
            return True
        return False

    def _is_in_quotes(self, c: Candidate, text: str) -> bool:
        """Check if the candidate is within quoted text."""
        before = text[:c.start]
        open_quotes = before.count('"') - before.count('\\"')
        if open_quotes % 2 == 1:
            return True
        return False

    def _is_irregular_verb_ok(self, c: Candidate, text: str) -> bool:
        """Handle irregular verb edge cases."""
        if c.category not in ("grammar", "tense"):
            return False

        # Check for specific irregular verbs that are their own past tense
        IRREGULAR_SAME_FORM = {"hit", "let", "put", "cut", "set", "read", "run",
                               "come", "become", "overcome", "forecast", "input",
                               "output", "shed", "spread", "upset", "bet", "bid",
                               "burst", "cost", "fit", "hit", "hurt", "knit",
                               "lay", "lead", "quit", "ride", "rise", "saw",
                               "sew", "shoot", "shut", "slit", "split", "spray",
                               "steal", "stick", "sting", "strike", "swear", "sweep",
                               "swing", "tear", "throw", "understand", "wake", "wear",
                               "win", "wound", "write"}

        original_lower = c.original.lower()
        if original_lower in IRREGULAR_SAME_FORM:
            # These verbs don't change form for past tense
            return True

        return False

    def _is_there_construction_ok(self, c: Candidate, text: str) -> bool:
        """Handle 'there is/are' constructions."""
        if c.category != "agreement":
            return False

        # Don't suppress existential there corrections from our detector
        if c.metadata.get("existential_there"):
            return False

        # Check if there's a "there" before the verb
        context_before = text[max(0, c.start - 30):c.start].lower()
        if "there" in context_before.split():
            return True  # Suppress "there is/are" issues

        return False

    def _is_passive_context(self, c: Candidate, text: str) -> bool:
        """Don't flag passive voice as a grammar error."""
        if c.category not in ("grammar", "tense"):
            return False

        # Check for passive indicators
        doc = self.nlp(text)
        for token in doc:
            if token.dep_ == "nsubjpass" or token.dep_ == "auxpass":
                if token.idx >= c.start - 50 and token.idx <= c.end + 50:
                    return True
        return False

    def _is_modal_edge_case(self, c: Candidate, text: str) -> bool:
        """Handle modal verb edge cases."""
        if c.category != "grammar":
            return False

        # Check if the verb is after a modal
        doc = self.nlp(text)
        for token in doc:
            if token.text.lower() in MODALS and token.idx < c.start:
                if c.start - token.idx < 20:  # Close enough
                    return True
        return False
