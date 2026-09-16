"""
text_preprocessor.py — Protect content that should not be checked as normal English.

Handles: URLs, emails, code blocks, HTML, Markdown, numbers, currency,
dates, abbreviations, acronyms, product names, company names, people's names,
technical vocabulary, quoted content.

Preserves exact character offsets for frontend underline alignment.
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class ProtectedSpan:
    """A span of text that should be skipped by grammar checkers."""
    start: int
    end: int
    text: str
    category: str  # url, email, code, html, markdown, number, currency, etc.


class TextPreprocessor:
    """Protects non-English content and extracts metadata for checkers."""

    # Patterns to protect (order matters — first match wins)
    PROTECT_PATTERNS = [
        # URLs
        (re.compile(r'https?://\S+|www\.\S+'), "url"),
        # Emails
        (re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'), "email"),
        # Code blocks (backtick)
        (re.compile(r'`[^`\n]+`'), "code_inline"),
        # Fenced code blocks
        (re.compile(r'```[\s\S]*?```'), "code_block"),
        # HTML tags
        (re.compile(r'<[^>]+>'), "html"),
        # Markdown images ![alt](url)
        (re.compile(r'!\[[^\]]*\]\([^)]*\)'), "markdown"),
        # Markdown links [text](url)
        (re.compile(r'\[[^\]]*\]\([^)]*\)'), "markdown"),
        # Markdown headings
        (re.compile(r'^#{1,6}\s+', re.MULTILINE), "markdown"),
        # Currency ($100, EUR 50, etc.)
        (re.compile(r'[$€£¥]\d[\d,.]*|\b\d[\d,.]*\s*(?:USD|EUR|GBP|JPY)\b'), "currency"),
        # Dates (2024-01-15, 01/15/2024, Jan 15 2024, etc.)
        (re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE), "date"),
        # Times (10:30, 10:30 AM, 14:00)
        (re.compile(r'\b\d{1,2}:\d{2}(?:\s*(?:AM|PM|am|pm))?\b'), "time"),
        # Numbers with units (100kg, 5km, 3.5GB)
        (re.compile(r'\b\d[\d,.]*\s*(?:kg|km|mb|gb|tb|mph|kph|mm|cm|dm|ft|in|oz|lb)\b', re.IGNORECASE), "number_unit"),
        # Bare numbers — NOT protected so detectors can catch preposition errors with years/dates
        # (re.compile(r'\b\d[\d,.]*\b'), "number"),
        # Abbreviations (U.S.A., e.g., i.e., etc.)
        (re.compile(r'\b(?:[A-Z]\.){2,}\b|\b(?:e\.g|i\.e|etc|vs|viz)\b', re.IGNORECASE), "abbreviation"),
        # All-caps words (acronyms: NASA, FBI, HTML)
        (re.compile(r'\b[A-Z]{2,}\b'), "acronym"),
        # Quoted content (preserve as-is) — only double quotes and curly quotes
        # Single quotes are apostrophes in contractions (don't, didn't), NOT quotes
        (re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d|\u2018[^\u2019]*\u2019'), "quoted"),
    ]

    # Words that are valid even though they look like errors
    COMMON_ABBREVIATIONS = {
        "etc", "eg", "ie", "vs", "dr", "mr", "mrs", "ms", "prof",
        "st", "ave", "blvd", "rd", "dept", "est", "inc", "ltd",
        "corp", "llc", "no", "tel", "fax", "www", "http", "https",
    }

    def __init__(self):
        pass

    def protect(self, text: str) -> Tuple[str, List[ProtectedSpan]]:
        """
        Returns (cleaned_text, protected_spans).
        cleaned_text replaces protected content with placeholders.
        protected_spans maps placeholder positions back to original offsets.
        """
        protected = []
        # Find all matches, sort by start position
        all_matches = []
        for pattern, category in self.PROTECT_PATTERNS:
            for m in pattern.finditer(text):
                all_matches.append((m.start(), m.end(), m.group(), category))

        # Sort by start, then by length (longest first)
        all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Remove overlapping spans (keep first/longest)
        kept = []
        last_end = 0
        for start, end, text_match, category in all_matches:
            if start >= last_end:
                kept.append((start, end, text_match, category))
                last_end = end

        # Build placeholder text
        result = []
        pos = 0
        for start, end, original, category in kept:
            # Add text before this match
            if start > pos:
                result.append(text[pos:start])
            # Add placeholder
            placeholder = f"__PROTECTED_{len(protected)}__"
            result.append(placeholder)
            protected.append(ProtectedSpan(
                start=start, end=end, text=original, category=category
            ))
            pos = end

        # Add remaining text
        if pos < len(text):
            result.append(text[pos:])

        return "".join(result), protected

    def restore(self, text: str, protected: List[ProtectedSpan]) -> str:
        """Replace placeholders back with original text."""
        for i, span in enumerate(protected):
            placeholder = f"__PROTECTED_{i}__"
            text = text.replace(placeholder, span.text)
        return text

    def get_checkable_spans(self, text: str, protected: List[ProtectedSpan]) -> List[Tuple[int, int]]:
        """Return spans of text that ARE checkable (not protected)."""
        if not protected:
            return [(0, len(text))]

        spans = []
        pos = 0
        for span in sorted(protected, key=lambda s: s.start):
            if span.start > pos:
                spans.append((pos, span.start))
            pos = span.end
        if pos < len(text):
            spans.append((pos, len(text)))
        return spans

    def is_protected(self, position: int, protected: List[ProtectedSpan]) -> bool:
        """Check if a character position falls within a protected span."""
        for span in protected:
            if span.start <= position < span.end:
                return True
        return False
