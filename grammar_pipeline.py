"""
grammar_pipeline.py — Central orchestrator for the grammar checking pipeline.

Pipeline stages:
  TEXT → PREPROCESSING → CANDIDATE DETECTION → CONTEXT ANALYSIS →
  CONFIDENCE SCORING → FALSE-POSITIVE FILTER → DEDUPLICATION →
  CORRECTION VALIDATION → CORRECTION RANKING → FINAL OUTPUT

Individual checkers produce CANDIDATES, not final errors.
The pipeline validates, filters, and deduplicates before output.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ── Final Error Object Schema (Phase 11) ──────────────────────────────

@dataclass
class GrammarError:
    """Confirmed grammar error — the ONLY object shown to users."""
    start: int
    end: int
    original: str
    replacement: str
    category: str          # spelling | grammar | punctuation | agreement | tense | pronouns | articles | prepositions | sentence_structure | word_usage
    severity: str          # error | warning | info
    confidence: float      # 0.0–1.0
    explanation: str
    rule_id: str           # e.g. "SPELLING_TYPING", "SVA_SINGULAR_VERB"
    context: str = ""      # surrounding text
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Candidate:
    """An unverified potential error detected by a checker."""
    start: int
    end: int
    original: str
    replacement: str
    category: str
    rule_id: str
    message: str
    checker_name: str      # which checker produced this
    raw_confidence: float  # checker's own confidence estimate
    context: str = ""
    sentence_text: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ContextInfo:
    """Contextual information about a candidate error."""
    sentence: str
    previous_sentence: str = ""
    next_sentence: str = ""
    subject: str = ""
    subject_number: str = ""       # singular | plural | unknown
    subject_person: str = ""       # 1st | 2nd | 3rd
    verb: str = ""
    verb_number: str = ""
    tense: str = ""
    sentence_type: str = ""        # declarative | question | imperative | exclamation
    clause_type: str = ""          # main | subordinate | relative
    is_proper_noun: bool = False
    is_technical_term: bool = False
    is_quoted_text: bool = False
    is_existential_there: bool = False
    is_passive: bool = False
    auxiliary_verbs: List[str] = field(default_factory=list)
    modal_verbs: List[str] = field(default_factory=list)
    compound_subject: bool = False
    collective_noun: bool = False
    uncountable_noun: bool = False
    embedding_depth: int = 0       # 0=main clause, 1=nested once, etc.


# ── Pipeline Configuration ────────────────────────────────────────────

class PipelineConfig:
    """Configurable pipeline parameters."""
    def __init__(self, **kwargs):
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.75)
        self.min_confidence_to_show = kwargs.get('min_confidence_to_show', 0.75)
        self.high_confidence_threshold = kwargs.get('high_confidence_threshold', 0.90)
        self.max_alternatives = kwargs.get('max_alternatives', 3)
        self.enable_correction_validation = kwargs.get('enable_correction_validation', True)
        self.enable_deduplication = kwargs.get('enable_deduplication', True)
        self.protected_terms = kwargs.get('protected_terms', set())


# ── Base Classes for Pipeline Stages ──────────────────────────────────

class CandidateDetector:
    """Base class for all candidate detection engines.
    
    Each detector receives preprocessed text and returns a list of
    Candidate objects. Candidates are NOT final errors — they are
    unverified potential issues that will be validated by the pipeline.
    """
    name: str = "base_detector"

    def detect(self, text: str, context: 'ContextAnalyzer') -> List[Candidate]:
        raise NotImplementedError


class ContextAnalyzer:
    """Analyzes sentence context for candidate validation.
    
    Uses spaCy NLP to extract:
    - Subject-verb relationships
    - Tense information
    - Clause structure
    - Proper nouns / technical terms
    - Quoted text
    """
    name: str = "context_analyzer"

    def analyze(self, text: str) -> List[ContextInfo]:
        raise NotImplementedError

    def analyze_sentence(self, sentence: str, doc=None) -> ContextInfo:
        raise NotImplementedError


class FalsePositiveFilter:
    """Filters out candidates that are likely false positives.
    
    Ask: Is the original actually incorrect? Could it be valid?
    If uncertain, suppress.
    """
    name: str = "fp_filter"

    def filter(self, candidates: List[Candidate], contexts: List[ContextInfo]) -> List[Candidate]:
        raise NotImplementedError


class CorrectionValidator:
    """Validates that a proposed correction actually fixes the error
    without introducing new errors.
    """
    name: str = "correction_validator"

    def validate(self, candidate: Candidate, sentence: str) -> Tuple[bool, str]:
        """Returns (is_valid, reason)."""
        raise NotImplementedError


# ── Main Pipeline ─────────────────────────────────────────────────────

class GrammarPipeline:
    """
    Central orchestrator.
    
    Flow:
      1. Split text into sentences
      2. For each sentence, run all candidate detectors
      3. Analyze context for each candidate
      4. Score confidence
      5. Filter false positives
      6. Deduplicate
      7. Validate corrections
      8. Rank and select top suggestions
      9. Return final GrammarError objects
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self._detectors: List[CandidateDetector] = []
        self._context_analyzer: Optional[ContextAnalyzer] = None
        self._fp_filter: Optional[FalsePositiveFilter] = None
        self._correction_validator: Optional[CorrectionValidator] = None

    def register_detector(self, detector: CandidateDetector):
        self._detectors.append(detector)

    def set_context_analyzer(self, analyzer: ContextAnalyzer):
        self._context_analyzer = analyzer

    def set_fp_filter(self, fp_filter: FalsePositiveFilter):
        self._fp_filter = fp_filter

    def set_correction_validator(self, validator):
        self._correction_validator = validator

    def check(self, text: str) -> List[GrammarError]:
        """
        Main entry point. Returns confirmed GrammarError objects only.
        """
        if not text or not text.strip():
            return []

        # Step 1: Preprocess — split into sentences
        sentences = self._split_sentences(text)

        # Step 2: Context analysis (if available)
        sentence_contexts = []
        if self._context_analyzer:
            sentence_contexts = self._context_analyzer.analyze(text)

        # Step 3: Run all candidate detectors
        all_candidates: List[Candidate] = []
        for sentence_info in sentences:
            sent_text = sentence_info["text"]
            sent_start = sentence_info["start"]

            # Get context for this sentence
            ctx = self._get_context_for_sentence(sent_text, sentence_contexts)

            for detector in self._detectors:
                try:
                    candidates = detector.detect(sent_text, self._context_analyzer)
                    # Offset positions to global text positions
                    for c in candidates:
                        c.start += sent_start
                        c.end += sent_start
                        c.context = sent_text
                    all_candidates.extend(candidates)
                except Exception as e:
                    pass  # Detectors should not crash the pipeline

        # Step 4: Confidence scoring
        scored_candidates = self._score_confidence(all_candidates)

        # Step 5: False-positive filter
        filtered = self._filter_false_positives(scored_candidates, text)

        # Step 6: Deduplication
        deduped = self._deduplicate(filtered) if self.config.enable_deduplication else filtered

        # Step 7: Correction validation
        if self.config.enable_correction_validation and self._correction_validator:
            validated = self._validate_corrections(deduped, text)
        else:
            validated = deduped

        # Step 8: Rank and convert to GrammarError
        final_errors = self._rank_and_convert(validated)

        return final_errors

    def _split_sentences(self, text: str) -> List[Dict]:
        """Split text into sentences with character offsets."""
        # Simple sentence splitter using regex
        # Handles periods, question marks, exclamation marks
        sentences = []
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+', text)
        current_pos = 0
        for part in parts:
            if part.strip():
                start = text.find(part, current_pos)
                if start == -1:
                    start = current_pos
                sentences.append({
                    "text": part,
                    "start": start,
                    "end": start + len(part),
                })
                current_pos = start + len(part)
        return sentences

    def _get_context_for_sentence(self, sentence: str, contexts: List[ContextInfo]) -> Optional[ContextInfo]:
        """Find the ContextInfo for a given sentence."""
        for ctx in contexts:
            if ctx.sentence.strip() == sentence.strip():
                return ctx
        return None

    def _score_confidence(self, candidates: List[Candidate]) -> List[Candidate]:
        """Score each candidate's confidence based on multiple signals."""
        for c in candidates:
            # Base confidence from the detector
            base = c.raw_confidence

            # Adjust based on context
            # (Phase 5 will add more sophisticated scoring)
            c.metadata["final_confidence"] = base

        return candidates

    def _filter_false_positives(self, candidates: List[Candidate], text: str) -> List[Candidate]:
        """Remove candidates that are likely false positives."""
        if not self._fp_filter:
            return candidates
        return self._fp_filter.filter(candidates, text)

    def _deduplicate(self, candidates: List[Candidate]) -> List[Candidate]:
        """Remove duplicate candidates (same error from multiple detectors)."""
        if not candidates:
            return []

        # Sort by confidence (highest first)
        candidates.sort(key=lambda c: c.metadata.get("final_confidence", 0), reverse=True)

        deduped = []
        used_ranges = []

        for c in candidates:
            # Check if this candidate overlaps with an already-selected one
            is_dup = False
            for used_start, used_end in used_ranges:
                if c.start < used_end and c.end > used_start:
                    # Overlapping — check if same error
                    if c.original.lower() == deduped[used_ranges.index((used_start, used_end))].original.lower():
                        is_dup = True
                        break

            if not is_dup:
                deduped.append(c)
                used_ranges.append((c.start, c.end))

        return deduped

    def _validate_corrections(self, candidates: List[Candidate], text: str) -> List[Candidate]:
        """Validate that corrections actually fix the error."""
        validated = []
        for c in candidates:
            if self._correction_validator:
                is_valid, reason = self._correction_validator.validate(c, text)
                if is_valid:
                    validated.append(c)
                else:
                    c.metadata["validation_rejected"] = reason
            else:
                validated.append(c)
        return validated

    def _rank_and_convert(self, candidates: List[Candidate]) -> List[GrammarError]:
        """Rank candidates and convert to final GrammarError objects."""
        # Sort by confidence (highest first)
        candidates.sort(key=lambda c: c.metadata.get("final_confidence", 0), reverse=True)

        errors = []
        for c in candidates:
            conf = c.metadata.get("final_confidence", c.raw_confidence)
            if conf < self.config.min_confidence_to_show:
                continue

            # Determine severity from confidence
            if conf >= self.config.high_confidence_threshold:
                severity = "error"
            elif conf >= 0.80:
                severity = "warning"
            else:
                severity = "info"

            error = GrammarError(
                start=c.start,
                end=c.end,
                original=c.original,
                replacement=c.replacement,
                category=c.category,
                severity=severity,
                confidence=round(conf, 2),
                explanation=c.message,
                rule_id=c.rule_id,
                context=c.context,
                alternatives=c.metadata.get("alternatives", []),
            )
            errors.append(error)

        return errors


# ── Placeholder implementations (to be filled in Phases 3-9) ──────────

class CorrectionValidator:
    """Validates corrections by applying them and re-checking."""
    name: str = "correction_validator"

    def validate(self, candidate: Candidate, full_text: str) -> Tuple[bool, str]:
        """Apply correction and check if it's valid."""
        # Phase 8 will implement full validation
        return True, "placeholder"
