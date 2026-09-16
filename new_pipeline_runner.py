"""
new_pipeline_runner.py — Connects the new grammar pipeline to the existing app.py.
This replaces the old WritingAnalyzer/GrammarChecker/AIEngine flow.
"""

from typing import List, Dict
from grammar_pipeline import GrammarPipeline, PipelineConfig, GrammarError
from spacy_context import SpaCyContextAnalyzer
from spacy_detectors import (
    SubjectVerbAgreementDetector,
    TenseConsistencyDetector,
    PronounCaseDetector,
    PossessiveDetector,
    ModalVerbDetector,
    CouldHaveDetector,
    UsedToDetector,
)
from extended_detectors import (
    SpellingDetector,
    WordUsageDetector,
    CompoundSubjectPronounDetector,
    ExistentialThereDetector,
    ArticleDetector,
    PrepositionDetector,
)
from fp_filter import SpaCyFalsePositiveFilter
from correction_validator import SpaCyCorrectionValidator


class NewGrammarRunner:
    """
    New grammar checking pipeline that replaces the old system.
    
    Flow:
      TEXT → CANDIDATE DETECTION (spaCy) → CONTEXT ANALYSIS →
      CONFIDENCE SCORING → FALSE-POSITIVE FILTER → DEDUPLICATION →
      CORRECTION VALIDATION → FINAL OUTPUT
    """

    def __init__(self):
        # Create pipeline config
        config = PipelineConfig(
            confidence_threshold=0.75,
            min_confidence_to_show=0.75,
            high_confidence_threshold=0.90,
            enable_correction_validation=True,
            enable_deduplication=True,
        )

        # Create pipeline
        self.pipeline = GrammarPipeline(config)

        # Set context analyzer
        context_analyzer = SpaCyContextAnalyzer()
        self.pipeline.set_context_analyzer(context_analyzer)

        # Register detectors
        self.pipeline.register_detector(SubjectVerbAgreementDetector())
        self.pipeline.register_detector(TenseConsistencyDetector())
        self.pipeline.register_detector(PronounCaseDetector())
        self.pipeline.register_detector(PossessiveDetector())
        self.pipeline.register_detector(ModalVerbDetector())
        self.pipeline.register_detector(CouldHaveDetector())
        self.pipeline.register_detector(UsedToDetector())
        # Extended detectors
        self.pipeline.register_detector(SpellingDetector())
        self.pipeline.register_detector(WordUsageDetector())
        self.pipeline.register_detector(CompoundSubjectPronounDetector())
        self.pipeline.register_detector(ExistentialThereDetector())
        self.pipeline.register_detector(ArticleDetector())
        self.pipeline.register_detector(PrepositionDetector())

        # Set false-positive filter
        self.pipeline.set_fp_filter(SpaCyFalsePositiveFilter())

        # Set correction validator
        self.pipeline.set_correction_validator(SpaCyCorrectionValidator())

    def check(self, text: str) -> List[Dict]:
        """
        Check text for grammar errors.
        Returns list of error dicts in the format expected by the frontend.
        """
        if not text or not text.strip():
            return []

        # Run the pipeline
        grammar_errors = self.pipeline.check(text)

        # Convert to frontend format
        return [self._to_frontend_format(e) for e in grammar_errors]

    def _to_frontend_format(self, error: GrammarError) -> Dict:
        """Convert GrammarError to frontend-compatible dict."""
        severity_map = {
            "error": "HIGH",
            "warning": "MEDIUM",
            "info": "LOW",
        }
        return {
            "original_text": error.original,
            "replacement": error.replacement,
            "category": error.category.upper(),
            "error_type": error.category,
            "message": error.explanation,
            "start_position": error.start,
            "end_position": error.end,
            "severity": severity_map.get(error.severity, "MEDIUM"),
            "confidence": error.confidence,
            "context": error.context,
            "explanation": error.explanation,
            "rule_id": error.rule_id,
            "alternatives": error.alternatives,
        }


# Singleton instance
_new_runner = None


def get_new_runner() -> NewGrammarRunner:
    """Get or create the singleton NewGrammarRunner."""
    global _new_runner
    if _new_runner is None:
        _new_runner = NewGrammarRunner()
    return _new_runner


def check_text_new(text: str) -> List[Dict]:
    """Convenience function to check text using the new pipeline."""
    runner = get_new_runner()
    return runner.check(text)
