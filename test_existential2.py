import sys
sys.path.insert(0, 'C:\\Users\\hemal\\OneDrive\\Documents\\grammar_checker')
from grammar_pipeline import GrammarPipeline, PipelineConfig, Candidate
from spacy_context import SpaCyContextAnalyzer
from extended_detectors import ExistentialThereDetector
from fp_filter import SpaCyFalsePositiveFilter
from correction_validator import SpaCyCorrectionValidator

config = PipelineConfig(confidence_threshold=0.75)
pipeline = GrammarPipeline(config)
pipeline.set_context_analyzer(SpaCyContextAnalyzer())
pipeline.register_detector(ExistentialThereDetector())

# Step 1: Get raw candidates
text = 'there was many people writing outside'
candidates = pipeline._detect_candidates(text, None)
print(f'Raw candidates: {len(candidates)}')
for c in candidates:
    print(f'  {c.original} -> {c.replacement} [{c.category}] conf={c.raw_confidence:.2f}')

# Step 2: Filter
fp_filter = SpaCyFalsePositiveFilter()
filtered = fp_filter.filter(candidates, text)
print(f'After FP filter: {len(filtered)}')
for c in filtered:
    print(f'  {c.original} -> {c.replacement} [{c.category}]')

# Step 3: Correction validator
validator = SpaCyCorrectionValidator()
for c in filtered:
    valid = validator.validate(c, text)
    print(f'  Validation: {c.original} -> {c.replacement} valid={valid}')
