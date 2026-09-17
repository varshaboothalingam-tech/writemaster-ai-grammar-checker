import sys
sys.path.insert(0, 'C:\\Users\\hemal\\OneDrive\\Documents\\grammar_checker')
import spacy
nlp = spacy.load('en_core_web_sm')
from grammar_pipeline import GrammarPipeline, PipelineConfig, Candidate
from spacy_context import SpaCyContextAnalyzer
from extended_detectors import ExistentialThereDetector
from fp_filter import SpaCyFalsePositiveFilter
from correction_validator import SpaCyCorrectionValidator

text = 'there was many people writing outside'
config = PipelineConfig(confidence_threshold=0.75)
pipeline = GrammarPipeline(config)
pipeline.set_context_analyzer(SpaCyContextAnalyzer())
pipeline.register_detector(ExistentialThereDetector())
pipeline.set_fp_filter(SpaCyFalsePositiveFilter())
pipeline.set_correction_validator(SpaCyCorrectionValidator())

# Manually trace
detector = ExistentialThereDetector()
candidates = detector.detect(text, None)
print(f'Detector produced {len(candidates)} candidates')
for c in candidates:
    print(f'  {c.original} -> {c.replacement} [{c.category}] conf={c.raw_confidence}')

# FP filter
fp = SpaCyFalsePositiveFilter()
filtered = fp.filter(candidates, text)
print(f'After FP filter: {len(filtered)}')

# Correction validator
validator = SpaCyCorrectionValidator()
for c in filtered:
    valid, reason = validator.validate(c, text)
    print(f'  Validation for "{c.original}" -> "{c.replacement}": valid={valid}, reason={reason}')
    # Debug: show corrected parse
    corrected = text[:c.start] + c.replacement + text[c.end:]
    print(f'  Corrected: "{corrected}"')
    doc = nlp(corrected)
    for t in doc:
        print(f'    {t.text} tag={t.tag_} dep={t.dep_} head={t.head.text}')
