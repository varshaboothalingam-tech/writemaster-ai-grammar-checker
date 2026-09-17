import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, NLPDetector, get_nlp, ContextEngine, ConfidenceEngine

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()
ctx = ContextEngine()
conf = ConfidenceEngine()

tests = [
    (46, "I didn't expected the meeting to take so long, so I haven't brought anything to eat."),
    (47, "My sister bought a new laptop because her old one was becoming slower and slower every days."),
]

for num, s in tests:
    clean_text, protected = p.preprocessor.protect(s)
    doc = nlp(clean_text)
    nlp_errors = det.detect(clean_text, doc)
    print(f"S{num}:")
    for e in nlp_errors:
        suppressed, reason = ctx.should_suppress(e, clean_text, doc)
        confidence = conf.score(e, (suppressed, reason))
        is_fp = p.fp_filter._is_false_positive(e, clean_text, doc)
        print(f"  {e.rule_id} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")
        print(f"    suppressed={suppressed} reason={reason} conf={confidence:.2f} is_fp={is_fp}")
    print()
