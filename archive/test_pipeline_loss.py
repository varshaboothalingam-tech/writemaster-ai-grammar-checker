import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()

tests = [
    "they was celebrating",
    "we was expecting",
    "team have worked on it",
    "The food were served after twenty minutes.",
    "problems which needs fixing",
    "competitors websites are important",
    "gives useful information to people",
]
for t in tests:
    doc = nlp(t)
    errs = det.detect(t, doc)
    pipeline = p.check(t)
    
    nlp_summary = [(e.rule_id, e.original, e.replacement) for e in errs]
    pipe_summary = [(i['rule_id'], i['original'], i['replacement']) for i in pipeline]
    
    print(f"\n=== '{t}' ===")
    print(f"  NLPDetector: {nlp_summary}")
    print(f"  Pipeline:    {pipe_summary}")
    
    if errs and not pipeline:
        clean, prot = p.preprocessor.protect(t)
        doc2 = nlp(clean)
        fast = p.fast_detector.detect(clean)
        nlp_c = p.nlp_detector.detect(clean, doc2)
        data_c = p.data_detector.detect(clean, doc2)
        print(f"  Preprocessed same as original: {clean == t}")
        
        for c in nlp_c:
            suppressed, reason = p.context_engine.should_suppress(c, clean, doc2)
            conf = p.confidence_engine.score(c, (suppressed, reason))
            print(f"  {c.rule_id} '{c.original}': suppressed={suppressed}, reason={reason}, conf={conf:.2f}")
