import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector, ContextEngine, ConfidenceEngine, FalsePositiveFilter, CorrectionValidator

nlp = get_nlp()
det = NLPDetector()
ctx = ContextEngine()
conf = ConfidenceEngine()
fpf = FalsePositiveFilter()
val = CorrectionValidator()

s1 = "My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem."
s2 = "When I ask her why she selling it, she tell me because the car was too boring and she dont like it."

for i, s in enumerate([s1, s2], 1):
    doc = nlp(s)
    raw = det.detect(s, doc)
    print(f"Sentence {i}:")
    for c in raw:
        suppressed, reason = ctx.should_suppress(c, s, doc)
        score = conf.score(c, (suppressed, reason))
        passed_ctx = score >= 0.70
        passed_fp = c in fpf.filter(raw, s, doc)
        is_valid, val_reason = val.validate(c, s)
        print(f"  {c.rule_id:30s} '{c.original}'->{c.replacement:10s} ctx_sup={suppressed} score={score:.2f} passed_ctx={passed_ctx} valid={is_valid} val_reason={val_reason}")
    print()
