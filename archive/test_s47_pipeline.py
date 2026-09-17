import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector, ContextEngine, ConfidenceEngine

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()
ctx = ContextEngine()
conf = ConfidenceEngine()

s = "My sister bought a new laptop because her old one was becoming slower and slower every days."
clean_text, protected = p.preprocessor.protect(s)
doc = nlp(clean_text)
candidates = det.detect(clean_text, doc)
print(f"Step 1 - NLP raw: {len(candidates)}")

for c in candidates:
    suppressed, reason = ctx.should_suppress(c, clean_text, doc)
    confidence = conf.score(c, (suppressed, reason))
    print(f"  {c.rule_id} suppressed={suppressed} reason='{reason}' conf={confidence:.2f}")
    if confidence < 0.70:
        print(f"    FILTERED by confidence < 0.70")
    else:
        c.raw_confidence = confidence

context_filtered = [c for c in candidates if conf.score(c, ctx.should_suppress(c, clean_text, doc)) >= 0.70]
print(f"Step 6 - After context: {len(context_filtered)}")

fp_filtered = p.fp_filter.filter(context_filtered, clean_text, doc)
print(f"Step 7 - After FP: {len(fp_filtered)}")

from unified_pipeline import deduplicate
deduped = deduplicate(fp_filtered)
print(f"Step 8 - After dedup: {len(deduped)}")

for c in deduped:
    is_valid, reason = p.validator.validate(c, clean_text)
    print(f"  {c.rule_id} valid={is_valid} reason={reason}")
    if not is_valid:
        print(f"    FILTERED by validator")
