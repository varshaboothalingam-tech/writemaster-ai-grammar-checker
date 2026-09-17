import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector, deduplicate

p = UnifiedPipeline()
nlp = get_nlp()

t = "they was celebrating"
clean, _ = p.preprocessor.protect(t)
doc = nlp(clean)

nlp_c = p.nlp_detector.detect(clean, doc)
print(f"NLPDetector: {len(nlp_c)} candidates")
for c in nlp_c:
    print(f"  {c.rule_id}: '{c.original}' -> '{c.replacement}' conf={c.raw_confidence:.2f}")

# Context
ctx = []
for c in nlp_c:
    suppressed, reason = p.context_engine.should_suppress(c, clean, doc)
    conf = p.confidence_engine.score(c, (suppressed, reason))
    print(f"  Context: suppressed={suppressed} reason='{reason}' conf={conf:.2f}")
    if conf >= 0.70:
        c.raw_confidence = conf
        ctx.append(c)

# FP filter
fp = p.fp_filter.filter(ctx, clean, doc)
print(f"After FP filter: {len(fp)} candidates")
for c in fp:
    print(f"  {c.rule_id}: '{c.original}' -> '{c.replacement}'")

# Dedup
deduped = deduplicate(fp)
print(f"After dedup: {len(deduped)} candidates")

# Validate
validated = []
for c in deduped:
    is_valid, reason = p.validator.validate(c, clean)
    print(f"  Validate {c.rule_id} '{c.original}': valid={is_valid} reason='{reason}'")
    if is_valid:
        validated.append(c)

print(f"After validate: {len(validated)}")
