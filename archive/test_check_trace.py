"""Trace the actual check() method steps."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp

p = UnifiedPipeline()
nlp = get_nlp()

text = """My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem.

When I ask her why she selling it, she tell me because the car was too old and it didn't started properly since two years. She also say that she was not satisfy with the car because the engine make too much noise and she didn't had enough money to fix it.

My cousin she was very upset because she have to spend lot of money on the car every month. She say that she working hard but the car keep break down and she can't afford it anymore. She was also worry about the safety because the brakes was not working properly and she have a big family who she need to drive to school every day.

She ask me if I know someone who want to buy a cheap car. I tell her that my friend have a brother who was looking for a car. She was very happy to hear that and she ask me to give her the brother number. I say I will call him and let her know.

My cousin she was very relief when I tell her that my friend brother was interested in buying her car. She say that she going to give him a good price because she want to get rid of it fast. She was also happy because she can finally buy a new car that she can rely on."""

# Step by step
clean_text, protected = p.preprocessor.protect(text)
print(f"Original len: {len(text)}, Clean len: {len(clean_text)}")

doc = nlp(clean_text)
fast_cands = p.fast_detector.detect(clean_text)
nlp_cands = p.nlp_detector.detect(clean_text, doc)
data_cands = p.data_detector.detect(clean_text, doc)
print(f"Fast: {len(fast_cands)}, NLP: {len(nlp_cands)}, Data: {len(data_cands)}")

all_cands = fast_cands + nlp_cands + data_cands
print(f"Total raw: {len(all_cands)}")

# Context
ctx_passed = []
for c in all_cands:
    suppressed, reason = p.context_engine.should_suppress(c, clean_text, doc)
    score = p.confidence_engine.score(c, (suppressed, reason))
    if score >= 0.70:
        c.raw_confidence = score
        ctx_passed.append(c)
print(f"After context: {len(ctx_passed)}")

# FP filter
fp_passed = p.fp_filter.filter(ctx_passed, clean_text, doc)
print(f"After FP filter: {len(fp_passed)}")

# Dedup
from unified_pipeline import deduplicate
deduped = deduplicate(fp_passed)
print(f"After dedup: {len(deduped)}")

# Validate
validated = []
for c in deduped:
    is_valid, reason = p.validator.validate(c, clean_text)
    if is_valid:
        validated.append(c)
    else:
        print(f"  DROPPED by validator: {c.rule_id} '{c.original}'->{c.replacement} reason={reason}")
print(f"After validate: {len(validated)}")
