"""Check what dedup removes."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, deduplicate

p = UnifiedPipeline()
nlp = get_nlp()

text = """My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem.

When I ask her why she selling it, she tell me because the car was too old and it didn't started properly since two years. She also say that she was not satisfy with the car because the engine make too much noise and she didn't had enough money to fix it.

My cousin she was very upset because she have to spend lot of money on the car every month. She say that she working hard but the car keep break down and she can't afford it anymore. She was also worry about the safety because the brakes was not working properly and she have a big family who she need to drive to school every day.

She ask me if I know someone who want to buy a cheap car. I tell her that my friend have a brother who was looking for a car. She was very happy to hear that and she ask me to give her the brother number. I say I will call him and let her know.

My cousin she was very relief when I tell her that my friend brother was interested in buying her car. She say that she going to give him a good price because she want to get rid of it fast. She was also happy because she can finally buy a new car that she can rely on."""

clean_text, protected = p.preprocessor.protect(text)
doc = nlp(clean_text)
nlp_cands = p.nlp_detector.detect(clean_text, doc)
data_cands = p.data_detector.detect(clean_text, doc)
all_cands = nlp_cands + data_cands

# Context + FP
passed = []
for c in all_cands:
    suppressed, reason = p.context_engine.should_suppress(c, clean_text, doc)
    score = p.confidence_engine.score(c, (suppressed, reason))
    if score >= 0.70:
        c.raw_confidence = score
        passed.append(c)
passed = p.fp_filter.filter(passed, clean_text, doc)

# Show before dedup
print(f"Before dedup ({len(passed)}):")
for c in passed:
    print(f"  {c.rule_id:30s} pos={c.start:3d}-{c.end:3d} '{c.original}' -> '{c.replacement}'")

deduped = deduplicate(passed)
dedup_ids = {(e.rule_id, e.start) for e in deduped}
for c in passed:
    if (c.rule_id, c.start) not in dedup_ids:
        print(f"\n  DEDUPED: {c.rule_id:30s} pos={c.start:3d} '{c.original}' -> '{c.replacement}'")
