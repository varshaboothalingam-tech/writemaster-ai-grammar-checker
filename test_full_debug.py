"""Check each sentence individually in the full text."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector, ContextEngine, ConfidenceEngine, FalsePositiveFilter
nlp = get_nlp()
det = NLPDetector()
ctx = ContextEngine()
conf = ConfidenceEngine()
fpf = FalsePositiveFilter()

text = """My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem.

When I ask her why she selling it, she tell me because the car was too old and it didn't started properly since two years. She also say that she was not satisfy with the car because the engine make too much noise and she didn't had enough money to fix it.

My cousin she was very upset because she have to spend lot of money on the car every month. She say that she working hard but the car keep break down and she can't afford it anymore. She was also worry about the safety because the brakes was not working properly and she have a big family who she need to drive to school every day.

She ask me if I know someone who want to buy a cheap car. I tell her that my friend have a brother who was looking for a car. She was very happy to hear that and she ask me to give her the brother number. I say I will call him and let her know.

My cousin she was very relief when I tell her that my friend brother was interested in buying her car. She say that she going to give him a good price because she want to get rid of it fast. She was also happy because she can finally buy a new car that she can rely on."""

doc = nlp(text)
raw = det.detect(text, doc)
# Filter through context + confidence
passed = []
for c in raw:
    suppressed, reason = ctx.should_suppress(c, text, doc)
    score = conf.score(c, (suppressed, reason))
    if score >= 0.70:
        passed.append(c)
    else:
        print(f"  SUPPRESSED: {c.rule_id:30s} pos={c.start:3d} '{c.original}' -> '{c.replacement}' reason='{reason}' score={score:.2f}")

filtered = fpf.filter(passed, text, doc)
filtered_ids = {(e.rule_id, e.start) for e in filtered}
for c in passed:
    if (c.rule_id, c.start) not in filtered_ids:
        print(f"  FP-FILTERED: {c.rule_id:30s} pos={c.start:3d} '{c.original}' -> '{c.replacement}'")

print(f"\n  Raw: {len(raw)}, Passed ctx: {len(passed)}, Passed FP: {len(filtered)}")
