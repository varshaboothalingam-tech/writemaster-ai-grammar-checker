import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, NLPDetector, get_nlp
from unified_pipeline import ContextEngine, ConfidenceEngine

p = UnifiedPipeline()
nlp = get_nlp()
det = NLPDetector()
ctx = ContextEngine()
conf = ConfidenceEngine()

s = "The number of people who uses this application have increased significantly during the last few months."
doc = nlp(s)

nlp_errors = det.detect(s, doc)
print(f"NLP raw: {len(nlp_errors)} issues")

for e in nlp_errors:
    suppressed, reason = ctx.should_suppress(e, s, doc)
    confidence = conf.score(e, (suppressed, reason))
    print(f"  {e.rule_id} [{e.start}:{e.end}] '{e.original}' -> '{e.replacement}'")
    print(f"    suppressed={suppressed} reason={reason} confidence={confidence:.2f}")
