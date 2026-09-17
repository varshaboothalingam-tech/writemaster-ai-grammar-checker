import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector
nlp = get_nlp()
det = NLPDetector()

full = """Our company has recently launched a new software product which is designed for helping small businesses manage their customer data. The development team have worked on it since almost one year, and they has spent many hours testing different features. Although the product looks very promising, there are still several problems which needs to be fixed before it can be released to the public. Some customers has reported that the application loads very slowly when they uploads large files. The developers are currently working on solving these issues, but they said that it may takes another few weeks. The marketing team have already prepared several articles and advertisements, so everybody are waiting for the final release. If the technical problems will be solved soon, the company expects that the product will becomes very popular among small businesses."""

doc = nlp(full)
errors = det.detect(full, doc)
print(f"Full paragraph: {len(errors)} errors")
for e in errors:
    ctx = full[max(0,e.start-20):min(len(full),e.end+20)].replace('\n',' ')
    print(f"  {e.rule_id:25s} [{e.start:4d}:{e.end:4d}] '{e.original}' -> '{e.replacement}' conf={e.raw_confidence:.2f}")
    print(f"         ...{ctx}...")

# Compare: same text but split into sentences
import re
sents = re.split(r'(?<=[.!?])\s+', full)
print(f"\nIndividual sentences: ", end="")
total = 0
for s in sents:
    doc2 = nlp(s)
    errs = det.detect(s, doc2)
    total += len(errs)
    for e in errs:
        print(f"\n  {e.rule_id:25s} '{e.original}' -> '{e.replacement}'")
        # Find which sentence
        idx = full.find(s[:40])
        print(f"    in sentence starting at {idx}: '{s[:60]}...'")
print(f"\n  Total individual: {total}")
