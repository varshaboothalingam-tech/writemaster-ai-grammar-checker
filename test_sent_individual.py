import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, NLPDetector

nlp = get_nlp()
det = NLPDetector()

tests = [
    "there are still several problems which needs to be fixed",
    "Some customers has reported that the application loads very slowly when they uploads large files",
    "The developers are currently working on solving these issues, but they said that it may takes another few weeks",
    "The marketing team have already prepared several articles and advertisements, so everybody are waiting for the final release",
]

for text in tests:
    doc = nlp(text)
    errs = det.detect(text, doc)
    print(f"Text: '{text[:60]}...'")
    for e in errs:
        print(f"  {e.rule_id}: '{e.original}' -> '{e.replacement}'")
    if not errs:
        print("  NO ERRORS FOUND!")
    print()
