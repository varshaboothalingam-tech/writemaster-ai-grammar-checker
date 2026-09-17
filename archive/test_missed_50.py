import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

missed = [
    "I prefer working from home than travelling to the office because it saves more time.",
    "He didn't told anyone about the problem because he was afraid that his manager might gets angry.",
    "The new employees needs to complete their training before they can starts working on real projects.",
    "I wish I can speak English more fluently because it would helps me communicate with international clients.",
    "He told me that he didn't understood the instructions because they were written in a very complicated way.",
]
for s in missed:
    r = p.check(s)
    if r:
        for i in r:
            print(f"  OK: '{s[:50]}...' -> {i['rule_id']}: {i['original']}->{i['replacement']}")
    else:
        print(f"  MISS: '{s[:50]}...'")
