import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

tests = [
    ("The food were served after twenty minutes.", "food were -> food was"),
    ("problems which needs fixing", "which needs -> which need"),
    ("for few weeks", "for few -> for a few"),
    ("the content are not enough", "content are -> content is"),
    ("search engines looks useful", "engines looks -> engines look"),
    ("competitors websites are important", "competitors websites -> competitors' websites"),
    ("gives useful information", "gives -> give"),
    ("they was celebrating", "they was -> they were"),
    ("we was expecting", "we was -> we were"),
    ("team have worked", "team have -> team has"),
    ("everybody are waiting", "everybody are -> everybody is"),
    ("noodles was too spicy", "noodles was -> noodles were"),
    ("he didn't had enough", "didn't had -> didn't have"),
    ("she had buy a phone", "had buy -> had bought"),
    ("to brings food", "to brings -> to bring"),
    ("most happiest day", "most happiest -> happiest"),
]

for t, expected in tests:
    r = p.check(t)
    if r:
        found = [f"{i['rule_id']}: '{i['original']}' -> '{i['replacement']}'" for i in r]
        print(f"  OK  '{t}' -> {found}")
    else:
        print(f"  MISS '{t}' (expected: {expected})")
