import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

# Individual sentence checks for the remaining missed errors
tests = [
    ("the content are not enough", "SVA/content"),
    ("for few weeks you should optimize", "MISSING_ARTICLE/for few"),
    ("competitors websites are important", "POSSESSIVE/competitors"),
    ("search engines looks useful", "SVA/engines"),
    ("she had buy a new phone", "PAST_PARTICIPLE/had buy"),
    ("he can brings food", "BASE_FORM/brings"),
    ("they was celebrating", "SVA/they was"),
    ("everybody are waiting", "SVA/everybody are"),
    ("we was expecting it", "SVA/we was"),
    ("food were served", "SVA/food were"),
    ("noodles was too spicy", "SVA/noodles was"),
    ("he didn't had enough", "DIDNT_PAST/had"),
    ("problems which needs fixing", "SVA/which needs"),
    ("team have worked", "SVA/team have"),
    ("team have already prepared", "SVA/team have"),
]
for t, label in tests:
    r = p.check(t)
    found = [(i['rule_id'], i['original'], i['replacement']) for i in r]
    status = "OK" if found else "MISSED"
    print(f"  [{status:6s}] '{t}' -> {found if found else 'NONE'}")
