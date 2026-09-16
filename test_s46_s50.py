import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

s = "I didn't expected the meeting to take so long"
issues = p.check(s)
print(f"Contracted: {[(e['rule_id'], e['original'], e['replacement']) for e in issues]}")

s2 = "I did not expected the meeting to take so long"
issues2 = p.check(s2)
print(f"Uncontracted: {[(e['rule_id'], e['original'], e['replacement']) for e in issues2]}")

# Full sentence 46
s3 = "I didn't expected the meeting to take so long, so I haven't brought anything to eat."
issues3 = p.check(s3)
print(f"Full S46: {[(e['rule_id'], e['original'], e['replacement']) for e in issues3]}")

# Full sentence 50
s4 = "When I was reaching home yesterday, I realized that I had forget my keys at the office and had to called my colleague for help."
issues4 = p.check(s4)
print(f"Full S50: {[(e['rule_id'], e['original'], e['replacement']) for e in issues4]}")
