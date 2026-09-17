import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

tests = [
    "I have many works to complete today",
    "receiving many positive feedbacks from customers",
    "Every students in the class were asked",
    "She works very hardly",
    "he didn't knew how to explain",
    "he didn't understood the instructions",
    "I didn't expected the meeting",
    "we had to waited for another one",
    "he had to called my colleague",
    "I had forget my keys",
]
for t in tests:
    issues = p.check(t)
    rules = [e['rule_id'] for e in issues]
    print(f"  {str(rules):40s} '{t[:60]}'")
