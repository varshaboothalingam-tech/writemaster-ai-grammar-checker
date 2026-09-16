import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

# With contraction
issues1 = p.check("She wouldn't let anyone help.")
print("Contracted:", [(e['rule_id'], e['original'], e['replacement'], e['confidence']) for e in issues1])

# Without contraction
issues2 = p.check("She would not let anyone help.")
print("Uncontracted:", [(e['rule_id'], e['original'], e['replacement'], e['confidence']) for e in issues2])
