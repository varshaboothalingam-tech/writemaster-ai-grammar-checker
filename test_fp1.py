import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
issues = p.check("Only after the meeting did I realize what had happened.")
for i in issues:
    print(f"  {i['rule_id']:35s} '{i['original']}' -> '{i['replacement']}' conf={i['confidence']:.0%}")
