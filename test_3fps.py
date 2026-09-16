import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
for t in ["She told me how much it cost.", "She wouldn't let anyone help."]:
    issues = p.check(t)
    print(f"'{t}'")
    for e in issues:
        print(f"  {e['rule_id']:30s} '{e['original']}' -> '{e['replacement']}' conf={e['confidence']:.0%}")
    print()
