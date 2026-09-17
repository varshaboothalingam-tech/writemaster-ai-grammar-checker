import sys, json
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
s = "A number of students are absent today."
errors = p.check(s)
print(f"Errors: {len(errors)}")
for e in errors:
    print(json.dumps(e, indent=2))
