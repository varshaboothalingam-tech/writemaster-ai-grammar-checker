"""Full diagnostic of user text vs expected errors."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

# Full user text (all paragraphs)
full_text = open(r'C:\Users\hemal\OneDrive\Documents\grammar_checker\sample_texts\real_writing.txt', encoding='utf-8').read()

issues = p.check(full_text)
print(f"Total issues found: {len(issues)}")
print()
for i, e in enumerate(issues, 1):
    print(f"  {i:2d}. [{e['rule_id']:30s}] ({e['confidence']:.0%}) '{e['original']}' -> '{e['replacement']}'")
    print(f"      {e['explanation']}")
