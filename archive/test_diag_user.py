"""Diagnose which detectors fire on user text sentences."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
tests = [
    'It have many problem.',
    'She also say that.',
    'I ask her why.',
    'She tell me because.',
    'It did not started properly.',
    'Since two years.',
    'My cousin she was telling me.',
    'The brakes was not working.',
    'She have to spend lot of money.',
    'She going to sell it.',
    'She selling it.',
    'They keep break down.',
    'The brother number.',
]
for t in tests:
    issues = p.check(t)
    ids = [i['rule_id'] for i in issues]
    rep = [f"{i['original']}->{i['replacement']}" for i in issues]
    result = ", ".join(rep) if rep else "NOTHING"
    print(f"  {t:45s} -> {result}")
