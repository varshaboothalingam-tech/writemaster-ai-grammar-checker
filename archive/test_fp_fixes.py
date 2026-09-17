import sys
sys.path.insert(0, '.')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
tests = [
    'I was on cloud nine.',
    'Neither the teacher nor the students were ready.',
    'Neither the manager nor the employees were aware of the change.',
    'Whatever you decide, I will support you.',
    'Wherever you go, I will follow.',
    'If it were not for your help, I would not have succeeded.',
    'Were I in your position, I would accept the offer.',
    'Much as I admire her courage, I cannot support her decision.',
    "They was going to the store.",
    "We was just talking.",
    "Ain't nobody got time for that.",
    'I have less money than last month.',
    'She has less knowledge.',
    'Everyone have their own opinion.',
    'The students were writing outside while they waited for the teacher.',
]
for t in tests:
    result = p.check(t)
    if result:
        print(f'  FP: "{t}" => {[(e["original"], e["replacement"], e["rule_id"]) for e in result]}')
    else:
        print(f'  OK: "{t}"')
