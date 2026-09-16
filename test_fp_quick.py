from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
tests = [
    'If I knew about the meeting, I would attend it.',
    'It is important that he attend the meeting.',
    'I went to the market and I bought some vegetables and then I came back home.',
    'She made a mistake because she was not careful enough.',
    'They was playing football when I called them.',
]
for t in tests:
    r = p.check(t)
    n = len(r)
    print(f'{n} issues: {t[:60]}')
    for c in r:
        print(f'  {c["rule_id"]}: {c["original"]} -> {c["replacement"]}')
    print()
