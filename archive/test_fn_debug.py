import requests
tests = [
    'Everyone have arrived for the meeting.',
    'Mathematics are hard for me.',
    'Neither of them are coming tonight.',
    'The group of students were studying hard.',
    'Five dollars are too much for that.',
    'The police is investigating the crime.',
    'We was at the cinema last night.',
    "I didn't saw him at the party.",
    'The cat sat on the mat it was comfortable.',
    'Yesterday I went to the store.',
    'However the results were inconclusive.',
    "The cat licked it's paw.",
    'Us went to the movies last night.',
    "She don't know the answer.",
    'I seen him yesterday at the park.',
    'She walked to the store and buys some milk.',
    'He is running yesterday morning.',
    'I was thinking and decide to leave.',
    'Because I was tired.',
    'Running through the park.',
    'Always she goes to school on time.',
    'I only eat vegetables.',
    'She is good in math and science.',
    'I listened the music carefully.',
    'He married with her last summer.',
]
for t in tests:
    r = requests.post('http://127.0.0.1:5001/api/check-v3', json={'text': t})
    issues = r.json().get('issues', [])
    if issues:
        print(f'OK: "{t}" -> {len(issues)} issues')
        for i in issues[:3]:
            print(f'   {i.get("rule","")} | {i.get("word","")} -> {i.get("suggestion","")}')
    else:
        print(f'MISS: "{t}"')
