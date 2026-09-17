import sys
sys.path.insert(0, 'C:\\Users\\hemal\\OneDrive\\Documents\\grammar_checker')

from unified_pipeline import check_text, get_nlp

print("=" * 70)
print("TEST 1: Regression paragraph (should detect errors)")
print("=" * 70)
text1 = "Yesterday my friend and me goes to a large stopping mall because we waited to buy some new clothe. When we reach the mall, there was many people writing outside because the stops was not spend yet."
result1 = check_text(text1)
print(f"Issues found: {len(result1)}")
for r in result1:
    print(f"  [{r['category']}] '{r['original']}' -> '{r['replacement']}' (conf={r['confidence']})")
    print(f"    {r['explanation']}")

print()
print("=" * 70)
print("TEST 2: Clean text (should have 0 errors)")
print("=" * 70)
text2 = "Yesterday my friend and I went to the shopping mall. There were many people waiting outside because the shops were not open yet."
result2 = check_text(text2)
print(f"Issues found: {len(result2)}")
for r in result2:
    print(f"  [{r['category']}] '{r['original']}' -> '{r['replacement']}'")

print()
print("=" * 70)
print("TEST 3: BIG WRONG TEXT (should detect and correct many errors)")
print("=" * 70)
text3 = """I have allot of freind that recieve there diploma yesterday. The ceremony was very beautifull and everyone was so happy. My sister is a university student and she have been studing for four years. She definately deserve a celebration. 

The mall have many restaurants and alot of shops. We was planning to go there but the weather was to bad. I should of check the forecast before we left. 

Can you tell me how much these shoes cost? He asked what I wanted to buy. She told me that the informations was correct. 

The team have won there last three games. Everyone have their own opinion about the best player. The news are very disturbing today. 

I definately going to the store tomorrow. Me and him is going to the movies. The data shows that there is many problems with the system. 

If I was you I would of went to the store. The children was playing outside because they was happy. We should of bought more clothe for the winter."""

result3 = check_text(text3)
print(f"Issues found: {len(result3)}")
for r in result3:
    print(f"  [{r['category']}] '{r['original']}' -> '{r['replacement']}' (conf={r['confidence']})")
    print(f"    {r['explanation']}")

print()
print("=" * 70)
print("TEST 4: Adversarial correct sentences (should be 0 errors)")
print("=" * 70)
adversarial = [
    "I wanted to improve my writing.",
    "The list of items is ready.",
    "The team was many players short.",
    "The information was useful.",
    "I asked how much the shoes cost.",
    "He asked what I wanted.",
    "Can you tell me how much these shoes cost?",
    "The company was founded in 2010.",
    "Everyone has their own opinion.",
    "She is a university student.",
    "There is less progress.",
    "He's gonna be there.",
    "The students were writing outside while they waited for the teacher.",
    "I definitely wanted to improve my writing.",
]
total_errors = 0
for sent in adversarial:
    result = check_text(sent)
    if result:
        print(f"  FP: '{sent}'")
        for r in result:
            print(f"    [{r['category']}] '{r['original']}' -> '{r['replacement']}'")
        total_errors += len(result)
print(f"Total false positives: {total_errors} / {len(adversarial)} sentences")
