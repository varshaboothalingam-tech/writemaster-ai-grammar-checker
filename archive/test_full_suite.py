import sys
sys.path.insert(0, 'C:\\Users\\hemal\\OneDrive\\Documents\\grammar_checker')
from unified_pipeline import check_text

# Adversarial correct sentences — should produce ZERO errors
correct_sentences = [
    "I wanted to improve my writing.",
    "The list of items is ready.",
    "The team was many players short.",
    "The information was useful.",
    "I asked how much the shoes cost.",
    "He asked what I wanted.",
    "Can you tell me how much these shoes cost?",
    "She is a university student.",
    "Everyone has their own opinion.",
    "The students were writing outside while they waited for the teacher.",
    "Yesterday my friend and I went to the shopping mall. There were many people waiting outside because the shops were not open yet.",
]

print("=== ADVERSARIAL CORRECT (should be 0) ===")
total_fp = 0
for text in correct_sentences:
    r = check_text(text)
    if r:
        total_fp += len(r)
        print(f'  FP: "{text}"')
        for x in r:
            print(f'    {x["original"]} -> {x["replacement"]} [{x["category"]}]')
    else:
        print(f'  OK: "{text}"')
print(f'  Total FPs: {total_fp}')
print()

# Big wrong text
wrong_text = """Yesterday my friend and me goes to a large stopping mall because we waited to buy some new clothe. When we reach the mall, there was many people writing outside because the stops was not spend yet. She have many friends. The team have won. The news are very disturbing. The weather was to bad. I would of went but I didnt had no time. He dont know nothing about it. Me and him is going to the store tommorrow. He runned fastly. She definately going. I should of studied more. We goes to school everyday. The childs was playing outside. He seen the accident yesterday. I aint got no money. She buyed a new dress. The informations was correct. We was going to the park. He dont have no friends. She have to leave now."""

print("=== BIG WRONG TEXT (should find many) ===")
r = check_text(wrong_text)
print(f'  Found {len(r)} issues')
for x in r:
    print(f'  [{x["start"]}:{x["end"]}] "{x["original"]}" -> "{x["replacement"]}" [{x["category"]}] conf={x["confidence"]}')
print()

# Regression paragraph
regression = "Yesterday my friend and me goes to a large stopping mall because we waited to buy some new clothe. When we reach the mall, there was many people writing outside because the stops was not spend yet."
print("=== REGRESSION PARAGRAPH ===")
r = check_text(regression)
print(f'  Found {len(r)} issues')
for x in r:
    print(f'  "{x["original"]}" -> "{x["replacement"]}" [{x["category"]}]')
print()
