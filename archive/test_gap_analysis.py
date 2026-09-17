import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

text = """Last Monday, I was planning to go to office early because my manager have scheduled an important meeting with all the team members. I reached the bus stop at around eight o'clock, but there was too many peoples waiting for the bus. The bus finally arrived after twenty minutes, and when I entered inside, I realized that I had forgot my wallet at home. I tried calling my brother, but he didn't answered the phone because he was sleeping. By the time I reached the office, the meeting already started and everyone were waiting for me. My manager asked me why I was late, and I explained him what had happened. He was not very happy, but he said that I should be more careful in future. Yesterday, my friend and me decided to visit a new restaurant which recently opened near our house. We have heard many good reviews about the food, so we was expecting it to be very good. When we arrived at the restaurant, there were a long queue outside because the place was already crowded. After waiting for almost forty minutes, we finally got a table. The waiter gave us the menu and asked what would we like to order. My friend ordered a chicken biryani while I decided to try the special noodles. The food were served after twenty minutes, but the noodles was too spicy for me. We asked the waiter if he can bring some water, but he didn't came back for several minutes. In the end, we paid the bill and left the restaurant feeling a little disappointed."""

result = p.check(text)
found_map = {}
for issue in result:
    key = issue['original'].lower() + '|' + issue['replacement'].lower()
    found_map[key] = issue['rule_id']

# Expected errors with expected original/replacement pairs
expected = [
    ("have", "has", "SVA", "Paragraph 1: manager have"),
    ("was", "were", "SVA/EXISTENTIAL", "Paragraph 1: there was"),
    ("peoples", "people", "IRREGULAR_PLURAL", "Paragraph 1: too many peoples"),
    ("entered inside", "entered", "REDUNDANT_WORD", "Paragraph 1: entered inside"),
    ("had forgot", "had forgotten", "PAST_PARTICIPLE", "Paragraph 1: had forgot"),
    ("answered", "answer", "DIDNT_PAST_FORM", "Paragraph 1: didn't answered"),
    ("were", "was", "SVA", "Paragraph 1: everyone were"),
    ("explained him", "explained to him", "EXPLAIN_DATIVE", "Paragraph 1: explained him"),
    ("in future", "in the future", "MISSING_ARTICLE", "Paragraph 1: in future"),
    ("me", "I", "COMPOUND_SUBJECT", "Paragraph 2: friend and me"),
    ("was", "were", "SVA", "Paragraph 2: we was expecting"),
    ("were", "was", "SVA", "Paragraph 2: there were a queue"),
    ("was", "were", "SVA", "Paragraph 2: noodles was"),
    ("food were", "food was", "SVA", "Paragraph 2: food were"),
    ("came", "come", "DIDNT_PAST_FORM", "Paragraph 2: didn't came"),
    ("fixed", "fix", "TO_PAST_PARTICIPLE", "Paragraph 3: to be fixed"),
    ("needs", "need", "SVA", "Paragraph 3: problems which needs"),
    ("has", "have", "SVA", "Paragraph 3: customers has"),
    ("uploads", "upload", "SVA", "Paragraph 3: they uploads"),
    ("takes", "take", "MODAL_WRONG_FORM", "Paragraph 3: may takes"),
    ("are", "is", "SVA", "Paragraph 3: everybody are"),
    ("becomes", "become", "MODAL_WRONG_FORM", "Paragraph 3: will becomes"),
    ("have", "has", "SVA", "Paragraph 3: team have"),
    ("have", "has", "SVA", "Paragraph 3: team have already"),
    ("grandparents'", "grandparents'", "POSSESSIVE", "Paragraph 4: grandparents house"),
    ("was", "were", "SVA", "Paragraph 4: they was"),
    ("were", "was", "SVA", "Paragraph 4: everyone were helping"),
    ("brings", "bring", "BASE_FORM", "Paragraph 4: to brings"),
    ("had", "have", "DIDNT_PAST_FORM", "Paragraph 4: didn't had"),
    ("buy", "bought", "PAST_PARTICIPLE", "Paragraph 4: had buy"),
    ("most happiest", "happiest", "DOUBLE_SUPERLATIVE", "Paragraph 4: most happiest"),
    ("learning", "to learn", "PARALLELISM", "Paragraph 5: started learning"),
    ("for few", "for a few", "MISSING_ARTICLE", "Paragraph 5: for few weeks"),
    ("content are", "content is", "SVA", "Paragraph 5: content are"),
    ("looks", "look", "SVA", "Paragraph 5: engines looks"),
    ("has", "have", "SVA", "Paragraph 5: pages has"),
    ("needs", "need", "SVA", "Paragraph 5: factors needs"),
    ("performs", "perform", "SVA", "Paragraph 5: pages performs"),
    ("spends", "spend", "SVA", "Paragraph 5: spends"),
    ("understands", "understand", "SVA", "Paragraph 5: understands"),
]

found_count = 0
missed_count = 0
for orig, repl, rule, desc in expected:
    key = orig.lower() + '|' + repl.lower()
    if key in found_map:
        found_count += 1
    else:
        # Try partial match
        found = False
        for fkey, frule in found_map.items():
            if orig.lower() in fkey.split('|')[0]:
                found_count += 1
                found = True
                break
        if not found:
            missed_count += 1
            print(f"  MISSED: {rule} '{orig}' -> '{repl}' - {desc}")

print(f"\n=== SUMMARY ===")
print(f"Found: {found_count}/{len(expected)}")
print(f"Missed: {missed_count}/{len(expected)}")
print(f"Detected in pipeline: {len(result)} total issues")
