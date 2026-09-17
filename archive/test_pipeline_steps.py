import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector

p = UnifiedPipeline()
nlp = get_nlp()

# Read the exact same text as test_user_writing_full.py
text = """Last Monday, I was planning to go to office early because my manager have scheduled an important meeting with all the team members. I reached the bus stop at around eight o'clock, but there was too many peoples waiting for the bus. The bus finally arrived after twenty minutes, and when I entered inside, I realized that I had forgot my wallet at home. I tried calling my brother, but he didn't answered the phone because he was sleeping. By the time I reached the office, the meeting already started and everyone were waiting for me. My manager asked me why I was late, and I explained him what had happened. He was not very happy, but he said that I should be more careful in future. Yesterday, my friend and me decided to visit a new restaurant which recently opened near our house. We have heard many good reviews about the food, so we was expecting it to be very good. When we arrived at the restaurant, there were a long queue outside because the place was already crowded. After waiting for almost forty minutes, we finally got a table. The waiter gave us the menu and asked what would we like to order. My friend ordered a chicken biryani while I decided to try the special noodles. The food were served after twenty minutes, but the noodles was too spicy for me. We asked the waiter if he can bring some water, but he didn't came back for several minutes. In the end, we paid the bill and left the restaurant feeling a little disappointed. Our company has recently launched a new software product which is designed for helping small businesses manage their customer data. The development team have worked on it since almost one year, and they has spent many hours testing different features. Although the product looks very promising, there are still several problems which needs to be fixed before it can be released to the public. Some customers has reported that the application loads very slowly when they uploads large files. The developers are currently working on solving these issues, but they said that it may takes another few weeks. The marketing team have already prepared several articles and advertisements, so everybody are waiting for the final release. If the technical problems will be solved soon, the company expects that the product will becomes very popular among small businesses."""

issues = p.check(text)
print(f"=== DETECTED {len(issues)} ISSUES ===")
for e in sorted(issues, key=lambda x: x['start']):
    print(f"  [{e['start']:4d}] {e['rule_id']:25s} '{e['original']}' -> '{e['replacement']}' conf={e['confidence']:.2f}")

# Now check individual sentences vs full text
clean, prot = p.preprocessor.protect(text)
doc = nlp(clean)
det = NLPDetector()
candidates = det.detect(clean, doc)
print(f"\nNLPDetector candidates: {len(candidates)}")
for c in candidates:
    print(f"  [{c.start:4d}:{c.end:4d}] {c.rule_id}: '{c.original}' -> '{c.replacement}' conf={c.raw_confidence:.2f}")
