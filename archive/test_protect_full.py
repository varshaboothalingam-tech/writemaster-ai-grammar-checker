import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

text = """Last Monday, I was planning to go to office early because my manager have scheduled an important meeting with all the team members. I reached the bus stop at around eight o'clock, but there was too many peoples waiting for the bus. The bus finally arrived after twenty minutes, and when I entered inside, I realized that I had forgot my wallet at home. I tried calling my brother, but he didn't answered the phone because he was sleeping. By the time I reached the office, the meeting already started and everyone were waiting for me. My manager asked me why I was late, and I explained him what had happened. He was not very happy, but he said that I should be more careful in future. Yesterday, my friend and me decided to visit a new restaurant which recently opened near our house. We have heard many good reviews about the food, so we was expecting it to be very good. When we arrived at the restaurant, there were a long queue outside because the place was already crowded. After waiting for almost forty minutes, we finally got a table. The waiter gave us the menu and asked what would we like to order. My friend ordered a chicken biryani while I decided to try the special noodles. The food were served after twenty minutes, but the noodles was too spicy for me. We asked the waiter if he can bring some water, but he didn't came back for several minutes. In the end, we paid the bill and left the restaurant feeling a little disappointed."""

clean, prot = p.preprocessor.protect(text)
print(f"Original length: {len(text)}")
print(f"Cleaned length: {len(clean)}")
print(f"Protected spans: {len(prot)}")
for ps in prot:
    ctx = text[max(0,ps.start-10):min(len(text),ps.end+10)]
    print(f"  [{ps.start}:{ps.end}] '{ps.text}' cat={ps.category}")
    print(f"    context: ...{ctx}...")

# Check if text is mangled
if clean != text:
    # Find differences
    for i in range(min(len(clean), len(text))):
        if clean[i] != text[i]:
            print(f"\nFirst diff at pos {i}: original='{text[max(0,i-5):i+5]}' clean='{clean[max(0,i-5):i+5]}'")
            break
else:
    print("\nText is identical (no protection applied)")
