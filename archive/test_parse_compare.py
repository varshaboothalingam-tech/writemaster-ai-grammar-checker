import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp
nlp = get_nlp()

# Compare parse: individual vs full text
individual = "Some customers has reported that the application loads very slowly."
full = """Last Monday, I was planning to go to office early because my manager have scheduled an important meeting with all the team members. I reached the bus stop at around eight o'clock, but there was too many peoples waiting for the bus. The bus finally arrived after twenty minutes, and when I entered inside, I realized that I had forgot my wallet at home. I tried calling my brother, but he didn't answered the phone because he was sleeping. By the time I reached the office, the meeting already started and everyone were waiting for me. My manager asked me why I was late, and I explained him what had happened. He was not very happy, but he said that I should be more careful in future. Yesterday, my friend and me decided to visit a new restaurant which recently opened near our house. We have heard many good reviews about the food, so we was expecting it to be very good. When we arrived at the restaurant, there were a long queue outside because the place was already crowded. After waiting for almost forty minutes, we finally got a table. The waiter gave us the menu and asked what would we like to order. My friend ordered a chicken biryani while I decided to try the special noodles. The food were served after twenty minutes, but the noodles was too spicy for me. We asked the waiter if he can bring some water, but he didn't came back for several minutes. In the end, we paid the bill and left the restaurant feeling a little disappointed. Our company has recently launched a new software product which is designed for helping small businesses manage their customer data. The development team have worked on it since almost one year, and they has spent many hours testing different features. Although the product looks very promising, there are still several problems which needs to be fixed before it can be released to the public. Some customers has reported that the application loads very slowly when they uploads large files."""

print("=== INDIVIDUAL PARSE ===")
doc = nlp(individual)
for sent in doc.sents:
    print(f"  Sentence: '{sent.text}'")
    for tok in sent:
        if tok.dep_ in ("nsubj", "nsubjpass"):
            verb = tok.head
            print(f"    nsubj={tok.text}({tok.tag_}) -> verb={verb.text}({verb.tag_}) dep={verb.dep_}")

print("\n=== FULL TEXT PARSE (looking for 'customers has') ===")
doc2 = nlp(full)
for sent in doc2.sents:
    if "customers" in sent.text:
        print(f"  Sentence: '{sent.text[:100]}...'")
        for tok in sent:
            print(f"    {tok.text:15s} tag={tok.tag_:4s} dep={tok.dep_:12s} head={tok.head.text}")
        break
