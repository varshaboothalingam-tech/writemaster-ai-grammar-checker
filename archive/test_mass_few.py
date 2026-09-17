import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp

nlp = get_nlp()

# Check "content are" parse
text1 = "the content are not enough because search engines looks useful information on the website"
doc1 = nlp(text1)
print("=== 'content are' ===")
for t in doc1:
    print(f"  {t.text:15s} tag={t.tag_:5s} dep={t.dep_:10s} head={t.head.text}")

# Check "for few weeks" parse  
text2 = "You should optimize your pages for few weeks"
doc2 = nlp(text2)
print("\n=== 'for few weeks' ===")
for t in doc2:
    print(f"  {t.text:15s} tag={t.tag_:5s} dep={t.dep_:10s} head={t.head.text}")
