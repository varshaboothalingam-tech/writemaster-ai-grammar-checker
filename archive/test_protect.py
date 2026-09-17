import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import get_nlp, UnifiedPipeline

p = UnifiedPipeline()
nlp = get_nlp()

s46 = "I didn't expected the meeting to take so long, so I haven't brought anything to eat."
clean46, prot46 = p.preprocessor.protect(s46)
print(f"S46 original: '{s46}'")
print(f"S46 cleaned:  '{clean46}'")
doc46 = nlp(clean46)
for ent in doc46.ents:
    print(f"  ENT: '{ent.text}' [{ent.start_char}:{ent.end_char}] label={ent.label_}")

s47 = "My sister bought a new laptop because her old one was becoming slower and slower every days."
doc47 = nlp(s47)
print(f"\nS47 parse:")
for tok in doc47:
    print(f"  {tok.text:12s} tag={tok.tag_:4s} pos={tok.pos_:6s} dep={tok.dep_}")
for ent in doc47.ents:
    print(f"  ENT: '{ent.text}' [{ent.start_char}:{ent.end_char}] label={ent.label_}")
