import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
s = "Yesterday my friend and me goes to a large stopping mall because we waited to buy some new clothe for the winter season. The mall have a lot of shops that sells different kind of things, and we was very exciting to go inside."
clean, prot = p.preprocessor.protect(s)
print(f"Original: '{s}'")
print(f"Cleaned:  '{clean}'")
print(f"Protected: {len(prot)} spans")
for ps in prot:
    print(f"  [{ps.start}:{ps.end}] '{ps.text}' cat={ps.category}")
