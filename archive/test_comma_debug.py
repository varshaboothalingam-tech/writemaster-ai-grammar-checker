import spacy
nlp = spacy.load('en_core_web_sm')

part1 = "She was tired"
part2 = "she went to bed."

doc1 = nlp(part1)
doc2 = nlp(part2)

print("Part1 tokens:")
for t in doc1:
    print(f"  {t.text} dep={t.dep_} pos={t.pos_}")

print("\nPart2 tokens:")
for t in doc2:
    print(f"  {t.text} dep={t.dep_} pos={t.pos_}")

# Check what the detector sees
has_subj1 = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc1)
has_subj2 = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc2)
has_verb1 = any(t.pos_ in ("VERB", "AUX") for t in doc1)
has_verb2 = any(t.pos_ in ("VERB", "AUX") for t in doc2)
first_has_subord = any(t.dep_ in ("advcl", "mark", "relcl") or t.tag_ in ("WRB", "WP", "WDT") for t in doc1)

print(f"\nhas_subj1={has_subj1} has_verb1={has_verb1}")
print(f"has_subj2={has_subj2} has_verb2={has_verb2}")
print(f"first_has_subord={first_has_subord}")
print(f"part2[0].isupper()={part2[0].isupper()}")
