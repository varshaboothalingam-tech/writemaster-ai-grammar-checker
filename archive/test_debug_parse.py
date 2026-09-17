import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Ain't nobody got time for that.")
for t in doc:
    print(f'{t.text:10s} tag={t.tag_:4s} pos={t.pos_:5s} dep={t.dep_:8s} head={t.head.text}')
print()
doc2 = nlp('I have less money than last month.')
for t in doc2:
    print(f'{t.text:10s} tag={t.tag_:4s} pos={t.pos_:5s} dep={t.dep_:8s} head={t.head.text}')
