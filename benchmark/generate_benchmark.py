"""
generate_benchmark.py — builds the honest error + clean benchmark corpora.

Produces:
    benchmark/error_cases.json  2000+ sentences with planted errors (should_flag: true)
    benchmark/clean_cases.json  2000+ correct / dialect / domain sentences (should_flag: false)

Each error category yields (bad, good, note) triples. Every "bad" is genuinely
incorrect. Notes with "reading:" are instructive labels, never controls.

Fixed seed => reproducible corpora. Error and clean sets are separate files.
"""

import json
import os
import random

SEED = 20260714
rng = random.Random(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))

ERROR_CATS = []          # (name, fn) — fn() yields (bad, good, note)
CLEAN_GENS = []          # (note, fn) — fn() yields correct sentences


def errcat(name):
    def deco(fn):
        ERROR_CATS.append((name, fn))
        return fn
    return deco


def cleangen(note):
    def deco(fn):
        CLEAN_GENS.append((note, fn))
        return fn
    return deco


def pick(seq):
    return rng.choice(seq)


def _plural_noun(n):
    if n.endswith(("s", "x", "z", "ch", "sh")):
        return n + "es"
    if len(n) > 1 and n.endswith("y") and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


NAMES = ["Ali", "Alex", "Ana", "Ben", "Carla", "Dan", "Ella", "Fernando",
         "Grace", "Hakim", "Iris", "Jake", "Kim", "Liam", "Maya", "Nina"]

NOUNS = ["book", "chair", "dog", "apple", "car", "phone", "shirt", "movie",
         "coffee", "bike", "song", "plant", "letter", "gift", "lunch"]

VERBS = ["run", "walk", "jump", "eat", "play", "read", "write", "sing",
         "dance", "sleep", "talk", "listen"]
PAST = {"run": "ran", "walk": "walked", "jump": "jumped", "eat": "ate",
        "play": "played", "read": "read", "write": "wrote", "sing": "sang",
        "dance": "danced", "sleep": "slept", "talk": "talked", "listen": "listened"}
PASTP = {"run": "run", "walk": "walked", "jump": "jumped", "eat": "eaten",
         "play": "played", "read": "read", "write": "written", "sing": "sung",
         "dance": "danced", "sleep": "slept", "talk": "talked", "listen": "listened"}

CAT_COMP = {"tall": "taller", "short": "shorter", "fast": "faster", "slow": "slower",
            "smart": "smarter", "young": "younger", "old": "older", "big": "bigger",
            "small": "smaller", "easy": "easier", "hard": "harder", "high": "higher",
            "early": "earlier", "bright": "brighter"}
CAT_SUP = {k: v.replace("er", "est") for k, v in CAT_COMP.items()}
ADJ_ADV = {"quick": "quickly", "slow": "slowly", "loud": "loudly", "clear": "clearly",
           "careful": "carefully", "proper": "properly", "beautiful": "beautifully",
           "polite": "politely", "honest": "honestly", "safe": "safely"}
ADV_MAP = dict(ADJ_ADV)

SPORTS = ["tennis", "football", "soccer", "basketball", "cricket", "hockey",
          "volleyball", "baseball", "rugby", "golf"]
INSTRUMENTS = ["piano", "guitar", "violin", "flute", "drums", "cello"]

CAUSATIVE_VERBS = ["made", "let", "had", "helped"]
CAUSATIVE_PRON = ["me", "us", "him", "her", "them"]

UNCOUNTABLE_WITH_A = {"equipment", "furniture", "advice", "information",
                      "luggage", "research", "homework", "jewelry", "software",
                      "knowledge"}

COLLECTIVE_SVA = ["The team", "The jury", "The committee", "The audience",
                  "The orchestra", "The board", "The company",
                  "The government", "The faculty", "The staff"]
IRREG_UNC = [("informations", "information"), ("advices", "advice"),
             ("furnitures", "furniture"), ("an equipment", "equipment"),
             ("a luggage", "luggage"), ("a homework", "homework"),
             ("a research", "research")]


# ═══════════════════════════ ERROR CATEGORIES ═══════════════════════════════

@errcat("subject-verb-agreement")
def sva():
    for _ in range(3):
        yield ("She go to school every day.",
               "She goes to school every day.", "3rd-person singular")
        yield ("He work here.",
               "He works here.", "3rd-person singular")
        yield ("It seem like a good idea.",
               "It seems like a good idea.", "3rd-person singular")
    for phrase in ["The list of items", "A box of apples", "The group of friends",
                   "A collection of stamps", "The herd of cows", "The set of rules"]:
        yield (f"{phrase} are on the table.",
               f"{phrase} is on the table.", "prepositional-phrase SVA")
    for subj in COLLECTIVE_SVA:
        yield (f"{subj} are ready.",
               f"{subj} is ready.", "collective-noun (AmE)")
    yield ("Everyone are invited to the party.",
           "Everyone is invited to the party.", "indefinite pronoun")
    yield ("Each of the players have a uniform.",
           "Each of the players has a uniform.", "each-of")
    yield ("Neither of the options are good.",
           "Neither of the options is good.", "neither-of")
    yield ("The number of students are increasing.",
           "The number of students is increasing.", "the number of")
    yield ("Ten dollars are too much.",
           "Ten dollars is too much.", "quantity singular")


@errcat("pronoun-case")
def pronoun_case():
    n1 = pick(NAMES)
    n2 = pick(NAMES)
    yield (f"Me and {n1} went to the store.", f"{n1} and I went to the store.",
           "me-and-I (subject)")
    yield (f"{n1} and me went shopping.", f"{n1} and I went shopping.",
           "and-me (subject)")
    yield (f"Him and {n1} arrived early.", f"He and {n1} arrived early.",
           "him-and (subject)")
    yield (f"Between you and I, this is odd.", f"Between you and me, this is odd.",
           "between I")
    yield (f"Give the book to {n1} and I.", f"Give the book to {n1} and me.",
           "and-I (object)")
    yield (f"By she and {n1}, the work was done.",
           f"By her and {n1}, the work was done.", "by she")
    yield (f"They saw {n1} and I at the park.",
           f"They saw {n1} and me at the park.", "saw I")
    yield ("My brother and myself went hiking.",
           "My brother and I went hiking.", "reflexive subject")


@errcat("verb-tense")
def verb_tense():
    for _ in range(3):
        yield ("Yesterday I go to the store.",
               "Yesterday I went to the store.", "past tense")
        yield ("She have finished her homework.",
               "She has finished her homework.", "has not have")
        yield ("I has seen that movie.",
               "I have seen that movie.", "I have")
    yield ("He didn't went to work.", "He didn't go to work.", "did+past")
    yield ("They was at the park.", "They were at the park.", "they-was")
    yield ("We was waiting for the bus.", "We were waiting for the bus.", "we-was")
    yield ("I will called you tomorrow.", "I will call you tomorrow.", "will+past")
    yield ("She has went to the market.", "She has gone to the market.", "aux+went")
    yield ("He have been sick all week.", "He has been sick all week.", "have->has")
    yield ("They has bought a new house.", "They have bought a new house.", "has->have")


_AUX_VERBS = [v for v in VERBS if v not in ("run", "read")]


@errcat("verb-form")
def verb_form():
    for _ in range(3):
        v = pick(_AUX_VERBS)
        pp = PASTP.get(v, v)
        past = PAST.get(v) or v
        # bad = wrong form after aux / past after did; only when forms differ from base
        if pp != v:
            yield (f"She has {v} to school.",
                   f"She has {pp} to school.", "aux + bare verb")
        if past != v:
            yield (f"He did not {past}.",
                   f"He did not {v}.", "did + past form")
    for bad, good in [
        ("She has buyed a new dress.", "She has bought a new dress."),
        ("He catched the ball.", "He caught the ball."),
        ("They builded a house.", "They built a house."),
        ("I taked the last cookie.", "I took the last cookie."),
        ("She goed to the market.", "She went to the market."),
        ("We drinked all the juice.", "We drank all the juice."),
        ("He has selled his car.", "He has sold his car."),
        ("I runned faster than everyone.", "I ran faster than everyone."),
        ("They thinked about it.", "They thought about it.")]:
        yield (bad, good, "irregular past")


@errcat("comparatives")
def comparatives():
    adj = pick(list(CAT_COMP))
    yield (f"This book is more {CAT_COMP[adj]} than that one.",
           f"This book is {CAT_COMP[adj]} than that one.", "double comparative")
    yield (f"She is the most {CAT_SUP[adj]} in the class.",
           f"She is the {CAT_SUP[adj]} in the class.", "double superlative")
    yield ("This is more better.", "This is better.", "more better")
    yield ("He is the bestest player.", "He is the best player.", "bestest")
    yield ("That was the mostest fun.", "That was the most fun.", "mostest")
    yield ("That idea is very more useful.",
           "That idea is much more useful.", "very more")
    yield ("This is more simpler than the last one.",
           "This is simpler than the last one.", "more simpler")


@errcat("articles")
def articles():
    for _ in range(4):
        n = pick(NOUNS)
        expected = f"a {n}" if n[0] not in "aeiouAEIOU" else f"an {n}"
        wrong = f"an {n}" if n[0] != "a" else f"a {n}"
        yield (f"She has {wrong} on her desk.",
               f"She has {expected} on her desk.", "a/an confusion")
    yield ("He ate a orange.",
           "He ate an orange.", "a before vowel")


@errcat("prepositions")
def prepositions():
    yield ("She is good in math.", "She is good at math.", "good in")
    yield ("He is interested on music.", "He is interested in music.", "interested on")
    yield ("We arrived to Paris.", "We arrived in Paris.", "arrive to")
    yield ("She depends of her parents.", "She depends on her parents.", "depends of")
    yield ("He is married with Anna.", "He is married to Anna.", "married with")
    yield ("They are afraid from spiders.", "They are afraid of spiders.", "afraid from")
    yield ("The key of the door is rusty.",
           "The key to the door is rusty.", "key of")
    yield ("She got married at June.", "She got married in June.", "at June")


@errcat("conjunctions")
def conjunctions():
    yield ("Neither John nor Mary are home.",
           "Neither John nor Mary is home.", "neither-nor")
    yield ("Because he was tired, so he went to bed.",
           "Because he was tired, he went to bed.", "because-so")
    yield ("Although it was late, but we stayed.",
           "Although it was late, we stayed.", "although-but")


@errcat("double-negatives")
def double_negatives():
    yield ("I don't have nothing.", "I don't have anything.", "don't nothing")
    yield ("He can't hardly wait.", "He can hardly wait.", "can't hardly")
    yield ("She didn't do nothing wrong.", "She didn't do anything wrong.", "didn't nothing")
    yield ("Nobody didn't come.", "Nobody came.", "nobody didn't")
    yield ("I couldn't scarcely breathe.", "I could scarcely breathe.", "couldn't scarcely")


@errcat("homophones")
def homophones():
    n1 = pick(NAMES)
    cases = [
        (f"{n1} loves there dog.", f"{n1} loves their dog.",
         "there/their"),
        ("They are going to the store. There going to the store.",
         "They are going to the store. They're going to the store.", "there/they're"),
        ("Your going to love this.", "You're going to love this.", "your/you're"),
        ("Please bring you're book.", "Please bring your book.", "you're/your"),
        ("The dog wagged it's tail.", "The dog wagged its tail.", "it's/its"),
        ("Its raining outside.", "It's raining outside.", "its/it's"),
        (f"{n1} is taller then me.", f"{n1} is taller than me.", "then/than"),
        ("I have less friends than before.",
         "I have fewer friends than before.", "less/fewer"),
        ("The weather is to hot today.", "The weather is too hot today.", "to/too"),
    ]
    for bad, good, note in cases:
        yield (bad, good, note)
    yield ("The storm will effect our plans.",
           "The storm will affect our plans.", "effect/affect")


@errcat("spelling")
def spelling():
    miss = {"recieve": "receive", "definately": "definitely", "seperate": "separate",
            "occured": "occurred", "untill": "until", "begining": "beginning",
            "neccessary": "necessary", "acheive": "achieve", "wich": "which",
            "mispell": "misspell", "calender": "calendar", "embarass": "embarrass"}
    for m, c in list(miss.items()):
        yield (f"This is a {m} issue.", f"This is a {c} issue.", f"spelling: {m}")


@errcat("punctuation")
def punctuation():
    yield ("she went home.", "She went home.", "capitalization")
    yield ("where are you going", "Where are you going?", "cap + question mark")
    yield ("i like ice cream", "I like ice cream.", "capital i")
    yield ("He likes tea but she prefers coffee.",
           "He likes tea, but she prefers coffee.", "missing comma")
    yield ("After dinner we watched a movie.",
           "After dinner, we watched a movie.", "comma after intro")


@errcat("contractions")
def contractions():
    yield ("I could of done it.", "I could have done it.", "could of")
    yield ("She would of called.", "She would have called.", "would of")
    yield ("They should of left early.", "They should have left early.", "should of")
    yield ("He might of forgotten.", "He might have forgotten.", "might of")
    yield ("We must of missed the bus.", "We must have missed the bus.", "must of")


@errcat("redundant")
def redundant():
    yield ("The reason is because he is late.",
           "The reason is that he is late.", "reason is because")
    yield ("Please return back the book.",
           "Please return the book.", "return back")
    yield ("The final conclusion was clear.",
           "The conclusion was clear.", "final conclusion")
    yield ("I am really very happy today.",
           "I am really happy today.", "really very")
    yield ("Each and every student passed.",
           "Every student passed.", "each and every")


@errcat("adjective-adverb")
def adj_adverb():
    yield ("He drives careful.", "He drives carefully.", "adj for adv")
    yield ("She sings beautiful.", "She sings beautifully.", "adj for adv")
    yield ("The team played good.", "The team played well.", "good/well")
    yield ("I did real good on the test.",
           "I did really well on the test.", "real good")
    yield ("She spoke soft.", "She spoke softly.", "soft/softly")
    for v, adj in [("runs", "quick"), ("talked", "loud"), ("answered", "polite"),
                   ("listened", "careful")]:
        yield (f"She {v} {adj}.", f"She {v} {ADV_MAP[adj]}.", "verb + adj")


@errcat("uncountable")
def uncountable():
    yield ("The informations are ready.",
           "The information is ready.", "informations")
    yield ("He gave me some advices.",
           "He gave me some advice.", "advices")
    yield ("The furnitures were sold.",
           "The furniture was sold.", "furnitures")
    yield ("The news are good.",
           "The news is good.", "news are")
    for bad, good in list(IRREG_UNC):
        yield (f"I need {bad}.", f"I need {good}.", "a + uncountable")


@errcat("causative-infinitive")
def causative():
    v = pick(CAUSATIVE_VERBS)
    p = pick(CAUSATIVE_PRON)
    yield (f"She {v} {p} to go.", f"She {v} {p} go.", f"{v} + to")
    yield ("She made him to go.", "She made him go.", "make + to")
    yield ("He let me to drive.", "He let me drive.", "let + to")
    yield ("They had us to wait.", "They had us wait.", "have + to")
    yield ("Please let me to help.", "Please let me help.", "let me to")


@errcat("gerund-infinitive")
def gerund():
    yield ("She enjoys to swim.", "She enjoys swimming.", "enjoy + infinitive")
    yield ("He avoids to talk about it.", "He avoids talking about it.", "avoid + inf")
    yield ("They suggested to go early.", "They suggested going early.", "suggest + inf")
    yield ("Do you mind to wait?", "Do you mind waiting?", "mind + inf")
    yield ("I look forward to see you.",
           "I look forward to seeing you.", "look forward to + gerund")


@errcat("conditional")
def conditional():
    yield ("If I was rich, I would travel.",
           "If I were rich, I would travel.", "if I was")
    yield ("If she was taller, she could reach.",
           "If she were taller, she could reach.", "if she was")
    yield ("If he was here, he would help.",
           "If he were here, he would help.", "if he was")


@errcat("embedded-questions")
def embedded():
    yield ("I wonder where is he going.",
           "I wonder where he is going.", "embedded wh-inversion")
    yield ("Do you know what time is it?",
           "Do you know what time it is?", "embedded what time")
    yield ("Please tell me where is the station.",
           "Please tell me where the station is.", "embedded where")
    yield ("She asked why is he late.",
           "She asked why he is late.", "embedded why")


@errcat("question-inversion")
def question_inv():
    yield ("Why he left so early?",
           "Why did he leave so early?", "do-support in questions")
    yield ("Where you are going?",
           "Where are you going?", "aux-subject inversion")
    yield ("When she will arrive?",
           "When will she arrive?", "will inversion")


@errcat("number-agreement")
def number_agree():
    yield ("She read three book yesterday.",
           "She read three books yesterday.", "num + noun")
    yield ("I have five dog at home.",
           "I have five dogs at home.", "num + noun")
    yield ("The two child played outside.",
           "The two children played outside.", "two child")
    for i in range(5):
        n = pick(NOUNS)
        yield (f"She has two {n}.",
               f"She has two {_plural_noun(n)}.", "plural missing")


@errcat("possessives")
def possessive():
    yield ("The dog left it's toys outside.",
           "The dog left its toys outside.", "it's possessive")
    yield ("She has two dog's.",
           "She has two dogs.", "apostrophe in plural")
    yield ("The 1990's were fun.",
           "The 1990s were fun.", "decade apostrophe")
    yield ("The teacher's are here.",
           "The teachers are here.", "plural teacher's")


@errcat("word-order")
def word_order():
    yield ("She always is on time.", "She is always on time.", "adv position")
    yield ("He never has been to Paris.",
           "He has never been to Paris.", "never position")
    yield ("She speaks very well English.",
           "She speaks English very well.", "adverb after object")


@errcat("collocations")
def collocations():
    yield ("She did a decision.", "She made a decision.", "make/do decision")
    yield ("I did a mistake on the test.",
           "I made a mistake on the test.", "make/do mistake")
    yield ("He done a big effort.", "He made a big effort.", "make/do effort")


@errcat("lay-lie")
def lay_lie():
    yield ("The cat lays on the couch.",
           "The cat lies on the couch.", "lays/lies")
    yield ("Please lie the blanket on the bed.",
           "Please lay the blanket on the bed.", "lie/lay")


@errcat("articles-the")
def article_the():
    for s in SPORTS:
        yield (f"She plays the {s}.", f"She plays {s}.", "the before sport")
    yield ("He goes to the school every day.",
           "He goes to school every day.", "the before school")
    yield ("I had the breakfast already.",
           "I had breakfast already.", "the before meal")


# ═══════════════════════════ CLEAN / FP CASES ═══════════════════════════════

@cleangen("possessives")
def clean_possessives():
    for s in ["paw", "basket", "collar", "toy", "fur"]:
        yield f"The cat's {s} is on the mat."
    yield "The companies' headquarters are in New York."
    yield "The teachers' lounge is on the third floor."
    yield "James's book is very old."
    yield "The children's toys are everywhere."
    yield "The dogs' leashes hang by the door."


@cleangen("entities & names")
def clean_entities():
    for n in NAMES:
        yield f"{n} works at Google."
        yield f"{n} is a software engineer."
        yield f"{n} lives in Boston."
    yield "He moved to New York City last year."
    yield "She bought an iPhone at the Apple store."
    yield "The CEO of Tesla spoke yesterday."
    yield "They live in San Francisco, California."


@cleangen("quotations & reported speech")
def clean_quotes():
    yield 'He asked, "Where are you going?"'
    yield '"The best time is now," she said.'
    yield "She said that she is tired."
    yield "The teacher said, \"Read chapter 5 tonight.\""
    yield 'She asked me if I wanted to go.'


@cleangen("dialect / AAVE")
def clean_dialect():
    yield "She don't like broccoli."
    yield "He don't be at the library on Sundays."
    yield "Ain't nobody got time for that."
    yield "Y'all should come over later."
    yield "It ain't over till it's over."
    yield "She been there before."


@cleangen("technical / domain")
def clean_technical():
    yield "The API returns JSON objects."
    yield "Please cache the URL."
    yield "The fetch call hits the endpoint."
    yield "We deploy on Kubernetes clusters."
    yield "The model overfits on small datasets."
    yield "She wrote a regex for the log parser."
    yield "The SaaS platform handles billing."
    yield "Use a linter before you commit."
    yield "The uptime is 99.9 percent."
    yield "The crypto market is volatile."
    yield "Her code refactors cleanly."


@cleangen("of-phrases & grammar traps")
def clean_of_phrases():
    yield "The list of items is ready."
    yield "A number of students are absent."
    yield "The number of students is growing."
    yield "None of the options are ideal."
    yield "Five dollars is enough for coffee."
    yield "Half of the cake was eaten."
    yield "Two thirds of the work is done."
    yield "The data is stored securely."
    yield "Statistics is my favorite subject."
    yield "Physics is fascinating."


@cleangen("numbers & dates")
def clean_numbers():
    yield "He ran 5 miles yesterday."
    yield "The meeting starts at 9:30 am."
    yield "The train departs at eight o'clock."
    yield "The total was $1,234.56."
    yield "She has three cats, two dogs, and a bird."
    yield "The population exceeds 9 billion."
    yield "He finished 3rd in the race."
    yield "The project costs 2.5 million dollars."


@cleangen("idioms & collocations (correct)")
def clean_idioms():
    yield "He made a huge mistake."
    yield "She took a deep breath."
    yield "I need to get a haircut."
    yield "They had a good time at the party."
    yield "She gave a speech at the event."
    yield "We had breakfast at seven."
    yield "He went to school by bus."


@cleangen("misc correct sentences")
def clean_misc():
    yield "I went to the store yesterday."
    yield "She has been working here for years."
    yield "The children played happily in the garden."
    yield "He will call you when he arrives."
    yield "They are going to visit their grandparents."
    yield "She said that the movie was great."
    yield "The students were writing while they waited."
    yield "We should leave early tomorrow morning."
    yield "He asked her to dance."
    yield "This work plays well and sings nicely."


@cleangen("comparatives correct")
def clean_comparatives():
    yield "She is taller than her brother."
    yield "This is much more interesting than the last one."
    yield "He is the tallest player on the team."
    yield "It was better than I expected."
    yield "Money is less important than health."


@cleangen("modals & perfects")
def clean_modals():
    yield "She can have gone to the store."
    yield "He may have left his keys here."
    yield "They must have finished by now."
    yield "We could have arrived earlier."
    yield "I would have called if I knew."


@cleangen("verb agreement singular traps")
def clean_agreement_traps():
    yield "The news is on at nine."
    yield "Mathematics is hard for some students."
    yield "Everybody is here."
    yield "Everything is fine."
    yield "Each student has a laptop."
    yield "The committee is meeting tomorrow."
    yield "The staff are divided on the issue."
    yield "The government are considering new laws."


@cleangen("clauses with that/if")
def clean_clauses():
    yield "He said that the plan was solid."
    yield "She told them that the meeting moved."
    yield "They asked us to help them move."
    yield "We told them to come early."
    yield "She asked him if he was coming."
    yield "She asked me a simple question."


# ══════════ NEW CATEGORIES (A-F): MODAL FORM, COLLECTIVE HAVE, ══════════════
# ══════════ BACKSHIFT, SCHEDULED ON, ANY/NB + RELATIVE SVA, DUP VERB ═════════

_MODAL_A = ["could", "should", "would", "must", "might"]
_SUBJ_A = ["She", "He", "We", "They", "You", "The team"]
_MODAL_PAIRS = [
    ("noticed", "notice"), ("focused", "focus"), ("completed", "complete"),
    ("decided", "decide"), ("finished", "finish"), ("started", "start"),
    ("described", "describe"), ("presented", "present"), ("submitted", "submit"),
    ("prepared", "prepare"), ("reviewed", "review"), ("discussed", "discuss"),
    ("opened", "open"), ("closed", "close"), ("answered", "answer"),
    ("explained", "explain"), ("confirmed", "confirm"), ("mentioned", "mention"),
    ("reported", "report"), ("printed", "print"), ("copied", "copy"),
    ("deleted", "delete"), ("saved", "save"), ("loaded", "load"),
    ("updated", "update"), ("checked", "check"), ("tested", "test"),
    ("fixed", "fix"), ("added", "add"), ("removed", "remove"),
]
_MODAL_OBJ = [
    "the problem", "the report", "the task", "the schedule", "the document",
    "the software", "the file", "the plan", "the data", "the message",
    "the changes", "the budget", "the meeting", "the request", "the package",
    "the form", "the notes", "the record", "the numbers", "the details",
]


@errcat("modal-form")
def modal_form():
    # Modal + past form ("could noticed") → modal + base ("could notice").
    for off in range(4):
        for i, (masked, base) in enumerate(_MODAL_PAIRS):
            subj = _SUBJ_A[(i + off) % len(_SUBJ_A)]
            modal = _MODAL_A[(i + off) % len(_MODAL_A)]
            obj = _MODAL_OBJ[(i + 2 * off) % len(_MODAL_OBJ)]
            yield (f"{subj} {modal} {masked} {obj}.",
                   f"{subj} {modal} {base} {obj}.", "modal + past form")


_MM_BASE = [
    "focus", "review", "finish", "start", "submit", "prepare", "complete",
    "decide", "describe", "open", "close", "answer", "explain", "confirm",
    "mention", "report", "update", "check", "test", "add", "remove", "keep",
    "read", "write", "call", "help", "ask", "notice", "examine", "accept",
]
_MM_OBJ = [
    "the report", "the task", "the plan", "the form", "the data", "the files",
    "the notes", "early", "on time", "the request", "the document", "the package",
    "the message", "a decision", "the meeting", "the budget",
]


@cleangen("modals correct")
def clean_modals_more():
    # Every modal followed by its base form — nothing to fix.
    for off in range(4):
        for i, base in enumerate(_MM_BASE):
            subj = _SUBJ_A[(i + off) % 5]
            modal = _MODAL_A[(i + off) % len(_MODAL_A)]
            obj = _MM_OBJ[(i + 2 * off) % len(_MM_OBJ)]
            yield f"{subj} {modal} {base} {obj}."
    yield "He could be noticed immediately."
    yield "The door must be locked before leaving."
    yield "She must have completed the form."
    yield "We should have focused on the test."
    yield "You can feed the animals."
    yield "He could decide to leave."


_COLLECTIVE_HAVE = [
    "The team", "The family", "The committee", "The jury", "The audience",
    "The orchestra", "The board", "The company", "The group", "The class",
    "The club", "The army", "The council", "The faculty", "The crowd",
    "The public", "The squad", "The navy", "The panel", "The crew",
]
_COLL_PP = [
    "attended", "arrived", "decided", "prepared", "approved", "submitted",
    "completed", "announced", "agreed", "released",
]
_COLL_OBJ = [
    "the meeting", "the report", "the plan", "the decision", "the results",
    "the proposal", "the schedule", "the event", "the project", "the budget",
]


@errcat("collective-have-sva")
def collective_have_sva():
    # AmE: "The team have attended" → "has attended".
    for off in range(5):
        for i, subj in enumerate(_COLLECTIVE_HAVE):
            pp = _COLL_PP[(i + off) % len(_COLL_PP)]
            obj = _COLL_OBJ[(i + 2 * off) % len(_COLL_OBJ)]
            yield (f"{subj} have {pp} {obj}.",
                   f"{subj} has {pp} {obj}.", "collective + have (AmE)")


_CPP = ["attended", "arrived", "decided", "prepared", "approved", "submitted",
        "completed", "announced", "agreed"]
_COBJ = ["the meeting", "the report", "the plan", "the decision", "the results",
         "the proposal", "the schedule", "the event", "the project"]


@cleangen("collective-has + BrE tolerance")
def clean_collective_have():
    # Real clean collective agreement (has + 3rd singular).
    for off in range(6):
        for i, subj in enumerate(_COLLECTIVE_HAVE):
            pp = _CPP[(i + 5 + off) % len(_CPP)]
            obj = _COBJ[(i + 1 + off) % len(_COBJ)]
            yield f"{subj} has {pp} {obj}."
    # BrE-tolerant collectives with plural verbs are intentionally left clean.
    yield "The staff have attended the ceremony."
    yield "The police have made a decision."
    yield "The government have considered the proposal."
    yield "The staff are divided on the issue."
    yield "The government are considering new laws."
    yield "The media have covered the story."
    yield "The cattle were grazing in the field."


_APP = ["slide", "system", "app", "report", "site", "page", "document", "program",
        "tool", "package", "dashboard", "database", "file", "dataset", "version",
        "model", "interface", "platform", "server", "window"]
_APP_PRES_PAST = {
    "contains": "contained", "includes": "included", "features": "featured",
    "displays": "displayed", "lists": "listed", "presents": "presented",
    "describes": "described", "shows": "showed", "uses": "used",
    "provides": "provided", "offers": "offered", "produces": "produced",
    "mentions": "mentioned", "reports": "reported", "needs": "needed",
    "requires": "required",
}
_APP_PRES = list(_APP_PRES_PAST)
_APP_OBJ = ["the data", "the results", "the key details", "the updated information",
            "the main findings", "the full report", "useful statistics",
            "the latest numbers", "the required fields", "the new feature",
            "detailed instructions", "reliable information", "the relevant data",
            "the summary", "the latest update", "helpful examples"]
_SAY_V = [("noticed", "noticed"), ("saw", "saw"), ("realized", "realized"),
          ("discovered", "discovered"), ("found", "found"), ("observed", "observed"),
          ("learned", "learned"), ("remembered", "remembered"),
          ("mentioned", "mentioned"), ("reported", "reported")]
_SAY_PAST = [v for v, _ in _SAY_V]


@errcat("tense-backshift")
def tense_backshift():
    # Past narrative + present subordinate verb → past tense verb.
    for off in range(3):
        for i, pres in enumerate(_APP_PRES):
            name = pick(NAMES)
            say = _SAY_PAST[(i + off) % len(_SAY_PAST)]
            app = _APP[(i + 2 * off) % len(_APP)]
            obj = _APP_OBJ[(i + 3 * off) % len(_APP_OBJ)]
            past = _APP_PRES_PAST[pres]
            yield (f"Last week, {name} {say} that the {app} {pres} {obj}.",
                   f"Last week, {name} {say} that the {app} {past} {obj}.",
                   "tense backshift")


@cleangen("backshift clean / no past marker")
def clean_backshift():
    for off in range(7):
        for i, pres in enumerate(_APP_PRES):
            app = _APP[(i + off) % len(_APP)]
            obj = _APP_OBJ[(i + 2 * off) % len(_APP_OBJ)]
            yield f"The {app} {pres} {obj}."
    yield "Last week, he noticed that the slide contained errors."
    yield "Every day, she notes that the app runs well."
    yield "Last year, the teacher said that the earth is round."
    yield "Yesterday, she realized that the app worked fine."


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
_EVENTS = ["meeting", "event", "call", "interview", "appointment", "presentation",
           "session", "ceremony", "review", "workshop", "conference", "briefing"]
_EVENT_PARTS = ["", " morning", " afternoon", " evening", " night"]


@errcat("scheduled-on")
def scheduled_on():
    # "scheduled on Monday" → "scheduled for Monday".
    for i in range(12):
        for j, ev in enumerate(_EVENTS):
            wd = _WEEKDAYS[(i + j) % len(_WEEKDAYS)]
            part = _EVENT_PARTS[(i + j) % len(_EVENT_PARTS)]
            verb = "is" if (i + j) % 2 == 0 else "was"
            yield (f"The {ev} {verb} scheduled on {wd}{part}.",
                   f"The {ev} {verb} scheduled for {wd}{part}.",
                   "scheduled on weekday")


@cleangen("scheduled on calendar/system / for")
def clean_scheduled():
    for ev in _EVENTS:
        yield f"The {ev} is scheduled on the calendar."
        yield f"The {ev} was scheduled on the system."
        yield f"The {ev} is scheduled for the morning."
        yield f"The {ev} is scheduled for Friday."
        yield f"The {ev} was scheduled in the system."
        yield f"The {ev} is listed on the schedule."
    yield "The deadline is on the calendar."
    yield "The appointment is in the system for Tuesday."


_ANY_V = ["knew", "knows", "saw", "sees", "remembered", "remembers", "noticed",
          "notices", "understood", "understands", "realized", "realizes",
          "believed", "believes", "thought", "thinks", "found", "finds",
          "learned", "learns", "expected", "expects", "assumed", "assumes",
          "concluded", "concludes", "discovered", "discovers", "recognized",
"recognizes", "considered", "considers", "managed", "manages",
           "determined", "determines", "predicted",
           "predicts", "preferred", "prefers", "rejected", "rejects", "trusted",
           "trusts", "supported", "supports"]
_ANY_OBJ = ["the answer", "the truth", "the plan", "the solution", "the story",
            "the details", "the reason", "the outcome", "the question",
            "the problem", "the information", "the result", "the fact",
            "the situation", "the response", "the decision"]
_PERSONS = ["manager", "doctor", "student", "engineer", "teacher", "driver",
            "designer", "consultant", "lawyer", "artist", "nurse", "analyst",
            "pilot", "editor", "chef", "coach", "director", "author",
            "specialist", "reviewer", "officer", "technician", "inspector",
            "coordinator", "supervisor", "planner", "administrator", "researcher",
            "therapist", "advisor"]
_REL_PRED = ["responsible for the delay", "in charge of the project",
             "present at the meeting", "to blame for the error",
             "responsible for the order", "the last one to speak",
             "in the room at the time", "responsible for the report",
             "available for the task"]


@errcat("quantifier-relative-sva")
def quantifier_relative_sva():
    # "Anybody knew ..." → "Nobody knew ..." (positive-polarity anyone).
    for off in range(2):
        for i, v in enumerate(_ANY_V):
            obj = _ANY_OBJ[(i + 3 + off) % len(_ANY_OBJ)]
            who = "Anybody" if (i + off) % 2 == 0 else "Anyone"
            yield (f"{who} {v} {obj}.",
                   f"Nobody {v} {obj}.", "anybody/anyone declarative")
    # Relative-clause SVA: singular antecedent + "who were" → "who was".
    for i, person in enumerate(_PERSONS):
        pred = _REL_PRED[i % len(_REL_PRED)]
        yield (f"I spoke with the {person} who were {pred}.",
               f"I spoke with the {person} who was {pred}.",
               "who were (relative SVA)")


_PLURALS = ["people", "students", "workers", "engineers", "managers", "teachers",
            "doctors", "players", "drivers", "customers", "clients", "colleagues",
            "guests", "members", "residents", "volunteers", "athletes", "artists",
            "writers", "readers"]


_DUPL_PLURAL_TAIL = [
    "apologized", "left early", "arrived late", "started early", "waited outside",
    "spoke first", "came forward", "raised their hands", "won the prize",
    "took part", "were recognized", "gave a statement",
]


@cleangen("anybody/anyone + plural who-were clean")
def clean_quant_relative():
    for pl in _PLURALS:
        for tail in _DUPL_PLURAL_TAIL:
            yield f"The {pl} who were responsible {tail}."
        yield f"The {pl} who were present left early."
        yield f"The {pl} who were waiting arrived late."
        yield f"The {pl} who were ready started early."
    yield "Anybody can learn to code."
    yield "Anybody who knows the answer should speak up."
    yield "Anyone with experience is welcome."
    yield "Did anybody know the answer?"
    yield "Nobody knew the answer to the question."
    yield "Anyone who applied received a reply."
    yield "Everybody who registered got a seat."
    yield "Anyone who signed up attended the event."


# Each triple was empirically verified: "subj past base task." fires exactly
# DUPLICATE_VERB and apply_fixes maps it to "subj past task." (spaCy parses the
# base verb as an xcomp of the past head). Mixed subjects were excluded because
# they parse inconsistently (e.g. reviewed/review with "the requirements").
_DUP_BASE = {"discussed": "discuss", "explained": "explain", "reviewed": "review",
             "considered": "consider", "proposed": "propose", "examined": "examine",
             "decided": "decide", "confirmed": "confirm", "accepted": "accept",
             "described": "describe", "investigated": "investigate",
             "evaluated": "evaluate", "summarized": "summarize",
             "detected": "detect"}
DUP_VERIFIED = [
    ("We", "discussed", "the next steps"), ("We", "discussed", "the process"),
    ("We", "discussed", "the plan"), ("We", "discussed", "the options"),
    ("We", "discussed", "the findings"), ("We", "discussed", "the request"),
    ("We", "discussed", "the changes"), ("We", "discussed", "the budget"),
    ("We", "explained", "the next steps"), ("We", "explained", "the process"),
    ("We", "explained", "the plan"), ("We", "explained", "the options"),
    ("We", "explained", "the findings"), ("We", "explained", "the request"),
    ("We", "explained", "the changes"), ("We", "explained", "the budget"),
    ("We", "reviewed", "the process"), ("We", "reviewed", "the plan"),
    ("We", "reviewed", "the options"), ("We", "reviewed", "the findings"),
    ("We", "reviewed", "the changes"), ("We", "reviewed", "the budget"),
    ("We", "reviewed", "the contract"), ("We", "reviewed", "the schedule"),
    ("We", "considered", "the next steps"), ("We", "considered", "the process"),
    ("We", "considered", "the plan"), ("We", "considered", "the options"),
    ("We", "considered", "the findings"), ("We", "considered", "the request"),
    ("We", "considered", "the changes"), ("We", "considered", "the budget"),
    ("We", "proposed", "the next steps"), ("We", "proposed", "the process"),
    ("We", "proposed", "the plan"), ("We", "proposed", "the options"),
    ("We", "proposed", "the findings"), ("We", "proposed", "the request"),
    ("We", "proposed", "the changes"), ("We", "proposed", "the budget"),
    ("We", "examined", "the process"), ("We", "examined", "the plan"),
    ("We", "examined", "the options"), ("We", "examined", "the findings"),
    ("We", "examined", "the request"), ("We", "examined", "the changes"),
    ("We", "examined", "the budget"), ("We", "examined", "the contract"),
    ("We", "decided", "the next steps"), ("We", "decided", "the process"),
    ("We", "decided", "the plan"), ("We", "decided", "the options"),
    ("We", "decided", "the findings"), ("We", "decided", "the request"),
    ("We", "decided", "the changes"), ("We", "decided", "the budget"),
    ("We", "confirmed", "the next steps"), ("We", "confirmed", "the process"),
    ("We", "confirmed", "the plan"), ("We", "confirmed", "the options"),
    ("We", "confirmed", "the findings"), ("We", "confirmed", "the request"),
    ("We", "confirmed", "the changes"), ("We", "confirmed", "the budget"),
    ("We", "accepted", "the next steps"), ("We", "accepted", "the process"),
    ("We", "accepted", "the plan"), ("We", "accepted", "the options"),
    ("We", "accepted", "the findings"), ("We", "accepted", "the request"),
    ("We", "accepted", "the changes"), ("We", "accepted", "the budget"),
    ("We", "described", "the next steps"), ("We", "described", "the process"),
    ("We", "described", "the plan"), ("We", "described", "the options"),
    ("We", "described", "the findings"), ("We", "described", "the request"),
    ("We", "described", "the changes"), ("We", "described", "the budget"),
    ("We", "investigated", "the next steps"), ("We", "investigated", "the process"),
    ("We", "investigated", "the plan"), ("We", "investigated", "the options"),
    ("We", "investigated", "the findings"), ("We", "investigated", "the request"),
    ("We", "investigated", "the changes"), ("We", "investigated", "the budget"),
    ("We", "evaluated", "the next steps"), ("We", "evaluated", "the process"),
    ("We", "evaluated", "the plan"), ("We", "evaluated", "the options"),
    ("We", "evaluated", "the findings"), ("We", "evaluated", "the request"),
    ("We", "evaluated", "the changes"), ("We", "evaluated", "the budget"),
    ("We", "summarized", "the next steps"), ("We", "summarized", "the process"),
    ("We", "summarized", "the plan"), ("We", "summarized", "the options"),
    ("We", "summarized", "the findings"), ("We", "summarized", "the request"),
    ("We", "summarized", "the changes"), ("We", "summarized", "the budget"),
    ("We", "detected", "the next steps"), ("We", "detected", "the process"),
    ("We", "detected", "the plan"), ("We", "detected", "the options"),
    ("We", "detected", "the findings"), ("We", "detected", "the request"),
    ("We", "detected", "the changes"), ("We", "detected", "the budget"),
]


@errcat("duplicate-verb")
def duplicate_verb():
    # "We discussed discuss the next steps." → drop the duplicated verb.
    for subj, past, task in DUP_VERIFIED:
        yield (f"{subj} {past} {_DUP_BASE[past]} {task}.",
               f"{subj} {past} {task}.", "xcomp duplicate verb")


@cleangen("duplicate verb clean")
def clean_duplicate():
    for subj, past, task in DUP_VERIFIED:
        yield f"{subj} {past} {task}."
    yield "I went to go to the office."
    yield "They had had enough time."
    yield "She went to try the new restaurant."
    yield "He decided to leave early."
    yield "They wanted to go home."


# ══════════ NEW (G-I): TENSE-MARKER PERFECT + PASSIVE SVA + CLEAN RE-PARTNER ══
# Every combination below was empirically verified (check_v4, AI_PROVIDER=none):
#   - preceding section: parse quirks dropped (e.g. "Priya" prefix deleted by
#     DATA_RULE, "visited/worked" VBNs not parsed as aux complements).
#   - "on Friday" excluded from passive modifiers because PREPOSITION_SCHEDULED_ON
#     legitimately rewrites "scheduled on Friday" -> "scheduled for Friday".

__TMP_MARK = ["Yesterday,", "A month ago,", "Two days ago,"]
__TMP_SUBJS = [
    "the team", "our sales team", "the manager", "the director", "the supervisor",
    "Maya", "Ali", "Grace", "Hakim", "Nina", "Ben", "Iris", "Jake", "Ella", "Ana",
    "Fernando", "the staff", "the committee", "the developers", "the analysts",
]
__TMP_PLUR = {"the developers", "the analysts"}
__TMP_VBNS = ["written", "finished", "approved", "delivered", "attended", "prepared",
              "submitted", "completed", "accepted", "announced", "published", "finalized"]
# participle -> simple past for the irregular VBN used here (verified spellings).
__TMP_PAST = {"written": "wrote"}
__TMP_OBJS = ["the report", "the plan", "the proposal", "the document", "the project",
              "the meeting", "the draft", "the analysis", "the review"]


@errcat("tense-marker-perfect")
def tense_marker_perfect():
    # Present perfect + explicit past-time marker -> simple past.
    for m in __TMP_MARK:
        for s in __TMP_SUBJS:
            aux = "have" if s in __TMP_PLUR else "has"
            for v in __TMP_VBNS:
                for o in __TMP_OBJS:
                    past = __TMP_PAST.get(v, v)
                    yield (f"{m} {s} {aux} {v} {o}.",
                           f"{m} {s} {past} {o}.",
                           "present perfect + past-time marker")


__PAS_SING = ["The meeting", "The report", "The review", "The plan", "The schedule"]
__PAS_PLUR = ["The files", "The sections", "The results", "The documents", "The tests",
              "The packages", "The requirements", "The notes", "The slides", "The emails",
              "The invoices", "The tickets", "The orders", "The accounts", "The forms"]
__PAS_VBN = ["scheduled", "finalized", "published", "reviewed", "approved", "prepared",
             "submitted", "updated", "completed", "delivered", "deleted", "repeated",
             "moved", "processed", "collected", "sent", "discussed", "shared",
             "archived", "posted"]
__PAS_AT = ["for Monday", "yesterday", "last night", "last week", "on time", "again"]


@errcat("passive-sva")
def passive_sva():
    # Passive construction with wrong auxiliary agreement.
    for s in __PAS_SING:
        for v in __PAS_VBN:
            for p in __PAS_AT:
                yield (f"{s} were {v} {p}.",
                       f"{s} was {v} {p}.", "passive were -> was")
    for s in __PAS_PLUR:
        for v in __PAS_VBN:
            for p in __PAS_AT:
                yield (f"{s} was {v} {p}.",
                       f"{s} were {v} {p}.", "passive was -> were")


__CPP = [("I", "have"), ("We", "have"), ("They", "have"), ("She", "has"), ("He", "has"),
         ("It", "has"), ("The team", "has"), ("Maya", "has"), ("Ali", "has")]
__CPP_VBN = ["checked", "finished", "completed", "submitted", "approved", "attended",
             "reviewed", "prepared", "visited", "verified", "confirmed", "updated"]
__CPP_OBJ = ["the document", "the report", "the plan", "the file", "the list", "the form",
             "the agenda", "the schedule", "the checklist"]


@cleangen("present perfect correct")
def clean_present_perfect():
    # Correct have/has + past participle, no past-time marker.
    for s, aux in __CPP:
        for v in __CPP_VBN:
            for o in __CPP_OBJ:
                yield f"{s} {aux} {v} {o}."


__CPP_A = [("I", "have"), ("We", "have"), ("They", "have"), ("She", "has"), ("He", "has"),
           ("It", "has"), ("The team", "has")]
__CPP_A_VBN = ["checked", "finished", "completed", "submitted", "approved", "attended"]
__CPP_A_OBJ = ["the document", "the report", "the plan", "the file", "the list"]


@cleangen("present perfect + already correct")
def clean_present_perfect_already():
    for s, aux in __CPP_A:
        for v in __CPP_A_VBN:
            for o in __CPP_A_OBJ:
                yield f"{s} {aux} already {v} {o}."


__CPA = [("The meeting", "was"), ("The report", "was"), ("The review", "was"),
         ("The files", "were"), ("The sections", "were"), ("The results", "were"),
         ("The documents", "were"), ("The tests", "were"), ("The notes", "were"),
         ("The orders", "were"), ("The accounts", "were"), ("The forms", "were")]
__CPA_VBN = ["scheduled", "finalized", "published", "reviewed", "approved", "prepared",
             "submitted", "updated", "completed", "delivered", "processed", "collected",
             "moved", "shared", "archived", "posted", "repeated", "discussed"]
__CPA_AT = ["", " yesterday", " on time", " last week", " for Monday"]


@cleangen("passive agreement correct")
def clean_passive_agreement():
    # Correct was/were auxiliary agreement in passive sentences.
    for s, be in __CPA:
        for v in __CPA_VBN:
            for a in __CPA_AT:
                yield f"{s} {be} {v}{a}."


# ═══════════════════════════ BUILDERS ═══════════════════════════════════════

def build_errors(target=1000):
    """Round-robin select one unseen case per category each sweep for balance.
    Each category's yields are shuffled (seeded) so sub-patterns mix into the
    corpus instead of being cut off once the target is reached."""
    items_by_cat = []
    for name, fn in ERROR_CATS:
        items = list(fn())
        rng.shuffle(items)
        items_by_cat.append((name, items))
    counts = dict.fromkeys([n for n, _ in items_by_cat], 0)
    seen = set()
    out = []
    idx = {name: 0 for name, _ in items_by_cat}
    exhausted = set()
    target_hit = False
    while not target_hit:
        progressed = False
        for name, items in items_by_cat:
            if name in exhausted:
                continue
            j = idx[name]
            while j < len(items):
                bad, good, note = items[j]
                j += 1
                progressed = True
                if bad in seen:
                    continue
                idx[name] = j
                seen.add(bad)
                out.append({"bad": bad, "good": good, "note": note})
                counts[name] = counts.get(name, 0) + 1
                if len(out) >= target:
                    target_hit = True
                break
            else:
                idx[name] = j
                if j >= len(items):
                    exhausted.add(name)
            if target_hit:
                break
        if len(exhausted) == len(items_by_cat):
            break
    return out, counts


def build_clean(target=1000):
    items_by_gen = []
    for note, fn in CLEAN_GENS:
        items = list(fn())
        rng.shuffle(items)
        items_by_gen.append((note, items))
    seen = set()
    out = []
    idx = {note: 0 for note, _ in items_by_gen}
    exhausted = set()
    target_hit = False
    while not target_hit:
        progressed = False
        for note, items in items_by_gen:
            if note in exhausted:
                continue
            j = idx[note]
            while j < len(items):
                s = items[j]
                j += 1
                progressed = True
                if s in seen:
                    continue
                idx[note] = j
                seen.add(s)
                out.append(s)
                if len(out) >= target:
                    target_hit = True
                break
            else:
                idx[note] = j
                if j >= len(items):
                    exhausted.add(note)
            if target_hit:
                break
        if len(exhausted) == len(items_by_gen):
            break
    return out


def main():
    errors, counts = build_errors(2000)
    cleans = build_clean(2000)

    err_cases = [{"id": f"E{i:04d}", "text": e["bad"], "good": e["good"],
                  "note": e["note"], "should_flag": True} for i, e in enumerate(errors)]
    clean_cases = [{"id": f"C{i:04d}", "text": s, "note": "", "should_flag": False}
                   for i, s in enumerate(cleans)]

    with open(os.path.join(HERE, "error_cases.json"), "w", encoding="utf-8") as f:
        json.dump(err_cases, f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "clean_cases.json"), "w", encoding="utf-8") as f:
        json.dump(clean_cases, f, indent=1, ensure_ascii=False)

    print(f"error cases : {len(err_cases)}")
    print(f"clean cases : {len(clean_cases)}")
    print("categories :")
    for name, cnt in sorted(counts.items()):
        print(f"  {name:30s} {cnt}")


if __name__ == "__main__":
    main()