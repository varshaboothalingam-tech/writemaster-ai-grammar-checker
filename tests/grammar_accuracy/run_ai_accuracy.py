"""run_ai_accuracy.py — controlled accuracy evaluation of the AI-first pipeline.

Generates a deterministic (seeded) corpus of exactly N error sentences, one
injected error per sentence, across the error types the tool claims to fix.
Runs ``pipeline.ai_core.check_ai_text(..., use_ai=False)`` offline (hermetic,
no network) and computes precision / recall / F1 plus sentence-repair accuracy
and clean-text false-positive rate.

Usage:
    python tests/grammar_accuracy/run_ai_accuracy.py [--count 2000] [--seed 42]

Results are written to:
    results/ai_accuracy_<seed>_<count>.json   (metrics + config)
    results/ai_accuracy_corpus_<seed>.json    (the frozen corpus for later
                                               comparison runs, e.g. with Gemini)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AI_PROVIDER", "none")
import sys  # noqa: E402
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.ai_core import check_ai_text  # noqa: E402
from pipeline.aggregator import _relocate_one  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ---------------------------------------------------------------------------
# Corpus generators: each returns a list of (bad, repaired) with ONE error.
# ---------------------------------------------------------------------------

SPELLING_PAIRS: List[Tuple[str, str]] = [  # (correct, misspelled)
    ("every", "evry"), ("because", "becaus"), ("address", "adress"),
    ("beautiful", "beautifull"), ("studying", "studing"), ("message", "mesage"),
    ("school", "skool"), ("occurring", "occuring"), ("tomorrow", "tommorow"),
    ("definitely", "definately"), ("receive", "recieve"), ("believe", "beleive"),
    ("separate", "seperate"), ("necessary", "neccessary"), ("occasion", "occassion"),
    ("accommodate", "acommodate"), ("embarrass", "embarass"), ("guarantee", "garrantee"),
    ("maintenance", "maintainance"), ("occurrence", "occurence"), ("privilege", "priviledge"),
    ("recommend", "reccommend"), ("restaurant", "restarant"), ("rhythm", "rythm"),
    ("vacuum", "vaccum"), ("weird", "wierd"), ("questionnaire", "questionaire"),
    ("misspell", "mispell"), ("noticeable", "noticable"), ("pastime", "passime"),
    ("persistent", "persistant"), ("preferred", "prefered"), ("intelligence", "inteligence"),
    ("independence", "independance"), ("government", "goverment"), ("finally", "finnaly"),
    ("experience", "experiance"), ("environment", "enviroment"), ("equipment", "equiptment"),
    ("knowledge", "knowlege"), ("language", "langauge"), ("necessary", "necesary"),
    ("opportunity", "oppurtunity"), ("original", "orignal"), ("particular", "particur"),
    ("possible", "possable"), ("preparation", "preperation"), ("pronunciation", "pronounciation"),
    ("publicly", "publicaly"), ("question", "questionn"), ("really", "realy"),
    ("remember", "remeber"), ("weather", "wether"), ("whether", "wether"),
    ("writing", "writting"), ("beginning", "begining"), ("calendar", "calender"),
    ("candidate", "candiate"), ("colleague", "collegue"), ("comfortable", "comfortible"),
    ("committee", "comitee"), ("comparison", "comperison"), ("complete", "complet"),
    ("confident", "confidant"), ("conscience", "consience"), ("conscious", "concious"),
    ("control", "controll"), ("decide", "decid"), ("description", "discription"),
    ("different", "diffrent"), ("disappear", "disapear"), ("discipline", "disipline"),
    ("foreign", "foriegn"), ("fortunate", "fortunite"), ("grammar", "grammer"),
    ("immediately", "immediatly"), ("independent", "independant"),
    ("appeared", "apeared"), ("attention", "attantion"), ("available", "availible"),
    ("bargain", "bargin"), ("benefit", "benifit"), ("boundary", "boundry"),
    ("brilliant", "briliant"), ("business", "bussiness"), ("cafeteria", "cafetaria"),
    ("ceiling", "ceilling"), ("challenge", "chalenge"), ("chocolate", "choclate"),
    ("convenience", "conveniance"), ("desperate", "desparate"), ("dilemma", "dillema"),
    ("especially", "espesially"), ("exaggerate", "exaggarate"), ("excellent", "excelent"),
    ("height", "heigth"), ("interference", "interferance"), ("irresistible", "irresistable"),
    ("millionaire", "millionare"), ("miniature", "miniture"), ("occasionally", "occassionally"),
    ("omission", "omisson"), ("optimistic", "optimisitic"), ("permanent", "pernament"),
    ("philosophy", "philophy"), ("potatoes", "potatos"), ("probably", "probly"),
    ("procedure", "proceedure"), ("scissors", "scisors"), ("shoulder", "sholder"),
    ("succeed", "succede"), ("tendency", "tendancy"), ("unnecessary", "unecessary"),
    ("valuable", "valuble"), ("February", "Feburary"), ("psycology", "psychology"),
]

SUBJECTS_SG = ["The dog", "The cat", "The boy", "The girl", "The student", "The child",
               "The bird", "The worker", "My friend", "The teacher"]
SUBJECTS_PL = ["The dogs", "The cats", "The boys", "The girls", "The students",
               "The children", "The birds", "The workers", "My friends", "The teachers"]
SUBJECTS_PRON = ["She", "He"]
VERBS_BASE = ["run", "walk", "play", "sing", "dance", "swim", "read", "write",
              "jump", "laugh", "work", "study", "talk", "eat", "drive", "sleep"]
OBJECTS = ["in the park", "at school", "in the garden", "at home", "to the market",
           "in the morning", "at night", "in the library", "to the store", "at the beach"]

AUX_HAS = ["has", "have"]
PP_PAIRS: List[Tuple[str, str]] = [("went", "gone"), ("ate", "eaten"), ("saw", "seen"),
                                   ("wrote", "written"), ("spoke", "spoken"),
                                   ("broke", "broken"), ("drove", "driven"),
                                   ("took", "taken"), ("ran", "run"), ("sang", "sung")]
BE_PAST_WRONG = ["went", "ate", "saw"]

HOMO_POOLS: List[Tuple[List[str], str, str]] = [  # (contexts, correct, wrong-at-start)
    (  # they're <- Their  (v4: their-going rule)
        ["going to the store", "welcome to join", "coming to dinner", "going to be late",
         "ready to start", "heading to work", "planning a trip", "moving to the city",
         "buying a house", "getting married", "starting a business", "playing outside",
         "working late", "having fun", "learning English", "waiting for the bus"],
        "they're", "Their"),
    (  # their <- There
        ["house is big", "children play here", "car is parked outside", "home is nearby",
         "team won the game", "books are on the shelf", "garden is full of flowers"],
        "their", "There"),
    (  # there <- Their
        ["is a mouse in the kitchen", "are many people outside", "goes the bell",
         "are two options"],
        "there", "Their"),
    (  # they're <- There
        ["waiting outside", "coming over", "working together", "always helpful"],
        "they're", "There"),
    (  # you're <- Your
        ["welcome to join", "going to love this", "really smart", "probably right",
         "home already", "going to need a coat", "doing great", "on the right track",
         "almost done", "invited too", "first in line", "the winner", "our last hope"],
        "you're", "Your"),
    (  # it's <- Its
        ["a nice day today", "raining outside", "going to rain", "been a long week",
         "a great idea", "time to leave", "my favorite color", "the best I can do",
         "not what I expected", "hard to say", "easy to do", "good to see you"],
        "it's", "Its"),
    (  # who's <- Whose
        ["coming to dinner", "car is this outside", "book is on the table", "going to win",
         "turn is it", "ready to go", "responsible for this", "calling at this hour",
         "at the door", "idea was this"],
        "who's", "Whose"),
    (  # too <- to
        ["much to handle", "big to fit", "hard to believe", "late for the event",
         "hot to drink", "fast to follow", "young to drive", "small to see",
         "tired to continue", "old to run"],
        "too", "to"),
    (  # two <- to
        ["of us are late", "days ago", "years old", "miles away",
         "options left", "hours later", "times a week", "books missing"],
        "two", "to"),
    (  # than <- then
        ["taller then me", "faster then him", "bigger then expected", "older then you",
         "better then before", "harder then it looks"],
        "than", "then"),
]

REPEAT_WORD_TEMPLATES: List[Tuple[str, str]] = [  # (bad, good) — one duplicated word
    ("The the cat is sleeping.", "The cat is sleeping."),
    ("The the dog barked loudly.", "The dog barked loudly."),
    ("The the sky is blue today.", "The sky is blue today."),
    ("I saw the the movie yesterday.", "I saw the movie yesterday."),
    ("Open the the door please.", "Open the door please."),
    ("We read the the book together.", "We read the book together."),
    ("I saw a a bird in the sky.", "I saw a bird in the sky."),
    ("She has a a new car.", "She has a new car."),
    ("He bought a a red bike.", "He bought a red bike."),
    ("There is a a big storm coming.", "There is a big storm coming."),
    ("They own a a small farm.", "They own a small farm."),
    ("It takes a a while to finish.", "It takes a while to finish."),
    ("Tom and and Jerry are friends.", "Tom and Jerry are friends."),
    ("We sang and and danced all night.", "We sang and danced all night."),
    ("He laughed and and smiled.", "He laughed and smiled."),
    ("They walked and and talked together.", "They walked and talked together."),
    ("It is in in the box.", "It is in the box."),
    ("She lives in in the city.", "She lives in the city."),
    ("We are in in the same class.", "We are in the same class."),
    ("The keys are in in the drawer.", "The keys are in the drawer."),
    ("One of of us is late.", "One of us is late."),
    ("It is made of of solid wood.", "It is made of solid wood."),
    ("The rest of of them left.", "The rest of them left."),
    ("A cup of of coffee is enough.", "A cup of coffee is enough."),
    ("She is very very happy today.", "She is very happy today."),
    ("It was very very cold outside.", "It was very cold outside."),
    ("This is very very important.", "This is very important."),
    ("He is very very tall.", "He is very tall."),
    ("I want to to go home now.", "I want to go home now."),
    ("We need to to finish this.", "We need to finish this."),
    ("She tried to to call him.", "She tried to call him."),
    ("They like to to play outside.", "They like to play outside."),
    ("The book is on on the table.", "The book is on the table."),
    ("We rely on on each other.", "We rely on each other."),
    ("She put the hat on on her head.", "She put the hat on her head."),
    ("The meeting is on on Monday.", "The meeting is on Monday."),
    ("This is my my favorite book.", "This is my favorite book."),
    ("He took my my phone.", "He took my phone."),
    ("She walked my my dog.", "She walked my dog."),
    ("They visited my my hometown.", "They visited my hometown."),
    ("It is is a good idea.", "It is a good idea."),
    ("This is is the last one.", "This is the last one."),
    ("He is is my best friend.", "He is my best friend."),
    ("The door is is open.", "The door is open."),
    ("She is is ready now.", "She is ready now."),
]

REPEAT_LETTERS: List[Tuple[str, str]] = [("goood", "good"), ("happpy", "happy"),
                                         ("greaat", "great"), ("coool", "cool"),
                                         ("neeed", "need"), ("eaassy", "easy"),
                                         ("sooon", "soon"), ("taall", "tall"),
                                         ("reallly", "really"), ("veryyy", "very"),
                                         ("quiick", "quick"), ("slooow", "slow"),
                                         ("nicce", "nice"), ("braave", "brave")]

A_N_NOUNS = ["apple", "orange", "umbrella", "egg", "hour", "uncle", "elephant",
             "apple pie", "empty room", "old car", "excellent choice", "honest answer",
             "open door", "important meeting", "unusual event", "amazing view",
             "elegant dress", "enormous building", "early start", "interesting idea"]
A_N_OPS = ["She bought", "He ate", "We saw", "I found", "They enjoyed",
           "She made", "He wrote", "We visited", "I read"]

TENSE_SUBJECTS = ["I", "We", "They", "You", "She", "He"]
TENSE_OBJECTS = ["in the park", "at school", "in the garden", "at home", "to the market",
                 "in the morning", "at night", "in the library", "to the store", "at the beach",
                 "to the hospital", "to the park", "to the theater", "to the gym", "into town"]

DONT_SUBJECTS = ["She", "He", "It"]
DONT_ACTS = ["like coffee", "want to go", "know the answer", "play tennis",
             "work here", "read books", "enjoy the movie", "drive a car",
             "speak Spanish", "take the train", "want tea", "cook dinner",
             "need help", "understand this", "remember that", "agree with you"]


def _mk_sva_sg() -> Tuple[str, str]:
    # singular subject + base verb (should be verb+s) -> inject missing -s
    subj = random.choice(SUBJECTS_SG)
    verb = random.choice(VERBS_BASE)
    obj = random.choice(OBJECTS)
    bad = f"{subj} {verb} {obj}."
    good = f"{subj} {verb}s {obj}."
    return bad, good


def _mk_sva_pl() -> Tuple[str, str]:
    # plural subject + singular verb (verb+s) -> inject stray -s
    subj = random.choice(SUBJECTS_PL)
    verb = random.choice(VERBS_BASE)
    obj = random.choice(OBJECTS)
    bad = f"{subj} {verb}s {obj}."
    good = f"{subj} {verb} {obj}."
    return bad, good


def _mk_sva_pron() -> Tuple[str, str]:
    subj = random.choice(SUBJECTS_PRON)
    verb = random.choice(VERBS_BASE)
    obj = random.choice(OBJECTS)
    bad = f"{subj} {verb} {obj}."
    good = f"{subj} {verb}s {obj}."
    return bad, good


def _mk_aux_pp() -> Tuple[str, str]:
    subj_aux = [("She", "has"), ("He", "has"), ("It", "has"), ("They", "have"),
                ("We", "have"), ("I", "have"), ("You", "have")]
    subj, aux = random.choice(subj_aux)
    wrong, right = random.choice(PP_PAIRS)
    obj = random.choice(OBJECTS)
    bad = f"{subj} {aux} {wrong} {obj}."
    good = f"{subj} {aux} {right} {obj}."
    return bad, good


def _mk_be_past() -> Tuple[str, str]:
    subj = random.choice(["They", "We", "I", "She", "He"])
    be = random.choice(["is", "are", "was", "were"])
    wrong = random.choice(BE_PAST_WRONG)
    obj = random.choice(OBJECTS)
    bad = f"{subj} {be} {wrong} {obj}."
    good = f"{subj} {wrong} {obj}." if be in ("is", "was") else f"{subj} {wrong} {obj}."
    return bad, good


def _mk_tense() -> Tuple[str, str]:
    subj = random.choice(TENSE_SUBJECTS)
    obj = random.choice(TENSE_OBJECTS)
    bad = f"{subj} go {obj} yesterday."
    good = f"{subj} went {obj} yesterday."
    return bad, good


def _mk_dont() -> Tuple[str, str]:
    subj = random.choice(DONT_SUBJECTS)
    act = random.choice(DONT_ACTS)
    bad = f"{subj} don't {act}."
    good = f"{subj} doesn't {act}."
    return bad, good


def _mk_homophone() -> Tuple[str, str]:
    ctxs, correct, wrong = random.choice(HOMO_POOLS)
    ctx = random.choice(ctxs)
    bad = f"{wrong} {ctx}."
    good = f"{correct} {ctx}."
    return bad, good


def _mk_repeat_word() -> Tuple[str, str]:
    return random.choice(REPEAT_WORD_TEMPLATES)


def _mk_repeat_letter() -> Tuple[str, str]:
    wrong, good_w = random.choice(REPEAT_LETTERS)
    ctx = random.choice(["This is", "It was", "That looks", "Here is",
                         "She has", "We got"])
    bad = f"{ctx} a {wrong} book."
    good = f"{ctx} a {good_w} book."
    return bad, good


def _mk_an() -> Tuple[str, str]:
    noun = random.choice(A_N_NOUNS)
    op = random.choice(A_N_OPS)
    bad = f"{op} a {noun} today."
    good = f"{op} an {noun} today."
    return bad, good


def _mk_spelling() -> Tuple[str, str]:
    correct, wrong = random.choice(SPELLING_PAIRS)
    ctx = random.choice([
        f"This is a very {correct} day.",
        f"The {correct} thing matters most.",
        f"I need to {correct} this note.",
        f"We saw the {correct} event.",
        f"Her {correct} work impressed everyone.",
        f"The plan is very {correct}.",
    ])
    bad = ctx.replace(correct, wrong, 1)
    return bad, ctx


CATEGORY_BUILDERS = [
    ("spelling", _mk_spelling),
    ("sva_singular", _mk_sva_sg),
    ("sva_plural", _mk_sva_pl),
    ("sva_pronoun", _mk_sva_pron),
    ("aux_past_participle", _mk_aux_pp),
    ("be_past_form", _mk_be_past),
    ("tense_marker", _mk_tense),
    ("dont_doesnt", _mk_dont),
    ("homophone", _mk_homophone),
    ("repeated_word", _mk_repeat_word),
    ("repeated_letter", _mk_repeat_letter),
    ("a_vs_an", _mk_an),
]

# Deterministic distribution summing to --count (default 2000)
DEFAULT_SPLIT = [500, 450, 300, 200, 250, 100, 100, 50, 100, 50, 100, 200]


def build_corpus(count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    tot = sum(DEFAULT_SPLIT)
    raw = [max(1, int(b * count / tot)) for b in DEFAULT_SPLIT]
    diff = count - sum(raw)
    i = 0
    while diff != 0:
        step = 1 if diff > 0 else -1
        target = i % len(raw)
        if raw[target] + step >= 1:
            raw[target] += step
            diff -= step
        i += 1

    corpus: List[Dict] = []
    for (cat, builder), n_wanted in zip(CATEGORY_BUILDERS, raw):
        attempts = 0
        made = 0
        while made < n_wanted and attempts < n_wanted * 500:
            attempts += 1
            bad, good = builder()
            if bad == good or norm(bad) == norm(good):
                continue
            if any(c["bad"] == bad for c in corpus):
                continue
            corpus.append({"bad": bad, "good": good, "category": cat, "seed": seed})
            made += 1
        if made < n_wanted:
            raise RuntimeError(
                f"category {cat!r} exhausted at {made}/{n_wanted} after {attempts} tries "
                f"— generator pool too small for quota"
            )
    if len(corpus) > count:
        corpus = corpus[:count]
    rng.shuffle(corpus)
    return corpus


def build_clean_set(n: int, seed: int) -> List[str]:
    rng = random.Random(seed + 1000)
    clean: List[str] = []
    builders = [b for _, b in CATEGORY_BUILDERS]
    while len(clean) < n:
        bad, good = random.choice(builders)()
        good = good[:1].upper() + good[1:]
        if good not in clean:
            clean.append(good)
    return clean


def proposal_fixes(original: str, expected: str, wrong: str, correct: str) -> bool:
    loc = _relocate_one(original, wrong)
    if loc is None:
        return False
    s, e = loc
    rebuilt = original[:s] + correct + original[e:]
    return norm(rebuilt) == norm(expected)


def evaluate(corpus: List[Dict], clean: List[str], seed: int) -> Dict:
    tp = fn = fp = 0
    repaired = 0
    total_proposals = 0
    matched_proposals = 0
    clean_fp_sentences = 0
    confidences = []
    per_error = []
    start = time.time()

    for item in corpus:
        res = check_ai_text(item["bad"], use_ai=False)
        props = res["errors"]
        total_proposals += len(props) if props else 0
        is_repaired = norm(res["corrected_text"]) == norm(item["good"])
        repaired += is_repaired
        hit = False
        for p in props:
            confidences.append(p.get("confidence", 0))
            if proposal_fixes(item["bad"], item["good"], p.get("wrong", ""), p.get("correct", "")):
                hit = True
                matched_proposals += 1
        if hit:
            tp += 1
            fp += len(props) - 1
        else:
            fn += 1
            fp += len(props) if props else 0
        per_error.append({
            "category": item["category"],
            "bad": item["bad"],
            "good": item["good"],
            "repaired": is_repaired,
            "detected": hit,
            "proposals": [{"wrong": p.get("wrong"), "correct": p.get("correct"),
                           "confidence": p.get("confidence")} for p in (props or [])],
        })

    for text in clean:
        res = check_ai_text(text, use_ai=False)
        props = res["errors"]
        if props:
            clean_fp_sentences += 1
        total_proposals += len(props) if props else 0
        fp += len(props) if props else 0

    total_injected = len(corpus)
    precision = tp / total_proposals if total_proposals else 0.0
    recall = tp / total_injected if total_injected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "config": {"count": total_injected, "seed": seed,
                   "clean_size": len(clean), "mode": "offline"},
        "metrics": {
            "total_injected": total_injected,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "total_proposals": total_proposals,
            "matched_proposals": matched_proposals,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "sentence_repair_accuracy": round(repaired / total_injected, 4),
            "clean_fp_rate": round(clean_fp_sentences / len(clean), 4),
            "clean_sentences_with_fp": clean_fp_sentences,
            "mean_confidence": round(statistics.mean(confidences), 4) if confidences else 0.0,
            "elapsed_seconds": round(time.time() - start, 2),
        },
        "by_category": _category_breakdown(per_error),
        "per_error": per_error,
    }


def _category_breakdown(per_error: List[Dict]) -> Dict:
    out: Dict[str, Dict] = {}
    for e in per_error:
        d = out.setdefault(e["category"], {"total": 0, "detected": 0, "repaired": 0})
        d["total"] += 1
        d["detected"] += int(e["detected"])
        d["repaired"] += int(e["repaired"])
    for d in out.values():
        d["recall"] = round(d["detected"] / d["total"], 4)
        d["repair_accuracy"] = round(d["repaired"] / d["total"], 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", type=int, default=200)
    args = ap.parse_args()

    corpus = build_corpus(args.count, args.seed)
    clean = build_clean_set(args.clean, args.seed)
    print(f"corpus: {len(corpus)} error sentences, {len(clean)} clean")

    results = evaluate(corpus, clean, args.seed)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    corpus_path = os.path.join(ROOT, "results", f"ai_accuracy_corpus_{args.seed}.json")
    result_path = os.path.join(ROOT, "results", f"ai_accuracy_{args.seed}_{args.count}.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    m = results["metrics"]
    cfg = results["config"]
    print("\n=== AI-first pipeline accuracy (offline rule path) ===")
    print(f"total injected errors        : {m['total_injected']}")
    print(f"sentence repair accuracy     : {m['sentence_repair_accuracy']:.2%}")
    print(f"detected (recall)            : {m['recall']:.2%}   ({m['true_positives']} / {m['total_injected']})")
    print(f"precision                    : {m['precision']:.2%}   ({m['true_positives']} TP / {m['total_proposals']} proposals)")
    print(f"F1                           : {m['f1']:.2%}")
    print(f"false positives              : {m['false_positives']}")
    print(f"clean-sentence FP rate       : {m['clean_fp_rate']:.2%}   ({m['clean_sentences_with_fp']} / {cfg['clean_size']})")
    print(f"mean proposal confidence     : {m['mean_confidence']}")
    print(f"elapsed                      : {m['elapsed_seconds']}s")

    print("\n=== By category ===")
    for cat, d in results["by_category"].items():
        print(f"  {cat:<18} n={d['total']:<4} recall={d['recall']:.2%}  repair={d['repair_accuracy']:.2%}")
    print(f"\nwrote: {result_path}")
    print(f"froze corpus at: {corpus_path}")


if __name__ == "__main__":
    main()