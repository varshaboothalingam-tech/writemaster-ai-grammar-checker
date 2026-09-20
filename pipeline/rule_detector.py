"""pipeline.rule_detector — dependency-light candidate generator.

Produces correction CANDIDATES only; it never rewrites text by itself.
Dependencies: Python stdlib + ``data/*.json`` + ``high_confidence_rules``
(which is stdlib-only). This means it can run inside a Vercel serverless
function without spaCy / NLTK.

Candidate schema (dict):
    {
      "wrong": str, "correct": str, "type": str,
      "start": int, "end": int, "confidence": float,
      "source": "rule", "rule_id": str, "message": str
    }
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set

try:
    from high_confidence_rules import HighConfidenceDetector
    _HC_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    HighConfidenceDetector = None  # type: ignore
    _HC_AVAILABLE = False

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

_WORD = r"[A-Za-z']+"

# Verb forms used by the agreement / tense rules.
_THIRD_SINGULAR = {"go": "goes", "have": "has", "do": "does", "play": "plays",
                   "like": "likes", "eat": "eats", "run": "runs", "work": "works",
                   "want": "wants", "need": "needs", "make": "makes",
                   "take": "takes", "know": "knows", "see": "sees", "say": "says",
                   "give": "gives", "come": "comes", "read": "reads",
                   "write": "writes", "study": "studies", "watch": "watches",
                   "teach": "teaches", "live": "lives", "speak": "speaks"}
_PLURAL_FORM = {"goes": "go", "has": "have", "does": "do", "plays": "play",
                "likes": "like", "eats": "eat", "runs": "run", "works": "work",
                "wants": "want", "needs": "need", "makes": "make", "takes": "take",
                "knows": "know", "sees": "see", "says": "say", "gives": "give",
                "comes": "come", "reads": "read", "writes": "write",
                "studies": "study", "watches": "watch", "teaches": "teach",
                "lives": "live", "speaks": "speak"}

_IRREGULAR_PAST = {
    "go": "went", "see": "saw", "eat": "ate", "come": "came", "take": "took",
    "make": "made", "have": "had", "do": "did", "get": "got", "know": "knew",
    "bring": "brought", "buy": "bought", "catch": "caught", "think": "thought",
    "say": "said", "give": "gave", "run": "ran", "sing": "sang",
    "write": "wrote", "speak": "spoke", "drive": "drove", "break": "broke",
    "choose": "chose", "find": "found", "forget": "forgot", "understand": "understood",
    "meet": "met", "leave": "left", "sleep": "slept", "keep": "kept",
    "feel": "felt", "tell": "told", "sell": "sold", "teach": "taught",
    "lose": "lost", "win": "won", "sit": "sat", "stand": "stood",
    "begin": "began", "drink": "drank", "swim": "swam", "fly": "flew",
    "ride": "rode", "wear": "wore", "draw": "drew", "throw": "threw",
    "spend": "spent",
}
_PAST_SIMPLE = set(_IRREGULAR_PAST.values())

_PAST_PARTICIPLE = {
    "went": "gone", "saw": "seen", "ate": "eaten", "came": "come", "took": "taken",
    "made": "made", "had": "had", "did": "done", "got": "gotten", "knew": "known",
    "brought": "brought", "bought": "bought", "caught": "caught",
    "thought": "thought", "said": "said", "gave": "given", "ran": "run",
    "sang": "sung", "wrote": "written", "spoke": "spoken", "drove": "driven",
    "broke": "broken", "chose": "chosen", "found": "found", "forgot": "forgotten",
    "understood": "understood", "met": "met", "left": "left", "slept": "slept",
    "kept": "kept", "felt": "felt", "told": "told", "sold": "sold",
    "taught": "taught", "lost": "lost", "won": "won", "sat": "sat",
    "stood": "stood", "began": "begun", "drank": "drunk", "swam": "swum",
    "flew": "flown", "rode": "ridden", "wore": "worn", "drew": "drawn",
    "threw": "thrown",
}
_BE_FORMS = {"is", "are", "was", "were", "am", "be", "been", "being"}
_HAVE_FORMS = {"has", "have", "had"}

_PAST_MARKERS = re.compile(
    r"\b(yesterday|the day before yesterday|last (?:night|week|month|year|monday|tuesday|"
    r"wednesday|thursday|friday|saturday|sunday|time)|\d+ (?:days?|weeks?|months?|years?) ago|"
    r"ago|previously|earlier)\b", re.I)

_GERUNDS = ("going|coming|doing|playing|working|watching|studying|eating|running|"
            "walking|talking|saying|giving|making|taking|reading|writing|speaking")

_PRONOUN_SUBJECT = {"i": "I", "he": "he", "she": "she", "it": "it",
                    "we": "we", "you": "you", "they": "they"}
_BE_FOR_PRONOUN = {"i": "am", "he": "is", "she": "is", "it": "is",
                   "we": "are", "you": "are", "they": "are"}

_A_VOWEL_EXCEPTIONS = {"university", "uniform", "user", "usual", "united", "european",
                       "euro", "one", "once", "unicorn", "union", "unique", "unit",
                       "useful", "used", "ukulele", "university"}
_AN_CONSONANT_EXCEPTIONS = {"hour", "honest", "honour", "honor", "heir", "honestly"}

_NUMBER_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "many", "several", "both", "few", "multiple", "various"}
_PLURAL_NOUNS = {"book": "books", "car": "cars", "cat": "cats", "dog": "dogs",
                 "day": "days", "apple": "apples", "pen": "pens", "student": "students",
                 "house": "houses", "school": "schools", "friend": "friends",
                 "boy": "boys", "girl": "girls", "city": "cities", "child": "children",
                 "man": "men", "woman": "women", "person": "people", "idea": "ideas",
                 "problem": "problems", "question": "questions", "answer": "answers",
                 "year": "years", "month": "months", "week": "weeks", "minute": "minutes"}

_DAY_MONTH_NAMES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

# Irregular bases whose PAST form is also a common noun ("take a break",
# "have a drink") — never convert them right after a determiner.
_NOUNY_IRREGULAR = {"break", "drink", "sleep", "leave", "ride", "catch",
                    "feel", "run", "find", "swim", "draw"}
_DETERMINERS = {"a", "an", "the", "my", "your", "our", "his", "her", "its",
                "their", "this", "that", "these", "those", "some", "any",
                "every", "each", "no", "one", "two"}

# 3rd-person-singular person nouns — a base verb right after one is a learner
# tense error in a past narrative ("My friend explain ...").
_PERSON_NOUNS = ("friend|classmate|student|teacher|professor|librarian|brother|"
                 "sister|mother|father|parent|uncle|aunt|grandmother|grandfather|"
                 "doctor|nurse|driver|waiter|chef|boy|girl|man|woman|kid|child|"
                 "neighbor|colleague|manager|boss|coach|writer|artist|player|"
                 "member|guest|pal|cook|worker")

# Verb bases the coordination/person-noun rules may tense-shift.
_PASTABLE_VERBS = set(_PLURAL_FORM.values()) | set(_IRREGULAR_PAST) | {
    "ask", "finish", "talk", "start", "search", "wait", "help", "listen",
    "watch", "use", "look", "try", "explain", "decide", "continue", "return",
    "remember", "forget", "promise", "leave", "stay", "arrive", "enter",
    "open", "close", "bring", "become", "walk", "study", "live", "play",
    "need", "want", "call", "sleep", "work", "read", "write", "speak",
    "arrive", "ask", "visit", "enjoy", "clean", "start", "teach", "learn",
    # silent-e verbs (past tense adds only -d: like->liked, love->loved)
    "like", "love", "move", "hope", "notice", "change", "provide",
    "receive", "believe", "invite", "describe", "improve", "include",
    "produce", "reduce", "refuse", "save", "smile", "dance", "face",
    "place", "race", "share", "prepare", "compare", "create", "complete",
    "increase", "decrease", "exercise", "balance", "praise", "achieve",
    "agree", "argue", "ensure", "imagine", "introduce", "manage",
    "mention", "observe", "participate", "announce", "convince",
    "examine", "explore", "practice", "apologize", "realize", "recognize",
    "arrange", "escape", "influence", "communicate", "celebrate",
    "educate", "relate", "separate", "translate", "emphasize", "tie",
    "die", "lie", "hate", "care", "share",
    # single-syllable verbs with doubled final consonants (stop->stopped)
    "stop", "plan", "skip", "drop", "shop", "beg", "grab", "rub", "hug",
    "jog", "pat", "nod", "tip", "wrap", "swap", "trap", "clap", "dip",
    "step", "pop", "rob", "spot", "slip", "log", "plot", "trim", "slap",
    "snap", "pet", "chat", "trip", "mop", "brag", "bar", "bat", "cop",
    "crop", "drum", "drag", "flag", "flip", "grip", "grin", "hop", "jam",
    "kiss", "map", "nag", "pin", "plug", "rag", "scan", "ship", "slam",
    "smug", "stab", "stir", "strip", "tan", "tap", "tag", "twig", "zip",
}

# A modal/auxiliary before "and" keeps the coordinated verb in the base form
# ("we will meet again tomorrow and finish") — shared by the tense rules.
_MODAL_IN_BETWEEN_RE = re.compile(
    r"\b(will|would|shall|should|can|could|may|might|must|"
    r"do|does|did|have|has|had|to)\b", re.I)

# Wrongly "-ed"-inflected irregular verbs ("flyed", "forgetted" ...).
_IRREGULAR_PAST_ED = {
    "flyed": "flew", "forgetted": "forgot", "runned": "ran", "catched": "caught",
    "buyed": "bought", "breaked": "broke", "taked": "took", "goed": "went",
    "sitted": "sat", "swimmed": "swam", "swimed": "swam", "singed": "sang",
    "drinked": "drank", "throwed": "threw", "drawed": "drew", "writed": "wrote",
    "speaked": "spoke", "telled": "told", "selled": "sold", "teached": "taught",
    "winned": "won", "standed": "stood", "sleeped": "slept", "feeled": "felt",
    "leaved": "left", "meeted": "met", "keeped": "kept", "payed": "paid",
    "sayed": "said", "finded": "found", "getted": "got", "hitted": "hit",
    "hurted": "hurt", "cutted": "cut", "putted": "put", "eated": "ate",
    "comed": "came", "maked": "made", "haved": "had", "begined": "began",
    "chosed": "chose", "losed": "lost", "stoled": "stole", "broked": "broke",
    "seed": "saw", "feeded": "fed", "seed": "saw", "knewed": "knew",
    "selled": "sold",
}

# Common learner misspellings of verbs/nouns ("slepping", "wonderfull").
_COMMON_TYPO_FIXES = {
    "slepping": "sleeping", "sleepingg": "sleeping", "wonderfull": "wonderful",
    "wonderfuly": "wonderfully", "beautifull": "beautiful", "succesfull": "successful",
    "occured": "occurred", "untill": "until", "littel": "little", "freinds": "friends",
    "becuase": "because", "recieve": "receive", "tommorow": "tomorrow",
    "morroco": "morocco", "definately": "definitely", "familar": "familiar",
    "govenment": "government", "enviroment": "environment", "assistant": "assistant",
}

# Wrong pluralisations of irregular nouns ("peoples", "childrens" ...).
_IRREGULAR_PLURAL_FORMS = {
    "peoples": "people", "persons": "people", "childrens": "children",
    "childs": "children", "childes": "children", "mans": "men", "mens": "men",
    "womans": "women", "womens": "women", "foots": "feet", "feets": "feet",
    "tooths": "teeth", "mouses": "mice", "sheeps": "sheep", "deers": "deer",
    "fishs": "fish", "gooses": "geese", "mices": "mice", "oxes": "oxen",
}

# Non-standard reflexive pronouns.
_REFLEXIVE_FORMS = {
    "theirselves": "themselves", "theirselfs": "themselves", "themselfs": "themselves",
    "hisself": "himself", "hershelves": "herself", "youself": "yourself",
    "youreself": "yourself", "meself": "myself", "ourselves": "ourselves",
    "ourselfs": "ourselves",
}

# Forms that cannot follow a negated auxiliary ("don't likes" -> "don't like").
_DONT_BASE = {
    "likes": "like", "liked": "like", "wants": "want", "wanted": "want",
    "needs": "need", "needed": "need", "goes": "go", "went": "go", "knows": "know",
    "knew": "know", "does": "do", "did": "do", "has": "have", "had": "have",
    "came": "come", "took": "take", "takes": "take", "saw": "see", "seen": "see",
    "said": "say", "says": "say", "made": "make", "got": "get", "gotten": "get",
    "played": "play", "works": "work", "worked": "work", "studies": "study",
    "studied": "study", "watches": "watch", "watched": "watch", "feels": "feel",
    "felt": "feel", "tells": "tell", "told": "tell", "thinks": "think",
    "thought": "think", "eats": "eat", "ate": "eat", "runs": "run", "ran": "run",
}

# Base verb -> -ing form used by the "be + base" rule ("I am go").
_BE_BASE_ING = {
    "go": "going", "come": "coming", "eat": "eating", "play": "playing",
    "run": "running", "work": "working", "study": "studying", "read": "reading",
    "write": "writing", "take": "taking", "make": "making", "watch": "watching",
    "walk": "walking", "swim": "swimming", "sing": "singing", "dance": "dancing",
    "drive": "driving", "sleep": "sleeping", "talk": "talking", "speak": "speaking",
    "listen": "listening", "learn": "learning", "teach": "teaching",
    "find": "finding", "get": "getting", "see": "seeing", "shower": "showering",
    "eat": "eating", "sit": "sitting", "shop": "shopping", "travel": "traveling",
    "wait": "waiting", "fly": "flying", "swim": "swimming", "visit": "visiting",
    "use": "using", "help": "helping", "call": "calling", "tell": "telling",
    "send": "sending", "give": "giving", "buy": "buying", "cook": "cooking",
}

# Base verb -> past participle, for "have/has/had + base" ("have ever spend").
_HAVE_BASE_PARTICIPLE = {
    "go": "gone", "do": "done", "see": "seen", "eat": "eaten", "write": "written",
    "speak": "spoken", "break": "broken", "choose": "chosen", "spend": "spent",
    "send": "sent", "build": "built", "buy": "bought", "bring": "brought",
    "catch": "caught", "teach": "taught", "think": "thought", "know": "known",
    "get": "gotten", "give": "given", "find": "found", "feel": "felt",
    "sleep": "slept", "keep": "kept", "leave": "left", "meet": "met",
    "lose": "lost", "win": "won", "run": "run", "sing": "sung", "drink": "drunk",
    "swim": "swum", "fly": "flown", "ride": "ridden", "wear": "worn",
    "throw": "thrown", "draw": "drawn", "forget": "forgotten",
    "understand": "understood", "tell": "told", "sell": "sold", "cut": "cut",
    "put": "put", "hurt": "hurt", "hit": "hit", "cost": "cost", "take": "taken",
    "make": "made", "come": "come",
}

# Past-tense signals that set a narrative in the past.
_PAST_SIGNAL_RE = re.compile(
    r"\b(went|saw|came|took|made|said|told|was|were|had|did|got|gave|knew|met|"
    r"left|felt|taught|bought|brought|found|forgot|flew|ate|ran|sat|stood|wrote|"
    r"spoke|missed|enjoyed|arrived|started|finished|talked|walked|played|helped|"
    r"called|visited|wanted|liked)\b", re.I)

# Words that mark the present/habitual and veto past-tense suggestions.
_HABITUAL_RE = re.compile(
    r"\b(every\s+day|everyday|every\s+\w+|always|usually|often|sometimes|never|"
    r"generally|normally|now|today|tomorrow|currently|rarely|frequent(ly)?)\b", re.I)

# Invariant verbs whose past tense is spelled identically to the present
# ("he read", "he put"). Never fabricate a visible past form for them.
_INVARIANT_VERBS = {
    "read", "put", "cut", "set", "hit", "cost", "hurt", "let",
    "spread", "bet", "upset", "cast", "split",
}

# Known plural noun forms (dictionary plurals + high-frequency vocabulary)
# used by the plural/agreement rules to avoid false positives on singular
# words that merely end in '-s' ("news", "bus", "status", "glass").
_KNOWN_PLURALS = set(_PLURAL_NOUNS.values())
_COMMON_PLURALS = _KNOWN_PLURALS | {
    "computers", "teachers", "parents", "animals", "things", "words",
    "reasons", "rooms", "tables", "chairs", "doors", "windows", "bags",
    "shoes", "kids", "rules", "letters", "phones", "pictures", "photos",
    "games", "movies", "songs", "messages", "emails", "calls", "jobs",
    "numbers", "colors", "colours", "names", "places", "countries",
    "languages", "stories", "bottles", "cups", "plates", "keys",
    "flowers", "trees", "birds", "fishes", "efforts", "goals", "rights",
    "achievements", "skills", "habits", "hours", "books",
}


def _load_json(name: str) -> Dict:
    try:
        with open(os.path.join(_DATA_DIR, name), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _regular_past(base: str) -> str:
    """Simple past of a REGULAR verb ('try'->'tried', 'stop'->'stopped')."""
    low = base.lower()
    if len(low) > 2 and low.endswith("y") and low[-2] not in "aeiouy":
        return low[:-1] + "ied"
    if low.endswith("e"):
        return low + "d"
    if (len(low) >= 3 and low[-1] not in "aeiouy" and low[-1] not in "xw"
            and low[-2] in "aeiou" and low[-3] not in "aeiou"):
        return low + low[-1] + "ed"
    return low + "ed"


_3SG_BASE = {v: k for k, v in _THIRD_SINGULAR.items()}
_IRREGULAR_BASE_BY_PAST = {v: k for k, v in _IRREGULAR_PAST.items()}
_BASE_BY_PARTICIPLE = {}
for _k, _v in _PAST_PARTICIPLE.items():
    _base = _IRREGULAR_BASE_BY_PAST.get(_k, _k)
    _BASE_BY_PARTICIPLE[_v] = _base
    _BASE_BY_PARTICIPLE.setdefault(_k, _base)


def _past_of_verb(word: str) -> str:
    """Past tense of a base or 3rd-person-singular present verb form
    ('calls'->'called', 'goes'->'went', 'study'->'studied', 'starts'->'started')."""
    low = word.lower()
    spec = _3SG_BASE.get(low)
    if spec:
        base = spec
    elif low.endswith("ies") and len(low) > 3:
        base = low[:-3] + "y"
    elif low.endswith("es"):
        base = low[:-2]
    elif low.endswith("s") and not low.endswith("ss"):
        base = low[:-1]
    else:
        base = low
    if base in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[base]
    return _regular_past(base)


def _is_verb_form(word: str) -> bool:
    """Is ``word`` a base or 3rd-person-singular present verb we know?"""
    low = word.lower()
    if low in _PASTABLE_VERBS or _IRREGULAR_BASE_BY_PAST.get(low):
        return True
    if low.endswith("ies") and low[:-3] + "y" in _PASTABLE_VERBS:
        return True
    if low.endswith("es") and low[:-2] in _PASTABLE_VERBS:
        return True
    if low.endswith("s") and not low.endswith("ss") and low[:-1] in _PASTABLE_VERBS:
        return True
    return False


def _regular_base_from_past(word: str) -> str:
    """Recover the base form of a REGULAR-looking past-tense verb, used when the
    base form is required in context ("didn't liked" -> "like", "couldn't
    stopped" -> "stop"). Handles silent-e verbs ('liked'->'like'),
    y->i ('studied'->'study') and doubled final consonants ('stopped'->'stop').
    Never strips a doubled consonant blindly, so 'rolled' stays 'roll'."""
    low = word.lower()
    if low.endswith("ed"):
        if low[:-1].endswith("e") and _is_verb_form(low[:-1]):
            return low[:-1]
        if low.endswith("ied"):
            return low[:-3] + "y"
        base = low[:-2]
        if (len(base) >= 2 and base[-1] == base[-2]
                and _is_verb_form(base[:-1])):
            base = base[:-1]
        return base if len(base) >= 2 else low
    return low


def _gerund_base(word: str) -> Optional[str]:
    """Recover the base form of a gerund for modal contexts ('going'->'go',
    'eating'->'eat', 'running'->'run', 'studying'->'study'). Returns None if the
    word doesn't look like a gerund or its base isn't a real verb. Handles
    doubled consonants ('running'->'run') and y->i ('studying'->'study')."""
    low = word.lower()
    if not low.endswith("ing") or len(low) <= 4:
        return None
    base = low[:-3]
    candidates = [base, base[:-1]]
    # 'studying' = study + ing (y->i): restore the trailing y
    if low.endswith("ying") and len(low) > 5:
        candidates.append(base[:-1] + "y")
    for candidate in candidates:
        if len(candidate) >= 2 and _is_verb_form(candidate):
            return candidate
    return None


class RuleDetector:
    """Deterministic, high-precision candidate generator."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = data_dir or _DATA_DIR
        spell = _load_json("spelling_dictionary.json")
        self.misspellings: Dict[str, str] = {
            k.lower(): v for k, v in (spell.get("common_misspellings") or {}).items()
        }
        self.valid_words: Set[str] = {
            w.lower() for w in (spell.get("valid_words") or [])
        }
        conf = _load_json("confusing_words.json")
        self.confusing: Dict = conf.get("confusing_words") or {}
        self._hc = HighConfidenceDetector() if _HC_AVAILABLE else None

    # ------------------------------------------------------------------ public
    def generate(self, text: str) -> List[Dict]:
        """Return a de-duplicated list of candidate dicts."""
        if not text or not text.strip():
            return []
        out: List[Dict] = []
        for fn in (
            self._spelling_dictionary,
            self._repeated_letters,
            self._repeated_words,
            self._high_confidence,
            self._subject_verb_agreement,
            self._be_plus_base,
            self._be_agreement,
            self._aux_past_participle,
            self._be_plus_past,
            self._did_plus_past,
            self._modal_past_base,
            self._double_negative,
            self._tense_past_marker,
            self._tense_past_marker_after,
            self._aux_base_participle,
            self._narrative_past,
            self._tense_coordination,
            self._articles,
            self._prepositions,
            self._homophones,
            self._count_plurals,
            self._irregular_plural,
            self._missing_auxiliary,
            self._reflexive_subject,
            self._reflexive_irregular,
            self._dont_base,
            self._day_month_cap,
            self._irregular_past_ed,
            self._common_typos,
            self._word_confusions,
            self._word_form,
            self._person_noun_be,
            self._missing_apostrophe,
            self._who_were,
            self._singular_noun_were,
            self._plural_was,
            self._tense_consistency,
            self._capitalization,
            self._punctuation,
        ):
            try:
                out.extend(fn(text))
            except Exception:  # never let one rule break the pipeline
                continue
        return _dedup(out)

    # --------------------------------------------------------------- spelling
    def _spelling_dictionary(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(_WORD, text):
            w = m.group(0)
            low = w.lower()
            if low in self.misspellings:
                corr = self.misspellings[low]
                if w[0].isupper() and corr:
                    corr = corr[0].upper() + corr[1:]
                out.append(self._cand(w, corr, "spelling", m, 0.97,
                                      "SPELLING_DICT",
                                      f"'{w}' is a common misspelling of '{corr}'."))
        return out

    def _repeated_letters(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(r"\b([A-Za-z]*?)([A-Za-z])\2{2,}([A-Za-z]*?)\b", text):
            word = m.group(0)
            if len(word) < 4:
                continue
            for reduce_to in (2, 1):
                cand = re.sub(r"([A-Za-z])\1{2,}",
                              lambda mm, n=reduce_to: mm.group(1) * n, word)
                if cand.lower() in self.valid_words and cand.lower() != word.lower():
                    if word[0].isupper():
                        cand = cand[0].upper() + cand[1:]
                    out.append(self._cand(word, cand, "spelling", m, 0.92,
                                          "REPEATED_LETTERS",
                                          f"'{word}' looks like a typing mistake for '{cand}'."))
                    break
        return out

    # ------------------------------------------------------------ repetition
    def _repeated_words(self, text: str) -> List[Dict]:
        out = []
        # "had had" is the past-perfect of "have" ("They had had enough") and
        # is legitimately ambiguous — never auto-flag it.
        _ALLOWED_DOUBLES = {
            "had had", "that that", "do do", "this this",
            "so so", "much much", "many many",
        }
        for m in re.finditer(r"\b(\w+)(\s+)\1\b", text, re.I):
            word = m.group(1)
            start = m.start()
            end = m.end()
            punct = re.sub(r"[^a-z0-9\s]", "", f"{word} {word}").lower()
            if punct in _ALLOWED_DOUBLES:
                continue
            correct = word
            out.append(self._cand(text[start:end], correct, "redundancy", m, 0.97,
                                  "REPEATED_WORD",
                                  f"'{word}' is repeated; remove the duplicate."))
        return out

    # -------------------------------------------------------- high-confidence
    def _high_confidence(self, text: str) -> List[Dict]:
        if self._hc is None:
            return []
        out = []
        for c in self._hc.detect(text):
            cat = getattr(c.category, "value", str(c.category)) if hasattr(c.category, "value") else str(c.category)
            out.append({
                "wrong": c.original,
                "correct": c.replacement,
                "type": _normalize_type(cat),
                "start": c.start,
                "end": c.end,
                "confidence": float(getattr(c, "raw_confidence", 0.9)),
                "source": "rule",
                "rule_id": getattr(c, "rule_id", "HC_RULE"),
                "message": getattr(c, "message", ""),
            })
        return out

    # ----------------------------------------------------- subject-verb agr.
    def _subject_verb_agreement(self, text: str) -> List[Dict]:
        out = []
        # he/she/it + base verb → 3rd person singular
        verbs = "|".join(_THIRD_SINGULAR)
        for m in re.finditer(rf"\b(he|she|it)\s+({verbs})\b", text, re.I):
            verb = m.group(2).lower()
            past_ctx = _PAST_SIGNAL_RE.search(text[max(0, m.start(2) - 140):m.start(2) + 24])
            if past_ctx:
                # in a past narrative "he say" -> "he said", "he suggest" ->
                # "he suggested"; leave the tense to NARRATIVE_PAST.
                continue
            out.append(self._cand(m.group(2), _THIRD_SINGULAR[verb], "subject_verb",
                                  _span(m, 2), 0.9, "SVA_3SG",
                                  f"'{m.group(1)}' is singular, so use '{_THIRD_SINGULAR[verb]}'."))
        # i/we/you/they + 3rd-person singular → plural form
        plurals = "|".join(_PLURAL_FORM)
        for m in re.finditer(rf"\b(i|we|you|they)\s+({plurals})\b", text, re.I):
            verb = m.group(2).lower()
            out.append(self._cand(m.group(2), _PLURAL_FORM[verb], "subject_verb",
                                  _span(m, 2), 0.9, "SVA_PLURAL",
                                  f"'{m.group(1)}' requires the base form '{_PLURAL_FORM[verb]}'."))
        # I has → I have ; she/he/it have → has
        for m in re.finditer(r"\b(i)\s+(has)\b", text, re.I):
            out.append(self._cand(m.group(2), "have", "subject_verb", _span(m, 2), 0.92,
                                  "SVA_HAVE", "'I' takes 'have'."))
        for m in re.finditer(r"\b(he|she|it)\s+(have)\b", text, re.I):
            out.append(self._cand(m.group(2), "has", "subject_verb", _span(m, 2), 0.92,
                                  "SVA_HAVE", f"'{m.group(1)}' takes 'has'."))
        # he/she/it + don't → doesn't (learner error)
        for m in re.finditer(r"\b(he|she|it)\s+(don't|do not)\b", text, re.I):
            word = text[m.start(2):m.end(2)]
            repl = "doesn't" if word.lower() == "don't" else "does not"
            out.append(self._cand(word, repl, "subject_verb", _span(m, 2), 0.9,
                                  "SVA_DOESNT", f"'{m.group(1)}' takes '{repl}'."))
        return out

    def _be_plus_base(self, text: str) -> List[Dict]:
        out = []
        bases = "|".join(_BE_BASE_ING)
        for m in re.finditer(rf"\b(am|is|are|was|were|be)\s+({bases})\b", text, re.I):
            be, base = m.group(1).lower(), m.group(2).lower()
            if be == "be":
                continue
            ing = _BE_BASE_ING[base]
            out.append(self._cand(m.group(2), ing, "verb_form", _span(m, 2), 0.85,
                                  "BE_PLUS_BASE",
                                  f"After '{be}', use the -ing form '{ing}'."))
        return out

    def _be_agreement(self, text: str) -> List[Dict]:
        out = []
        wrong = {"i": "am", "he": "is", "she": "is", "it": "is",
                 "we": "are", "you": "are", "they": "are"}
        for m in re.finditer(r"\b(i|he|she|it|we|you|they)\s+(am|is|are|was|were)\b", text, re.I):
            pron, verb = m.group(1).lower(), m.group(2).lower()
            target_present = wrong[pron]
            target_past = "was" if pron in ("i", "he", "she", "it") else "were"
            if verb in ("am", "is", "are"):
                if verb != target_present:
                    out.append(self._cand(m.group(2), target_present, "subject_verb",
                                          _span(m, 2), 0.92, "BE_AGREEMENT",
                                          f"'{pron}' takes '{target_present}'."))
            else:  # was/were
                if verb != target_past:
                    out.append(self._cand(m.group(2), target_past, "subject_verb",
                                          _span(m, 2), 0.9, "BE_AGREEMENT_PAST",
                                          f"'{pron}' takes '{target_past}'."))
        return out

    # ------------------------------------------------------------- verb form
    def _aux_past_participle(self, text: str) -> List[Dict]:
        out = []
        past = "|".join(_PAST_PARTICIPLE)
        for m in re.finditer(rf"\b(has|have|had|is|are|was|were|am|be|being)\s+({past})\b", text, re.I):
            aux, verb = m.group(1).lower(), m.group(2).lower()
            part = _PAST_PARTICIPLE[verb]
            if aux in _HAVE_FORMS and part != verb:
                out.append(self._cand(m.group(2), part, "verb_form", _span(m, 2), 0.95,
                                      "AUX_PAST_PARTICIPLE",
                                      f"After '{aux}', use the past participle '{part}'."))
        return out

    def _be_plus_past(self, text: str) -> List[Dict]:
        """'They is went home' → 'They went home' (drop the be-verb).

        A verb that is itself a valid past participle (identity participle, e.g.
        "sold", "left") is a PASSIVE construction ("were sold") and must NOT be
        rewritten."""
        out = []
        past = "|".join(_PAST_SIMPLE)
        for m in re.finditer(rf"\b(is|are|was|were|am)\s+({past})\b", text, re.I):
            word = m.group(2).lower()
            part = _PAST_PARTICIPLE.get(word)
            if part is not None and part == word:
                continue  # passive, not a learner error
            out.append(self._cand(m.group(0), m.group(2), "verb_form", _span(m, 0), 0.9,
                                  "BE_PLUS_PAST",
                                  f"Remove '{m.group(1)}'; the past tense '{m.group(2)}' stands alone."))
        return out

    def _did_plus_past(self, text: str) -> List[Dict]:
        out = []
        _base_of = {v: k for k, v in _IRREGULAR_PAST.items()}
        pat = "|".join(sorted(_base_of, key=len, reverse=True))
        for m in re.finditer(rf"\b(did|didn'?t|did not)\s+({pat})\b", text, re.I):
            base = _base_of[m.group(2).lower()]
            out.append(self._cand(m.group(2), base, "verb_form", _span(m, 2), 0.92,
                                  "DIDNT_PAST_FORM",
                                  f"After '{m.group(1)}', use the base form '{base}'."))
        # regular verbs: "didn't wanted" -> "didn't want"
        for m in re.finditer(r"\b(did|didn'?t|did not)\s+([a-z]{2,}ed)\b", text, re.I):
            word = m.group(2)
            if word.lower() in _IRREGULAR_PAST_ED:
                continue
            base = _regular_base_from_past(word)
            if word.lower() == base.lower():
                continue
            out.append(self._cand(word, base, "verb_form", _span(m, 2), 0.85,
                                  "DIDNT_PAST_FORM",
                                  f"After '{m.group(1)}', use the base form '{base}'."))
        return out

    def _modal_past_base(self, text: str) -> List[Dict]:
        """'I couldn't understood the lesson.' -> modal verbs take the base form
        ('couldn't understand'). Also 'should have went' -> 'should have gone'."""
        out = []
        _MODAL = (r"\b(can|could|may|might|must|shall|should|will|would|"
                  r"can't|cannot|couldn't|shouldn't|wouldn't|won't|mustn't|"
                  r"daren't|needn't)\b")
        for past, base in _BASE_BY_PARTICIPLE.items():
            for m in re.finditer(rf"{_MODAL}\s+({re.escape(past)})\b", text, re.I):
                if base.lower() == m.group(2).lower():
                    continue
                out.append(self._cand(m.group(2), base, "verb_form", _span(m, 2), 0.9,
                                      "MODAL_PAST_BASE",
                                      f"After '{m.group(1)}', use the base form '{base}'."))
        # regular verbs: "couldn't wanted" -> "couldn't want"
        for m in re.finditer(rf"{_MODAL}\s+([a-z]{{2,}}ed)\b", text, re.I):
            word = m.group(2)
            low = word.lower()
            if low in _IRREGULAR_PAST_ED or low in _PAST_SIMPLE:
                continue
            base = _regular_base_from_past(word)
            if low == base.lower():
                continue
            out.append(self._cand(word, base, "verb_form", _span(m, 2), 0.85,
                                  "MODAL_PAST_BASE",
                                  f"After '{m.group(1)}', use the base form '{base}'."))
        # gerund after modal: "would going" -> "would go", "will eating" -> "will eat"
        for m in re.finditer(rf"{_MODAL}\s+([a-z]{{2,}}ing)\b", text, re.I):
            ger = m.group(2)
            low = ger.lower()
            base = _gerund_base(low)
            if base is None or base == low:
                continue
            out.append(self._cand(ger, base, "verb_form", _span(m, 2), 0.85,
                                  "MODAL_GERUND_BASE",
                                  f"After '{m.group(1)}', use the base form '{base}'."))
        return out

    def _double_negative(self, text: str) -> List[Dict]:
        out = []
        neg = r"don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|ain'?t|cannot|can'?t|never|nobody|nothing"
        repl = {"no": "any", "nothing": "anything", "never": "ever",
                "nobody": "anybody", "nowhere": "anywhere"}
        words = "|".join(repl)
        for m in re.finditer(rf"\b({neg})\s+[a-z']+\s+({words})\b", text, re.I):
            out.append(self._cand(m.group(2), repl[m.group(2).lower()], "double_negative",
                                  _span(m, 2), 0.85, "DOUBLE_NEGATIVE",
                                  "Double negatives are incorrect; use a single negative."))
        return out

    def _tense_past_marker(self, text: str) -> List[Dict]:
        out = []
        bases = "|".join(_IRREGULAR_PAST)
        # A modal/auxiliary immediately before the base requires the base form
        # ("could have arrived earlier" -> NOT "had arrived earlier").
        _AUX = {"have", "has", "had", "will", "would", "shall", "should",
                "can", "could", "may", "might", "must", "do", "does", "did", "to",
                # negative contractions — a base verb after these is already
                # correct ("she doesn't eat", "they couldn't go")
                "don't", "doesn't", "didn't", "can't", "cannot", "won't",
                "wouldn't", "couldn't", "shouldn't", "mustn't", "ain't"}
        for marker in _PAST_MARKERS.finditer(text):
            window = text[max(0, marker.start() - 120): marker.start()]
            for m in re.finditer(rf"\b({bases})\b", window, re.I):
                tokens = m.string[:m.start(1)].rstrip().split()
                if tokens and tokens[-1].strip(".,;:!?").lower() in _AUX:
                    continue
                skip_tail = [t.strip(".,;:!?").lower() for t in tokens][-3:]
                # "he don't (usually) get sick": a negative auxiliary anywhere
                # in the previous 3 words means the base verb is CORRECT.
                if any(t in ("don't", "doesn't", "didn't", "can't", "cannot",
                             "won't", "wouldn't", "couldn't", "shouldn't",
                             "mustn't", "ain't", "not", "never") for t in skip_tail):
                    continue
                abs_start = max(0, marker.start() - 120) + m.start(1)
                abs_end = abs_start + len(m.group(1))
                out.append({
                    "wrong": m.group(1),
                    "correct": _IRREGULAR_PAST[m.group(1).lower()],
                    "type": "tense",
                    "start": abs_start,
                    "end": abs_end,
                    "confidence": 0.85,
                    "source": "rule",
                    "rule_id": "TENSE_PAST_MARKER",
                    "message": f"With '{marker.group(0)}', use the past tense '{_IRREGULAR_PAST[m.group(1).lower()]}'.",
                })
        return out

    def _tense_past_marker_after(self, text: str) -> List[Dict]:
        """'Last sunday my friends and I go' -> the marker comes FIRST and the
        base verb AFTER it. Catch irregular base verbs a few words after a
        past-time marker in the same sentence."""
        out = []
        bases = "|".join(_IRREGULAR_PAST)
        _AUX = {"have", "has", "had", "will", "would", "shall", "should",
                "can", "could", "may", "might", "must", "do", "does", "did", "to",
                # negative contractions — a base verb after these is already
                # correct ("she doesn't eat", "they couldn't go")
                "don't", "doesn't", "didn't", "can't", "cannot", "won't",
                "wouldn't", "couldn't", "shouldn't", "mustn't", "ain't"}
        for marker in _PAST_MARKERS.finditer(text):
            after = text[marker.end():marker.end() + 80]
            cut = min([i for i in (after.find("."), after.find("!"), after.find("?"),
                                   after.find(";")) if i != -1] or [len(after)])
            after = after[:cut]
            for m in re.finditer(rf"\b({bases})\b", after, re.I):
                tokens = m.string[:m.start(1)].rstrip().split()
                if tokens and tokens[-1].strip(".,;:!?").lower() in _AUX:
                    continue
                if len(tokens) >= 2 \
                        and tokens[-2].strip(".,;:!?").lower() in _AUX \
                        and tokens[-1].strip(".,;:!?").lower() in ("not", "never"):
                    continue
                abs_start = marker.end() + m.start(1)
                if _HABITUAL_RE.search(text[max(0, abs_start - 40):abs_start + 24]):
                    continue
                out.append({
                    "wrong": m.group(1),
                    "correct": _IRREGULAR_PAST[m.group(1).lower()],
                    "type": "tense",
                    "start": abs_start,
                    "end": abs_start + len(m.group(1)),
                    "confidence": 0.8,
                    "source": "rule",
                    "rule_id": "TENSE_PAST_MARKER_AFTER",
                    "message": f"With '{marker.group(0)}', use the past tense '{_IRREGULAR_PAST[m.group(1).lower()]}'.",
                })
        return out

    def _aux_base_participle(self, text: str) -> List[Dict]:
        """'have/has/had + base irregular verb' -> past participle ('have ever
        spend' -> 'have ever spent')."""
        out = []
        bases = "|".join(_HAVE_BASE_PARTICIPLE)
        for m in re.finditer(
                rf"\b(has|have|had)\s+(ever|never|just|already|always|also)?\s*({bases})\b",
                text, re.I):
            aux, verb = m.group(1).lower(), m.group(3).lower()
            part = _HAVE_BASE_PARTICIPLE[verb]
            if part == verb:
                continue
            out.append(self._cand(m.group(3), part, "verb_form", _span(m, 3), 0.9,
                                  "HAVE_PAST_PARTICIPLE",
                                  f"After '{aux}', use the past participle '{part}'."))
        return out

    def _narrative_past(self, text: str) -> List[Dict]:
        """In a past-tense narrative, present-tense irregular verbs are usually
        past-tense errors. Uses a sticky narrative state: once past, it stays
        past until a present-tense adverb resets it."""
        out = []
        bases = "|".join(_IRREGULAR_PAST)
        _AUX = {"have", "has", "had", "will", "would", "shall", "should",
                "can", "could", "may", "might", "must", "do", "does", "did", "to",
                # negative contractions — a base verb after these is already
                # correct ("she doesn't eat", "they couldn't go")
                "don't", "doesn't", "didn't", "can't", "cannot", "won't",
                "wouldn't", "couldn't", "shouldn't", "mustn't", "ain't"}
        _RESET = {"now", "today", "tomorrow", "present", "currently", "right now",
                  "daily", "everyday"}
        _ALREADY_ROUGH = _PAST_SIMPLE | set(_PAST_PARTICIPLE) | {
            "was", "were", "is", "are", "am", "be", "been", "being", "been",
            "became", "become"}
        narrative_past = False
        for sm in re.finditer(r"[^.!?\n]+[.!?]*", text):
            sent, sbase = sm.group(0), sm.start()
            sent_past = narrative_past or _PAST_MARKERS.search(sent) is not None \
                or _PAST_SIGNAL_RE.search(sent) is not None
            if sent_past:
                for m in re.finditer(rf"\b({bases})\b", sent, re.I):
                    verb = m.group(1).lower()
                    tokens = sent[:m.start(1)].rstrip().split()
                    if tokens and tokens[-1].strip(".,;:!?").lower() in _AUX:
                        continue
                    # "I could not understand" — the AUX is two tokens back,
                    # separated by a negator; the base form stays ("could not
                    # understand", "might not come").
                    if len(tokens) >= 2 \
                            and tokens[-2].strip(".,;:!?").lower() in _AUX \
                            and tokens[-1].strip(".,;:!?").lower() in ("not", "never"):
                        continue
                    # "We take a SHORT BREAK" / "have a drink" — the word is a
                    # NOUN when a determiner sits 1-2 words back ("a short
                    # break", "a break"), never rewrite it to its past form.
                    if verb in _NOUNY_IRREGULAR:
                        lookback = sent[max(0, m.start(1) - 30):m.start(1)]
                        if any(t.strip(".,;:!?()\"'").lower() in _DETERMINERS
                               for t in lookback.split()[-2:]):
                            continue
                    # Habitual adverbs rescue a present-tense base VERB ("she
                    # usually go", "they never leave"). A genuine habit marker
                    # sits BEFORE the verb. An adverb AFTER the verb only
                    # rescues if it does NOT open a nested clause ("my friend
                    # say [that she has never seen]" -> habitual 'never' belongs
                    # to the embedded clause and must NOT rescue 'say').
                    before = sent[max(0, m.start(1) - 40):m.start(1)]
                    if _HABITUAL_RE.search(before):
                        continue
                    after = sent[m.end(1):m.end(1) + 40].lstrip()
                    if _HABITUAL_RE.search(after) and not re.match(
                            r"(?i)^(that|which|who|whom|where|when|while|"
                            r"after|before)\b", after):
                        continue
                    after = sent[m.end(1):m.end(1) + 40].lstrip().split(None, 1)
                    if verb in ("have", "has", "had"):
                        nxt = (after[0].strip(".,;:!?") if after else "").lower()
                        if (nxt in _PAST_PARTICIPLE or nxt in _PAST_SIMPLE
                                or nxt in ("ever", "never", "just", "already",
                                           "always", "also", "been")):
                            continue
                    out.append({
                        "wrong": m.group(1),
                        "correct": _IRREGULAR_PAST[verb],
                        "type": "tense",
                        "start": sbase + m.start(1),
                        "end": sbase + m.end(1),
                        "confidence": 0.8,
                        "source": "rule",
                        "rule_id": "NARRATIVE_PAST",
                        "message": f"The narrative is in the past; use '{_IRREGULAR_PAST[verb]}'.",
                    })
                # 3rd-person + regular/base verb -> past ('he calls' -> 'he called',
                # 'he start' -> 'he started'); a 3sg -s ending is handled through
                # _past_of_verb so 'calls' becomes 'called', never 'callsed'.
                for m in re.finditer(r"\b(he|she|it)\s+([a-z]{2,})\b", sent, re.I):
                    word, verb = m.group(2), m.group(2).lower()
                    if (_HABITUAL_RE.search(sent[max(0, m.start(2) - 40):m.end(2) + 24])
                            or verb in _IRREGULAR_PAST
                            or verb.endswith("ing") or verb in _AUX
                            or verb in _ALREADY_ROUGH or verb.endswith("ed")
                            or verb.endswith("ly") or verb in _INVARIANT_VERBS
                            or not _is_verb_form(verb)):
                        continue
                    past = _past_of_verb(verb)
                    if past == verb:
                        continue
                    out.append(self._cand(word, past, "tense",
                                          (sbase + m.start(2), sbase + m.end(2)), 0.8,
                                          "NARRATIVE_REGULAR",
                                          f"The narrative is in the past; use '{past}'."))
                # 3rd-person person-noun + base verb ('My friend explain ...').
                for m in re.finditer(rf"\b({_PERSON_NOUNS})\s+([a-z]{{2,}})\b", sent, re.I):
                    word, verb = m.group(2), m.group(2).lower()
                    if (_HABITUAL_RE.search(sent[max(0, m.start(2) - 50):m.end(2) + 24])
                            or verb in _IRREGULAR_PAST
                            or verb.endswith("ing") or verb in _AUX
                            or verb in _ALREADY_ROUGH or verb.endswith("ed")):
                        continue
                    if not _is_verb_form(verb):
                        continue
                    past = _past_of_verb(verb)
                    if past == verb:
                        continue
                    out.append(self._cand(word, past, "tense",
                                          (sbase + m.start(2), sbase + m.end(2)), 0.8,
                                          "NARRATIVE_PERSON_3SG",
                                          f"'{m.group(1)}' is singular; in past narration use '{past}'."))
                for m in re.finditer(r"\b(will)\b", sent, re.I):
                    out.append(self._cand(m.group(1), "would", "tense",
                                          (sbase + m.start(1), sbase + m.end(1)), 0.79,
                                          "NARRATIVE_WOULD",
                                          "In reported past narration, 'will' is usually 'would'."))
                # coordinated verb after 'and' in a past narrative: 'another
                # student come into the library and starts talking' ->
                # 'came ... and started'. A modal before it ('we will meet
                # again tomorrow and finish') keeps the base form.
                for m in re.finditer(
                        rf"\band\s+(?:(then|also|suddenly|quickly|immediately|still|"
                        rf"just|soon|later|finally|slowly)\s+)?([a-z]{{2,}})\b",
                        sent, re.I):
                    word, verb = m.group(2), m.group(2).lower()
                    if _MODAL_IN_BETWEEN_RE.search(sent[:m.start(2)]):
                        continue
                    if (verb in _PAST_SIMPLE or verb.endswith("ed")
                            or verb.endswith("ing") or verb in _AUX
                            or verb in _ALREADY_ROUGH):
                        continue
                    if not _is_verb_form(verb):
                        continue
                    past = _past_of_verb(verb)
                    if past == verb:
                        continue
                    out.append(self._cand(word, past, "tense",
                                          (sbase + m.start(2), sbase + m.end(2)), 0.8,
                                          "TENSE_COORD",
                                          f"The narrative is in the past; use '{past}'."))
            if _RESET & set(re.findall(_WORD, sent.lower())):
                narrative_past = False
            elif _PAST_SIGNAL_RE.search(sent):
                narrative_past = True
        return out

    def _tense_coordination(self, text: str) -> List[Dict]:
        """'The student came into the library and starts talking.' — a verb
        coordinated with a past verb stays in the past ('started'). Skips when a
        modal sits between them: 'we will meet again tomorrow and finish ...'."""
        out = []
        _ADV = r"(?:then|also|suddenly|quickly|immediately|still|just|soon|later|finally|slowly)\s+"
        past_list = sorted(_PAST_SIMPLE, key=len, reverse=True)
        for v1 in past_list:
            for m in re.finditer(
                    rf"\b({re.escape(v1)})\b([^.!?;]{{0,100}}?)\band\s+(?:{_ADV})?([a-z]{{2,}})\b",
                    text, re.I):
                between = m.group(2)
                word, verb = m.group(3), m.group(3).lower()
                if _MODAL_IN_BETWEEN_RE.search(between):
                    continue
                if verb in _PAST_SIMPLE or verb.endswith("ed"):
                    continue
                if not _is_verb_form(verb):
                    continue
                past = _past_of_verb(verb)
                if past == verb:
                    continue
                out.append(self._cand(word, past, "tense", _span(m, 3), 0.8,
                                      "TENSE_COORD",
                                      f"Coordinate with '{m.group(1)}': use the past '{past}'."))
        for m in re.finditer(
                rf"\b([a-z]{{3,}}ed)\b([^.!?;]{{0,100}}?)\band\s+(?:{_ADV})?([a-z]{{2,}})\b",
                text, re.I):
            v1, word, verb = m.group(1), m.group(3), m.group(3).lower()
            between = m.group(2)
            if _MODAL_IN_BETWEEN_RE.search(between):
                continue
            if verb in _PAST_SIMPLE or verb.endswith("ed"):
                continue
            if not _is_verb_form(verb):
                continue
            past = _past_of_verb(verb)
            if past == verb:
                continue
            out.append(self._cand(word, past, "tense", _span(m, 3), 0.8,
                                  "TENSE_COORD",
                                  f"Coordinate with '{v1}': use the past '{past}'."))
        return out

    # -------------------------------------------------------------- articles
    def _articles(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(rf"\b(a)\s+({_WORD})", text, re.I):
            nxt = m.group(2)
            low = nxt.lower()
            if low[:1] in "aeiou" and low not in _A_VOWEL_EXCEPTIONS:
                out.append(self._cand(m.group(1), "an", "article", _span(m, 1), 0.9,
                                      "ARTICLE_AN", f"Use 'an' before '{nxt}'."))
        for m in re.finditer(rf"\b(an)\s+({_WORD})", text, re.I):
            nxt = m.group(2)
            low = nxt.lower()
            if low[:1] not in "aeiou" and low not in _AN_CONSONANT_EXCEPTIONS:
                out.append(self._cand(m.group(1), "a", "article", _span(m, 1), 0.88,
                                      "ARTICLE_A", f"Use 'a' before '{nxt}'."))
        return out

    # ---------------------------------------------------------- prepositions
    def _prepositions(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(r"\b(in|on|at)\s+((?:monday|tuesday|wednesday|thursday|"
                             r"friday|saturday|sunday))\b", text, re.I):
            if m.group(1).lower() != "on":
                out.append(self._cand(m.group(1), "on", "preposition", _span(m, 1), 0.9,
                                      "PREP_WEEKDAY", f"Use 'on' before days of the week."))
        for m in re.finditer(r"\b(at|on)\s+(the\s+)?(morning|afternoon|evening)\b", text, re.I):
            out.append(self._cand(m.group(1), "in", "preposition", _span(m, 1), 0.88,
                                  "PREP_TIME_OF_DAY", "Use 'in' with parts of the day."))
        for m in re.finditer(r"\b(am)\s+(agree|disagree)\b", text, re.I):
            out.append(self._cand(m.group(0), m.group(2), "verb_form", _span(m, 0), 0.85,
                                  "AGREE_NO_BE", f"Use '{m.group(2)}' without 'am'."))
        return out

    # ------------------------------------------------------------ homophones
    def _homophones(self, text: str) -> List[Dict]:
        out = []
        for key, entry in self.confusing.items():
            if not isinstance(entry, dict):
                continue
            key_l = key.lower()
            rules = entry.get("rules") or []
            for rule in rules:
                correct = (rule.get("correct") or "").lower()
                if not correct or correct == key_l:
                    continue
                ctx = rule.get("context_words") or []
                if not ctx:
                    continue
                ctx_alt = "|".join(re.escape(c) for c in ctx)
                pat = rf"\b({re.escape(key_l)})\s+({ctx_alt})\b"
                for m in re.finditer(pat, text, re.I):
                    out.append(self._cand(key, correct, "word_choice",
                                          _span(m, 1), 0.9, f"HOMOPHONE_{key_l.upper()}",
                                          rule.get("reason", f"Consider '{correct}' here.")))
        return out

    # --------------------------------------------------------- count/plurals
    def _count_plurals(self, text: str) -> List[Dict]:
        out = []
        nums = "|".join(_NUMBER_WORDS)
        nouns = "|".join(_PLURAL_NOUNS)
        for m in re.finditer(rf"\b({nums})\s+({nouns})\b", text, re.I):
            noun = m.group(2).lower()
            out.append(self._cand(m.group(2), _PLURAL_NOUNS[noun], "plural",
                                  _span(m, 2), 0.8, "COUNT_PLURAL",
                                  f"'{m.group(1)}' needs the plural '{_PLURAL_NOUNS[noun]}'."))
        return out

    def _irregular_plural(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(_WORD, text):
            low = m.group(0).lower()
            if low in _IRREGULAR_PLURAL_FORMS:
                corr = _IRREGULAR_PLURAL_FORMS[low]
                out.append(self._cand(m.group(0), corr, "plural", m, 0.9,
                                      "IRREGULAR_PLURAL",
                                      f"'{m.group(0)}' is already plural; use '{corr}'."))
        return out

    def _reflexive_irregular(self, text: str) -> List[Dict]:
        out = []
        pat = r"\b(" + "|".join(_REFLEXIVE_FORMS) + r")\b"
        for m in re.finditer(pat, text, re.I):
            w = m.group(1)
            corr = _REFLEXIVE_FORMS[w.lower()]
            out.append(self._cand(w, corr, "pronoun", m, 0.9, "REFLEXIVE_FORM",
                                  f"Use '{corr}'."))
        return out

    def _dont_base(self, text: str) -> List[Dict]:
        out = []
        forms = "|".join(_DONT_BASE)
        for m in re.finditer(rf"\b(don'?t|do not|doesn'?t|does not)\s+({forms})\b",
                             text, re.I):
            word = m.group(2)
            corr = _DONT_BASE[word.lower()]
            if word[0].isupper():
                corr = corr[0].upper() + corr[1:]
            out.append(self._cand(word, corr, "verb_form", _span(m, 2), 0.85,
                                  "DONT_BASE",
                                  f"After '{m.group(1)}', use the base form '{corr}'."))
        return out

    def _day_month_cap(self, text: str) -> List[Dict]:
        out = []
        names = "|".join(sorted(_DAY_MONTH_NAMES, key=len, reverse=True))
        for m in re.finditer(rf"\b({names})\b", text, re.I):
            w = m.group(1)
            if w[0].isupper():
                continue
            repl = w[0].upper() + w[1:]
            out.append(self._cand(w, repl, "capitalization", m, 0.9,
                                  "PROPER_NOUN_CAP",
                                  f"Days and months are capitalized: '{repl}'."))
        return out

    def _irregular_past_ed(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(_WORD, text):
            low = m.group(0).lower()
            if low in _IRREGULAR_PAST_ED:
                corr = _IRREGULAR_PAST_ED[low]
                if m.group(0)[0].isupper():
                    corr = corr[0].upper() + corr[1:]
                out.append(self._cand(m.group(0), corr, "verb_form", m, 0.95,
                                      "IRREGULAR_PAST_ED",
                                      f"The past form is '{corr}', not '{m.group(0)}'."))
        return out

    def _singular_noun_were(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(
                r"\b(a|an|the|this|that|my|your|our|his|her|its|one)\s+(\w+)\s+(were)\b",
                text, re.I):
            noun = m.group(2).lower()
            if noun.endswith("s") or noun in ("information", "news", "staff",
                                              "team", "family"):
                continue
            out.append(self._cand(m.group(3), "was", "subject_verb", _span(m, 3), 0.8,
                                  "SVA",
                                  f"'{m.group(2)}' is singular, so use 'was'."))
        return out

    def _tense_consistency(self, text: str) -> List[Dict]:
        """'Yesterday, the team has attended the meeting.' -> the past-time
        marker clashes with the present-perfect, so drop the auxiliary."""
        out = []
        for marker in _PAST_MARKERS.finditer(text):
            window = text[marker.end():marker.end() + 90]
            cut = min([i for i in (window.find("."), window.find("!"),
                                   window.find("?"), window.find(";"))
                       if i != -1] or [len(window)])
            window = window[:cut]
            for m in re.finditer(
                    r"\b(has|have|had)\s+(also|just|already|ever|never)?\s*"
                    r"([a-z]+(?:ed|n|t))\b", window, re.I):
                verb = m.group(3).lower()
                plain = verb if verb in _PAST_SIMPLE or verb.endswith("ed") else None
                if plain is None or plain == m.group(1).lower() == "had":
                    continue
                start = marker.end() + m.start(1)
                end = marker.end() + m.end(0)
                wrong = text[start:end]
                out.append({
                    "wrong": wrong,
                    "correct": plain,
                    "type": "tense",
                    "start": start,
                    "end": end,
                    "confidence": 0.85,
                    "source": "rule",
                    "rule_id": "TENSE_CONSISTENCY",
                    "message": f"With '{marker.group(0)}', use the simple past '{plain}' instead of '{wrong}'.",
                })
        return out

    def _common_typos(self, text: str) -> List[Dict]:
        out = []
        pat = r"\b(" + "|".join(_COMMON_TYPO_FIXES) + r")\b"
        for m in re.finditer(pat, text, re.I):
            w = m.group(1)
            corr = _COMMON_TYPO_FIXES[w.lower()]
            if w[0].isupper():
                corr = corr[0].upper() + corr[1:]
            out.append(self._cand(w, corr, "spelling", m, 0.88, "COMMON_TYPO",
                                  f"Common misspelling; use '{corr}'."))
        return out

    def _word_confusions(self, text: str) -> List[Dict]:
        """High-precision, context-aware word confusions without a POS tagger.

        'everyday' used adverbially ('I go there everyday') is almost always a
        learner error for 'every day' — but only flag it when it is NOT the
        attributive adjective ('everyday life', 'everyday routine')."""
        out = []
        _NOUN_Y = {"life", "routine", "use", "basis", "chores", "tasks", "work",
                   "wear", "items", "things", "clothes", "people", "activities",
                   "problems", "objects", "events", "occurrence", "occurrences",
                   "experience", "experiences", "decision", "decisions",
                   "activity", "habits", "thing", "object", "task", "problem",
                   "event", "item", "clothing", "chores"}
        for m in re.finditer(r"\beveryday\b(?![-\w])", text, re.I):
            after = text[m.end():m.end() + 30].lstrip()
            nxt = re.match(r"[a-z]+", after, re.I)
            if nxt and nxt.group(0).lower() in _NOUN_Y:
                continue  # "an everyday life" — attributive adjective, correct
            out.append(self._cand(m.group(0), "every day", "word_choice", m, 0.85,
                                  "EVERYDAY_DAY",
                                  "The adverbial phrase for 'each day' is 'every day'."))
        return out

    def _person_noun_be(self, text: str) -> List[Dict]:
        """3rd-person singular 'person' noun + wrong be-form: 'My brother are...'
        -> 'My brother is...'; 'were' -> 'was'."""
        out = []
        for m in re.finditer(rf"\b({_PERSON_NOUNS})\s+(are|were)\b", text, re.I):
            repl = "was" if m.group(2).lower() == "were" else "is"
            out.append(self._cand(m.group(2), repl, "subject_verb", _span(m, 2), 0.82,
                                  "PERSON_NOUN_BE",
                                  f"'{m.group(1)}' is singular, so use '{repl}'."))
        return out

    def _missing_apostrophe(self, text: str) -> List[Dict]:
        out = []
        _APOS = {"dont": "don't", "doesnt": "doesn't", "didnt": "didn't",
                 "cant": "can't", "wont": "won't", "wouldnt": "wouldn't",
                 "couldnt": "couldn't", "shouldnt": "shouldn't", "isnt": "isn't",
                 "arent": "aren't", "wasnt": "wasn't", "werent": "weren't"}
        pat = r"\b(" + "|".join(_APOS) + r")\b"
        for m in re.finditer(pat, text, re.I):
            w = m.group(1).lower()
            prev = ""
            if m.start() > 0:
                prev = text[max(0, m.start() - 14):m.start()].split()[-1]
                prev = prev.lower().strip(".,;:!?()\"'")
            rep = _APOS[w]
            if w == "dont" and prev in ("he", "she", "it"):
                rep = "doesn't"
            if w == "wont" and prev in ("i", "you", "we", "they"):
                rep = "won't"
            out.append(self._cand(m.group(0), rep, "spelling", m, 0.92,
                                  "MISSING_APOSTROPHE",
                                  f"The word '{m.group(0)}' is missing an apostrophe; use '{rep}'."))
        return out

    def _who_were(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(r"\b(a|an|the|one)\s+(\w+)\s+who\s+(were)\b", text, re.I):
            noun = m.group(2).lower()
            # "the children who were" — a plural head keeps 'were'.
            if (noun in _COMMON_PLURALS
                    or (noun.endswith("s")
                        and not noun.endswith("ss") and not noun.endswith("is"))):
                continue
            out.append(self._cand(m.group(3), "was", "subject_verb", _span(m, 3), 0.75,
                                  "WHO_WAS",
                                  f"'{m.group(2)}' is singular, so use 'was'."))
        return out

    def _word_form(self, text: str) -> List[Dict]:
        # "I was very worry" -> "I was very worried"; "so worry", "really worry"
        # A closed set of noun/adjective confusions that are safe and specific.
        out = []
        _NOUN_ADJ = {"worry": "worried", "tire": "tired", "disappoint":
                     "disappointed", "surprise": "surprised", "excite":
                     "excited", "confuse": "confused", "interest":
                     "interested"}
        for noun, adj in _NOUN_ADJ.items():
            for m in re.finditer(
                    rf"\b(?:is|are|am|was|were|be|been|become|feel|feels|felt)\s+"
                    rf"(?:very|so|really|quite|too|extremely|more)\s+\b({noun})\b",
                    text, re.I):
                out.append(self._cand(m.group(1), adj, "word_form", m, 0.8,
                                      "WORD_FORM_EMOTION_ADJ",
                                      f"Use the adjective form '{adj}' after a linking verb."))
        return out

    def _plural_was(self, text: str) -> List[Dict]:
        out = []
        # irregular plurals + "was" -> "were"
        for m in re.finditer(r"\b(childrens?|peoples?|men|women|friends?|they)\s+(was)\b",
                             text, re.I):
            out.append(self._cand(m.group(2), "were", "subject_verb", _span(m, 2), 0.87,
                                  "PLURAL_WAS",
                                  f"'{m.group(1)}' is plural, so use 'were'."))
        # determiner/possessive + plural-noun + "was" -> "were"
        for m in re.finditer(
                r"\b((?:my|our|your|their|these|those)\s+\w+s)\s+(was)\b", text, re.I):
            noun = m.group(1).rsplit(" ", 1)[-1].lower()
            if noun not in _COMMON_PLURALS:
                continue
            out.append(self._cand(m.group(2), "were", "subject_verb", _span(m, 2), 0.83,
                                  "PLURAL_NOUN_WAS",
                                  f"'{m.group(1)}' is plural, so use 'were'."))
        # "the monkeys was" / "the other birds was" — only when the noun is a
        # known plural form; singular words merely ending in '-s' stay untouched.
        for m in re.finditer(
                r"\b(the)\s+(?:\w+\s+)?(\w+s)\s+(was)\b", text, re.I):
            noun = m.group(2).lower()
            if (noun not in _COMMON_PLURALS
                    and not (noun.endswith(("ies", "ves"))
                             and noun not in ("series", "species"))):
                continue
            out.append(self._cand(m.group(3), "were", "subject_verb", _span(m, 3), 0.83,
                                  "PLURAL_NOUN_WAS",
                                  f"'{m.group(2)}' is plural, so use 'were'."))
        # plural quantifier + plural noun + "was": "several students was trying"
        for m in re.finditer(
                r"\b((?:many|several|both|few|various|numerous|all|some|two|three|"
                r"four|five|six|seven|eight|nine|ten|\d+)\s+\w+s)\s+(was)\b", text, re.I):
            noun = m.group(1).rsplit(" ", 1)[-1].lower()
            if (noun not in _COMMON_PLURALS
                    and not (noun.endswith(("ies", "ves"))
                             and noun not in ("series", "species"))):
                continue
            out.append(self._cand(m.group(2), "were", "subject_verb", _span(m, 2), 0.85,
                                  "PLURAL_NOUN_WAS",
                                  f"'{m.group(1)}' is plural, so use 'were'."))
        # "there wasn't many tables" -> "there weren't many tables"
        for m in re.finditer(
                r"\b(wasn'?t|was not)\s+(many|several|both|a few|two|three|four|five|"
                r"six|seven|eight|nine|ten)\b", text, re.I):
            repl = "weren't" if "n't" in m.group(1) else "were not"
            out.append(self._cand(m.group(1), repl, "subject_verb", _span(m, 1), 0.88,
                                  "WASNT_MANY",
                                  f"After a plural quantifier use '{repl}'."))
        # proper name + "don't" -> "doesn't" (learner error)
        for m in re.finditer(r"\b([A-Z][a-z]{2,})\s+(don'?t)\b", text):
            out.append(self._cand(m.group(2), "doesn't", "subject_verb", _span(m, 2), 0.87,
                                  "NAME_DOESNT",
                                  f"'{m.group(1)}' is singular, so use 'doesn't'."))
        return out

    # --------------------------------------------------------- missing words
    def _missing_auxiliary(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(rf"\b(i|he|she|it|we|you|they)\s+({_GERUNDS})\b", text, re.I):
            pron = m.group(1).lower()
            be = _BE_FOR_PRONOUN[pron]
            out.append(self._cand(m.group(0), f"{m.group(1)} {be} {m.group(2)}",
                                  "missing_word", _span(m, 0), 0.6, "MISSING_AUX",
                                  f"Possible missing '{be}' before '{m.group(2)}'."))
        return out

    # -------------------------------------------------------------- pronouns
    def _reflexive_subject(self, text: str) -> List[Dict]:
        out = []
        # A reflexive is never a subject pronoun; flag it only when it sits in
        # subject position (sentence start or right after a coordinating
        # conjunction). Object/emphatic reflexives ("I did it myself") stay.
        for m in re.finditer(
                r"(?:^|[.!?;]\s+|\band\s+|\bor\s+|\bbut\s+|\bso\s+)"
                r"(myself|yourself|himself|herself|ourselves|themselves)\b",
                text, re.I):
            out.append(self._cand(m.group(1), "", "pronoun", _span(m, 1), 0.6,
                                  "REFLEXIVE_SUBJECT",
                                  f"'{m.group(1)}' is usually not a subject pronoun."))
        return out

    # -------------------------------------------------------- capitalization
    def _capitalization(self, text: str) -> List[Dict]:
        out = []
        # standalone lowercase i
        for m in re.finditer(r"(?<![\w'’])i(?![\w'’A-Z])", text):
            out.append(self._cand("i", "I", "capitalization", m, 0.99, "CAP_I",
                                  "The pronoun 'I' is always capitalized."))
        # sentence-start lowercase letter
        for m in re.finditer(r"(^|[.!?]\s+)([a-z])", text):
            letter = m.group(2)
            pos = m.start(2)
            out.append({
                "wrong": letter,
                "correct": letter.upper(),
                "type": "capitalization",
                "start": pos,
                "end": pos + 1,
                "confidence": 0.7,
                "source": "rule",
                "rule_id": "CAP_SENTENCE_START",
                "message": "Capitalize the first letter of a sentence.",
            })
        return out

    # ------------------------------------------------------------ punctuation
    def _punctuation(self, text: str) -> List[Dict]:
        out = []
        for m in re.finditer(r"\s+([,.;:!?])", text):
            out.append({
                "wrong": m.group(0),
                "correct": m.group(1),
                "type": "punctuation",
                "start": m.start(),
                "end": m.end(),
                "confidence": 0.95,
                "source": "rule",
                "rule_id": "SPACE_BEFORE_PUNCT",
                "message": "Remove the space before punctuation.",
            })
        for m in re.finditer(r"([.!?,])\1+", text):
            out.append({
                "wrong": m.group(0),
                "correct": m.group(1),
                "type": "punctuation",
                "start": m.start(),
                "end": m.end(),
                "confidence": 0.85,
                "source": "rule",
                "rule_id": "DOUBLE_PUNCT",
                "message": f"Repeated punctuation '{m.group(0)}'.",
            })
        stripped = text.rstrip()
        if stripped and stripped[-1] not in ".!?\"')":
            out.append({
                "wrong": stripped[-1],
                "correct": stripped[-1] + ".",
                "type": "punctuation",
                "start": len(stripped) - 1,
                "end": len(stripped),
                "confidence": 0.6,
                "source": "rule",
                "rule_id": "MISSING_END_PUNCT",
                "message": "Add ending punctuation.",
            })
        return out

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _cand(wrong: str, correct: str, type_: str, m, confidence: float,
              rule_id: str, message: str) -> Dict:
        if isinstance(m, re.Match):
            start, end = m.span()
        elif hasattr(m, "span"):
            start, end = m.span()
        else:
            start, end = m[0], m[1]
        return {
            "wrong": wrong,
            "correct": correct,
            "type": type_,
            "start": start,
            "end": end,
            "confidence": confidence,
            "source": "rule",
            "rule_id": rule_id,
            "message": message,
        }


def _span(m: re.Match, group: int):
    """Fake a match object whose span() targets a capture group."""
    class _S:
        def __init__(self, s, e):
            self._s, self._e = s, e

        def start(self):
            return self._s

        def end(self):
            return self._e

        def span(self):
            return (self._s, self._e)
    return _S(m.start(group), m.end(group))


def _normalize_type(value: str) -> str:
    v = (value or "").lower()
    mapping = {
        "spelling": "spelling", "grammar": "grammar", "punctuation": "punctuation",
        "word_usage": "word_choice", "verb_form": "verb_form", "tense": "tense",
        "subject_verb_agreement": "subject_verb", "subject_verb": "subject_verb",
        "article": "article", "pronoun": "pronoun", "preposition": "preposition",
        "sentence_structure": "structure", "capitalization": "capitalization",
        "redundancy": "redundancy", "typo": "spelling",
    }
    return mapping.get(v, v or "grammar")


def _dedup(items: List[Dict]) -> List[Dict]:
    best: Dict[Tuple, Dict] = {}
    for it in items:
        key = (it["start"], it["end"], it["correct"].lower())
        cur = best.get(key)
        if cur is None or it["confidence"] > cur["confidence"]:
            best[key] = it
    return list(best.values())


_default_detector: Optional[RuleDetector] = None


def get_rule_detector() -> RuleDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = RuleDetector()
    return _default_detector


def detect_rules(text: str) -> List[Dict]:
    return get_rule_detector().generate(text)
