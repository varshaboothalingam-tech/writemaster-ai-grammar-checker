"""
unified_pipeline.py — Single unified grammar checking pipeline.

Consolidates all duplicate engines (SVA 4x, tense 4x, pronouns 3x, etc.)
into one clean pipeline with proper detection → context → validation → ranking flow.

Pipeline:
  TEXT → PREPROCESSOR → TOKENIZER → FAST DETECTION → NLP DETECTION →
  CONTEXT ANALYSIS → CONFIDENCE SCORING → FP FILTER → CORRECTION →
  VALIDATION → DEDUPLICATION → RANKING → OUTPUT
"""

import re
import os
import json
import math
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum

import spacy
from nltk.corpus import words as nltk_words, wordnet
from text_preprocessor import TextPreprocessor
from spellchecker import SpellChecker
from high_confidence_rules import HighConfidenceDetector

# ── Single shared spaCy model ───────────────────────────────────────────
_NLP = None
def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════

class ErrorCategory(Enum):
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    WORD_USAGE = "word_usage"
    VERB_FORM = "verb_form"
    SUBJECT_VERB_AGREEMENT = "agreement"
    TENSE = "tense"
    ARTICLE = "articles"
    ARTICLES = "articles"
    PRONOUN = "pronouns"
    PRONOUNS = "pronouns"
    PREPOSITION = "prepositions"
    PREPOSITIONS = "prepositions"
    SENTENCE_STRUCTURE = "sentence_structure"
    CAPITALIZATION = "capitalization"
    TYPO = "spelling"
    CLARITY = "clarity"
    STYLE = "style"
    TONE = "tone"
    READABILITY = "readability"
    ADVERB = "adverb"
    CONJUNCTION = "conjunction"
    MODAL = "modal"
    WORD_ORDER = "grammar"


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    STYLE = "style"


@dataclass
class ErrorCandidate:
    """Raw candidate error — before validation."""
    start: int
    end: int
    original: str
    replacement: str
    category: ErrorCategory
    rule_id: str
    message: str
    detector: str
    raw_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalError:
    """Validated, ranked error — shown to user."""
    start: int
    end: int
    original: str
    replacement: str
    category: str
    subcategory: str
    severity: str
    confidence: float
    explanation: str
    rule_id: str
    detector: str
    validated: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "original": self.original,
            "replacement": self.replacement,
            "category": self.category,
            "subcategory": self.subcategory,
            "severity": self.severity,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "rule_id": self.rule_id,
            "detector": self.detector,
            "validated": self.validated,
        }


# ══════════════════════════════════════════════════════════════════════════
#  LINGUISTIC CONSTANTS (single source of truth)
# ══════════════════════════════════════════════════════════════════════════

IRREGULAR_VERBS = {
    "be": {"past_sg": "was", "past_pl": "were", "pp": "been", "present_3sg": "is", "present_pl": "are"},
    "have": {"past": "had", "pp": "had", "present_3sg": "has"},
    "do": {"past": "did", "pp": "done", "present_3sg": "does"},
    "go": {"past": "went", "pp": "gone", "present_3sg": "goes"},
    "say": {"past": "said", "pp": "said", "present_3sg": "says"},
    "get": {"past": "got", "pp": "gotten", "present_3sg": "gets"},
    "make": {"past": "made", "pp": "made", "present_3sg": "makes"},
    "know": {"past": "knew", "pp": "known", "present_3sg": "knows"},
    "think": {"past": "thought", "pp": "thought", "present_3sg": "thinks"},
    "take": {"past": "took", "pp": "taken", "present_3sg": "takes"},
    "see": {"past": "saw", "pp": "seen", "present_3sg": "sees"},
    "come": {"past": "came", "pp": "come", "present_3sg": "comes"},
    "give": {"past": "gave", "pp": "given", "present_3sg": "gives"},
    "find": {"past": "found", "pp": "found", "present_3sg": "finds"},
    "tell": {"past": "told", "pp": "told", "present_3sg": "tells"},
    "become": {"past": "became", "pp": "become", "present_3sg": "becomes"},
    "leave": {"past": "left", "pp": "left", "present_3sg": "leaves"},
    "feel": {"past": "felt", "pp": "felt", "present_3sg": "feels"},
    "put": {"past": "put", "pp": "put", "present_3sg": "puts"},
    "bring": {"past": "brought", "pp": "brought", "present_3sg": "brings"},
    "begin": {"past": "began", "pp": "begun", "present_3sg": "begins"},
    "keep": {"past": "kept", "pp": "kept", "present_3sg": "keeps"},
    "hold": {"past": "held", "pp": "held", "present_3sg": "holds"},
    "write": {"past": "wrote", "pp": "written", "present_3sg": "writes"},
    "stand": {"past": "stood", "pp": "stood", "present_3sg": "stands"},
    "hear": {"past": "heard", "pp": "heard", "present_3sg": "hears"},
    "let": {"past": "let", "pp": "let", "present_3sg": "lets"},
    "mean": {"past": "meant", "pp": "meant", "present_3sg": "means"},
    "set": {"past": "set", "pp": "set", "present_3sg": "sets"},
    "meet": {"past": "met", "pp": "met", "present_3sg": "meets"},
    "run": {"past": "ran", "pp": "run", "present_3sg": "runs"},
    "pay": {"past": "paid", "pp": "paid", "present_3sg": "pays"},
    "sit": {"past": "sat", "pp": "sat", "present_3sg": "sits"},
    "speak": {"past": "spoke", "pp": "spoken", "present_3sg": "speaks"},
    "lie": {"past": "lay", "pp": "lain", "present_3sg": "lies"},
    "lead": {"past": "led", "pp": "led", "present_3sg": "leads"},
    "understand": {"past": "understood", "pp": "understood", "present_3sg": "understands"},
    "watch": {"past": "watched", "pp": "watched", "present_3sg": "watches"},
    "follow": {"past": "followed", "pp": "followed", "present_3sg": "follows"},
    "stop": {"past": "stopped", "pp": "stopped", "present_3sg": "stops"},
    "create": {"past": "created", "pp": "created", "present_3sg": "creates"},
    "speak": {"past": "spoke", "pp": "spoken", "present_3sg": "speaks"},
    "read": {"past": "read", "pp": "read", "present_3sg": "reads"},
    "spend": {"past": "spent", "pp": "spent", "present_3sg": "spends"},
    "grow": {"past": "grew", "pp": "grown", "present_3sg": "grows"},
    "open": {"past": "opened", "pp": "opened", "present_3sg": "opens"},
    "walk": {"past": "walked", "pp": "walked", "present_3sg": "walks"},
    "win": {"past": "won", "pp": "won", "present_3sg": "wins"},
    "teach": {"past": "taught", "pp": "taught", "present_3sg": "teaches"},
    "buy": {"past": "bought", "pp": "bought", "present_3sg": "buys"},
    "drive": {"past": "drove", "pp": "driven", "present_3sg": "drives"},
    "eat": {"past": "ate", "pp": "eaten", "present_3sg": "eats"},
    "fall": {"past": "fell", "pp": "fallen", "present_3sg": "falls"},
    "fly": {"past": "flew", "pp": "flown", "present_3sg": "flies"},
    "break": {"past": "broke", "pp": "broken", "present_3sg": "breaks"},
    "choose": {"past": "chose", "pp": "chosen", "present_3sg": "chooses"},
    "draw": {"past": "drew", "pp": "drawn", "present_3sg": "draws"},
    "drink": {"past": "drank", "pp": "drunk", "present_3sg": "drinks"},
    "ride": {"past": "rode", "pp": "ridden", "present_3sg": "rides"},
    "ring": {"past": "rang", "pp": "rung", "present_3sg": "rings"},
    "sing": {"past": "sang", "pp": "sung", "present_3sg": "sings"},
    "swim": {"past": "swam", "pp": "swum", "present_3sg": "swims"},
    "throw": {"past": "threw", "pp": "thrown", "present_3sg": "throws"},
    "wear": {"past": "wore", "pp": "worn", "present_3sg": "wears"},
    "forget": {"past": "forgot", "pp": "forgotten", "present_3sg": "forgets"},
    "forgive": {"past": "forgave", "pp": "forgiven", "present_3sg": "forgives"},
    "hide": {"past": "hid", "pp": "hidden", "present_3sg": "hides"},
    "shake": {"past": "shook", "pp": "shaken", "present_3sg": "shakes"},
    "rise": {"past": "rose", "pp": "risen", "present_3sg": "rises"},
    "bite": {"past": "bit", "pp": "bitten", "present_3sg": "bites"},
    "blow": {"past": "blew", "pp": "blown", "present_3sg": "blows"},
    "catch": {"past": "caught", "pp": "caught", "present_3sg": "catches"},
    "deal": {"past": "dealt", "pp": "dealt", "present_3sg": "deals"},
    "dig": {"past": "dug", "pp": "dug", "present_3sg": "digs"},
    "feed": {"past": "fed", "pp": "fed", "present_3sg": "feeds"},
    "freeze": {"past": "froze", "pp": "frozen", "present_3sg": "freezes"},
    "hang": {"past": "hung", "pp": "hung", "present_3sg": "hangs"},
    "lay": {"past": "laid", "pp": "laid", "present_3sg": "lays"},
    "lose": {"past": "lost", "pp": "lost", "present_3sg": "loses"},
    "sell": {"past": "sold", "pp": "sold", "present_3sg": "sells"},
    "send": {"past": "sent", "pp": "sent", "present_3sg": "sends"},
    "shoot": {"past": "shot", "pp": "shot", "present_3sg": "shoots"},
    "shut": {"past": "shut", "pp": "shut", "present_3sg": "shuts"},
    "steal": {"past": "stole", "pp": "stolen", "present_3sg": "steals"},
    "strike": {"past": "struck", "pp": "struck", "present_3sg": "strikes"},
    "tear": {"past": "tore", "pp": "torn", "present_3sg": "tears"},
    "wake": {"past": "woke", "pp": "woken", "present_3sg": "wakes"},
}

# Verbs whose past tense doubles the final consonant (stop -> stopped).
# Short CVC verbs plus common multi-syllable verbs stressed on the final
# syllable (submit -> submitted, occur -> occurred).
_DOUBLE_FINAL_VERBS = {
    "stop", "plan", "drop", "grab", "shop", "chat", "hop", "nod", "beg",
    "rub", "pad", "dot", "spot", "clap", "slip", "snap", "trim", "hug",
    "jog", "drum", "map", "log", "pat", "tag", "fan", "ban", "sin", "rot",
    "dip", "fit", "pit", "pin", "tip", "jam", "stem", "scrap", "wrap",
    "flip", "grip", "skim", "slim",
    "commit", "submit", "admit", "permit", "omit", "regret",
    "occur", "refer", "prefer", "transfer", "confer", "defer", "deter",
    "recur", "patrol", "rebel", "repel", "excel", "expel", "compel",
    "propel", "equip",
}


def _regular_past_tense(base: str) -> str:
    """Simple past form for regular verbs (''arrive'' -> ''arrived'')."""
    if base.endswith("e"):
        return base + "d"
    if base in _DOUBLE_FINAL_VERBS:
        return base + base[-1] + "ed"
    if len(base) > 1 and base.endswith("y") and base[-2] not in "aeiou":
        return base[:-1] + "ied"
    return base + "ed"


BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
HAVE_FORMS = {"have", "has", "had", "having"}
DO_FORMS = {"do", "does", "did"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must", "dare", "need", "ought", "'ll", "'d"}
SINGULAR_PRONOUNS = {"i", "he", "she", "it", "this", "that", "everyone", "everybody",
                      "someone", "somebody", "nobody", "anyone", "anybody", "each",
                      "every", "either", "neither", "one"}
PLURAL_PRONOUNS = {"we", "they", "these", "those"}
UNCOUNTABLE_NOUNS = {"information", "knowledge", "evidence", "advice", "furniture",
                      "luggage", "equipment", "progress", "chaos", "music", "news",
                      "math", "physics", "economics", "politics", "ethics", "traffic",
                      "weather", "rice", "sugar", "water", "milk", "bread", "money",
                      "research", "homework", "education", "experience", "health",
                      "love", "happiness", "software", "hardware", "furniture",
                      "luggage", "baggage", "equipment", "machinery", "pricing",
                      "feedback", "software", "content", "documentation", "news",
                      "mathematics", "physics", "economics", "politics", "ethics",
                      "linguistics", "athletics", "gymnastics", "measles", "mumps",
                      "rabies", "shingles", "series", "species",
                      "noise", "hope", "competition", "corruption", "patience",
                      "time", "effort", "work", "music", "art", "freedom",
                      "courage", "interest", "authority", "responsibility",
                      "opportunity", "influence", "approval", "pressure",
                      "anxiety", "distraction", "clutter", "talent",
                      "hair", "space", "room", "breadth", "strength",
                      "courage", "ambition", "passion", "enthusiasm",
                      "encouragement", "motivation", "vocabulary",
                      "practice", "homework", "progress", "information",
                      "knowledge", "evidence", "advice", "furniture",
                      "luggage", "baggage", "equipment", "machinery",
                      "data", "media", "criteria", "phenomena",
                      "food", "fruit", "meat", "wood", "paper",
                      "cloth", "gold", "silver", "steel", "iron", "glass",
                      "plastic", "cotton", "wool", "silk", "rubber",
                      "electricity", "energy", "power", "light", "heat",
                      "sunlight", "moonlight", "starlight", "darkness",
                      "silence", "noise", "sound", "music", "thunder",
                      "rain", "snow", "ice", "fog", "wind", "air",
                      "dust", "smoke", "fire", "flame", "ash",
                      "blood", "sweat", "tears", "saliva", "spit",
                      "dirt", "mud", "sand", "gravel", "soil", "earth",
                      "rock", "stone", "clay", "chalk", "coal",
                      "flesh", "bone", "skin", "hair", "nail",
                      "garbage", "trash", "rubbish", "waste", "debris",
                      "junk", "scrap", "residue", "sediment", "sludge",
                      "sewage", "manure", "compost", "fertilizer",
                      "cash", "coin", "currency", "capital", "wealth",
                      "poverty", "richness", "richness", "debt", "credit",
                      "interest", "profit", "loss", "revenue", "income",
                      "salary", "wage", "pay", "compensation", "bonus",
                      "tax", "duty", "tariff", "fee", "charge", "cost",
                      "price", "value", "worth", "merit", "quality",
                      "quantity", "amount", "number", "total", "sum",
                      "average", "rate", "ratio", "percentage", "fraction",
                      "proportion", "share", "portion", "slice", "piece",
                      "bit", "drop", "grain", "speck", "morsel", "crumb",
                      "scrap", "shred", "sliver", "splinter", "chip",
                      "flake", "particle", "atom", "molecule", "cell",
                      "tissue", "organ", "body", "system", "structure",
                      "framework", "skeleton", "foundation", "base",
                      "ground", "floor", "ceiling", "wall", "roof",
                      "door", "window", "gate", "fence", "barrier",
                      "border", "boundary", "edge", "rim", "brink",
                      "threshold", "margin", "verge", "precipice", "cliff",
                      "slope", "hill", "mountain", "valley", "canyon",
                      "gorge", "ravine", "gully", "ditch", "trench",
                      "channel", "stream", "creek", "brook", "river",
                      "lake", "pond", "pool", "puddle", "ocean", "sea",
                      "bay", "gulf", "cove", "harbor", "port", "dock",
                      "pier", "wharf", "jetty", "breakwater", "dam",
                      "dike", "levee", "embankment", "causeway", "bridge",
                      "tunnel", "passage", "corridor", "hallway", "aisle"}
COLLECTIVE_NOUNS = {"team", "group", "family", "committee", "class", "staff",
                     "company", "organization", "government", "police", "jury",
                     "audience", "crowd", "community", "public", "army", "club",
                     "board", "council", "panel", "faculty", "band", "choir",
                     "orchestra", "network", "system", "series", "couple"}

# Common misspellings (extended)
MISSPELLINGS = {
    "recieve": "receive", "freind": "friend", "seperate": "separate",
    "occured": "occurred", "definately": "definitely", "accomodate": "accommodate",
    "untill": "until", "wierd": "weird", "truely": "truly", "enviroment": "environment",
    "goverment": "government", "neccessary": "necessary", "priviledge": "privilege",
    "relevent": "relevant", "successfull": "successful", "grammer": "grammar",
    "independant": "independent", "libary": "library", "maintainance": "maintenance",
    "posession": "possession", "publically": "publicly", "responsable": "responsible",
    "similiar": "similar", "suprise": "surprise", "wheter": "whether", "wich": "which",
    "yeild": "yield",     "accross": "across", "arguement": "argument", "calender": "calendar",
    "catagory": "category", "comming": "coming", "commitee": "committee",
    "concious": "conscious", "decison": "decision", "dissapear": "disappear",
    "embarass": "embarrass", "existance": "existence", "fourty": "forty",
    "garentee": "guarantee", "happend": "happened", "incidently": "incidentally",
    "independance": "independence", "knowlege": "knowledge", "liason": "liaison",
    "lenght": "length", "liberry": "library", "maintenence": "maintenance",
    "mispell": "misspell", "noticable": "noticeable", "occassion": "occasion",
    "occurance": "occurrence", "paralell": "parallel", "peice": "piece",
    "posess": "possess", "preceed": "precede", "privelege": "privilege",
    "probaly": "probably", "pronounciation": "pronunciation", "questionaire": "questionnaire",
    "recieved": "received", "recomend": "recommend", "refrence": "reference",
    "rellevant": "relevant", "restaraunt": "restaurant", "seize": "seize",
    "sentance": "sentence", "succesful": "successful", "supercede": "supersede",
    "tommorow": "tomorrow", "tounge": "tongue", "truely": "truly",
    "tyranny": "tyranny", "unforseen": "unforeseen", "unfortunatly": "unfortunately",
    "unneccessary": "unnecessary", "whereever": "wherever", "wich": "which",
    "writting": "writing", "allot": "a lot", "alot": "a lot",
    "beautifull": "beautiful", "studing": "studying",     "clothe": "clothes", "studing": "studying", "infomation": "information",
    "informations": "information", "milage": "mileage", "miniture": "miniature",
    "mischievious": "mischievous", "neccessity": "necessity",
    "occassion": "occasion", "occurrance": "occurrence", "persistance": "persistence",
    "pharoah": "pharaoh", "politican": "politician",
    "preceeding": "preceding", "realy": "really",
    "reciept": "receipt", "refered": "referred",
    "religous": "religious", "ryhme": "rhyme",
    "seige": "siege", "similiar": "similar",
    "sincerly": "sincerely", "speach": "speech", "strenght": "strength",
    "supposably": "supposedly", "supposebly": "supposedly", "tatoo": "tattoo",
    "temperture": "temperature", "therefor": "therefore", "thier": "their",
    "tommorow": "tomorrow", "tounge": "tongue",
    "unneccessary": "unnecessary", "wierd": "weird", "whereever": "wherever",
    "writting": "writing", "calender": "calendar", "definately": "definitely",
    "excersize": "exercise", "existance": "existence", "happend": "happened",
    "harrassment": "harassment", "ignorence": "ignorance", "innoculate": "inoculate",
    "inteligence": "intelligence", "jewelery": "jewelry", "judgement": "judgment",
    "loose": "lose", "milage": "mileage", "miniture": "miniature",
    "mischievious": "mischievous", "neccessity": "necessity", "occassion": "occasion",
    "occurrance": "occurrence", "persistance": "persistence", "pharoah": "pharaoh",
    "pigeon": "pigeon", "politican": "politician", "posession": "possession",
    "preceeding": "preceding", "publically": "publicly", "realy": "really",
    "reciept": "receipt", "refered": "referred", "relevent": "relevant",
    "religous": "religious", "restaraunt": "restaurant", "ryhme": "rhyme",
    "seige": "siege", "sensei": "sensei", "similiar": "similar",
    "sincerly": "sincerely", "speach": "speech", "strenght": "strength",
    "supposably": "supposedly", "supposebly": "supposedly", "tatoo": "tattoo",
    "temperture": "temperature", "therefor": "therefore", "thier": "their",
    "tommorow": "tomorrow", "tounge": "tongue", "truely": "truly",
    "tyrany": "tyranny", "unforseen": "unforeseen", "unfortunatly": "unfortunately",
    "unneccessary": "unnecessary", "wierd": "weird", "whereever": "wherever",
    "arguement": "argument", "catagory": "category", "decison": "decision",
    "excersize": "exercise", "garentee": "guarantee", "harrassment": "harassment",
    "inteligence": "intelligence", "jewelery": "jewelry", "knowlege": "knowledge",
    "lenght": "length", "mispell": "misspell", "noticable": "noticeable",
    "occassion": "occasion", "occurrance": "occurrence", "paralell": "parallel",
    "peice": "piece", "posess": "possess", "preceed": "precede",
    "privelege": "privilege", "probaly": "probably", "reciept": "receipt",
    "refered": "referred", "restaraunt": "restaurant", "sentance": "sentence",
    "succesful": "successful", "supercede": "supersede", "tounge": "tongue",
    "unneccessary": "unnecessary", "writting": "writing", "alot": "a lot",
    "could of": "could have", "would of": "would have", "should of": "should have",
    "might of": "might have", "must of": "must have", "may of": "may have",
    "use to": "used to", "suppose to": "supposed to", "accidently": "accidentally",
    "irregardless": "regardless", "conversate": "converse", "stopping mall": "shopping mall",
    "clothe": "clothes",
    "seperated": "separated", "seperate": "separate", "seperating": "separating",
    "necessery": "necessary", "neccessary": "necessary", "neccesary": "necessary",
    "occurence": "occurrence", "occurrance": "occurrence",
    "privledge": "privilege", "privelege": "privilege",
    "acheive": "achieve", "acheived": "achieved",
    "beleive": "believe", "beleived": "believed",
    "dependant": "dependent",
    "foriegn": "foreign",
    "arguement": "argument", "catagory": "category",
    "commited": "committed", "occuring": "occurring",
    "prefered": "preferred", "begining": "beginning",
    "runing": "running", "planing": "planning",
    "goverment": "government", "enviroment": "environment",
    "independant": "independent", "consistant": "consistent",
    "existance": "existence", "permanant": "permanent",
    "relevent": "relevant", "adequet": "adequate",
    "apparant": "apparent", "equivelant": "equivalent",
    "maintainence": "maintenance", "persistant": "persistent",
    "reffered": "referred", "commitee": "committee",
    "disapear": "disappear", "embarass": "embarrass",
    "harrased": "harassed", "untill": "until",
    "wich": "which", "wheter": "whether",
    "writting": "writing", "commiting": "committing",
    "begining": "beginning", "runing": "running",
    "occassionally": "occasionally",
    "accomodation": "accommodation", "anomoly": "anomaly",
    "calender": "calendar", "concensus": "consensus",
    "definately": "definitely", "dilema": "dilemma",
    "flourescent": "fluorescent", "grammer": "grammar",
    "hieght": "height", "humourous": "humorous",
    "innoculate": "inoculate", "jeopardise": "jeopardize",
    "judgement": "judgment",
    "liason": "liaison", "manoeuvre": "maneuver",
    "millenium": "millennium", "miniscule": "minuscule",
    "mischievious": "mischievous", "mischevious": "mischievous",
    "neccessary": "necessary", "noticable": "noticeable",
    "occurence": "occurrence", "parliment": "parliament",
    "perseverence": "perseverance", "posess": "possess",
    "pronounciation": "pronunciation", "realy": "really",
    "referance": "reference", "restaraunt": "restaurant",
    "ryhme": "rhyme", "seige": "siege",
    "sentance": "sentence", "succesful": "successful",
    "supercede": "supersede", "supposibly": "supposedly",
    "temperture": "temperature", "therefor": "therefore",
    "thier": "their", "tommorow": "tomorrow",
    "tounge": "tongue", "truely": "truly",
    "tyranny": "tyranny", "unforseen": "unforeseen",
    "unfortunatly": "unfortunately", "unneccessary": "unnecessary",
    "whereever": "wherever", "wierd": "weird",
    "recieved": "received", "reciept": "receipt",
    "recomend": "recommend", "refered": "referred",
    "religous": "religious", "similiar": "similar",
    "sincerly": "sincerely", "speach": "speech",
    "strenght": "strength", "tatoo": "tattoo",
    "politican": "politician", "publically": "publicly",
    "excersize": "exercise", "garentee": "guarantee",
    "harrassment": "harassment", "ignorence": "ignorance",
    "inteligence": "intelligence", "jewelery": "jewelry",
    "knowlege": "knowledge", "lenght": "length",
    "mispell": "misspell", "peice": "piece",
    "posession": "possession", "preceed": "precede",
    "privelege": "privilege", "probaly": "probably",
    "sentance": "sentence", "succesful": "successful",
    "temperture": "temperature", "thier": "their",
    "tounge": "tongue", "unneccessary": "unnecessary",
    "writting": "writing",
    "harrass": "harass", "harrassment": "harassment",
    "immedately": "immediately", "tomatos": "tomatoes",
    "visious": "vicious", "adress": "address",
    "cemetary": "cemetery", "changable": "changeable",
    "collegue": "colleague", "comitment": "commitment",
    "comparision": "comparison", "completly": "completely",
    "independance": "independence", "liason": "liaison",
    "millenium": "millennium", "miniscule": "minuscule",
    "neccessary": "necessary", "occurance": "occurrence",
    "realy": "really", "seperate": "separate",
    "truely": "truly", "untill": "until",
}

# Word usage confusables
CONFUSABLES = {
    "their": {"there", "they're"}, "there": {"their", "they're"}, "they're": {"their", "there"},
    "your": {"you're"}, "you're": {"your"},
    "its": {"it's"}, "it's": {"its"},
    "affect": {"effect"}, "effect": {"affect"},
    "accept": {"except"}, "except": {"accept"},
    "lose": {"loose"}, "loose": {"lose"},
    "then": {"than"}, "than": {"then"},
    "which": {"witch"}, "witch": {"which"},
    "whose": {"who's"}, "who's": {"whose"},
    "weather": {"whether"}, "whether": {"weather"},
    "advice": {"advise"}, "advise": {"advice"},
    "principle": {"principal"}, "principal": {"principle"},
    "compliment": {"complement"}, "complement": {"compliment"},
    "desert": {"dessert"}, "dessert": {"desert"},
    "stationary": {"stationery"}, "stationery": {"stationary"},
}

# Object pronouns that should be subject pronouns in compound subjects
OBJECT_TO_SUBJECT = {"me": "I", "him": "he", "her": "she", "them": "they", "us": "we"}

# Weak/style words
WEAK_WORDS = {"very", "really", "quite", "rather", "somewhat", "basically",
              "actually", "practically", "virtually", "literally", "honestly",
              "totally", "completely", "absolutely", "definitely", "certainly",
              "probably", "possibly", "just", "simply", "easily", "clearly",
              "obviously", "apparently", "supposedly", "hopefully", "stuff",
              "things", "got", "getting", "nice", "good", "bad", "great"}

FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "actually",
                "literally", "honestly", "frankly", "sort of", "kind of"}


# ══════════════════════════════════════════════════════════════════════════
#  TIER 1: FAST DETECTION (spelling, typos, repeated words)
# ══════════════════════════════════════════════════════════════════════════

class FastDetector:
    """Very fast first-pass detection for high-confidence errors."""

    def __init__(self):
        self._valid_words = self._load_valid_words()

    def _load_valid_words(self) -> Set[str]:
        words = set()
        try:
            words = {w.lower() for w in nltk_words.words()}
        except Exception:
            pass
        # Add inflections
        base = list(words)
        for w in base:
            if len(w) < 2:
                continue
            words.add(w + "s")
            words.add(w + "ed")
            words.add(w + "ing")
            words.add(w + "er")
            words.add(w + "est")
            if w.endswith("e"):
                words.add(w[:-1] + "ed")
                words.add(w[:-1] + "ing")
            if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
                words.add(w[:-1] + "ies")
        # Add common words
        common = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "being", "have", "has", "had", "do", "does", "did", "will",
                  "would", "could", "should", "may", "might", "must", "shall",
                  "can", "need", "dare", "ought", "used", "to", "of", "in",
                  "for", "on", "with", "at", "by", "from", "up", "about",
                  "into", "through", "during", "before", "after", "above",
                  "below", "between", "under", "again", "further", "then",
                  "once", "here", "there", "when", "where", "why", "how",
                  "all", "both", "each", "few", "more", "most", "other",
                  "some", "such", "no", "nor", "not", "only", "own", "same",
                  "so", "than", "too", "very", "s", "t", "don", "now", "d",
                  "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn",
                  "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
                  "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
                  "weren", "won", "wouldn", "gonna", "wanna", "gotta",
                  "kinda", "sorta", "dunno", "gotta", "im", "ive", "dont",
                  "doesnt", "didnt", "cant", "wont", "isnt", "arent", "wasnt",
                  "werent", "hasnt", "havent", "hadnt", "shouldnt", "wouldnt",
                  "couldnt", "thats", "hes", "shes", "its", "thats", "whos",
                  "whats", "wheres", "hows", "whys", "dont", "doesnt", "didnt",
                  "cant", "wont", "isnt", "arent", "wasnt", "werent", "hasnt",
                  "havent", "hadnt", "shouldnt", "wouldnt", "couldnt", "mustnt",
                  "neednt", "shant", "lets", "thats", "whos", "hows"}
        words.update(common)
        # Add contractions
        contractions = {"can't", "won't", "don't", "doesn't", "didn't", "isn't",
                        "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
                        "shouldn't", "wouldn't", "couldn't", "mustn't", "needn't",
                        "that's", "who's", "what's", "where's", "how's", "there's",
                        "here's", "let's", "he's", "she's", "it's", "we're",
                        "they're", "you're", "i'm", "we're", "they're", "you're",
                        "i've", "we've", "they've", "you've", "i'll", "we'll",
                        "they'll", "you'll", "he'll", "she'll", "it'll", "i'd",
                        "we'd", "they'd", "you'd", "he'd", "she'd", "it'd"}
        words.update(c.replace("'", "") for c in contractions)
        words.update(contractions)
        return words

    def detect(self, text: str) -> List[ErrorCandidate]:
        candidates = []

        # 1. Repeated words ("the the", "is is")
        for m in re.finditer(r'\b(\w+)\s+\1\b', text, re.IGNORECASE):
            word = m.group(1)
            if word.lower() not in {"that", "had", "is", "are", "was", "were"}:
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0), replacement=word,
                    category=ErrorCategory.GRAMMAR,
                    rule_id="REPEATED_WORD",
                    message=f'Word "{word}" is repeated.',
                    detector="fast_detector",
                    raw_confidence=0.98,
                ))

        # 2. Spelling (misspellings dict — check BEFORE valid words)
        for m in re.finditer(r'\b([a-zA-Z]+)\b', text):
            word = m.group(1)
            lower = word.lower()
            if len(lower) < 3:
                continue
            if lower in MISSPELLINGS:
                prev_a = re.search(r'\ba\s+$', text[:m.start()])
                if prev_a and lower[0] in "aeiou":
                    # Misspelled vowel-word after "a": own the whole span so the
                    # article rule cannot also fire; keep the user's article
                    # ("a acheive" → "a achieve").
                    candidates.append(ErrorCandidate(
                        start=prev_a.start(), end=m.end(),
                        original=prev_a.group(0) + word,
                        replacement=f"a {MISSPELLINGS[lower]}",
                        category=ErrorCategory.SPELLING,
                        rule_id="SPELL_MISSPELLING",
                        message=f'"{word}" may be misspelled. Did you mean "{MISSPELLINGS[lower]}"?',
                        detector="fast_detector",
                        raw_confidence=0.95,
                    ))
                else:
                    candidates.append(ErrorCandidate(
                        start=m.start(), end=m.end(),
                        original=word, replacement=MISSPELLINGS[lower],
                        category=ErrorCategory.SPELLING,
                        rule_id="SPELL_MISSPELLING",
                        message=f'"{word}" may be misspelled. Did you mean "{MISSPELLINGS[lower]}"?',
                        detector="fast_detector",
                        raw_confidence=0.95,
                    ))
            elif lower not in self._valid_words:
                pass  # Unknown word but not in misspellings dict — skip

        # 3. Contextual "there" vs "their" — "there car" → "their car"
        for m in re.finditer(r'\bthere\b', text, re.IGNORECASE):
            word = m.group(0)
            idx = m.start()
            after = text[idx + 5:idx + 25].strip()
            next_word = after.split()[0] if after.split() else ""
            skip_words = BE_FORMS | HAVE_FORMS | DO_FORMS | MODALS | {
                "stands", "sits", "lies", "lives", "comes", "goes", "runs",
                "walks", "talks", "works", "plays", "was", "were", "will",
                "would", "could", "should", "may", "might", "must", "shall",
                "can", "is", "are", "be", "been", "have", "has", "had",
                "seem", "seems", "seemed", "appear", "appears", "appeared",
                "remain", "remains", "remained", "exist", "exists", "existed",
                "stand", "stood", "lie", "exist", "cannot", "used", "used",
                "might", "may", "must", "shall", "need", "ought",
                "shortly", "also", "just", "only", "always", "never", "often",
                "already", "still", "even", "quite", "very", "really",
                "much", "more", "most", "less", "least", "many", "few",
                "been", "being", "done", "doing", "going", "coming",
                "something", "nothing", "everything", "anything",
                "someplace", "nowhere", "everywhere", "anywhere",
                "somebody", "nobody", "everybody", "anybody",
            }
            if next_word and next_word.lower().rstrip('.,;:!?') not in skip_words and len(next_word) > 1:
                if not next_word[0].isupper():
                    candidates.append(ErrorCandidate(
                        start=idx, end=idx + 5,
                        original=word, replacement="their",
                        category=ErrorCategory.WORD_USAGE,
                        rule_id="THERE_THEIR",
                        message=f'Use "their" (possessive) instead of "there".',
                        detector="fast_detector",
                        raw_confidence=0.80,
                    ))

        # 4. "to" → "too" before adjectives/adverbs: "to bad", "to much", "to many"
        too_adj_pattern = re.compile(r'\bto\s+(bad|big|small|much|many|old|new|hard|easy|fast|slow|hot|cold|warm|cool|long|short|tall|wide|deep|thick|thin|fat|ugly|pretty|nice|good|great|poor|rich|early|late|quick|small)\b', re.IGNORECASE)
        for m in too_adj_pattern.finditer(text):
            word = m.group(0)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=word, replacement="too " + m.group(1),
                category=ErrorCategory.WORD_USAGE,
                rule_id="TO_TOO",
                message=f'Use "too" (meaning excessive) instead of "to" before "{m.group(1)}".',
                detector="fast_detector",
                        raw_confidence=0.88,
                    ))

        # 5. Context-aware confusables
        # "Their going" → "They're going" (possessive before verb = contraction needed)
        for m in re.finditer(r'\btheir\b\s+(\w+)', text, re.IGNORECASE):
            next_word = m.group(1).lower()
            # "their" before a common verb/auxiliary = likely "they're"
            if next_word in ("going", "coming", "running", "walking", "playing",
                             "doing", "making", "being", "getting", "having",
                             "not", "really", "always", "never", "also",
                             "so", "too", "very", "just", "now", "here",
                             "there", "been", "done", "ready", "finished",
                             "leaving", "coming", "the", "a", "an"):
                # Before determiners/articles — possessive, skip
                if next_word in ("the", "a", "an"):
                    pass
                else:
                    # "their" before verb/auxiliary/adverb = likely "they're"
                    candidates.append(ErrorCandidate(
                        start=m.start(), end=m.end() + 1 + len(next_word),
                        original=m.group(0), replacement="they're " + next_word,
                        category=ErrorCategory.WORD_USAGE,
                        rule_id="THEIR_THEYRE",
                        message=f'Use "they\'re" (they are) instead of "their".',
                        detector="fast_detector",
                        raw_confidence=0.82,
                    ))

        # "Your welcome" → "You're welcome"
        for m in re.finditer(r'\byour\b\s+(welcome|the\s+one|right|wrong|absolutely|being|going|not|always|never|just|now|here|there)\b', text, re.IGNORECASE):
            match_text = m.group(0)
            replacement = "you're " + m.group(1)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=match_text, replacement=replacement,
                category=ErrorCategory.WORD_USAGE,
                rule_id="YOUR_YOURE",
                message=f'Use "you\'re" (you are) instead of "your".',
                detector="fast_detector",
                raw_confidence=0.90,
            ))

        # "Its important" → "It's important" (its + adjective = it's)
        for m in re.finditer(r'\bits\b\s+(important|not|a|an|the|really|very|quite|too|so|been|going|not|been|my|your|his|her|our|their|one|true|false|clear|obvious|essential|necessary|possible|impossible)\b', text, re.IGNORECASE):
            match_text = m.group(0)
            replacement = "it's " + m.group(1)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=match_text, replacement=replacement,
                category=ErrorCategory.WORD_USAGE,
                rule_id="ITS_ITS",
                message=f'Use "it\'s" (it is) instead of "its" before "{m.group(1)}".',
                detector="fast_detector",
                raw_confidence=0.88,
            ))

        # "Who's book" → "Whose book" (who's + noun = whose)
        for m in re.finditer(r'\bwho\'?s\b\s+(\w+)', text, re.IGNORECASE):
            next_word = m.group(1).lower()
            # If next word is a noun/determiner (not a verb), it should be "whose"
            if next_word in ("book", "book", "car", "dog", "cat", "house", "phone",
                             "idea", "turn", "fault", "name", "child", "children",
                             "mother", "father", "sister", "brother", "friend",
                             "the", "a", "an", "this", "that", "these", "those",
                             "my", "your", "his", "her", "our", "their"):
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0), replacement="whose",
                    category=ErrorCategory.WORD_USAGE,
                    rule_id="WHOS_WHOSE",
                    message=f'Use "whose" (possessive) instead of "who\'s" before "{m.group(1)}".',
                    detector="fast_detector",
                    raw_confidence=0.85,
                ))

        # "it's tail" → "its tail" (it's + noun = possessive "its")
        for m in re.finditer(r'\bit\'s\b\s+(\w+)', text, re.IGNORECASE):
            next_word = m.group(1).lower()
            # Check if next word is a determiner, possessive, or noun (possessive context)
            possessive_followers = {"tail", "fur", "wings", "head", "eye", "eyes", "ear",
                             "ears", "nose", "mouth", "leg", "legs", "foot", "feet",
                             "hand", "hands", "body", "skin", "feathers", "claws",
                             "teeth", "horn", "horns", "shell", "back", "side",
                             "name", "color", "colour", "size", "shape", "sound",
                             "own", "way", "time", "place", "purpose", "paw", "paws",
                             "wings", "feathers", "tail", "fur", "bark", "leaves",
                             "roots", "branches", "trunk", "blossoms", "flowers"}
            # "it's" + a determiner is ALWAYS "it is a/..." — possessive "its"
            # never precedes a determiner, so only flag possession nouns.
            if next_word in possessive_followers:
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0), replacement=re.sub(r"\bit'?s\b", "its", m.group(0), count=1, flags=re.I),
                    category=ErrorCategory.WORD_USAGE,
                    rule_id="ITS_POSSESSIVE",
                    message=f'Use "its" (possessive) instead of "it\'s" before "{next_word}".',
                    detector="fast_detector",
                    raw_confidence=0.82,
                ))

        return candidates


# ══════════════════════════════════════════════════════════════════════════
#  TIER 2: NLP-BASED DETECTION (SVA, tense, articles, pronouns, etc.)
# ══════════════════════════════════════════════════════════════════════════

def _gerundize(base: str) -> str:
    """Form the -ing gerund of a base verb with standard doubling rules."""
    if base.endswith("e") and len(base) > 3:
        return base[:-1] + "ing"
    if (base.endswith(("c", "g", "m", "n", "p", "r", "s", "t"))
            and len(base) >= 3 and base[-3] not in "aeiou"
            and base[-2] in "aeiou" and base[-1] not in "aeiouwxy"):
        return base + base[-1] + "ing"
    return base + "ing"


def _gerund_base(gerund: str) -> str:
    """Best-effort recovery of the base verb from a gerund ("go to X").

    Used only as a regex backstop for 'want going' → 'want to go'; cases it
    mis-guesses (e.g. ambiguous consonant+ing) are usually covered by the
    POS-based branch, which uses the lemma.
    """
    if not gerund.endswith("ing") or len(gerund) <= 4:
        return gerund
    if gerund.endswith("ying") and len(gerund) > 5:
        return gerund[:-4] + "ie"
    b = gerund[:-3]
    if b.endswith("ee"):
        return b
    if len(b) > 1 and b[-1] == b[-2]:
        return b[:-1]
    if b and b[-1] in "cgtvzkpd":
        return b + "e"
    return b


IRREGULAR_PLURALS = {
    "child": "children", "person": "people", "man": "men", "woman": "women",
    "foot": "feet", "tooth": "teeth", "goose": "geese", "mouse": "mice",
    "ox": "oxen", "cactus": "cacti", "focus": "foci", "crisis": "crises",
    "analysis": "analyses", "basis": "bases", "datum": "data", "index": "indices",
}


def _pluralize(word: str) -> str:
    """Best-effort plural form for a (mostly regular) noun."""
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith("f") and len(word) > 1:
        return word[:-1] + "ves"
    return word + "s"


class NLPDetector:
    """spaCy-based grammar detection for complex errors."""

    def __init__(self):
        pass

    def detect(self, text: str, doc=None) -> List[ErrorCandidate]:
        if doc is None:
            nlp = get_nlp()
            doc = nlp(text)

        candidates = []
        candidates.extend(self._check_sva(text, doc))
        candidates.extend(self._check_tense(text, doc))
        candidates.extend(self._check_pronouns(text, doc))
        candidates.extend(self._check_articles(text, doc))
        candidates.extend(self._check_existential_there(text, doc))
        candidates.extend(self._check_compound_subject_pronouns(text, doc))
        candidates.extend(self._check_possessives(text, doc))
        candidates.extend(self._check_word_usage(text, doc))
        candidates.extend(self._check_prepositions(text, doc))
        candidates.extend(self._check_past_participle(text, doc))
        candidates.extend(self._check_comma_splice(text, doc))
        candidates.extend(self._check_modal_errors(text, doc))
        candidates.extend(self._check_sentence_fragments(text, doc))
        candidates.extend(self._check_do_support(text, doc))
        candidates.extend(self._check_have_base_form(text, doc))
        candidates.extend(self._check_was_were_base(text, doc))
        candidates.extend(self._check_double_comparative(text, doc))
        candidates.extend(self._check_run_on_sentences(text, doc))
        candidates.extend(self._check_double_subject(text, doc))
        candidates.extend(self._check_possessive_apostrophes(text, doc))
        candidates.extend(self._check_contraction_apostrophes(text, doc))
        candidates.extend(self._check_didnt_past(text, doc))
        candidates.extend(self._check_stative_verbs(text, doc))
        candidates.extend(self._check_preposition_collocations_nlp(text, doc))
        candidates.extend(self._check_since_for_duration(text, doc))
        candidates.extend(self._check_one_of_plural(text, doc))
        candidates.extend(self._check_article_missing(text, doc))
        candidates.extend(self._check_article_a_an(text, doc))
        candidates.extend(self._check_tense_consistency(text, doc))
        candidates.extend(self._check_run_on_no_punct(text, doc))
        candidates.extend(self._check_fragment(text, doc))
        candidates.extend(self._check_parallelism(text, doc))
        candidates.extend(self._check_object_pronoun(text, doc))
        candidates.extend(self._check_possessive_its(text, doc))
        candidates.extend(self._check_missing_auxiliary(text, doc))
        candidates.extend(self._check_missing_plural(text, doc))
        candidates.extend(self._check_noun_possessive(text, doc))
        candidates.extend(self._check_uncountable_plural(text, doc))
        candidates.extend(self._check_every_singular(text, doc))
        candidates.extend(self._check_prefer_than(text, doc))
        candidates.extend(self._check_explain_dative(text, doc))
        candidates.extend(self._check_irregular_plurals(text, doc))
        candidates.extend(self._check_mass_noun_sva(text, doc))
        candidates.extend(self._check_missing_article_phrases(text, doc))
        candidates.extend(self._check_possessive_plurals(text, doc))
        candidates.extend(self._check_duplicate_verb(text, doc))
        candidates.extend(self._check_scheduled_on(text, doc))
        candidates.extend(self._check_backshift(text, doc))
        candidates.extend(self._check_anybody_declarative(text, doc))
        candidates.extend(self._check_advanced_patterns(text, doc))
        return candidates


    def _check_advanced_patterns(self, text: str, doc) -> List[ErrorCandidate]:
        """Batch of advanced pattern detectors for common L2 English errors."""
        candidates = []
        nlp_model = get_nlp()

        # 1. Adjective used as adverb after verb: "speaks English very good" → "well"
        ADJ_TO_ADV = {
            "good": "well", "bad": "badly", "quick": "quickly", "slow": "slowly",
            "loud": "loudly", "quiet": "quietly",
            "beautiful": "beautifully", "careful": "carefully", "successful": "successfully",
            "angry": "angrily", "perfect": "perfectly", "gentle": "gently",
            "brave": "bravely", "clear": "clearly", "easy": "easily",
            "happy": "happily", "sad": "sadly", "deep": "deeply",
            "near": "nearly", "high": "highly",
        }
        for token in doc:
            if token.dep_ in ("advmod", "acomp") and token.lower_ in ADJ_TO_ADV:
                head = token.head
                if head.pos_ in ("VERB", "AUX") or head.dep_ == "ROOT":
                    # Skip adjectives after linking verbs: "was careful", "seems good"
                    LINKING_VERBS = {"be", "is", "are", "was", "were", "am", "been", "being",
                                     "seem", "seems", "seemed", "appear", "appears", "appeared",
                                     "become", "becomes", "became",
                                     "feel", "feels", "felt", "look", "looks", "looked",
                                     "sound", "sounds", "sounded", "taste", "tastes", "tasted",
                                     "smell", "smells", "smelled", "remain", "remains", "remained"}
                    if head.lower_ in LINKING_VERBS:
                        continue
                    replacement = ADJ_TO_ADV[token.lower_]
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement=replacement,
                        category=ErrorCategory.ADVERB,
                        rule_id="ADVERB_FORM",
                        message=f'Use the adverb form "{replacement}" after a verb instead of "{token.text}".',
                        detector="nlp_detector", raw_confidence=0.88,
                    ))
            # "completed the task successful" — flat adverb position
            if token.dep_ == "oprd" and token.lower_ in ADJ_TO_ADV:
                replacement = ADJ_TO_ADV[token.lower_]
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx + len(token.text),
                    original=token.text, replacement=replacement,
                    category=ErrorCategory.ADVERB,
                    rule_id="ADVERB_FORM",
                    message=f'Use the adverb form "{replacement}" instead of "{token.text}".',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 1b. "real good" → "really good" (adjective "real" as adverb)
        for m in re.finditer(
                r"\breal\s+(good|bad|well|big|small|nice|new|hard|fast|slow|"
                r"quick|happy|sad|easy|difficult|great|tired|hot|cold)\b",
                text, re.I):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.start() + len("real"),
                original="real", replacement="really",
                category=ErrorCategory.ADVERB,
                rule_id="ADVERB_FORM",
                message='Use the adverb form "really" instead of "real".',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 2. Preposition collocations (expanded)
        PREP_COLLOC = {
            "responsible": {"of": "for"},
            "interested": {"to": "in", "on": "in"},
            "married": {"with": "to"},
            "good": {"in": "at"},
            "different": {"than": "from", "to": "from"},
            "knowledge": {"about": "of", "in": "of"},
            "arrived": {"to": "in"},
            "born": {"on": "in"},
            "waiting": {"your": None},  # missing preposition
            "discuss": {"about": None},  # no preposition needed
        }
        # Exceptions for preposition collocations — fixed expressions
        _PREP_EXCEPTIONS = {
            "arrived": {"in": {"time"}},  # "arrived in time" is correct
        }
        for token in doc:
            lower = token.lower_
            if lower in PREP_COLLOC:
                for child in token.children:
                    if child.dep_ == "prep" and child.lower_ in PREP_COLLOC[lower]:
                        # Check exceptions
                        if lower in _PREP_EXCEPTIONS and child.lower_ in _PREP_EXCEPTIONS[lower]:
                            pobjs = [c for c in child.children if c.dep_ == "pobj"]
                            if pobjs and pobjs[0].lower_ in _PREP_EXCEPTIONS[lower][child.lower_]:
                                continue
                        correct = PREP_COLLOC[lower][child.lower_]
                        if correct is None:
                            # Remove the preposition
                            candidates.append(ErrorCandidate(
                                start=child.idx, end=child.idx + len(child.text) + 1,
                                original=child.text + " ", replacement="",
                                category=ErrorCategory.PREPOSITION,
                                rule_id="PREPOSITION_COLLOCATION",
                                message=f'Do not use a preposition after "{lower}".',
                                detector="nlp_detector", raw_confidence=0.85,
                            ))
                        else:
                            candidates.append(ErrorCandidate(
                                start=child.idx, end=child.idx + len(child.text),
                                original=child.text, replacement=correct,
                                category=ErrorCategory.PREPOSITION,
                                rule_id="PREPOSITION_COLLOCATION",
                                message=f'Use "{correct}" instead of "{child.text}" after "{lower}".',
                                detector="nlp_detector", raw_confidence=0.88,
                            ))
            # "interested to learn" → "interested in learning"
            if lower == "interested":
                for child in token.children:
                    if child.dep_ == "prep" and child.lower_ == "to":
                        candidates.append(ErrorCandidate(
                            start=child.idx, end=child.idx + len(child.text),
                            original=child.text, replacement="in",
                            category=ErrorCategory.PREPOSITION,
                            rule_id="PREPOSITION_COLLOCATION",
                            message='Use "in" after "interested" (e.g., "interested in learning").',
                            detector="nlp_detector", raw_confidence=0.85,
                        ))
                # Also check: "to" is aux of xcomp verb under interested
                for child in token.children:
                    if child.dep_ == "xcomp":
                        for sub in child.children:
                            if sub.dep_ == "aux" and sub.lower_ == "to":
                                candidates.append(ErrorCandidate(
                                    start=sub.idx, end=sub.idx + len(sub.text),
                                    original=sub.text, replacement="in",
                                    category=ErrorCategory.PREPOSITION,
                                    rule_id="PREPOSITION_COLLOCATION",
                                    message='Use "in" after "interested" (e.g., "interested in learning").',
                                    detector="nlp_detector", raw_confidence=0.85,
                                ))

        # 3. Missing preposition: "waiting your response" → "waiting for"
        MISSING_PREP = {
            "waiting": ("for", "your"),
        }
        for token in doc:
            if token.lower_ in MISSING_PREP:
                correct_prep, expected_after = MISSING_PREP[token.lower_]
                for child in token.children:
                    if child.dep_ == "dobj" and child.lower_ == expected_after:
                        candidates.append(ErrorCandidate(
                            start=token.idx + len(token.text), end=child.idx,
                            original="", replacement=f" {correct_prep}",
                            category=ErrorCategory.PREPOSITION,
                            rule_id="MISSING_PREPOSITION",
                            message=f'Use "{token.text} {correct_prep} {child.text}" (add preposition).',
                            detector="nlp_detector", raw_confidence=0.85,
                        ))

        # 4. "reach at" / "enter into" — unnecessary preposition
        UNNECESSARY_PREP = {
            "reached": {"at": None},
            "entered": {"into": None},
        }
        for token in doc:
            if token.lower_ in UNNECESSARY_PREP:
                for child in token.children:
                    if child.dep_ == "prep" and child.lower_ in UNNECESSARY_PREP[token.lower_]:
                        candidates.append(ErrorCandidate(
                            start=child.idx - 1, end=child.idx + len(child.text),
                            original=" " + child.text, replacement="",
                            category=ErrorCategory.PREPOSITION,
                            rule_id="UNNECESSARY_PREPOSITION",
                            message=f'Do not use a preposition after "{token.text}".',
                            detector="nlp_detector", raw_confidence=0.85,
                        ))

        # 5. "born on 2001" → "born in"
        for token in doc:
            if token.lower_ == "born":
                for child in token.children:
                    if child.dep_ == "prep" and child.lower_ == "on":
                        nxt = None
                        for c in child.children:
                            if c.dep_ == "pobj":
                                nxt = c
                                break
                        if nxt and nxt.pos_ == "NUM":
                            candidates.append(ErrorCandidate(
                                start=child.idx, end=child.idx + len(child.text),
                                original=child.text, replacement="in",
                                category=ErrorCategory.PREPOSITION,
                                rule_id="PREPOSITION_COLLOCATION",
                                message='Use "in" with years (e.g., "born in 2001").',
                                detector="nlp_detector", raw_confidence=0.90,
                            ))

        # 6. Gerund/infinitive errors
        VERB_GERUND = {  # verb + gerund (not infinitive)
            "enjoy", "enjoys", "enjoyed", "finish", "finishes", "finished",
            "mind", "minds", "minded", "suggest", "suggests", "suggested",
            "practice", "practices", "practiced", "consider", "considers",
            "considered", "keep", "keeps", "kept", "avoid", "avoids", "avoided",
            "imagine", "imagines", "imagined", "risk", "risks", "miss", "misses",
            "missed", "deny", "denies", "denied", "admit", "admits", "admitted",
            "appreciate", "appreciates", "appreciated", "delay", "delays",
            "delayed", "escape", "escapes", "escaped", "can't help", "feel like",
        }
        VERB_INFINITIVE = {  # verb + infinitive (not gerund)
            "want", "wants", "wanted", "wanting", "need", "needs", "needed",
            "decide", "decides", "decided", "hope", "hopes", "hoped", "hoping",
            "plan", "plans", "planned", "planning", "expect", "expects",
            "expected", "expecting", "promise", "promises", "promised",
            "refuse", "refuses", "refused", "offer", "offers", "offered",
            "agree", "agrees", "agreed", "pretend", "pretends", "pretended",
            "manage", "manages", "managed", "afford", "affords", "afforded",
            "fail", "fails", "failed", "choose", "chooses", "chose", "chosen",
            "tend", "tends", "seem", "seems", "appear", "appears",
            "happen", "happens", "claim", "claims",
        }
        for token in doc:
            if token.lower_ in VERB_GERUND and token.pos_ in ("VERB", "AUX"):
                for child in token.children:
                    if child.dep_ in ("xcomp", "ccomp") and child.tag_ == "VB":
                        candidates.append(ErrorCandidate(
                            start=child.idx, end=child.idx + len(child.text),
                            original=child.text, replacement=_gerundize(child.text),
                            category=ErrorCategory.GRAMMAR,
                            rule_id="GERUND_INFINITIVE",
                            message=f'After "{token.text}", use a gerund (-ing form) instead of an infinitive.',
                            detector="nlp_detector", raw_confidence=0.85,
                        ))
            if token.lower_ in VERB_INFINITIVE and token.pos_ in ("VERB", "AUX"):
                for child in token.children:
                    if child.dep_ in ("xcomp", "ccomp") and child.tag_ == "VBG":
                        candidates.append(ErrorCandidate(
                            start=child.idx, end=child.idx + len(child.text),
                            original=child.text, replacement=f"to {child.lemma_}",
                            category=ErrorCategory.GRAMMAR,
                            rule_id="GERUND_INFINITIVE",
                            message=f'After "{token.text}", use an infinitive ("to {child.lemma_}") instead of a gerund.',
                            detector="nlp_detector", raw_confidence=0.85,
                        ))

        # 7. "enjoy to watch" → "enjoy watching" (regex-based)
        enjoy_inf_pattern = re.compile(r'\b(enjoys?|enjoyed|finishes?|finished|practices?|practiced|practicing|considers?|considered|keeps?|kept|avoids?|avoided|imagines?|imagined|miss(?:es|ed)?|den(?:ies|yed|y)|admits?|admitted|appreciates?|appreciated|delays?|delayed|escapes?|escaped|enjoy|finish|practice|consider|keep|avoid|imagine|miss|deny|admit|appreciate|delay|escape)\s+to\s+(\w+)\b', re.IGNORECASE)
        for m in enjoy_inf_pattern.finditer(text):
            verb = m.group(1).lower()
            inf = m.group(2)
            if verb in VERB_GERUND:
                gerund = _gerundize(inf)
                candidates.append(ErrorCandidate(
                    start=m.start(2), end=m.end(2),
                    original=inf, replacement=gerund,
                    category=ErrorCategory.GRAMMAR,
                    rule_id="GERUND_INFINITIVE",
                    message=f'After "{verb}", use the gerund form "{gerund}" instead of "to {inf}".',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 8. "want going" → "want to go"
        want_gerund_pattern = re.compile(r'\b((?:want(?:s|ing|ed)?|need(?:s|ed)?|decide(?:s|d)?|hope(?:s|d)?|plan(?:s|ned|ning)?|expect(?:s|ed|ing)?|promise(?:s|d)?|refuse(?:s|d)?|offer(?:s|ed)?|agree(?:s|d)?|pretend(?:s|ed)?|manage(?:s|d)?|afford(?:s|ed)?|fail(?:s|ed)?|choose(?:s)?|tend(?:s)?|seem(?:s)?|appear(?:s)?|happen(?:s)?|claim(?:s)?))\s+(\w+ing)\b', re.IGNORECASE)
        for m in want_gerund_pattern.finditer(text):
            verb = m.group(1).lower()
            gerund = m.group(2)
            if verb in VERB_INFINITIVE:
                base = _gerund_base(gerund)
                candidates.append(ErrorCandidate(
                    start=m.start(2), end=m.end(2),
                    original=gerund, replacement=f"to {base}",
                    category=ErrorCategory.GRAMMAR,
                    rule_id="GERUND_INFINITIVE",
                    message=f'After "{verb}", use the infinitive form "to {base}".',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 9. Dative errors: "said me" → "said to me", "explained me" → "explained to me"
        # "told me that..." is correct (told accepts direct object), but "said me" needs "to"
        DATIVE_PREP = {"said", "told", "asked", "explained", "suggested", "replied", "mentioned", "whispered", "shouted"}
        DATIVE_ACCEPTS_DO = {"told"}  # These verbs accept direct object without "to"
        for token in doc:
            if token.lower_ in DATIVE_PREP and token.pos_ in ("VERB", "AUX"):
                # Skip "told me that..." (correct) but not "said me that..." (wrong)
                if token.lower_ in DATIVE_ACCEPTS_DO:
                    has_ccomp = any(c.dep_ == "ccomp" for c in token.children)
                    if has_ccomp:
                        continue
                for child in token.children:
                    if child.dep_ in ("dobj",) and child.lower_ in ("me", "him", "her", "us", "them"):
                        has_prep = any(c.dep_ == "prep" and c.lower_ == "to" for c in token.children)
                        if not has_prep:
                            candidates.append(ErrorCandidate(
                                start=token.idx + len(token.text), end=child.idx,
                                original="", replacement=" to",
                                category=ErrorCategory.GRAMMAR,
                                rule_id="DATIVE_PREPOSITION",
                                message=f'Use "{token.text} to {child.text}" (add "to" before the indirect object).',
                                detector="nlp_detector", raw_confidence=0.85,
                            ))

        # 10. "made me to laugh" → "made me laugh" (causative + bare infinitive)
        CAUSATIVE = {"make", "makes", "made", "let", "lets", "have", "has", "had"}
        for token in doc:
            if token.lower_ in CAUSATIVE and token.pos_ in ("VERB", "AUX"):
                for child in token.children:
                    if child.dep_ in ("ccomp", "xcomp") and child.text.lower() == "to":
                        for grandchild in child.children:
                            if grandchild.pos_ == "VERB":
                                candidates.append(ErrorCandidate(
                                    start=child.idx, end=grandchild.idx + len(grandchild.text),
                                    original=f"to {grandchild.text}", replacement=grandchild.text,
                                    category=ErrorCategory.GRAMMAR,
                                    rule_id="CAUSATIVE_BARE_INFINITIVE",
                                    message=f'After causative verbs (make/let/have), use the bare infinitive without "to".',
                                    detector="nlp_detector", raw_confidence=0.90,
                                ))
        # Also check "made me to laugh" pattern via regex
        # Exclude "have been to" (correct idiom: "I have been to Paris")
        causative_to_pattern = re.compile(r'\b(made?|lets?|have|has|had)\s+(\w+)\s+to\s+(\w+)', re.IGNORECASE)
        for m in causative_to_pattern.finditer(text):
            verb = m.group(1).lower()
            if verb in CAUSATIVE and m.group(2).lower() != "been":
                candidates.append(ErrorCandidate(
                    start=m.start(3) - 4, end=m.end(3),
                    original=f"to {m.group(3)}", replacement=m.group(3),
                    category=ErrorCategory.GRAMMAR,
                    rule_id="CAUSATIVE_BARE_INFINITIVE",
                    message=f'After causative verbs, use bare infinitive without "to".',
                    detector="nlp_detector", raw_confidence=0.90,
                ))

        # 11. "told me to completing" → "told me to complete"
        for token in doc:
            if token.lower_ in ("to",) and token.tag_ == "TO":
                # spaCy: "to" is aux of the verb (head). Check head, not children.
                head_verb = token.head
                if head_verb.tag_ == "VBG" and head_verb.pos_ == "VERB":
                    candidates.append(ErrorCandidate(
                        start=head_verb.idx, end=head_verb.idx + len(head_verb.text),
                        original=head_verb.text, replacement=head_verb.lemma_,
                        category=ErrorCategory.GRAMMAR,
                        rule_id="INFINITIVE_FORM",
                        message=f'After "to", use the base form "{head_verb.lemma_}" instead of "{head_verb.text}".',
                        detector="nlp_detector", raw_confidence=0.90,
                    ))
                # Also check children (some parses may differ)
                for child in token.children:
                    if child.tag_ == "VBG" and child.pos_ == "VERB":
                        candidates.append(ErrorCandidate(
                            start=child.idx, end=child.idx + len(child.text),
                            original=child.text, replacement=child.lemma_,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="INFINITIVE_FORM",
                            message=f'After "to", use the base form "{child.lemma_}" instead of "{child.text}".',
                            detector="nlp_detector", raw_confidence=0.90,
                        ))

        # 12. Redundant pronoun in relative clause: "which I bought it" → "which I bought"
        for token in doc:
            if token.dep_ in ("dobj",) and token.pos_ == "PRON" and token.lower_ in ("it", "him", "her", "them", "me"):
                head = token.head
                # Check if head has a relative clause dependency
                if head.dep_ == "relcl" or any(c.dep_ == "relcl" for c in head.children):
                    continue
                # Check if there's a relative pronoun (which/who/that) governing this clause
                for t2 in doc:
                    if t2.dep_ in ("nsubj", "dobj") and t2.tag_ in ("WP", "WDT"):
                        if t2.head == head or t2.head.head == head:
                            candidates.append(ErrorCandidate(
                                start=token.idx - 1 if token.idx > 0 else token.idx,
                                end=token.idx + len(token.text),
                                original=" " + token.text, replacement="",
                                category=ErrorCategory.PRONOUN,
                                rule_id="REDUNDANT_PRONOUN",
                                message=f'Remove the redundant pronoun "{token.text}" — it is already implied by the relative clause.',
                                detector="nlp_detector", raw_confidence=0.82,
                            ))
                            break

        # 13. "which" for people → "who": "the girl which won" → "the girl who won"
        # Only flag when the antecedent is clearly a person word
        PERSON_WORDS = {"girl", "boy", "man", "woman", "child", "children", "baby",
                        "person", "people", "student", "teacher", "doctor", "nurse",
                        "engineer", "manager", "director", "professor", "worker",
                        "friend", "neighbor", "colleague", "customer", "client",
                        "member", "player", "artist", "writer", "actor", "actress",
                        "singer", "dancer", "chef", "driver", "artist", "host",
                        "guest", "speaker", "winner", "leader", "president", "chairman",
                        "member", "representative", "volunteer", "citizen", "resident"}
        for token in doc:
            if token.lower_ == "which" and token.tag_ == "WDT":
                antecedent = None
                # Strategy 1: "which" is nsubj of a relcl verb — antecedent is the head of the relcl
                if token.dep_ == "nsubj" and token.head.dep_ == "relcl":
                    antecedent = token.head.head  # the noun the relative clause modifies
                # Strategy 2: look for sibling nsubj/dobj
                if antecedent is None:
                    for child in token.head.children:
                        if child.dep_ in ("nsubj", "dobj") and child != token:
                            antecedent = child
                            break
                # Strategy 3: look for relcl and its head's children
                if antecedent is None:
                    for t2 in doc:
                        if t2.dep_ == "relcl" and t2.head == token.head:
                            for c in t2.head.children:
                                if c.pos_ in ("NOUN", "PROPN") and c.dep_ in ("nsubj",):
                                    antecedent = c
                                    break
                if antecedent and antecedent.pos_ in ("NOUN", "PROPN"):
                    # Only flag if the antecedent is a person word
                    if antecedent.lower_ not in PERSON_WORDS:
                        continue
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="who",
                        category=ErrorCategory.PRONOUN,
                        rule_id="WHICH_WHO",
                        message='Use "who" instead of "which" when referring to people.',
                        detector="nlp_detector", raw_confidence=0.85,
                    ))

        # 14. "Although...but" → remove "but"
        although_but = re.compile(r'\b(although|though|even though)\s+[^,]+,\s*but\b', re.IGNORECASE)
        for m in although_but.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=m.group(0).replace(" but", ""),
                category=ErrorCategory.CONJUNCTION,
                rule_id="ALTHOUGH_BUT",
                message='Use "although" or "but", not both together. Remove "but".',
                detector="nlp_detector", raw_confidence=0.88,
            ))

        # 15. "Despite of" → "Despite"
        despite_of = re.compile(r'\bdespite\s+of\b', re.IGNORECASE)
        for m in despite_of.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement="despite",
                category=ErrorCategory.PREPOSITION,
                rule_id="DESPITE_OF",
                message='Use "despite" (not "despite of").',
                detector="nlp_detector", raw_confidence=0.92,
            ))

        # 16. "In spite he was" → "In spite of" or "Despite"
        in_spite_pattern = re.compile(r'\bin spite\s+(?!of\b)(\w)', re.IGNORECASE)
        for m in in_spite_pattern.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original="in spite " + m.group(1), replacement="in spite of " + m.group(1),
                category=ErrorCategory.PREPOSITION,
                rule_id="IN_SPITE_OF",
                message='Use "in spite of" (not "in spite" without "of").',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 17. "The reason is because" → "The reason is that"
        reason_because = re.compile(r'\bthe reason (is|was|were|are) because\b', re.IGNORECASE)
        for m in reason_because.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=f"the reason {m.group(1)} that",
                category=ErrorCategory.GRAMMAR,
                rule_id="REDUNDANT_BECAUSE",
                message='Use "the reason is that" (not "the reason is because").',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 18. "must to finish" → "must finish" (modal + to)
        MODALS_SET = {"can", "could", "may", "might", "must", "shall", "should", "will", "would", "need", "ought", "dare"}
        for token in doc:
            if token.dep_ == "aux" and token.lower_ in MODALS_SET:
                for child in token.head.children:
                    if child.dep_ == "aux" and child.lower_ == "to" and child.tag_ == "TO":
                        candidates.append(ErrorCandidate(
                            start=child.idx, end=child.idx + len(child.text) + 1,
                            original="to ", replacement="",
                            category=ErrorCategory.MODAL,
                            rule_id="MODAL_TO",
                            message=f'Do not use "to" after the modal "{token.text}".',
                            detector="nlp_detector", raw_confidence=0.92,
                        ))

        # 19. "can able to" → "can" (redundant)
        can_able = re.compile(r'\bcan\s+able\s+to\b', re.IGNORECASE)
        for m in can_able.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement="can",
                category=ErrorCategory.GRAMMAR,
                rule_id="REDUNDANT_ABLE",
                message='Use "can" (not "can able to" — they mean the same thing).',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 20. "I am agree" → "I agree"
        am_agree = re.compile(r'\bI\s+am\s+agree\b', re.IGNORECASE)
        for m in am_agree.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement="I agree",
                category=ErrorCategory.GRAMMAR,
                rule_id="ADJECTIVE_FORM",
                message='"Agree" is a verb, not an adjective. Say "I agree" (not "I am agree").',
                detector="nlp_detector", raw_confidence=0.92,
            ))

        # 21. "I am knowing" → "I know" (stative verb)
        am_knowing = re.compile(r'\bI\s+am\s+knowing\b', re.IGNORECASE)
        for m in am_knowing.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement="I know",
                category=ErrorCategory.GRAMMAR,
                rule_id="STATIVE_BE",
                message='"Know" is a stative verb and is not used in the progressive form. Say "I know".',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 22. "I wish I was knowing" → "I wish I knew"
        wish_was_knowing = re.compile(r'\bI wish I was knowing\b', re.IGNORECASE)
        for m in wish_was_knowing.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement="I wish I knew",
                category=ErrorCategory.TENSE,
                rule_id="STATIVE_BE",
                message='Use "I wish I knew" (simple past after "wish" for present wishes).',
                detector="nlp_detector", raw_confidence=0.88,
            ))

        # 23. "too much hot" → "too hot"
        too_much_adj = re.compile(r'\btoo much\s+(hot|cold|big|small|loud|quiet|fast|slow|hard|easy|good|bad|nice|ugly|long|short|tall|wide|deep|thick|thin|fat|old|new|young|rich|poor|bright|dark|strong|weak|clean|dirty|dry|wet|soft|sharp|bitter|sweet|sour|spicy|salty)\b', re.IGNORECASE)
        for m in too_much_adj.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=f"too {m.group(1)}",
                category=ErrorCategory.GRAMMAR,
                rule_id="TOO_MUCH_ADJ",
                message=f'Use "too {m.group(1)}" (not "too much {m.group(1)}"). "Too much" is used with uncountable nouns.',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 24. "less books" → "fewer books" (countable)
        less_countable = re.compile(r'\bless\s+(\w*s)\b', re.IGNORECASE)
        for m in less_countable.finditer(text):
            word = m.group(1)
            if word.lower() not in UNCOUNTABLE_NOUNS and not word.lower().endswith("ness"):
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0), replacement=f"fewer {word}",
                    category=ErrorCategory.GRAMMAR,
                    rule_id="LESS_FEWER",
                    message=f'Use "fewer {word}" (not "less {word}") for countable nouns.',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 25. "much people" → "many people"
        much_countable = re.compile(r'\bmuch\s+people\b', re.IGNORECASE)
        for m in much_countable.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement="many people",
                category=ErrorCategory.GRAMMAR,
                rule_id="MUCH_MANY",
                message='Use "many people" (not "much people"). "Many" is used with countable nouns.',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 26. "few water" → "little water" (uncountable)
        few_uncountable = re.compile(r'\bfew\s+(water|money|information|advice|furniture|luggage|equipment|homework|news|music|work|bread|milk|tea|coffee|rice|sugar|salt|oil|wood|paper|glass|gold|silver|iron|cotton|leather|plastic|rubber|cloth|fruit|meat|fish|cheese|butter|wine|beer|juice|oil|air|land|weather|raining|snow|ice|fire|earth|mud|dust|sand|smoke|air|space|time|money|knowledge|education|health|happiness|love|hate|anger|fear|trouble|damage|progress|research|traffic|evidence|homework|housework|work|baggage|luggage|furniture|equipment|machinery|software|hardware)\b', re.IGNORECASE)
        for m in few_uncountable.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=f"little {m.group(1)}",
                category=ErrorCategory.GRAMMAR,
                rule_id="FEW_LITTLE",
                message=f'Use "little {m.group(1)}" (not "few {m.group(1)}") for uncountable nouns.',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 27. Missing serial comma: "reading writing and painting" → "reading, writing, and painting"
        for sent in doc.sents:
            sent_text = sent.text
            items = re.findall(r'\b(\w+ing)\s+(\w+ing)\s+and\s+(\w+ing)\b', sent_text)
            for item_match in items:
                full = f"{item_match[0]} {item_match[1]} and {item_match[2]}"
                if ", " not in sent_text.split(full)[0][-5:]:
                    m = re.search(re.escape(full), sent_text)
                    if m:
                        first_comma_pos = sent.start_char + m.start() + len(item_match[0])
                        candidates.append(ErrorCandidate(
                            start=first_comma_pos, end=first_comma_pos,
                            original="", replacement=", ",
                            category=ErrorCategory.PUNCTUATION,
                            rule_id="MISSING_SERIAL_COMMA",
                            message='Add commas to separate items in a list (e.g., "reading, writing, and painting").',
                            detector="nlp_detector", raw_confidence=0.80,
                        ))

        # 28. "Where was I going" (inverted word order in embedded question)
        embedded_q_pattern = re.compile(r'\b(when|where|why|how|what|which|who|whom|whose)\s+(was|were|is|are|do|does|did|can|could|will|would|shall|should|may|might|must)\s+(I|he|she|it|we|they|you)\b', re.IGNORECASE)
        for m in embedded_q_pattern.finditer(text):
            # Skip if it's a direct question at start of sentence
            prefix = text[:m.start()].strip()
            if not prefix or prefix.endswith('.'):
                continue
            word1 = m.group(1)
            aux = m.group(2)
            pronoun = m.group(3)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0),
                replacement=f"{word1} {pronoun} {aux}",
                category=ErrorCategory.WORD_ORDER,
                rule_id="EMBEDDED_QUESTION_ORDER",
                message=f'In an embedded question, use statement word order: "{word1} {pronoun} {aux}" (not "{aux} {pronoun}").',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 29. "I don't know what does he wants" → "what he wants"
        does_he_pattern = re.compile(r'\bwhat\s+does\s+(\w+)\s+(\w+)\b', re.IGNORECASE)
        for m in does_he_pattern.finditer(text):
            subject = m.group(1)
            verb = m.group(2)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0),
                replacement=f"what {subject} {verb}",
                category=ErrorCategory.WORD_ORDER,
                rule_id="EMBEDDED_QUESTION_ORDER",
                message=f'In an embedded question, use statement word order: "what {subject} {verb}".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 30. Inverted adverbs: "Never I have" → "I have never"
        inverted_adv = re.compile(r'\b(Never|Only|Rarely|Seldom|Hardly|Barely|Not only)\s+(I|he|she|it|we|they|you)\s+(\w+)\b', re.IGNORECASE)
        for m in inverted_adv.finditer(text):
            prefix = text[:m.start()].strip()
            if not prefix or prefix.endswith('.'):
                # Start of sentence — inverted order
                adv = m.group(1)
                pronoun = m.group(2)
                verb = m.group(3)
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0),
                    replacement=f"{pronoun} {verb} {adv.lower()}",
                    category=ErrorCategory.WORD_ORDER,
                    rule_id="INVERTED_ADVERB_ORDER",
                    message=f'Put the subject before the verb: "{pronoun} {verb} {adv.lower()}".',
                    detector="nlp_detector", raw_confidence=0.82,
                ))

        # 31. "Theyre" → "They're", "Its" → "It's" (missing apostrophe in contraction)
        for contraction in ["theyre", "were", "youre", "hes", "shes", "thats", "whats"]:
            pattern = re.compile(r'\b' + contraction + r'\b', re.IGNORECASE)
            for m in pattern.finditer(text):
                word = m.group(0)
                correct_map = {"theyre": "they're", "were": "we're", "youre": "you're",
                              "hes": "he's", "shes": "she's", "thats": "that's", "whats": "what's"}
                # Only flag if it's not already a valid word
                if contraction == "were" and word.lower() == "were":
                    continue  # "were" is valid
                if contraction in correct_map:
                    candidates.append(ErrorCandidate(
                        start=m.start(), end=m.end(),
                        original=word, replacement=correct_map[contraction],
                        category=ErrorCategory.PUNCTUATION,
                        rule_id="MISSING_APOSTROPHE",
                        message=f'Use the contraction "{correct_map[contraction]}" (with apostrophe).',
                        detector="nlp_detector", raw_confidence=0.88,
                    ))

        # 32. "He is work in this company" → "He works" or "He is working"
        is_base_verb = re.compile(r'\b(is|are|was|were)\s+(work|go|come|see|eat|drink|run|walk|talk|play|write|read|speak|listen|watch|help|start|stop|finish|open|close|take|give|make|put|get|set|run)\b', re.IGNORECASE)
        for m in is_base_verb.finditer(text):
            aux = m.group(1)
            verb = m.group(2)
            if aux.lower() in ("is", "are") and verb.lower() in ("work",):
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0),
                    replacement=f"{aux} {verb}ing" if aux.lower() in ("is", "are") else f"{verb}s",
                    category=ErrorCategory.TENSE,
                    rule_id="PROGRESSIVE_FORM",
                    message=f'Use "{aux} {verb}ing" (progressive) or "{verb}s" (simple present).',
                    detector="nlp_detector", raw_confidence=0.82,
                ))

        # 33. "She told that" → "She told me that"
        for token in doc:
            if token.lower_ in ("told", "tells", "tell") and token.pos_ in ("VERB", "AUX"):
                has_obj = any(c.dep_ in ("dobj",) for c in token.children)
                has_indirect = any(c.dep_ in ("iobj",) for c in token.children)
                if not has_obj and not has_indirect:
                    for child in token.children:
                        if child.dep_ in ("ccomp", "advcl"):
                            candidates.append(ErrorCandidate(
                                start=token.idx + len(token.text), end=child.idx,
                                original="", replacement=" me",
                                category=ErrorCategory.GRAMMAR,
                                rule_id="MISSING_INDIRECT_OBJECT",
                                message=f'"Tell" requires an indirect object: "told me that..."',
                                detector="nlp_detector", raw_confidence=0.82,
                            ))
                            break

        # 34. "He asked me that whether" → "He asked me whether" (redundant "that")
        asked_that_whether = re.compile(r'\basked?\s+(\w+\s+)?that\s+whether\b', re.IGNORECASE)
        for m in asked_that_whether.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=m.group(0).replace(" that ", " "),
                category=ErrorCategory.GRAMMAR,
                rule_id="REDUNDANT_THAT",
                message='Remove "that" — "asked whether" is sufficient.',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 35. "Parallelism: cooking, dancing and to sing" → "cooking, dancing, and singing"
        # Already handled by _check_parallelism

        # 36. "extra verb: more than tea is" → "more than tea"
        for sent in doc.sents:
            sent_text = sent.text
            extra_verb_match = re.search(r'more than\s+\w+\s+(is|are|was|were|has|have|had)\b', sent_text, re.IGNORECASE)
            if extra_verb_match:
                candidates.append(ErrorCandidate(
                    start=sent.start_char + extra_verb_match.start(1),
                    end=sent.start_char + extra_verb_match.end(1),
                    original=extra_verb_match.group(1), replacement="",
                    category=ErrorCategory.GRAMMAR,
                    rule_id="EXTRA_VERB",
                    message='Remove the extra verb after "than" — "more than X" does not need a verb.',
                    detector="nlp_detector", raw_confidence=0.82,
                ))

        # 37. "They was" → "They were" (pronoun SVA fix for SPAcy misparse)
        PRONOUN_WAS = {"they": "were", "we": "were", "you": "were"}
        for token in doc:
            if token.dep_ == "nsubj" and token.lower_ in PRONOUN_WAS:
                head = token.head
                if head.text.lower() == "was" and head.pos_ in ("VERB", "AUX"):
                    # Check if there's also an "and" making it compound
                    replacement = PRONOUN_WAS[token.lower_]
                    candidates.append(ErrorCandidate(
                        start=head.idx, end=head.idx + len(head.text),
                        original=head.text, replacement=replacement,
                        category=ErrorCategory.SUBJECT_VERB_AGREEMENT,
                        rule_id="SVA",
                        message=f'The pronoun "{token.text}" requires "{replacement}" (not "was").',
                        detector="nlp_detector", raw_confidence=0.90,
                    ))

        # 38. "We were discuss" → "We were discussing"
        # Exclude: hurt (passive), open/close (state verbs), and other ambiguous forms
        were_base = re.compile(r'\b(were|was)\s+(discuss|go|come|see|eat|drink|run|walk|talk|play|write|read|speak|listen|watch|help|make|take|give|put|get|set|begin|start|stop|finish|try|use|need|want|like|love|hate|feel|think|know|believe|understand|remember|forget|seem|appear|become|remain|keep|turn|grow|fall|rise|hang|stand|sit|lie|fit|cost|cut|hit|let|put|read|set|shut|spread|throw|quit|bet|cast|burst|cast|cost|cut|hit|let|put|quit|read|set|shut|spread|throw)\b', re.IGNORECASE)
        for m in were_base.finditer(text):
            aux = m.group(1)
            verb = m.group(2)
            if aux.lower() in ("were", "was"):
                candidates.append(ErrorCandidate(
                    start=m.start(2), end=m.end(2),
                    original=verb, replacement=f"{verb}ing",
                    category=ErrorCategory.TENSE,
                    rule_id="PROGRESSIVE_FORM",
                    message=f'After "{aux}", use the present participle: "{aux} {verb}ing".',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 39. "He is work" → "He works" / "He is working"
        for token in doc:
            if token.dep_ == "ROOT" and token.tag_ == "VB" and token.text.lower() in ("work",):
                for child in token.children:
                    if child.dep_ == "aux" and child.text.lower() in ("is", "are", "was", "were"):
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement=f"{token.text}ing",
                            category=ErrorCategory.TENSE,
                            rule_id="PROGRESSIVE_FORM",
                            message=f'After "{child.text}", use "{token.text}ing" (progressive form).',
                            detector="nlp_detector", raw_confidence=0.82,
                        ))

        # 40. "has been send" → "has been sent" (passive with wrong pp)
        for token in doc:
            if token.text.lower() in HAVE_FORMS and token.dep_ == "aux":
                for child in token.children:
                    if child.dep_ == "auxpass" and child.text.lower() == "been":
                        for grandchild in child.head.children:
                            if grandchild.dep_ == "ROOT" or grandchild == child.head:
                                main_v = child.head if child.head != child else None
                                if main_v is None:
                                    main_v = token.head
                                if main_v and main_v.tag_ == "VB":
                                    lemma = main_v.lemma_
                                    if lemma in IRREGULAR_VERBS:
                                        pp = IRREGULAR_VERBS[lemma].get("pp")
                                        if pp and main_v.text.lower() != pp:
                                            candidates.append(ErrorCandidate(
                                                start=main_v.idx, end=main_v.idx + len(main_v.text),
                                                original=main_v.text, replacement=pp,
                                                category=ErrorCategory.TENSE,
                                                rule_id="WRONG_PAST_PARTICIPLE",
                                                message=f'In passive voice ("been {main_v.text}"), use the past participle "{pp}".',
                                                detector="nlp_detector", raw_confidence=0.88,
                                            ))

        # 41. "I will call you after I will finish" → "after I finish" (no future in time clauses)
        after_will = re.compile(r'\b(after|when|before|until|as soon as|once)\s+(I|he|she|it|we|they|you)\s+will\s+(\w+)', re.IGNORECASE)
        for m in after_will.finditer(text):
            prefix = text[:m.start()].strip()
            if not prefix or prefix[-1] in ".?!;":
                continue
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0),
                replacement=f"{m.group(1)} {m.group(2)} {m.group(3)}",
                category=ErrorCategory.TENSE,
                rule_id="TIME_CLAUSE_TENSE",
                message=f'In time clauses ("{m.group(1)}..."), use present tense instead of "will".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 42. "If I will get" → "If I get" (no future in conditionals)
        if_will = re.compile(r'\bif\s+(I|he|she|it|we|they|you)\s+will\s+(\w+)', re.IGNORECASE)
        for m in if_will.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0),
                replacement=f"if {m.group(1)} {m.group(2)}",
                category=ErrorCategory.TENSE,
                rule_id="CONDITIONAL_TENSE",
                message='Use present tense in "if" clauses (not "will").',
                detector="nlp_detector", raw_confidence=0.88,
            ))

        # 43. "He will send the document yesterday" → "would send"
        will_past_time = re.compile(r'\bwill\s+(\w+)\b.*\b(yesterday|last week|last month|last year|ago)\b', re.IGNORECASE)
        for m in will_past_time.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.start(1) + len(m.group(1)),
                original=m.group(1),
                replacement=m.group(1),
                category=ErrorCategory.TENSE,
                rule_id="TENSE_PAST_MARKER",
                message=f'The time marker "{m.group(2)}" requires past tense, not "will".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 44. Missing comma after introductory clause: "When I reached the station the train..."
        for sent in doc.sents:
            sent_text = sent.text
            intro_match = re.match(r'\b(When|While|If|Because|Although|Though|Since|Until|Before|After|As soon as|Once|Where|Unless|Whereas)\s+[^,]+?\s+(the|a|an|my|his|her|its|our|their|this|that|these|those|I|he|she|it|we|they|you)\b', sent_text, re.IGNORECASE)
            if intro_match and ',' not in intro_match.group(0):
                # Skip questions (end with ?)
                if sent_text.strip().endswith("?"):
                    continue
                # Skip sentences where the match covers most of the sentence (short sentence)
                total_words = len(sent_text.split())
                if total_words < 7:
                    continue
                # Also check if comma exists AFTER the match in the sentence
                after_match = sent_text[intro_match.end():]
                if ',' in after_match[:15]:
                    continue
                if len(sent_text) > len(intro_match.group(0)):
                    comma_pos = sent.start_char + intro_match.end() - 1
                    next_word = intro_match.group(2)
                    candidates.append(ErrorCandidate(
                        start=comma_pos - len(next_word), end=comma_pos - len(next_word) + 1,
                        original=" ", replacement=", ",
                        category=ErrorCategory.PUNCTUATION,
                        rule_id="MISSING_COMMA_INTRO_CLAUSE",
                        message='Add a comma after the introductory clause.',
                        detector="nlp_detector", raw_confidence=0.80,
                    ))

        # 45. "its raining" → "it's raining" (possessive vs contraction)
        its_raining = re.compile(r'\bits\s+(raining|snowing|raining|cold|hot|warm|cool|sunny|cloudy|windy|dark|light|late|early|time|getting)\b', re.IGNORECASE)
        for m in its_raining.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0),
                replacement=f"it's {m.group(1)}",
                category=ErrorCategory.WORD_USAGE,
                rule_id="ITS_CONTRACTION",
                message=f'Use "it\'s" (contraction of "it is") before "{m.group(1)}".',
                detector="nlp_detector", raw_confidence=0.88,
            ))

        # 46. "She is having two cars" → "She has two cars" (stative possession)
        # But "is having dinner/breakfast/lunch" is correct (activity, not possession)
        is_having = re.compile(r'\b(I|he|she|it|we|they|you)\s+is\s+having\b', re.IGNORECASE)
        for m in is_having.finditer(text):
            after = text[m.end():m.end()+30].strip().lower()
            meal_words = {"breakfast", "lunch", "dinner", "a meal", "snack", "coffee", "tea",
                         "conversation", "chat", "discussion", "meeting", "interview",
                         "problem", "issue", "difficulty", "trouble", "fun", "good time"}
            # Skip possessives/determiners: "his breakfast", "her lunch"
            after_stripped = re.sub(r'^(my|his|her|its|our|your|their|a|an|the|some|the)\s+', '', after)
            if any(after_stripped.startswith(mw) for mw in meal_words):
                continue
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0),
                replacement=f"{m.group(1)} has",
                category=ErrorCategory.TENSE,
                rule_id="STATIVE_POSSESSION",
                message=f'Use "{m.group(1)} has" (simple present) for possession. "Have" as a stative verb is not used in progressive.',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 47. "They're going to bring" — flag "Theyre" without apostrophe
        theyre_no_apost = re.compile(r'\b(theyre|youre|weire|hes|shes|thats|whats|its)\b', re.IGNORECASE)
        for m in theyre_no_apost.finditer(text):
            word = m.group(1).lower()
            contractions = {"theyre": "they're", "youre": "you're", "hes": "he's", "shes": "she's",
                           "thats": "that's", "whats": "what's"}
            if word in contractions:
                # Skip if it's already a valid word
                if word in ("its",):
                    continue  # "its" is valid as possessive
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(1), replacement=contractions[word],
                    category=ErrorCategory.PUNCTUATION,
                    rule_id="MISSING_APOSTROPHE",
                    message=f'Use the contraction "{contractions[word]}" (with apostrophe).',
                    detector="nlp_detector", raw_confidence=0.88,
                ))

        # 48. "He didnt" → "He didn't"
        didnt_pattern = re.compile(r'\b(didnt|doesnt|dont|isnt|arent|wasnt|werent|hasnt|havent|hadnt|couldnt|shouldnt|wouldnt|cant|wont|mustnt|shant|neednt)\b', re.IGNORECASE)
        for m in didnt_pattern.finditer(text):
            word = m.group(1).lower()
            apostrophe_map = {
                "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
                "isnt": "isn't", "arent": "aren't", "wasnt": "wasn't",
                "werent": "weren't", "hasnt": "hasn't", "havent": "haven't",
                "hadnt": "hadn't", "couldnt": "couldn't", "shouldnt": "shouldn't",
                "wouldnt": "wouldn't", "cant": "can't", "wont": "won't",
                "mustnt": "mustn't", "shant": "shan't", "neednt": "needn't",
            }
            if word in apostrophe_map:
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(1), replacement=apostrophe_map[word],
                    category=ErrorCategory.PUNCTUATION,
                    rule_id="MISSING_APOSTROPHE",
                    message=f'Use "{apostrophe_map[word]}" (with apostrophe).',
                    detector="nlp_detector", raw_confidence=0.90,
                ))

        return candidates

    def _check_sva(self, text: str, doc) -> List[ErrorCandidate]:
        """Subject-verb agreement."""
        candidates = []
        for sent in doc.sents:
            sent_start = sent.start_char

            for token in sent:
                if token.dep_ not in ("nsubj", "nsubjpass"):
                    continue

                subject = token
                main_verb = token.head

                # Check the main verb AND its auxiliaries
                verbs_to_check = []
                if main_verb.pos_ in ("VERB", "AUX"):
                    verbs_to_check.append(main_verb)
                for child in main_verb.children:
                    if child.dep_ == "aux" and child.pos_ in ("AUX", "VERB"):
                        verbs_to_check.append(child)
                    elif child.dep_ == "auxpass" and child.tag_ == "VBD":
                        # Passive "was/were + VBN": the participle doesn't inflect,
                        # but the be-auxiliary must agree ("The meeting were" -> "was",
                        # "Two sections was" -> "were").
                        verbs_to_check.append(child)

                for verb in verbs_to_check:
                    if verb.pos_ not in ("VERB", "AUX"):
                        continue

                    # Possessive "'s" before a plural verb usually hides a plural
                    # noun ("The teacher's are here." → "teachers are"). The
                    # POSSESSIVE_PLURAL rule owns this; don't fight it with SVA.
                    if subject.text.lower().endswith("'s") and verb.text.lower() in ("are", "were", "have"):
                        continue

                    # Skip verbs in dependent/subordinate clauses (xcomp)
                    verb_root = verb.head if verb.dep_ == "aux" else verb
                    if verb_root.dep_ in ("xcomp",):
                        continue

                    # Skip modal aux
                    if verb.dep_ == "aux" and verb.text.lower() in MODALS:
                        continue

                    # Skip if do/does/did aux
                    if verb.dep_ == "aux" and verb.text.lower() in DO_FORMS:
                        continue

                    # Skip when a have/has/had auxiliary is present —
                    # HAVE_BASE_FORM owns "has walk" → "has walked" (avoids "has walks").
                    if any(t.dep_ == "aux" and t.text.lower() in HAVE_FORMS
                           for t in verb.children):
                        continue

                    # Skip VB (base form) used as subjunctive/imperative: "God bless", "long live"
                    # Only skip if NO subject (imperative) or after "let/make" (causative, handled below)
                    # Don't skip if there's a clear subject — "manager give" is an SVA error
                    if verb.tag_ == "VB" and verb.dep_ not in ("aux", "auxpass"):
                        has_any_aux = any(t.dep_ in ("aux", "auxpass") for t in verb.children)
                        has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in verb.children)
                        if not has_any_aux and not has_subj:
                            continue
                        # Fixed subjunctive expressions: "God bless", "long live", etc.
                        _fixed_subjunctives = {
                            ("god", "bless"), ("long", "live"), ("god", "save"),
                            ("heaven", "help"), ("heaven", "forbid"),
                        }
                        if has_subj:
                            subj_token = next((t for t in verb.children if t.dep_ in ("nsubj", "nsubjpass")), None)
                            if subj_token and (subj_token.lower_, verb.lower_) in _fixed_subjunctives:
                                continue

                    # Skip subjunctive after "important/essential/necessary/crucial/vital that"
                    SUBJUNCTIVE_ADJS = {"important", "essential", "necessary", "crucial",
                                        "vital", "desirable", "imperative", "critical",
                                        "recommended", "suggested", "proposed", "requested",
                                        "demanded", "insisted", "ordered", "urged",
                                        "better", "worse", "best", "worst"}
                    if verb.tag_ in ("VB", "VBP") and verb.dep_ not in ("aux", "auxpass"):
                        # Check if preceded by "ADJ that" pattern
                        # Structure: verb.head = aux/be, verb.head has child ADJ (acomp)
                        # e.g., "It is important that he attend" → attend.head=is, is has child important
                        has_subj_context = False
                        for child in verb.children:
                            if child.dep_ == "mark" and child.lower_ == "that":
                                if verb.head.pos_ == "ADJ":
                                    has_subj_context = True
                                    break
                                # Check siblings of verb's head for adjectives
                                for sibling in verb.head.children:
                                    if sibling.pos_ == "ADJ" and sibling.dep_ in ("acomp", "attr"):
                                        has_subj_context = True
                                        break
                        if has_subj_context:
                            continue

                    # Skip "have/has/had" after a modal (e.g. "will have finished")
                    if verb.text.lower() in HAVE_FORMS and verb.head != verb:
                        if any(t.dep_ == "aux" and t.text.lower() in MODALS and t.head == verb.head for t in sent):
                            continue

                    # Skip VB (base form) after do-aux (e.g. "did she win", "does he go")
                    if verb.tag_ == "VB" and any(t.dep_ == "aux" and t.text.lower() in DO_FORMS and t.head == verb for t in sent):
                        continue

                    # Skip VB (base form) after modal (e.g. "could believe", "will be", "cannot come")
                    _modal_forms = MODALS | {"cannot", "can't", "won't", "wouldn't", "couldn't",
                                              "shouldn't", "mightn't", "mustn't", "shan't"}
                    def _is_modal(t):
                        """Check if token is a modal, handling contractions like ca (from ca-nt)."""
                        tl = t.text.lower()
                        if tl in _modal_forms or tl in MODALS:
                            return True
                        # Handle spaCy splitting contractions: "ca" from "can't"
                        if t.tag_ == "MD":
                            return True
                        return False
                    if verb.tag_ == "VB" and any(t.dep_ == "aux" and _is_modal(t) and t.head == verb for t in sent):
                        continue
                    # Skip aux VB when a modal also auxes the same head (e.g., "would be retiring")
                    if verb.dep_ == "aux" and verb.tag_ == "VB":
                        if any(t.dep_ == "aux" and _is_modal(t) and t.head == verb.head for t in verb.head.children):
                            continue

                    # Skip past tense (doesn't change for number) — except was/were
                    if verb.tag_ == "VBD" and verb.text.lower() not in ("was", "were"):
                        continue

                    # Skip non-finite verbs (VBN = past participle)
                    if verb.tag_ == "VBN":
                        continue

                    # Skip VBG (present participle)
                    if verb.tag_ == "VBG":
                        continue

                    # Skip bare infinitive after causative/perception verbs (let/make/have/see/hear/watch/help)
                    if verb.dep_ in ("xcomp", "ccomp"):
                        aux = verb.head
                        CAUSATIVE = {"let", "make", "made", "letting", "making", "made",
                                     "have", "has", "had", "having",
                                     "see", "saw", "seen", "seeing",
                                     "hear", "heard", "hearing",
                                     "watch", "watched", "watching",
                                     "help", "helped", "helping"}
                        if aux.text.lower() in CAUSATIVE:
                            continue

                    # Skip verbs that are same form in present and past (cost/cut/put/hit/shut...)
                    SAME_FORM_PAST = {"cost", "cut", "put", "hit", "shut", "set", "let",
                                      "hurt", "quit", "read", "spread", "bet", "cast",
                                      "fit", "forecast", "lay", "lead", "run", "quit"}
                    if verb.text.lower() in SAME_FORM_PAST and verb.tag_ == "VBP":
                        continue

                    # Determine subject number
                    subj_number = self._get_subject_number(subject, doc, text)
                    if subj_number is None:
                        continue

                    # Get expected verb form
                    expected = self._get_expected_verb(verb, subj_number, sent)
                    if expected is None:
                        continue

                    current = verb.text.lower()
                    if current == expected:
                        continue

                    replacement = expected
                    if verb.lemma_ in IRREGULAR_VERBS:
                        irr = IRREGULAR_VERBS[verb.lemma_]
                        if verb.tag_ == "VBZ" and subj_number == "plural":
                            replacement = irr.get("present_pl", verb.lemma_)
                        elif verb.tag_ in ("VBP", "VB") and subj_number == "singular":
                            replacement = irr.get("present_3sg", verb.lemma_ + "s")

                    candidates.append(ErrorCandidate(
                        start=sent_start + verb.idx,
                        end=sent_start + verb.idx + len(verb.text),
                        original=verb.text,
                        replacement=replacement,
                        category=ErrorCategory.SUBJECT_VERB_AGREEMENT,
                        rule_id="SVA",
                        message=f'The subject "{subject.text}" is {subj_number}, but the verb "{verb.text}" does not agree.',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                        metadata={"subject": subject.text, "subject_number": subj_number, "verb": verb.text},
                    ))
                    break  # Only report one error per subject
        return candidates

    def _get_subject_number(self, subject, doc, text=None) -> Optional[str]:
        # "neither...nor": verb agrees with nearest subject (the one after nor)
        if subject.text.lower() in ("neither", "either"):
            for child in subject.children:
                if child.dep_ == "conj" and child.text.lower() == "nor":
                    for child2 in child.children:
                        if child2.dep_ in ("conj", "attr") or child2.pos_ in ("NOUN", "PRON", "PROPN"):
                            return self._get_subject_number(child2, doc, text)
        for child in subject.children:
            if child.dep_ == "cc" and child.text.lower() == "nor":
                for child2 in subject.children:
                    if child2.dep_ == "conj":
                        return self._get_subject_number(child2, doc, text)

        # Compound subjects with "and" are plural: "He and I are", "The cat and the dog are"
        # Exception: fixed expressions treated as singular: "bread and butter is"
        for child in subject.children:
            if child.dep_ == "cc" and child.lower_ == "and":
                # Find the conj to build the full compound text
                conj = None
                for c in subject.children:
                    if c.dep_ == "conj":
                        conj = c
                        break
                if conj:
                    start = min(subject.idx, conj.idx)
                    end = conj.idx + len(conj.text)
                    compound_text = text[start:end].lower()
                    fixed_singular = {"bread and butter", "ham and eggs", "fish and chips",
                                      "peanut butter and jelly", "mac and cheese", "spaghetti and meatballs",
                                      "trial and error", "rock and roll", "ready and willing",
                                      "each and every", "tried and tested", "cut and paste"}
                    if compound_text in fixed_singular:
                        break  # don't return plural, let normal POS check handle it
                return "plural"

        lower = subject.lower_

        # Check uncountable nouns BEFORE spaCy POS tag (spaCy may tag them NNS)
        if lower in UNCOUNTABLE_NOUNS:
            return "singular"

        # Always-plural nouns (spaCy may tag them NN)
        always_plural = {"police", "people", "cattle", "children", "data"}
        if lower in always_plural:
            return "plural"

        # "neither" / "either" as pronoun → singular
        if lower in ("neither", "either", "each", "every", "one"):
            return "singular"

        # Compound quantities ("five dollars", "ten euros") → singular (treated as one amount)
        if subject.tag_ == "NNS" and subject.pos_ == "NOUN":
            money_words = {"dollars", "yen", "euros", "pounds", "cents", "rupees"}
            if lower in money_words:
                for child in subject.children:
                    if child.dep_ == "nummod":
                        return "singular"

        # Use tag_ (fine-grained) for PRP/NNS/NN/NNPS/CD, use pos_ (coarse) for NOUN/PRON
        if subject.tag_ in ("NNS", "NNPS"):
            return "plural"
        if subject.tag_ in ("NN", "NNP"):
            return "singular"
        if subject.tag_ == "CD":
            return "plural" if subject.text not in ("1", "one", "a", "an") else "singular"
        if subject.tag_ == "PRP":
            if subject.lower_ in ("he", "she", "it"):
                return "singular"
            if subject.lower_ in ("i", "you", "we", "they"):
                return "plural"
            return None
        if subject.pos_ in ("NOUN", "PROPN"):
            return "singular"
        if subject.pos_ == "PRON":
            if subject.lower_ in PLURAL_PRONOUNS:
                return "plural"
            if subject.lower_ in ("i", "you"):
                return "plural"
        # WP/WDT (who/which/that in relative clauses) — infer from antecedent
        # In spaCy: "who" (WP) → head=verb → head=antecedent
        if subject.tag_ in ("WP", "WDT"):
            verb = subject.head
            antecedent = verb.head if verb else None
            if antecedent and antecedent.tag_ in ("NN", "NNP", "NNPS"):
                return "singular"
            if antecedent and antecedent.tag_ in ("NNS", "NNPS"):
                # "one of the people who..." — "who" may refer to "one" (singular)
                # Walk up from antecedent to see if it's in a "one of" structure
                parent = antecedent.head
                while parent and parent.dep_ == "prep":
                    parent = parent.head
                if parent and parent.text.lower() == "one":
                    return None
                return "plural"
            return None
        return None

    def _get_expected_verb(self, verb, subj_number: str, sent) -> Optional[str]:
        lower = verb.text.lower()
        # was/were — "I was" is always correct
        if lower == "was":
            if subj_number == "plural":
                # Check if subject is "I" — "I was" is always correct
                for t in sent:
                    if t.dep_ in ("nsubj", "nsubjpass") and t.lower_ == "i" and t.head == verb:
                        return None
                    if t.dep_ in ("nsubj", "nsubjpass") and t.lower_ == "i" and verb.head != verb and t.head == verb.head:
                        return None
                return "were"
            return None
        if lower == "were":
            if subj_number == "singular":
                # "were" with singular is the subjunctive mood — only flag when the
                # sentence clearly uses an indicative "were" (no if/wish trigger).
                # e.g., "If I were", "I wish she were", "If the decision were"
                # are correct; "The meeting were" is wrong.
                for t in sent:
                    if t.dep_ in ("nsubj", "nsubjpass"):
                        prev_words = [w.lower_ for w in sent[:t.i]]
                        if any(w in ("if", "wish", "wished") for w in prev_words):
                            return None
                return "was"
            return None
        if verb.tag_ == "VBZ":
            return verb.lemma_ if subj_number == "plural" else None
        if verb.tag_ in ("VBP", "VB"):
            if subj_number == "singular" and self._is_third_person(verb, sent):
                # Check irregular verbs first
                lemma = verb.lemma_
                if lemma in IRREGULAR_VERBS:
                    irr = IRREGULAR_VERBS[lemma]
                    return irr.get("present_3sg", verb.text + "s")
                return verb.lemma_ + "s"
            return None
        return None

    def _is_third_person(self, verb, sent) -> bool:
        for child in sent:
            if child.head == verb and child.dep_ in ("nsubj", "nsubjpass"):
                if child.tag_ == "PRP" and child.lower_ in ("i", "you", "we", "they"):
                    return False
                if child.tag_ in ("NNS", "NNPS"):
                    if child.lower_ not in UNCOUNTABLE_NOUNS:
                        # Check if compound quantity (nummod child → treated as singular)
                        has_nummod = any(c.dep_ == "nummod" for c in child.children)
                        if not has_nummod:
                            return False
                return True
        if verb.head != verb:
            for child in sent:
                if child.head == verb.head and child.dep_ in ("nsubj", "nsubjpass"):
                    if child.tag_ == "PRP" and child.lower_ in ("i", "you", "we", "they"):
                        return False
                    if child.tag_ in ("NNS", "NNPS"):
                        if child.lower_ not in UNCOUNTABLE_NOUNS:
                            return False
                    return True
        return False

    def _check_tense(self, text: str, doc) -> List[ErrorCandidate]:
        """Tense consistency within sentences."""
        candidates = []
        for sent in doc.sents:
            # Check for past-time markers
            past_markers = {"yesterday", "ago", "earlier", "previously"}
            has_past_marker = False
            for t in sent:
                if t.lower_ in past_markers:
                    has_past_marker = True
                    break
                # "last" is only a past marker when followed by time word (last week/month/year)
                if t.lower_ == "last":
                    # Skip if preceded by "than" — "than last month" is a comparison, not past marker
                    if t.i > 0 and doc[t.i - 1].lower_ == "than":
                        pass
                    else:
                        time_words = {"week", "month", "year", "time", "night", "day", "season", "summer", "winter", "spring", "fall", "autumn", "century", "decade", "quarter", "hour", "minute", "moment", "period", "term", "semester", "instance", "occasion"}
                        for child in t.children:
                            if child.dep_ == "amod" and child.lower_ in time_words:
                                has_past_marker = True
                                break
                        if t.i + 1 < len(doc):
                            nxt = doc[t.i + 1]
                            if nxt.lower_ in time_words:
                                has_past_marker = True
                    if has_past_marker:
                        break
            # Check for present-time markers
            present_markers = {"now", "today", "currently", "always", "every", "usually"}
            has_present_marker = any(t.lower_ in present_markers for t in sent)

            verbs = [t for t in sent if t.pos_ in ("VERB",) and t.tag_ in ("VBD", "VBP", "VBZ", "VB")]
            if len(verbs) < 2 and not has_past_marker:
                continue

            # Check if all main verbs are same tense
            tenses = {}
            for v in verbs:
                # Skip non-finite
                if v.tag_ in ("VBG", "VBN"):
                    continue
                # Skip auxiliary
                if any(t.dep_ == "aux" and t.head == v for t in sent):
                    continue
                # Skip verbs in subordinate/relative/dependent clauses
                if v.dep_ in ("advcl", "relcl", "ccomp", "xcomp", "acl"):
                    continue
                tense = "past" if v.tag_ == "VBD" else "present"
                tenses[v] = tense

            if len(set(tenses.values())) > 1:
                if not has_present_marker:
                    # Find the present tense verb that should be past
                    for v, tense in tenses.items():
                        if tense == "present" and v.tag_ in ("VBP", "VB"):
                            if v.lemma_ in IRREGULAR_VERBS:
                                irr = IRREGULAR_VERBS[v.lemma_]
                                past = irr.get("past", v.text + "ed")
                            else:
                                past = v.text + "ed" if not v.text.endswith("e") else v.text + "d"

                            candidates.append(ErrorCandidate(
                                start=sent.start_char + v.idx,
                                end=sent.start_char + v.idx + len(v.text),
                                original=v.text,
                                replacement=past,
                                category=ErrorCategory.TENSE,
                                rule_id="TENSE_CONSISTENCY",
                                message=f'Tense inconsistency: consider using past tense "{past}" to match the context.',
                                detector="nlp_detector",
                                raw_confidence=0.82,
                                metadata={"tense": tense, "expected": "past"},
                            ))
                            break

            # Even if all verbs are same tense, check if past marker forces past
            if has_past_marker and not has_present_marker:
                for v, tense in tenses.items():
                    if tense == "present" and v.tag_ in ("VBP", "VB"):
                        if v.lemma_ in IRREGULAR_VERBS:
                            irr = IRREGULAR_VERBS[v.lemma_]
                            past = irr.get("past", v.text + "ed")
                        else:
                            past = v.text + "ed" if not v.text.endswith("e") else v.text + "d"

                        candidates.append(ErrorCandidate(
                            start=sent.start_char + v.idx,
                            end=sent.start_char + v.idx + len(v.text),
                            original=v.text,
                            replacement=past,
                            category=ErrorCategory.TENSE,
                            rule_id="TENSE_PAST_MARKER",
                            message=f'Use past tense "{past}" with time expression like "yesterday".',
                            detector="nlp_detector",
                            raw_confidence=0.85,
                            metadata={"tense": tense, "expected": "past"},
                        ))
                        break
        return candidates

    def _check_pronouns(self, text: str, doc) -> List[ErrorCandidate]:
        """Pronoun case errors."""
        candidates = []
        for token in doc:
            # Subject pronouns used as objects
            if token.dep_ in ("dobj", "pobj", "iobj") and token.lower_ in ("i", "he", "she", "we", "they"):
                replacement = {"i": "me", "he": "him", "she": "her", "we": "us", "they": "them"}.get(token.lower_)
                if replacement:
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement=replacement,
                        category=ErrorCategory.PRONOUN,
                        rule_id="PRONOUN_OBJECT_CASE",
                        message=f'Use the object form "{replacement}" after a preposition or as an object.',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                    ))
            # Object pronouns used as subjects: "Him went", "Them are", "Me forgot", "Us went"
            if token.dep_ in ("nsubj", "nsubjpass") and token.lower_ in ("me", "him", "her", "them", "us"):
                replacement = {"me": "I", "him": "he", "her": "she", "them": "they", "us": "we"}.get(token.lower_)
                if replacement:
                    # Skip "me" after perception/causative verbs: "Let me know", "It helps me"
                    # These verbs take bare infinitive with object pronoun
                    head = token.head
                    if token.lower_ == "me" and head.lower_ in (
                        "let", "lets", "letting",
                        "help", "helps", "helping",
                        "make", "makes", "making",
                        "have", "has", "having",
                        "see", "sees", "seeing",
                        "hear", "hears", "hearing",
                        "watch", "watches", "watching",
                        "notice", "notices", "noticing",
                        "feel", "feels", "feeling",
                        "love", "loves", "loving",
                        "like", "likes", "liking",
                        "want", "wants", "wanting",
                        "need", "needs", "needing",
                        "ask", "asks", "asking",
                        "tell", "tells", "telling",
                        "expect", "expects", "expecting",
                        "invite", "invites", "inviting",
                        "remind", "reminds", "reminding",
                        "encourage", "encourages", "encouraging",
                        "allow", "allows", "allowing",
                        "permit", "permits", "permitting",
                        "enable", "enables", "enabling",
                        "force", "forces", "forcing",
                        "cause", "causes", "causing",
                    ):
                        continue
                    # Skip "me" in ccomp/clausal complement — spaCy often misparses
                    # ditransitive constructions: "she makes me cookies"
                    if token.lower_ == "me" and head.dep_ in ("ccomp", "xcomp"):
                        continue
                    # Also skip if the head's head is a causative verb
                    if token.lower_ == "me" and head.head and head.head.lower_ in (
                        "make", "makes", "let", "lets", "have", "has",
                        "help", "helps", "tell", "tells", "ask", "asks",
                    ):
                        continue
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement=replacement,
                        category=ErrorCategory.PRONOUN,
                        rule_id="PRONOUN_SUBJECT_CASE",
                        message=f'Use the subject form "{replacement}" as the subject of a verb.',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                    ))
            # Compound subject "Me and Ana went" → "Ana and I went" (self last).
            # Only "me" reorders: benchmark gold keeps "Him and Ana" → "He and Ana"
            # (plain subject-case fix), while "Me and X" → "X and I".
            if token.dep_ in ("nsubj", "nsubjpass") and token.lower_ == "me":
                next_i = token.i + 1
                if next_i + 1 < len(doc) and doc[next_i].lower_ == "and":
                    name = doc[next_i + 1]
                    if name.pos_ == "PROPN" and next_i + 2 < len(doc) and doc[next_i + 2].pos_ in ("VERB", "AUX"):
                        comp_subject_map = {"me": "I", "him": "he", "her": "she",
                                            "them": "they", "us": "we"}
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=name.idx + len(name.text),
                            original=text[token.idx:name.idx + len(name.text)],
                            replacement=f"{name.text} and {comp_subject_map[token.lower_]}",
                            category=ErrorCategory.PRONOUN,
                            rule_id="PRONOUN_ORDER",
                            message=f'Place yourself last in a compound subject: "{name.text} and {comp_subject_map[token.lower_]}".',
                            detector="nlp_detector",
                            raw_confidence=0.95,
                        ))
            # "them" as determiner before noun → "those": "them new toys" → "those new toys"
            if token.lower_ == "them" and token.tag_ == "DT":
                nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if nxt and nxt.pos_ in ("NOUN", "ADJ"):
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="those",
                        category=ErrorCategory.PRONOUN,
                        rule_id="PRONOUN_DETERMINER",
                        message='Use "those" as a determiner before a noun, not "them".',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                    ))

        # "Between you and I" → "Between you and me" (object form after preposition)
        for i, token in enumerate(doc):
            if token.lower_ == "i" and i > 0 and doc[i-1].lower_ == "and":
                # Only check 1-2 tokens back — avoid finding prepositions from earlier clauses
                has_prep = False
                for j in range(i-2, max(i-3, -1), -1):
                    if j < 0:
                        break
                    if doc[j].pos_ in ("ADP", "SCONJ") or doc[j].dep_ in ("prep", "mark"):
                        has_prep = True
                        break
                if has_prep:
                        # This is "prep X and I" pattern — should be "prep X and me"
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement="me",
                            category=ErrorCategory.PRONOUN,
                            rule_id="PRONOUN_OBJECT_CASE",
                            message=f'Use the object form "me" after a preposition (e.g., "between you and me").',
                            detector="nlp_detector",
                            raw_confidence=0.90,
                        ))
                        break

        return candidates

    def _check_articles(self, text: str, doc) -> List[ErrorCandidate]:
        """a/an errors."""
        candidates = []
        vowels = set("aeiou")
        yoo_u = {"university", "uniform", "unique", "unit", "united", "universal",
                 "unified", "union", "useful", "user", "universe", "usage", "usual",
                 "usually", "utensil", "utility", "utilize", "eulogy", "euphemism",
                 "euphoria", "european", "unanimous", "unicorn"}
        an_exceptions = {"hour", "honest", "honour", "heir", "herb", "umbrella"}

        for i, token in enumerate(doc):
            if token.lower_ not in ("a", "an") or i + 1 >= len(doc):
                continue
            next_token = doc[i + 1]
            if next_token.pos_ not in ("NOUN", "PROPN", "ADJ"):
                continue

            word = next_token.text.lower()
            use_an = False
            if word in yoo_u:
                use_an = False
            elif word in an_exceptions or word[0] in vowels:
                use_an = True

            if token.lower_ == "a" and use_an:
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx + len(token.text),
                    original=token.text, replacement="an",
                    category=ErrorCategory.ARTICLE,
                    rule_id="ARTICLE_A_AN",
                    message=f'Use "an" before "{word}" (vowel sound).',
                    detector="nlp_detector",
                    raw_confidence=0.92,
                ))
            elif token.lower_ == "an" and not use_an:
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx + len(token.text),
                    original=token.text, replacement="a",
                    category=ErrorCategory.ARTICLE,
                    rule_id="ARTICLE_AN_A",
                    message=f'Use "a" before "{word}" (consonant sound).',
                    detector="nlp_detector",
                    raw_confidence=0.92,
                ))
        return candidates

    def _check_prepositions(self, text: str, doc) -> List[ErrorCandidate]:
        """Common preposition collocation errors."""
        candidates = []

        # adjective + wrong preposition patterns
        adj_prep = {
            "interested in": ("interested on", "interested in"),
            "interested on": ("interested on", "interested in"),
            "afraid of": ("afraid of", "afraid of"),
            "afraid from": ("afraid from", "afraid of"),
            "good at": ("good at", "good at"),
            "good in": ("good in", "good at"),
            "afraid of": ("afraid from", "afraid of"),
            "responsible for": ("responsible for", "responsible for"),
            "responsible to": ("responsible to", "responsible for"),
            "capable of": ("capable of", "capable of"),
            "capable for": ("capable for", "capable of"),
            "famous for": ("famous for", "famous for"),
            "famous in": ("famous in", "famous for"),
            "proud of": ("proud of", "proud of"),
            "proud about": ("proud about", "proud of"),
            "sick of": ("sick of", "sick of"),
            "sick from": ("sick from", "sick of"),
            "tired of": ("tired of", "tired of"),
            "tired from": ("tired from", "tired of"),
            "different from": ("different from", "different from"),
            "different than": ("different than", "different from"),
            "aware of": ("aware of", "aware of"),
            "aware about": ("aware about", "aware of"),
            "dependent on": ("dependent on", "dependent on"),
            "dependent from": ("dependent from", "dependent on"),
            "independent of": ("independent of", "independent of"),
            "independent from": ("independent from", "independent of"),
            "married to": ("married to", "married to"),
            "married with": ("married with", "married to"),
            "consist of": ("consist of", "consist of"),
            "consist in": ("consist in", "consist of"),
            "apologize for": ("apologize for", "apologize for"),
            "apologize about": ("apologize about", "apologize for"),
            "believe in": ("believe in", "believe in"),
            "believe on": ("believe on", "believe in"),
            "listen to": ("listen to", "listen to"),
            "listen the": ("listen the", "listen to"),
            "listen music": ("listen music", "listen to music"),
            "depend on": ("depend on", "depend on"),
            "depend of": ("depend of", "depend on"),
            "arrive at": ("arrive at", "arrive at"),
            "arrive to": ("arrive to", "arrive in"),
            "arrive in": ("arrive in", "arrive in"),
            "discuss about": ("discuss about", "discuss"),
            "discuss the": ("discuss the", "discuss the"),
            "discussed about": ("discussed about", "discussed"),
            "agree with": ("agree with", "agree with"),
            "am agree": ("am agree", "agree"),
            "is agree": ("is agree", "agree"),
            "are agree": ("are agree", "agree"),
            "working in the": ("working in the", "working on the"),
            "working in a": ("working in a", "working on a"),
            "discussed about the": ("discussed about the", "discussed the"),
            "afraid with": ("afraid with", "afraid of"),
            "afraid from": ("afraid from", "afraid of"),
            "wait from": ("wait from", "wait for"),
            "waiting from": ("waiting from", "waiting for"),
            "put in the": ("put in the", "put on the"),
            "arrived at to": ("arrived at to", "arrived at"),
        }

        # "went to home" → "went home" (home doesn't take a preposition)
        for token in doc:
            if token.text.lower() == "to" and token.dep_ == "prep":
                for child in token.children:
                    if child.dep_ == "pobj" and child.text.lower() in ("home", "here", "there", "outside", "inside", "upstairs", "downstairs"):
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=child.idx + len(child.text),
                            original=text[token.idx:child.idx + len(child.text)],
                            replacement=child.text,
                            category=ErrorCategory.PREPOSITION,
                            rule_id="UNNEEDED_PREPOSITION",
                            message=f'Do not use a preposition before "{child.text}" — it is used without a preposition after verbs of movement.',
                            detector="nlp_detector",
                            raw_confidence=0.88,
                        ))
                        break

        for token in doc:
            if token.pos_ not in ("ADJ", "VERB", "NOUN"):
                continue
            for child in token.children:
                if child.dep_ != "prep":
                    continue
                # Build the pattern using lemma for verbs
                lemma = token.lemma_ if token.pos_ == "VERB" else token.text.lower()
                pattern = f"{lemma} {child.text.lower()}"
                if pattern in adj_prep:
                    wrong, correct = adj_prep[pattern]
                    if pattern == wrong and pattern != correct:
                        # Get the full phrase including pobj
                        pobj = None
                        for c2 in child.children:
                            if c2.dep_ == "pobj":
                                pobj = c2
                                break
                        if pobj:
                            correct_prep = correct.split()[-1]
                            if correct_prep in {"in", "on", "at", "to", "for", "of", "from",
                                                "with", "about", "into", "onto", "by", "after",
                                                "before", "over", "under", "around", "through",
                                                "between", "among", "against", "during", "without"}:
                                # Simple preposition swap: replace only the preposition
                                # token so any words between it and the pobj survive.
                                candidates.append(ErrorCandidate(
                                    start=child.idx, end=child.idx + len(child.text),
                                    original=child.text, replacement=correct_prep,
                                    category=ErrorCategory.PREPOSITION,
                                    rule_id="PREPOSITION_COLLOCATION",
                                    message=f'Use "{correct_prep}" instead of "{child.text}" after "{token.text}".',
detector="nlp_detector",
                                    raw_confidence=0.88,
                                ))
                            else:
                                candidates.append(ErrorCandidate(
                                    start=child.idx, end=pobj.idx + len(pobj.text),
                                    original=text[child.idx:pobj.idx + len(pobj.text)],
                                    replacement=f"{correct_prep} {pobj.text}",
                                    category=ErrorCategory.PREPOSITION,
                                    rule_id="PREPOSITION_COLLOCATION",
                                    message=f'Use "{correct_prep}" instead of "{child.text}" after "{token.text}".',
                                    detector="nlp_detector",
                                    raw_confidence=0.88,
                                ))

        # 8. Missing comma after introductory word/phrase
        intro_words = {"therefore", "moreover", "furthermore", "nevertheless",
                       "meanwhile", "otherwise", "fortunately", "unfortunately", "suddenly",
                       "originally", "basically", "obviously", "clearly",
                       "certainly", "definitely", "probably", "perhaps",
                       "apparently", "essentially", "surprisingly",
                       "actually", "recently", "usually",
                       "normally", "typically",
                       "immediately", "occasionally",
                       "often", "sometimes",
                       "instead", "meanwhile"}
        for m in re.finditer(r'\b(' + '|'.join(re.escape(w) for w in intro_words) + r')\s+(\w)', text, re.IGNORECASE):
            word = m.group(1)
            # Only flag when word is at start of sentence (after period/start-of-text, optional whitespace)
            prefix = text[:m.start()].rstrip()
            if prefix and not prefix.endswith('.'):
                continue
            space_end = m.start() + len(word)
            rest = text[space_end:].lstrip()
            if rest.startswith(',') or rest.startswith('.'):
                continue
            candidates.append(ErrorCandidate(
                start=space_end, end=space_end + 1,
                original=" ",
                replacement=", ",
                category=ErrorCategory.PUNCTUATION,
                rule_id="MISSING_COMMA_INTRO",
                message=f'Add a comma after the introductory word "{word}".',
                detector="fast_detector",
                raw_confidence=0.82,
            ))

        # Missing comma after introductory phrase: "Yesterday I went" → "Yesterday, I went"
        # "After eating dinner she" → "After eating dinner, she"
        for m in re.finditer(r'\b(after|before|while|when|since|until|although|though|if|because|unless|whereas|whenever|wherever)\s+[\w\s]+?\s+(she|he|it|they|we|i|you)\b', text, re.IGNORECASE):
            if m.start() > 5:
                continue
            # Skip short introductory phrases (<5 words) - comma is optional
            intro_words = m.group(0).split()
            if len(intro_words) < 5:
                continue
            # Skip questions: "When did you go?" doesn't need a comma
            remaining = text[m.start():]
            if '?' in remaining[:50]:
                continue
            prefix = text[max(0, m.start()-7):m.start()].lower().strip()
            if prefix in ("not", "only", "not only"):
                continue
            end_phrase = m.end() - len(m.group(2).strip())
            before_pronoun = text[m.start():end_phrase].rstrip()
            if ',' not in before_pronoun:
                # Also check if comma follows the match: "If I were you, I would..."
                after_match = text[m.end():m.end()+5].lstrip()
                if after_match.startswith(','):
                    continue
                candidates.append(ErrorCandidate(
                    start=end_phrase - 1, end=end_phrase,
                    original=" ",
                    replacement=", ",
                    category=ErrorCategory.PUNCTUATION,
                    rule_id="MISSING_COMMA_INTRO_PHRASE",
                    message=f'Add a comma after the introductory clause.',
                    detector="fast_detector",
                    raw_confidence=0.78,
                ))

        # 9. Missing apostrophe: "dogs bone" → "dog's bone"
        for m in re.finditer(r'\b(\w+)s\s+(bone|bones|tail|food|house|home|name|collar|leash|bath|bed|toy|toys|owner|owners)\b', text, re.IGNORECASE):
            word = m.group(1)
            full_word = word.lower() + "s"
            if full_word in ("the", "his", "her", "my", "your", "its", "our", "their", "this", "that", "these", "those", "what", "which", "who", "whom", "his", "hers"):
                continue
            if word.lower() in ("the", "his", "her", "my", "your", "its", "our", "their", "this", "that", "these", "those"):
                continue
            if not word.endswith("'") and not word.endswith("s'"):
                if len(word) > 2 and word.lower().endswith(("es", "ss", "ch", "sh")):
                    continue
                candidates.append(ErrorCandidate(
                    start=m.start(1), end=m.end(1) + 1,
                    original=word + "s",
                    replacement=word + "'s",
                    category=ErrorCategory.PUNCTUATION,
                    rule_id="MISSING_APOSTROPHE",
                    message='Use an apostrophe to show possession: "' + word + "'s\".",
                    detector="fast_detector",
                    raw_confidence=0.80,
                ))

        # Missing comma before "so" connecting independent clauses
        for m in re.finditer(r'(\w)\s+so\s+(?=[A-Z])', text):
            before_so = text[max(0, m.start()-20):m.start()+1].strip()
            if before_so and not before_so.endswith((',', ';', ':')):
                after_so = text[m.end():m.end()+10]
                if after_so and after_so[0].isupper():
                    candidates.append(ErrorCandidate(
                        start=m.start() + 1, end=m.start() + 2,
                        original=" ",
                        replacement=", ",
                        category=ErrorCategory.PUNCTUATION,
                        rule_id="MISSING_COMMA_BEFORE_SO",
                        message='Add a comma before "so" when it connects two independent clauses.',
                        detector="fast_detector",
                        raw_confidence=0.82,
                    ))

        return candidates
    def _check_existential_there(self, text: str, doc) -> List[ErrorCandidate]:
        """There is/are agreement with real subject."""
        candidates = []
        for token in doc:
            if token.lower_ != "there" or token.dep_ != "expl":
                continue
            verb = token.head
            if verb.pos_ != "VERB":
                continue

            real_subject = None
            for child in verb.children:
                if child.dep_ == "attr":
                    real_subject = child
                    break
            if real_subject is None:
                continue

            # Handle "lot of X", "number of X"
            if real_subject.lower_ in ("lot", "number", "plenty", "lots", "couple", "series", "range"):
                for child in real_subject.children:
                    if child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                real_subject = grandchild
                                break

            is_plural = self._is_existential_plural(real_subject)
            if is_plural is None:
                continue

            be_singular = {"is", "was", "has"}
            be_plural = {"are", "were", "have"}

            if is_plural and verb.text.lower() in be_singular:
                mapping = {"is": "are", "was": "were", "has": "have"}
                replacement = mapping.get(verb.text.lower())
                if replacement:
                    candidates.append(ErrorCandidate(
                        start=verb.idx, end=verb.idx + len(verb.text),
                        original=verb.text, replacement=replacement,
                        category=ErrorCategory.SUBJECT_VERB_AGREEMENT,
                        rule_id="EXISTENTIAL_THERE_SVA",
                        message=f'With the plural subject "{real_subject.text}", use "{replacement}".',
                        detector="nlp_detector",
                        raw_confidence=0.92,
                        metadata={"existential_there": True, "subject": real_subject.text},
                    ))
            elif not is_plural and verb.text.lower() in be_plural:
                mapping = {"are": "is", "were": "was", "have": "has"}
                replacement = mapping.get(verb.text.lower())
                if replacement:
                    candidates.append(ErrorCandidate(
                        start=verb.idx, end=verb.idx + len(verb.text),
                        original=verb.text, replacement=replacement,
                        category=ErrorCategory.SUBJECT_VERB_AGREEMENT,
                        rule_id="EXISTENTIAL_THERE_SVA",
                        message=f'With the singular subject "{real_subject.text}", use "{replacement}".',
                        detector="nlp_detector",
                        raw_confidence=0.92,
                        metadata={"existential_there": True, "subject": real_subject.text},
                    ))
        return candidates

    def _is_existential_plural(self, token) -> Optional[bool]:
        if token.lower_ in UNCOUNTABLE_NOUNS:
            return False
        PLURAL_QUANTIFIERS = {"many", "several", "few", "numerous", "countless",
                              "plenty", "lots", "tons", "dozens", "hundreds",
                              "thousands", "millions", "billions", "various",
                              "multiple", "numerous", "some", "these", "those"}
        if token.lower_ in PLURAL_QUANTIFIERS:
            return True
        for prep in token.children:
            if prep.dep_ == "prep" and prep.lower_ == "of":
                for pobj in prep.children:
                    if pobj.dep_ == "pobj" and pobj.lower_ in PLURAL_QUANTIFIERS:
                        return True
        if token.tag_ in ("NNS", "NNPS"):
            return True
        if token.tag_ in ("NN", "NNP"):
            if token.text.lower() in ("fish", "sheep", "deer", "moose", "elk", "bison", "buffalo", "salmon", "trout", "carp", "shrimp"):
                return True
            return False
        if token.tag_ == "CD":
            return token.text not in ("1", "one", "a", "an")
        if token.lower_ in PLURAL_PRONOUNS:
            return True
        if token.lower_ in ("i", "you"):
            return True
        if token.lower_ in ("people", "students", "children", "men", "women"):
            return True
        return None

    def _check_compound_subject_pronouns(self, text: str, doc) -> List[ErrorCandidate]:
        """Object pronouns in compound subjects: 'friend and me' → 'friend and I'."""
        candidates = []
        for token in doc:
            if token.lower_ not in OBJECT_TO_SUBJECT:
                continue
            # Pattern: "X and me" where X is noun/pronoun
            if token.dep_ == "conj" and token.i > 0:
                prev = doc[token.i - 1]
                if prev.text.lower() in ("and", "or") and token.i >= 2:
                    two_back = doc[token.i - 2]
                    if two_back.dep_ in ("nsubj", "nsubjpass") or two_back.pos_ in ("NOUN", "PROPN"):
                        replacement = OBJECT_TO_SUBJECT[token.lower_]
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement=replacement,
                            category=ErrorCategory.PRONOUN,
                            rule_id="COMPOUND_SUBJECT_PRONOUN",
                            message=f'In a compound subject, use "{replacement}" instead of "{token.text}".',
                            detector="nlp_detector",
                            raw_confidence=0.90,
                        ))
            # Pattern: "me and X" where me is nsubj
            elif token.dep_ in ("nsubj", "nsubjpass") and token.i > 0:
                prev = doc[token.i - 1]
                if prev.text.lower() in ("and", "or") and token.i >= 2:
                    two_back = doc[token.i - 2]
                    if two_back.pos_ in ("NOUN", "PROPN", "PRON"):
                        replacement = OBJECT_TO_SUBJECT[token.lower_]
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement=replacement,
                            category=ErrorCategory.PRONOUN,
                            rule_id="COMPOUND_SUBJECT_PRONOUN_2",
                            message=f'In a compound subject, use "{replacement}" instead of "{token.text}".',
                            detector="nlp_detector",
                            raw_confidence=0.90,
                        ))
        return candidates

    def _check_possessives(self, text: str, doc) -> List[ErrorCandidate]:
        """its/it's, your/you're, their/they're, who's/whose."""
        candidates = []
        for token in doc:
            lower = token.text.lower()

            # it's → its (possessive)
            if lower == "it's":
                # Check if it's possessive (no verb after)
                next_tokens = list(doc[token.i + 1:token.i + 3])
                if next_tokens and next_tokens[0].pos_ in ("NOUN", "ADJ"):
                    # "it's name" should be "its name"
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="its",
                        category=ErrorCategory.PRONOUN,
                        rule_id="POSSESSIVE_ITS",
                        message='Use "its" (possessive) instead of "it\'s" (contraction of "it is").',
                        detector="nlp_detector",
                        raw_confidence=0.88,
                    ))

            # their → there/they're
            if lower == "their":
                next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if next_token and next_token.text.lower() in BE_FORMS:
                    # "their going" → "they're going"
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="they're",
                        category=ErrorCategory.WORD_USAGE,
                        rule_id="THEIR_THEYRE",
                        message='Use "they\'re" (contraction of "they are") instead of "their".',
                        detector="nlp_detector",
                        raw_confidence=0.85,
                    ))

            # who's → whose (possessive)
            if lower == "who's":
                next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if next_token and next_token.pos_ in ("NOUN",):
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="whose",
                        category=ErrorCategory.PRONOUN,
                        rule_id="WHOSE_WHO",
                        message='Use "whose" (possessive) instead of "who\'s" (contraction of "who is").',
                        detector="nlp_detector",
                        raw_confidence=0.85,
                    ))
        return candidates

    def _check_word_usage(self, text: str, doc) -> List[ErrorCandidate]:
        """Contextual word usage errors."""
        candidates = []
        lower_text = text.lower()

        # "could of" → "could have"
        for wrong, correct in [("could of", "could have"), ("would of", "would have"),
                               ("should of", "should have"), ("might of", "might have"),
                               ("must of", "must have"), ("may of", "may have")]:
            idx = lower_text.find(wrong)
            while idx != -1:
                end = idx + len(wrong)
                before_ok = idx == 0 or not text[idx - 1].isalpha()
                after_ok = end >= len(text) or not text[end].isalpha()
                if before_ok and after_ok:
                    candidates.append(ErrorCandidate(
                        start=idx, end=end,
                        original=text[idx:end], replacement=correct,
                        category=ErrorCategory.WORD_USAGE,
                        rule_id=f"WOULD_OF_{wrong.split()[0].upper()}",
                        message=f'Use "{correct}" instead of "{wrong}".',
                        detector="nlp_detector",
                        raw_confidence=0.95,
                    ))
                idx = lower_text.find(wrong, end)

        # "affect" vs "effect" (basic context)
        for m in re.finditer(r'\b(affect|effect)\b', text, re.IGNORECASE):
            word = m.group(1).lower()
            before = text[max(0, m.start() - 10):m.start()].lower()
            if word == "effect" and ("a " in before or "an " in before):
                pass
            elif word == "affect" and ("have " in before or "has " in before or "had " in before):
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(1), replacement="effect",
                    category=ErrorCategory.WORD_USAGE,
                    rule_id="AFFECT_EFFECT",
                    message='Use "effect" (noun) after "have/has/had".',
                    detector="nlp_detector",
                    raw_confidence=0.80,
                ))

        # "could care less" → "couldn't care less"
        for wrong, correct in [("could care less", "couldn't care less"),
                               ("can care less", "can't care less")]:
            idx = lower_text.find(wrong)
            if idx != -1:
                end = idx + len(wrong)
                before_ok = idx == 0 or not text[idx - 1].isalpha()
                after_ok = end >= len(text) or not text[end].isalpha()
                if before_ok and after_ok:
                    candidates.append(ErrorCandidate(
                        start=idx, end=end,
                        original=text[idx:end], replacement=correct,
                        category=ErrorCategory.WORD_USAGE,
                        rule_id="COULD_CARE_LESS",
                        message=f'The correct expression is "{correct}".',
                        detector="nlp_detector",
                        raw_confidence=0.92,
                    ))

        # "use to" (past) → "used to": "I use to walk" → "I used to walk"
        for m in re.finditer(r'\b(use to)\b', text, re.IGNORECASE):
            before = text[max(0, m.start() - 20):m.start()].lower()
            # Only flag in past tense contexts: "used to", "would use to", or after past markers
            if any(marker in before for marker in ["used ", "would", "did", "didn't", "was", "were", "had"]):
                continue
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(1), replacement="used to",
                category=ErrorCategory.WORD_USAGE,
                rule_id="USE_TO",
                message='Use "used to" (with -d) for past habits or states.',
                detector="nlp_detector",
                raw_confidence=0.88,
            ))

        # "was suppose to" → "was supposed to"
        for m in re.finditer(r'\b(suppose to)\b', text, re.IGNORECASE):
            before = text[max(0, m.start() - 15):m.start()].lower()
            if any(marker in before for marker in ["was", "were", "is", "are", "been"]):
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(1), replacement="supposed to",
                    category=ErrorCategory.WORD_USAGE,
                    rule_id="SUPPOSE_TO",
                    message='Use "supposed to" (with -ed) after a form of "be".',
                    detector="nlp_detector",
                    raw_confidence=0.90,
                ))

        # "learnt" → "learned" (US English preference)
        for m in re.finditer(r'\b(learnt|dreamt|leapt)\b', text, re.IGNORECASE):
            word = m.group(1).lower()
            correct = {"learnt": "learned", "dreamt": "dreamed", "leapt": "leaped"}[word]
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(1), replacement=correct,
                category=ErrorCategory.WORD_USAGE,
                rule_id="US_ENGLISH_FORM",
                message=f'In US English, use "{correct}" instead of "{word}".',
                detector="nlp_detector",
                raw_confidence=0.75,
            ))

        # "seen" without auxiliary → "saw": "I seen him" → "I saw him"
        for token in doc:
            if token.text.lower() == "seen" and token.tag_ in ("VBN", "VBD"):
                # Check if it has an auxiliary (have/has/had) — direct child or contracted
                has_aux = False
                for t in token.children:
                    if t.dep_ in ("aux", "auxpass") and t.text.lower() in HAVE_FORMS:
                        has_aux = True
                        break
                # Also check for contracted forms: "I've", "We've", "They've"
                if token.i > 0:
                    prev = doc[token.i - 1]
                    if prev.text.lower().endswith("ve") or prev.text.lower() in ("have", "has", "had"):
                        has_aux = True
                if not has_aux:
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="saw",
                        category=ErrorCategory.WORD_USAGE,
                        rule_id="SEEN_SAW",
                        message='Use "saw" (simple past) without an auxiliary verb. "Seen" requires "have/has/had".',
                        detector="nlp_detector",
                        raw_confidence=0.88,
                    ))

        # "done" without auxiliary → "did": "I done it" → "I did it"
        for token in doc:
            if token.text.lower() == "done" and token.tag_ in ("VBN", "VBD"):
                has_aux = False
                for t in token.children:
                    if t.dep_ in ("aux", "auxpass") and t.text.lower() in HAVE_FORMS:
                        has_aux = True
                        break
                # Also check for "be" auxiliary (AAVE: "She be done")
                for t in token.children:
                    if t.dep_ in ("aux", "auxpass") and t.text.lower() in BE_FORMS:
                        has_aux = True
                        break
                if has_aux:
                    continue
                # Skip "done" used as adjective/complement: "well done", "done", "I'm done"
                # Check if "done" is a root or complement (acomp/xcomp/attr)
                if token.dep_ in ("acomp", "xcomp", "attr", "ROOT"):
                    continue
                # Check if "done" follows "well" or "all" (collocations)
                prev_token = token.nbor(-1) if token.i > 0 else None
                if prev_token and prev_token.lower_ in ("well", "all"):
                    continue
                # Check if "done" is a response/standalone
                if token.dep_ == "ROOT" and len(list(doc)) <= 5:
                    continue
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="did",
                        category=ErrorCategory.WORD_USAGE,
                        rule_id="DONE_DID",
                        message='Use "did" (simple past) without an auxiliary verb. "Done" requires "have/has/had".',
                        detector="nlp_detector",
                        raw_confidence=0.85,
                    ))

        return candidates


    def _check_past_participle(self, text: str, doc) -> List[ErrorCandidate]:
        """Wrong past participle after have/has/had: 'have went' → 'have gone'.
        Also: 'had to waited' → 'had to wait' (to + VBN should be to + VB)."""
        candidates = []
        for sent in doc.sents:
            for token in sent:
                if token.text.lower() not in HAVE_FORMS:
                    continue
                if token.dep_ != "aux":
                    continue
                main_verb = token.head
                if main_verb.tag_ not in ("VBD", "VBN", "VB"):
                    continue
                # Check if the verb's past form matches but pp is different
                lemma = main_verb.lemma_
                if lemma in IRREGULAR_VERBS:
                    irr = IRREGULAR_VERBS[lemma]
                    pp = irr.get("pp")
                    past = irr.get("past")
                    if pp and past and main_verb.text.lower() == past and pp != past:
                        candidates.append(ErrorCandidate(
                            start=sent.start_char + main_verb.idx,
                            end=sent.start_char + main_verb.idx + len(main_verb.text),
                            original=main_verb.text,
                            replacement=pp,
                            category=ErrorCategory.TENSE,
                            rule_id="WRONG_PAST_PARTICIPLE",
                            message=f'After "{token.text}", use the past participle "{pp}" instead of "{main_verb.text}".',
                            detector="nlp_detector",
                            raw_confidence=0.92,
                        ))
                    elif pp and main_verb.tag_ == "VB" and main_verb.text.lower() != pp:
                        if main_verb.text.lower() in ("forget", "get", "see", "know",
                                                       "think", "say", "take", "give",
                                                       "come", "go", "do", "make", "find"):
                            candidates.append(ErrorCandidate(
                                start=sent.start_char + main_verb.idx,
                                end=sent.start_char + main_verb.idx + len(main_verb.text),
                                original=main_verb.text,
                                replacement=pp,
                                category=ErrorCategory.TENSE,
                                rule_id="WRONG_PAST_PARTICIPLE",
                                message=f'After "{token.text}", use the past participle "{pp}" instead of "{main_verb.text}".',
                                detector="nlp_detector",
                                raw_confidence=0.88,
                            ))
                else:
                    # Regular verb: "have finish" → "have finished" (VBN = lemma means wrong form)
                    if main_verb.tag_ == "VBN" and main_verb.text.lower() == lemma:
                        correct_pp = lemma + "ed" if not lemma.endswith("e") else lemma + "d"
                        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
                            correct_pp = lemma[:-1] + "ied"
                        if main_verb.text.lower() != correct_pp:
                            candidates.append(ErrorCandidate(
                                start=sent.start_char + main_verb.idx,
                                end=sent.start_char + main_verb.idx + len(main_verb.text),
                                original=main_verb.text,
                                replacement=correct_pp,
                                category=ErrorCategory.TENSE,
                                rule_id="WRONG_PAST_PARTICIPLE",
                                message=f'After "{token.text}", use the past participle "{correct_pp}" instead of "{main_verb.text}".',
                                detector="nlp_detector",
                                raw_confidence=0.88,
                            ))
        # "to + VBN" → "to + VB" (e.g., "had to waited" → "had to wait")
        # Skip "to have been" / "to have gone" — perfect infinitives are correct
        # Skip "to be finished" / "to be seen" — passive infinitives are correct
        for token in doc:
            if token.text.lower() == "to" and token.tag_ == "TO" and token.dep_ == "aux":
                main_verb = token.head
                if main_verb.tag_ == "VBN" and main_verb.dep_ == "xcomp":
                    has_have = any(t.text.lower() in HAVE_FORMS and t.dep_ in ("aux", "auxpass")
                                   for t in main_verb.children)
                    if has_have:
                        continue
                    # Skip passive infinitive: "to be VBN"
                    has_be = any(t.text.lower() in BE_FORMS and t.dep_ in ("aux", "auxpass")
                                 for t in main_verb.children)
                    if has_be:
                        continue
                    base = main_verb.lemma_
                    candidates.append(ErrorCandidate(
                        start=main_verb.idx, end=main_verb.idx + len(main_verb.text),
                        original=main_verb.text, replacement=base,
                        category=ErrorCategory.TENSE,
                        rule_id="TO_PAST_PARTICIPLE",
                        message=f'After "to", use the base form "{base}" instead of "{main_verb.text}".',
                        detector="nlp_detector", raw_confidence=0.92,
                    ))
                elif main_verb.tag_ == "VBZ" and main_verb.dep_ in ("xcomp", "acl"):
                    base = main_verb.lemma_
                    candidates.append(ErrorCandidate(
                        start=main_verb.idx, end=main_verb.idx + len(main_verb.text),
                        original=main_verb.text, replacement=base,
                        category=ErrorCategory.GRAMMAR,
                        rule_id="INFINITIVE_FORM",
                        message=f'After "to", use the base form "{base}" instead of "{main_verb.text}".',
                        detector="nlp_detector", raw_confidence=0.92,
                    ))

        # "had forget" → "had forgotten": "had" as ROOT with VBN that should be pp
        for token in doc:
            if token.text.lower() in ("had", "has", "have") and token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ in ("xcomp", "ccomp", "conj") and child.tag_ in ("VBN", "VBD"):
                        lemma = child.lemma_
                        if lemma in IRREGULAR_VERBS:
                            irr = IRREGULAR_VERBS[lemma]
                            pp = irr.get("pp")
                            past = irr.get("past")
                            if pp and child.text.lower() == past and pp != past:
                                candidates.append(ErrorCandidate(
                                    start=child.idx, end=child.idx + len(child.text),
                                    original=child.text, replacement=pp,
                                    category=ErrorCategory.TENSE,
                                    rule_id="WRONG_PAST_PARTICIPLE",
                                    message=f'After "{token.text}", use the past participle "{pp}" instead of "{child.text}".',
                                    detector="nlp_detector", raw_confidence=0.92,
                ))

        # === BATCH FIX: Missing detectors for common L2 errors ===

        # 1. Double negative: "didn't see nobody" → "didn't see anybody"
        neg_words = {"n't", "not", "no", "never", "neither", "nor", "nowhere", "hardly", "barely", "scarcely"}
        neg_indefinite_map = {
            "nobody": "anybody", "nothing": "anything", "nowhere": "anywhere",
            "neither": "either", "none": "any", "no one": "anyone",
        }
        for token in doc:
            if token.lower_ in ("nobody", "nothing", "nowhere", "neither", "none", "no one"):
                # Check if there's a negation in the sentence before this word
                for t in doc:
                    if t.i >= token.i:
                        break
                    if t.lower_ in neg_words or (t.dep_ == "neg"):
                        replacement = neg_indefinite_map.get(token.lower_, "any" + token.lower_[2:])
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement=replacement,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="DOUBLE_NEGATIVE",
                            message=f'Double negative: use "{replacement}" with a negative verb.',
                            detector="nlp_detector", raw_confidence=0.92,
                        ))
                        break

        # 2. "waiting your response" → "waiting for your response"
        WAIT_FOR = {"waiting", "waited", "wait"}
        for token in doc:
            if token.lower_ in WAIT_FOR:
                # Check if next meaningful word is a dobj without preposition
                for child in token.children:
                    if child.dep_ == "dobj":
                        candidates.append(ErrorCandidate(
                            start=token.idx + len(token.text), end=token.idx + len(token.text),
                            original=" ", replacement=" for ",
                            category=ErrorCategory.PREPOSITION,
                            rule_id="MISSING_PREPOSITION",
                            message=f'Use "waiting for" + object (e.g., "waiting for your response").',
                            detector="nlp_detector", raw_confidence=0.85,
                        ))
                        break

        # 3. "enjoyed to watch" → "enjoyed watching" (gerund after enjoy/look forward to/etc.)
        GERUND_VERBS = {"enjoy", "enjoyed", "enjoys", "enjoying",
                        "finish", "finished", "finishes", "finishing",
                        "mind", "minded", "minds", "minding",
                        "avoid", "avoided", "avoids", "avoiding",
                        "consider", "considered", "considers", "considering",
                        "suggest", "suggested", "suggests", "suggesting",
                        "practice", "practiced", "practices", "practicing",
                        "keep", "kept", "keeps", "keeping",
                        "quit", "quits", "quitting",
                        "risk", "risks", "risking"}
        for token in doc:
            if token.lower_ in GERUND_VERBS:
                for child in token.children:
                    if child.dep_ == "xcomp" and child.tag_ == "VB":
                        to_children = [t for t in child.children if t.dep_ == "aux" and t.tag_ == "TO"]
                        if to_children:
                            gerund = _gerundize(child.text)
                            candidates.append(ErrorCandidate(
                                start=to_children[0].idx, end=child.idx + len(child.text),
                                original=f"to {child.text}", replacement=gerund,
                                category=ErrorCategory.GRAMMAR,
                                rule_id="GERUND_INFINITIVE",
                                message=f'After "{token.text}", use the gerund form "{gerund}" (e.g., "{token.text} {gerund}").',
                                detector="nlp_detector", raw_confidence=0.88,
                            ))

        # 4. "What I want is improving" → "What I want is to improve" (infinitive after "is")
        what_want_pat = re.compile(r'^(What\s+(I|you|he|she|it|we|they)\s+want(s)?\s+is)\s+(\w+)', re.IGNORECASE)
        for sent in doc.sents:
            m = what_want_pat.match(sent.text)
            if m and m.group(4).endswith("ing"):
                gerund = m.group(4)
                base = gerund[:-3] if gerund.endswith("ing") and len(gerund) > 4 else gerund
                candidates.append(ErrorCandidate(
                    start=sent.start_char + m.start(4), end=sent.start_char + m.end(4),
                    original=gerund, replacement=f"to {base}",
                    category=ErrorCategory.GRAMMAR,
                    rule_id="GERUND_INFINITIVE",
                    message=f'After "What ... is", use the infinitive "to {base}" instead of "{gerund}".',
                    detector="nlp_detector", raw_confidence=0.82,
                ))

        # 5. Passive wrong PP: "has been send" → "has been sent"
        IRREGULAR_PP = {
            "send": "sent", "see": "seen", "take": "taken", "give": "given",
            "write": "written", "speak": "spoken", "break": "broken",
            "choose": "chosen", "drive": "driven", "eat": "eaten",
            "fall": "fallen", "fly": "flown", "forget": "forgotten",
            "freeze": "frozen", "grow": "grown", "know": "known",
            "ride": "ridden", "ring": "rung", "rise": "risen",
            "show": "shown", "swim": "swum", "throw": "thrown",
            "wear": "worn", "bear": "borne/born", "tear": "torn",
            "wake": "woken", "hide": "hidden", "lie": "lain",
            "shake": "shaken", "stand": "stood", "understand": "understood",
            "hold": "held", "sit": "sat", "win": "won",
            "become": "become", "come": "come", "run": "run",
            "begin": "begun", "drink": "drunk", "sing": "sung",
            "sink": "sunk", "swim": "swum", "blow": "blown",
            "draw": "drawn", "grow": "grown", "know": "known",
            "throw": "thrown", "catch": "caught", "teach": "taught",
            "bring": "brought", "buy": "bought", "think": "thought",
            "seek": "sought", "feel": "felt", "keep": "kept",
            "sleep": "slept", "leave": "left", "mean": "meant",
            "meet": "met", "pay": "paid", "say": "said",
            "tell": "told", "sell": "sold", "build": "built",
            "bend": "bent", "lend": "lent", "send": "sent",
            "spend": "spent", "lose": "lost", "lead": "led",
            "read": "read", "put": "put", "cut": "cut",
            "hit": "hit", "let": "let", "set": "set",
            "shut": "shut", "cost": "cost", "hurt": "hurt",
        }
        for token in doc:
            if token.dep_ == "ROOT" and token.tag_ == "VBN":
                # Check if this is passive: has/had been + VBN
                has_been = any(t.dep_ == "auxpass" and t.lower_ in ("been", "be", "being") for t in token.children)
                has_aux = any(t.dep_ in ("aux", "auxpass") for t in token.children)
                has_do_aux = any(t.dep_ == "aux" and t.lower_ in DO_FORMS for t in token.children)
                if has_do_aux:
                    # "He did not sang." → the correction belongs to DIDNT_PAST_FORM
                    # (did + base "sing"), not the passive/perfect past-participle check.
                    continue
                if has_been or (has_aux and token.dep_ == "ROOT"):
                    lemma = token.lemma_.lower()
                    if lemma in IRREGULAR_PP:
                        expected = IRREGULAR_PP[lemma]
                        if token.lower_ != expected:
                            candidates.append(ErrorCandidate(
                                start=token.idx, end=token.idx + len(token.text),
                                original=token.text, replacement=expected,
                                category=ErrorCategory.VERB_FORM,
                                rule_id="WRONG_PP",
                                message=f'The past participle of "{lemma}" is "{expected}", not "{token.text}".',
                                detector="nlp_detector", raw_confidence=0.92,
                            ))

        # 6. Adverb form after verb: "speaks English very good" → "well"
        # spaCy may parse "good" as advcl (not advmod) — check both
        _ADJ_TO_ADV_LOCAL = {
            "good": "well", "bad": "badly", "quick": "quickly", "slow": "slowly",
            "fast": "fast", "hard": "hard", "loud": "loudly", "quiet": "quietly",
            "beautiful": "beautifully", "careful": "carefully", "successful": "successfully",
            "angry": "angrily", "perfect": "perfectly", "gentle": "gently",
            "brave": "bravely", "clear": "clearly", "easy": "easily",
            "happy": "happily", "sad": "sadly", "deep": "deeply",
            "near": "nearly", "high": "highly", "straight": "straight",
        }
        for token in doc:
            if token.lower_ in _ADJ_TO_ADV_LOCAL and token.pos_ == "ADJ":
                head = token.head
                # Check if adjective is used as adverb (advmod, acomp, advcl, dobj-misparse)
                if token.dep_ in ("advmod", "acomp", "advcl") and head.pos_ in ("VERB", "AUX"):
                    LINKING_VERBS_SET = {"be", "is", "are", "was", "were", "am", "been", "being",
                                         "seem", "seems", "seemed", "appear", "appears",
                                         "become", "becomes", "became",
                                         "feel", "feels", "felt", "look", "looks", "looked"}
                    if head.lower_ not in LINKING_VERBS_SET:
                        replacement = _ADJ_TO_ADV_LOCAL[token.lower_]
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement=replacement,
                            category=ErrorCategory.ADVERB,
                            rule_id="ADVERB_FORM",
                            message=f'Use the adverb form "{replacement}" instead of "{token.text}".',
                            detector="nlp_detector", raw_confidence=0.87,
                        ))
            # Also check dobj-misparse: "completed the task successful" — "successful" as dobj
            if token.lower_ in _ADJ_TO_ADV_LOCAL and token.pos_ == "ADJ" and token.dep_ == "dobj":
                replacement = _ADJ_TO_ADV_LOCAL[token.lower_]
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx + len(token.text),
                    original=token.text, replacement=replacement,
                    category=ErrorCategory.ADVERB,
                    rule_id="ADVERB_FORM",
                    message=f'Use the adverb form "{replacement}" instead of "{token.text}".',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 7. Missing comma before relative clause: "My friend, who lives in Chennai is"
        for token in doc:
            if token.dep_ == "relcl" and token.pos_ == "VERB":
                subj_token = None
                for child in token.children:
                    if child.dep_ == "nsubj" and child.pos_ == "PRON":
                        subj_token = child
                        break
                if subj_token and subj_token.lower_ in ("who", "which", "that"):
                    token_before = doc[subj_token.i - 1] if subj_token.i > 0 else None
                    if token_before and token_before.text != ",":
                        # Only flag non-restrictive clauses:
                        # 1. Antecedent is preceded by a possessive/my/his/her or "the"
                        # 2. OR the clause itself has internal commas (list)
                        antecedent = token.head
                        has_internal_comma = any(t.text == "," for t in token.children)
                        is_non_restrictive = False
                        if antecedent.i > 0:
                            word_before = doc[antecedent.i - 1]
                            if word_before.lower_ in ("my", "his", "her", "its", "our", "your", "their"):
                                is_non_restrictive = True
                        # Check: comma AFTER the relative clause subtree (not inside it)
                        rel_tokens = sorted(token.subtree, key=lambda t: t.i)
                        if rel_tokens:
                            last_rel_token = rel_tokens[-1]
                            if last_rel_token.i + 1 < len(doc):
                                token_after_clause = doc[last_rel_token.i + 1]
                                if token_after_clause.text == ",":
                                    # Make sure it's not part of a list (comma followed by adj/noun modifying same antecedent)
                                    if last_rel_token.i + 2 < len(doc):
                                        next_after_comma = doc[last_rel_token.i + 2]
                                        if next_after_comma.pos_ not in ("ADJ", "NOUN", "PROPN") or next_after_comma.head != antecedent:
                                            is_non_restrictive = True
                        if is_non_restrictive or has_internal_comma:
                            candidates.append(ErrorCandidate(
                                start=token_before.idx + len(token_before.text) if token_before else token.idx,
                                end=token_before.idx + len(token_before.text) + 1 if token_before else token.idx + 1,
                                original=" ", replacement=", ",
                                category=ErrorCategory.PUNCTUATION,
                                rule_id="MISSING_COMMA",
                                message='Add a comma before the relative clause (or after the clause if non-restrictive).',
                                detector="nlp_detector", raw_confidence=0.78,
                            ))

        # 8. Parallelism: "cooking, dancing and to sing" → "cooking, dancing, and singing"
        for token in doc:
            if token.dep_ == "conj" and token.tag_ == "VB":
                # Find the list head
                list_head = token.head
                to_children = [t for t in token.children if t.dep_ == "aux" and t.tag_ == "TO"]
                if to_children and list_head.pos_ in ("NOUN", "VERB"):
                    # Check if head is a gerund (NN from VBG) or verb
                    head_is_gerund = list_head.tag_ == "NN" and list_head.lemma_ != list_head.text
                    # Or check siblings for gerunds
                    siblings = [t for t in list_head.children if t.dep_ == "conj"]
                    if head_is_gerund or any(s.tag_ in ("NN", "VBG") for s in siblings):
                        gerund = token.lemma_ + "ing"
                        candidates.append(ErrorCandidate(
                            start=to_children[0].idx, end=token.idx + len(token.text),
                            original=f"to {token.text}", replacement=gerund,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="PARALLELISM",
                            message=f'Maintain parallel structure: use "{gerund}" to match the other list items.',
                            detector="nlp_detector", raw_confidence=0.82,
                        ))

        # === NEW BATCH: Additional L2 error detectors ===

        # 9. Missing "to" after "going": "They are going visit" → "They are going to visit"
        for token in doc:
            if token.lower_ == "going" and token.pos_ in ("VERB", "AUX"):
                for child in token.children:
                    if child.dep_ == "xcomp" and child.tag_ == "VB":
                        to_exists = any(t.dep_ == "aux" and t.tag_ == "TO" for t in child.children)
                        if not to_exists:
                            candidates.append(ErrorCandidate(
                                start=token.idx + len(token.text), end=child.idx,
                                original=" " + child.text, replacement=" to " + child.text,
                                category=ErrorCategory.GRAMMAR,
                                rule_id="MISSING_TO",
                                message=f'Use "going to {child.text}" (missing "to").',
                                detector="nlp_detector", raw_confidence=0.88,
                            ))

        # 10. Unnecessary preposition "in" before "next week": "in next week" → "next week"
        # Only flag time expressions, not "in this house" (which is correct)
        UNNECESSARY_PREP = {
            ("in", "next"), ("in", "last"),
            ("on", "next"), ("on", "last"),
        }
        for token in doc:
            if token.pos_ == "ADP":
                nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if nxt and (token.lower_, nxt.lower_) in UNNECESSARY_PREP:
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text) + 1,
                        original=token.text + " ", replacement="",
                        category=ErrorCategory.PREPOSITION,
                        rule_id="UNNECESSARY_PREPOSITION",
                        message=f'Remove the unnecessary preposition "{token.text}" before "{nxt.text}".',
                        detector="nlp_detector", raw_confidence=0.85,
                    ))

        # 11. Missing article: "to office" → "to the office"
        for token in doc:
            if token.pos_ == "ADP" and token.dep_ == "prep":
                nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if nxt and nxt.lower_ == "office" and nxt.pos_ == "NOUN":
                    candidates.append(ErrorCandidate(
                        start=nxt.idx, end=nxt.idx + len(nxt.text),
                        original=nxt.text, replacement="the office",
                        category=ErrorCategory.ARTICLE,
                        rule_id="ARTICLE_MISSING",
                        message=f'Use "the office" (missing article "the").',
                        detector="nlp_detector", raw_confidence=0.80,
                    ))

        # 12. Missing plural: "two sister" → "two sisters" (number + singular noun)
        for token in doc:
            if token.pos_ == "NOUN" and token.tag_ == "NN":
                for child in token.children:
                    if child.dep_ == "nummod" and child.pos_ == "NUM":
                        # "one" takes singular — "one thing" is correct
                        if child.lower_ == "one":
                            continue
                        word = token.text.lower()
                        if not word.endswith("s") and token.lemma_.lower() not in {"sheep", "deer", "fish", "series", "species", "news"}:
                            plural = _pluralize(word)
                            end = token.idx + len(token.text)
                            possess = None
                            if end < len(text):
                                rest = text[end:]
                                m_s = re.match(r"\s*'s\b", rest)
                                if m_s:
                                    possess = m_s.group(0).strip()
                                    end = end + m_s.end()
                            candidates.append(ErrorCandidate(
                                start=token.idx, end=end,
                                original=token.text if not possess else token.text + possess,
                                replacement=plural,
                                category=ErrorCategory.GRAMMAR,
                                rule_id="MISSING_PLURAL",
                                message=f'Use the plural form "{plural}" with the number {child.text}.',
                                detector="nlp_detector", raw_confidence=0.90,
                            ))

        # 13. Modal + to + verb: "must to finish" → "must finish" (regex-based)
        # Note: "need to" is correct in standard English, so exclude "need"
        MODALS_LIST = {"must", "should", "could", "would", "may", "might", "shall", "can", "dare"}
        for m in re.finditer(r'\b(must|should|could|would|may|might|shall|can|dare)\s+to\s+(\w+)\b', text, re.IGNORECASE):
            modal = m.group(1).lower()
            verb = m.group(2)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=f"{modal} {verb}",
                category=ErrorCategory.MODAL,
                rule_id="MODAL_UNNEEDED_TO",
                message=f'Do not use "to" after the modal "{modal}". Use "{modal} {verb}" instead.',
                detector="nlp_detector", raw_confidence=0.92,
            ))

        # 14. "look forward to meet" → "look forward to meeting" (to as preposition)
        lf_match = re.search(r'\b(look(?:ing)?\s+forward\s+to)\s+(\w+)\b', text, re.IGNORECASE)
        if lf_match:
            next_word = lf_match.group(2)
            if not next_word.endswith("ing") and next_word.lower() not in ("the", "a", "an", "my", "your", "his", "her", "its", "our", "their"):
                if next_word.endswith("ie"):
                    gerund = next_word[:-2] + "ying"
                elif next_word.endswith("e") and not next_word.endswith("ee"):
                    gerund = next_word[:-1] + "ing"
                else:
                    gerund = next_word + "ing"
                candidates.append(ErrorCandidate(
                    start=lf_match.start(2), end=lf_match.end(2),
                    original=next_word, replacement=gerund,
                    category=ErrorCategory.GRAMMAR,
                    rule_id="GERUND_INFINITIVE",
                    message=f'After "look forward to" (preposition), use the gerund form "{gerund}".',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 15. Redundant pronoun: "which I bought it" → "which I bought"
        for token in doc:
            if token.lower_ in ("it", "him", "her", "them", "me", "us") and token.pos_ == "PRON":
                if token.dep_ in ("dobj", "pobj"):
                    head = token.head
                    has_rel = head.dep_ == "relcl" or any(c.dep_ == "relcl" for c in head.children)
                    if not has_rel:
                        for c in head.children:
                            if c.dep_ == "relcl":
                                has_rel = True
                                break
                    if has_rel:
                        # Only flag if the relative pronoun (that/which/who) is the dobj/pobj of the same verb
                        # In "that sets him apart", "him" is NOT redundant — it's a separate object
                        rel_pronoun_is_dobj = False
                        for c in head.children:
                            if c.dep_ == "relcl":
                                for rc_child in c.children:
                                    if rc_child.tag_ in ("WP", "WDT") and rc_child.dep_ in ("dobj", "pobj"):
                                        rel_pronoun_is_dobj = True
                                        break
                        # Also check if the head verb itself has a relative pronoun as dobj/pobj
                        for c in head.children:
                            if c.tag_ in ("WP", "WDT") and c.dep_ in ("dobj", "pobj"):
                                rel_pronoun_is_dobj = True
                                break
                        if not rel_pronoun_is_dobj:
                            continue
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text) + 1,
                            original=token.text + " ", replacement="",
                            category=ErrorCategory.GRAMMAR,
                            rule_id="REDUNDANT_PRONOUN",
                            message=f'Remove the redundant pronoun "{token.text}".',
                            detector="nlp_detector", raw_confidence=0.88,
                        ))

        # 16. Redundant subject in relative clause: "who he called me" → "who called me"
        for token in doc:
            if token.lower_ in ("he", "she", "it", "they", "we", "you", "him", "her") and token.pos_ == "PRON":
                if token.dep_ == "nsubj":
                    head = token.head
                    if head.dep_ == "relcl":
                        # Check if there's a relative pronoun already
                        for t in doc[:token.i]:
                            if t.lower_ in ("who", "which", "that") and t.dep_ in ("nsubj", "nsubjpass"):
                                candidates.append(ErrorCandidate(
                                    start=token.idx, end=token.idx + len(token.text) + 1,
                                    original=token.text + " ", replacement="",
                                    category=ErrorCategory.GRAMMAR,
                                    rule_id="REDUNDANT_PRONOUN",
                                    message=f'Remove the redundant pronoun "{token.text}".',
                                    detector="nlp_detector", raw_confidence=0.88,
                                ))
                                break

        # 17. Redundant adverb: "where I was born there" → "where I was born"
        for token in doc:
            if token.lower_ == "there" and token.pos_ == "ADV":
                for t in doc:
                    if t.lower_ == "where" and t.i < token.i:
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text) + 1,
                            original=token.text + " ", replacement="",
                            category=ErrorCategory.GRAMMAR,
                            rule_id="REDUNDANT_ADVERB",
                            message=f'Remove the redundant adverb "there" — "where" already indicates the location.',
                            detector="nlp_detector", raw_confidence=0.85,
                        ))
                        break

        # 18. Word order: "I always am checking" → "I am always checking"
        # Skip fronted adverbs: "Rarely have we..." is correct
        ADV_PRE_VERB = {"always", "never", "often", "usually", "sometimes", "rarely", "seldom", "frequently", "generally", "normally"}
        for token in doc:
            if token.lower_ in ADV_PRE_VERB and token.dep_ == "advmod":
                head = token.head
                if head.pos_ in ("VERB", "AUX"):
                    # Skip if adverb is at the start (fronted adverb — correct inversion)
                    if token.i == 0 or (token.i == 1 and doc[0].pos_ == "PUNCT"):
                        continue
                    aux_before = [t for t in head.children if t.dep_ in ("aux", "auxpass") and t.i < head.i and t.i > token.i]
                    if aux_before:
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement=f"[move after {aux_before[0].text}]",
                            category=ErrorCategory.WORD_ORDER,
                            rule_id="ADVERB_POSITION",
                            message=f'Move "{token.text}" after the auxiliary verb "{aux_before[0].text}" (e.g., "I {aux_before[0].text} {token.text}").',
                            detector="nlp_detector", raw_confidence=0.82,
                        ))

        # 19. "Only after X I understood" → subject-auxiliary inversion
        # Skip if already has inversion (e.g., "Only after X did I realize")
        for sent in doc.sents:
            sent_text = sent.text
            m = re.match(r'^(Only\s+(after|before|when|if|since|until)\s+.+?)\s+(I|he|she|we|they|you)\s+(\w+)', sent_text, re.IGNORECASE)
            if m:
                subject = m.group(3)
                verb_after = m.group(4).lower()
                # Check if there's already an auxiliary before the subject
                text_before_subject = sent_text[:m.start(3)].strip().rstrip()
                words_before = text_before_subject.split()
                already_has_aux = words_before and words_before[-1].lower() in ("did", "does", "do", "has", "have", "had", "was", "were", "can", "could", "will", "would", "shall", "should", "may", "might", "must")
                if not already_has_aux:
                    candidates.append(ErrorCandidate(
                        start=sent.start_char + m.start(3), end=sent.start_char + m.end(3),
                        original=subject, replacement=f"did {subject.lower()}",
                        category=ErrorCategory.WORD_ORDER,
                        rule_id="WORD_ORDER",
                        message=f'After "Only {m.group(2)}...", use subject-auxiliary inversion.',
                        detector="nlp_detector", raw_confidence=0.80,
                    ))

        # 20. Repetitive "and": "X, and Y, and Z, and W"
        for sent in doc.sents:
            sent_text = sent.text
            if sent_text.count(" and ") >= 3:
                parts = re.split(r',\s*and\s+', sent_text.rstrip('.'))
                if len(parts) >= 3:
                    candidates.append(ErrorCandidate(
                        start=sent.start_char, end=sent.end_char,
                        original=sent.text, replacement="[restructure]",
                        category=ErrorCategory.SENTENCE_STRUCTURE,
                        rule_id="REPETITIVE_CONJUNCTION",
                        message='Too many coordinated clauses. Consider breaking into shorter sentences.',
                        detector="nlp_detector", raw_confidence=0.75,
                    ))

        # 21. Dative: "suggested me to take" → "suggested that I take"
        DATIVE_VERBS = {"suggest", "suggested", "suggests", "recommend", "recommended",
                        "propose", "proposed", "demand", "demanded", "insist", "insisted"}
        DATIVE_SUBJ_MAP = {"me": "I", "him": "he", "her": "she", "us": "we", "them": "they"}
        for token in doc:
            if token.lower_ in DATIVE_VERBS:
                for child in token.children:
                    if child.dep_ == "ccomp" and child.pos_ == "VERB":
                        for sub in child.children:
                            if sub.dep_ == "nsubj" and sub.pos_ == "PRON" and sub.lower_ in DATIVE_SUBJ_MAP:
                                subj_case = DATIVE_SUBJ_MAP[sub.lower_]
                                candidates.append(ErrorCandidate(
                                    start=child.idx, end=child.idx + len(child.text),
                                    original=f"{sub.text} to {child.text}",
                                    replacement=f"that {subj_case} {child.lemma_}",
                                    category=ErrorCategory.GRAMMAR,
                                    rule_id="DATIVE",
                                    message=f'Use "{token.text} that {subj_case} {child.lemma_}" instead of "{token.text} {sub.text} to {child.text}".',
                                    detector="nlp_detector", raw_confidence=0.82,
                                ))

        # 22. "told me to completing" → "told me to complete" (prep+pcomp VBG pattern)
        # Only fire when prep is "to" AND not a prepositional phrase like "look forward to"
        for token in doc:
            if token.pos_ == "ADP" and token.dep_ == "prep" and token.lower_ == "to":
                # Check if this is a "look forward to" type pattern (to = preposition)
                head_verb = token.head
                # Check children of head_verb for "forward" (look forward to)
                has_forward = any(c.lower_ == "forward" for c in head_verb.children)
                if has_forward:
                    continue
                for child in token.children:
                    if child.dep_ == "pcomp" and child.tag_ == "VBG":
                        candidates.append(ErrorCandidate(
                            start=child.idx, end=child.idx + len(child.text),
                            original=child.text, replacement=child.lemma_,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="INFINITIVE_FORM",
                            message=f'After "to", use the base form "{child.lemma_}" instead of "{child.text}".',
                            detector="nlp_detector", raw_confidence=0.88,
                        ))

        # 23. Redundant pronoun after preposition in relative clause: "talking about him" → "talking about"
        for token in doc:
            if token.lower_ in ("him", "her", "them", "it", "me", "us") and token.pos_ == "PRON":
                if token.dep_ == "pobj":
                    # Check if the prep is part of a relative clause
                    prep = token.head
                    if prep.dep_ == "prep":
                        rel_verb = prep.head
                        if rel_verb.dep_ == "relcl":
                            candidates.append(ErrorCandidate(
                                start=token.idx - len(token.text), end=token.idx + len(token.text),
                                original=f" {token.text}", replacement="",
                                category=ErrorCategory.GRAMMAR,
                                rule_id="REDUNDANT_PRONOUN",
                                message=f'Remove the redundant pronoun "{token.text}" from the relative clause.',
                                detector="nlp_detector", raw_confidence=0.85,
                            ))

        # 24. Missing comma after relative clause: "My friend, who lives in Chennai is" → "My friend, who lives in Chennai, is"
        # Only flag non-restrictive clauses (comma before who/which)
        for token in doc:
            if token.dep_ == "relcl":
                # Find the relative pronoun (who/which/whose/whom)
                rel_pronoun = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ == "PRON":
                        rel_pronoun = child
                        break
                if not rel_pronoun:
                    continue
                # Check if there's a comma before the relative pronoun (non-restrictive)
                if rel_pronoun.i > 0 and doc[rel_pronoun.i - 1].text == ",":
                    rel_tokens = [t for t in token.subtree]
                    if rel_tokens:
                        last_rel = max(rel_tokens, key=lambda x: x.i)
                        next_token = doc[last_rel.i + 1] if last_rel.i + 1 < len(doc) else None
                        if next_token and next_token.text != "," and next_token.pos_ in ("VERB", "AUX"):
                            candidates.append(ErrorCandidate(
                                start=last_rel.idx + len(last_rel.text), end=last_rel.idx + len(last_rel.text) + 1,
                                original=" ", replacement=", ",
                                category=ErrorCategory.PUNCTUATION,
                                rule_id="MISSING_COMMA",
                                message='Add a comma after the relative clause.',
                                detector="nlp_detector", raw_confidence=0.78,
                            ))

        # 25. Parallelism fix: check root list item for -ing ending
        for token in doc:
            if token.dep_ == "conj" and token.tag_ == "VB":
                to_children = [t for t in token.children if t.dep_ == "aux" and t.tag_ == "TO"]
                if to_children:
                    # Walk up to find the root of the list
                    list_head = token.head
                    # Check if ANY item in the list (direct or transitive conj) ends in -ing
                    found_gerund = list_head.text.lower().endswith("ing")
                    if not found_gerund:
                        # Check sibling conjuncts
                        for sib in list_head.children:
                            if sib.dep_ == "conj" and sib.text.lower().endswith("ing"):
                                found_gerund = True
                                break
                    if found_gerund:
                        gerund = token.lemma_ + "ing" if not token.lemma_.endswith("e") else token.lemma_[:-1] + "ing"
                        candidates.append(ErrorCandidate(
                            start=to_children[0].idx, end=token.idx + len(token.text),
                            original=f"to {token.text}", replacement=gerund,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="PARALLELISM",
                            message=f'Maintain parallel structure: use "{gerund}" to match the other list items.',
                            detector="nlp_detector", raw_confidence=0.82,
                        ))

        # === BATCH 3: High-frequency SEO/web-dev error patterns ===

        # 26. "this are" → "these are" (demonstrative pronoun agreement)
        this_are_pat = re.compile(r'\b(this)\s+(are)\b', re.IGNORECASE)
        for m in this_are_pat.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement="these are",
                category=ErrorCategory.SUBJECT_VERB_AGREEMENT, rule_id="SVA",
                message='"this" is singular — use "these are" for plural.', detector="nlp_detector", raw_confidence=0.92,
            ))

        # 27. "are/is discuss/go/come/etc" → progressive: "are discussing"
        PROG_VERBS = {"discuss": "discussing", "go": "going", "come": "coming", "see": "seeing",
                      "eat": "eating", "drink": "drinking", "run": "running", "walk": "walking",
                      "talk": "talking", "play": "playing", "write": "writing", "read": "reading",
                      "speak": "speaking", "listen": "listening", "watch": "watching", "help": "helping",
                      "make": "making", "take": "taking", "give": "giving", "work": "working",
                      "look": "looking", "try": "trying", "use": "using", "need": "needing"}
        prog_pat = re.compile(r'\b(is|are|am|was|were)\s+(discuss|go|come|see|eat|drink|run|walk|talk|play|write|read|speak|listen|watch|help|make|take|give|work|look|try|use|need)\b', re.IGNORECASE)
        for m in prog_pat.finditer(text):
            aux = m.group(1)
            verb = m.group(2).lower()
            if verb in PROG_VERBS:
                candidates.append(ErrorCandidate(
                    start=m.start(2), end=m.end(2), original=m.group(2), replacement=PROG_VERBS[verb],
                    category=ErrorCategory.TENSE, rule_id="PROGRESSIVE_FORM",
                    message=f'After "{aux}", use the present participle: "{aux} {PROG_VERBS[verb]}".',
                    detector="nlp_detector", raw_confidence=0.88,
                ))

        # 28. "will discussed/goed/etc" → "will discuss" (base form after modal)
        will_past = re.compile(r'\b(will|would|shall|should|can|could|may|might|must)\s+(discussed|went|came|saw|ate|drank|ran|walked|talked|played|wrote|read|spoke|listened|watched|helped|made|took|gave|worked|finished|completed|started|joined|sent|received|told|asked|replied|explained|suggested|mentioned|called|contacted|visited|returned|entered|clicked)\b', re.IGNORECASE)
        for m in will_past.finditer(text):
            modal = m.group(1)
            past = m.group(2)
            base_map = {"went": "go", "came": "come", "saw": "see", "ate": "eat", "drank": "drink",
                        "ran": "run", "played": "play", "wrote": "write", "spoke": "speak",
                        "made": "make", "took": "take", "gave": "give", "finished": "finish",
                        "completed": "complete", "started": "start", "joined": "join",
                        "sent": "send", "received": "receive", "told": "tell", "asked": "ask",
                        "replied": "reply", "explained": "explain", "suggested": "suggest",
                        "mentioned": "mention", "called": "call", "contacted": "contact",
                        "visited": "visit", "returned": "return", "entered": "enter",
                        "clicked": "click", "discussed": "discuss", "walked": "walk",
                        "talked": "talk", "watched": "watch", "helped": "help",
                        "listened": "listen", "worked": "work", "read": "read"}
            base = base_map.get(past, past)
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.end(2), original=past, replacement=base,
                category=ErrorCategory.TENSE, rule_id="MODAL_WRONG_FORM",
                message=f'After "{modal}", use the base form "{base}" instead of "{past}".',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 29. "let me knows/lets me know" → "let me know" (bare infinitive after let)
        let_knows = re.compile(r'\b(let)\s+(\w+)\s+(knows)\b', re.IGNORECASE)
        for m in let_knows.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(3), end=m.end(3), original="knows", replacement="know",
                category=ErrorCategory.GRAMMAR, rule_id="BARE_INFINITIVE",
                message='After "let me", use the bare infinitive "know" instead of "knows".',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 30. "returned back" → "returned" (redundant "back")
        returned_back = re.compile(r'\b(returned)\s+(back)\b', re.IGNORECASE)
        for m in returned_back.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(1), end=m.end(2), original=m.group(0), replacement=m.group(1),
                category=ErrorCategory.GRAMMAR, rule_id="REDUNDANT_WORD",
                message='"returned" already means "went back" — remove the redundant "back".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 31. "contact with the client" → "contact the client" (unnecessary preposition)
        contact_with = re.compile(r'\b(contact)\s+with\b', re.IGNORECASE)
        for m in contact_with.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(1), end=m.end(), original=m.group(0), replacement="contact",
                category=ErrorCategory.PREPOSITION, rule_id="UNNECESSARY_PREPOSITION",
                message='"contact" is transitive — remove "with".', detector="nlp_detector", raw_confidence=0.85,
            ))

        # 32. "in monday/tuesday/etc" → "on Monday" (day preposition)
        in_day = re.compile(r'\b(in)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.IGNORECASE)
        for m in in_day.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(1), end=m.end(2), original=m.group(0),
                replacement=f"on {m.group(2)}",
                category=ErrorCategory.PREPOSITION, rule_id="PREPOSITION_COLLOCATION",
                message=f'Use "on {m.group(2)}" (days take "on").', detector="nlp_detector", raw_confidence=0.88,
            ))

        # 33. "on 2024/2025/etc" → "in 2024" (year preposition)
        on_year = re.compile(r'\b(on)\s+((?:19|20)\d{2})\b', re.IGNORECASE)
        for m in on_year.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(1), end=m.end(2), original=m.group(0),
                replacement=f"in {m.group(2)}",
                category=ErrorCategory.PREPOSITION, rule_id="PREPOSITION_COLLOCATION",
                message=f'Use "in {m.group(2)}" (years take "in").', detector="nlp_detector", raw_confidence=0.88,
            ))

        # 34. "at next week/month" → remove "at" (unnecessary)
        at_next = re.compile(r'\b(at)\s+(next\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b', re.IGNORECASE)
        for m in at_next.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(1), end=m.end(2), original=m.group(0), replacement=m.group(2),
                category=ErrorCategory.PREPOSITION, rule_id="UNNECESSARY_PREPOSITION",
                message=f'Remove "at" before "{m.group(2)}".', detector="nlp_detector", raw_confidence=0.85,
            ))

        # 35. "in next week/month" → remove "in" (unnecessary)
        in_next = re.compile(r'\b(in)\s+(next\s+(week|month|year))\b', re.IGNORECASE)
        for m in in_next.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(1), end=m.end(2), original=m.group(0), replacement=m.group(2),
                category=ErrorCategory.PREPOSITION, rule_id="UNNECESSARY_PREPOSITION",
                message=f'Remove "in" before "{m.group(2)}".', detector="nlp_detector", raw_confidence=0.85,
            ))

        # 36. "an information" → "information" (uncountable)
        an_info = re.compile(r'\ban?\s+information\b', re.IGNORECASE)
        for m in an_info.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement="information",
                category=ErrorCategory.ARTICLE, rule_id="UNCOUNTABLE_PLURAL",
                message='"information" is uncountable — remove the article.', detector="nlp_detector", raw_confidence=0.90,
            ))

        # 37. "a feedback" → "feedback" (uncountable)
        a_feedback = re.compile(r'\ba\s+feedback\b', re.IGNORECASE)
        for m in a_feedback.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement="feedback",
                category=ErrorCategory.ARTICLE, rule_id="UNCOUNTABLE_PLURAL",
                message='"feedback" is uncountable — remove the article.', detector="nlp_detector", raw_confidence=0.90,
            ))

        # 38. "a good news" → "good news" (uncountable)
        a_news = re.compile(r'\b(a)\s+(good\s+)?news\b', re.IGNORECASE)
        for m in a_news.finditer(text):
            repl = f"{m.group(2) or ''}news".strip()
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement=repl,
                category=ErrorCategory.ARTICLE, rule_id="UNCOUNTABLE_PLURAL",
                message='"news" is uncountable — remove the article.', detector="nlp_detector", raw_confidence=0.90,
            ))

        # 39. "staffs" → "staff" (uncountable)
        staffs = re.compile(r'\bstaffs\b', re.IGNORECASE)
        for m in staffs.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original="staffs", replacement="staff",
                category=ErrorCategory.ARTICLE, rule_id="UNCOUNTABLE_PLURAL",
                message='"staff" is uncountable — use "staff" (not "staffs").',
                detector="nlp_detector", raw_confidence=0.90,
            ))

        # 40. "did a mistake" → "made a mistake" (collocation)
        did_mistake = re.compile(r'\b(did)\s+a\s+mistake\b', re.IGNORECASE)
        for m in did_mistake.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement="made a mistake",
                category=ErrorCategory.WORD_USAGE, rule_id="COLLOCATION",
                message='Use "made a mistake" (not "did a mistake").',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 41. "make a photo" → "take a photo" (collocation)
        make_photo = re.compile(r'\b(make)\s+a\s+photo\b', re.IGNORECASE)
        for m in make_photo.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement="take a photo",
                category=ErrorCategory.WORD_USAGE, rule_id="COLLOCATION",
                message='Use "take a photo" (not "make a photo").',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 42. "take an action" → "take action" (uncountable)
        take_action = re.compile(r'\b(take)\s+an?\s+action\b', re.IGNORECASE)
        for m in take_action.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement="take action",
                category=ErrorCategory.ARTICLE, rule_id="UNCOUNTABLE_PLURAL",
                message='"action" is uncountable here — use "take action".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 43. "unless you don't hurry" → "unless you hurry" (double negative)
        unless_dont = re.compile(r'\b(unless)\s+(\w+)\s+(don\'?t|doesn\'?t|do\s+not)\s+(\w+)', re.IGNORECASE)
        for m in unless_dont.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(3), end=m.end(4), original=f"{m.group(3)} {m.group(4)}", replacement=m.group(4),
                category=ErrorCategory.GRAMMAR, rule_id="DOUBLE_NEGATIVE",
                message=f'"unless" already implies negation — use "{m.group(4)}" without "{m.group(3)}".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 44. "if I will get" → "if I get" (no future tense in if-clause)
        if_will = re.compile(r'\b(if)\s+(i|he|she|it|we|they|you)\s+(will|would)\s+(\w+)', re.IGNORECASE)
        for m in if_will.finditer(text):
            subj = m.group(2)
            future = m.group(3)
            verb = m.group(4)
            if future.lower() == "will":
                candidates.append(ErrorCandidate(
                    start=m.start(3), end=m.end(4), original=f"{future} {verb}",
                    replacement=verb,
                    category=ErrorCategory.TENSE, rule_id="CONDITIONAL_TENSE",
                    message=f'In "if" clauses, don\'t use future tense — use "{verb}".',
                    detector="nlp_detector", raw_confidence=0.88,
                ))

        # 45. "he was completed the project" → "he completed" (passive misuse with active verb)
        was_completed = re.compile(r'\b(\w+)\s+(was)\s+(completed)\s+the\b', re.IGNORECASE)
        for m in was_completed.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.end(3), original="was completed",
                replacement="completed",
                category=ErrorCategory.TENSE, rule_id="PASSIVE_MISUSE",
                message='Use active voice "completed" instead of passive "was completed".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 46. "while i was work" → "while i was working" (progressive after was/were)
        was_base_prog = re.compile(r'\b(while|when)\s+\w+\s+(was|were)\s+(work|go|come|eat|drink|run|walk|talk|play|write|read|speak|listen|watch|help|make|take|give|discuss)\b', re.IGNORECASE)
        for m in was_base_prog.finditer(text):
            verb = m.group(3).lower()
            if verb in PROG_VERBS:
                candidates.append(ErrorCandidate(
                    start=m.start(3), end=m.end(3), original=m.group(3), replacement=PROG_VERBS[verb],
                    category=ErrorCategory.TENSE, rule_id="PROGRESSIVE_FORM",
                    message=f'After "was/were", use the present participle: "{PROG_VERBS[verb]}".',
                    detector="nlp_detector", raw_confidence=0.88,
                ))

        # 47. "when she arrived, we already left" → "we had already left" (past perfect)
        when_past_already = re.compile(r'\b(when)\s+\w+\s+\w+,?\s+(\w+)\s+(already)\s+(left|gone|finished|completed|arrived)\b', re.IGNORECASE)
        for m in when_past_already.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.end(4), original=f"{m.group(2)} {m.group(3)} {m.group(4)}",
                replacement=f"{m.group(2)} had {m.group(3)} {m.group(4)}",
                category=ErrorCategory.TENSE, rule_id="PAST_PERFECT",
                message='Use past perfect "had already left" for the earlier action.',
                detector="nlp_detector", raw_confidence=0.80,
            ))

        # 48. "i have already finished the task yesterday" → "i finished" (tense conflict)
        have_already_past = re.compile(r'\b(i|he|she|it|we|they|you)\s+(have|has)\s+(already\s+)?(finished|completed|done|sent|received|bought|made|written|read|seen|eaten|drunk|gone|come|left|arrived)\s+.*?\b(yesterday|last\s+(week|month|year|night|time|monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b', re.IGNORECASE)
        for m in have_already_past.finditer(text):
            subj = m.group(1)
            past_form = m.group(4)
            time_word = m.group(6)
            base_map2 = {"finished": "finished", "completed": "completed", "done": "done", "sent": "sent",
                         "received": "received", "bought": "bought", "made": "made", "written": "wrote",
                         "read": "read", "seen": "saw", "eaten": "ate", "drunk": "drank",
                         "gone": "went", "come": "came", "left": "left", "arrived": "arrived"}
            past = base_map2.get(past_form, past_form)
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.start(5),
                original=f"{m.group(2)} {m.group(3) or ''}".strip(),
                replacement="",
                category=ErrorCategory.TENSE, rule_id="TENSE_CONSISTENCY",
                message=f'With "{time_word}", use simple past: "{subj} {past}".',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 49. "looking forward for" → "looking forward to" (wrong preposition)
        lff = re.compile(r'\b(looking\s+forward\s+for)\b', re.IGNORECASE)
        for m in lff.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0), replacement="looking forward to",
                category=ErrorCategory.PREPOSITION, rule_id="PREPOSITION_COLLOCATION",
                message='Use "looking forward to" (not "looking forward for").',
                detector="nlp_detector", raw_confidence=0.88,
            ))

        # 50. "in our team" → "on our team" (team preposition)
        in_our_team = re.compile(r'\b(in)\s+(our|the|their|a|his|her)\s+team\b', re.IGNORECASE)
        for m in in_our_team.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(1), end=m.end(2), original=m.group(0),
                replacement=f"on {m.group(2)} team",
                category=ErrorCategory.PREPOSITION, rule_id="PREPOSITION_COLLOCATION",
                message=f'Use "on {m.group(2)} team" (not "in").', detector="nlp_detector", raw_confidence=0.82,
            ))

        # 51. "visited Bangalore before?" → "visited Bangalore?" (redundant "before" in questions)
        visited_before_q = re.compile(r'\b(visited|gone\s+to|been\s+to)\s+(\w+)\s+(before)\?', re.IGNORECASE)
        for m in visited_before_q.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(3), end=m.end(3), original=" before", replacement="",
                category=ErrorCategory.GRAMMAR, rule_id="REDUNDANT_WORD",
                message='Remove "before" — the question form implies past experience.',
                detector="nlp_detector", raw_confidence=0.78,
            ))

        # 52. "he is working here since 2022" → "has been working" (stative since)
        is_working_since = re.compile(r'\b(\w+)\s+(is|are)\s+(working|living|studying|teaching)\s+.*?\bsince\b', re.IGNORECASE)
        for m in is_working_since.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.start(3),
                original=f"{m.group(2)} {m.group(3)}", replacement=f"has been {m.group(3)}",
                category=ErrorCategory.TENSE, rule_id="TENSE_CONSISTENCY",
                message=f'With "since", use perfect progressive: "has been {m.group(3)}".',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 53. "enough easy" → "easy enough" (adjective + enough word order)
        enough_adj = re.compile(r'\b(enough)\s+(easy|hard|fast|slow|good|bad|big|small|long|short|old|new|hot|cold|warm|cool|dark|light|late|early|nice|great|poor|rich|strong|weak|tall|deep|wide|thin|thick|safe|dangerous|difficult|simple|complex|clear|quiet|loud|soft|rough|smooth)\b', re.IGNORECASE)
        for m in enough_adj.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=f"{m.group(2)} enough",
                category=ErrorCategory.WORD_ORDER, rule_id="WORD_ORDER",
                message=f'Use "{m.group(2)} enough" (adjective before "enough").',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 54. "reply me" → "reply to me" (missing preposition)
        reply_me = re.compile(r'\b(reply)\s+(me|him|her|us|them)\b', re.IGNORECASE)
        for m in reply_me.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=f"{m.group(1)} to {m.group(2)}",
                category=ErrorCategory.PREPOSITION, rule_id="MISSING_PREPOSITION",
                message=f'Use "{m.group(1)} to {m.group(2)}" (add "to").',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 55. "i have five years experience" → "five years of experience" (missing "of")
        years_experience = re.compile(r'\b(\d+\s+years)\s+experience\b', re.IGNORECASE)
        for m in years_experience.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=f"{m.group(1)} of experience",
                category=ErrorCategory.PREPOSITION, rule_id="MISSING_PREPOSITION",
                message=f'Use "{m.group(1)} of experience" (add "of").',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 56. "sent email to" → "sent an email to" (missing article)
        sent_email = re.compile(r'\b(sent|send|sends)\s+(email)\b', re.IGNORECASE)
        for m in sent_email.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.end(2), original="email", replacement="an email",
                category=ErrorCategory.ARTICLE, rule_id="MISSING_ARTICLE",
                message=f'Use "an email" (add article).', detector="nlp_detector", raw_confidence=0.78,
            ))

        # 57. "clicked on the button" → "clicked the button" (unnecessary "on")
        clicked_on = re.compile(r'\b(clicked)\s+on\s+(the|a|an|this|that|every|each)\s+(\w+)', re.IGNORECASE)
        for m in clicked_on.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=f"{m.group(1)} {m.group(2)} {m.group(3)}",
                category=ErrorCategory.PREPOSITION, rule_id="UNNECESSARY_PREPOSITION",
                message='"click" is transitive — remove "on".', detector="nlp_detector", raw_confidence=0.78,
            ))

        # 58. "sends me the file" (imperative with 3rd person) → "send me"
        sends_me = re.compile(r'\b(please\s+)?(sends|gives|tells|asks|shows|brings|takes|makes|lets|helps)\s+(me|him|her|us|them)\b', re.IGNORECASE)
        for m in sends_me.finditer(text):
            verb = m.group(2)
            # Only flag imperatives: sentence start, after "please", or after semicolon/newline
            before = text[:m.start()].rstrip()
            is_imperative = (
                not before or
                before.endswith(('.', '!', '?', ';', '\n')) or
                before.lower().endswith('please')
            )
            if not is_imperative:
                continue
            base_map3 = {"sends": "send", "gives": "give", "tells": "tell", "asks": "ask",
                         "shows": "show", "brings": "bring", "takes": "take", "makes": "make",
                         "lets": "let", "helps": "help"}
            base = base_map3.get(verb.lower(), verb)
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.end(2), original=verb, replacement=base,
                category=ErrorCategory.SUBJECT_VERB_AGREEMENT, rule_id="SVA",
                message=f'After "please" or in imperative, use the base form "{base}".',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 59. "very much faster" → "much faster" (redundant "very")
        very_much_comp = re.compile(r'\bvery\s+much\s+(faster|slower|bigger|smaller|better|worse|easier|harder|stronger|weaker|longer|shorter|hotter|colder|warmer|newer|older)\b', re.IGNORECASE)
        for m in very_much_comp.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=f"much {m.group(1)}",
                category=ErrorCategory.ADVERB, rule_id="REDUNDANT_ADVERB",
                message=f'Use "much {m.group(1)}" (remove redundant "very").',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 60. "where you are going" → "where are you going" (question word order inversion)
        WH_WORDS = {"where", "when", "why", "how", "what", "which", "who", "whom", "whose"}
        wh_subj_aux = re.compile(
            r'\b(' + '|'.join(WH_WORDS) + r')\s+'
            r'(i|you|he|she|it|we|they|my|your|his|her|its|our|their|\w+)\s+'
            r'(is|are|am|was|were|do|does|did|can|could|will|would|shall|should|may|might|must|have|has|had)\s+',
            re.IGNORECASE
        )
        for m in wh_subj_aux.finditer(text):
            wh = m.group(1)
            subj = m.group(2)
            aux = m.group(3)
            # Only flag in actual questions (sentences ending with ? or wh-word not at start)
            after = text[m.end():].strip()
            before_text = text[:m.start()].strip().lower()
            is_question = after.endswith('?') or before_text == ''
            if not is_question:
                # Also allow if the WH-word is after a known prefix word
                if before_text and before_text.split()[-1] in ('today', 'yesterday', 'last', 'this', 'during', 'before', 'after', 'meeting', 'deadline', 'call', 'morning', 'week', 'time', 'project', 'report'):
                    is_question = True
            if not is_question:
                continue
            # Skip compound interrogatives: "which one do", "what time does", "how much does", etc.
            compound_wh = {"one", "time", "much", "many", "kind", "type", "sort",
                          "way", "thing", "person", "people", "place", "reason",
                          "long", "far", "old", "tall", "fast", "soon", "often"}
            if subj.lower() in compound_wh:
                continue
            # Skip if it's a relative clause (who/which/that after a noun)
            before = text[:m.start()].rstrip()
            if before:
                last_token = before.split()[-1].lower() if before.split() else ""
                if last_token in ("the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their"):
                    continue
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0).rstrip(),
                replacement=f"{wh} {aux} {subj} ",
                category=ErrorCategory.WORD_ORDER, rule_id="QUESTION_INVERSION",
                message=f'In questions, the auxiliary verb comes before the subject: "{wh} {aux} {subj}...".',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 60b. "what time the meeting starts" → "what time does the meeting start" (missing auxiliary in wh-questions)
        wh_noun_verb = re.compile(
            r'\b(what)\s+(time|kind|type|sort|way)\s+(the|a|an|this|that|my|your|his|her|our|their)\s+(\w+)\s+(starts?|ends?|runs?|works?|moves?|sits?|stands?|reads?|writes?|plays?|lives?|loves?|hates?|wants?|needs?|likes?|comes?|goes?|makes?|takes?|gives?|gets?|puts?|says?|tells?|asks?|uses?|finds?|knows?|thinks?|sees?|hears?|feels?|tries?|helps?|calls?|asks?|opens?|closes?|shows?|turns?|follows?|brings?|holds?|keeps?|leaves?|meets?|pays?|runs?|sells?|sends?|sets?|teaches?|works?)\b',
            re.IGNORECASE
        )
        for m in wh_noun_verb.finditer(text):
            wh_word = m.group(1)
            compound = m.group(2)
            article = m.group(3)
            noun = m.group(4)
            verb = m.group(5)
            base_verb = verb.rstrip('s') if verb.endswith('s') and not verb.endswith('ss') else verb
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=f"{wh_word} {compound} {article} {noun} {base_verb}",
                category=ErrorCategory.WORD_ORDER, rule_id="QUESTION_INVERSION",
                message=f'In questions, use "does" + base form: "{wh_word} {compound} {article} {noun} does {base_verb}...".',
                detector="nlp_detector", raw_confidence=0.80,
            ))

        # 60c. Collective noun SVA: "the team are" → "the team is" (American English)
        COLLECTIVE_NOUNS = {"team", "group", "staff", "company", "organization", "committee",
                           "family", "class", "audience", "public", "government", "council",
                           "board", "jury", "army", "police", "news", "math", "physics"}
        for token in doc:
            if token.lower_ in COLLECTIVE_NOUNS and token.pos_ in ("NOUN", "PROPN"):
                # Find the verb this noun is subject of
                if token.dep_ == "nsubj":
                    verb = token.head
                    if verb.pos_ in ("AUX", "VERB") and verb.lower_ in ("are", "were"):
                        sent = token.sent
                        candidates.append(ErrorCandidate(
                            start=sent.start_char + verb.idx,
                            end=sent.start_char + verb.idx + len(verb.text),
                            original=verb.text,
                            replacement="is" if verb.lower_ == "are" else "was",
                            category=ErrorCategory.SUBJECT_VERB_AGREEMENT,
                            rule_id="SVA_COLLECTIVE",
                            message=f'The collective noun "{token.text}" takes a singular verb in American English.',
                            detector="nlp_detector", raw_confidence=0.75,
                        ))

        # 61. "three year experience" / "five years experience" → needs "of" (word-number + years + experience)
        word_num_years_exp = re.compile(
            r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
            r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
            r'thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\s+'
            r'(year|years|month|months|week|weeks|day|days|hour|hours)\s+'
            r'(experience|work|study|studies|teaching|training|practice|research|learning|knowledge)\b',
            re.IGNORECASE
        )
        for m in word_num_years_exp.finditer(text):
            time_word = m.group(2)
            noun = m.group(3)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=f"{m.group(1)} {time_word} of {noun}",
                category=ErrorCategory.PREPOSITION, rule_id="MISSING_PREPOSITION",
                message=f'Use "{m.group(1)} {time_word} of {noun}" (add "of").',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 62. "digit years experience" → "digit years of experience" (e.g., "three 3 years experience")
        digit_years_exp = re.compile(
            r'\b(\d+)\s+(year|years|month|months|week|weeks|day|days|hour|hours)\s+'
            r'(experience|work|study|studies|teaching|training|practice|research|learning|knowledge)\b',
            re.IGNORECASE
        )
        for m in digit_years_exp.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=f"{m.group(1)} {m.group(2)} of {m.group(3)}",
                category=ErrorCategory.PREPOSITION, rule_id="MISSING_PREPOSITION",
                message=f'Use "{m.group(1)} {m.group(2)} of {m.group(3)}" (add "of").',
                detector="nlp_detector", raw_confidence=0.82,
            ))

        # 63. "made a research" / "do a research" → uncountable "research"
        made_a_research = re.compile(r'\b(made|did|done|conduct|conducted|perform|performed|complete|completed|finish|finished|start|started|begin|began|prepare|prepared)\s+(a\s+research)\b', re.IGNORECASE)
        for m in made_a_research.finditer(text):
            candidates.append(ErrorCandidate(
                start=m.start(2), end=m.end(2), original=m.group(2), replacement="research",
                category=ErrorCategory.ARTICLE, rule_id="UNCOUNTABLE_PLURAL",
                message='"research" is uncountable — remove the article "a".',
                detector="nlp_detector", raw_confidence=0.85,
            ))

        # 64. "a research" generic (after any verb)
        a_research = re.compile(r'\ba\s+research\b', re.IGNORECASE)
        for m in a_research.finditer(text):
            # Don't double-detect if detector #63 already caught it
            if not any(c.start <= m.start() < c.end for c in candidates):
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(), original=m.group(0), replacement="research",
                    category=ErrorCategory.ARTICLE, rule_id="UNCOUNTABLE_PLURAL",
                    message='"research" is uncountable — remove the article "a".',
                    detector="nlp_detector", raw_confidence=0.85,
                ))

        # 65. "buy new laptop" → "buy a new laptop" (missing article before singular countable noun after verb)
        # Pattern: verb + adjective + singular noun (no article)
        MISSING_ARTICLE_PATTERNS = [
            (r'\b(buy|bought|need|needed|get|got|find|found|order|ordered|receive|received|want|wanted|use|used|install|installed)\s+([a-z]+)\s+(laptop|computer|phone|tablet|monitor|keyboard|mouse|desk|chair|car|bike|book|pen|table|door|window|screen|camera|router|server|software|tool|device|machine|printer|scanner|headphone|speaker)\b',
             'missing article before singular countable noun'),
        ]
        for pat, _ in MISSING_ARTICLE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                # Check that there's no article before the adjective
                adj = m.group(2)
                if adj.lower() not in ('a', 'an', 'the', 'this', 'that', 'these', 'those'):
                    candidates.append(ErrorCandidate(
                        start=m.start(2), end=m.end(3), original=f"{adj} {m.group(3)}",
                        replacement=f"a {adj} {m.group(3)}",
                        category=ErrorCategory.ARTICLE, rule_id="ARTICLE_MISSING",
                        message=f'Add "a" before "{adj} {m.group(3)}" (missing article).',
                        detector="nlp_detector", raw_confidence=0.78,
                    ))

        # 65b. Sentence-initial missing article: "Disadvantage is..." → "A disadvantage is..."
        # Pattern: sentence starts with a singular countable noun without an article
        SINGULAR_NOUNS_START = {
            "disadvantage", "advantage", "problem", "solution", "result", "reason",
            "person", "student", "teacher", "doctor", "man", "woman", "child",
            "company", "government", "society", "country", "world", "system",
            "process", "method", "example", "fact", "idea", "issue", "case",
            "question", "answer", "point", "way", "thing", "place", "time",
            "number", "group", "part", "area", "form", "level", "type",
            "plan", "program", "project", "policy", "law", "rule", "act",
            "device", "machine", "tool", "car", "house", "building", "school",
            "university", "library", "hospital", "market", "store", "shop",
            " restaurant", "hotel", "airport", "station", "park", "garden",
            "book", "article", "paper", "report", "letter", "email", "message",
            "website", "page", "screen", "computer", "phone", "table",
            "team", "group", "class", "family", "friend", "neighbor",
        }
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            # Get first content word (skip punctuation, adverbs)
            first_content = None
            for token in sent:
                if token.pos_ in ("NOUN", "PROPN") and token.dep_ in ("nsubj", "attr", "dobj", "pobj", "ROOT"):
                    first_content = token
                    break
                if token.pos_ not in ("PUNCT", "SPACE", "ADV", "ADJ"):
                    break
            if first_content and first_content.pos_ == "NOUN":
                if first_content.text.lower() in SINGULAR_NOUNS_START:
                    # Check there's no determiner before it
                    has_det = any(c.dep_ in ("det",) for c in first_content.children)
                    if not has_det:
                        # Check it's at or near start of sentence
                        if first_content.i - sent.start < 3:
                            candidates.append(ErrorCandidate(
                                start=first_content.idx, end=first_content.idx + len(first_content.text),
                                original=first_content.text, replacement=f"a {first_content.text.lower()}",
                                category=ErrorCategory.ARTICLE, rule_id="ARTICLE_MISSING",
                                message=f'Add "a" before "{first_content.text}" (missing article at sentence start).',
                                detector="nlp_detector", raw_confidence=0.72,
                            ))

        # 66. "if you study hard, you would pass" → "you will pass" (wrong conditional type)
        # Only flag when if-clause uses present/base (zero/first conditional) but result uses "would"
        # Don't flag 2nd conditional (if + past, would) or 3rd (if + had + pp, would have)
        if_cond_result = re.compile(
            r'\b(if)\s+(.+?),\s+'
            r'(i|you|he|she|it|we|they)\s+(would|could|might)\s+(\w+)',
            re.IGNORECASE
        )
        for m in if_cond_result.finditer(text):
            modal = m.group(4)
            verb = m.group(5)
            if_clause = m.group(2).lower()
            if modal.lower() == "would":
                # Check if-clause tense: skip if it uses past tense (2nd/3rd conditional)
                # Use spaCy to check for VBD (past tense) tags in the if-clause
                if_clause_doc = get_nlp()(if_clause)
                has_past_tense = any(t.tag_ == "VBD" for t in if_clause_doc)
                if has_past_tense:
                    continue
                candidates.append(ErrorCandidate(
                    start=m.start(4), end=m.end(5),
                    original=f"{m.group(4)} {verb}", replacement=f"will {verb}",
                    category=ErrorCategory.TENSE, rule_id="CONDITIONAL_TENSE",
                    message=f'In a zero/first conditional, use "will {verb}" (not "would {verb}") in the result clause.',
                    detector="nlp_detector", raw_confidence=0.82,
                ))

        return candidates

    def _check_comma_splice(self, text: str, doc) -> List[ErrorCandidate]:
        """Comma splice: 'I went to the store, I bought milk.' → add conjunction or use period."""
        candidates = []
        nlp_model = get_nlp()
        for sent in doc.sents:
            sent_text = sent.text
            parts = re.split(r',\s*', sent_text)
            if len(parts) != 2:
                continue
            part1, part2 = parts[0].strip(), parts[1].strip()
            if not part1 or not part2:
                continue
            # Skip if part1 contains a semicolon — semicolon is correct punctuation
            if ';' in part1:
                continue
            doc1 = nlp_model(part1)
            doc2 = nlp_model(part2)
            # Skip if first clause is subordinate (whatever/wherever/if/because/when/etc.)
            first_has_subord = any(t.dep_ in ("advcl", "mark", "relcl") or t.tag_ in ("WRB", "WP", "WDT") for t in doc1)
            if first_has_subord:
                continue
            first_words = [t.lower_ for t in doc1]
            # Skip subordinating adverbs at start: "However hard he tried", "Wherever you go"
            if first_words and first_words[0] in ("however", "wherever", "whenever", "whatever", "whoever"):
                continue
            # Skip inverted conditionals: "Were I...", "Had I...", "Should you..."
            if first_words and first_words[0] in ("were", "had", "should") and len(first_words) > 1:
                if first_words[1] in ("i", "he", "she", "it", "you", "we", "they"):
                    continue
            # Skip correlative conjunctions: "Not only...but", "The more...the more"
            lower_text_part = part1.lower()
            if "not only" in lower_text_part or "the more" in lower_text_part or "the less" in lower_text_part:
                continue
            # Skip "The X-er, the Y-er" correlative comparatives
            if re.match(r'^the\s+\w+(?:er|ier)\b', lower_text_part) and re.match(r'^the\s+\w+(?:er|ier)\b', part2.lower()):
                continue
            has_subj1 = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc1)
            has_subj2 = any(t.dep_ in ("nsubj", "nsubjpass") for t in doc2)
            has_verb1 = any(t.pos_ in ("VERB", "AUX") for t in doc1)
            has_verb2 = any(t.pos_ in ("VERB", "AUX") for t in doc2)
            # Skip if followed by coordinating conjunction: "X, but Y" is valid
            second_words = [t.lower_ for t in doc2]
            if second_words and second_words[0] in ("and", "but", "or", "nor", "for", "yet", "so"):
                continue
            if has_subj1 and has_verb1 and has_subj2 and has_verb2:
                comma_pos = text.find(',', sent.start_char)
                if comma_pos != -1:
                    replacement_text = part2[0].upper() + part2[1:] if part2 else part2
                    candidates.append(ErrorCandidate(
                        start=comma_pos, end=comma_pos + 1,
                        original=",",
                        replacement=". " + replacement_text,
                        category=ErrorCategory.PUNCTUATION,
                        rule_id="COMMA_SPLICE",
                        message='Two independent clauses joined by a comma. Use a period, semicolon, or add a conjunction.',
                        detector="nlp_detector",
                        raw_confidence=0.80,
                    ))
        return candidates

    def _check_modal_errors(self, text: str, doc) -> List[ErrorCandidate]:
        """Modal verb errors: 'can sings' → 'can sing', 'must to go' → 'must go', etc."""
        candidates = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ != "aux" or token.text.lower() not in MODALS:
                    continue
                # Skip "'d" when it's "had" (past perfect: "We'd been")
                if token.text.lower() in ("'d",):
                    # "'d" = "would" or "had" — skip if head is "been" or VBN (past perfect)
                    if token.head.text.lower() in ("been", "have", "has") or token.head.tag_ in ("VBN", "VBG"):
                        continue
                main_verb = token.head
                sent_start = sent.start_char
                # Pattern 1: Modal + VBZ/VBD/VBN → Modal + VB (base form)
                # Skip when an "of" follows the modal ("could of done") — that's
                # the "could have" pattern handled by WOULD_OF_*.
                of_between = token.i + 1 < len(doc) and doc[token.i + 1].lower_ == "of"
                base = None
                if main_verb.tag_ in ("VBZ", "VBD"):
                    base = main_verb.lemma_ if not of_between else None
                elif main_verb.tag_ == "VBN":
                    # "could noticed" → "could notice" — but not "must be locked"
                    # (the VBN already has a be/have aux, i.e. a passive/perfect chain)
                    if not any(t.dep_ in ("aux", "auxpass") and t.lower_ in (BE_FORMS | HAVE_FORMS) for t in main_verb.children):
                        base = main_verb.lemma_ if not of_between else None
                elif (main_verb.tag_ == "VB" and main_verb.dep_ not in ("aux", "auxpass")
                        and main_verb.text.lower().endswith("ed")
                        and main_verb.lemma_ == main_verb.text.lower()
                        and not any(t.dep_ in ("aux", "auxpass") and t.lower_ in (BE_FORMS | HAVE_FORMS) for t in main_verb.children)) and not of_between:
                    # Misparsed -ed form (e.g., "should focused" → tag VB, lemma "focused")
                    base = self._modal_ed_to_base(main_verb.text.lower())
                if base and base != main_verb.text.lower():
                    candidates.append(ErrorCandidate(
                        start=sent_start + main_verb.idx,
                        end=sent_start + main_verb.idx + len(main_verb.text),
                        original=main_verb.text,
                        replacement=base,
                        category=ErrorCategory.GRAMMAR,
                        rule_id="MODAL_WRONG_FORM",
                        message=f'After the modal "{token.text}", use the base form "{base}" instead of "{main_verb.text}".',
                        detector="nlp_detector",
                        raw_confidence=0.92,
                    ))
                # Skip "be" after modal — correct for progressive: "shouldn't be doing"
                if main_verb.text.lower() == "be" and main_verb.tag_ == "VB":
                    continue
                # Skip if modal head is not the main verb but an intermediate aux
                # e.g., "should be doing" — "should" head is "doing" (VBG), but "be" is between
                if main_verb.tag_ == "VBG":
                    # VBG after modal is only wrong if there's no "be" intermediate
                    # Check if any child of main_verb is "be" as aux
                    has_be_aux = any(t.text.lower() in BE_FORMS and t.dep_ in ("aux", "auxpass") for t in main_verb.children)
                    if has_be_aux:
                        continue
                    # Also skip if the sentence has a "be" aux between modal and main verb
                    for t in sent:
                        if t.text.lower() in BE_FORMS and t.dep_ == "aux" and t.head == main_verb:
                            has_be_aux = True
                            break
                    if has_be_aux:
                        continue
                    base = main_verb.lemma_
                    candidates.append(ErrorCandidate(
                        start=sent_start + main_verb.idx,
                        end=sent_start + main_verb.idx + len(main_verb.text),
                        original=main_verb.text,
                        replacement=base,
                        category=ErrorCategory.GRAMMAR,
                        rule_id="MODAL_WRONG_FORM",
                        message=f'After the modal "{token.text}", use the base form "{base}" instead of "{main_verb.text}".',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                    ))
                    continue
                # Pattern 2b: Modal + xcomp(VB) where xcomp has "to" aux → Modal + VB
                # "must to finish" → spaCy: must→finish(xcomp), finish has child to(aux)
                if main_verb.tag_ == "VB" and main_verb.dep_ == "xcomp":
                    to_children = [t for t in main_verb.children if t.dep_ == "aux" and t.tag_ == "TO"]
                    if to_children:
                        # "must to finish" → "must finish"
                        candidates.append(ErrorCandidate(
                            start=sent_start + to_children[0].idx,
                            end=sent_start + main_verb.idx + len(main_verb.text),
                            original=to_children[0].text + " " + main_verb.text,
                            replacement=main_verb.text,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="MODAL_UNNEEDED_TO",
                            message=f'Do not use "to" after the modal "{token.text}". Use "{token.text} {main_verb.text}" instead.',
                            detector="nlp_detector",
                            raw_confidence=0.92,
                        ))
                # Pattern 2: Modal + "to" + VB → Modal + VB
                if main_verb.text.lower() == "to" and main_verb.tag_ == "TO":
                    for child in main_verb.children:
                        if child.dep_ == "xcomp" and child.pos_ in ("VERB", "AUX"):
                            candidates.append(ErrorCandidate(
                                start=sent_start + main_verb.idx,
                                end=sent_start + child.idx + len(child.text),
                                original=main_verb.text + " " + child.text,
                                replacement=child.text,
                                category=ErrorCategory.GRAMMAR,
                                rule_id="MODAL_UNNEEDED_TO",
                                message=f'Do not use "to" after the modal "{token.text}". Use "{token.text} {child.text}" instead.',
                                detector="nlp_detector",
                                raw_confidence=0.92,
                            ))
                            break
                # Pattern 3: Modal + has → Modal + have (base form)
                if main_verb.text.lower() in ("has",) and main_verb.tag_ == "VBZ":
                    candidates.append(ErrorCandidate(
                        start=sent_start + main_verb.idx,
                        end=sent_start + main_verb.idx + len(main_verb.text),
                        original=main_verb.text,
                        replacement="have",
                        category=ErrorCategory.GRAMMAR,
                        rule_id="MODAL_WRONG_FORM",
                        message=f'After the modal "{token.text}", use "have" instead of "has".',
                        detector="nlp_detector",
                        raw_confidence=0.92,
                    ))
                # Pattern 4: Modal + VBG → Modal + VB
                # "may going" → "may go"
                if main_verb.tag_ == "VBG":
                    base = main_verb.lemma_
                    candidates.append(ErrorCandidate(
                        start=sent_start + main_verb.idx,
                        end=sent_start + main_verb.idx + len(main_verb.text),
                        original=main_verb.text,
                        replacement=base,
                        category=ErrorCategory.GRAMMAR,
                        rule_id="MODAL_WRONG_FORM",
                        message=f'After the modal "{token.text}", use the base form "{base}" instead of "{main_verb.text}".',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                    ))

            # Pattern 5: Double modals — "could should", "could would", "can could"
            modal_tokens = [t for t in sent if t.dep_ == "aux" and t.text.lower() in MODALS]
            if len(modal_tokens) >= 2:
                # Only flag if both modals govern the same verb (true double modal)
                # Skip if modals are in separate clauses (e.g., "we can...or we can")
                m1, m2 = modal_tokens[0], modal_tokens[1]
                if m1.head == m2.head or m1.head == m2 or m2.head == m1:
                    m2 = modal_tokens[1]
                    candidates.append(ErrorCandidate(
                        start=sent_start + m2.idx,
                        end=sent_start + m2.idx + len(m2.text),
                        original=modal_tokens[0].text + " " + m2.text,
                        replacement=modal_tokens[0].text,
                        category=ErrorCategory.GRAMMAR,
                        rule_id="DOUBLE_MODAL",
                        message=f'Use only one modal verb. Remove "{m2.text}".',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                    ))

        return candidates

    # Verb bases that legitimately end in "-ed" — never strip them after a modal.
    _MODAL_ED_EXCEPTIONS = {
        "feed", "breed", "proceed", "exceed", "succeed", "recede", "concede",
        "accede", "supersede", "precede", "seed", "need", "weed", "speed",
        "bleed", "knead", "bead", "deed", "greed", "creed", "steed", "swede",
        "embed", "ambed"}

    # Mis-tagged "-ed" forms whose stem keeps a silent "e" — string surgery alone
    # is ambiguous ("focused" vs "closed" share the same suffix shape), so the
    # correct base is curated here.
    _MODAL_ED_EBASE = {
        "closed": "close", "prepared": "prepare", "completed": "complete",
        "created": "create", "saved": "save", "moved": "move", "used": "use",
        "raised": "raise", "shared": "share", "installed": "install",
        "upgraded": "upgrade", "updated": "update", "deleted": "delete",
        "located": "locate", "organised": "organise", "organized": "organize",
    }

    def _modal_ed_to_base(self, word):
        """Recover the base form of a mis-tagged VBN/VBD that spaCy tagged VB.

        "focused" → "focus", "studied" → "study", "stopped" → "stop",
        "closed" → "close" (silent-e stem, curated since suffix surgery is
        ambiguous).
        Returns None when the word is an exception (its base genuinely ends in -ed)
        or when we cannot confidently strip the suffix.
        """
        w = word
        if w in self._MODAL_ED_EXCEPTIONS:
            return None
        if w in self._MODAL_ED_EBASE:
            return self._MODAL_ED_EBASE[w]
        if w.endswith("ied"):
            base = w[:-3] + "y"
        elif w.endswith("ded") and len(w) > 3 and w[-4] == w[-3]:
            # "stopped" → "stop", "planned" → "plan"
            base = w[:-2]
            if len(base) > 1 and base[-1] == base[-2]:
                base = base[:-1]
        else:
            base = w[:-2]
            if not base:
                return None
            # "noticed" → "notic" → "notice"  (verbs whose stem needs a final e)
            if base.endswith("c") or base.endswith("v"):
                base = base + "e"
        # Sanity: base must differ and not itself end in "ed" (e.g., "focused"→"focus" ok;
        # "needed" → "need" is an exception covered above).
        if base == w or base.endswith("ed"):
            return None
        return base

    def _check_duplicate_verb(self, text: str, doc) -> List[ErrorCandidate]:
        """Adjacent repeated verbs ('have have', 'discussed discuss')."""
        candidates = []
        for sent in doc.sents:
            sent_start = sent.start_char
            tokens = [t for t in sent if not t.is_space and not t.is_punct]
            for i, token in enumerate(tokens):
                # Case A: "discussed discuss" — xcomp with the same lemma as its head.
                if (token.dep_ == "xcomp" and token.pos_ == "VERB"
                        and token.head.pos_ == "VERB" and token.head.lemma_ == token.lemma_
                        and not any(t.dep_ == "aux" and t.tag_ == "TO" for t in token.children)):
                    head = token.head
                    if head.text.lower() != token.text.lower():
                        candidates.append(ErrorCandidate(
                            start=sent_start + head.idx,
                            end=sent_start + token.idx + len(token.text),
                            original=head.text + " " + token.text,
                            replacement=head.text,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="DUPLICATE_VERB",
                            message=f'Remove the repeated verb "{token.text}" — "{head.text}" already covers it.',
                            detector="nlp_detector",
                            raw_confidence=0.90,
                        ))
                        continue
                # Case B: same spelled verb twice in a row ("have have", "is is").
                if i == 0:
                    continue
                prev = tokens[i - 1]
                if prev.text.lower() != token.text.lower():
                    continue
                if token.pos_ not in ("VERB", "AUX") or prev.pos_ not in ("VERB", "AUX"):
                    continue
                if token.lower_ == "had" and prev.lower_ == "had":
                    continue  # valid past perfect: "had had"
                candidates.append(ErrorCandidate(
                    start=sent_start + token.idx,
                    end=sent_start + token.idx + len(token.text),
                    original=prev.text + " " + token.text,
                    replacement=prev.text,
                    category=ErrorCategory.GRAMMAR,
                    rule_id="DUPLICATE_VERB",
                    message=f'Remove the repeated verb "{token.text}".',
                    detector="nlp_detector",
                    raw_confidence=0.92,
                ))
        return candidates

    _WEEKDAY_NAMES = {"monday", "tuesday", "wednesday", "thursday", "friday",
                      "saturday", "sunday"}
    _SCHEDULED_ON_EXCLUDE = {"calendar", "system", "agenda", "schedule", "list",
                             "document", "spreadsheet", "time", "date", "plan"}

    def _check_scheduled_on(self, text: str, doc) -> List[ErrorCandidate]:
        """'scheduled on Monday' → 'scheduled for Monday' (days, not calendar/system)."""
        candidates = []
        for token in doc:
            if token.lemma_ != "schedule" or token.pos_ != "VERB":
                continue
            for child in token.children:
                if child.dep_ != "prep" or child.lower_ != "on":
                    continue
                subtree_words = [c.lower_ for c in child.subtree]
                if not any(w in self._WEEKDAY_NAMES for w in subtree_words):
                    continue
                if any(w in self._SCHEDULED_ON_EXCLUDE for w in subtree_words):
                    continue
                candidates.append(ErrorCandidate(
                    start=child.idx,
                    end=child.idx + len(child.text),
                    original=child.text,
                    replacement="for",
                    category=ErrorCategory.PREPOSITION,
                    rule_id="PREPOSITION_SCHEDULED_ON",
                    message=f'Use "for" with days when something is scheduled ("scheduled for Monday").',
                    detector="nlp_detector",
                    raw_confidence=0.85,
                ))
                break
        return candidates

    _BACKSHIFT_PRESENT_TIMES = {"now", "today", "currently", "still", "yet",
                                "always", "nowadays", "these days"}

    def _has_past_time_marker(self, sent) -> bool:
        toks = [t for t in sent if not t.is_space]
        for i, t in enumerate(toks):
            tl = t.lower_
            if tl == "yesterday" or tl == "previously" or tl == "formerly":
                return True
            if tl == "ago" and i > 0 and toks[i - 1].pos_ == "NUM":
                return True
            if tl == "last" and i + 1 < len(toks) and toks[i + 1].lower_ in {
                    "week", "month", "year", "night", "summer", "winter", "fall",
                    "autumn", "spring", "weekend", "monday", "tuesday", "wednesday",
                    "thursday", "friday", "saturday", "sunday"}:
                return True
            if tl in ("in", "on") and i + 1 < len(toks) and toks[i + 1].pos_ == "NUM":
                return True  # "in 2019", "on 5 June"
        return False

    def _has_past_head_chain(self, token):
        cur = token.head
        seen = 0
        while cur is not None and seen < 10:
            if cur.pos_ == "VERB" and cur.tag_ in ("VBD", "VBN", "VBG"):
                return True
            if cur == cur.head:
                break  # reached a ROOT (self-head) verb that was not past
            cur = cur.head
            seen += 1
        return False

    def _simple_past(self, token):
        """Present → simple past, or None if we cannot inflect confidently."""
        base = token.lemma_
        if token.tag_ in ("VBZ", "VBP", "VB"):
            if base in IRREGULAR_VERBS and "past" in IRREGULAR_VERBS[base]:
                return IRREGULAR_VERBS[base]["past"]
            if base in BE_FORMS:
                return {"am": "was", "is": "was", "are": "were"}.get(base)
            if base in HAVE_FORMS:
                return "had"
            if base in DO_FORMS:
                return "did"
            if base.endswith("e"):
                return base + "d"
            return base + "ed"
        return None

    def _check_backshift(self, text: str, doc) -> List[ErrorCandidate]:
        """Present-tense subordinate verb in an explicit past narrative → past tense.

        "Last week ... noticed that the slide contains ..." → "contained".
        Conservative: requires an explicit past-time marker, a past verb in the
        head chain, and a standalone present verb (no aux, no timeless markers).
        """
        candidates = []
        for sent in doc.sents:
            if not self._has_past_time_marker(sent):
                continue
            for token in sent:
                if token.pos_ != "VERB" or token.dep_ not in ("ccomp", "advcl"):
                    continue
                if token.tag_ not in ("VBZ", "VBP", "VB"):
                    continue
                if any(c.dep_ in ("aux", "auxpass") for c in token.children):
                    continue  # progressive/passive — leave tense alone
                if any(t.lower_ in self._BACKSHIFT_PRESENT_TIMES for t in token.subtree):
                    continue
                if not self._has_past_head_chain(token):
                    continue
                if token.lemma_ in BE_FORMS:
                    continue  # "is/was" stative copulas are too risky
                past = self._simple_past(token)
                if not past or past == token.text.lower():
                    continue
                candidates.append(ErrorCandidate(
                    start=sent.start_char + token.idx,
                    end=sent.start_char + token.idx + len(token.text),
                    original=token.text,
                    replacement=past,
                    category=ErrorCategory.TENSE,
                    rule_id="TENSE_BACKSHIFT",
                    message=f'In this past narrative, shift "{token.text}" to the past tense "{past}".',
                    detector="nlp_detector",
                    raw_confidence=0.80,
                ))
        return candidates

    def _check_anybody_declarative(self, text: str, doc) -> List[ErrorCandidate]:
        """Positive declarative 'anybody/anyone' + finite lexical verb → 'nobody/-one'.

        "Anybody knew the answer." → "Nobody knew the answer."
        Positive-polarity anyone + a factive past/present verb is non-standard;
        a modal, a relative ('anybody who'), or 'with' constructions are left alone.
        """
        candidates = []
        for sent in doc.sents:
            toks = [t for t in sent if not t.is_space]
            if not toks:
                continue
            subj = toks[0]
            if subj.lower_ not in ("anybody", "anyone"):
                continue
            verb = subj.head
            if verb is None or verb.pos_ != "VERB" or verb.dep_ not in ("ROOT", "ccomp"):
                continue
            # The subject must directly precede its verb (no relative/with/NP after it).
            next_tok = toks[1] if len(toks) > 1 else None
            if next_tok is None or not (next_tok == verb or next_tok.lower_ in ("'d", "n't")):
                continue
            if verb.tag_ not in ("VBZ", "VBP", "VBD"):
                continue
            if any(t.dep_ == "aux" and (t.lower_ in MODALS or t.lower_ in DO_FORMS) for t in verb.children):
                continue  # "anybody can/may/does..." — grammatical
            if any(t.dep_ == "neg" for t in verb.children):
                continue
            has_q = sent.text.strip().endswith("?")
            if has_q:
                continue
            repl = "Nobody" if subj.text[:1].isupper() else "nobody"
            candidates.append(ErrorCandidate(
                start=sent.start_char + subj.idx,
                end=sent.start_char + subj.idx + len(subj.text),
                original=subj.text,
                replacement=repl,
                category=ErrorCategory.WORD_USAGE,
                rule_id="ANYBODY_DECLARATIVE",
                message=f'Use "{repl}" instead of "{subj.text}" in a positive statement.',
                detector="nlp_detector",
                raw_confidence=0.85,
            ))
        return candidates

    def _check_sentence_fragments(self, text: str, doc) -> List[ErrorCandidate]:
        """Detect sentence fragments: missing subject or missing finite verb."""
        candidates = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text or sent_text[0] == '"':
                continue

            has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in sent)
            has_finite_verb = any(t.tag_ in ("VBP", "VBZ", "VBD", "MD") for t in sent)
            has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)

            # Fragment: has verb but no subject and no finite verb
            # "Running through the park." / "Went to the store yesterday."
            if has_verb and not has_subj and not has_finite_verb:
                if len(sent_text.split()) >= 2:
                    candidates.append(ErrorCandidate(
                        start=sent.start_char,
                        end=sent.end_char,
                        original=sent_text,
                        replacement=None,
                        category=ErrorCategory.SENTENCE_STRUCTURE,
                        rule_id="SENTENCE_FRAGMENT",
                        message='This appears to be a sentence fragment. It may be missing a subject or a finite verb.',
                        detector="nlp_detector",
                        raw_confidence=0.80,
                    ))

            # Fragment: starts with subordinating conjunction and no main clause
            # "Because I was tired."
            first_token = list(sent)[0] if len(list(sent)) > 0 else None
            if first_token and first_token.lower_ in ("because", "although", "though", "while", "when", "if", "since", "unless", "until", "after", "before"):
                # Check if there's a main clause (ROOT verb that's not in a dependent clause)
                root_verbs = [t for t in sent if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")]
                if not root_verbs or (len(root_verbs) == 1 and root_verbs[0].dep_ == "advcl"):
                    candidates.append(ErrorCandidate(
                        start=sent.start_char,
                        end=sent.end_char,
                        original=sent_text,
                        replacement=None,
                        category=ErrorCategory.SENTENCE_STRUCTURE,
                        rule_id="SENTENCE_FRAGMENT",
                        message=f'This appears to be a sentence fragment. "Because..." clauses need a main clause.',
                        detector="nlp_detector",
                        raw_confidence=0.80,
                    ))

        return candidates

    def _check_do_support(self, text: str, doc) -> List[ErrorCandidate]:
        """'don't' with 3rd person singular → 'doesn't': 'She don't know' → 'She doesn't know'."""
        candidates = []
        for token in doc:
            is_do_aux = False
            # spaCy splits "don't" → "do" + "nt" (or "n't"). "nt" is child of ROOT, not "do"
            if token.text.lower() in ("don't", "dont"):
                is_do_aux = True
            elif token.text.lower() == "do" and token.dep_ == "aux":
                main_verb = token.head
                if any(c.text.lower() in ("n't", "nt") and c.dep_ == "neg"
                       for c in main_verb.children):
                    is_do_aux = True
            if not is_do_aux:
                continue
            main_verb = token.head
            subjects = [c for c in main_verb.children if c.dep_ in ("nsubj", "nsubjpass")]
            if not subjects:
                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            for child in subjects:
                if child.tag_ == "PRP" and child.lower_ in ("he", "she", "it"):
                    don_t_text = "don't"
                    don_t_end = token.idx + len(token.text)
                    for c in main_verb.children:
                        if c.dep_ == "neg" and c.text.lower() in ("n't", "nt"):
                            don_t_end = c.idx + len(c.text)
                            don_t_text = text[token.idx:c.idx + len(c.text)]
                            break
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=don_t_end,
                        original=don_t_text, replacement="doesn't",
                        category=ErrorCategory.GRAMMAR,
                        rule_id="DO_SUPPORT",
                        message=f'Use "doesn\'t" with third-person singular subjects (he/she/it).',
                        detector="nlp_detector",
                        raw_confidence=0.92,
                    ))
                    break
                if child.tag_ in ("NN", "NNP"):
                    don_t_text = "don't"
                    don_t_end = token.idx + len(token.text)
                    for c in main_verb.children:
                        if c.dep_ == "neg" and c.text.lower() in ("n't", "nt"):
                            don_t_end = c.idx + len(c.text)
                            don_t_text = text[token.idx:c.idx + len(c.text)]
                            break
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=don_t_end,
                        original=don_t_text, replacement="doesn't",
                        category=ErrorCategory.GRAMMAR,
                        rule_id="DO_SUPPORT",
                        message=f'Use "doesn\'t" with singular noun subjects.',
                        detector="nlp_detector",
                        raw_confidence=0.88,
                    ))
                    break
        return candidates

    def _check_have_base_form(self, text: str, doc) -> List[ErrorCandidate]:
        """'have' + base verb → 'have' + past participle: 'have eat' → 'have eaten'."""
        candidates = []
        for token in doc:
            if token.text.lower() not in HAVE_FORMS:
                continue
            if token.dep_ != "aux":
                continue
            main_verb = token.head
            if main_verb.tag_ == "VB" and main_verb.dep_ == "ROOT":
                # Check if the verb is in its base form (not VBN/VBD)
                lemma = main_verb.lemma_
                if lemma in IRREGULAR_VERBS:
                    pp = IRREGULAR_VERBS[lemma].get("pp")
                    if pp and pp != main_verb.text.lower():
                        candidates.append(ErrorCandidate(
                            start=main_verb.idx, end=main_verb.idx + len(main_verb.text),
                            original=main_verb.text, replacement=pp,
                            category=ErrorCategory.TENSE,
                            rule_id="HAVE_BASE_FORM",
                            message=f'After "{token.text}", use the past participle "{pp}" instead of "{main_verb.text}".',
                            detector="nlp_detector",
                            raw_confidence=0.90,
                        ))
                else:
                    # Regular verb: form its past participle (-ed family).
                    base = main_verb.text.lower()
                    if base.endswith("e"):
                        pp = base + "d"
                    elif base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
                        pp = base[:-1] + "ied"
                    elif (len(base) >= 3 and base[-2:] != "en"
                          and base[-3] not in "aeiou"
                          and base[-2] in "aeiou" and base[-1] not in "aeiouwxy"):
                        pp = base + base[-1] + "ed"
                    else:
                        pp = base + "ed"
                    candidates.append(ErrorCandidate(
                        start=main_verb.idx, end=main_verb.idx + len(main_verb.text),
                        original=main_verb.text, replacement=pp,
                        category=ErrorCategory.TENSE,
                        rule_id="HAVE_BASE_FORM",
                        message=f'After "{token.text}", use the past participle "{pp}" instead of "{main_verb.text}".',
                        detector="nlp_detector",
                        raw_confidence=0.90,
                    ))
        return candidates

    def _check_was_were_base(self, text: str, doc) -> List[ErrorCandidate]:
        """'was/were' + base verb → 'was/were' + VBG: 'was study' → 'was studying'."""
        candidates = []
        for token in doc:
            if token.text.lower() not in ("was", "were"):
                continue
            # was/were must be ROOT (not in subordinate clause like "Were I...")
            if token.dep_ not in ("ROOT", "conj"):
                continue
            # Find the main verb dependent on was/were
            for child in token.children:
                if child.tag_ == "VB" and child.dep_ in ("attr", "acomp", "relcl"):
                    base = child.lemma_
                    if base in ("be", "do", "have"):
                        continue
                    if base.endswith("e"):
                        vbg = base[:-1] + "ing"
                    elif len(base) > 2 and base[-1] not in "aeiou" and base[-2] in "aeiou" and base[-1] not in "wxy":
                        vbg = base + base[-1] + "ing"
                    else:
                        vbg = base + "ing"
                    candidates.append(ErrorCandidate(
                        start=child.idx, end=child.idx + len(child.text),
                        original=child.text, replacement=vbg,
                        category=ErrorCategory.TENSE,
                        rule_id="WAS_WERE_BASE_FORM",
                        message=f'After "{token.text}", use the present participle "{vbg}" instead of "{child.text}".',
                        detector="nlp_detector",
                        raw_confidence=0.88,
                    ))
        return candidates

    def _check_double_comparative(self, text: str, doc) -> List[ErrorCandidate]:
        """'more taller' → 'taller', 'most biggest' → 'biggest'."""
        candidates = []
        for token in doc:
            if token.text.lower() in ("more", "most") and token.dep_ == "advmod":
                head = token.head
                if head.pos_ == "ADJ":
                    # Check if adjective already has comparative/superlative suffix
                    adj = head.text.lower()
                    is_comparative = adj.endswith("er") or adj.startswith("more")
                    is_superlative = adj.endswith("est") or adj.startswith("most")
                    if token.text.lower() == "more" and is_comparative:
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=head.idx + len(head.text),
                            original=token.text + " " + head.text,
                            replacement=head.text,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="DOUBLE_COMPARATIVE",
                            message=f'Use just "{head.text}" — do not use "more" with a comparative adjective.',
                            detector="nlp_detector",
                            raw_confidence=0.90,
                        ))
                    elif token.text.lower() == "most" and is_superlative:
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=head.idx + len(head.text),
                            original=token.text + " " + head.text,
                            replacement=head.text,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="DOUBLE_SUPERLATIVE",
                            message=f'Use just "{head.text}" — do not use "most" with a superlative adjective.',
                            detector="nlp_detector",
                            raw_confidence=0.90,
                        ))
        return candidates

    def _check_run_on_sentences(self, text: str, doc) -> List[ErrorCandidate]:
        """Run-on sentences: two independent clauses with no punctuation between them."""
        candidates = []
        for sent in doc.sents:
            tokens = list(sent)
            for i, token in enumerate(tokens):
                if token.pos_ in ("VERB", "AUX") and token.dep_ == "ROOT":
                    before_tokens = tokens[:i]
                    after_tokens = tokens[i+1:]
                    has_punct_before = any(t.pos_ == "PUNCT" for t in before_tokens)
                    has_subj_before = any(t.dep_ in ("nsubj", "nsubjpass") for t in before_tokens)
                    # Only flag if there's a finite verb (not just aux) before ROOT
                    has_finite_verb_before = any(t.pos_ in ("VERB", "AUX") and t.tag_ in ("VBD", "VBP", "VBZ") and t.dep_ not in ("aux", "auxpass") for t in before_tokens)
                    has_subord_before = any(t.dep_ in ("advcl", "mark", "relcl", "acl", "ccomp") or t.tag_ in ("WRB", "WP", "WDT") for t in before_tokens)
                    has_subord_after = any(t.dep_ in ("advcl", "mark") or t.lower_ in ("while", "when", "because", "although", "though", "if", "since", "until", "before", "after", "unless", "whereas") for t in after_tokens)
                    if has_subj_before and has_finite_verb_before and not has_punct_before and not has_subord_before and not has_subord_after:
                        second_clause_start = token.idx
                        prev_text = text[sent.start_char:second_clause_start].rstrip()
                        if prev_text and not prev_text.endswith(('.', '!', '?', ';', ':')):
                            candidates.append(ErrorCandidate(
                                start=second_clause_start - 1, end=second_clause_start,
                                original=" ",
                                replacement=". ",
                                category=ErrorCategory.PUNCTUATION,
                                rule_id="RUN_ON_SENTENCE",
                                message='These appear to be two independent clauses without punctuation. Use a period, semicolon, or conjunction.',
                                detector="nlp_detector",
                                raw_confidence=0.82,
                            ))
                            break
        return candidates

    def _check_double_subject(self, text: str, doc) -> List[ErrorCandidate]:
        """Double subjects: 'My friend he is nice' -> 'My friend is nice'."""
        candidates = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.tag_ == "PRP":
                    head = token.head
                    for child in head.children:
                        if child.dep_ in ("nsubj", "nsubjpass") and child != token and child.pos_ in ("NOUN", "PROPN"):
                            candidates.append(ErrorCandidate(
                                start=token.idx, end=token.idx + len(token.text) + 1,
                                original=token.text + " ",
                                replacement="",
                                category=ErrorCategory.SENTENCE_STRUCTURE,
                                rule_id="DOUBLE_SUBJECT",
                                message=f'Remove the redundant pronoun "{token.text}" -- the subject is already "{child.text}".',
                                detector="nlp_detector", raw_confidence=0.82,
                            ))
                            break
                if token.dep_ in ("nsubj", "nsubjpass") and token.tag_ == "PRP":
                    head = token.head
                    for child in head.children:
                        if child.dep_ == "appos" and child.pos_ in ("NOUN", "PROPN") and child != token:
                            candidates.append(ErrorCandidate(
                                start=child.idx, end=child.idx + len(child.text) + 1,
                                original=child.text + " ",
                                replacement="",
                                category=ErrorCategory.SENTENCE_STRUCTURE,
                                rule_id="DOUBLE_SUBJECT",
                                message=f'Remove the redundant noun "{child.text}" -- the subject is "{token.text}".',
                                detector="nlp_detector", raw_confidence=0.82,
                            ))
                            break
        return candidates

    def _check_double_comparative_adj(self, text: str, doc) -> List[ErrorCandidate]:
        """'faster than you' comparison with wrong target: 'My car is faster than you'."""
        return []

    def _check_possessive_apostrophes(self, text: str, doc) -> List[ErrorCandidate]:
        """Missing possessive apostrophes: 'childrens' → 'children's', 'womens' → 'women's'."""
        candidates = []
        possessive_patterns = {
            "childrens": "children's", "womens": "women's", "mens": "men's",
            "peoples": "people's",
        }
        for m in re.finditer(r'\b(\w+)\b', text):
            word = m.group(1).lower()
            if word in possessive_patterns:
                # Check if it's followed by a noun (possessive context)
                after = text[m.end():m.end() + 20].strip()
                if after and after[0].isalpha() and after[0].islower():
                    candidates.append(ErrorCandidate(
                        start=m.start(), end=m.end(),
                        original=m.group(1), replacement=possessive_patterns[word],
                        category=ErrorCategory.PUNCTUATION,
                        rule_id="POSSESSIVE_APOSTROPHE",
                        message=f'Use an apostrophe to show possession: "{possessive_patterns[word]}".',
                        detector="nlp_detector",
                        raw_confidence=0.85,
                    ))
        return candidates

    def _check_contraction_apostrophes(self, text: str, doc) -> List[ErrorCandidate]:
        """Missing contraction apostrophes: 'Shes' → 'She's', 'Hes' → 'He's', 'Were' → 'We're'."""
        candidates = []
        contractions = {
            "shes": "she's", "hes": "he's", "its": None,  # "its" is valid (possessive)
            "thats": "that's", "whos": "who's", "whats": "what's",
            "heres": "here's", "theres": "there's",
            "youll": "you'll", "theyll": "they'll", "well": None,  # "well" is valid
            "ill": None,  # "I'll" or "ill" (sick) — context needed
            "were": None,  # "were" is valid past tense of "be"
        }
        # Check "were going" → "we're going" (but "were" can be valid)
        for m in re.finditer(r'\b(shes|hes|thats|whos|whats|heres|theres|youll|theyll)\b', text, re.IGNORECASE):
            word = m.group(1).lower()
            if word in contractions and contractions[word]:
                # Check it's at start of sentence or after space
                if m.start() == 0 or text[m.start()-1] == ' ':
                    candidates.append(ErrorCandidate(
                        start=m.start(), end=m.end(),
                        original=m.group(1), replacement=contractions[word],
                        category=ErrorCategory.PUNCTUATION,
                        rule_id="MISSING_CONTRACTION_APOSTROPHE",
                        message=f'Add an apostrophe: "{contractions[word]}" instead of "{m.group(1)}".',
                        detector="nlp_detector",
                        raw_confidence=0.88,
                    ))
        return candidates

    def _check_didnt_past(self, text: str, doc) -> List[ErrorCandidate]:
        """'didn't saw' -> 'didn't see': after do-aux, use base form not past tense."""
        candidates = []
        DO_FORMS = {"did", "do", "does", "didn't", "don't", "doesn't", "didnt", "dont", "doesnt"}
        for token in doc:
            if token.dep_ == "aux" and token.text.lower() in DO_FORMS:
                main_verb = token.head
                if main_verb.tag_ in ("VBD", "VBN"):
                    base = main_verb.lemma_
                    candidates.append(ErrorCandidate(
                        start=main_verb.idx, end=main_verb.idx + len(main_verb.text),
                        original=main_verb.text, replacement=base,
                        category=ErrorCategory.TENSE,
                        rule_id="DIDNT_PAST_FORM",
                        message=f'After "{token.text}", use the base form "{base}" instead of "{main_verb.text}".',
                        detector="nlp_detector",
                        raw_confidence=0.95,
                    ))
        return candidates

    def _check_stative_verbs(self, text: str, doc) -> List[ErrorCandidate]:
        """'I am agree' → 'I agree': stative verbs don't take 'be' auxiliary."""
        candidates = []
        STATIVE_VERBS = {"agree", "believe", "know", "understand", "want", "need",
                         "prefer", "owe", "own", "belong", "consist", "contain",
                         "depend", "deserve", "doubt", "fear", "feel", "forget",
                         "hate", "imagine", "include", "involve", "lack", "like",
                         "love", "mean", "mind", "notice", "owe", "possess",
                         "prefer", "realize", "recognize", "remember", "respect",
                         "seem", "smell", "sound", "suppose", "taste", "think",
                         "understand", "want", "wish"}
        BE_FORMS = {"am", "is", "are", "was", "were", "be", "being", "been"}
        for token in doc:
            if token.text.lower() in BE_FORMS and token.dep_ in ("aux", "auxpass"):
                main_verb = token.head
                if main_verb.lemma_ in STATIVE_VERBS and main_verb.dep_ == "ROOT":
                    # Don't flag continuous tenses: "was feeling", "is thinking", etc.
                    if main_verb.tag_ == "VBG":
                        continue
                    # Don't flag past participles: "is recognized", "was considered" (passive/adjective)
                    if main_verb.tag_ == "VBN":
                        continue
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="",
                        category=ErrorCategory.GRAMMAR,
                        rule_id="STATIVE_BE",
                        message=f'Use "{main_verb.text}" directly — stative verbs like "{main_verb.lemma_}" don\'t take "be" auxiliaries.',
                        detector="nlp_detector",
                        raw_confidence=0.85,
                    ))
        return candidates

    def _check_preposition_collocations_nlp(self, text: str, doc) -> List[ErrorCandidate]:
        """NLP-based preposition collocation checks: 'good in' → 'good at', 'listened the' → 'listened to'."""
        candidates = []
        adj_prep = {
            ("good", "in"): "at", ("bad", "in"): "at",
            ("better", "in"): "at", ("worse", "in"): "at",
            ("excellent", "in"): "at", ("proficient", "in"): "at",
            ("weak", "in"): "at", ("strong", "in"): "at",
        }
        verb_no_prep = {"listened", "listens", "listen", "heard", "hears", "hear",
                        "observed", "observes", "observe"}
        verb_wrong_prep = {
            ("married", "with"): None,
            ("discussed", "about"): "of",
            ("entered", "to"): None,
        }
        for token in doc:
            if token.pos_ == "ADJ" and token.text.lower() in [p[0] for p in adj_prep]:
                # Check both direct children AND siblings (preposition may be child of parent verb)
                prep_candidates = list(token.children)
                if token.head != token:
                    prep_candidates.extend(token.head.children)
                for child in prep_candidates:
                    if child.dep_ == "prep" and child.text.lower() in [p[1] for p in adj_prep]:
                        key = (token.text.lower(), child.text.lower())
                        if key in adj_prep:
                            correct = adj_prep[key]
                            pobj = None
                            for c2 in child.children:
                                if c2.dep_ == "pobj":
                                    pobj = c2
                                    break
                            if pobj:
                                candidates.append(ErrorCandidate(
                                    start=child.idx, end=pobj.idx + len(pobj.text),
                                    original=text[child.idx:pobj.idx + len(pobj.text)],
                                    replacement=f"{correct} {pobj.text}",
                                    category=ErrorCategory.PREPOSITION,
                                    rule_id="ADJ_PREPOSITION",
                                    message=f'Use "{correct}" instead of "{child.text}" after "{token.text}".',
                                    detector="nlp_detector",
                                    raw_confidence=0.85,
                                ))
            # "listened the music" → "listened to the music"
            if token.text.lower() in verb_no_prep and token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ == "dobj":
                        candidates.append(ErrorCandidate(
                            start=token.idx + len(token.text), end=child.idx,
                            original="",
                            replacement=" to",
                            category=ErrorCategory.PREPOSITION,
                            rule_id="MISSING_PREPOSITION",
                            message=f'Use "to" after "{token.text}" — e.g., "{token.text} to {child.text}".',
                            detector="nlp_detector",
                            raw_confidence=0.82,
                        ))
            # "married with her" → "married her" (no preposition)
            if token.text.lower() == "married" and token.dep_ == "ROOT":
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "with":
                        candidates.append(ErrorCandidate(
                            start=child.idx, end=child.idx + len(child.text),
                            original=child.text, replacement="",
                            category=ErrorCategory.PREPOSITION,
                            rule_id="UNNECESSARY_PREPOSITION",
                            message=f'Do not use "with" after "married" — just say "married {child.text}".',
                            detector="nlp_detector",
                            raw_confidence=0.85,
                        ))
        return candidates

    def _check_since_for_duration(self, text: str, doc) -> List[ErrorCandidate]:
        """'since two years' or 'from two years' -> 'for two years': use 'for' with durations."""
        candidates = []
        for m in re.finditer(r'\b(since|from)\s+(two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(years?|months?|weeks?|days?|hours?|minutes?|decades?|centuries?)\b', text, re.IGNORECASE):
            wrong_prep = m.group(1)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=f"for {m.group(2)} {m.group(3)}",
                category=ErrorCategory.PREPOSITION,
                rule_id="SINCE_FOR_DURATION",
                message=f'Use "for" (not "{wrong_prep}") with a duration of time.',
                detector="nlp_detector", raw_confidence=0.92,
            ))
        return candidates

    def _check_one_of_plural(self, text: str, doc) -> List[ErrorCandidate]:
        """'one of my cousin' -> 'one of my cousins'."""
        IRREGULAR_PLURALS = {
            "people", "children", "men", "women", "mice", "geese",
            "teeth", "feet", "lice", "oxen", "phenomena", "criteria",
            "data", "media", "alumni", "fungi", "syllabi", "stimuli",
        }
        candidates = []
        for sent in doc.sents:
            tokens = list(sent)
            for i, t in enumerate(tokens):
                if t.lower_ != "one" or t.i + 2 >= len(doc):
                    continue
                if doc[t.i + 1].lower_ != "of":
                    continue
                of_tok = doc[t.i + 1]
                det_tokens = [c for c in of_tok.children if c.dep_ == "pobj"]
                if not det_tokens:
                    pobj = None
                    for j in range(t.i + 2, min(t.i + 8, len(doc))):
                        if doc[j].dep_ == "pobj" and doc[j].head.lower_ == "of":
                            pobj = doc[j]
                            break
                    if pobj is None:
                        continue
                    noun_tok = pobj
                else:
                    noun_tok = det_tokens[0]
                    if noun_tok.tag_ in ("DT", "JJ", "JJS", "JJR"):
                        for child in noun_tok.children:
                            if child.dep_ in ("det", "amod"):
                                noun_tok = child
                                break
                if noun_tok.tag_ in ("DT", "JJ", "JJS", "JJR"):
                    found = False
                    for j in range(noun_tok.i + 1, min(noun_tok.i + 5, len(doc))):
                        if doc[j].pos_ in ("NOUN", "PROPN") and doc[j].tag_ in ("NN", "NNP"):
                            noun_tok = doc[j]
                            found = True
                            break
                    if not found:
                        continue
                if noun_tok.pos_ not in ("NOUN", "PROPN"):
                    continue
                if noun_tok.tag_ not in ("NN", "NNP"):
                    continue
                noun = noun_tok.text
                lower_noun = noun.lower()
                if lower_noun in IRREGULAR_PLURALS:
                    continue
                if lower_noun in UNCOUNTABLE_NOUNS:
                    continue
                plural = noun + "s"
                candidates.append(ErrorCandidate(
                    start=noun_tok.idx, end=noun_tok.idx + len(noun),
                    original=noun, replacement=plural,
                    category=ErrorCategory.GRAMMAR,
                    rule_id="ONE_OF_PLURAL",
                    message=f'After "one of", use the plural form "{plural}".',
                    detector="nlp_detector", raw_confidence=0.90,
                ))
        return candidates

    def _check_article_missing(self, text: str, doc) -> List[ErrorCandidate]:
        """'He is teacher' -> 'He is a teacher'."""
        INDEFINITE_PRONOUNS = {
            "nothing", "something", "everything", "anything",
            "nobody", "somebody", "everybody", "anybody",
            "no one", "someone", "everyone", "anyone",
            "none", "neither", "either", "each", "every",
        }
        candidates = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ not in ("attr", "oprd"):
                    continue
                if token.tag_ not in ("NN", "NNP"):
                    continue
                if token.lower_ in UNCOUNTABLE_NOUNS | {"people", "children", "police"}:
                    continue
                if token.lower_ in INDEFINITE_PRONOUNS:
                    continue
                has_det = any(c.dep_ == "det" for c in token.children)
                if has_det:
                    continue
                has_poss = any(c.dep_ == "poss" for c in token.children)
                if has_poss:
                    continue
                has_adj = any(c.dep_ in ("amod", "compound") for c in token.children)
                if not has_adj:
                    continue
                words_before = text[:token.idx].rstrip().split()
                if words_before and words_before[-1].lower() in (
                    "a", "an", "the", "its", "his", "her", "my", "your", "our", "their",
                ):
                    continue
                if words_before and words_before[-1].endswith("'s"):
                    continue
                prev_tok = token.nbor(-1) if token.i > 0 else None
                if prev_tok and prev_tok.dep_ == "det":
                    continue
                if prev_tok and prev_tok.dep_ == "poss":
                    continue
                if words_before and len(words_before) >= 2 and words_before[-2].lower() in (
                    "there", "it",
                ) and words_before[-1].lower() in ("is", "was", "are", "were", "has", "had"):
                    continue
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx + len(token.text),
                    original=token.text, replacement=f"a {token.text}",
                    category=ErrorCategory.ARTICLE,
                    rule_id="MISSING_ARTICLE",
                    message=f'Add "a" before "{token.text}".',
                    detector="nlp_detector", raw_confidence=0.82,
                ))
        return candidates

    def _check_article_a_an(self, text: str, doc) -> List[ErrorCandidate]:
        """'a MBA' -> 'an MBA', 'a apple' -> 'an apple'."""
        candidates = []
        for m in re.finditer(r'\b(a)\s+([aeiou]\w*)', text, re.IGNORECASE):
            word = m.group(2)
            if word.lower() in ("university", "uniform", "united", "unique", "useful", "user", "union", "unit", "one", "once", "european", "euro", "unanimous", "unicorn", "universal", "universe", "urban", "usage", "usual", "usually", "utensil", "utility", "eulogy", "euphemism", "euphoria"):
                continue
            if word[0].lower() in 'aeiou':
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0), replacement=f"an {word}",
                    category=ErrorCategory.ARTICLE,
                    rule_id="A_AN",
                    message=f'Use "an" before vowel sounds.',
                    detector="nlp_detector", raw_confidence=0.92,
                ))
        for m in re.finditer(r'\b(an)\s+([^aeiouAEIOU]\w*)', text, re.IGNORECASE):
            word = m.group(2)
            if word.lower() in ("hour", "honest", "honor", "honour", "heir", "herb", "umbrella", "mp3", "mba", "nba", "fbi", "cia"):
                continue
            if word[0].lower() not in 'aeiou':
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(0), replacement=f"a {word}",
                    category=ErrorCategory.ARTICLE,
                    rule_id="A_AN",
                    message=f'Use "a" before consonant sounds.',
                    detector="nlp_detector", raw_confidence=0.92,
                ))
        for m in re.finditer(r'\b(a)\s+((?:M|N|F|S|X|L|H|U|A|R|B|C|D|G|J|K|P|Q|T|V|W|Y|Z)(?:BA|BA|MBA|PhD|Dd))\b', text):
            word = m.group(2)
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement=f"an {word}",
                category=ErrorCategory.ARTICLE,
                rule_id="A_AN",
                message=f'Use "an" before the sound of "{word[0]}" (vowel sound).',
                detector="nlp_detector", raw_confidence=0.92,
            ))
        return candidates

    def _check_tense_consistency(self, text: str, doc) -> List[ErrorCandidate]:
        """'walked...buys' -> 'walked...bought': mixed tenses in coordinated clauses."""
        candidates = []
        for sent in doc.sents:
            tokens = list(sent)
            has_past_time = any(t.lower_ in ("yesterday", "ago", "earlier") or
                               (t.dep_ == "npadvmod" and t.tag_ == "RB") for t in tokens)

            past_verbs = []
            present_verbs = []
            for t in tokens:
                if t.dep_ in ("ROOT", "conj") and t.pos_ in ("VERB", "AUX"):
                    if t.tag_ == "VBD" and t.text.lower() not in ("was", "were"):
                        past_verbs.append(t)
                    elif t.tag_ in ("VBZ", "VBP") and t.text.lower() not in ("am", "is", "are", "do", "does", "did"):
                        present_verbs.append(t)

            if has_past_time and present_verbs:
                for pv in present_verbs:
                    lemma = pv.lemma_
                    if lemma in IRREGULAR_VERBS and IRREGULAR_VERBS[lemma].get("past"):
                        replacement = IRREGULAR_VERBS[lemma]["past"]
                    elif lemma == "be":
                        replacement = "was"
                    elif pv.tag_ in ("VBZ", "VBP"):
                        replacement = _regular_past_tense(lemma)
                    else:
                        replacement = pv.text
                    if replacement != pv.text:
                        candidates.append(ErrorCandidate(
                            start=pv.idx, end=pv.idx + len(pv.text),
                            original=pv.text, replacement=replacement,
                            category=ErrorCategory.TENSE,
                            rule_id="TENSE_CONSISTENCY",
                            message=f'Use past tense "{replacement}" for consistency.',
                            detector="nlp_detector", raw_confidence=0.88,
                        ))

            # "Yesterday, our sales team has attended ..." -> present perfect with an
            # unambiguous completed-past marker is a tense conflict -> simple past.
            if has_past_time:
                _perf_marker = re.compile(
                    r"\b(yesterday|ago)\b"
                    r"|\blast\s+(week|month|year|time|night|day|monday|tuesday|wednesday|thursday|friday|saturday|sunday|summer|winter|spring|fall|autumn|decade|century|quarter|semester)\b"
                    r"|\b(in|on)\s+(18|19|20)\d{2}\b",
                    re.IGNORECASE,
                )
                sent_text = sent.text
                _m = _perf_marker.search(sent_text)
                if _m:
                    # "…since yesterday / since last week" starts an open period and
                    # is fine with present perfect — not a conflict.
                    pre = sent_text[max(0, _m.start() - 16):_m.start()].lower()
                    if "since" not in re.split(r"[^a-z']", pre):
                        for t in tokens:
                            if t.text.lower() in ("has", "have") and t.dep_ == "aux":
                                head = t.head
                                if head.tag_ != "VBN" or head.lemma_ in ("be", "have", "do"):
                                    continue
                                lemma = head.lemma_
                                past = IRREGULAR_VERBS.get(lemma, {}).get("past")
                                if not past:
                                    past = _regular_past_tense(lemma)
                                span_end = head.idx + len(head.text)
                                if past:
                                    candidates.append(ErrorCandidate(
                                        start=t.idx, end=span_end,
                                        original=sent_text[t.idx - sent.start_char:span_end - sent.start_char],
                                        replacement=past,
                                        category=ErrorCategory.TENSE,
                                        rule_id="TENSE_CONSISTENCY",
                                        message=f'With the past-time marker, use simple past "{past}".',
                                        detector="nlp_detector", raw_confidence=0.88,
                                    ))

            if past_verbs and present_verbs and not has_past_time:
                for pv in present_verbs:
                    for pt in past_verbs:
                        if abs(pv.i - pt.i) <= 5:
                            lemma = pv.lemma_
                            if lemma in IRREGULAR_VERBS and IRREGULAR_VERBS[lemma].get("past"):
                                replacement = IRREGULAR_VERBS[lemma]["past"]
                            elif lemma == "be":
                                replacement = "was"
                            elif pv.tag_ in ("VBZ", "VBP"):
                                replacement = _regular_past_tense(lemma)
                            else:
                                replacement = pv.text
                            if replacement != pv.text:
                                candidates.append(ErrorCandidate(
                                    start=pv.idx, end=pv.idx + len(pv.text),
                                    original=pv.text, replacement=replacement,
                                    category=ErrorCategory.TENSE,
                                    rule_id="TENSE_CONSISTENCY",
                                    message=f'Use past tense "{replacement}" for consistency with "{pt.text}".',
                                    detector="nlp_detector", raw_confidence=0.85,
                                ))

        # "works hardly" → "works hard": "hardly" means "barely", not "with effort"
        for token in doc:
            if token.text.lower() == "hardly" and token.tag_ == "RB":
                if token.i == 0 or (token.i == 1 and doc[0].text in (".", "!", "?", '"', "'")):
                    continue
                head = token.head
                if head.pos_ == "VERB":
                    candidates.append(ErrorCandidate(
                        start=token.idx, end=token.idx + len(token.text),
                        original=token.text, replacement="hard",
                        category=ErrorCategory.WORD_USAGE,
                        rule_id="HARDLY_HARD",
                        message=f'Use "hard" (adverb meaning "with effort") instead of "hardly" (which means "barely").',
                        detector="nlp_detector", raw_confidence=0.85,
                    ))

        return candidates
    def _check_run_on_no_punct(self, text: str, doc) -> List[ErrorCandidate]:
        """'The cat sat on the mat it was comfortable' -> add period.
        
        Only fires on truly independent clauses without punctuation.
        Never fires on relative clauses (relcl) or after subordinating markers.
        """
        candidates = []
        SUBORD_MARKERS = {
            "that", "which", "who", "whom", "whose", "where", "when",
            "what", "why", "how", "whether", "if", "because", "since",
            "although", "though", "while", "whereas", "unless", "until",
            "before", "after", "as", "once", "than", "lest",
            "whatever", "however", "wherever", "whenever", "whoever",
            "whichever", "whereby", "wherein",
        }
        CONJUNCTIONS = {
            "and", "but", "or", "nor", "yet", "so", "for",
            "however", "therefore", "moreover", "furthermore", "nevertheless",
        }
        for sent in doc.sents:
            tokens = list(sent)
            for i, token in enumerate(tokens):
                if token.dep_ not in ("ccomp", "advcl") or token.pos_ not in ("VERB",):
                    continue
                if token.dep_ in ("ccomp", "advcl") and token.tag_ in ("VBG", "VBN"):
                    continue
                if token.dep_ == "advcl" and any(t.dep_ == "aux" and t.lower_ == "to" for t in token.children):
                    continue
                CAUSATIVES = {"let", "make", "made", "have", "had", "get", "got"}
                if token.dep_ == "ccomp" and token.head.lower_ in CAUSATIVES:
                    continue
                if token.i == sent.start:
                    continue
                prev_token = tokens[i - 1] if i > 0 else None
                if prev_token is None:
                    continue
                if prev_token.dep_ == "cc":
                    continue
                if prev_token.lower_ in CONJUNCTIONS:
                    continue
                punct_before = text[prev_token.idx + len(prev_token.text):token.idx].strip()
                if punct_before in ('.', ',', ';', ':', '—', '–'):
                    continue
                if any(t.pos_ == "PUNCT" and t.i > token.i for t in tokens):
                    continue
                if token.lower_ in SUBORD_MARKERS:
                    continue
                if any(t.dep_ == "mark" and t.lower_ in SUBORD_MARKERS for t in token.children):
                    continue
                has_subord = any(
                    t.dep_ == "mark" and t.i < token.i
                    for t in tokens
                )
                if has_subord:
                    continue
                token_subtree_words = {t.lower_ for t in token.subtree}
                if token_subtree_words & SUBORD_MARKERS:
                    continue
                prev_left = tokens[i - 2] if i >= 2 else None
                if prev_left is not None and prev_left.lower_ in CONJUNCTIONS:
                    continue
                candidates.append(ErrorCandidate(
                    start=prev_token.idx + len(prev_token.text),
                    end=token.idx,
                    original=" ",
                    replacement=". ",
                    category=ErrorCategory.PUNCTUATION,
                    rule_id="RUN_ON_NO_PUNCT",
                    message=f'Possible run-on. Add a period before "{token.text}".',
                    detector="nlp_detector", raw_confidence=0.80,
                ))
                break
        return candidates

    def _check_fragment(self, text: str, doc) -> List[ErrorCandidate]:
        """'Running through the park.' -> 'I was running through the park.'"""
        candidates = []
        for sent in doc.sents:
            tokens = list(sent)
            if len(tokens) < 2:
                continue
            root = next((t for t in tokens if t.dep_ == "ROOT"), None)
            if root is None:
                continue
            if root.tag_ == "VBG" and root.dep_ == "ROOT":
                has_aux = any(t.dep_ in ("aux", "auxpass") for t in root.children)
                has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in root.children)
                if not has_aux and not has_subj:
                    candidates.append(ErrorCandidate(
                        start=sent.start_char, end=sent.end_char,
                        original=sent.text, replacement=f"I was {sent.text[0].lower()}{sent.text[1:]}",
                        category=ErrorCategory.SENTENCE_STRUCTURE,
                        rule_id="FRAGMENT",
                        message="Possible sentence fragment. Add a subject and auxiliary verb.",
                        detector="nlp_detector", raw_confidence=0.75,
                    ))
        return candidates

    def _check_parallelism(self, text: str, doc) -> List[ErrorCandidate]:
        """'to swim, running, and to bike' -> 'to swim, to run, and to bike'."""
        candidates = []
        for sent in doc.sents:
            tokens = list(sent)
            infinitives = []
            gerunds = []
            PHRASAL_GET = {"going", "done", "started", "running", "moving", "lost", "caught", "stuck"}
            for t in tokens:
                if t.tag_ == "VBG" and t.dep_ in ("conj", "appos", "xcomp"):
                    if t.dep_ == "xcomp" and t.head.lower_ == "get" and t.lower_ in PHRASAL_GET:
                        continue
                    gerunds.append(t)
                if t.text.lower() == "to" and t.i + 1 < len(doc):
                    nxt = doc[t.i + 1]
                    if nxt.tag_ == "VB":
                        infinitives.append(nxt)
            if infinitives and gerunds:
                for g in gerunds:
                    replacement = g.lemma_ if g.lemma_ else g.text
                    candidates.append(ErrorCandidate(
                        start=g.idx, end=g.idx + len(g.text),
                        original=g.text, replacement=f"to {replacement}",
                        category=ErrorCategory.GRAMMAR,
                        rule_id="PARALLELISM",
                        message=f'Keep the list parallel. Use "to {replacement}".',
                        detector="nlp_detector", raw_confidence=0.85,
                    ))
        return candidates

    def _check_object_pronoun(self, text: str, doc) -> List[ErrorCandidate]:
        """'him and I' -> 'him and me'."""
        candidates = []
        for m in re.finditer(r'\b(to|for|with|between|among|like)\s+(\w+)\s+and\s+(I|he|she|we|they)\b', text, re.IGNORECASE):
            pronoun = m.group(3)
            pronoun_map = {"I": "me", "he": "him", "she": "her", "we": "us", "they": "them"}
            if pronoun in pronoun_map:
                i_pos = text.find(pronoun, m.start())
                if i_pos >= 0:
                    candidates.append(ErrorCandidate(
                        start=i_pos, end=i_pos + len(pronoun),
                        original=pronoun, replacement=pronoun_map[pronoun],
                        category=ErrorCategory.PRONOUNS,
                        rule_id="OBJECT_PRONOUN",
                        message=f'Use the object form "{pronoun_map[pronoun]}" after "{m.group(1)}".',
                        detector="nlp_detector", raw_confidence=0.88,
                    ))
        return candidates

    def _check_possessive_its(self, text: str, doc) -> List[ErrorCandidate]:
        """'it's' meaning possessive -> 'its'."""
        candidates = []
        for m in re.finditer(r"\bit's\b", text, re.IGNORECASE):
            after = text[m.end():].lstrip()
            next_word_match = re.match(r'(\w+)', after)
            if next_word_match:
                nw = next_word_match.group(1).lower()
                if nw in ("own", "time", "true", "clear", "possible", "not", "a", "an", "the",
                          "been", "going", "just", "also", "always", "never", "only"):
                    continue
            before = text[:m.start()].rstrip()
            prev_word_match = re.search(r'(\w+)\s*$', before)
            if prev_word_match:
                pw = prev_word_match.group(1).lower()
                if pw in ("is", "was", "were", "has", "had", "will", "would", "could", "should",
                          "might", "can", "may", "shall", "must", "that", "this", "what", "which", "who"):
                    continue
            candidates.append(ErrorCandidate(
                start=m.start(), end=m.end(),
                original=m.group(0), replacement="its",
                category=ErrorCategory.GRAMMAR,
                rule_id="POSSESSIVE_ITS",
                message=f'Use "its" (possessive) instead of "it\'s" (contraction of "it is").',
                detector="nlp_detector", raw_confidence=0.90,
            ))
        return candidates

    def _check_missing_auxiliary(self, text: str, doc) -> List[ErrorCandidate]:
        """'She going to sell it' -> 'She is going to sell it': VBG without auxiliary.
        
        Checks ROOT, ccomp, and conj VBGs — any finite-clause position.
        """
        candidates = []
        BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
        CONTRACTION_AUX = {"'m", "'re", "'s", "n't"}
        for sent in doc.sents:
            for token in sent:
                if token.tag_ != "VBG":
                    continue
                if token.dep_ not in ("ROOT", "ccomp", "conj", "advcl"):
                    continue
                has_aux = any(t.dep_ in ("aux", "auxpass") and t.lower_ in BE_FORMS for t in token.children)
                if has_aux:
                    continue
                has_subj = any(t.dep_ in ("nsubj", "nsubjpass") for t in token.children)
                if not has_subj:
                    continue
                subj = next((t for t in token.children if t.dep_ in ("nsubj", "nsubjpass")), None)
                if subj is None or subj.tag_ != "PRP":
                    continue
                prev_token = doc[subj.i - 1] if subj.i > 0 else None
                if prev_token and prev_token.text in CONTRACTION_AUX:
                    continue
                for child in token.children:
                    if child.dep_ == "aux" and child.pos_ == "AUX":
                        has_aux = True
                        break
                if has_aux:
                    continue
                if subj.lower_ == "i":
                    aux = "am"
                elif subj.lower_ in ("he", "she", "it"):
                    aux = "is"
                else:
                    aux = "are"
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx,
                    original="", replacement=f"{aux} ",
                    category=ErrorCategory.GRAMMAR,
                    rule_id="MISSING_AUXILIARY",
                    message=f'Add "{aux}" before "{token.text}" (progressive tense).',
                    detector="nlp_detector", raw_confidence=0.88,
                ))
        return candidates

    def _check_missing_plural(self, text: str, doc) -> List[ErrorCandidate]:
        """'many problem' -> 'many problems': plural quantifier + singular noun."""
        candidates = []
        PLURAL_QUANTIFIERS = {"many", "several", "few", "numerous", "various", "multiple",
                              "both", "few", "these", "those"}
        COMPOUND_NOUNS = {
            "student loan", "credit card", "phone number", "book club",
            "car park", "fire station", "police officer", "post office",
            "washing machine", "coffee table", "dinner table", "parking lot",
        }
        for sent in doc.sents:
            tokens = list(sent)
            for token in sent:
                if token.tag_ == "NN" and token.pos_ == "NOUN":
                    for child in token.children:
                        if child.lower_ in PLURAL_QUANTIFIERS and child.dep_ in ("det", "amod"):
                            has_article_after = False
                            for t in tokens:
                                if t.i > child.i and t.i < token.i and t.lower_ in ("a", "an"):
                                    has_article_after = True
                                    break
                            if has_article_after:
                                continue
                            noun = token.text
                            if noun.lower() in UNCOUNTABLE_NOUNS:
                                continue
                            if noun.lower() in ("sheep", "fish", "deer", "moose", "species", "series"):
                                continue
                            next_tok = doc[token.i + 1] if token.i + 1 < len(doc) else None
                            if next_tok and next_tok.pos_ in ("NOUN", "PROPN"):
                                continue
                            plural = _pluralize(noun.lower())
                            end = token.idx + len(token.text)
                            possess = None
                            if next_tok and next_tok.text.strip() == "'s":
                                possess = "'s"
                                end = next_tok.idx + len(next_tok.text)
                            candidates.append(ErrorCandidate(
                                start=token.idx, end=end,
                                original=noun if not possess else noun + possess,
                                replacement=plural,
                                category=ErrorCategory.GRAMMAR,
                                rule_id="MISSING_PLURAL",
                                message=f'Use the plural form "{plural}" with "{child.text}".',
                                detector="nlp_detector", raw_confidence=0.88,
                            ))
                            break
        return candidates

    def _check_noun_possessive(self, text: str, doc) -> List[ErrorCandidate]:
        """'the brother number' -> 'the brother's number': missing possessive.
        
        Very conservative: only fires on clear family/person + noun patterns.
        """
        candidates = []
        PEOPLE_NOUNS = {"brother", "sister", "mother", "father", "mom", "dad",
                        "husband", "wife", "son", "daughter", "uncle", "aunt",
                        "cousin", "friend", "neighbor", "teacher", "doctor",
                        "boss", "manager", "director", "president", "king",
                        "queen", "prince", "princess", "hero", "villain",
                        "student", "worker", "player", "driver", "farmer"}
        for sent in doc.sents:
            tokens = list(sent)
            for i in range(len(tokens) - 1):
                t1 = tokens[i]
                t2 = tokens[i + 1]
                if t1.pos_ in ("NOUN", "PROPN") and t1.tag_ in ("NN", "NNP"):
                    if t2.pos_ in ("NOUN", "PROPN") and t2.tag_ in ("NN", "NNP"):
                        if t1.dep_ in ("poss",) or t2.dep_ in ("compound",):
                            continue
                        if t1.lower_ not in PEOPLE_NOUNS:
                            continue
                        before = text[:t1.idx].rstrip().split()
                        if not before:
                            continue
                        last_word = before[-1].lower()
                        if last_word not in ("the", "a", "an", "my", "his", "her", "its", "your", "our", "their"):
                            continue
                        candidates.append(ErrorCandidate(
                            start=t1.idx + len(t1.text),
                            end=t1.idx + len(t1.text),
                            original="", replacement="'s ",
                            category=ErrorCategory.PUNCTUATION,
                            rule_id="POSSESSIVE_NOUN",
                            message=f'Add an apostrophe: "{t1.text}\'s".',
                            detector="nlp_detector", raw_confidence=0.75,
                        ))
        return candidates

    def _check_uncountable_plural(self, text: str, doc) -> List[ErrorCandidate]:
        """'feedbacks' -> 'feedback': uncountable nouns that shouldn't be pluralized."""
        candidates = []
        UNCOUNTABLE_PLURAL = {
            "informations": "information", "feedbacks": "feedback", "advices": "advice",
            "furnitures": "furniture", "luggages": "luggage", "equipments": "equipment",
            "homeworks": "homework", "researches": "research", "evidences": "evidence",
            "knowledges": "knowledge",
        }
        for token in doc:
            lower = token.text.lower()
            if lower in UNCOUNTABLE_PLURAL and token.tag_ == "NNS":
                singular = UNCOUNTABLE_PLURAL[lower]
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx + len(token.text),
                    original=token.text, replacement=singular,
                    category=ErrorCategory.GRAMMAR,
                    rule_id="UNCOUNTABLE_PLURAL",
                    message=f'"{token.text}" is uncountable. Use "{singular}" instead.',
                    detector="nlp_detector", raw_confidence=0.92,
                ))
                # Fix the verb too: "informations are" → "information is".
                # spaCy links the plural noun to its predicate as head.
                verb = token.head
                V_SINGULAR_MAP = {"are": "is", "were": "was",
                                  "have": "has", "do": "does"}
                if verb.pos_ in ("VERB", "AUX"):
                    verb_lower = verb.text.lower()
                    if verb_lower in V_SINGULAR_MAP:
                        candidates.append(ErrorCandidate(
                            start=verb.idx, end=verb.idx + len(verb.text),
                            original=verb.text, replacement=V_SINGULAR_MAP[verb_lower],
                            category=ErrorCategory.GRAMMAR,
                            rule_id="SVA",
                            message=f'After "{singular}", use the singular verb "{V_SINGULAR_MAP[verb_lower]}" instead of "{verb.text}".',
                            detector="nlp_detector", raw_confidence=0.93,
                        ))
                    elif verb.pos_ == "VERB":
                        # "The furnitures were sold" — the finite verb is an aux
                        # of the head verb ("were" heads the passive "sold").
                        for aux in verb.children:
                            if aux.dep_ in ("aux", "auxpass") and aux.text.lower() in V_SINGULAR_MAP:
                                candidates.append(ErrorCandidate(
                                    start=aux.idx, end=aux.idx + len(aux.text),
                                    original=aux.text, replacement=V_SINGULAR_MAP[aux.text.lower()],
                                    category=ErrorCategory.GRAMMAR,
                                    rule_id="SVA",
                                    message=f'After "{singular}", use the singular verb "{V_SINGULAR_MAP[aux.text.lower()]}" instead of "{aux.text}".',
                                    detector="nlp_detector", raw_confidence=0.93,
                                ))
                                break
        return candidates

    def _check_every_singular(self, text: str, doc) -> List[ErrorCandidate]:
        """'every students' -> 'every student': every + singular noun."""
        candidates = []
        EVERY_WORDS = {"every", "each", "either", "neither"}
        for token in doc:
            if token.text.lower() in EVERY_WORDS and token.tag_ == "DT":
                for child in token.children:
                    if child.dep_ == "det" and child == token:
                        continue
                next_tok = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if next_tok and next_tok.tag_ == "NNS" and next_tok.pos_ == "NOUN":
                    singular = next_tok.lemma_
                    if singular and singular != next_tok.text.lower():
                        candidates.append(ErrorCandidate(
                            start=next_tok.idx, end=next_tok.idx + len(next_tok.text),
                            original=next_tok.text, replacement=singular,
                            category=ErrorCategory.GRAMMAR,
                            rule_id="EVERY_SINGULAR",
                            message=f'After "{token.text}", use the singular form "{singular}" instead of "{next_tok.text}".',
                            detector="nlp_detector", raw_confidence=0.90,
                        ))
        return candidates

    def _check_prefer_than(self, text: str, doc) -> List[ErrorCandidate]:
        """'prefer X than Y' -> 'prefer X to Y'"""
        candidates = []
        lower = text.lower()
        idx = 0
        while True:
            pos = lower.find("prefer ", idx)
            if pos == -1:
                break
            rest = lower[pos + 7:]
            # Look for "than" within 50 chars of "prefer"
            than_pos = rest.find("than")
            if than_pos != -1 and than_pos < 50:
                start = pos + 7 + than_pos
                candidates.append(ErrorCandidate(
                    start=start, end=start + 4,
                    original=text[start:start+4],
                    replacement="to",
                    category=ErrorCategory.PREPOSITION,
                    rule_id="PREFER_THAN",
                    message="'prefer' takes 'to' not 'than'. Use 'prefer X to Y'.",
                    raw_confidence=0.92, detector="nlp_detector"
                ))
            idx = pos + 7
        return candidates

    def _check_explain_dative(self, text: str, doc) -> List[ErrorCandidate]:
        """'explain/suggest + person' -> 'explain to + person'
        spaCy may parse 'me' as dative or dobj."""
        candidates = []
        for token in doc:
            if token.lemma_ in {"explain", "suggest", "report", "announce", "describe"}:
                if token.dep_ in ("ROOT", "relcl", "ccomp", "advcl", "conj"):
                    children = list(token.children)
                    has_dobj = any(c.dep_ in ("dobj", "dative") for c in children)
                    has_prep_to = any(c.dep_ == "prep" and c.lower_ == "to" for c in children)
                    if has_dobj and not has_prep_to:
                        dobj = [c for c in children if c.dep_ in ("dobj", "dative")][0]
                        if dobj.pos_ == "PRON" and dobj.tag_ == "PRP":
                            prep_start = token.idx + len(token.text)
                            candidates.append(ErrorCandidate(
                                start=prep_start, end=prep_start,
                                original="",
                                replacement="to ",
                                category=ErrorCategory.PREPOSITION,
                                rule_id="EXPLAIN_DATIVE",
                                message=f"'{token.text}' requires 'to' before the indirect object. Use '{token.text} to {dobj.text}'.",
                                raw_confidence=0.88, detector="nlp_detector"
                            ))
        return candidates

    def _check_irregular_plurals(self, text: str, doc) -> List[ErrorCandidate]:
        """'peoples' -> 'people', 'childs' -> 'children', etc."""
        candidates = []
        IRREGULAR_PLURALS = {
            "peoples": "people", "childs": "children", "oxs": "oxen",
            "mouses": "mice", "die": "dice", "personses": "persons",
            "fishs": "fish", "sheeps": "sheep", "deers": "deer",
        }
        for token in doc:
            lower = token.text.lower()
            if lower in IRREGULAR_PLURALS and token.tag_ == "NNS":
                correct = IRREGULAR_PLURALS[lower]
                candidates.append(ErrorCandidate(
                    start=token.idx, end=token.idx + len(token.text),
                    original=token.text, replacement=correct,
                    category=ErrorCategory.GRAMMAR,
                    rule_id="IRREGULAR_PLURAL",
                    message=f'"{token.text}" is not a standard plural. Use "{correct}" instead.',
                    detector="nlp_detector", raw_confidence=0.95,
                ))
        return candidates

    def _check_mass_noun_sva(self, text: str, doc) -> List[ErrorCandidate]:
        """'content are' -> 'content is': mass nouns need singular verb."""
        candidates = []
        MASS_NOUNS = {
            "content", "equipment", "furniture", "luggage", "advice",
            "information", "feedback", "research", "homework", "evidence",
            "knowledge", "music", "traffic", "weather", "software", "hardware",
            "money", "bread", "rice", "water", "milk", "sugar", "salt",
            "gold", "silver", "paper", "plastic", "wood", "glass", "cloth",
            "cotton", "leather", "rubber", "steel", "iron", "coal", "oil",
            "gas", "electricity", "power", "energy", "light", "heat",
            "noise", "pollution", "poverty", "wealth", "happiness",
            "sadness", "anger", "fear", "love", "hate", "respect",
            "trust", "faith", "hope", "luck", "courage", "patience",
            "anger", "pride", "shame", "guilt", "surprise", "joy",
            "travel", "work", "study", "play", "rest", "sleep",
        }
        for token in doc:
            if token.dep_ not in ("nsubj", "nsubjpass"):
                continue
            subject = token
            lower = subject.text.lower()
            if lower not in MASS_NOUNS:
                continue
            main_verb = subject.head
            if main_verb.pos_ not in ("VERB", "AUX"):
                continue
            if main_verb.tag_ in ("VBP", "VBZ"):
                if main_verb.tag_ == "VBP":
                    # Check if it's a "be" verb that should be "is"
                    if main_verb.text.lower() == "are":
                        candidates.append(ErrorCandidate(
                            start=main_verb.idx, end=main_verb.idx + len(main_verb.text),
                            original=main_verb.text, replacement="is",
                            category=ErrorCategory.SUBJECT_VERB_AGREEMENT,
                            rule_id="MASS_NOUN_SVA",
                            message=f'"{lower}" is a mass noun and takes a singular verb. Use "is" instead of "are".',
                            detector="nlp_detector", raw_confidence=0.92,
                        ))
        return candidates

    def _check_missing_article_phrases(self, text: str, doc) -> List[ErrorCandidate]:
        """'in future' -> 'in the future', 'for few weeks' -> 'for a few weeks'."""
        candidates = []
        lower = text.lower()
        # "in future" -> "in the future" (British vs American, but standard in most contexts)
        idx = 0
        while True:
            pos = lower.find("in future", idx)
            if pos == -1:
                break
            # Make sure it's a standalone phrase (not "in futuristic" etc.)
            after = pos + 9
            if after < len(lower) and lower[after].isalpha():
                idx = pos + 1
                continue
            before = pos - 1
            if before >= 0 and lower[before].isalpha():
                idx = pos + 1
                continue
            candidates.append(ErrorCandidate(
                start=pos, end=after,
                original="in future", replacement="in the future",
                category=ErrorCategory.ARTICLES,
                rule_id="MISSING_ARTICLE",
                message='Use "in the future" instead of "in future".',
                detector="nlp_detector", raw_confidence=0.85,
            ))
            idx = pos + 1

        # "for few" -> "for a few" (missing article before few)
        idx = 0
        while True:
            pos = lower.find("for few", idx)
            if pos == -1:
                break
            after = pos + 7
            if after < len(lower) and lower[after].isalpha():
                idx = pos + 1
                continue
            before = pos - 1
            if before >= 0 and lower[before].isalpha():
                idx = pos + 1
                continue
            candidates.append(ErrorCandidate(
                start=pos, end=after,
                original="for few", replacement="for a few",
                category=ErrorCategory.ARTICLES,
                rule_id="MISSING_ARTICLE",
                message='Use "for a few" instead of "for few".',
                detector="nlp_detector", raw_confidence=0.88,
            ))
            idx = pos + 1

        return candidates

    def _check_possessive_plurals(self, text: str, doc) -> List[ErrorCandidate]:
        """'grandparents house' -> 'grandparents\\' house': missing possessive apostrophe on plural nouns."""
        candidates = []
        # Words ending in -s that are mass nouns (not plural possessives)
        MASS_NOUNS_ENDING_S = {
            "ethics", "physics", "mathematics", "linguistics", "economics",
            "politics", "statistics", "genetics", "electronics", "mechanics",
            "dynamics", "acoustics", "aeronautics", "analytics", "tactics",
            "logistics", "tropics", "classics", "odds", "goods", "remains",
            "Proceedings", "Times", "News", "series", "species", "means",
        }
        for token in doc:
            if token.tag_ == "NNS" and token.pos_ == "NOUN":
                lower = token.text.lower()
                if lower in MASS_NOUNS_ENDING_S:
                    continue
                # Check if the next token is a noun
                next_idx = token.i + 1
                if next_idx >= len(doc):
                    continue
                next_tok = doc[next_idx]
                if next_tok.pos_ != "NOUN":
                    continue
                # Skip if the next token has a determiner (e.g., "competitors the websites" = wrong parse)
                if any(c.dep_ == "det" for c in next_tok.children):
                    continue
                # Skip if dep_ is compound (modifier, not possessive)
                if next_tok.dep_ == "compound":
                    continue
                # Only flag if relationship looks possessive
                if next_tok.dep_ in ("poss",) or (next_tok.head == token and token.dep_ not in ("compound", "amod")):
                    if not token.text.endswith("'") and not token.text.endswith("'s"):
                        candidates.append(ErrorCandidate(
                            start=token.idx + len(token.text) - 1,
                            end=token.idx + len(token.text),
                            original=token.text[-1],
                            replacement="'",
                            category=ErrorCategory.PUNCTUATION,
                            rule_id="POSSESSIVE_PLURAL",
                            message=f'Add an apostrophe: "{token.text}\'" (plural possessive).',
                            detector="nlp_detector", raw_confidence=0.80,
                        ))
        return candidates

# ══════════════════════════════════════════════════════════════════════════
#  DATA-DRIVEN DETECTOR — Loads patterns from all JSON datasets
# ══════════════════════════════════════════════════════════════════════════

class DataDrivenDetector:
    """Detects errors using pattern lookup from all project JSON datasets.
    
    Loads:
    - data/common_mistakes.json (~130 grammar patterns + 51 spelling)
    - data/grammar_errors.json (~95 error entries)
    - data/confusing_words.json (22 words with context rules)
    - data/spelling_dictionary.json (~400 misspellings)
    """

    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.exact_phrases = {}       # lowercased phrase -> (correct, category, rule_id, message)
        self.spelling_map = {}        # misspelled -> correct
        self.confusable_rules = {}    # word -> [(context_check, correct, reason)]
        self._loaded = False
        self._spell = None            # pyspellchecker instance

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        self._load_common_mistakes()
        self._load_grammar_errors()
        self._load_spelling_dict()
        self._load_confusing_words()
        try:
            self._spell = SpellChecker()
        except Exception:
            self._spell = None

    def _load_common_mistakes(self):
        try:
            path = os.path.join(self.data_dir, "common_mistakes.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            grammar = data.get("common_grammar_mistakes", {})
            cat_map = {
                "subject_verb_agreement": ("agreement", "SVA_DATA"),
                "verb_tense": ("tense", "TENSE_DATA"),
                "pronoun_errors": ("pronouns", "PRONOUN_DATA"),
                "word_confusion": ("word_usage", "WORD_CONFUSION_DATA"),
                "double_negatives": ("grammar", "DOUBLE_NEGATIVE_DATA"),
                "comma_rules": ("punctuation", "COMMA_DATA"),
                "possessive_errors": ("pronouns", "POSSESSIVE_DATA"),
                "punctuation": ("punctuation", "PUNCTUATION_DATA"),
            }
            for cat_name, entries in grammar.items():
                if cat_name not in cat_map:
                    continue
                category, rule_prefix = cat_map[cat_name]
                for entry in entries:
                    incorrect = entry.get("incorrect", "").strip()
                    correct = entry.get("correct", "").strip()
                    explanation = entry.get("explanation", "")
                    if not incorrect or not correct or incorrect == correct:
                        continue
                    lower = incorrect.lower()
                    if lower not in self.exact_phrases:
                        self.exact_phrases[lower] = (correct, category, rule_prefix, explanation)
            # Spelling
            for entry in data.get("common_spelling_mistakes", []):
                word = entry.get("word", "").lower()
                corr = entry.get("correct", "")
                if word and corr and word != corr:
                    self.spelling_map[word] = corr
        except Exception:
            pass

    def _load_grammar_errors(self):
        try:
            path = os.path.join(self.data_dir, "grammar_errors.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cat_map = {
                "subject_verb_agreement": ("agreement", "SVA_DATA"),
                "verb_tense": ("tense", "TENSE_DATA"),
                "pronouns": ("pronouns", "PRONOUN_DATA"),
                "articles": ("articles", "ARTICLE_DATA"),
                "plurals": ("grammar", "PLURAL_DATA"),
                "comparatives": ("grammar", "COMPARATIVE_DATA"),
                "modals": ("grammar", "MODAL_DATA"),
                "negatives": ("grammar", "NEGATIVE_DATA"),
                "sentence_structure": ("sentence_structure", "SENTENCE_DATA"),
                "punctuation": ("punctuation", "PUNCTUATION_DATA"),
            }
            for cat_name, entries in data.items():
                if cat_name not in cat_map:
                    continue
                category, rule_prefix = cat_map[cat_name]
                for entry in entries:
                    incorrect = entry.get("incorrect", "").strip()
                    correct = entry.get("correct", "").strip()
                    explanation = entry.get("explanation", "")
                    if not incorrect or not correct or incorrect == correct:
                        continue
                    lower = incorrect.lower()
                    if lower not in self.exact_phrases:
                        self.exact_phrases[lower] = (correct, category, rule_prefix, explanation)
        except Exception:
            pass

    def _load_spelling_dict(self):
        try:
            path = os.path.join(self.data_dir, "spelling_dictionary.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for word, correct in data.get("common_misspellings", {}).items():
                if word and correct and word != correct:
                    self.spelling_map[word.lower()] = correct
        except Exception:
            pass

    def _load_confusing_words(self):
        try:
            path = os.path.join(self.data_dir, "confusing_words.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for word, info in data.get("confusing_words", {}).items():
                rules = info.get("rules", [])
                self.confusable_rules[word.lower()] = rules
        except Exception:
            pass

    def detect(self, text: str, doc=None) -> List[ErrorCandidate]:
        self._ensure_loaded()
        candidates = []
        candidates.extend(self._check_exact_phrases(text))
        candidates.extend(self._check_spelling(text))
        candidates.extend(self._check_spelling_fuzzy(text))
        candidates.extend(self._check_confusables(text, doc))
        return candidates

    def _check_exact_phrases(self, text: str) -> List[ErrorCandidate]:
        """Match exact error phrases from datasets."""
        candidates = []
        lower_text = text.lower()
        for phrase, (correct, category, rule_id, message) in self.exact_phrases.items():
            idx = lower_text.find(phrase)
            while idx != -1:
                end = idx + len(phrase)
                before_ok = idx == 0 or not text[idx - 1].isalpha()
                after_ok = end >= len(text) or not text[end].isalpha()
                if before_ok and after_ok:
                    # Skip if applying the replacement would not change the text
                    # e.g., "Thank you." matches "thank you" -> "Thank you." — period already there
                    # The replacement may add punctuation that's already in the text after the match
                    remaining_after = text[end:]  # text after the matched phrase
                    replacement_suffix = correct[len(phrase):]  # extra chars in replacement vs phrase
                    if remaining_after.lstrip().startswith(replacement_suffix.lstrip()):
                        idx = lower_text.find(phrase, end)
                        continue
                    candidates.append(ErrorCandidate(
                        start=idx, end=end,
                        original=text[idx:end], replacement=correct,
                        category=ErrorCategory(category.lower() if category.lower() in [e.value for e in ErrorCategory] else "grammar"),
                        rule_id=rule_id,
                        message=message or f'Use "{correct}" instead of "{phrase}".',
                        detector="data_driven",
                        raw_confidence=0.90,
                    ))
                idx = lower_text.find(phrase, end)
        return candidates

    def _check_spelling(self, text: str) -> List[ErrorCandidate]:
        """Match misspellings from spelling dictionary."""
        candidates = []
        for m in re.finditer(r'\b([a-zA-Z]+)\b', text):
            word = m.group(1).lower()
            if word in self.spelling_map:
                correct = self.spelling_map[word]
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=m.group(1), replacement=correct,
                    category=ErrorCategory.SPELLING,
                    rule_id="SPELLING_DATA",
                    message=f'Spelling: "{correct}" instead of "{m.group(1)}".',
                    detector="data_driven",
                    raw_confidence=0.90,
                ))
        return candidates

    def _check_spelling_fuzzy(self, text: str) -> List[ErrorCandidate]:
        """Use pyspellchecker for words not in our dictionaries."""
        candidates = []
        if self._spell is None:
            return candidates
        for m in re.finditer(r'\b([a-zA-Z]+)\b', text):
            word = m.group(1)
            lower = word.lower()
            if len(lower) < 4:
                continue
            if lower in self.spelling_map:
                continue
            if lower in self._spell:
                continue
            # Skip parts of contractions ("wouldn" from "wouldn't", "hasn" from "hasn't")
            end_pos = m.end()
            if end_pos < len(text) and text[end_pos] == "'":
                next_char_pos = end_pos + 1
                if next_char_pos < len(text) and text[next_char_pos].lower() == 't':
                    continue
            # Skip CamelCase compound words ("TheMechanics")
            if word[0].isupper() and any(c.isupper() for c in word[1:]):
                continue
            candidates_tuple = self._spell.candidates(lower)
            if candidates_tuple is None:
                continue
            candidates_list = list(candidates_tuple)
            if len(candidates_list) == 0:
                continue
            best = self._spell.correction(lower)
            if best and best != lower:
                candidates.append(ErrorCandidate(
                    start=m.start(), end=m.end(),
                    original=word, replacement=best,
                    category=ErrorCategory.SPELLING,
                    rule_id="SPELLING_FUZZY",
                    message=f'"{word}" may be misspelled. Did you mean "{best}"?',
                    detector="data_driven",
                    raw_confidence=0.75,
                ))
        return candidates

    def _check_confusables(self, text: str, doc) -> List[ErrorCandidate]:
        """Check confusing words using context rules from dataset."""
        candidates = []
        if doc is None:
            nlp = get_nlp()
            doc = nlp(text)
        for token in doc:
            lower = token.text.lower()
            if lower not in self.confusable_rules:
                continue
            rules = self.confusable_rules[lower]
            for rule in rules:
                context_words = [w.lower() for w in rule.get("context_words", [])]
                correct = rule.get("correct", "")
                if not correct or not context_words:
                    continue
                # Check if any context word appears near this token
                window = [t.lower_ for t in doc[max(0, token.i-3):token.i+4]]
                if any(cw in window for cw in context_words):
                    if lower != correct.lower():
                        # Skip if "their" is possessive (followed by noun/determiner)
                        if lower == "their" and correct.lower() == "there":
                            nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
                            if nxt and nxt.pos_ in ("NOUN", "PROPN", "ADJ", "DET", "PRON"):
                                continue
                        if lower == "less" and correct.lower() == "fewer":
                            nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
                            if nxt and nxt.pos_ in ("NOUN", "PROPN"):
                                if nxt.lower_ in UNCOUNTABLE_NOUNS:
                                    continue
                                if nxt.tag_ == "NN":
                                    continue
                        candidates.append(ErrorCandidate(
                            start=token.idx, end=token.idx + len(token.text),
                            original=token.text, replacement=correct,
                            category=ErrorCategory.WORD_USAGE,
                            rule_id="CONFUSABLE_DATA",
                            message=rule.get("reason", f'Use "{correct}" in this context.'),
                            detector="data_driven",
                            raw_confidence=0.85,
                        ))
                        break
        return candidates


class ContextEngine:
    """Analyzes whether a candidate error is actually wrong in context."""

    # Reporting/asking verbs that introduce embedded questions
    REPORTING_VERBS = {"ask", "asked", "tell", "told", "say", "said", "wonder",
                       "wondered", "know", "knew", "think", "thought", "believe",
                       "believed", "explain", "explained", "show", "showed",
                       "determine", "determined", "find", "found", "discover",
                       "discovered", "learn", "learned", "realize", "realized"}

    def should_suppress(self, candidate: ErrorCandidate, text: str, doc=None) -> Tuple[bool, str]:
        """Returns (should_suppress, reason)."""
        if doc is None:
            nlp = get_nlp()
            doc = nlp(text)

        # 1. Embedded question — don't apply inversion rules
        if candidate.rule_id in ("EMBEDDED_QUESTION",):
            return True, "Embedded question structure is correct"

        # 2. Reporting verb + embedded clause
        # Removed: errors in embedded clauses are still real errors

        # 3. Past tense with past time context
        if candidate.category == ErrorCategory.TENSE.value and candidate.rule_id != "TENSE_BACKSHIFT":
            if self._has_past_time_context(text, doc):
                return True, "Past time context makes past tense correct"

        # 4. Proper nouns / named entities — but exempt grammar errors
        _entity_exempt_rules = {"PRONOUN_SUBJECT_CASE", "PRONOUN_OBJECT_CASE", "SUBJECT_PRONOUN", "OBJECT_PRONOUN", "PRONOUN_CASE", "SVA", "AGREEMENT", "DO_SUPPORT", "WAS_WERE_BASE", "EVERY_SINGULAR", "UNCOUNTABLE_PLURAL", "DIDNT_PAST_FORM", "MODAL_WRONG_FORM", "SPELL_MISSPELLING", "SPELLING_DATA", "SPELLING_FUZZY", "POSSESSIVE_NOUN", "REPETITIVE_CONJUNCTION", "MISSING_PREPOSITION", "MISSING_PLURAL", "MISSING_ARTICLE", "ARTICLE_MISSING", "ARTICLE_A_AN", "QUESTION_INVERSION", "CONDITIONAL_TENSE", "COLLOCATION", "REDUNDANT_WORD", "REDUNDANT_ADVERB", "WORD_ORDER", "BARE_INFINITIVE", "DOUBLE_NEGATIVE", "PASSIVE_MISUSE", "PAST_PERFECT", "PROGRESSIVE_FORM", "TENSE_CONSISTENCY", "PREPOSITION_COLLOCATION", "UNNECESSARY_PREPOSITION", "MISSING_COMMA", "CONTRACTION", "ADVERB_FORM", "ADJECTIVE_FORM", "DATA_RULE_DECADE_APOSTROPHE", "DUPLICATE_VERB", "PREPOSITION_SCHEDULED_ON", "TENSE_BACKSHIFT", "ANYBODY_DECLARATIVE"}
        if (candidate.rule_id not in _entity_exempt_rules
                and not candidate.rule_id.startswith("DATA_RULE")
                and self._is_proper_noun(candidate, doc)):
            return True, "Proper noun / named entity"

        # 5. Collective nouns (British English)
        if candidate.category == ErrorCategory.SUBJECT_VERB_AGREEMENT.value:
            if candidate.metadata.get("subject", "").lower() in COLLECTIVE_NOUNS:
                return True, "Collective noun — both singular and plural acceptable"

        # 6. Uncountable nouns
        if candidate.category == ErrorCategory.SUBJECT_VERB_AGREEMENT.value:
            if candidate.metadata.get("subject", "").lower() in UNCOUNTABLE_NOUNS:
                return True, "Uncountable noun takes singular verb"

        # 7. Singular they
        if candidate.category == ErrorCategory.SUBJECT_VERB_AGREEMENT.value:
            subj = candidate.metadata.get("subject", "").lower()
            if subj in SINGULAR_PRONOUNS:
                return True, "Singular they — acceptable in modern English"

        # 8. Quoted text
        if self._is_in_quotes(candidate, text):
            return True, "Quoted text"

        return False, ""

    def _is_in_embedded_clause(self, candidate: ErrorCandidate, doc) -> bool:
        """Check if candidate is inside an embedded question/clause."""
        for token in doc:
            if token.i * 1 <= candidate.start <= (token.i + 1) * 5:
                # Check if this token is in a ccomp/xcomp after a reporting verb
                if token.dep_ in ("ccomp", "xcomp"):
                    head = token.head
                    if head.text.lower() in self.REPORTING_VERBS:
                        return True
        return False

    def _has_past_time_context(self, text: str, doc) -> bool:
        past_markers = {"yesterday", "ago", "before", "last", "previously",
                        "earlier", "formerly", "once", "in the past"}
        for token in doc:
            if token.lower_ in past_markers:
                return True
        return False

    def _is_proper_noun(self, candidate: ErrorCandidate, doc) -> bool:
        for ent in doc.ents:
            if ent.start_char <= candidate.start < ent.end_char:
                return True
        return False

    def _is_in_quotes(self, candidate: ErrorCandidate, text: str) -> bool:
        before = text[:candidate.start]
        return (before.count('"') - before.count('\\"')) % 2 == 1


# ══════════════════════════════════════════════════════════════════════════
#  CONFIDENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════

class ConfidenceEngine:
    """Calculates final confidence from multiple evidence sources."""

    THRESHOLDS = {
        "high": 0.95,
        "good": 0.85,
        "review": 0.75,
        "suppress": 0.70,
    }

    def score(self, candidate: ErrorCandidate, context_result: Tuple[bool, str]) -> float:
        base = candidate.raw_confidence
        suppressed, reason = context_result

        if suppressed:
            return 0.0  # Context says this is correct

        # Boost confidence for certain categories
        if candidate.category == ErrorCategory.SPELLING.value:
            base = min(base + 0.03, 1.0)
        if candidate.category == ErrorCategory.SUBJECT_VERB_AGREEMENT.value:
            if candidate.metadata.get("existential_there"):
                base = min(base + 0.02, 1.0)

        return base


# ══════════════════════════════════════════════════════════════════════════
#  FALSE-POSITIVE FILTER
# ══════════════════════════════════════════════════════════════════════════

class FalsePositiveFilter:
    """15+ strategies to filter false positives."""

    def filter(self, candidates: List[ErrorCandidate], text: str, doc=None) -> List[ErrorCandidate]:
        if doc is None:
            nlp = get_nlp()
            doc = nlp(text)

        filtered = []
        for c in candidates:
            if not self._is_false_positive(c, text, doc):
                filtered.append(c)
        return filtered

    def _is_false_positive(self, c: ErrorCandidate, text: str, doc) -> bool:
        lower_text = text.lower()
        # Strategy 1: Protected content
        if c.metadata.get("protected"):
            return True

        # Strategy 2: Inside named entity — but don't suppress grammar corrections
        _entity_exempt_rules = {"PRONOUN_SUBJECT_CASE", "PRONOUN_OBJECT_CASE", "SUBJECT_PRONOUN", "OBJECT_PRONOUN", "PRONOUN_CASE", "SVA", "AGREEMENT", "DO_SUPPORT", "WAS_WERE_BASE", "EVERY_SINGULAR", "UNCOUNTABLE_PLURAL", "DIDNT_PAST_FORM", "MODAL_WRONG_FORM", "SPELL_MISSPELLING", "SPELLING_DATA", "SPELLING_FUZZY", "REPETITIVE_CONJUNCTION", "MISSING_PREPOSITION", "MISSING_PLURAL", "MISSING_ARTICLE", "ARTICLE_MISSING", "ARTICLE_A_AN", "QUESTION_INVERSION", "CONDITIONAL_TENSE", "COLLOCATION", "REDUNDANT_WORD", "REDUNDANT_ADVERB", "WORD_ORDER", "BARE_INFINITIVE", "DOUBLE_NEGATIVE", "PASSIVE_MISUSE", "PAST_PERFECT", "PROGRESSIVE_FORM", "TENSE_CONSISTENCY", "PREPOSITION_COLLOCATION", "UNNECESSARY_PREPOSITION", "MISSING_COMMA", "CONTRACTION", "ADVERB_FORM", "ADJECTIVE_FORM", "DATIVE", "EXPLAIN_DATIVE", "INFINITIVE_FORM", "GERUND_INFINITIVE", "MODAL_UNNEEDED_TO", "REDUNDANT_ABLE", "DOUBLE_COMPARATIVE", "DOUBLE_SUPERLATIVE", "WHICH_WHO", "WRONG_PAST_PARTICIPLE", "CAUSATIVE_BARE_INFINITIVE", "TENSE_PAST_MARKER", "ADJ_PREPOSITION", "DUPLICATE_VERB", "PREPOSITION_SCHEDULED_ON", "TENSE_BACKSHIFT", "ANYBODY_DECLARATIVE"}
        if c.rule_id not in _entity_exempt_rules and not c.rule_id.startswith("DATA_RULE"):
            for ent in doc.ents:
                if ent.start_char <= c.start < ent.end_char:
                    return True

        # Strategy 3: Repetition detection — "the the" is OK if one is a proper noun
        if c.rule_id == "REPEATED_WORD":
            # Allow "had had", "that that" etc.
            if c.original.lower().split()[0] in {"had", "that", "is", "are", "was", "were", "do", "does", "did"}:
                return True

        # Strategy 4: its/it's — suppress dummy subject "it's" before weather/verbs
        if c.rule_id == "POSSESSIVE_ITS":
            next_word = ""
            remaining = text[c.end:].lstrip()
            if remaining:
                next_word = remaining.split()[0].lower() if remaining.split() else ""
            dummy_subj_words = {
                "raining", "snowing", "raining", "hailing", "sleeting",
                "cold", "hot", "warm", "cool", "dark", "light", "early", "late",
                "getting", "been", "going", "about", "time", "important",
                "necessary", "essential", "clear", "obvious", "possible",
                "likely", "unlikely", "okay", "ok", "fine", "wrong",
                "difficult", "easy", "hard", "nice", "great", "good",
                "better", "worse", "better", "worse",
            }
            if next_word in dummy_subj_words:
                return True
            if c.raw_confidence < 0.90:
                return True

        # Strategy 5: existential there — don't suppress, corrections should be made

        # Strategy 6: Collective nouns — British vs American English
        # "The team are" — still flag it (American English prefers "is")
        # "have/had/do/did" with a collective — flag in AmE ("Our team have attended"),
        # except for genuinely plural / British-tolerant words (police, cattle, media,
        # staff, government) which behave like Strategy 25's BrE handling.
        if c.rule_id == "SVA" and doc is not None:
            subj = c.metadata.get("subject", "").lower()
            collective_nouns = {"team", "family", "jury", "orchestra", "committee", "staff",
                        "group", "class", "audience", "crowd", "government", "company",
                        "club", "army", "band", "board", "council", "faculty",
                        "fleet", "gang", "herd", "majority", "minority", "mob",
                        "navy", "public", "squad", "troop", "village",
                        "data", "media", "police", "people", "cattle", "poultry", "children"}
            if subj in collective_nouns:
                # Only suppress if the verb is plural (British English pattern)
                verb_text = c.original.lower()
                if verb_text in ("are", "were"):
                    return True
                # BrE-tolerant collectives with "have/had/do/did" — suppress only when
                # the next word isn't a predicate adjective that must agree in AmE.
                if verb_text in ("have", "had", "do", "did") and subj in {
                        "staff", "government", "police", "media", "cattle"}:
                    after = text[c.end:].lstrip()
                    nxt = after.split()[0].lower().strip(".,!?;:") if after else ""
                    if nxt not in ("ready", "late", "early", "happy", "available", "sick",
                                   "tired", "present", "absent", "away", "here", "there"):
                        return True

        # Strategy 8: In embedded clause — only suppress inside quoted speech
        if c.rule_id in ("EMBEDDED_QUESTION",):
            n_quotes = text[:c.start].count('"')
            if n_quotes % 2 == 1:
                return True

        # Strategy 9: AAVE habitual "be" — "He be working", "She be at the library", "He don't be there"
        if c.rule_id in ("SVA", "DO_SUPPORT") and c.original.lower() in ("be", "don't", "doesn't"):
            text_before = text[:c.start].rstrip()
            if text_before:
                last_word = text_before.split()[-1].lower() if text_before.split() else ""
                if c.original.lower() in ("be",) and last_word not in MODALS | DO_FORMS | {"will", "would", "could", "should", "may", "might", "must", "shall", "can"}:
                    return True
                # "don't be" pattern — AAVE habitual
                if "don't be" in lower_text or "doesn't be" in lower_text:
                    return True

        # Strategy 9b: AAVE "was" with plural pronouns — now corrected (standard English)

        # Strategy 9c: AAVE "ain't" — "Ain't nobody got time"
        if c.rule_id in ("SVA", "AGREEMENT"):
            text_before = text[:c.start].rstrip()
            if text_before:
                last_word = text_before.split()[-1].lower() if text_before.split() else ""
                if last_word in ("ain't", "aint"):
                    return True
            # Also handle "Ai" token (spaCy splits "ain't" → "Ai" + "n't")
            if c.original.lower() == "ai":
                return True

        # Strategy 9d: Dialect patterns — suppress prescriptive corrections for known dialect constructions
        # "seen" without auxiliary (dialect past tense — suppress when subject is a pronoun)
        if c.rule_id in ("WORD_USAGE", "SEEN_SAW") and c.original.lower() == "seen":
            lower_t = text.lower()
            if re.search(r"\b(don't|doesn't|didn't)\b.*\b(nothing|nobody|nowhere)\b", lower_t):
                return True
            if re.search(r"\b(they|we|you|i)\s+was\b", lower_t):
                return True
            if re.search(r"\b(i|he|she|it|we|they|you)\s+seen\b", lower_t):
                return True
        # "don't" + 3rd person (dialect) — "She don't", "He don't"
        if c.rule_id == "DO_SUPPORT":
            subj = None
            if doc is not None:
                for ent in doc.ents:
                    pass
                for tok in doc:
                    if tok.dep_ in ("nsubj", "nsubjpass") and tok.head.text.lower() in ("do", "don't"):
                        subj = tok.lower_
                        break
            if subj in ("she", "he", "it"):
                # Check if this is a known dialect pattern in the sentence
                if "don't" in lower_text and any(w in lower_text for w in ("nothing", "nobody", "nowhere")):
                    return True
        # "have" with "everyone" / "everybody" — standard English requires singular "has"
        # Don't suppress — "everyone have" IS an error in standard English

        # Strategy 10: "a number of" + plural verb — valid ("A number of students are absent")
        # "the number of" takes SINGULAR verb, so don't suppress
        if c.rule_id == "SVA":
            subj = c.metadata.get("subject", "").lower()
            if subj == "number":
                text_before = text[:c.start].rstrip()
                lower_before = text_before.lower()
                idx = lower_before.rfind("number of")
                if idx >= 2:
                    two_before = lower_before[idx-2:idx]
                    if two_before == "a ":
                        return True

        # Strategy 11: Compound quantity — "five dollars" is singular, so "is" is correct

        # Strategy 12: "None of them are" — "none" can be plural
        if c.rule_id == "SVA":
            subj = c.metadata.get("subject", "").lower()
            if subj == "none":
                return True

        # Strategy 13: Fraction phrases — verb agrees with the "of" phrase
        if c.rule_id == "SVA" and doc is not None:
            subj_text = c.metadata.get("subject", "").lower()
            fraction_words = {"half", "third", "thirds", "quarter", "quarters", "percent",
                              "part", "parts", "majority", "minority", "rest", "lot"}
            for frac in fraction_words:
                if frac in subj_text:
                    # Check if "of" phrase follows — verb should agree with object of "of"
                    for token in doc:
                        if token.text.lower() == "of" and token.dep_ == "prep":
                            for pobj in token.children:
                                if pobj.dep_ == "pobj":
                                    return True
                    break

        # Strategy 14: Cardinal numbers in prepositional phrases — "on cloud nine", "at three o'clock"
        if c.rule_id == "SVA" and doc is not None:
            subj_text = c.metadata.get("subject", "").lower()
            for token in doc:
                if token.text.lower() == subj_text and token.dep_ in ("nsubj", "nsubjpass"):
                    if token.tag_ == "CD":
                        for child in token.children:
                            if child.dep_ == "compound":
                                return True
                        if token.dep_ == "pobj":
                            return True
                    break

        # Strategy 15: Indefinite pronouns — "everyone" takes singular verb ("has", not "have")

        # Strategy 16: Predicative adjective "good" — "it's very good", "she looks good"
        # Only correct "good -> well" when it modifies a verb directly (works good).
        if c.rule_id in ("ADVERB_FORM", "ADJECTIVE_FORM") and c.original.lower() == "good":
            prev = ""
            before = text[:c.start].rstrip()
            if before:
                prev = before.split()[-1].lower() if before.split() else ""
            linking = {"is", "am", "are", "was", "were", "be", "been", "being", "seem", "seems",
                       "seemed", "seeming", "sound", "sounds", "look", "looks", "looked",
                       "feel", "feels", "felt", "taste", "tastes", "smell", "smells",
                       "appear", "appears", "become", "becomes", "became", "get", "gets", "got",
                       "turn", "turns", "turned", "stay", "stays", "remain", "remains", "grow", "grows"}
            adverb_mods = {"very", "really", "pretty", "quite", "too", "so", "extremely", "perfectly",
                           "fairly", "rather", "particularly", "incredibly", "totally", "absolutely"}
            if prev in linking or prev in adverb_mods:
                return True

        # Strategy 17: "o'clock" never pluralizes — "eight o'clock" is correct
        if c.rule_id == "MISSING_PLURAL" and "o'clock" in text[c.start:c.end].lower():
            return True
        if c.rule_id == "MISSING_PLURAL":
            after = text[c.end:].lstrip()
            if after and after.split()[0].lower().strip(".,!?;:") == "o'clock":
                return True

        # Strategy 18: Common-technology vocabulary — "tech", "app", "api" are not typos
        if c.rule_id == "SPELLING_FUZZY" and c.original.lower() in {
            "tech", "app", "api", "url", "uri", "apps", "software", "smartphone",
            "smartphones", "laptop", "laptops", "cyber", "phishing", "malware",
            "ransomware", "crypto", "bitcoin", "google", "apple", "microsoft",
            "amazon", "netflix", "youtube", "whatsapp", "facebook", "twitter",
            "instagram", "linkedin", "blockchain", "startup", "semantic"}:  # fmt: off
            return True

        # Strategy 19: Already-clean punctuation candidates — text already capitalized & terminated
        if c.rule_id == "PUNCTUATION_DATA":
            stripped = text.strip()
            if stripped and stripped[0].isupper() and stripped[-1] in ".!?":
                return True

        # Strategy 20: CAUSATIVE_BARE_INFINITIVE — only valid when the word after the
        # flagged "to" is actually a verb (bare infinitive). "can have gone to the store",
        # "went to school" are NOT causatives.
        if c.rule_id == "CAUSATIVE_BARE_INFINITIVE":
            drug = False
            token_after = ""
            if doc is not None:
                for tok in doc:
                    if tok.idx >= c.end and tok.text.strip():
                        token_after = tok.text.lower()
                        break
                if token_after:
                    for tok in doc:
                        if tok.text.lower() == token_after and tok.pos_ == "VERB":
                            drug = True
                            break
            if not drug:
                return True
            # modal + have + participle ("can have gone") — not causative
            sentence = text.rstrip()
            if re.search(r"\b(can|could|may|might|must|shall|should|will|would)\s+"
                         r"(have|have)\s+\w+ed\b", sentence, re.I):
                return True

        # Strategy 21: DATIVE_PREPOSITION — verbs that already take a direct object
        # ("ask me", "tell them", "taught us") must not get an inserted "to".
        if c.rule_id in ("DATIVE_PREPOSITION", "DATIVE_SUFFIX"):
            low = text.lower()
            if re.search(r"\b(ask|asks|asked|asking|tell|tells|told|telling|teach|teaches|"
                         r"taught|teaching|promise|promises|promised|charge|charges|charged|"
                         r"cost|costs|cost|fine|fines|fined|grant|grants|granted|deny|denies|"
                         r"denied|refuse|refuses|refused|save|saves|saved|envy|envi(e)?s|"
                         r"forgive|forgives|forgave|spare|spares|spared|wish|wishes|wished|"
                         r"cause|causes|caused|bet|bets)\b", low):
                return True

        # Strategy 22: Pronoun case — only flag pronouns in genuinely subject positions.
        # A pronoun preceded by a verb (help them, asked us, told him) is an object.
        if c.rule_id in ("PRONOUN_SUBJECT_CASE", "PRONOUN_OBJECT_CASE") and \
                doc is not None and c.original.lower() in ("me", "him", "her", "us", "them", "i", "we", "they"):
            before = text[:c.start].rstrip()
            phrase_before = before[-12:] if before else ""
            # Coordinated ("John and me", "Me and John") or sentence-initial: keep checking
            if re.search(r"(^|[.!?;])\s*$", before) or re.search(r"\b(and|or|but|then)\s*$", before, re.I):
                pass  # genuine subject-candidate — keep
            else:
                # If the pronoun directly follows a verb, it is an object — reject
                prev_word = before.split()[-1].lower() if before.split() else ""
                if prev_word:
                    for tok in doc:
                        if tok.text.lower() == prev_word:
                            if tok.pos_ in ("VERB", "AUX", "ADP", "CONJ"):
                                return True
                            break

        # Strategy 23: its/it's — contraction followed by a non-noun is valid
        # ("It's over", "It's here") — only flag possessive before a NOUN.
        if c.rule_id in ("POSSESSIVE_ITS", "DATA_RULE_ITS_POSSESSIVE", "ITS_POSSESSIVE") \
                and doc is not None:
            next_pos = ""
            for tok in doc:
                if tok.idx >= c.end and tok.text.strip():
                    next_pos = tok.pos_
                    break
            if c.original.lower() == "it's":
                if next_pos and next_pos != "NOUN":
                    return True
            else:  # original written as "its" (possessive) — wrong before a verb
                if next_pos == "NOUN":
                    return True

        # Strategy 24: "there" + adverb ("there before", "there too") is not their
        if c.rule_id in ("THERE_THEIR", "DATA_RULE_THERE_THEIR"):
            after = text[c.end:].lstrip()
            if after:
                nxt = after.split()[0].lower().strip(".,!?;:")
                if nxt in ("before", "yet", "today", "tomorrow", "soon", "now",
                           "too", "away", "around", "alone", "later", "so",
                           "already", "then", "first", "also"):
                    return True

        # Strategy 25: collective SVA — some collectives are plural in practice
        # (BrE "staff are divided" is fine) but not with a plain adjective
        # ("staff are ready" should still flag in AmE).
        if c.rule_id in ("DATA_RULE_COLLECTIVE_SVA", "DATA_RULE_GROUP_OF_SVA"):
            span = text[c.start:c.end].lower()
            after = text[c.end:].lstrip()
            nxt = after.split()[0].lower().strip(".,!?;:") if after else ""
            if any(w in span for w in ("staff are", "staff were", "government are",
                                       "government were", "police are", "police were",
                                       "media are", "media were", "cattle are",
                                       "cattle were")) and nxt not in (
                    "ready", "late", "early", "happy", "available", "sick", "tired",
                    "present", "absent", "away", "here", "there"):
                return True

        # Strategy 26: -ics subjects are singular ("Statistics is", "Physics is")
        if c.rule_id == "SVA":
            subj = (c.metadata.get("subject", "") or "").lower()
            if subj.endswith("ics"):
                return True

        # Strategy 27: TO_TOO before a proper noun ("to New York") is valid
        if c.rule_id == "TO_TOO":
            after = text[c.end:].lstrip()
            if after and after[0].isupper():
                return True

        # Strategy 28: MISSING_PLURAL after a decimal quantity ("99.9 percent")
        if c.rule_id == "MISSING_PLURAL":
            before = text[:c.start]
            if re.search(r"\d+\.\d+\s*\w*$", before):
                return True
            after = text[c.end:].lstrip()
            if after and after.split()[0].lower().strip(".,!?;:") in ("percent", "percents"):
                return True

        # Strategy 29: SPELLING_FUZZY — technology vocabulary is not a typo
        if c.rule_id == "SPELLING_FUZZY" and c.original.lower() in {
            "uptime", "regex", "overfit", "overfits", "overfitting", "dataset",
            "datasets", "endpoint", "endpoints", "linter", "cache", "caching",
            "deploy", "deploys", "deployed", "refactor", "refactors",
            "refactoring", "tokenizer", "embeddings", "vector", "vectors",
            "billing", "marketplace", "startup", "startups", "json", "saas"}:
            return True

        return False


# ══════════════════════════════════════════════════════════════════════════
#  CORRECTION VALIDATOR
# ══════════════════════════════════════════════════════════════════════════

class CorrectionValidator:
    """Validates that corrections are actually better."""

    # Content-word POS we require to be preserved across the edit (meaning check).
    _CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}
    # Non-content, order-insensitive words that differ trivially between versions.
    _SEMANTIC_STOP = {
        "be", "is", "are", "was", "were", "been", "being", "am",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "not", "no", "nor", "so", "too", "very",
    }

    def validate(self, candidate: ErrorCandidate, text: str) -> Tuple[bool, str]:
        if candidate.replacement is None:
            return False, "No replacement"

        if candidate.replacement and candidate.replacement.strip() == candidate.original.strip():
            return False, "Correction is a no-op (replacement equals original)"

        corrected = text[:candidate.start] + candidate.replacement + text[candidate.end:]

        nlp = get_nlp()
        try:
            orig_doc = nlp(text)
            corr_doc = nlp(corrected)
        except Exception:
            return True, "Validation skipped (parse error)"

        # Meaning preservation: content words outside the edited span must survive.
        ratio = self._content_overlap_ratio(orig_doc, corr_doc, candidate.start, candidate.end,
                                            len(candidate.replacement))
        if ratio < 0.50:
            return False, f"Correction changes meaning (content overlap {ratio:.2f})"

        return True, "Valid"

    def _content_overlap_ratio(self, orig_doc, corr_doc, start, end, repl_len=0) -> float:
        """Ratio of content words preserved outside the edited span."""
        a = self._content_words(orig_doc, start, end)
        b = self._content_words(corr_doc, start, start + (repl_len or (end - start)))
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _content_words(self, doc, wstart, wend):
        words = []
        for t in doc:
            if t.idx < wend and t.idx + len(t.text) > wstart:
                continue
            if t.pos_ in self._CONTENT_POS and t.lower_ not in self._SEMANTIC_STOP:
                words.append(t.lemma_.lower())
        return words


# ══════════════════════════════════════════════════════════════════════════
#  DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════

def deduplicate(candidates: List[ErrorCandidate]) -> List[ErrorCandidate]:
    """Remove duplicate candidates at same position."""
    if not candidates:
        return []

    # Sort: replacements first, then by confidence (highest first)
    candidates.sort(key=lambda c: (0 if c.replacement else 1, -c.raw_confidence))

    deduped = []
    used_ranges = []
    for c in candidates:
        is_dup = False
        for used_start, used_end in used_ranges:
            if c.start < used_end and c.end > used_start:
                is_dup = True
                break
        if not is_dup:
            deduped.append(c)
            used_ranges.append((c.start, c.end))
    return deduped


# ══════════════════════════════════════════════════════════════════════════
#  MAIN UNIFIED PIPELINE
# ══════════════════════════════════════════════════════════════════════════

class UnifiedPipeline:
    """
    Single unified grammar checking pipeline.

    Flow:
      TEXT → PREPROCESS → FAST DETECT → NLP DETECT →
      CONTEXT → CONFIDENCE → FP FILTER → VALIDATE →
      DEDUPLICATE → RANK → OUTPUT
    """

    def __init__(self):
        self.fast_detector = FastDetector()
        self.nlp_detector = NLPDetector()
        self.data_detector = DataDrivenDetector()
        self.high_conf_detector = HighConfidenceDetector()
        self.context_engine = ContextEngine()
        self.confidence_engine = ConfidenceEngine()
        self.fp_filter = FalsePositiveFilter()
        self.validator = CorrectionValidator()
        self.preprocessor = TextPreprocessor()

    def check(self, text: str) -> List[Dict]:
        """Main entry point. Returns list of error dicts for the frontend."""
        if not text or not text.strip():
            return []

        nlp = get_nlp()

        # Step 1: Protect non-English content
        clean_text, protected = self.preprocessor.protect(text)

        # Step 2: Parse
        doc = nlp(clean_text)

        # Step 3: Fast detection (Tier 1)
        fast_candidates = self.fast_detector.detect(clean_text)

        # Step 4: NLP detection (Tier 2)
        nlp_candidates = self.nlp_detector.detect(clean_text, doc)

        # Step 4b: Data-driven detection (Tier 2b — pattern lookup from JSON datasets)
        data_candidates = self.data_detector.detect(clean_text, doc)

        # Step 4c: High-confidence data-distilled rules (Tier 2c)
        high_conf_candidates = self.high_conf_detector.detect(clean_text, doc)

        # Step 5: Merge all candidates
        all_candidates = fast_candidates + nlp_candidates + data_candidates + high_conf_candidates

        # Step 6: Context analysis
        context_filtered = []
        for c in all_candidates:
            suppressed, reason = self.context_engine.should_suppress(c, clean_text, doc)
            confidence = self.confidence_engine.score(c, (suppressed, reason))
            if confidence >= 0.70:
                c.raw_confidence = confidence
                context_filtered.append(c)

        # Step 7: False-positive filter
        fp_filtered = self.fp_filter.filter(context_filtered, clean_text, doc)

        # Step 8: Deduplication
        deduped = deduplicate(fp_filtered)

        # Step 9: Validate corrections
        validated = []
        for c in deduped:
            is_valid, reason = self.validator.validate(c, clean_text)
            if is_valid:
                validated.append(c)

        # Step 10: Rank and convert to output format
        validated.sort(key=lambda c: c.raw_confidence, reverse=True)

        errors = []
        for c in validated:
            if c.raw_confidence < 0.70:
                continue
            severity = "error" if c.raw_confidence >= 0.90 else "warning" if c.raw_confidence >= 0.80 else "info"
            errors.append({
                "start": c.start,
                "end": c.end,
                "original": c.original,
                "replacement": c.replacement,
                "category": c.category.value.upper(),
                "subcategory": c.rule_id,
                "severity": severity,
                "confidence": round(c.raw_confidence, 2),
                "explanation": c.message,
                "rule_id": c.rule_id,
                "detector": c.detector,
                "validated": True,
                "message": c.message,
                "error_type": c.category.value,
                "alternatives": [c.replacement] if c.replacement else [],
                "context": clean_text,
            })

        return errors


# ══════════════════════════════════════════════════════════════════════════
#  STYLE/CLARITY/TONE (separate system — never shown as grammar errors)
# ══════════════════════════════════════════════════════════════════════════

class StyleEngine:
    """Style suggestions — shown separately from grammar errors."""

    def analyze(self, text: str) -> List[Dict]:
        suggestions = []
        lower = text.lower()

        # Filler words
        for filler in FILLER_WORDS:
            if filler in lower:
                idx = lower.find(filler)
                suggestions.append({
                    "start": idx, "end": idx + len(filler),
                    "original": text[idx:idx + len(filler)],
                    "category": "style",
                    "message": f'Consider removing filler word "{filler}".',
                    "severity": "info",
                })

        # Weak words
        for m in re.finditer(r'\b(\w+)\b', text):
            if m.group(1).lower() in WEAK_WORDS:
                suggestions.append({
                    "start": m.start(), "end": m.end(),
                    "original": m.group(1),
                    "category": "style",
                    "message": f'"{m.group(1)}" is a weak word. Consider a stronger alternative.',
                    "severity": "info",
                })

        return suggestions


class ReadabilityEngine:
    """Readability scoring."""

    def calculate(self, text: str) -> Dict:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        word_count = len(words)
        sentence_count = max(len(sentences), 1)

        # Flesch-Kincaid
        syllable_count = sum(self._count_syllables(w) for w in words)
        avg_sentence_len = word_count / sentence_count
        avg_syllables = syllable_count / max(word_count, 1)

        fk_grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59
        fk_score = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables

        if fk_score >= 90:
            level = "Very Easy"
        elif fk_score >= 80:
            level = "Easy"
        elif fk_score >= 70:
            level = "Fairly Easy"
        elif fk_score >= 60:
            level = "Standard"
        elif fk_score >= 50:
            level = "Fairly Difficult"
        elif fk_score >= 30:
            level = "Difficult"
        else:
            level = "Very Confusing"

        return {
            "score": round(fk_score, 1),
            "grade": round(fk_grade, 1),
            "level": level,
            "sentences": sentence_count,
            "words": word_count,
            "avg_words_sentence": round(avg_sentence_len, 1),
            "reading_time": round(word_count / 200, 1),
            "speaking_time": round(word_count / 130, 1),
        }

    def _count_syllables(self, word: str) -> int:
        word = word.lower().strip(".,!?;:'\"")
        if not word:
            return 0
        count = 0
        vowels = "aeiouy"
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e"):
            count -= 1
        return max(count, 1)


class ToneEngine:
    """Tone detection."""

    PROFESSIONAL = {"furthermore", "moreover", "consequently", "therefore", "hence",
                    "accordingly", "nevertheless", "notwithstanding", "hitherto"}
    FRIENDLY = {"hey", "hi", "hello", "awesome", "great", "love", "thanks",
                "please", "glad", "happy", "wonderful", "fantastic", "amazing"}
    FORMAL = {"hereby", "herein", "aforementioned", "notwithstanding", "henceforth",
              "whereas", "whereby", "forthwith"}
    CASUAL = {"gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "yeah",
              "ok", "cool", "awesome", "stuff", "things", "like"}

    def analyze(self, text: str) -> Dict:
        lower = text.lower()
        words = set(re.findall(r'\b\w+\b', lower))

        scores = {}
        scores["professional"] = len(words & self.PROFESSIONAL) / max(len(words), 1)
        scores["friendly"] = len(words & self.FRIENDLY) / max(len(words), 1)
        scores["formal"] = len(words & self.FORMAL) / max(len(words), 1)
        scores["casual"] = len(words & self.CASUAL) / max(len(words), 1)

        # Heuristic boosts
        if text.count("?") > 1:
            scores["friendly"] += 0.1
        if len(text.split()) > 20:
            scores["formal"] += 0.05

        # Determine dominant tone
        dominant = max(scores, key=scores.get)
        if scores[dominant] < 0.01:
            dominant = "neutral"

        return {"dominant": dominant, "scores": scores}


# ══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = UnifiedPipeline()
    return _pipeline


def check_text(text: str) -> List[Dict]:
    """Quick check — returns errors only."""
    return get_pipeline().check(text)


def check_text_full(text: str) -> Dict:
    """Full check — returns errors + scores + readability + tone."""
    pipeline = get_pipeline()
    errors = pipeline.check(text)
    readability = ReadabilityEngine().calculate(text)
    tone = ToneEngine().analyze(text)
    style = StyleEngine().analyze(text)
    return {
        "errors": errors,
        "readability": readability,
        "tone": tone,
        "style": style,
    }
