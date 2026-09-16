"""
Writing Intelligence Engine - Unified pipeline for grammar, spelling, punctuation,
capitalization, word usage, style, and clarity checking.
"""

import re
import json
import os
import math
from typing import List, Dict, Tuple, Optional, Set


# ---------------------------------------------------------------------------
# Error data structure
# ---------------------------------------------------------------------------
def make_error(
    text: str,
    start: int,
    end: int,
    category: str,
    message: str,
    suggestion: str = "",
    severity: str = "MEDIUM",
    confidence: float = 0.8,
    context: str = "",
    explanation: str = "",
) -> Dict:
    return {
        "original_text": text,
        "replacement": suggestion,
        "category": category,
        "error_type": _category_to_type(category),
        "message": message,
        "explanation": explanation or message,
        "start_position": start,
        "end_position": end,
        "severity": severity,
        "confidence": round(confidence, 2),
        "context": context,
    }


def _category_to_type(cat: str) -> str:
    mapping = {
        "SPELLING": "spelling",
        "GRAMMAR": "grammar",
        "PUNCTUATION": "punctuation",
        "CAPITALIZATION": "capitalization",
        "WORD_USAGE": "word_usage",
        "CONTEXTUAL_WORD_USAGE": "contextual_word_usage",
        "SENTENCE_STRUCTURE": "sentence_structure",
        "REDUNDANCY": "redundancy",
        "MISSING_WORD": "missing_word",
        "EXTRA_WORD": "extra_word",
        "TENSE": "tense",
        "SUBJECT_VERB_AGREEMENT": "subject_verb_agreement",
        "STYLE": "style",
        "CLARITY": "clarity",
        "READABILITY": "readability",
    }
    return mapping.get(cat, "grammar")


# ---------------------------------------------------------------------------
# Spelling engine
# ---------------------------------------------------------------------------
class SpellingEngine:
    def __init__(self, data_dir: str):
        self._misspellings: Dict[str, str] = {}
        self._valid_words: Set[str] = set()
        self._ignored_patterns: List[re.Pattern] = []
        self._custom_words: Set[str] = set()
        self._load(data_dir)

    def _load(self, data_dir: str):
        path = os.path.join(data_dir, "spelling_dictionary.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._misspellings = data.get("common_misspellings", {})
            raw_valid = data.get("valid_words", [])
            self._valid_words = {w.lower() for w in raw_valid}
            for pat in data.get("ignored_patterns", []):
                self._ignored_patterns.append(re.compile(pat))
        except Exception:
            pass
        cm_path = os.path.join(data_dir, "common_mistakes.json")
        try:
            with open(cm_path, "r", encoding="utf-8") as f:
                cm = json.load(f)
            for pair in cm.get("common_spelling_mistakes", []):
                if isinstance(pair, dict):
                    wrong = pair.get("wrong", "").lower()
                    right = pair.get("correct", "")
                    if wrong and right:
                        self._misspellings[wrong] = right
                elif isinstance(pair, (list, tuple)) and len(pair) == 2:
                    self._misspellings[pair[0].lower()] = pair[1]
        except Exception:
            pass
        extra_misspellings = {
            "sayed": "said", "fastly": "fast", "messae": "message",
            "rerom": "error", "cehcj": "check", "dcorrcet": "correct",
            "conrntb": "content", "thsi": "this", "comming": "coming",
            "recieve": "receive", "freind": "friend", "msitake": "mistake",
            "tommorow": "tomorrow", "dont": "don't", "cant": "can't",
            "doesnt": "doesn't", "didnt": "didn't", "wouldnt": "wouldn't",
            "couldnt": "couldn't", "shouldnt": "shouldn't",
            "wont": "won't", "isnt": "isn't", "arent": "aren't",
            "wasnt": "wasn't", "werent": "weren't", "hasnt": "hasn't",
            "havent": "haven't", "hadnt": "hadn't",
            "alot": "a lot", "untill": "until", "wierd": "weird",
            "truely": "truly", "occured": "occurred",
            "enviroment": "environment", "goverment": "government",
            "neccessary": "necessary", "priviledge": "privilege",
            "relevent": "relevant", "successfull": "successful",
            "grammer": "grammar", "independant": "independent",
            "libary": "library", "maintainance": "maintenance",
            "posession": "possession", "publically": "publicly",
            "responsable": "responsible", "similiar": "similar",
            "suprise": "surprise", "wheter": "whether", "wich": "which",
            "yeild": "yield", "accross": "across",
        }
        self._misspellings.update(extra_misspellings)
        self._valid_words.update(self._load_common_words())
        self._valid_words.update(self._load_contractions())
        self._valid_words.update(self._load_word_forms())
        self._valid_words.update(self._load_nltk_words())

    def _load_nltk_words(self) -> Set[str]:
        try:
            import nltk
            from nltk.corpus import words as nltk_words
            base = {w.lower() for w in nltk_words.words()}
            # Generate common inflections from base words
            inflections = set()
            for w in list(base):
                if len(w) < 2:
                    continue
                # Plurals
                if w.endswith("s"):
                    inflections.add(w)
                else:
                    inflections.add(w + "s")
                    if w.endswith(("sh", "ch", "x", "z", "s")):
                        inflections.add(w + "es")
                    elif w.endswith("y") and w[-2] not in "aeiou":
                        inflections.add(w[:-1] + "ies")
                    elif w.endswith("f"):
                        inflections.add(w[:-1] + "ves")
                    elif w.endswith("fe"):
                        inflections.add(w[:-2] + "ves")
                # Verb forms
                if not w.endswith("e"):
                    inflections.add(w + "ed")
                    inflections.add(w + "ing")
                    inflections.add(w + "er")
                    inflections.add(w + "est")
                else:
                    if not w.endswith("ee"):
                        inflections.add(w[:-1] + "ed")
                        inflections.add(w[:-1] + "ing")
                    else:
                        inflections.add(w + "d")
                        inflections.add(w + "ing")
                    inflections.add(w + "r")
                    inflections.add(w + "st")
                # Double consonant -ing/-ed (e.g., stop -> stopping, stopped)
                if len(w) >= 3 and w[-1] in "bcdfgklmnprst" and w[-2] in "aeiou" and w[-3] not in "aeiou":
                    inflections.add(w + w[-1] + "ing")
                    inflections.add(w + w[-1] + "ed")
                # Adjective comparatives/superlatives
                if w.endswith("e"):
                    inflections.add(w + "r")
                    inflections.add(w + "st")
                elif w.endswith("y"):
                    inflections.add(w[:-1] + "ier")
                    inflections.add(w[:-1] + "iest")
                else:
                    inflections.add(w + "er")
                    inflections.add(w + "est")
            return base | inflections
        except Exception:
            return set()

    def _load_contractions(self) -> Set[str]:
        return {
            "don't", "can't", "won't", "isn't", "aren't", "wasn't",
            "weren't", "hasn't", "haven't", "hadn't", "doesn't",
            "didn't", "couldn't", "shouldn't", "wouldn't", "mustn't",
            "needn't", "mightn't", "shalln't", "shan't", "let's",
            "that's", "who's", "what's", "where's", "when's", "how's",
            "there's", "here's", "he's", "she's", "it's", "we're",
            "you're", "they're", "we've", "you've", "they've", "we'll",
            "you'll", "they'll", "he'll", "she'll", "it'll", "i'm",
            "i've", "i'll", "i'd", "you'd", "he'd", "she'd", "we'd",
            "they'd", "it'd", "who'd", "what'd", "where'd", "how'd",
            "i've", "o'clock", "y'all", "ma'am", "ne'er", "e'er",
            "fo'c'sle", "gov'nor",
        }

    def _load_word_forms(self) -> Set[str]:
        extras = {
            "games", "dogs", "cats", "birds", "trees", "boys", "girls",
            "friends", "teachers", "students", "workers", "things",
            "houses", "rooms", "doors", "windows", "books", "tables",
            "chairs", "roads", "paths", "fields", "trees", "rivers",
            "mountains", "islands", "countries", "cities", "towns",
            "plays", "plays", "works", "moves", "lives", "believes",
            "provides", "includes", "continues", "learns", "changes",
            "watches", "follows", "stops", "creates", "offers",
            "remembers", "considers", "appears", "waits", "serves",
            "expects", "stays", "remains", "suggests", "raises",
            "passes", "requires", "reports", "decides", "develops",
            "reaches", "pulls", "kills", "falls", "fights", "rides",
            "shakes", "rises", "hides", "bites", "forgets", "steals",
            "freezes", "wakes", "tries", "wants", "likes", "needs",
            "asks", "says", "tells", "uses", "helps", "shows",
            "hears", "seems", "turns", "sets", "puts", "gets",
            "starts", "opens", "closes", "walks", "talks", "looks",
            "reads", "feels", "becomes", "leaves", "calls", "makes",
            "loves", "hates", "wishes", "hopes", "plans", "orders",
            "forces", "teaches", "studies", "practices", "cooks",
            "bakes", "cleans", "washes", "drives", "flies", "swims",
            "climbs", "jumps", "dances", "sings", "draws", "paints",
            "builds", "fixes", "repairs", "measures", "tests", "checks",
            "walked", "talked", "looked", "read", "felt", "became",
            "left", "called", "made", "loved", "hated", "wished",
            "hoped", "planned", "ordered", "forced", "taught", "studied",
            "practiced", "cooked", "baked", "cleaned", "washed",
            "drove", "flew", "swam", "climbed", "jumped", "danced",
            "sang", "drew", "painted", "built", "fixed", "repaired",
            "measured", "tested", "checked",
            "walking", "talking", "looking", "reading", "feeling",
            "becoming", "leaving", "calling", "making", "loving",
            "hating", "wishing", "hoping", "planning", "ordering",
            "forcing", "teaching", "studying", "practicing", "cooking",
            "baking", "cleaning", "washing", "driving", "flying",
            "swimming", "climbing", "jumping", "dancing", "singing",
            "drawing", "painting", "building", "fixing", "repairing",
            "measuring", "testing", "checking",
            "bigger", "biggest", "smaller", "smallest", "faster",
            "fastest", "slower", "slowest", "harder", "hardest",
            "easier", "easiest", "better", "best", "worse", "worst",
            "longer", "longest", "shorter", "shortest", "taller",
            "tallest", "older", "oldest", "younger", "youngest",
            "newer", "newest", "hotter", "hottest", "colder", "coldest",
            "warmer", "warmest", "cooler", "coolest", "dryer", "driest",
            "wetter", "wettest", "darker", "darkest", "lighter", "lightest",
            "louder", "loudest", "quieter", "quietest", "sweeter",
            "sweetest", "fresher", "freshest", "richer", "richest",
            "poorer", "poorest", "stronger", "strongest", "weaker",
            "weakest", "safer", "safest", "fuller", "fullest",
            "emptier", "emptiest", "cleaner", "cleanest", "dirtier",
            "dirtiest", "rougher", "roughest", "smoother", "smoothest",
            "wider", "widest", "narrower", "narrowest", "deeper",
            "deepest", "thicker", "thickest", "thinner", "thinnest",
            "heavier", "heaviest", "cheaper", "cheapest", "freer",
            "freest", "busier", "busiest", "smarter", "smartest",
            "braver", "bravest", "shyer", "shyest", "prouder",
            "proudest", "angrier", "angriest", "calmer", "calmest",
            "sadder", "saddest", "sicker", "sickest", "prettier",
            "prettiest", "uglier", "ugliest", "funnier", "funniest",
            "seriouer", "seriouser", "kind", "kindest", "ruder",
            "rudest", "wiser", "wisest", "truer", "truest", "falser",
            "falsest", "realler", "realest", "fakse", "faksest",
            "simpler", "simplest", "easier", "easiest", "harder",
            "hardest", "more", "most", "less", "least", "better",
            "best", "worse", "worst", "farther", "farthest",
            "further", "furthest", "closer", "closest", "nearer",
            "nearest", "higher", "highest", "lower", "lowest",
            "larger", "largest", "smaller", "smallest", "fewer",
            "fewest", "many", "much", "several", "enough",
            "features", "feature", "learning", "learned", "perfectly",
            "perfect", "perfects", "phone", "phones", "phones",
            "message", "messages", "content", "contents", "correct",
            "corrects", "error", "errors", "check", "checks",
            "coming", "comes", "said", "this", "friend", "friends",
            "fast", "faster", "fastest",
        }
        return extras

    def _load_common_words(self) -> Set[str]:
        extras = {
            "fox", "jump", "jumps", "lazy", "brown", "quick", "dog", "cat",
            "honest", "apple", "hour", "person", "john", "india", "happy",
            "really", "big", "was", "decided", "hardly", "play", "plays",
            "playing", "played", "goes", "going", "went", "gone", "ate",
            "eaten", "eat", "saw", "see", "seen", "gave", "given", "give",
            "took", "taken", "take", "knew", "known", "know", "thought",
            "think", "made", "make", "ran", "run", "runs", "running",
            "wrote", "written", "write", "spoke", "spoken", "speak",
            "drove", "driven", "drive", "broke", "broken", "break",
            "chose", "chosen", "choose", "did", "done", "do", "does",
            "doing", "flew", "flown", "fly", "grew", "grown", "grow",
            "threw", "thrown", "throw", "wore", "worn", "wear",
            "drew", "drawn", "draw", "began", "begun", "begin", "drank",
            "drunk", "drink", "rang", "rung", "ring", "swam", "swum",
            "swim", "sang", "sung", "sing", "sat", "sit", "sits",
            "stood", "stand", "stands", "found", "find", "finds", "felt",
            "feel", "feels", "kept", "keep", "keeps", "slept", "sleep",
            "sleeps", "left", "leave", "leaves", "brought", "bring",
            "brings", "bought", "buy", "buys", "caught", "catch",
            "catches", "taught", "teach", "teaches", "told", "tell",
            "tells", "sold", "sell", "sells", "spent", "spend", "spends",
            "built", "build", "builds", "sent", "send", "sends", "won",
            "win", "wins", "lost", "lose", "loses", "held", "hold",
            "holds", "met", "meet", "meets", "put", "puts", "cut",
            "cuts", "cost", "costs", "hit", "hits", "let", "lets",
            "shut", "shuts", "hurt", "hurts",
            "played", "playing", "coming", "doing", "being", "getting",
            "making", "taking", "saying", "telling", "giving", "finding",
            "thinking", "keeping", "leaving", "bringing", "buying",
            "catching", "teaching", "selling", "spending", "building",
            "sending", "winning", "losing", "holding", "meeting",
            "reading", "putting", "cutting", "hurting", "starting",
            "stopping", "opening", "closing", "changing", "running",
            "walking", "talking", "looking", "helping", "working",
            "sitting", "standing", "lying", "dying",
            "good", "great", "nice", "bad", "best", "worst", "better",
            "worse", "fast", "slow", "quick", "hard", "soft", "long",
            "short", "tall", "small", "large", "old", "young", "new",
            "hot", "cold", "warm", "cool", "dry", "wet", "dark",
            "light", "bright", "loud", "quiet", "sweet", "fresh",
            "rich", "poor", "strong", "weak", "safe", "full", "empty",
            "clean", "dirty", "rough", "smooth", "wide", "narrow",
            "deep", "thick", "thin", "heavy", "cheap", "free", "busy",
            "smart", "brave", "shy", "proud", "angry", "calm",
            "sad", "sick", "beautiful", "ugly", "pretty", "funny",
            "serious", "kind", "rude", "wise", "honest", "true",
            "false", "real", "fake", "simple", "easy", "difficult",
            "possible", "necessary", "useful", "important", "famous",
            "ordinary", "special", "normal", "strange", "usual",
            "common", "rare", "general", "specific", "exact", "clear",
            "obvious", "modern", "foreign", "native", "local",
            "huge", "tiny", "massive", "wild", "tame", "silent",
            "noisy", "fancy", "royal", "noble", "sacred", "holy",
            "earth", "sky", "sea", "ocean", "river", "lake",
            "mountain", "hill", "valley", "forest", "desert", "island",
            "beach", "coast", "field", "garden", "park", "road",
            "street", "path", "bridge", "building", "house", "home",
            "room", "door", "window", "wall", "floor", "roof",
            "table", "chair", "bed", "desk", "kitchen", "bathroom",
            "bedroom", "yard", "morning", "evening", "night",
            "today", "tomorrow", "yesterday", "always", "never",
            "sometimes", "often", "usually", "already", "still",
            "just", "only", "also", "too", "very", "quite", "rather",
            "almost", "nearly", "enough", "perhaps", "maybe",
            "actually", "truly", "about", "above", "across", "after",
            "against", "along", "among", "around", "before", "behind",
            "below", "beside", "between", "beyond", "during", "except",
            "inside", "outside", "near", "off", "past", "since",
            "through", "toward", "under", "until", "upon", "within",
            "without", "however", "therefore", "moreover",
            "furthermore", "nevertheless", "meanwhile", "although",
            "though", "unless", "where", "when", "how", "why", "who",
            "whom", "whose", "which", "that", "this", "these",
            "those", "what", "everyone", "someone", "anyone", "nobody",
            "somebody", "anybody", "everything", "something", "nothing",
            "each", "every", "either", "neither", "both", "few",
            "many", "much", "several", "all", "some", "none", "most",
            "less", "another", "other", "such", "own", "same",
            "different", "else", "even", "yet", "ago", "if",
            "because", "as", "like", "than", "besides", "but",
            "not", "once", "twice", "half", "double", "pair",
            "am", "are", "were", "been", "being", "can", "could",
            "will", "would", "shall", "should", "may", "might",
            "must", "need", "used", "gets", "got", "gotten",
            "gets", "got", "gotten", "seem", "seemed", "hear",
            "heard", "move", "moved", "live", "lived", "believe",
            "believed", "provide", "provided", "include", "included",
            "continue", "continued", "learn", "learned", "change",
            "changed", "lead", "led", "understand", "understood",
            "watch", "watched", "follow", "followed", "stop", "stopped",
            "create", "created", "spend", "spent", "offer", "offered",
            "remember", "remembered", "consider", "considered",
            "appear", "appeared", "wait", "waited", "serve", "served",
            "expect", "expected", "stay", "stayed", "remain", "remained",
            "suggest", "suggested", "raise", "raised", "pass", "passed",
            "require", "required", "report", "reported", "decide",
            "decided", "develop", "developed", "reach", "reached",
            "pull", "pulled", "kill", "killed", "fall", "fell",
            "fallen", "fight", "fought", "draw", "drew", "drawn",
            "ride", "rode", "ridden", "shake", "shook", "shaken",
            "rise", "rose", "risen", "hide", "hid", "hidden",
            "bite", "bit", "bitten", "teach", "taught", "forget",
            "forgot", "forgotten", "forgive", "forgave", "forgiven",
            "steal", "stole", "stolen", "freeze", "froze", "frozen",
            "blow", "blew", "blown", "wake", "woke", "woken",
            "stole", "stolen", "freeze", "froze", "frozen",
        }
        return extras

    def add_custom_word(self, word: str):
        self._custom_words.add(word.lower())

    def _is_ignored(self, word: str, pos: int, full_text: str) -> bool:
        for pat in self._ignored_patterns:
            if pat.search(word):
                return True
        return False

    def _edit_distance_one(self, word: str) -> Set[str]:
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = {L + R[1:] for L, R in splits if R}
        transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
        replaces = {L + c + R[1:] for L, R in splits if R for c in letters}
        inserts = {L + c + R for L, R in splits for c in letters}
        return deletes | transposes | replaces | inserts

    def _known(self, words: Set[str]) -> Set[str]:
        return words & (self._valid_words | self._custom_words)

    def _candidates(self, word: str) -> Set[str]:
        known = self._known({word})
        if known:
            return known
        e1 = self._edit_distance_one(word)
        known1 = self._known(e1)
        if known1:
            return known1
        e2 = set()
        for w in e1:
            e2 |= self._edit_distance_one(w)
        known2 = self._known(e2)
        if known2:
            return known2
        return {word}

    def _phonetic_key(self, word: str) -> str:
        w = word.lower()
        w = re.sub(r"[^a-z]", "", w)
        if not w:
            return ""
        key = w[0]
        prev = w[0]
        for c in w[1:]:
            consonants = "bdfgjklmnpqrstvwxyz"
            if c in consonants and c != prev:
                key += c
            prev = c
        return key

    def _phonetic_candidates(self, word: str) -> List[str]:
        key = self._phonetic_key(word)
        if not key:
            return []
        candidates = []
        for vw in self._valid_words:
            if self._phonetic_key(vw) == key and len(vw) > 2:
                candidates.append(vw)
        return candidates[:5]

    def _is_url_or_email(self, word: str) -> bool:
        return bool(re.match(r"https?://|www\.|@|\.com|\.org|\.net|\.io", word.lower()))

    def _is_number(self, word: str) -> bool:
        return bool(re.match(r"^[0-9]+\.?[0-9]*[%$]?$", word))

    def check(self, text: str) -> List[Dict]:
        errors = []
        if not text or not text.strip():
            return errors

        words_with_pos = []
        for m in re.finditer(r"[a-zA-Z']+", text):
            words_with_pos.append((m.group(), m.start(), m.end()))

        for word, start, end in words_with_pos:
            lower = word.lower()

            if self._is_ignored(word, start, text):
                continue
            if self._is_url_or_email(word):
                continue
            if self._is_number(word):
                continue
            if lower in self._custom_words:
                continue
            if len(lower) <= 1:
                continue

            if lower in self._misspellings:
                correction = self._misspellings[lower]
                if correction.lower() != lower:
                    errors.append(make_error(
                        text=word, start=start, end=end,
                        category="SPELLING",
                        message=f'"{word}" is misspelled. Did you mean "{correction}"?',
                        suggestion=correction,
                        severity="HIGH",
                        confidence=0.98,
                        explanation=f'The correct spelling is "{correction}".',
                    ))
                continue

            if lower in self._valid_words:
                continue

            candidates = self._candidates(lower)
            phonetic = self._phonetic_candidates(lower)
            all_candidates = list(candidates - {lower}) + [
                p for p in phonetic if p not in candidates
            ]

            if all_candidates:
                best = min(all_candidates, key=lambda c: self._edit_distance(lower, c))
                dist = self._edit_distance(lower, best)
                if dist == 1 and len(lower) >= 4:
                    conf = 0.92
                    errors.append(make_error(
                        text=word, start=start, end=end,
                        category="SPELLING",
                        message=f'"{word}" may be misspelled. Did you mean "{best}"?',
                        suggestion=best,
                        severity="HIGH",
                        confidence=conf,
                        explanation=f'The closest valid word is "{best}".',
                    ))
                elif dist == 2 and len(lower) >= 5:
                    conf = 0.75
                    errors.append(make_error(
                        text=word, start=start, end=end,
                        category="SPELLING",
                        message=f'"{word}" may be misspelled. Did you mean "{best}"?',
                        suggestion=best,
                        severity="MEDIUM",
                        confidence=conf,
                        explanation=f'The closest valid word is "{best}".',
                    ))
        return errors

    def _edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) > 30 or len(s2) > 30:
            return 3
        if abs(len(s1) - len(s2)) > 3:
            return 3
        d = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(len(s1) + 1):
            d[i][0] = i
        for j in range(len(s2) + 1):
            d[0][j] = j
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + cost,
                )
        return d[len(s1)][len(s2)]


# ---------------------------------------------------------------------------
# Grammar engine
# ---------------------------------------------------------------------------
class GrammarEngine:
    SINGULAR_PRP = {"he", "she", "it", "this", "that", "everyone", "everybody",
                    "someone", "somebody", "anyone", "anybody", "nobody",
                    "nothing", "everything", "each", "every", "either",
                    "neither", "no one"}
    PLURAL_PRP = {"they", "we", "these", "those"}
    misspelled_words: Set[str] = set()
    BE_FORMS = {
        "i": "am", "he": "is", "she": "is", "it": "is",
        "we": "are", "they": "are", "you": "are",
        "this": "is", "that": "is", "these": "are", "those": "are",
        "everyone": "is", "somebody": "is", "nobody": "is",
        "each": "is", "every": "is", "nothing": "is",
    }
    BE_PAST = {
        "i": "was", "he": "was", "she": "was", "it": "was",
        "we": "were", "they": "were", "you": "were",
        "this": "was", "that": "was", "these": "were", "those": "were",
        "everyone": "was", "somebody": "was", "nobody": "was",
        "each": "was", "every": "was", "nothing": "was",
    }
    HAVE_FORMS = {
        "i": "have", "he": "has", "she": "has", "it": "has",
        "we": "have", "they": "have", "you": "have",
        "everyone": "has", "somebody": "has", "nobody": "has",
        "each": "has", "every": "has", "nothing": "has",
    }
    DO_FORMS = {
        "i": "do", "he": "does", "she": "does", "it": "does",
        "we": "do", "they": "do", "you": "do",
        "everyone": "does", "somebody": "does", "nobody": "does",
        "each": "does", "every": "does", "nothing": "does",
    }
    IRREGULAR_PAST = {
        "go": "went", "eat": "ate", "see": "saw", "come": "came",
        "take": "took", "give": "gave", "know": "knew",
        "think": "thought", "make": "made", "run": "ran",
        "write": "wrote", "speak": "spoke", "drive": "drove",
        "break": "broke", "choose": "chose", "do": "did",
        "fly": "flew", "grow": "grew", "throw": "threw",
        "wear": "wore", "draw": "drew", "begin": "began",
        "drink": "drank", "ring": "rang", "swim": "swam",
        "sing": "sang", "sit": "sat", "stand": "stood",
        "find": "found", "feel": "felt", "keep": "kept",
        "sleep": "slept", "leave": "left", "bring": "brought",
        "buy": "bought", "catch": "caught", "teach": "taught",
        "tell": "told", "sell": "sold", "spend": "spent",
        "build": "built", "send": "sent", "win": "won",
        "lose": "lost", "hold": "held", "meet": "met",
        "read": "read", "put": "put", "cut": "cut",
        "cost": "cost", "hit": "hit", "let": "let",
        "set": "set", "shut": "shut", "hurt": "hurt",
    }
    IRREGULAR_PAST_PART = {
        "go": "gone", "eat": "eaten", "see": "seen", "come": "come",
        "take": "taken", "give": "given", "know": "known",
        "think": "thought", "make": "made", "run": "run",
        "write": "written", "speak": "spoken", "drive": "driven",
        "break": "broken", "choose": "chosen", "do": "done",
        "fly": "flown", "grow": "grown", "throw": "thrown",
        "wear": "worn", "draw": "drawn", "begin": "begun",
        "drink": "drunk", "ring": "rung", "swim": "swum",
        "sing": "sung", "sit": "sat", "stand": "stood",
        "find": "found", "feel": "felt", "keep": "kept",
        "sleep": "slept", "leave": "left", "bring": "brought",
        "buy": "bought", "catch": "caught", "teach": "taught",
        "tell": "told", "sell": "sold", "spend": "spent",
        "build": "built", "send": "sent", "win": "won",
        "lose": "lost", "hold": "held", "meet": "met",
        "read": "read", "shake": "shaken", "take": "taken",
    }
    IRREGULAR_PLURAL = {
        "child": "children", "man": "men", "woman": "women",
        "foot": "feet", "tooth": "teeth", "mouse": "mice",
        "person": "people", "ox": "oxen", "goose": "geese",
    }
    IRREGULAR_SINGULAR = {v: k for k, v in IRREGULAR_PLURAL.items()}
    MODALS = {"can", "could", "will", "would", "shall", "should",
              "may", "might", "must"}
    COMMON_PREPOSITIONS = {
        "good at", "bad at", "interested in", "afraid of",
        "fond of", "proud of", "capable of", "responsible for",
        "dependent on", "different from", "listen to", "look at",
        "wait for", "belong to", "consist of", "depend on",
        "succeed in", "participate in", "believe in", "result in",
        "concentrate on", "insist on", "rely on", "agree with",
        "apologize for", "apply for", "argue with", "ask for",
        "believe in", "care about", "complain about", "congratulate on",
        "consist of", "dream about", "explain to", "feel about",
        "hear about", "hope for", "know about", "laugh at",
        "learn about", "look after", "look for", "pay for",
        "plan for", "point at", "prepare for", "provide for",
        "refer to", "remember to", "respond to", "search for",
        "smile at", "think about", "think of", "worry about",
    }
    CONFUSING_WORDS = {
        "your": ("you're", "possessive vs contraction", "Use 'you're' for 'you are'."),
        "you're": ("your", "contraction vs possessive", "Use 'your' for possession."),
        "their": ("they're", "possessive vs contraction", "Use 'they're' for 'they are'."),
        "they're": ("their", "contraction vs possessive", "Use 'their' for possession."),
        "there": ("their", "location vs possessive", "Use 'their' for possession."),
        "its": ("it's", "possessive vs contraction", "Use 'it's' for 'it is'."),
        "it's": ("its", "contraction vs possessive", "Use 'its' for possession."),
        "then": ("than", "sequence vs comparison", "Use 'than' for comparisons."),
        "than": ("then", "comparison vs sequence", "Use 'then' for time sequence."),
        "affect": ("effect", "verb vs noun", "Use 'effect' as a noun."),
        "effect": ("affect", "noun vs verb", "Use 'affect' as a verb."),
        "accept": ("except", "receive vs exclude", "Use 'except' for exclusion."),
        "except": ("accept", "exclusion vs receive", "Use 'accept' for receiving."),
        "loose": ("lose", "not tight vs misplace", "Use 'lose' for misplacing."),
        "lose": ("loose", "misplace vs not tight", "Use 'loose' for not tight."),
        "weather": ("whether", "climate vs if", "Use 'whether' for 'if'."),
        "whether": ("weather", "if vs climate", "Use 'weather' for climate."),
        "advice": ("advise", "noun vs verb", "Use 'advise' as a verb."),
        "advise": ("advice", "verb vs noun", "Use 'advice' as a noun."),
        "too": ("to", "excess/also vs preposition", "Use 'to' as a preposition."),
        "to": ("too", "preposition vs excess", "Use 'too' for excess."),
        "their": ("there", "possessive vs location", "Use 'there' for location."),
        "principle": ("principal", "rule vs person", "Use 'principal' for a person/leader."),
        "principal": ("principle", "person vs rule", "Use 'principle' for a rule."),
        "compliment": ("complement", "praise vs complete", "Use 'complement' to mean complete."),
        "complement": ("compliment", "complete vs praise", "Use 'compliment' for praise."),
        "stationary": ("stationery", "still vs writing paper", "Use 'stationery' for writing paper."),
        "stationery": ("stationary", "writing paper vs still", "Use 'stationary' for not moving."),
    }

    def check(self, text: str) -> List[Dict]:
        errors = []
        if not text or not text.strip():
            return errors
        sentences = self._split_sentences(text)
        for sent_text in sentences:
            sent_start = text.find(sent_text)
            errors.extend(self._check_subject_verb(sent_text, sent_start))
            errors.extend(self._check_verb_tense(sent_text, sent_start))
            errors.extend(self._check_articles(sent_text, sent_start))
            errors.extend(self._check_pronouns(sent_text, sent_start))
            errors.extend(self._check_double_negatives(sent_text, sent_start))
            errors.extend(self._check_confusing_words(sent_text, sent_start))
            errors.extend(self._check_missing_verb(sent_text, sent_start))
            errors.extend(self._check_extra_words(sent_text, sent_start))
            errors.extend(self._check_redundancy(sent_text, sent_start))
            errors.extend(self._check_plural_singular(sent_text, sent_start))
        return errors

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p for p in parts if p.strip()]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z']+", text)

    def _get_subject_verb_pairs(self, tokens: List[str]) -> List[Tuple[str, str, int, int]]:
        pairs = []
        i = 0
        while i < len(tokens):
            t = tokens[i].lower()
            if t in self.SINGULAR_PRP or t in self.PLURAL_PRP:
                for j in range(i + 1, min(i + 5, len(tokens))):
                    v = tokens[j].lower()
                    if v in ("is", "am", "are", "was", "were"):
                        pairs.append((t, v, i, j))
                        break
                    if v in ("has", "have"):
                        pairs.append((t, v, i, j))
                        break
                    if v in ("do", "does", "don't", "dont", "doesn't", "doesnt"):
                        pairs.append((t, v, i, j))
                        break
                    if v in self.MODALS:
                        break
                    if self._looks_like_verb(v):
                        pairs.append((t, v, i, j))
                        break
                    if v not in ("not", "n't", "never", "really", "very", "also",
                                  "always", "often", "sometimes", "usually", "just",
                                  "only", "even", "still", "already", "yet",
                                  "dont", "don't", "doesnt", "doesn't",
                                  "didnt", "didn't", "cant", "can't",
                                  "wont", "won't", "isnt", "isn't",
                                  "arent", "aren't", "wasnt", "wasn't",
                                  "werent", "weren't", "hasnt", "hasn't",
                                  "havent", "haven't", "hadnt", "hadn't",
                                  "wouldnt", "wouldn't", "couldnt", "couldn't",
                                  "shouldnt", "shouldn't", "mustnt", "mustn't"):
                        break
            else:
                is_det = t in ("a", "an", "the", "my", "your", "his", "her",
                               "its", "our", "their", "this", "that", "these",
                               "those", "some", "no", "every", "each", "any")
                is_adj = (len(t) > 3 and t.endswith(("ful", "ous", "ive", "ing",
                          "ent", "ant", "ible", "able", "ial", "al"))) or t in (
                    "big", "old", "new", "hot", "cold", "fast", "slow", "hard",
                    "soft", "long", "short", "tall", "good", "bad", "nice",
                    "great", "little", "small", "large", "young", "happy",
                    "beautiful", "important", "different", "possible", "real",
                    "human", "own", "other", "same", "next", "last", "first")
                noun_idx = -1
                if is_det:
                    for k in range(i + 1, min(i + 4, len(tokens))):
                        tk = tokens[k].lower()
                        if tk in ("is", "am", "are", "was", "were"):
                            break
                        if self._looks_like_verb(tk):
                            break
                        noun_idx = k
                        break
                elif t[0].isupper() and len(t) > 1 and t.isalpha() and t.lower() not in self.MODALS:
                    noun_idx = i

                if noun_idx >= 0:
                    subj = tokens[noun_idx].lower()
                    for j in range(noun_idx + 1, min(noun_idx + 3, len(tokens))):
                        v = tokens[j].lower()
                        if v in ("is", "am", "are", "was", "were"):
                            pairs.append((subj, v, noun_idx, j))
                            break
                        if v in ("has", "have"):
                            pairs.append((subj, v, noun_idx, j))
                            break
                        if v in ("do", "does", "don't", "dont", "doesn't", "doesnt"):
                            pairs.append((subj, v, noun_idx, j))
                            break
                        if v in self.MODALS:
                            break
                        if self._looks_like_verb(v):
                            pairs.append((subj, v, noun_idx, j))
                            break
                        if v not in ("not", "n't", "never", "really", "very",
                                      "also", "always", "often", "usually"):
                            break
            i += 1
        return pairs

    def _looks_like_verb(self, word: str) -> bool:
        w = word.lower()
        if w in self.MODALS:
            return True
        if w in ("is", "am", "are", "was", "were", "be", "been", "being"):
            return True
        if w in ("has", "have", "had"):
            return True
        if w in ("do", "does", "did"):
            return True
        if w.endswith("ing") and len(w) > 4:
            return True
        if w.endswith("ed") and len(w) > 3:
            return True
        if w.endswith("es") and len(w) > 3:
            return True
        if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
            return True
        base = w.rstrip("s")
        if base in self.IRREGULAR_PAST or base in self.IRREGULAR_PAST_PART:
            return True
        if w in self.IRREGULAR_PAST or w in self.IRREGULAR_PAST_PART:
            return True
        if w in set(self.IRREGULAR_PAST.values()) or w in set(self.IRREGULAR_PAST_PART.values()):
            return True
        if w in self._COMMON_BASE_VERBS:
            return True
        return False

    _COMMON_BASE_VERBS = {
        "go", "eat", "see", "come", "take", "give", "know", "think",
        "make", "run", "write", "speak", "drive", "break", "choose",
        "fly", "grow", "throw", "wear", "draw", "begin", "drink",
        "ring", "swim", "sing", "sit", "stand", "find", "feel",
        "keep", "sleep", "leave", "bring", "buy", "catch", "teach",
        "tell", "sell", "spend", "build", "send", "win", "lose",
        "hold", "meet", "put", "cut", "hit", "let", "shut", "hurt",
        "play", "work", "move", "live", "believe", "provide", "include",
        "continue", "learn", "change", "watch", "follow", "stop",
        "create", "offer", "remember", "consider", "appear", "wait",
        "serve", "expect", "stay", "remain", "suggest", "raise",
        "pass", "require", "report", "decide", "develop", "reach",
        "pull", "kill", "fall", "fight", "ride", "shake", "rise",
        "hide", "bite", "forget", "forgive", "steal", "freeze", "wake",
        "try", "want", "like", "need", "ask", "say", "tell", "use",
        "help", "show", "hear", "seem", "turn", "set", "put", "get",
        "set", "start", "open", "close", "walk", "talk", "look",
        "read", "feel", "become", "leave", "call", "make", "be",
        "have", "do", "may", "might", "must", "shall", "should",
        "will", "would", "can", "could", "dare", "ought", "need",
        "love", "hate", "want", "wish", "hope", "plan", "try",
        "ask", "order", "force", "help", "show", "teach", "study",
        "practice", "cook", "bake", "clean", "wash", "drive", "fly",
        "swim", "climb", "jump", "dance", "sing", "draw", "paint",
        "build", "fix", "repair", "paint", "measure", "test", "check",
    }

    def _check_subject_verb(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        pairs = self._get_subject_verb_pairs(tokens)
        for subj, verb, si, vi in pairs:
            word = tokens[vi]
            pos_in_text = self._word_position(sent, vi, sent_start)

            is_singular_subj = (subj in self.SINGULAR_PRP or
                (subj not in self.PLURAL_PRP and
                 not (subj.endswith("s") and not subj.endswith("ss") and len(subj) > 2)))
            is_plural_subj = (subj in self.PLURAL_PRP or
                (subj.endswith("s") and not subj.endswith("ss") and len(subj) > 2))

            if verb in ("is", "was"):
                if subj in self.PLURAL_PRP:
                    expected = "are" if verb == "is" else "were"
                    errors.append(make_error(
                        text=word, start=pos_in_text, end=pos_in_text + len(word),
                        category="GRAMMAR",
                        message=f'The subject "{subj}" requires "{expected}", not "{verb}".',
                        suggestion=expected,
                        severity="HIGH",
                        confidence=0.95,
                        context=sent,
                        explanation=f'"{subj}" is plural and requires the plural form of "be".',
                    ))
            elif verb == "are":
                if subj in self.SINGULAR_PRP:
                    expected = "is"
                    errors.append(make_error(
                        text=word, start=pos_in_text, end=pos_in_text + len(word),
                        category="GRAMMAR",
                        message=f'The subject "{subj}" requires "{expected}", not "{verb}".',
                        suggestion=expected,
                        severity="HIGH",
                        confidence=0.95,
                        context=sent,
                    ))
            elif verb == "am":
                if subj != "i":
                    errors.append(make_error(
                        text=word, start=pos_in_text, end=pos_in_text + len(word),
                        category="GRAMMAR",
                        message=f'"am" is only used with "I". Use "is" or "are" with "{subj}".',
                        suggestion="is" if subj in self.SINGULAR_PRP else "are",
                        severity="HIGH",
                        confidence=0.95,
                        context=sent,
                    ))
            elif verb == "were":
                if subj in ("he", "she", "it", "this", "that"):
                    expected = "was"
                    errors.append(make_error(
                        text=word, start=pos_in_text, end=pos_in_text + len(word),
                        category="GRAMMAR",
                        message=f'The subject "{subj}" requires "{expected}", not "{verb}".',
                        suggestion=expected,
                        severity="HIGH",
                        confidence=0.95,
                        context=sent,
                    ))

            if verb == "have" and is_singular_subj:
                errors.append(make_error(
                    text=word, start=pos_in_text, end=pos_in_text + len(word),
                    category="GRAMMAR",
                    message=f'The subject "{subj}" requires "has", not "have".',
                    suggestion="has",
                    severity="HIGH",
                    confidence=0.90,
                    context=sent,
                ))
            elif verb == "has" and (subj in ("i", "we", "they", "you") or is_plural_subj):
                errors.append(make_error(
                    text=word, start=pos_in_text, end=pos_in_text + len(word),
                    category="GRAMMAR",
                    message=f'The subject "{subj}" requires "have", not "has".',
                    suggestion="have",
                    severity="HIGH",
                    confidence=0.95,
                    context=sent,
                ))

            if verb == "does" and subj in ("i", "we", "they", "you"):
                errors.append(make_error(
                    text=word, start=pos_in_text, end=pos_in_text + len(word),
                    category="GRAMMAR",
                    message=f'The subject "{subj}" requires "do", not "does".',
                    suggestion="do",
                    severity="HIGH",
                    confidence=0.95,
                    context=sent,
                ))
            elif verb == "do" and subj in ("he", "she", "it"):
                errors.append(make_error(
                    text=word, start=pos_in_text, end=pos_in_text + len(word),
                    category="GRAMMAR",
                    message=f'The subject "{subj}" requires "does", not "do".',
                    suggestion="does",
                    severity="HIGH",
                    confidence=0.95,
                    context=sent,
                ))
            elif verb in ("don't", "dont") and is_singular_subj:
                errors.append(make_error(
                    text=word, start=pos_in_text, end=pos_in_text + len(word),
                    category="GRAMMAR",
                    message=f'The subject "{subj}" requires "doesn\'t", not "don\'t".',
                    suggestion="doesn't",
                    severity="HIGH",
                    confidence=0.95,
                    context=sent,
                    explanation=f'Singular subjects (he/she/it) require "doesn\'t", not "don\'t".',
                ))
            elif verb in ("don't", "dont") and is_plural_subj:
                errors.append(make_error(
                    text=word, start=pos_in_text, end=pos_in_text + len(word),
                    category="GRAMMAR",
                    message=f'The subject "{subj}" requires "don\'t", not "doesn\'t".',
                    suggestion="don't",
                    severity="HIGH",
                    confidence=0.95,
                    context=sent,
                ))

            if is_singular_subj and verb not in (
                "is", "am", "are", "was", "were", "has", "have", "had",
                "do", "does", "did",
                "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
            ) and verb in self._COMMON_BASE_VERBS:
                if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
                    expected = verb + "es"
                elif verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
                    expected = verb[:-1] + "ies"
                else:
                    expected = verb + "s"
                errors.append(make_error(
                    text=word, start=pos_in_text, end=pos_in_text + len(word),
                    category="GRAMMAR",
                    message=f'The subject "{subj}" requires "{expected}", not "{verb}".',
                    suggestion=expected,
                    severity="HIGH",
                    confidence=0.90,
                    context=sent,
                    explanation=f'Singular subjects (he/she/it) require verbs ending in -s/-es.',
                ))

            if is_plural_subj and verb not in (
                "is", "am", "are", "was", "were", "has", "have", "had",
                "do", "does", "did",
            ) and verb.endswith("s") and not verb.endswith("ss"):
                if verb.endswith("ies"):
                    expected = verb[:-3] + "y"
                elif verb.endswith("es"):
                    expected = verb[:-2]
                else:
                    expected = verb[:-1]
                if expected in self._COMMON_BASE_VERBS:
                    errors.append(make_error(
                        text=word, start=pos_in_text, end=pos_in_text + len(word),
                        category="GRAMMAR",
                        message=f'The subject "{subj}" requires "{expected}", not "{verb}".',
                        suggestion=expected,
                        severity="HIGH",
                        confidence=0.90,
                        context=sent,
                        explanation=f'Plural subjects (they/we) require the base form of the verb.',
                    ))
        return errors

    def _check_verb_tense(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]

        for i, t in enumerate(lower_tokens):
            if t in ("didn't", "didnt", "did", "don't", "dont", "doesn't", "doesnt"):
                for j in range(i + 1, min(i + 4, len(lower_tokens))):
                    nt = lower_tokens[j]
                    if nt in ("not", "n't", "never"):
                        continue
                    if t in ("didn't", "didnt", "did", "don't", "dont", "doesn't", "doesnt"):
                        base = None
                        for base_form, past in self.IRREGULAR_PAST.items():
                            if past == nt:
                                base = base_form
                                break
                        if not base:
                            for base_form, pp in self.IRREGULAR_PAST_PART.items():
                                if pp == nt:
                                    base = base_form
                                    break
                        if base:
                            word = tokens[j]
                            pos = self._word_position(sent, j, sent_start)
                            errors.append(make_error(
                                text=word, start=pos, end=pos + len(word),
                                category="GRAMMAR",
                                message=f'After "didn\'t", use the base form "{base}", not "{nt}".',
                                suggestion=base,
                                severity="HIGH",
                                confidence=0.95,
                                context=sent,
                                explanation='After "did/didn\'t", use the base form of the verb.',
                            ))
                        if nt.endswith("ed") and len(nt) > 3:
                            base_form = nt[:-2]
                            if nt.endswith("ied"):
                                base_form = nt[:-3] + "y"
                            elif nt.endswith("pped") or nt.endswith("tted") or nt.endswith("nned"):
                                base_form = nt[:-3]
                            if base_form != base:
                                word = tokens[j]
                                pos = self._word_position(sent, j, sent_start)
                                errors.append(make_error(
                                    text=word, start=pos, end=pos + len(word),
                                    category="GRAMMAR",
                                    message=f'After "didn\'t", use the base form, not the past tense "{nt}".',
                                    suggestion=base_form,
                                    severity="HIGH",
                                    confidence=0.85,
                                    context=sent,
                                ))
                        break
                    break

        for i, t in enumerate(lower_tokens):
            if t in self.MODALS:
                for j in range(i + 1, min(i + 4, len(lower_tokens))):
                    nt = lower_tokens[j]
                    if nt in ("not", "n't", "never"):
                        continue
                    if nt.endswith("s") and not nt.endswith("ss") and len(nt) > 3:
                        if nt.endswith("es"):
                            base = nt[:-2]
                        else:
                            base = nt[:-1]
                        word = tokens[j]
                        pos = self._word_position(sent, j, sent_start)
                        errors.append(make_error(
                            text=word, start=pos, end=pos + len(word),
                            category="GRAMMAR",
                            message=f'After "{t}", use the base form "{base}", not "{nt}".',
                            suggestion=base,
                            severity="HIGH",
                            confidence=0.90,
                            context=sent,
                            explanation=f'After modal verbs, use the base form of the verb.',
                        ))
                    break

        for i, t in enumerate(lower_tokens):
            if t in ("is", "am", "are", "was", "were"):
                for j in range(i + 1, min(i + 4, len(lower_tokens))):
                    nt = lower_tokens[j]
                    if nt in ("not", "n't", "never", "also", "always", "still",
                              "already", "just", "only", "even", "very", "really",
                              "quite", "rather", "too"):
                        continue
                    if self._looks_like_verb(nt) and not nt.endswith("ing"):
                        if t in ("is", "am", "are"):
                            word = tokens[j]
                            pos = self._word_position(sent, j, sent_start)
                            ing_form = self._to_ing(nt)
                            errors.append(make_error(
                                text=word, start=pos, end=pos + len(word),
                                category="GRAMMAR",
                                message=f'After "{t}", use the -ing form (e.g., "{ing_form}").',
                                suggestion=ing_form,
                                severity="MEDIUM",
                                confidence=0.75,
                                context=sent,
                                explanation=f'After "is/am/are", use the present participle (-ing form).',
                            ))
                    break
        return errors

    def _to_ing(self, verb: str) -> str:
        v = verb.lower()
        if v.endswith("e") and not v.endswith("ee") and not v.endswith("ye") and len(v) > 2:
            return v[:-1] + "ing"
        if re.search(r"[^aeiou][aeiou][^aeiouwxy]$", v) and len(v) > 3:
            return v + v[-1] + "ing"
        return v + "ing"

    def _check_articles(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]
        vowels = set("aeiou")
        silent_h = {"hour", "honest", "honor", "honour", "heir", "herb", "homage"}
        y_sound_u = {"university", "uniform", "unique", "united", "unit",
                     "universal", "union", "usage", "useful", "usual",
                     "usually", "utensil", "utility"}
        consonant_sound_o = {"one", "once", "unanimous"}

        for i, t in enumerate(lower_tokens):
            if t == "a" and i + 1 < len(lower_tokens):
                nxt = lower_tokens[i + 1]
                if nxt in silent_h or nxt in consonant_sound_o:
                    word = tokens[i]
                    pos = self._word_position(sent, i, sent_start)
                    errors.append(make_error(
                        text=word, start=pos, end=pos + len(word),
                        category="GRAMMAR",
                        message=f'Use "an" before "{nxt}" (vowel sound).',
                        suggestion="an",
                        severity="HIGH",
                        confidence=0.95,
                        context=sent,
                        explanation=f'"{nxt}" starts with a vowel sound, so use "an".',
                    ))
                elif nxt in y_sound_u:
                    pass
                elif nxt and nxt[0] in vowels and nxt not in y_sound_u:
                    word = tokens[i]
                    pos = self._word_position(sent, i, sent_start)
                    errors.append(make_error(
                        text=word, start=pos, end=pos + len(word),
                        category="GRAMMAR",
                        message=f'Use "an" before "{nxt}" (vowel sound).',
                        suggestion="an",
                        severity="HIGH",
                        confidence=0.92,
                        context=sent,
                    ))
            elif t == "an" and i + 1 < len(lower_tokens):
                nxt = lower_tokens[i + 1]
                if nxt in consonant_sound_o:
                    pass
                elif nxt in y_sound_u:
                    word = tokens[i]
                    pos = self._word_position(sent, i, sent_start)
                    errors.append(make_error(
                        text=word, start=pos, end=pos + len(word),
                        category="GRAMMAR",
                        message=f'Use "a" before "{nxt}" (consonant sound).',
                        suggestion="a",
                        severity="HIGH",
                        confidence=0.92,
                        context=sent,
                        explanation=f'"{nxt}" starts with a "y" sound (consonant), so use "a".',
                    ))
                elif nxt and nxt[0] not in vowels and nxt not in silent_h:
                    word = tokens[i]
                    pos = self._word_position(sent, i, sent_start)
                    errors.append(make_error(
                        text=word, start=pos, end=pos + len(word),
                        category="GRAMMAR",
                        message=f'Use "a" before "{nxt}" (consonant sound).',
                        suggestion="a",
                        severity="HIGH",
                        confidence=0.90,
                        context=sent,
                    ))
        return errors

    def _check_pronouns(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]

        for i, t in enumerate(lower_tokens):
            if t == "i" and tokens[i] != "I":
                pos = self._word_position(sent, i, sent_start)
                errors.append(make_error(
                    text=tokens[i], start=pos, end=pos + len(tokens[i]),
                    category="CAPITALIZATION",
                    message='The pronoun "I" should always be capitalized.',
                    suggestion="I",
                    severity="HIGH",
                    confidence=0.99,
                    context=sent,
                ))

        if lower_tokens:
            first = lower_tokens[0]
            obj_to_subj = {
                "me": "I", "him": "he", "her": "she", "us": "we",
                "them": "they", "myself": "I", "himself": "he",
                "herself": "she", "itself": "it", "ourselves": "we",
                "themselves": "they",
            }
            if first in obj_to_subj:
                word = tokens[0]
                pos = self._word_position(sent, 0, sent_start)
                suggestion = obj_to_subj[first]
                rest = sent[pos - sent_start + len(word):]
                errors.append(make_error(
                    text=word, start=pos, end=pos + len(word),
                    category="GRAMMAR",
                    message=f'Use the subject form "{suggestion}" at the start of a sentence, not "{word}".',
                    suggestion=suggestion,
                    severity="HIGH",
                    confidence=0.90,
                    context=sent,
                    explanation=f'"{word}" is an object pronoun. Use the subject form "{suggestion}" as the sentence subject.',
                ))

        for i in range(len(lower_tokens) - 1):
            if lower_tokens[i] in ("me", "him", "her", "us", "them"):
                if lower_tokens[i] == "me" and i == 0:
                    if i + 2 < len(lower_tokens) and lower_tokens[i + 1] == "and":
                        word = tokens[0]
                        pos = self._word_position(sent, 0, sent_start)
                        errors.append(make_error(
                            text=word, start=pos, end=pos + len(word),
                            category="GRAMMAR",
                            message='Use "I" (not "me") as a compound subject. Say "I and ...".',
                            suggestion="I",
                            severity="HIGH",
                            confidence=0.88,
                            context=sent,
                        ))
        return errors

    def _check_double_negatives(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]
        strict_neg = {"no", "not", "nothing", "nobody", "no one", "none",
                      "never", "neither", "nor", "nowhere"}
        contraction_neg = {"n't", "dont", "doesn't", "didn't", "won't",
                          "wouldn't", "couldn't", "shouldn't", "can't",
                          "isn't", "aren't", "wasn't", "weren't",
                          "hasn't", "haven't", "hadn't"}

        neg_positions = []
        for i, t in enumerate(lower_tokens):
            if t in strict_neg or t in contraction_neg:
                neg_positions.append(i)

        if len(neg_positions) >= 2:
            for k in range(len(neg_positions) - 1):
                i1 = neg_positions[k]
                i2 = neg_positions[k + 1]
                if i2 - i1 <= 5:
                    t1 = tokens[i1]
                    t2 = tokens[i2]
                    pos = self._word_position(sent, i2, sent_start)
                    errors.append(make_error(
                        text=t2, start=pos, end=pos + len(t2),
                        category="GRAMMAR",
                        message=f'Double negative: "{t1}" and "{t2}". Use only one negative.',
                        suggestion="",
                        severity="HIGH",
                        confidence=0.90,
                        context=sent,
                        explanation='Two negatives make a positive. Use only one negative word.',
                    ))
        return errors

    def _check_confusing_words(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]
        words = sent.split()

        for i, t in enumerate(lower_tokens):
            if t in self.CONFUSING_WORDS:
                replacement, desc, explanation = self.CONFUSING_WORDS[t]
                context_ok = self._confusing_word_context(
                    t, lower_tokens, i
                )
                if context_ok is not None:
                    correct = context_ok
                    if correct != t:
                        word = tokens[i]
                        pos = self._word_position(sent, i, sent_start)
                        conf = 0.78
                        if t in ("then", "than") and correct == "than":
                            prev = lower_tokens[i - 1] if i > 0 else ""
                            nxt = lower_tokens[i + 1] if i + 1 < len(lower_tokens) else ""
                            comp = {"more", "less", "better", "worse", "rather", "other", "bigger", "smaller", "taller", "faster", "slower", "higher", "lower", "greater", "wider", "older", "younger", "easier", "harder", "longer", "shorter", "stronger", "weaker", "nicer", "richer", "poorer", "smarter"}
                            if prev in comp or nxt in ("his", "hers", "its", "ours", "yours", "theirs", "my", "your", "her", "our", "their", "a", "an", "the"):
                                conf = 0.92
                        if t == "its" and correct == "it's":
                            conf = 0.90
                        errors.append(make_error(
                            text=word, start=pos, end=pos + len(word),
                            category="WORD_USAGE",
                            message=f'Possible confusion: did you mean "{correct}"? ({desc})',
                            suggestion=correct,
                            severity="MEDIUM",
                            confidence=conf,
                            context=sent,
                            explanation=explanation,
                        ))
        return errors

    def _confusing_word_context(self, word: str, tokens: List[str], idx: int) -> Optional[str]:
        if word in ("your", "you're"):
            next_words = {t.lower() for t in tokens[idx + 1: idx + 3]}
            verb_tags = {"going", "coming", "doing", "being", "getting",
                        "welcome", "right", "wrong", "correct", "smart",
                        "kind", "beautiful", "ready", "sure", "fine",
                        "okay", "able", "willing", "trying", "looking",
                        "sitting", "standing", "making", "taking"}
            if next_words & verb_tags:
                return "you're"
            noun_tags = {"car", "house", "dog", "cat", "book", "room",
                        "name", "phone", "family", "friend", "bag",
                        "clothes", "mother", "father", "brother", "sister",
                        "homework", "school", "work"}
            if next_words & noun_tags:
                return "your"
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1].lower()
                if nxt in ("is", "are", "was", "were", "has", "have", "had",
                           "will", "would", "can", "could", "should", "may",
                           "might", "must"):
                    return "you're"
                if nxt[0:1] in "bcdfghjklmnpqrstvwxyz" and nxt not in verb_tags:
                    return "your"
            return None

        if word in ("their", "they're", "there"):
            next_words = {t.lower() for t in tokens[idx + 1: idx + 3]}
            if word == "their":
                noun_tags = {"car", "house", "dog", "cat", "book", "room",
                            "children", "parents", "friends", "phone",
                            "house", "bag", "clothes", "school", "work"}
                if next_words & noun_tags:
                    return "their"
            if word == "there":
                exist_tags = {"is", "are", "was", "were", "will", "can",
                             "not", "here", "over"}
                if next_words & exist_tags:
                    return "there"
                if next_words & {"is", "are", "was", "were"}:
                    return "there"
            if word == "they're":
                verb_tags = {"going", "coming", "doing", "playing",
                            "happy", "ready", "right", "wrong"}
                if next_words & verb_tags:
                    return "they're"
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1].lower()
                if nxt in ("is", "are", "was", "were"):
                    return "there"
                if nxt in ("going", "coming", "doing", "being", "getting",
                           "happy", "ready", "right", "wrong"):
                    return "they're"
                adj_tags = {"beautiful", "smart", "tall", "short", "big",
                           "small", "old", "young", "good", "bad"}
                if nxt in adj_tags:
                    return "they're"
            return None

        if word in ("its", "it's"):
            next_words = {t.lower() for t in tokens[idx + 1: idx + 3]}
            if word == "its":
                noun_tags = {"car", "house", "dog", "cat", "book", "room",
                            "color", "shape", "tail", "purpose", "name"}
                if next_words & noun_tags:
                    return "its"
            if word == "it's":
                if next_words & {"a", "an", "raining", "sunny", "cold",
                                "hot", "been", "not", "time", "important"}:
                    return "it's"
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1].lower()
                if nxt in ("a", "an", "not", "been", "raining", "sunny"):
                    return "it's"
                verb_tags = {"is", "are", "was", "were", "has", "have", "had",
                            "will", "would", "can", "could", "should"}
                if nxt in verb_tags:
                    return "it's"
                adj_words = {"very", "really", "quite", "so", "too", "also",
                            "always", "never", "just", "already", "still",
                            "good", "bad", "great", "nice", "fine", "wrong",
                            "right", "ready", "busy", "free", "safe", "sure",
                            "true", "important", "beautiful", "smart"}
                if nxt in adj_words:
                    return "it's"
                if nxt not in ("is", "was") and nxt[0:1] in "bcdfghjklmnpqrstvwxyz":
                    return "its"
            return None

        if word in ("then", "than"):
            comp_words = {"more", "less", "better", "worse", "rather",
                         "other", "bigger", "smaller", "taller",
                         "faster", "slower", "higher", "lower",
                         "greater", "wider", "older", "younger",
                         "easier", "harder", "longer", "shorter",
                         "stronger", "weaker", "cleaner", "dirtier",
                         "nicer", "ruder", "safer", "richer", "poorer",
                         "smarter", "braver", "calmer", "sadder"}
            if word == "than":
                prev_words = {t.lower() for t in tokens[max(0, idx - 2): idx]}
                if prev_words & comp_words:
                    return "than"
            if word == "then":
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1].lower()
                    if nxt in comp_words:
                        return "than"
                if idx > 0:
                    prev = tokens[idx - 1].lower()
                    if prev in comp_words:
                        return "than"
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1].lower()
                    if nxt in ("his", "hers", "its", "ours", "yours", "theirs",
                               "my", "your", "her", "our", "their"):
                        return "than"
            return None

        if word in ("affect", "effect"):
            if word == "affect":
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1].lower()
                    if nxt in ("the", "my", "our", "his", "her", "this", "that"):
                        return "affect"
                    if nxt in ("has", "had", "have"):
                        return "effect"
            if word == "effect":
                if idx > 0:
                    prev = tokens[idx - 1].lower()
                    if prev in ("has", "had", "have", "the", "an"):
                        return "effect"
                    if prev in ("will", "can", "does", "do", "to"):
                        return "affect"
            return None

        if word in ("accept", "except"):
            if word == "accept":
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1].lower()
                    if nxt in ("for", "all", "every", "one", "but"):
                        return "except"
            if word == "except":
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1].lower()
                    if nxt in ("will", "can", "to", "the", "this", "it"):
                        return "accept"
            return None

        if word in ("loose", "lose"):
            if word == "loose":
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1].lower()
                    if nxt in ("weight", "game", "match", "chance", "hope"):
                        return "lose"
            if word == "lose":
                if idx + 1 < len(tokens):
                    nxt = tokens[idx + 1].lower()
                    if nxt in ("clothes", "fit", "tight", "shoe"):
                        return "loose"
            return None

        if word in ("advice", "advise"):
            if word == "advice":
                if idx > 0:
                    prev = tokens[idx - 1].lower()
                    if prev in ("to", "you", "him", "her", "them", "will", "can"):
                        return "advise"
            if word == "advise":
                if idx > 0:
                    prev = tokens[idx - 1].lower()
                    if prev in ("give", "some", "good", "his", "her", "the", "need"):
                        return "advice"
            return None

        return None

    def _check_missing_verb(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]
        if len(lower_tokens) < 3:
            return errors

        skip_words = {"not", "n't", "never", "really", "very", "also",
                      "always", "often", "sometimes", "usually", "just",
                      "only", "even", "still", "already", "yet", "too",
                      "quite", "rather", "almost", "nearly", "enough",
                      "perhaps", "maybe", "actually", "basically",
                      "honestly", "literally", "fortunately",
                      "unfortunately", "obviously", "clearly",
                      "well", "oh", "so", "then", "now", "here",
                      "there", "where", "when", "how", "why"}

        has_verb = False
        for t in lower_tokens:
            if t in skip_words:
                continue
            if (self._looks_like_verb(t) or
                t in ("is", "am", "are", "was", "were",
                      "has", "have", "had", "do", "does", "did")):
                has_verb = True
                break

        has_subject = False
        subject_idx = -1
        for i, t in enumerate(lower_tokens):
            if t in skip_words:
                continue
            if t in self.SINGULAR_PRP or t in self.PLURAL_PRP or t in ("i",):
                has_subject = True
                subject_idx = i
                break
            if len(t) > 2 and t[0].isupper():
                has_subject = True
                subject_idx = i
                break
            if len(t) > 3 and t.endswith(("er", "or", "ist", "ant", "ent")):
                has_subject = True
                subject_idx = i
                break
            break

        if has_subject and not has_verb and len(lower_tokens) >= 3:
            first = tokens[0]
            pos = self._word_position(sent, 0, sent_start)
            if lower_tokens[0] not in self.MODALS:
                errors.append(make_error(
                    text=first, start=pos, end=pos + len(first),
                    category="SENTENCE_STRUCTURE",
                    message=f'This sentence may be missing a verb after "{first}".',
                    suggestion="",
                    severity="MEDIUM",
                    confidence=0.60,
                    context=sent,
                    explanation='A complete sentence needs a subject and a verb.',
                ))
        return errors

    def _check_extra_words(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]

        patterns = [
            (["did", "not"], ["didn't"], "Use the contraction"),
            (["do", "not"], ["don't"], "Use the contraction"),
            (["does", "not"], ["doesn't"], "Use the contraction"),
            (["did", "not"], ["didn't"], "Use the contraction"),
        ]

        for i in range(len(lower_tokens) - 1):
            if lower_tokens[i] == "did" and i + 2 < len(lower_tokens):
                if lower_tokens[i + 1] == "not" or lower_tokens[i + 1] == "n't":
                    nxt = lower_tokens[i + 2] if i + 2 < len(lower_tokens) else ""
                    if nxt.endswith("ed") and len(nxt) > 3:
                        word = tokens[i + 2]
                        pos = self._word_position(sent, i + 2, sent_start)
                        base = nxt[:-2] if nxt.endswith("ed") else nxt[:-1]
                        errors.append(make_error(
                            text=word, start=pos, end=pos + len(word),
                            category="GRAMMAR",
                            message=f'After "didn\'t", use the base form "{base}", not "{word}".',
                            suggestion=base,
                            severity="HIGH",
                            confidence=0.92,
                            context=sent,
                        ))

        for i in range(len(lower_tokens) - 1):
            if lower_tokens[i] in ("is", "am", "are", "was", "were"):
                if i + 1 < len(lower_tokens) and lower_tokens[i + 1] == "being":
                    if i + 2 < len(lower_tokens) and lower_tokens[i + 2].endswith("ed"):
                        pass
                    if i + 1 < len(lower_tokens):
                        nxt = lower_tokens[i + 1] if i + 1 < len(lower_tokens) else ""
                        if nxt == "being" and i + 2 < len(lower_tokens):
                            nxt2 = lower_tokens[i + 2]
                            if nxt2.endswith("ed"):
                                word = tokens[i + 1]
                                pos = self._word_position(sent, i + 1, sent_start)
                                errors.append(make_error(
                                    text=word, start=pos, end=pos + len(word),
                                    category="GRAMMAR",
                                    message=f'"is being" with past participle may be incorrect. Use "is" or "was" + past participle.',
                                    suggestion="",
                                    severity="LOW",
                                    confidence=0.60,
                                    context=sent,
                                ))
        return errors

    def _check_redundancy(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]

        for i in range(1, len(lower_tokens)):
            if lower_tokens[i] == lower_tokens[i - 1]:
                skip_words = {"that", "this", "the", "a", "an", "is", "was"}
                if lower_tokens[i] not in skip_words:
                    word = tokens[i]
                    pos = self._word_position(sent, i, sent_start)
                    errors.append(make_error(
                        text=word, start=pos, end=pos + len(word),
                        category="REDUNDANCY",
                        message=f'Repeated word: "{word}". Consider removing one instance.',
                        suggestion="",
                        severity="MEDIUM",
                        confidence=0.90,
                        context=sent,
                        explanation='The same word appears twice in a row.',
                    ))
        return errors

    def _check_plural_singular(self, sent: str, sent_start: int) -> List[Dict]:
        errors = []
        tokens = self._tokenize(sent)
        lower_tokens = [t.lower() for t in tokens]

        for i, t in enumerate(lower_tokens):
            if t in self.IRREGULAR_PLURAL:
                for j in range(max(0, i - 4), i):
                    if lower_tokens[j] in ("two", "three", "four", "five", "six",
                                            "seven", "eight", "nine", "ten",
                                            "many", "several", "few", "these",
                                            "those"):
                        break
                    if lower_tokens[j] == "one" or lower_tokens[j] in self.SINGULAR_PRP:
                        word = tokens[i]
                        pos = self._word_position(sent, i, sent_start)
                        singular = self.IRREGULAR_SINGULAR.get(t, t)
                        errors.append(make_error(
                            text=word, start=pos, end=pos + len(word),
                            category="GRAMMAR",
                            message=f'The singular form is "{singular}" (not "{word}") when used with "{lower_tokens[j]}".',
                            suggestion=singular,
                            severity="MEDIUM",
                            confidence=0.70,
                            context=sent,
                        ))
                        break
        return errors

    def _word_position(self, sent: str, token_idx: int, sent_start: int) -> int:
        tokens_in_sent = re.findall(r"[a-zA-Z']+", sent)
        if token_idx >= len(tokens_in_sent):
            return sent_start + len(sent)
        target = tokens_in_sent[token_idx]
        pos = 0
        for i, m in enumerate(re.finditer(r"[a-zA-Z']+", sent)):
            if i == token_idx:
                return sent_start + m.start()
            pos = m.end()
        return sent_start


# ---------------------------------------------------------------------------
# Punctuation engine
# ---------------------------------------------------------------------------
class PunctuationEngine:
    def check(self, text: str) -> List[Dict]:
        errors = []
        if not text or not text.strip():
            return errors

        errors.extend(self._check_missing_end_punctuation(text))
        errors.extend(self._check_spacing_around_punctuation(text))
        errors.extend(self._check_multiple_punctuation(text))
        errors.extend(self._check_comma_splices(text))
        errors.extend(self._check_apostrophes(text))
        errors.extend(self._check_missing_commas(text))
        return errors

    def _check_missing_end_punctuation(self, text: str) -> List[Dict]:
        errors = []
        stripped = text.rstrip()
        if stripped and stripped[-1] not in ".!?":
            errors.append(make_error(
                text=stripped[-1], start=len(text) - 1, end=len(text),
                category="PUNCTUATION",
                message="Sentence is missing end punctuation (period, question mark, or exclamation mark).",
                suggestion=".",
                severity="MEDIUM",
                confidence=0.85,
                explanation="Every sentence should end with appropriate punctuation.",
            ))
        return errors

    def _check_spacing_around_punctuation(self, text: str) -> List[Dict]:
        errors = []
        for m in re.finditer(r"\s[,.:;!?]", text):
            errors.append(make_error(
                text=m.group(), start=m.start(), end=m.end(),
                category="PUNCTUATION",
                message=f'Extra space before "{m.group().strip()}".',
                suggestion=m.group().strip(),
                severity="LOW",
                confidence=0.85,
                explanation="Remove the space before punctuation.",
            ))
        for m in re.finditer(r"[,.:;](?=[a-zA-Z])", text):
            errors.append(make_error(
                text=m.group(), start=m.start(), end=m.end() + 1,
                category="PUNCTUATION",
                message=f'Missing space after "{m.group()}".',
                suggestion=m.group() + " ",
                severity="MEDIUM",
                confidence=0.88,
                explanation="Add a space after the comma/period.",
            ))
        return errors

    def _check_multiple_punctuation(self, text: str) -> List[Dict]:
        errors = []
        for m in re.finditer(r"[.!?]{2,}", text):
            if m.group() not in ("...",):
                errors.append(make_error(
                    text=m.group(), start=m.start(), end=m.end(),
                    category="PUNCTUATION",
                    message=f'Multiple punctuation marks: "{m.group()}". Use a single mark.',
                    suggestion=m.group()[0],
                    severity="LOW",
                    confidence=0.75,
                    explanation="Use a single punctuation mark.",
                ))
        return errors

    def _check_comma_splices(self, text: str) -> List[Dict]:
        errors = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            parts = re.split(r',\s*', sent)
            if len(parts) >= 3:
                for i in range(len(parts) - 1):
                    p1 = parts[i].strip()
                    p2 = parts[i + 1].strip()
                    if (p1 and p2 and
                        len(p1.split()) >= 2 and len(p2.split()) >= 2 and
                        p1[0].isupper() and p2[0].isupper()):
                        comma_pos = text.find(p1[-1] + "," + p2[0])
                        if comma_pos >= 0:
                            errors.append(make_error(
                                text=",", start=comma_pos, end=comma_pos + 1,
                                category="PUNCTUATION",
                                message='Possible comma splice. Consider using a period or semicolon.',
                                suggestion=".",
                                severity="MEDIUM",
                                confidence=0.65,
                                context=sent,
                                explanation="Two independent clauses are joined only by a comma.",
                            ))
                            break
        return errors

    def _check_apostrophes(self, text: str) -> List[Dict]:
        errors = []
        contractions = {
            "dont": "don't", "doesnt": "doesn't", "didnt": "didn't",
            "isnt": "isn't", "wasnt": "wasn't", "werent": "weren't",
            "hasnt": "hasn't", "havent": "haven't", "hadnt": "hadn't",
            "wouldnt": "wouldn't", "couldnt": "couldn't",
            "shouldnt": "shouldn't", "cant": "can't", "wont": "won't",
            "arent": "aren't", "aint": "isn't",
            "its": None, "your": None, "their": None,
            "theyre": "they're", "youre": "you're", "were": None,
            "wont": "won't",
        }
        for wrong, right in contractions.items():
            if right is None:
                continue
            for m in re.finditer(r'\b' + wrong + r'\b', text, re.IGNORECASE):
                errors.append(make_error(
                    text=m.group(), start=m.start(), end=m.end(),
                    category="PUNCTUATION",
                    message=f'Missing apostrophe: "{m.group()}" should be "{right}".',
                    suggestion=right,
                    severity="MEDIUM",
                    confidence=0.80,
                    explanation=f'The contraction "{right}" requires an apostrophe.',
                ))
        return errors

    def _check_missing_commas(self, text: str) -> List[Dict]:
        errors = []
        intro_words = {"however", "therefore", "moreover", "furthermore",
                       "nevertheless", "consequently", "meanwhile",
                       "fortunately", "unfortunately", "basically",
                       "obviously", "clearly", "finally", "first",
                       "second", "third", "also", "then", "yes", "no",
                       "well", "oh", "although", "though", "while",
                       "if", "when", "after", "before", "since",
                       "unless", "until", "as soon as"}
        tokens = re.findall(r"[a-zA-Z']+", text)
        lower_tokens = [t.lower() for t in tokens]

        for i, t in enumerate(lower_tokens):
            if t in intro_words:
                word_pos = text.find(tokens[i])
                next_comma = text.find(",", word_pos + len(tokens[i]))
                next_word_end = text.find(" ", word_pos + len(tokens[i]))
                if next_comma < 0 or (next_word_end >= 0 and next_comma > next_word_end + 20):
                    if i + 1 < len(lower_tokens):
                        errors.append(make_error(
                            text=tokens[i], start=word_pos,
                            end=word_pos + len(tokens[i]),
                            category="PUNCTUATION",
                            message=f'Consider adding a comma after introductory word "{tokens[i]}".',
                            suggestion=tokens[i] + ",",
                            severity="LOW",
                            confidence=0.65,
                            context=text,
                            explanation="Introductory words are typically followed by a comma.",
                        ))
        return errors


# ---------------------------------------------------------------------------
# Capitalization engine
# ---------------------------------------------------------------------------
class CapitalizationEngine:
    PROPER_NOUNS = {
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "january", "february", "march",
        "april", "may", "june", "july", "august", "september",
        "october", "november", "december", "english", "french",
        "spanish", "german", "chinese", "japanese", "arabic",
        "hindu", "hindi", "latin", "mathematics", "physics",
        "chemistry", "biology", "history", "geography",
    }

    def check(self, text: str) -> List[Dict]:
        errors = []
        if not text or not text.strip():
            return errors
        errors.extend(self._check_sentence_start(text))
        errors.extend(self._check_i_capitalization(text))
        errors.extend(self._check_proper_nouns(text))
        errors.extend(self._check_all_caps(text))
        return errors

    def _check_sentence_start(self, text: str) -> List[Dict]:
        errors = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        pos = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                pos += 1
                continue
            first_char = sent[0]
            if first_char.isalpha() and first_char.islower():
                errors.append(make_error(
                    text=first_char, start=pos, end=pos + 1,
                    category="CAPITALIZATION",
                    message=f'Sentences should begin with a capital letter. "{first_char}" should be "{first_char.upper()}".',
                    suggestion=first_char.upper(),
                    severity="HIGH",
                    confidence=0.95,
                    explanation="The first word of a sentence must be capitalized.",
                ))
            pos += len(sent) + 1
        return errors

    def _check_i_capitalization(self, text: str) -> List[Dict]:
        errors = []
        for m in re.finditer(r'\bi\b', text):
            if m.start() == 0 or text[m.start() - 1] in " \n\t":
                errors.append(make_error(
                    text="i", start=m.start(), end=m.end(),
                    category="CAPITALIZATION",
                    message='The pronoun "I" should always be capitalized.',
                    suggestion="I",
                    severity="HIGH",
                    confidence=0.99,
                    explanation='The pronoun "I" is always capitalized in English.',
                ))
        return errors

    def _check_proper_nouns(self, text: str) -> List[Dict]:
        errors = []
        words = re.findall(r"[a-zA-Z']+", text)
        lower_words = [w.lower() for w in words]

        for i, w in enumerate(words):
            if w[0].isupper() and i > 0:
                prev_word = words[i - 1] if i > 0 else ""
                if prev_word in (".", "!", "?", '"', "("):
                    continue
                if w.lower() in self.PROPER_NOUNS:
                    continue
                if len(w) <= 2:
                    continue

        return errors

    def _check_all_caps(self, text: str) -> List[Dict]:
        errors = []
        for m in re.finditer(r'\b[A-Z]{3,}\b', text):
            word = m.group()
            if word in ("I",):
                continue
            if word.isupper() and any(c.islower() for c in text[max(0, m.start() - 5):m.start()]):
                errors.append(make_error(
                    text=word, start=m.start(), end=m.end(),
                    category="CAPITALIZATION",
                    message=f'Unnecessary ALL CAPS: "{word}".',
                    suggestion=word.capitalize(),
                    severity="LOW",
                    confidence=0.70,
                    explanation="Avoid using ALL CAPS for emphasis.",
                ))
        return errors


# ---------------------------------------------------------------------------
# Style engine
# ---------------------------------------------------------------------------
class StyleEngine:
    FILLER_WORDS = {
        "basically", "actually", "literally", "honestly", "just",
        "really", "quite", "very", "extremely", "absolutely",
        "totally", "completely", "definitely", "certainly",
        "obviously", "clearly", "simply", "merely",
    }
    WEAK_WORDS = {
        "very": ["extremely", "remarkably", "exceptionally"],
        "really": ["truly", "genuinely", "particularly"],
        "good": ["excellent", "great", "superb", "outstanding"],
        "bad": ["terrible", "awful", "horrible", "dreadful"],
        "big": ["large", "enormous", "vast", "massive"],
        "small": ["tiny", "little", "compact", "minute"],
        "nice": ["pleasant", "delightful", "charming", "lovely"],
        "said": ["stated", "remarked", "noted", "explained"],
        "thing": ["matter", "aspect", "element", "factor"],
        "a lot": ["many", "much", "numerous", "frequent"],
        "important": ["crucial", "vital", "essential", "significant"],
        "interesting": ["fascinating", "intriguing", "compelling", "engaging"],
        "beautiful": ["gorgeous", "stunning", "exquisite", "elegant"],
        "ugly": ["hideous", "unsightly", "grotesque", "repulsive"],
        "happy": ["joyful", "elated", "thrilled", "delighted"],
        "sad": ["sorrowful", "mournful", "melancholy", "heartbroken"],
        "angry": ["furious", "enraged", "livid", "irate"],
        "fast": ["swift", "rapid", "quick", "speedy"],
        "slow": ["sluggish", "leisurely", "unhurried", "plodding"],
        "smart": ["intelligent", "brilliant", "sharp", "astute"],
        "dumb": ["foolish", "senseless", "inane", "mindless"],
        "strong": ["powerful", "mighty", "robust", "sturdy"],
        "weak": ["feeble", "frail", "fragile", "delicate"],
        "hot": ["scorching", "blistering", "sweltering", "blazing"],
        "cold": ["freezing", "frigid", "icy", "bitter"],
        "old": ["ancient", "antique", "aged", "venerable"],
        "new": ["modern", "novel", "innovative", "fresh"],
        "long": ["extensive", "lengthy", "prolonged", "extended"],
        "short": ["brief", "concise", "abbreviated", "curtailed"],
        "hard": ["difficult", "challenging", "demanding", "rigorous"],
        "easy": ["simple", "straightforward", "effortless", "uncomplicated"],
        "rich": ["wealthy", "affluent", "prosperous", "opulent"],
        "poor": ["impoverished", "destitute", "indigent", "needy"],
        "funny": ["hilarious", "amusing", "comical", "entertaining"],
        "sad": ["depressing", "somber", "bleak", "grim"],
        "scary": ["terrifying", "frightening", "alarming", "horrifying"],
        "weird": ["bizarre", "peculiar", "strange", "unusual"],
        "crazy": ["lunatic", "frenzied", "wild", "absurd"],
        "stupid": ["idiotic", "foolish", "asinine", "moronic"],
        "boring": ["dull", "tedious", "monotonous", "tiresome"],
        "cool": ["impressive", "remarkable", "extraordinary", "noteworthy"],
        "amazing": ["astonishing", "incredible", "remarkable", "stunning"],
        "terrible": ["dreadful", "atrocious", "abominable", "appalling"],
        "wonderful": ["magnificent", "splendid", "marvelous", "superb"],
        "awful": ["horrendous", "abysmal", "dreadful", "lousy"],
        "fine": ["acceptable", "adequate", "satisfactory", "passable"],
        "pretty": ["attractive", "appealing", "charming", "lovely"],
        "clean": ["spotless", "immaculate", "pristine", "scrupulous"],
        "dirty": ["filthy", "grimy", "squalid", "sordid"],
        "quiet": ["silent", "hushed", "mute", "noiseless"],
        "loud": ["deafening", "booming", "thunderous", "earsplitting"],
        "safe": ["secure", "protected", "shielded", "guarded"],
        "dangerous": ["hazardous", "perilous", "risky", "treacherous"],
        "correct": ["accurate", "precise", "exact", "right"],
        "wrong": ["incorrect", "inaccurate", "erroneous", "fallacious"],
        "enough": ["sufficient", "adequate", "ample", "sufficient"],
        "help": ["assist", "support", "aid", "facilitate"],
        "show": ["demonstrate", "display", "exhibit", "illustrate"],
        "get": ["obtain", "acquire", "procure", "secure"],
        "make": ["create", "produce", "construct", "fabricate"],
        "give": ["provide", "supply", "furnish", "deliver"],
        "take": ["seize", "grab", "acquire", "capture"],
        "use": ["utilize", "employ", "leverage", "apply"],
        "try": ["attempt", "endeavor", "strive", "endeavour"],
        "start": ["begin", "commence", "initiate", "launch"],
        "end": ["conclude", "terminate", "finish", "culminate"],
        "need": ["require", "demand", "necessitate", "call for"],
        "want": ["desire", "crave", "long for", "yearn for"],
        "think": ["consider", "contemplate", "ponder", "reflect"],
        "feel": ["sense", "perceive", "experience", "detect"],
        "look": ["appear", "seem", "resemble", "look like"],
        "come": ["arrive", "approach", "appear", "emerge"],
        "go": ["proceed", "advance", "depart", "travel"],
        "put": ["place", "position", "situate", "deposit"],
        "run": ["sprint", "dash", "race", "hasten"],
        "move": ["relocate", "transfer", "shift", "reposition"],
        "live": ["reside", "dwell", "inhabit", "occupy"],
        "belong": ["pertain", "relate", "apply", "correspond"],
    }
    CLICHES = {
        "at the end of the day", "think outside the box",
        "low hanging fruit", "move the needle",
        "circle back", "touch base", "boil the ocean",
        "paradigm shift", "synergy", "leverage",
        "deep dive", "game changer", "on the same page",
        "bottom line", "win-win", "back to basics",
        "only time will tell", "the fact of the matter",
        "it goes without saying", "needless to say",
        "for all intents and purposes", "each and every",
        "last but not least", "easier said than done",
        "better late than never", "actions speak louder than words",
        "every cloud has a silver lining", "blessing in disguise",
        "time heals all wounds", "when life gives you lemons",
        "the ball is in your court", "break a leg",
    }

    def check(self, text: str) -> List[Dict]:
        errors = []
        if not text or not text.strip():
            return errors
        errors.extend(self._check_filler_words(text))
        errors.extend(self._check_weak_words(text))
        errors.extend(self._check_cliches(text))
        errors.extend(self._check_passive_voice(text))
        errors.extend(self._check_wordiness(text))
        errors.extend(self._check_long_sentences(text))
        return errors

    def _check_filler_words(self, text: str) -> List[Dict]:
        errors = []
        for fw in self.FILLER_WORDS:
            for m in re.finditer(r'\b' + fw + r'\b', text, re.IGNORECASE):
                errors.append(make_error(
                    text=m.group(), start=m.start(), end=m.end(),
                    category="STYLE",
                    message=f'Filler word "{m.group()}". Consider removing it for stronger writing.',
                    suggestion="",
                    severity="LOW",
                    confidence=0.70,
                    explanation=f'"{m.group()}" adds little meaning and weakens the sentence.',
                ))
        return errors

    def _check_weak_words(self, text: str) -> List[Dict]:
        errors = []
        for word, alternatives in self.WEAK_WORDS.items():
            for m in re.finditer(r'\b' + word + r'\b', text, re.IGNORECASE):
                errors.append(make_error(
                    text=m.group(), start=m.start(), end=m.end(),
                    category="STYLE",
                    message=f'Consider replacing "{m.group()}" with a more precise word like "{alternatives[0]}".',
                    suggestion=alternatives[0],
                    severity="LOW",
                    confidence=0.65,
                    explanation=f'"{m.group()}" is vague. Stronger alternatives exist.',
                ))
        return errors

    def _check_cliches(self, text: str) -> List[Dict]:
        errors = []
        text_lower = text.lower()
        for cliche in self.CLICHES:
            idx = text_lower.find(cliche)
            if idx >= 0:
                errors.append(make_error(
                    text=text[idx: idx + len(cliche)], start=idx,
                    end=idx + len(cliche),
                    category="STYLE",
                    message=f'Cliche detected: "{cliche}". Consider rephrasing.',
                    suggestion="",
                    severity="LOW",
                    confidence=0.75,
                    explanation="Cliches make writing feel generic.",
                ))
        return errors

    def _check_passive_voice(self, text: str) -> List[Dict]:
        errors = []
        pattern = r'\b(is|are|was|were|be|been|being)\s+(being\s+)?(\w+ed)\b'
        for m in re.finditer(pattern, text, re.IGNORECASE):
            errors.append(make_error(
                text=m.group(), start=m.start(), end=m.end(),
                category="STYLE",
                message=f'Passive voice detected: "{m.group()}". Consider active voice.',
                suggestion="",
                severity="LOW",
                confidence=0.65,
                explanation="Active voice is generally clearer and more direct.",
            ))
        return errors

    def _check_wordiness(self, text: str) -> List[Dict]:
        errors = []
        wordy = {
            "in order to": "to",
            "due to the fact that": "because",
            "at this point in time": "now",
            "in the event that": "if",
            "for the purpose of": "to",
            "in the process of": "while",
            "on a daily basis": "daily",
            "in a timely manner": "promptly",
            "at the present time": "currently",
            "in the near future": "soon",
            "has the ability to": "can",
            "is able to": "can",
            "make a decision": "decide",
            "give consideration to": "consider",
            "arrive at a conclusion": "conclude",
            "perform an analysis of": "analyze",
            "conduct an investigation of": "investigate",
        }
        text_lower = text.lower()
        for phrase, replacement in wordy.items():
            idx = text_lower.find(phrase)
            if idx >= 0:
                errors.append(make_error(
                    text=text[idx: idx + len(phrase)], start=idx,
                    end=idx + len(phrase),
                    category="STYLE",
                    message=f'Wordy phrase: "{phrase}" can be simplified to "{replacement}".',
                    suggestion=replacement,
                    severity="LOW",
                    confidence=0.80,
                    explanation=f'"{replacement}" is more concise than "{phrase}".',
                ))
        return errors

    def _check_long_sentences(self, text: str) -> List[Dict]:
        errors = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            words = sent.split()
            if len(words) > 40:
                sent_start = text.find(sent)
                errors.append(make_error(
                    text=sent[:50] + "...", start=sent_start,
                    end=sent_start + len(sent),
                    category="CLARITY",
                    message=f'Long sentence ({len(words)} words). Consider breaking it into shorter sentences.',
                    suggestion="",
                    severity="LOW",
                    confidence=0.70,
                    context=sent,
                    explanation="Very long sentences are harder to read and understand.",
                ))
        return errors


# ---------------------------------------------------------------------------
# Readability calculator
# ---------------------------------------------------------------------------
class ReadabilityEngine:
    def calculate(self, text: str) -> Dict:
        if not text or not text.strip():
            return {
                "score": 0, "level": "N/A", "grade": 0,
                "sentences": 0, "words": 0, "syllables": 0,
                "avg_words_sentence": 0, "reading_time": 0,
                "speaking_time": 0, "complex_word_pct": 0,
                "passive_voice_pct": 0,
            }

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r"[a-zA-Z']+", text)
        total_syllables = sum(self._count_syllables(w) for w in words)
        avg_sent_len = len(words) / max(len(sentences), 1)
        avg_syllables = total_syllables / max(len(words), 1)

        flesch = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syllables)
        flesch = max(0, min(100, flesch))
        grade = 0.39 * avg_sent_len + 11.8 * avg_syllables - 15.59
        grade = max(0, grade)

        if flesch >= 80:
            level = "Very Easy"
        elif flesch >= 60:
            level = "Easy"
        elif flesch >= 50:
            level = "Standard"
        elif flesch >= 30:
            level = "Difficult"
        else:
            level = "Very Difficult"

        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        complex_pct = (complex_words / max(len(words), 1)) * 100

        passive_pattern = r'\b(is|are|was|were|be|been|being)\s+\w+ed\b'
        passive_count = len(re.findall(passive_pattern, text, re.IGNORECASE))
        passive_pct = (passive_count / max(len(sentences), 1)) * 100

        return {
            "score": round(flesch, 1),
            "level": level,
            "grade": round(grade, 1),
            "sentences": len(sentences),
            "words": len(words),
            "syllables": total_syllables,
            "avg_words_sentence": round(avg_sent_len, 1),
            "reading_time": round(len(words) / 200, 1),
            "speaking_time": round(len(words) / 130, 1),
            "complex_word_pct": round(complex_pct, 1),
            "passive_voice_pct": round(passive_pct, 1),
        }

    def _count_syllables(self, word: str) -> int:
        word = word.lower().strip()
        if len(word) <= 2:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for c in word:
            is_vowel = c in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and not word.endswith("le"):
            count -= 1
        if word.endswith("ed") and len(word) > 3 and not word.endswith(("ted", "ded")):
            count -= 1
        return max(1, count)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------
class WritingAnalyzer:
    def __init__(self, data_dir: str):
        self.spelling = SpellingEngine(data_dir)
        self.grammar = GrammarEngine()
        self.punctuation = PunctuationEngine()
        self.capitalization = CapitalizationEngine()
        self.style = StyleEngine()
        self.readability = ReadabilityEngine()
        self.contextual = None
        self._ext_punct = None
        self._data_dir = data_dir
        self._user_dict_path = os.path.join(data_dir, "user_dictionary.json")
        try:
            from contextual_engine import ContextualEngine
            self.contextual = ContextualEngine(data_dir)
        except Exception:
            pass
        try:
            from punctuation_checker import PunctuationChecker
            self._ext_punct = PunctuationChecker()
        except ImportError:
            pass
        try:
            from suggestion_validator import validate_candidates
            self._validate = validate_candidates
        except ImportError:
            self._validate = None

    def analyze(self, text: str, config: Optional[Dict] = None) -> Dict:
        if not text or not text.strip():
            return {
                "errors": [],
                "readability": self.readability.calculate(""),
                "stats": {"word_count": 0, "sentence_count": 0, "error_count": 0},
            }

        cfg = config or {}
        min_confidence = cfg.get("min_confidence", 0.60)
        enabled_checks = cfg.get("enabled_checks", [
            "spelling", "grammar", "punctuation", "capitalization",
            "word_usage", "style", "readability", "contextual",
        ])

        all_errors: List[Dict] = []

        if "spelling" in enabled_checks:
            all_errors.extend(self.spelling.check(text))
        if "grammar" in enabled_checks:
            self.grammar.misspelled_words = {
                e["original_text"].lower() for e in all_errors
                if e["category"] == "SPELLING"
            }
            all_errors.extend(self.grammar.check(text))
        if "contextual" in enabled_checks and self.contextual:
            all_errors.extend(self.contextual.check(text))
        if "punctuation" in enabled_checks:
            all_errors.extend(self.punctuation.check(text))
            if self._ext_punct:
                all_errors.extend(self._ext_punct.check(text))
        if "capitalization" in enabled_checks:
            all_errors.extend(self.capitalization.check(text))
        if "style" in enabled_checks:
            all_errors.extend(self.style.check(text))

        deduped = self._deduplicate(all_errors)
        filtered = [e for e in deduped if e["confidence"] >= min_confidence]

        # Run through suggestion validator for false-positive control
        if self._validate:
            all_to_validate = filtered[:]
            # Supporting signals = errors from OTHER checkers for cross-validation
            all_supporting = filtered[:]
            validated = self._validate(
                all_to_validate,
                text,
                supporting_signals=all_supporting,
                user_dict_path=self._user_dict_path,
            )
            filtered = validated

        filtered.sort(key=lambda e: (-_severity_order(e["severity"]), -e["confidence"]))

        readability = self.readability.calculate(text) if "readability" in enabled_checks else {}
        words = re.findall(r"[a-zA-Z']+", text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        return {
            "errors": filtered,
            "readability": readability,
            "stats": {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "error_count": len(filtered),
            },
        }

    def _deduplicate(self, errors: List[Dict]) -> List[Dict]:
        CATEGORY_PRIORITY = {
            "CONTEXTUAL_WORD_USAGE": 0, "GRAMMAR": 1, "WORD_USAGE": 2,
            "SPELLING": 3, "PUNCTUATION": 4, "CAPITALIZATION": 5, "STYLE": 6,
        }
        position_map = {}
        for err in errors:
            start = err.get("start_position", 0)
            end = err.get("end_position", 0)
            pos_key = (start, end, err["original_text"].lower().strip())
            cat_priority = CATEGORY_PRIORITY.get(err["category"], 10)
            if pos_key not in position_map:
                position_map[pos_key] = err
            else:
                existing = position_map[pos_key]
                existing_priority = CATEGORY_PRIORITY.get(existing["category"], 10)
                if cat_priority < existing_priority:
                    position_map[pos_key] = err
                elif cat_priority == existing_priority and err["confidence"] > existing["confidence"]:
                    position_map[pos_key] = err

        # Remove overlapping errors — keep the one with higher confidence + longer span
        # Only remove if one is FULLY contained within the other (not just overlapping)
        filtered = []
        for err in errors:
            start = err.get("start_position", 0)
            end = err.get("end_position", 0)
            pos_key = (start, end, err["original_text"].lower().strip())
            if pos_key in position_map and position_map[pos_key] is not err:
                continue
            overlaps = False
            for kept in list(filtered):
                ks = kept.get("start_position", 0)
                ke = kept.get("end_position", 0)
                # Only remove if one span fully contains the other
                if (start >= ks and end <= ke) or (ks >= start and ke <= end):
                    keep_len = ke - ks
                    cur_len = end - start
                    keep_conf = kept.get("confidence", 0)
                    cur_conf = err.get("confidence", 0)
                    if cur_conf > keep_conf or (cur_conf == keep_conf and cur_len > keep_len):
                        filtered.remove(kept)
                    else:
                        overlaps = True
                        break
            if not overlaps:
                filtered.append(err)

        seen = {}
        result = []
        for err in filtered:
            key = (
                err["category"],
                err["original_text"].lower().strip(),
                err.get("start_position", 0),
            )
            if key in seen:
                existing = seen[key]
                if err["confidence"] > existing["confidence"]:
                    result.remove(existing)
                    seen[key] = err
                    result.append(err)
            else:
                seen[key] = err
                result.append(err)
        return result


def _severity_order(sev: str) -> int:
    return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(sev, 1)
