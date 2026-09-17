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


def _load_json(name: str) -> Dict:
    try:
        with open(os.path.join(_DATA_DIR, name), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


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
            self._be_agreement,
            self._aux_past_participle,
            self._be_plus_past,
            self._did_plus_past,
            self._tense_past_marker,
            self._articles,
            self._prepositions,
            self._homophones,
            self._count_plurals,
            self._missing_auxiliary,
            self._reflexive_subject,
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
        # he/she/it + don't → doesn't (learner error; MEDIUM so AI decides)
        for m in re.finditer(r"\b(he|she|it)\s+(don't|do not)\b", text, re.I):
            word = text[m.start(2):m.end(2)]
            repl = "doesn't" if word.lower() == "don't" else "does not"
            out.append(self._cand(word, repl, "subject_verb", _span(m, 2), 0.6,
                                  "SVA_DOESNT", f"'{m.group(1)}' takes '{repl}'."))
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
        past = "|".join(_PAST_SIMPLE)
        for m in re.finditer(rf"\b(did)\s+({past})\b", text, re.I):
            out.append(self._cand(m.group(0), m.group(2), "verb_form", _span(m, 0), 0.9,
                                  "DID_PLUS_PAST",
                                  f"After 'did', use the base verb or drop 'did' here."))
        return out

    def _tense_past_marker(self, text: str) -> List[Dict]:
        out = []
        bases = "|".join(_IRREGULAR_PAST)
        # A modal/auxiliary immediately before the base requires the base form
        # ("could have arrived earlier" -> NOT "had arrived earlier").
        _AUX = {"have", "has", "had", "will", "would", "shall", "should",
                "can", "could", "may", "might", "must", "do", "does", "did", "to"}
        for marker in _PAST_MARKERS.finditer(text):
            window = text[max(0, marker.start() - 120): marker.start()]
            for m in re.finditer(rf"\b({bases})\b", window, re.I):
                preceding = m.string[:m.start(1)].rstrip().rsplit(None, 1)
                if preceding and preceding[-1].strip(".,;:!?").lower() in _AUX:
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
        m = re.search(r"\b(myself|yourself|himself|herself|ourselves|themselves)\b", text, re.I)
        if m:
            out.append(self._cand(m.group(0), "", "pronoun", m, 0.5, "REFLEXIVE_SUBJECT",
                                  f"'{m.group(0)}' is usually not a subject pronoun."))
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
        start, end = m.start(), m.end()
        if isinstance(m, re.Match):
            start, end = m.span()
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
    seen = set()
    out = []
    for it in items:
        key = (it["start"], it["end"], it["correct"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


_default_detector: Optional[RuleDetector] = None


def get_rule_detector() -> RuleDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = RuleDetector()
    return _default_detector


def detect_rules(text: str) -> List[Dict]:
    return get_rule_detector().generate(text)
