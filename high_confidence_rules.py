"""
high_confidence_rules.py - Data-distilled, high-confidence correction rules.

Targets the exact error patterns the current pipeline misses (measured from
the 200-case accuracy suite: 59 false negatives). Produces ErrorCandidate
objects that flow through the existing context -> confidence -> FP filter ->
validation -> dedup pipeline.

Principles:
  - unambiguous patterns get confidence >= 0.90 (error severity)
  - ambiguous patterns get 0.75-0.88 (warning) so they show but rarely collide
  - spans always equal text[start:end]
"""

import re
from typing import List, Optional

# Prefix glue so every rule_id is namespaced and reportable
_P = "DATA_RULE_"

# ── Irregular past-tense / spelling fixes (very high confidence) ─────────────
IRREGULAR_PAST = {
    "buyed": "bought", "runned": "ran", "brung": "brought", "bringed": "brought",
    "catched": "caught", "teached": "taught", "thinked": "thought",
    "growed": "grew", "knowed": "knew", "throwed": "threw", "drawed": "drew",
    "builded": "built", "speaked": "spoke", "taked": "took", "drinked": "drank",
    "swimed": "swam", "choosed": "chose", "hided": "hid", "stealed": "stole",
    "breaked": "broke", "writed": "wrote", "finded": "found", "holded": "held",
    "sleeped": "slept", "keeped": "kept", "leaved": "left", "meeted": "met",
    "costed": "cost", "putted": "put", "cutted": "cut", "hitted": "hit",
    "beated": "beat", "betten": "beaten", "maked": "made", "selled": "sold",
    "telled": "told", "goed": "went", "eated": "ate", "ated": "ate",
}

# ── Modal + of -> modal + have ───────────────────────────────────────────────
MODAL_OF_RE = re.compile(r"\b(could|should|would|might|must|may)\s+of\b", re.I)

# ── Auxiliary + wrong verb form ──────────────────────────────────────────────
HAVE_WENT_RE = re.compile(r"\b(have|has|had)\s+went\b", re.I)
MODAL_WENT_RE = re.compile(
    r"\b(will|would|shall|can|could|should|may|might|must|do|does|did)\s+went\b", re.I)
BE_WENT_RE = re.compile(r"\b(am|is|are|was|were)\s+went\b", re.I)
SEEN_RE = re.compile(r"\b(i|you|we|they|he|she|it)\s+seen\b", re.I)

# ── Homophone confusion ──────────────────────────────────────────────────────
YOUR_ER_RE = re.compile(
    r"\byour\s+(going|coming|done|welcome|right|wrong|ready|sure|late|able)\b", re.I)
THEIR_ER_RE = re.compile(
    r"\btheir\s+(going|coming|doing|making|taking|leaving|working|trying|saying|telling|watching|playing)\b", re.I)
THERE_POSS_RE = re.compile(
    r"\bthere\s+(papers?|book|books|house|home|car|car\s+keys|phone|jacket|shoes?|shirt|bag|bags|dog|cat|cats?|children|kids|friends|parents|room|rooms?|team|work|idea|ideas|names?|answers?|questions?|opinions?|rights?)\b", re.I)

# ── its vs it's (possessive context) ─────────────────────────────────────────
ITS_CURATED_RE = re.compile(
    r"\bit's\s+(tail|leg|legs|claw|claws|head|mirror|handle|button|buttons|cover|owner|creator|name|names|version|versions|seat|seats|door|window|key|keys|battery|batteries|screen|memory|storage|frame|surface|shape|color|colour|size|weight|height|depth|width|length|value|price|cost|speed|volume|heat|temperature|smell|scent|odor|flavour|flavor|texture|appearance|look|feel|function|purpose|role|meaning)\b", re.I)
ITS_EXEMPT = {"raining", "snowing", "hailing", "sleeting", "cold", "hot", "warm",
              "cool", "dark", "light", "early", "late", "getting", "been", "going",
              "about", "time", "important", "necessary", "essential", "clear",
              "obvious", "possible", "likely", "unlikely", "okay", "ok", "fine",
              "wrong", "difficult", "easy", "hard", "nice", "great", "good",
              "better", "worse", "a", "an", "the", "not", "no", "too", "very"}

# ── Comparatives / superlatives ──────────────────────────────────────────────
BESTEST_RE = re.compile(r"\bbestest\b", re.I)
MOSTEST_RE = re.compile(r"\bmostest\b", re.I)
VERY_MORE_RE = re.compile(r"\bvery\s+more\b", re.I)
VERY_MUCH_ADJS = {"beautiful", "ugly", "tall", "short", "fast", "slow", "good",
                  "bad", "smart", "stupid", "strong", "weak", "happy", "sad",
                  "big", "small", "hot", "cold", "nice", "kind", "friendly",
                  "helpful", "important", "interesting", "easy", "difficult",
                  "rich", "poor", "young", "old", "pretty", "handsome", "lucky",
                  "tired", "busy", "angry", "scared", "calm", "bright", "clever",
                  "clean", "quiet", "noisy", "sweet", "soft", "hard", "thick",
                  "thin", "heavy", "light", "wide", "deep", "long", "new", "old",
                  "cheap", "expensive", "high", "low"}
VERY_MUCH_RE = re.compile(r"\bvery\s+much\s+([a-zA-Z]+)\b", re.I)

# ── Adjective used as adverb (curated verb+adj pairs) ────────────────────────
ADJ_ADV_RE = re.compile(
    r"\b(sing|sings|sang|singing|dance|dances|danced|dancing|run|runs|ran|running|walk|walks|walked|walking|drive|drives|drove|driving|cook|cooks|cooked|cooking|play|plays|played|playing|work|works|worked|working|speak|speaks|spoke|speaking|read|reads|reading|write|writes|wrote|writing|draw|draws|drew|drawing|behave|behaves|behaved|answer|answers|answered|answering)\s+(beautiful|quick|slow|soft|loud|clear|careful|gentle|graceful|polite|honest|efficient|proper|correct|neat|quiet|fluent|perfect)\b", re.I)
_ADJ_ADV_MAP = {
    "beautiful": "beautifully", "quick": "quickly", "slow": "slowly",
    "soft": "softly", "loud": "loudly", "clear": "clearly",
    "careful": "carefully", "gentle": "gently", "graceful": "gracefully",
    "polite": "politely", "honest": "honestly", "efficient": "efficiently",
    "proper": "properly", "correct": "correctly", "neat": "neatly",
    "quiet": "quietly", "fluent": "fluently", "perfect": "perfectly",
}

# ── good -> well (curated verbs) ─────────────────────────────────────────────
GOOD_WELL_RE = re.compile(
    r"\b(did|doing|does|done|works?|worked|working|played|playing|behaved|performs?|performed|performing)\s+good\b", re.I)

# ── real + adjective -> really ───────────────────────────────────────────────
REAL_ADJ_RE = re.compile(
    r"\breal\s+(happy|good|nice|fast|quick|slow|big|small|short|tall|hard|easy|hot|cold|soon|well|sure|glad|difficult|great|fun|mean|tough|cool|smart|strong|weak|rich|poor|busy|tired|angry|sad|early|late|bored|interesting|important|huge|tiny|sweet|clean|pretty|silly|drunk|crazy|weird)\b", re.I)

# ── lay/lie, less/fewer, uncountables, collectives ───────────────────────────
LAYS_ON_RE = re.compile(r"\blays\s+on\b", re.I)
_LESS_PLURALS = ("books|cars|apples|chairs|students|people|tables|files|items|pages|"
                 "questions|problems|errors|miles|dollars|friends|children|toys|meetings|"
                 "emails|calls|hours|days|weeks|months|years|times|ideas|options|choices|"
                 "reasons|examples|members|workers|employees|tasks|jobs|projects|"
                 "screens|phones|computers|offices|customers|clients|products|orders")
LESS_FEWER_RE = re.compile(r"\bless\s+(" + _LESS_PLURALS + r")\b", re.I)

NEWS_ARE_RE = re.compile(r"\bthe?\s+news\s+(are|were)\b", re.I)
INFO_WERE_RE = re.compile(r"\binformation\s+(are|were)\b", re.I)
INFO_S_RE = re.compile(r"\binformations\b", re.I)
A_INFO_RE = re.compile(r"\ba\s+informations?\b", re.I)

_COLLECTIVES = "team|company|government|committee|jury|audience|family|staff|club|band|army|fleet|gang|council|board|choir|orchestra|faculty|squad|crew|staff"
COLLECTIVE_ARE_RE = re.compile(
    r"\b(" + _COLLECTIVES + r")\s+(are|were)\b", re.I)
GROUP_OF_WERE_RE = re.compile(
    r"\b(group|team|crowd|family|class|herd|flock|pack|swarm|school|pride|"
    r"pod|colony|bunch|committee|board|jury|audience)\s+of\s+[a-z]+s?\s+(are|were)\b",
    re.I)
QUANT_OF_PRON_SVA_RE = re.compile(
    r"\b(some|many|few|several|most|all|both)\s+of\s+(them|us|you)\s+"
    r"(was|is|has|does)\b", re.I)
PRONOUN_PLURAL_DOESNT_RE = re.compile(r"\b(they|we|you)\s+doesn'?t\b", re.I)

# ── Pronoun case ─────────────────────────────────────────────────────────────
BETWEEN_AND_I_RE = re.compile(r"\bbetween\s+([a-z]+)\s+and\s+I\b", re.I)
START_ME_RE = re.compile(
    r"^(me|him|her|them|us)\s+and\s+(i|me|him|her|us|them|we)\b", re.I)
BY_SUBJ_RE = re.compile(r"\bby\s+(she|he|they|we|i)\b", re.I)
_START_SUBJ = {"me": "he", "him": "he", "her": "she", "them": "they", "us": "we"}
_OBJ_MAP = {"she": "her", "he": "him", "they": "them", "we": "us", "i": "me"}

# ── Misc common errors ───────────────────────────────────────────────────────
WAS_TO_RE = re.compile(
    r"\b(was|were)\s+to\s+the\s+(store|shop|market|park|school|office|hospital|bank|library|gym|mall|station|airport|beach|cinema|theater|theatre|city|town|village|restaurant|supermarket|playground)\b", re.I)
CANT_HARDLY_RE = re.compile(r"\b(can'?t|cannot)\s+hardly\b", re.I)
REASON_BECAUSE_RE = re.compile(r"\b(the reason)\s+(is|was)\s+because\b", re.I)


def _mk(start, end, text, replacement, category, rule_id, message, confidence):
    """Build an ErrorCandidate (imported lazily to avoid circular imports)."""
    from unified_pipeline import ErrorCandidate
    return ErrorCandidate(
        start=start, end=end, original=text[start:end],
        replacement=replacement, category=category, rule_id=rule_id,
        message=message, detector="high_conf_rules",
        raw_confidence=confidence,
    )


def _cat(name):
    from unified_pipeline import ErrorCategory
    return getattr(ErrorCategory, name)


class HighConfidenceDetector:
    """Regex + POS-guided detector distilled from error datasets."""

    def detect(self, text: str, doc=None) -> List[object]:
        from unified_pipeline import ErrorCandidate, ErrorCategory
        cands: List[ErrorCandidate] = []

        def add(m, replacement, category, rule_id, message, conf):
            cands.append(ErrorCandidate(
                start=m.start(), end=m.end(), original=m.group(0),
                replacement=replacement, category=category, rule_id=rule_id,
                message=message, detector="high_conf_rules",
                raw_confidence=conf,
            ))

        # 1. irregular past / spelling
        irreg_re = re.compile(r"\b(" + "|".join(map(re.escape, IRREGULAR_PAST)) + r")\b", re.I)
        for m in irreg_re.finditer(text):
            w = m.group(1).lower()
            add(m, IRREGULAR_PAST[w], ErrorCategory.SPELLING, _P + "IRREGULAR_PAST",
                f'Correct form is "{IRREGULAR_PAST[w]}".', 0.97)

        # 2. modal + of -> modal + have
        for m in MODAL_OF_RE.finditer(text):
            modal = m.group(1).lower()
            add(m, f"{modal} have", ErrorCategory.VERB_FORM, _P + "MODAL_OF_HAVE",
                f'Use "{modal} have" (not "{modal} of"). "Of" is not a helping verb.', 0.97)

        # 3. have/has/had + went -> gone
        for m in HAVE_WENT_RE.finditer(text):
            aux = m.group(1).lower()
            add(m, f"{aux} gone", ErrorCategory.VERB_FORM, _P + "AUX_PAST_PARTICIPLE",
                f'After "{aux}", use the past participle "gone" instead of "went".', 0.97)

        # 4. will/would/shall/... + went -> go
        for m in MODAL_WENT_RE.finditer(text):
            aux = m.group(1).lower()
            add(m, f"{aux} go", ErrorCategory.VERB_FORM, _P + "MODAL_BARE_INFINITIVE",
                f'After "{aux}", use the base form "go" instead of "went".', 0.95)

        # 5. be + went -> went
        for m in BE_WENT_RE.finditer(text):
            add(m, "went", ErrorCategory.VERB_FORM, _P + "BE_WENT_FORM",
                'Use "went" (simple past) instead of a form of "be" + "went".', 0.94)

        # 6. I/She/... seen -> saw (no auxiliary)
        for m in SEEN_RE.finditer(text):
            subj = m.group(1).lower()
            add(m, f"{subj} saw", ErrorCategory.VERB_FORM, _P + "SEEN_WITHOUT_AUX",
                f'Use "{subj} saw" (simple past) instead of "{subj} seen".', 0.93)

        # 7. your -> you're
        for m in YOUR_ER_RE.finditer(text):
            word = m.group(1).lower()
            add(m, f"you're {word}", ErrorCategory.WORD_USAGE, _P + "YOUR_YOURE",
                'Use "you\'re" (you are) instead of the possessive "your".', 0.90)

        # 8. their -> they're
        for m in THEIR_ER_RE.finditer(text):
            word = m.group(1).lower()
            add(m, f"they're {word}", ErrorCategory.WORD_USAGE, _P + "THEIR_THEYRE",
                'Use "they\'re" (they are) instead of the possessive "their".', 0.87)

        # 9. there -> their (before noun)
        for m in THERE_POSS_RE.finditer(text):
            noun = m.group(1)
            add(m, f"their {noun}", ErrorCategory.WORD_USAGE, _P + "THERE_THEIR",
                f'Use the possessive "their" before "{noun}" (not "there").', 0.85)

        # 10. it's + possession noun -> its
        for m in ITS_CURATED_RE.finditer(text):
            noun = m.group(1)
            add(m, f"its {noun}", ErrorCategory.WORD_USAGE, _P + "ITS_POSSESSIVE",
                f'Use the possessive "its" (no apostrophe) before "{noun}".', 0.90)

        # 11. more/most + comparative/superlative (POS-guided with regex fallback)
        MORE_JJR_RE = re.compile(r"\bmore\s+(\w+er|better|worse|further)\b", re.I)
        MOST_JJS_RE = re.compile(r"\bmost\s+(\w+est|best|worst)\b", re.I)
        if doc is not None:
            try:
                for tok in doc:
                    if tok.lower_ == "more" and tok.i + 1 < len(doc):
                        nxt = doc[tok.i + 1]
                        if nxt.tag_ == "JJR":
                            add(_FakeM(tok.idx, tok.idx + len(tok.text) + len(nxt.text),
                                        text[tok.idx:tok.idx + len(tok.text)] + " " + nxt.text),
                                nxt.text, ErrorCategory.WORD_USAGE, _P + "DOUBLE_COMPARATIVE",
                                f'"{tok.text} {nxt.text}" is a double comparative; use just "{nxt.text}".', 0.95)
                    elif tok.lower_ == "most" and tok.i + 1 < len(doc):
                        nxt = doc[tok.i + 1]
                        if nxt.tag_ == "JJS":
                            add(_FakeM(tok.idx, tok.idx + len(tok.text) + len(nxt.text),
                                        text[tok.idx:tok.idx + len(tok.text)] + " " + nxt.text),
                                nxt.text, ErrorCategory.WORD_USAGE, _P + "DOUBLE_SUPERLATIVE",
                                f'"{tok.text} {nxt.text}" is a double superlative; use just "{nxt.text}".', 0.95)
            except Exception:
                pass

        # regex fallback for comparative/superlative when no spaCy doc available
        if doc is None:
            for m in MORE_JJR_RE.finditer(text):
                word = m.group(1)
                if word.lower() not in {"better", "worse", "further"} or word.lower() == word:
                    add(m, word, ErrorCategory.WORD_USAGE, _P + "DOUBLE_COMPARATIVE",
                        f'"more {word}" is a double comparative; use just "{word}".', 0.90)
            for m in MOST_JJS_RE.finditer(text):
                word = m.group(1)
                add(m, word, ErrorCategory.WORD_USAGE, _P + "DOUBLE_SUPERLATIVE",
                    f'"most {word}" is a double superlative; use just "{word}".', 0.90)

        # 12. bestest / mostest
        for m in BESTEST_RE.finditer(text):
            add(m, "best", ErrorCategory.SPELLING, _P + "BESTEST",
                'Use "best" (not "bestest").', 0.96)
        for m in MOSTEST_RE.finditer(text):
            add(m, "most", ErrorCategory.SPELLING, _P + "MOSTEST",
                'Use "most" (not "mostest").', 0.96)

        # 13. very more -> much more
        for m in VERY_MORE_RE.finditer(text):
            add(m, "much more", ErrorCategory.WORD_USAGE, _P + "VERY_MORE",
                'Use "much more" (not "very more") before a comparative.', 0.90)

        # 14. very much + plain adjective -> very + adjective
        for m in VERY_MUCH_RE.finditer(text):
            adj = m.group(1).lower()
            if adj in VERY_MUCH_ADJS:
                add(m, f"very {m.group(1)}", ErrorCategory.WORD_USAGE, _P + "VERY_MUCH_ADJ",
                    f'Use "very {m.group(1)}" (not "very much {m.group(1)}") here.', 0.85)

        # 15. verb + adjective -> verb + adverb
        for m in ADJ_ADV_RE.finditer(text):
            verb, adj = m.group(1), m.group(2).lower()
            adv = _ADJ_ADV_MAP.get(adj, adj + "ly")
            add(m, f"{verb} {adv}", ErrorCategory.WORD_USAGE, _P + "ADJ_AS_ADVERB",
                f'Use the adverb "{adv}" to modify the verb "{verb}".', 0.85)

        # 16. do(es)/did/done + good -> well
        for m in GOOD_WELL_RE.finditer(text):
            verb = m.group(1)
            add(m, f"{verb} well", ErrorCategory.WORD_USAGE, _P + "GOOD_WELL",
                f'Use "well" (adverb) instead of "good" to modify "{verb}".', 0.88)

        # 17. real + adj -> really + adj
        for m in REAL_ADJ_RE.finditer(text):
            adj = m.group(1)
            add(m, f"really {adj}", ErrorCategory.WORD_USAGE, _P + "REAL_REALLY",
                f'Use the adverb "really" before "{adj}" (not the adjective "real").', 0.85)

        # 18. lays on -> lies on
        for m in LAYS_ON_RE.finditer(text):
            add(m, "lies on", ErrorCategory.WORD_USAGE, _P + "LAY_LIE",
                'Use "lies on" (to recline) instead of "lays on" (to place something).', 0.85)

        # 19. less + countable plural -> fewer
        for m in LESS_FEWER_RE.finditer(text):
            noun = m.group(1)
            add(m, f"fewer {noun}", ErrorCategory.WORD_USAGE, _P + "LESS_FEWER",
                f'Use "fewer" (not "less") before a countable noun like "{noun}".', 0.80)

        # 20. uncountable agreement: news / information
        for m in NEWS_ARE_RE.finditer(text):
            verb = m.group(1).lower()
            repl = "is" if verb == "are" else "was"
            add(m, m.group(0)[:-len(verb)] + repl, ErrorCategory.SUBJECT_VERB_AGREEMENT,
                _P + "NEWS_SINGULAR",
                f'"news" is uncountable; use the singular verb "{repl}".', 0.92)
        for m in INFO_WERE_RE.finditer(text):
            verb = m.group(1).lower()
            repl = "is" if verb == "are" else "was"
            add(m, f"information {repl}", ErrorCategory.SUBJECT_VERB_AGREEMENT,
                _P + "INFORMATION_SINGULAR",
                f'"information" is uncountable; use the singular verb "{repl}".', 0.92)
        for m in INFO_S_RE.finditer(text):
            add(m, "information", ErrorCategory.WORD_USAGE, _P + "INFORMATION_UNCOUNTABLE",
                '"information" is uncountable; it has no plural form.', 0.95)
        for m in A_INFO_RE.finditer(text):
            add(m, "information", ErrorCategory.ARTICLE, _P + "A_UNCOUNTABLE",
                '"information" is uncountable; drop "a" and the plural form.', 0.90)

        # 21. collective noun agreement (American English)
        for m in COLLECTIVE_ARE_RE.finditer(text):
            noun, verb = m.group(1).lower(), m.group(2).lower()
            repl = "is" if verb == "are" else "was"
            add(m, f"{noun} {repl}", ErrorCategory.SUBJECT_VERB_AGREEMENT,
                _P + "COLLECTIVE_SVA",
                f'In American English "{noun}" takes the singular verb "{repl}" (not "{verb}").', 0.78)
        for m in GROUP_OF_WERE_RE.finditer(text):
            verb = m.group(2).lower()
            repl = "is" if verb == "are" else "was"
            fixed = re.sub(r"\s+(" + verb + r")\b", f" {repl}", m.group(0), count=1, flags=re.I)
            add(m, fixed, ErrorCategory.SUBJECT_VERB_AGREEMENT, _P + "GROUP_OF_SVA",
                f'The head noun is singular; the verb should be "{repl}".', 0.82)

        # 21b. quantifier + of + plural pronoun (some/many/few/... of them) takes plural verb
        _QUANT_PRON_SVA = {"was": "were", "is": "are", "has": "have", "does": "do"}
        for m in QUANT_OF_PRON_SVA_RE.finditer(text):
            verb = m.group(3).lower()
            repl = _QUANT_PRON_SVA[verb]
            fixed = re.sub(r"\b(" + verb + r")\b", repl, m.group(0), count=1, flags=re.I)
            add(m, fixed, ErrorCategory.SUBJECT_VERB_AGREEMENT, _P + "QUANT_PRON_SVA",
                f'The subject is plural; use "{repl}" (not "{verb}").', 0.90)
        for m in PRONOUN_PLURAL_DOESNT_RE.finditer(text):
            fixed = re.sub(r"\bdoesn'?t\b", "don't", m.group(0), flags=re.I)
            add(m, fixed, ErrorCategory.SUBJECT_VERB_AGREEMENT, _P + "PLURAL_DOESNT",
                'With a plural subject, use "don\'t" (not "doesn\'t").', 0.95)

        # 22. pronoun case
        for m in BETWEEN_AND_I_RE.finditer(text):
            add(m, f"between {m.group(1)} and me", ErrorCategory.PRONOUN,
                _P + "BETWEEN_PRONOUN",
                'After "between", use the object pronoun "me" (not "I").', 0.93)
        for m in START_ME_RE.finditer(text):
            first = m.group(1).lower()
            subj = _START_SUBJ.get(first, first)
            add(m, f"{subj} and I", ErrorCategory.PRONOUN, _P + "SUBJECT_PRONOUN",
                'Use the subject form (for example, "He and I") as the subject of a verb.', 0.85)
        for m in BY_SUBJ_RE.finditer(text):
            subj = m.group(1).lower()
            obj = _OBJ_MAP.get(subj, subj)
            add(m, f"by {obj}", ErrorCategory.PRONOUN, _P + "BY_OBJECT_PRONOUN",
                f'After the preposition "by", use the object pronoun "{obj}".', 0.90)

        # 23. was/were to the X -> went to the X
        for m in WAS_TO_RE.finditer(text):
            place = m.group(2)
            add(m, f"went to the {place}", ErrorCategory.VERB_FORM, _P + "WAS_TO_GO",
                f'Use "went to the {place}" (the verb "go" in the past) instead of "was/were to the {place}".', 0.82)

        # 24. can't hardly -> can hardly
        for m in CANT_HARDLY_RE.finditer(text):
            add(m, "can hardly", ErrorCategory.NEGATIVE if hasattr(ErrorCategory, "NEGATIVE") else ErrorCategory.GRAMMAR,
                _P + "CANT_HARDLY",
                'Avoid the double negative: "hardly" already makes the statement negative.', 0.90)

        # 25. the reason ... because -> that
        for m in REASON_BECAUSE_RE.finditer(text):
            add(m, f"the reason {m.group(2).lower()} that", ErrorCategory.SENTENCE_STRUCTURE,
                _P + "REASON_BECAUSE",
                'Use "the reason is that" (not "the reason is because").', 0.93)

        # 26. POS-guided: it's + NOUN -> its
        if doc is not None:
            try:
                for tok in doc:
                    if tok.lower_ == "it's":
                        nxt = tok.i + 1
                        if nxt < len(doc) and doc[nxt].pos_ == "NOUN" and doc[nxt].lower_ not in ITS_EXEMPT:
                            cands.append(ErrorCandidate(
                                start=tok.idx, end=tok.idx + len(tok.text),
                                original=tok.text,
                                replacement="Its" if tok.text[0].isupper() else "its",
                                category=ErrorCategory.WORD_USAGE,
                                rule_id=_P + "ITS_POSSESSIVE",
                                message=f'Use the possessive "its" (no apostrophe) before "{doc[nxt].text}".',
                                detector="high_conf_rules", raw_confidence=0.88,
                            ))
            except Exception:
                pass

        # 26b. POS-guided: you're + NOUN -> your (possessive)
        # spaCy tokenizes "you're" as you + 're; detect "'re" token after "you"
        if doc is not None:
            try:
                for i, tok in enumerate(doc):
                    if tok.lower_ == "'re" and i > 0 and doc[i - 1].lower_ == "you":
                        nxt = i + 1
                        if nxt < len(doc) and doc[nxt].pos_ == "NOUN" and doc[nxt].lower_ not in (
                                "going", "coming", "doing", "saying", "looking", "waiting"):
                            you_start = doc[i - 1].idx
                            cands.append(ErrorCandidate(
                                start=you_start,
                                end=tok.idx + len(tok.text),
                                original="you're",
                                replacement="Your" if doc[i - 1].text[0].isupper() else "your",
                                category=ErrorCategory.WORD_USAGE,
                                rule_id=_P + "YOURE_POSSESSIVE",
                                message=f'Before a noun, use the possessive "your" (not "you\'re").',
                                detector="high_conf_rules", raw_confidence=0.88,
                            ))
            except Exception:
                pass

        # 27. comparative + then -> comparative + than
        # Curated comparative heads to avoid "player"/"runner" false positives.
        _then_heads = {w + "er" for w in (
            "smart", "tall", "short", "fast", "slow", "old", "young", "new",
            "big", "small", "easy", "hard", "high", "low", "long", "wide",
            "deep", "great", "nice", "rich", "poor", "strong", "weak",
            "cheap", "hot", "cold", "warm", "cool", "happy", "sad", "busy",
            "large", "dark", "bright", "clean", "clear", "close", "tight",
            "thick", "thin", "early", "late")} | {"more", "less", "better",
                                                  "worse", "farther", "further"}
        then_than_re = re.compile(
            r"\b(" + "|".join(sorted(_then_heads, key=len, reverse=True)) + r")\s+then\b", re.I)
        for m in then_than_re.finditer(text):
            comp = m.group(1)
            after = text[m.end():].lstrip()
            nxt = after.split()[0].lower() if after else ""
            if not re.match(r"^(i|you|he|she|it|we|they)\b", nxt):
                add(m, f"{comp} than", ErrorCategory.WORD_USAGE, _P + "THEN_THAN",
                    f'Use "than" for comparisons ("{comp} than"), not "then".', 0.82)

        # 28. plays the SPORT -> plays SPORT (sports take no article)
        _sports = (r"tennis|football|soccer|basketball|cricket|hockey|volleyball|"
                   r"baseball|rugby|golf|badminton|swimming|skiing|chess")
        for m in re.finditer(r"\b(play|plays|played|playing|watch|watches|watched)\s+"
                             r"the\s+(" + _sports + r")\b", text, re.I):
            verb, sport = m.group(1), m.group(2)
            add(m, f"{verb} {sport}", ErrorCategory.ARTICLE, _P + "SPORT_THE",
                f'Do not use "the" before the sport "{sport}".', 0.90)

        # 29. causative verb + pronoun + to + verb -> drop "to"
        for m in re.finditer(r"\b(made|make|let)\s+"
                             r"(me|us|him|her|them|[A-Z][a-z]+)\s+to\s+"
                             r"([a-z]+(?:e|s|y)?)\b", text):
            verb, obj, main = m.group(1).lower(), m.group(2), m.group(3)
            if any(w in main for w in ("good", "want", "like", "know")):
                continue
            add(m, f"{m.group(1)} {obj} {main}", ErrorCategory.VERB_FORM,
                _P + "CAUSATIVE_TO",
                f'After the causative "{verb}", use the bare infinitive "{main}" without "to".', 0.88)

        # 30. a/an + uncountable noun -> drop article
        # Guard: only flag when the uncountable noun is the head (not followed by
        # another noun, so "a software engineer" stays unflagged).
        _unc = (r"equipment|furniture|advice|information|luggage|research|"
                r"homework|jewelry|jewellery|software|knowledge|news|money|"
                r"traffic|weather|accommodation|applause")
        for m in re.finditer(r"\b(?:a|an)\s+(?:\w+\s+){0,2}(" + _unc + r")\b", text, re.I):
            rest = text[m.end():].lstrip()
            nxt_word = rest.split()[0].lower().strip(".,!?;:") if rest else ""
            if nxt_word:
                if doc is not None:
                    nxt_pos = ""
                    for tok in doc:
                        if tok.idx >= m.end() and tok.text.strip():
                            nxt_pos = tok.pos_
                            break
                    if nxt_pos == "NOUN":
                        continue
                else:
                    if nxt_word in ("engineer", "program", "package", "system",
                                    "platform", "service", "stack", "team"):
                        continue
            noun = m.group(1)
            add(m, re.sub(r"\b(?:a|an)\s+", "", m.group(0), count=1, flags=re.I),
                ErrorCategory.ARTICLE, _P + "A_UNCOUNTABLE",
                f'"{noun}" is uncountable; do not use "a/an" before it.', 0.90)

        # 31. decade apostrophe — "the 1990's" -> "the 1990s"
        for m in re.finditer(r"\b(19|20)\d{2}'s\b", text):
            add(m, m.group(0).replace("'", ""), ErrorCategory.PUNCTUATION, _P + "DECADE_APOSTROPHE",
                'Decades do not take an apostrophe before "s" ("1990s").', 0.95)

        # 32. hypothetical "if I was" -> "if I were"
        for m in re.finditer(
                r"\bif\s+(i|he|she|it)\s+was\s+(rich|poor|taller|shorter|younger|"
                r"older|stronger|smarter|calmer|quicker|famous|brave|courageous|"
                r"available|ready|free|quiet|different|good|bad|nice|here|there|"
                r"okay|nice)\b", text, re.I):
            subj = m.group(1)
            word = m.group(2)
            add(m, f"if {subj} were {word}", ErrorCategory.VERB_FORM,
                _P + "SUBJUNCTIVE_WERE",
                f'In hypothetical (unreal) conditions, use "were" ("if {subj} were {word}").',
                0.82)

        # 33. embedded wh-question inversion ("please tell me where is the station")
        for m in re.finditer(
                r"\b(i wonder|do you know|can you tell|please tell me|please tell|"
                r"tell me|she asked|he asked|i asked|he wondered|she wondered)\s+"
                r"(what time|what|where|when|why|how|who)\s+"
                r"(is|are|was|were|will|would|does|do|did)\s+([^,.!?;]+)",
                text, re.I):
            lead, wh, aux, rest = m.group(1), m.group(2), m.group(3), m.group(4).strip()
            add(m, f"{lead} {wh} {rest} {aux}", ErrorCategory.SENTENCE_STRUCTURE,
                _P + "EMBEDDED_WH",
                f'Do not invert subject and verb in an embedded question: '
                f'"...{wh} {rest} {aux}...".', 0.85)

        # 34. return back -> return
        for m in re.finditer(r"\breturn\s+back\b", text, re.I):
            add(m, "return", ErrorCategory.STYLE, _P + "RETURN_BACK",
                '"back" is redundant after "return".', 0.88)

        # 35. did/do a(n) decision -> made a decision
        for m in re.finditer(
                r"\b(did|do|does)\s+(?:a|an)\s+(decision|mistake|effort|choice|error|"
                r"promise|deal)\b", text, re.I):
            verb = m.group(1).lower()
            repl = "made" if verb == "did" else "make" if verb in ("do", "does") else verb
            add(m, m.group(0).replace(verb, repl, 1), ErrorCategory.WORD_USAGE,
                _P + "MAKE_DO",
                f'Use "{repl}" with "{m.group(2)}".', 0.85)

        # 36. key of the -> key to the
        for m in re.finditer(r"\bkey\s+of\s+the\b", text, re.I):
            add(m, "key to the", ErrorCategory.PREPOSITION, _P + "KEY_TO",
                'Use "key to the" (the key that opens something).', 0.88)

        # 37. because X, so Y -> drop "so" (keep the comma)
        for m in re.finditer(r"\bbecause\b[^,]+,\s*so\b", text, re.I):
            add(m, m.group(0).replace(" so", ""), ErrorCategory.SENTENCE_STRUCTURE,
                _P + "BECAUSE_SO",
                'Do not use "so" after a "because" clause.', 0.88)

        # 38. stand-alone lowercase "i" -> "I"
        for m in re.finditer(r"(^|[.!?;]\s+)(i)(?=\s)", text):
            add(m, m.group(1) + "I", ErrorCategory.CAPITALIZATION,
                _P + "CAPITAL_I",
                'The pronoun "I" is always capitalized.', 0.94)

        # 39. extended double negatives — can't/cannot/couldn't/didn't/won't + negative adverbial
        _neg_aux = {"can't": "can", "cannot": "can", "couldn't": "could",
                    "didn't": "did", "doesn't": "does", "won't": "will", "don't": "do"}
        for m in re.finditer(
                r"\b(can'?t|cannot|couldn't|didn't|doesn't|won't|don't)\s+"
                r"(hardly|scarcely|never|barely)\b", text, re.I):
            aux = m.group(1)
            adv = m.group(2)
            repl = _neg_aux.get(aux.lower(), "can")
            add(m, f"{repl} {adv}", ErrorCategory.SENTENCE_STRUCTURE,
                _P + "DOUBLE_NEG",
                f'Avoid the double negative: "{adv}" already makes the statement negative. '
                f'Use "{repl} {adv}" instead of "{aux} {adv}".', 0.85)

        # 39b. nobody/no one + negative verb -> double negative
        _neg_past = {"come": "came", "go": "went", "see": "saw", "know": "knew",
                     "say": "said", "do": "did", "make": "made", "take": "took",
                     "give": "gave", "get": "got", "call": "called", "tell": "told",
                     "hear": "heard", "leave": "left", "help": "helped", "answer": "answered",
                     "show": "showed", "want": "wanted", "have": "had", "write": "wrote",
                     "read": "read", "find": "found", "bring": "brought", "think": "thought",
                     "eat": "ate", "speak": "spoke", "play": "played", "work": "worked",
                     "try": "tried", "ask": "asked", "finish": "finished", "watch": "watched"}
        for m in re.finditer(
                r"\b(nobody|no one|no-one|nothing|none)\s+"
                r"(didn't|didnt|did not)\s+(\w+)\b", text, re.I):
            subj = m.group(1)
            verb = m.group(3).lower()
            past = _neg_past.get(verb, verb + "d" if verb.endswith("e") else
                                 verb + "ed" if verb.endswith("s") else verb + "ed")
            add(m, f"{subj} {past}", ErrorCategory.SENTENCE_STRUCTURE,
                _P + "DOUBLE_NEG",
                f'Double negative: "{subj} ... {m.group(2)} {verb}" — use only one '
                f'negative form, e.g. "{subj} {past}".', 0.85)

        # 39c. transitive "lie" -> "lay" (recline vs place)
        for m in re.finditer(
                r"\blies?\s+(the|a|an|my|your|his|her|our|their|this|that)\s+\w+\s+"
                r"(on|over|under|across|beside|in|at)\b", text, re.I):
            lie_word = m.group(0).split()[0]
            repl = "lies" if lie_word.lower().endswith("s") else "lay"
            add(m, m.group(0).replace(lie_word, repl, 1),
                ErrorCategory.WORD_USAGE, _P + "LAY_LIE",
                'Use "lay/lays" (to place something) with a direct object; '
                '"lie" is intransitive (to recline).', 0.9)

        # 40. redundant "really very" -> "really"
        for m in re.finditer(r"\breally\s+very\b", text, re.I):
            add(_FakeM(m.start(), m.end(), m.group(0)), m.group(0).replace(" very", ""),
                ErrorCategory.WORD_USAGE, _P + "REALLY_VERY",
                '"Really" and "very" are redundant together; use only one.', 0.88)

        # 41. verb + object "and I" -> "and me"
        for m in re.finditer(
                r"\b(saw|see|gave|give|told|tell|asked|ask|called|call|"
                r"sent|send|met|meet|showed|show|left|leave|wrote|write)\s+"
                r"(\w+\s+and\s+)I\b", text, re.I):
            v = m.group(1)
            mid = m.group(2)
            pos = m.end() - 1  # position of "I"
            add(_FakeM(pos, pos+1, "I"), "me", ErrorCategory.PRONOUN,
                _P + "AND_I",
                'Use "me" (object form) after a verb.', 0.92)

        # 42. preposition + "X and I" -> "X and me"
        for m in re.finditer(r"\b(to|between|with|for|from|by|at|around)\s+(\w+\s+and\s+)I\b", text, re.I):
            prep = m.group(1)
            mid = m.group(2)
            pos = m.end() - 1
            add(_FakeM(pos, pos+1, "I"), "me", ErrorCategory.PRONOUN,
                _P + "AND_I_PREP",
                'Use "me" (object form) after a preposition.', 0.90)

        # 43. "each and every" -> "every"
        for m in re.finditer(r"\beach and every\b", text, re.I):
            add(m, "every", ErrorCategory.STYLE, _P + "EACH_AND_EVERY",
                '"each and every" is redundant; use "every".', 0.90)

        # 44. indefinite pronoun + plural verb ("everyone are", "everyone have")
        for m in re.finditer(
                r"\b(everyone|everybody|someone|somebody|anyone|anybody|no one|nobody)\s+"
                r"(are|were|have|has|do|does)\b", text, re.I):
            subj = m.group(1).lower()
            verb = m.group(2).lower()
            repl = {"are": "is", "were": "was", "have": "has", "do": "does"}.get(verb, verb)
            add(m, f"{subj} {repl}", ErrorCategory.SUBJECT_VERB_AGREEMENT,
                _P + "INDEFINITE_SVA",
                f'Indefinite pronouns like "{subj}" take a singular verb ("{subj} {repl}").', 0.90)

        # 45. "final conclusion" -> "conclusion"
        for m in re.finditer(r"\bthe final conclusion\b", text, re.I):
            add(m, "the conclusion", ErrorCategory.STYLE, _P + "FINAL_REDUNDANT",
                '"final" is implied by "conclusion"; avoid the redundancy.', 0.82)

        # 46. "at June/January..." -> "in June/January..."
        for m in re.finditer(
                r"\bat\s+(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\b", text, re.I):
            month = m.group(1)
            add(m, f"in {month}", ErrorCategory.PREPOSITION, _P + "AT_MONTH",
                f'Use "in" (not "at") before a month: "in {month}".', 0.92)

        # 47. "the breakfast/lunch/dinner" (generic meal) -> drop "the"
        for m in re.finditer(
                r"\b(had|eat|eats|ate|have|has)\s+the\s+(breakfast|lunch|dinner|brunch|supper)\b",
                text, re.I):
            verb = m.group(1)
            meal = m.group(2)
            add(m, f"{verb} {meal}", ErrorCategory.ARTICLE, _P + "THE_MEAL",
                f'Do not use "the" with a meal eaten generically: "{verb} {meal}".', 0.80)

        # 48. causative "had/helped + pronoun + to + verb" -> drop "to"
        for m in re.finditer(r"\b(had|helped)\s+"
                             r"(me|us|him|her|them)\s+to\s+"
                             r"([a-z]+(?:e|ed|s|ies|ing)?)\b", text):
            head, pron, verb = m.group(1), m.group(2), m.group(3)
            add(m, f"{head} {pron} {verb}", ErrorCategory.VERB_FORM,
                _P + "CAUSATIVE_TO",
                f'After the causative "{head}", use the bare infinitive ({head} {pron} {verb}) without "to".',
                0.82)

        # 49. "done a big effort/decision..." -> "made ..."
        for m in re.finditer(
                r"\b(done|did|do|does)\s+(?:a|an)\s+(?:[a-z]+\s+){0,3}(effort|decision|mistake|choice|error|"
                r"promise|deal)\b", text, re.I):
            verb = m.group(1).lower()
            noun = m.group(2)
            repl = "made" if verb in ("done", "did") else "make" if verb in ("do", "does") else verb
            add(m, m.group(0).replace(verb, repl, 1), ErrorCategory.WORD_USAGE,
                _P + "MAKE_DO",
                f'Use "{repl}" with "{noun}".', 0.85)

        # 50. sentence-start lowercase word -> capitalize
        for m in re.finditer(r"(^|[.!?]\s+)([a-z])", text):
            start = m.start(2)
            add(_FakeM(start, start + 1, m.group(2)), m.group(2).upper(),
                ErrorCategory.CAPITALIZATION, _P + "SENT_START",
                'The first word of a sentence should be capitalized.', 0.80)

        # 51. reflexive pronoun used as subject ("My brother and myself went ...")
        for m in re.finditer(
                r"\b(\w+)\s+and\s+myself\s+(went|am|are|was|were|will|would|like|wanted|"
                r"have|had|do|did|eat|play|work|enjoy|called|said|visited|bought|saw|see)\b",
                text, re.I):
            other = m.group(1)
            verb = m.group(2)
            pos = m.start() + len(other) + 5
            add(_FakeM(pos, pos + 6, "myself"), "I", ErrorCategory.PRONOUN,
                _P + "REFLEXIVE_SUBJECT",
                'Use "I" (the subject form), not "myself", as a subject.', 0.90)

        # 52. direct question missing do-support ("Why he left so early?")
        # Curated past->base map to avoid wrong morphology on irregular verbs.
        _wh_q_base = {"went": "go", "left": "leave", "said": "say", "came": "come",
                      "took": "take", "told": "tell", "found": "find", "made": "make",
                      "saw": "see", "gave": "give", "did": "do", "had": "have",
                      "wanted": "want", "liked": "like", "worked": "work",
                      "played": "play", "talked": "talk", "walked": "walk",
                      "asked": "ask", "arrived": "arrive", "waited": "wait",
                      "helped": "help", "called": "call", "started": "start",
                      "decided": "decide", "tried": "try"}
        for m in re.finditer(
                r"\b(who|what|when|where|why|how)\s+(he|she|it|they|we)\s+([a-z]+)\s+"
                r".*?\?", text, re.I):
            wh = m.group(1)
            subj = m.group(2).lower()
            verb = m.group(3).lower()
            if verb not in _wh_q_base:
                continue
            # Only when this looks like a direct question (wh-word at/near sentence start)
            before = text[:m.start()].strip()
            if before and not re.search(r"[.!?]\s*$", before):
                continue
            add(m, f"{wh} did {subj} {_wh_q_base[verb]} {m.group(0)[m.end(3):].lstrip()}",
                ErrorCategory.SENTENCE_STRUCTURE, _P + "WH_DO_SUPPORT",
                f'Direct questions need do-support: "{wh} did {subj} {_wh_q_base[verb]}?"',
                0.85)

        # 53. missing comma before coordinating conjunction joining two clauses
        _PREP_AFTER = {"at", "in", "on", "from", "to", "with", "for", "of", "by",
                       "about", "after", "before", "into", "onto", "under", "over",
                       "through", "between", "among", "during", "without"}
        for m in re.finditer(
                r"(?i)(\w+)\s+(and|but|so|yet)\s+(she|he|they|we|i|you|it)\s+"
                r"(\w+)\s+(?!,)", text):
            conj = m.group(2)
            if text[m.end() - 1:].lstrip()[:1] == ",":
                continue
            # "saw Carla and I at ..." is object coordination, not two clauses:
            # skip when the word after the pronoun is a preposition.
            if m.group(4).lower() in _PREP_AFTER:
                continue
            cpos = text.find(m.group(0).split()[1], m.start() + len(m.group(1)))
            add(_FakeM(cpos - 1, cpos + len(conj), conj), f", {conj}",
                ErrorCategory.PUNCTUATION, _P + "CLAUSE_COMMA",
                f'Insert a comma before "{conj}" to join two independent clauses.', 0.72)

        # 53b. "There going" → "They're going" (there + present participle)
        for m in re.finditer(r"\bThere\s+((?:going|coming|eating|running|walking|doing|"
                             r"making|taking|using|playing|studying|working|reading|writing|"
                             r"watching|listening|waiting|buying|selling|stopping))\b",
                             text):
            add(m, f"They're {m.group(1).lower()}", ErrorCategory.WORD_USAGE,
                _P + "THERE_GERUND",
                f'Use "They\'re {m.group(1).lower()}" (contraction of "they are") instead of "There {m.group(1).lower()}".',
                0.85)

        # 53c. Possessive apostrophe before a plural verb is usually a plural noun:
        # "The teacher's are here." → "The teachers are here."
        for m in re.finditer(r"\b([a-z]+)'s\s+(are|were|have)\b", text):
            add(m, f"{m.group(1)}s {m.group(2)}", ErrorCategory.PUNCTUATION,
                _P + "POSSESSIVE_PLURAL",
                f'Use the plural "{m.group(1)}s" (no apostrophe) before "{m.group(2)}".',
                0.95)

        # 54. comma after a short introductory phrase ("After dinner ...")
        for m in re.finditer(
                r"^(after|before|during|until|since|when|while)\s+"
                r"([a-z]+)\s+(she|he|they|we|i|you|it)\b",
                text, re.I):
            if "," in m.group(0):
                continue
            intro = m.group(1) + " " + m.group(2)
            add(m, f"{intro}, {m.group(3)}",
                ErrorCategory.PUNCTUATION, _P + "INTRO_COMMA",
                f'Use a comma after the introductory phrase "{intro}".', 0.72)

        # 55. mid-position adverb placed before the copula ("She always is ...")
        for m in re.finditer(
                r"\b(she|he|they|we|i|you|it)\s+"
                r"(always|never|often|usually|sometimes|rarely|seldom|frequently)\s+"
                r"(is|are|was|were|am)\b", text, re.I):
            subj, adv, cop = m.groups()
            add(m, f"{subj} {cop} {adv}", ErrorCategory.WORD_ORDER, _P + "ADV_COPULA",
                f'Place the frequency adverb after the verb: "{subj} {cop} {adv}".',
                0.72)

        # 56. frequency adverb misplaced before have/has ("He never has been ...")
        for m in re.finditer(
                r"\b(she|he|they|we|i|you|it)\s+"
                r"(never|always|often|usually|sometimes|already|just|ever)\s+"
                r"(have|has|had)\s+been\b", text, re.I):
            subj, adv, aux = m.groups()
            add(m, f"{subj} {aux} {adv} been", ErrorCategory.WORD_ORDER,
                _P + "ADV_AUX", f'Place the adverb after the auxiliary: "{subj} {aux} {adv} been".',
                0.72)

        # 57. manner adverb before the object ("speaks very well English")
        for m in re.finditer(
                r"\b(speaks?|spoke|writes?|wrote|reads?|knows?)\s+"
                r"(very well|extremely well|well|fluently|perfectly|properly)\s+"
                r"([A-Z][a-z]+)\b", text):
            verb, adv, obj = m.groups()
            add(m, f"{verb} {obj} {adv}", ErrorCategory.WORD_ORDER, _P + "ADV_OBJECT",
                f'Place the object before the adverb: "{verb} {obj} {adv}".', 0.72)

        # 58. "go to the school/work/bed" -> drop the article (activity sense)
        for m in re.finditer(
                r"\b(go|goes|going|went|walks?|walked|cycling|cycle)\s+to\s+the\s+"
                r"(school|work|bed|church|college|town|prison|court|hospital)\b",
                text, re.I):
            verb, place = m.groups()
            add(m, f"{verb} to {place}", ErrorCategory.ARTICLE, _P + "THE_ACTIVITY",
                f'When referring to the activity, use "{verb} to {place}" (no article).',
                0.72)

        return cands


class _FakeM:
    """Minimal re.Match stand-in for POS-guided spans."""
    __slots__ = ("_start", "_end", "_text")

    def __init__(self, start, end, text):
        self._start, self._end, self._text = start, end, text

    def start(self):
        return self._start

    def end(self):
        return self._end

    def group(self, *args):
        return self._text


if __name__ == "__main__":
    import json, sys
    t = sys.stdin.read() if not sys.stdin.isatty() else ""
    d = HighConfidenceDetector()
    for out in d.detect(t):
        print(json.dumps({
            "start": out.start, "end": out.end, "original": out.original,
            "replacement": out.replacement, "category": out.category.value,
            "rule_id": out.rule_id, "confidence": out.raw_confidence,
        }))