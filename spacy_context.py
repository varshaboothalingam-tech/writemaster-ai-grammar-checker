"""
spacy_context.py — spaCy-based context analyzer.
Uses dependency parsing, POS tagging, and NER to extract sentence context.
"""

import spacy
from typing import List, Dict, Optional
from grammar_pipeline import ContextAnalyzer, ContextInfo

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# ── Constants ─────────────────────────────────────────────────────────

SINGULAR_PRONOUNS = {"i", "he", "she", "it", "this", "that", "everyone", "everyone",
                     "each", "every", "nobody", "no one", "somebody", "someone",
                     "anybody", "anyone", "everybody", "every"}

PLURAL_PRONOUNS = {"we", "they", "these", "those"}

BE_FORMS = {"is", "am", "are", "was", "were", "be", "been", "being"}
HAVE_FORMS = {"has", "have", "had"}
DO_FORMS = {"does", "do", "did"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}

COLLECTIVE_NOUNS = {
    "team", "group", "jury", "committee", "family", "class", "crowd",
    "audience", "staff", "faculty", "company", "government", "majority",
    "minority", "number", "series", "species", "data", "media",
}

UNCOUNTABLE_NOUNS = {
    "information", "furniture", "equipment", "advice", "homework",
    "luggage", "baggage", "music", "news", "weather", "traffic",
    "rice", "bread", "water", "milk", "sugar", "salt", "flour",
    "money", "cash", "change", "knowledge", "experience", "education",
    "research", "progress", "evidence", "proof", "feedback",
    "software", "hardware", "electronics", "machinery",
    "research", "data", "evidence", "advice", "feedback",
    "courage", "patience", "freedom", "happiness", "love",
    "anger", "fear", "joy", "sadness", "excitement",
}

IRREGULAR_VERBS = {
    "go": {"past": "went", "pp": "gone", "present_3sg": "goes", "base": "go"},
    "have": {"past": "had", "pp": "had", "present_3sg": "has", "base": "have"},
    "be": {"past_sg": "was", "past_pl": "were", "pp": "been", "present_1sg": "am", "present_3sg": "is", "present_pl": "are", "base": "be"},
    "do": {"past": "did", "pp": "done", "present_3sg": "does", "base": "do"},
    "say": {"past": "said", "pp": "said", "present_3sg": "says", "base": "say"},
    "get": {"past": "got", "pp": "gotten", "present_3sg": "gets", "base": "get"},
    "make": {"past": "made", "pp": "made", "present_3sg": "makes", "base": "make"},
    "know": {"past": "knew", "pp": "known", "present_3sg": "knows", "base": "know"},
    "think": {"past": "thought", "pp": "thought", "present_3sg": "thinks", "base": "think"},
    "take": {"past": "took", "pp": "taken", "present_3sg": "takes", "base": "take"},
    "see": {"past": "saw", "pp": "seen", "present_3sg": "sees", "base": "see"},
    "come": {"past": "came", "pp": "come", "present_3sg": "comes", "base": "come"},
    "want": {"past": "wanted", "pp": "wanted", "present_3sg": "wants", "base": "want"},
    "give": {"past": "gave", "pp": "given", "present_3sg": "gives", "base": "give"},
    "use": {"past": "used", "pp": "used", "present_3sg": "uses", "base": "use"},
    "find": {"past": "found", "pp": "found", "present_3sg": "finds", "base": "find"},
    "tell": {"past": "told", "pp": "told", "present_3sg": "tells", "base": "tell"},
    "ask": {"past": "asked", "pp": "asked", "present_3sg": "asks", "base": "ask"},
    "work": {"past": "worked", "pp": "worked", "present_3sg": "works", "base": "work"},
    "seem": {"past": "seemed", "pp": "seemed", "present_3sg": "seems", "base": "seem"},
    "feel": {"past": "felt", "pp": "felt", "present_3sg": "feels", "base": "feel"},
    "try": {"past": "tried", "pp": "tried", "present_3sg": "tries", "base": "try"},
    "leave": {"past": "left", "pp": "left", "present_3sg": "leaves", "base": "leave"},
    "call": {"past": "called", "pp": "called", "present_3sg": "calls", "base": "call"},
    "need": {"past": "needed", "pp": "needed", "present_3sg": "needs", "base": "need"},
    "become": {"past": "became", "pp": "become", "present_3sg": "becomes", "base": "become"},
    "keep": {"past": "kept", "pp": "kept", "present_3sg": "keeps", "base": "keep"},
    "let": {"past": "let", "pp": "let", "present_3sg": "lets", "base": "let"},
    "begin": {"past": "began", "pp": "begun", "present_3sg": "begins", "base": "begin"},
    "show": {"past": "showed", "pp": "shown", "present_3sg": "shows", "base": "show"},
    "hear": {"past": "heard", "pp": "heard", "present_3sg": "hears", "base": "hear"},
    "play": {"past": "played", "pp": "played", "present_3sg": "plays", "base": "play"},
    "run": {"past": "ran", "pp": "run", "present_3sg": "runs", "base": "run"},
    "move": {"past": "moved", "pp": "moved", "present_3sg": "moves", "base": "move"},
    "live": {"past": "lived", "pp": "lived", "present_3sg": "lives", "base": "live"},
    "believe": {"past": "believed", "pp": "believed", "present_3sg": "believes", "base": "believe"},
    "bring": {"past": "brought", "pp": "brought", "present_3sg": "brings", "base": "bring"},
    "happen": {"past": "happened", "pp": "happened", "present_3sg": "happens", "base": "happen"},
    "write": {"past": "wrote", "pp": "written", "present_3sg": "writes", "base": "write"},
    "sit": {"past": "sat", "pp": "sat", "present_3sg": "sits", "base": "sit"},
    "stand": {"past": "stood", "pp": "stood", "present_3sg": "stands", "base": "stand"},
    "lose": {"past": "lost", "pp": "lost", "present_3sg": "loses", "base": "lose"},
    "pay": {"past": "paid", "pp": "paid", "present_3sg": "pays", "base": "pay"},
    "meet": {"past": "met", "pp": "met", "present_3sg": "meets", "base": "meet"},
    "include": {"past": "included", "pp": "included", "present_3sg": "includes", "base": "include"},
    "continue": {"past": "continued", "pp": "continued", "present_3sg": "continues", "base": "continue"},
    "set": {"past": "set", "pp": "set", "present_3sg": "sets", "base": "set"},
    "learn": {"past": "learned", "pp": "learned", "present_3sg": "learns", "base": "learn"},
    "change": {"past": "changed", "pp": "changed", "present_3sg": "changes", "base": "change"},
    "lead": {"past": "led", "pp": "led", "present_3sg": "leads", "base": "lead"},
    "understand": {"past": "understood", "pp": "understood", "present_3sg": "understands", "base": "understand"},
    "watch": {"past": "watched", "pp": "watched", "present_3sg": "watches", "base": "watch"},
    "follow": {"past": "followed", "pp": "followed", "present_3sg": "follows", "base": "follow"},
    "stop": {"past": "stopped", "pp": "stopped", "present_3sg": "stops", "base": "stop"},
    "create": {"past": "created", "pp": "created", "present_3sg": "creates", "base": "create"},
    "speak": {"past": "spoke", "pp": "spoken", "present_3sg": "speaks", "base": "speak"},
    "read": {"past": "read", "pp": "read", "present_3sg": "reads", "base": "read"},
    "spend": {"past": "spent", "pp": "spent", "present_3sg": "spends", "base": "spend"},
    "grow": {"past": "grew", "pp": "grown", "present_3sg": "grows", "base": "grow"},
    "open": {"past": "opened", "pp": "opened", "present_3sg": "opens", "base": "open"},
    "walk": {"past": "walked", "pp": "walked", "present_3sg": "walks", "base": "walk"},
    "win": {"past": "won", "pp": "won", "present_3sg": "wins", "base": "win"},
    "teach": {"past": "taught", "pp": "taught", "present_3sg": "teaches", "base": "teach"},
    "offer": {"past": "offered", "pp": "offered", "present_3sg": "offers", "base": "offer"},
    "remember": {"past": "remembered", "pp": "remembered", "present_3sg": "remembers", "base": "remember"},
    "love": {"past": "loved", "pp": "loved", "present_3sg": "loves", "base": "love"},
    "consider": {"past": "considered", "pp": "considered", "present_3sg": "considers", "base": "consider"},
    "appear": {"past": "appeared", "pp": "appeared", "present_3sg": "appears", "base": "appear"},
    "buy": {"past": "bought", "pp": "bought", "present_3sg": "buys", "base": "buy"},
    "wait": {"past": "waited", "pp": "waited", "present_3sg": "waits", "base": "wait"},
    "serve": {"past": "served", "pp": "served", "present_3sg": "serves", "base": "serve"},
    "die": {"past": "died", "pp": "died", "present_3sg": "dies", "base": "die"},
    "send": {"past": "sent", "pp": "sent", "present_3sg": "sends", "base": "send"},
    "expect": {"past": "expected", "pp": "expected", "present_3sg": "expects", "base": "expect"},
    "build": {"past": "built", "pp": "built", "present_3sg": "builds", "base": "build"},
    "stay": {"past": "stayed", "pp": "stayed", "present_3sg": "stays", "base": "stay"},
    "fall": {"past": "fell", "pp": "fallen", "present_3sg": "falls", "base": "fall"},
    "cut": {"past": "cut", "pp": "cut", "present_3sg": "cuts", "base": "cut"},
    "reach": {"past": "reached", "pp": "reached", "present_3sg": "reaches", "base": "reach"},
    "kill": {"past": "killed", "pp": "killed", "present_3sg": "kills", "base": "kill"},
    "raise": {"past": "raised", "pp": "raised", "present_3sg": "raises", "base": "raise"},
    "pass": {"past": "passed", "pp": "passed", "present_3sg": "passes", "base": "pass"},
    "sell": {"past": "sold", "pp": "sold", "present_3sg": "sells", "base": "sell"},
    "decide": {"past": "decided", "pp": "decided", "present_3sg": "decides", "base": "decide"},
    "return": {"past": "returned", "pp": "returned", "present_3sg": "returns", "base": "return"},
    "explain": {"past": "explained", "pp": "explained", "present_3sg": "explains", "base": "explain"},
    "hope": {"past": "hoped", "pp": "hoped", "present_3sg": "hopes", "base": "hope"},
    "develop": {"past": "developed", "pp": "developed", "present_3sg": "develops", "base": "develop"},
    "carry": {"past": "carried", "pp": "carried", "present_3sg": "carries", "base": "carry"},
    "break": {"past": "broke", "pp": "broken", "present_3sg": "breaks", "base": "break"},
    "receive": {"past": "received", "pp": "received", "present_3sg": "receives", "base": "receive"},
    "agree": {"past": "agreed", "pp": "agreed", "present_3sg": "agrees", "base": "agree"},
    "support": {"past": "supported", "pp": "supported", "present_3sg": "supports", "base": "support"},
    "hold": {"past": "held", "pp": "held", "present_3sg": "holds", "base": "hold"},
    "produce": {"past": "produced", "pp": "produced", "present_3sg": "produces", "base": "produce"},
    "happen": {"past": "happened", "pp": "happened", "present_3sg": "happens", "base": "happen"},
    "provide": {"past": "provided", "pp": "provided", "present_3sg": "provides", "base": "provide"},
    "enter": {"past": "entered", "pp": "entered", "present_3sg": "enters", "base": "enter"},
    "visit": {"past": "visited", "pp": "visited", "present_3sg": "visits", "base": "visit"},
    "love": {"past": "loved", "pp": "loved", "present_3sg": "loves", "base": "love"},
    "hit": {"past": "hit", "pp": "hit", "present_3sg": "hits", "base": "hit"},
}


# ── spaCy Context Analyzer ───────────────────────────────────────────

class SpaCyContextAnalyzer(ContextAnalyzer):
    """Uses spaCy NLP to analyze sentence context."""

    name = "spacy_context"

    def __init__(self):
        self.nlp = nlp

    def analyze(self, text: str) -> List[ContextInfo]:
        """Analyze all sentences in the text."""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        contexts = []

        for i, sent in enumerate(sentences):
            prev_sent = sentences[i - 1].text if i > 0 else ""
            next_sent = sentences[i + 1].text if i < len(sentences) - 1 else ""
            ctx = self._analyze_sentence(sent, prev_sent, next_sent)
            contexts.append(ctx)

        return contexts

    def analyze_sentence(self, sentence: str, doc=None) -> ContextInfo:
        """Analyze a single sentence."""
        if doc is None:
            doc = self.nlp(sentence)
        sents = list(doc.sents)
        if sents:
            return self._analyze_sentence(sents[0], "", "")
        return ContextInfo(sentence=sentence)

    def _analyze_sentence(self, sent, prev_sent: str, next_sent: str) -> ContextInfo:
        """Extract context from a spaCy sentence span."""
        ctx = ContextInfo(
            sentence=sent.text,
            previous_sentence=prev_sent,
            next_sentence=next_sent,
        )

        tokens = list(sent)

        # Detect sentence type
        if sent.text.strip().endswith("?"):
            ctx.sentence_type = "question"
        elif sent.text.strip().endswith("!"):
            ctx.sentence_type = "exclamation"
        elif any(t.dep_ in ("advcl", "csubj") for t in tokens):
            ctx.sentence_type = "declarative"
        else:
            ctx.sentence_type = "declarative"

        # Find subject and verb
        for token in tokens:
            # Subject
            if token.dep_ in ("nsubj", "nsubjpass"):
                ctx.subject = token.text
                ctx.subject_number = self._get_number(token)
                ctx.subject_person = self._get_person(token)
                ctx.is_proper_noun = token.ent_type_ != ""

            # Verb
            if token.dep_ in ("ROOT", "conj") and token.pos_ in ("VERB", "AUX"):
                if not ctx.verb:
                    ctx.verb = token.text
                    ctx.verb_number = self._get_verb_number(token, ctx.subject_number)

            # Auxiliary verbs
            if token.dep_ == "aux" and token.pos_ == "AUX":
                ctx.auxiliary_verbs.append(token.text.lower())
                if token.text.lower() in MODALS:
                    ctx.modal_verbs.append(token.text.lower())

        # Detect existential "there"
        for token in tokens:
            if token.text.lower() == "there" and token.dep_ in ("expl", "attr"):
                ctx.is_existential_there = True
                break

        # Detect passive
        for token in tokens:
            if token.dep_ == "nsubjpass" or (token.pos_ == "AUX" and token.lemma_ == "be" and any(t.dep_ == "auxpass" for t in token.children)):
                ctx.is_passive = True
                break

        # Detect compound subject
        for token in tokens:
            if token.dep_ == "conj" and token.head.dep_ in ("nsubj", "nsubjpass"):
                ctx.compound_subject = True
                break

        # Detect collective noun
        if ctx.subject.lower() in COLLECTIVE_NOUNS:
            ctx.collective_noun = True

        # Detect uncountable noun
        if ctx.subject.lower() in UNCOUNTABLE_NOUNS:
            ctx.uncountable_noun = True

        # Detect quoted text
        if '"' in sent.text or "'" in sent.text:
            ctx.is_quoted_text = True

        # Detect clause type
        for token in tokens:
            if token.dep_ == "mark":
                ctx.clause_type = "subordinate"
                break
            if token.dep_ == "relcl":
                ctx.clause_type = "relative"
                break
        if not ctx.clause_type:
            ctx.clause_type = "main"

        # Detect tense
        ctx.tense = self._detect_tense(tokens)

        return ctx

    def _get_number(self, token) -> str:
        """Determine if a token is singular or plural."""
        if token.lower_ in SINGULAR_PRONOUNS:
            return "singular"
        if token.lower_ in PLURAL_PRONOUNS:
            return "plural"
        if token.tag_ == "NN" or token.tag_ == "NNP":
            return "singular"
        if token.tag_ in ("NNS", "NNPS"):
            return "plural"
        return "unknown"

    def _get_person(self, token) -> str:
        """Determine the person (1st, 2nd, 3rd)."""
        if token.lower_ in ("i", "we", "me", "us", "my", "our"):
            return "1st"
        if token.lower_ in ("you", "your", "yours"):
            return "2nd"
        return "3rd"

    def _get_verb_number(self, verb_token, subject_number: str) -> str:
        """Determine verb number from form."""
        tag = verb_token.tag_
        if tag in ("VBZ",):
            return "singular"
        if tag in ("VBP", "VBN", "VBG"):
            if subject_number == "singular":
                return "singular"
            return "plural"
        if tag == "VBD":
            if verb_token.lower_ in ("was",):
                return "singular"
            if verb_token.lower_ in ("were",):
                return "plural"
        return "unknown"

    def _detect_tense(self, tokens) -> str:
        """Detect the tense of the sentence."""
        auxiliaries = [t for t in tokens if t.dep_ == "aux" and t.pos_ == "AUX"]
        root = [t for t in tokens if t.dep_ == "ROOT"]

        if not root:
            return "unknown"

        root_verb = root[0]
        aux_texts = [a.lower_ for a in auxiliaries]

        # Present perfect: has/have + past participle
        if any(a in ("has", "have", "had") for a in aux_texts):
            if any(t.tag_ == "VBN" for t in tokens):
                if "had" in aux_texts:
                    return "past_perfect"
                return "present_perfect"

        # Future: will/shall + base
        if any(a in ("will", "shall") for a in aux_texts):
            return "future"

        # Modal: can/could/would/should/may/might/must + base
        if any(a in MODALS for a in aux_texts):
            return "modal"

        # Past tense
        if root_verb.tag_ == "VBD":
            return "past"

        # Present tense
        if root_verb.tag_ in ("VBZ", "VBP", "VB"):
            return "present"

        # Present continuous: is/are/am + VBG
        if any(a in ("is", "am", "are", "was", "were") for a in aux_texts):
            if any(t.tag_ == "VBG" for t in tokens):
                return "present_continuous" if any(a in ("is", "am", "are") for a in aux_texts) else "past_continuous"

        return "unknown"
