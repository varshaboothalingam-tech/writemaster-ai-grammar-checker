import re
from typing import List, Dict, Optional, Tuple
from nlp_engine import (
    analyze_text, get_children, get_dependents, find_subject, find_object,
    get_verb_chain, is_compound_subject, get_compound_subjects,
    NlpDocument, NlpSentence, NlpToken, get_subtree_tokens
)

BE_VERBS = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being'}
HAVE_VERBS = {'has', 'have', 'had'}
DO_VERBS = {'does', 'do', 'did'}
MODALS = {'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}
AUX_VERBS = BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS

SINGULAR_PRP = {'he', 'she', 'it'}
PLURAL_PRP = {'we', 'they'}
ALL_SUBJECT_PRP = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}

PRONOUN_BE = {
    'i': ('am', 'was', 'are', 'were'),
    'you': ('are', 'were', 'is', 'was'),
    'he': ('is', 'was'),
    'she': ('is', 'was'),
    'it': ('is', 'was'),
    'we': ('are', 'were'),
    'they': ('are', 'were'),
}

PRONOUN_HAVE = {
    'i': 'have', 'you': 'have', 'he': 'has', 'she': 'has', 'it': 'has',
    'we': 'have', 'they': 'have',
}

PRONOUN_DO = {
    'i': 'do', 'you': 'do', 'he': 'does', 'she': 'does', 'it': 'does',
    'we': 'do', 'they': 'do',
}

IRREGULAR_VERBS = {
    'go': {'past': 'went', 'past_part': 'gone', 'present_3rd': 'goes', 'present_part': 'going'},
    'eat': {'past': 'ate', 'past_part': 'eaten', 'present_3rd': 'eats', 'present_part': 'eating'},
    'have': {'past': 'had', 'past_part': 'had', 'present_3rd': 'has', 'present_part': 'having'},
    'do': {'past': 'did', 'past_part': 'done', 'present_3rd': 'does', 'present_part': 'doing'},
    'be': {'past': 'was', 'past_part': 'been', 'present_3rd': 'is', 'present_part': 'being'},
    'see': {'past': 'saw', 'past_part': 'seen', 'present_3rd': 'sees', 'present_part': 'seeing'},
    'take': {'past': 'took', 'past_part': 'taken', 'present_3rd': 'takes', 'present_part': 'taking'},
    'give': {'past': 'gave', 'past_part': 'given', 'present_3rd': 'gives', 'present_part': 'giving'},
    'come': {'past': 'came', 'past_part': 'come', 'present_3rd': 'comes', 'present_part': 'coming'},
    'know': {'past': 'knew', 'past_part': 'known', 'present_3rd': 'knows', 'present_part': 'knowing'},
    'think': {'past': 'thought', 'past_part': 'thought', 'present_3rd': 'thinks', 'present_part': 'thinking'},
    'say': {'past': 'said', 'past_part': 'said', 'present_3rd': 'says', 'present_part': 'saying'},
    'get': {'past': 'got', 'past_part': 'gotten', 'present_3rd': 'gets', 'present_part': 'getting'},
    'make': {'past': 'made', 'past_part': 'made', 'present_3rd': 'makes', 'present_part': 'making'},
    'find': {'past': 'found', 'past_part': 'found', 'present_3rd': 'finds', 'present_part': 'finding'},
    'tell': {'past': 'told', 'past_part': 'told', 'present_3rd': 'tells', 'present_part': 'telling'},
    'run': {'past': 'ran', 'past_part': 'run', 'present_3rd': 'runs', 'present_part': 'running'},
    'write': {'past': 'wrote', 'past_part': 'written', 'present_3rd': 'writes', 'present_part': 'writing'},
    'drive': {'past': 'drove', 'past_part': 'driven', 'present_3rd': 'drives', 'present_part': 'driving'},
    'speak': {'past': 'spoke', 'past_part': 'spoken', 'present_3rd': 'speaks', 'present_part': 'speaking'},
    'choose': {'past': 'chose', 'past_part': 'chosen', 'present_3rd': 'chooses', 'present_part': 'choosing'},
    'break': {'past': 'broke', 'past_part': 'broken', 'present_3rd': 'breaks', 'present_part': 'breaking'},
    'wake': {'past': 'woke', 'past_part': 'woken', 'present_3rd': 'wakes', 'present_part': 'waking'},
    'sing': {'past': 'sang', 'past_part': 'sung', 'present_3rd': 'sings', 'present_part': 'singing'},
    'swim': {'past': 'swam', 'past_part': 'swum', 'present_3rd': 'swims', 'present_part': 'swimming'},
    'begin': {'past': 'began', 'past_part': 'begun', 'present_3rd': 'begins', 'present_part': 'beginning'},
    'bring': {'past': 'brought', 'past_part': 'brought', 'present_3rd': 'brings', 'present_part': 'bringing'},
    'buy': {'past': 'bought', 'past_part': 'bought', 'present_3rd': 'buys', 'present_part': 'buying'},
    'catch': {'past': 'caught', 'past_part': 'caught', 'present_3rd': 'catches', 'present_part': 'catching'},
    'feel': {'past': 'felt', 'past_part': 'felt', 'present_3rd': 'feels', 'present_part': 'feeling'},
    'keep': {'past': 'kept', 'past_part': 'kept', 'present_3rd': 'keeps', 'present_part': 'keeping'},
    'leave': {'past': 'left', 'past_part': 'left', 'present_3rd': 'leaves', 'present_part': 'leaving'},
    'lose': {'past': 'lost', 'past_part': 'lost', 'present_3rd': 'loses', 'present_part': 'losing'},
    'meet': {'past': 'met', 'past_part': 'met', 'present_3rd': 'meets', 'present_part': 'meeting'},
    'pay': {'past': 'paid', 'past_part': 'paid', 'present_3rd': 'pays', 'present_part': 'paying'},
    'read': {'past': 'read', 'past_part': 'read', 'present_3rd': 'reads', 'present_part': 'reading'},
    'sit': {'past': 'sat', 'past_part': 'sat', 'present_3rd': 'sits', 'present_part': 'sitting'},
    'stand': {'past': 'stood', 'past_part': 'stood', 'present_3rd': 'stands', 'present_part': 'standing'},
    'teach': {'past': 'taught', 'past_part': 'taught', 'present_3rd': 'teaches', 'present_part': 'teaching'},
    'understand': {'past': 'understood', 'past_part': 'understood', 'present_3rd': 'understands', 'present_part': 'understanding'},
    'win': {'past': 'won', 'past_part': 'won', 'present_3rd': 'wins', 'present_part': 'winning'},
    'fly': {'past': 'flew', 'past_part': 'flown', 'present_3rd': 'flies', 'present_part': 'flying'},
    'grow': {'past': 'grew', 'past_part': 'grown', 'present_3rd': 'grows', 'present_part': 'growing'},
    'show': {'past': 'showed', 'past_part': 'shown', 'present_3rd': 'shows', 'present_part': 'showing'},
    'throw': {'past': 'threw', 'past_part': 'thrown', 'present_3rd': 'throws', 'present_part': 'throwing'},
    'wear': {'past': 'wore', 'past_part': 'worn', 'present_3rd': 'wears', 'present_part': 'wearing'},
    'fall': {'past': 'fell', 'past_part': 'fallen', 'present_3rd': 'falls', 'present_part': 'falling'},
    'hang': {'past': 'hung', 'past_part': 'hung', 'present_3rd': 'hangs', 'present_part': 'hanging'},
    'lead': {'past': 'led', 'past_part': 'led', 'present_3rd': 'leads', 'present_part': 'leading'},
    'lie': {'past': 'lay', 'past_part': 'lain', 'present_3rd': 'lies', 'present_part': 'lying'},
    'rise': {'past': 'rose', 'past_part': 'risen', 'present_3rd': 'rises', 'present_part': 'rising'},
    'shake': {'past': 'shook', 'past_part': 'shaken', 'present_3rd': 'shakes', 'present_part': 'shaking'},
}

PAST_TO_BASE = {}
for base, forms in IRREGULAR_VERBS.items():
    PAST_TO_BASE[forms['past']] = base
    PAST_TO_BASE[forms['past_part']] = base
PAST_TO_BASE['went'] = 'go'
PAST_TO_BASE['ate'] = 'eat'
PAST_TO_BASE['saw'] = 'see'
PAST_TO_BASE['ran'] = 'run'
PAST_TO_BASE['wrote'] = 'write'
PAST_TO_BASE['drove'] = 'drive'
PAST_TO_BASE['spoke'] = 'speak'
PAST_TO_BASE['chose'] = 'choose'
PAST_TO_BASE['broke'] = 'break'
PAST_TO_BASE['woke'] = 'wake'
PAST_TO_BASE['sang'] = 'sing'
PAST_TO_BASE['swam'] = 'swim'
PAST_TO_BASE['began'] = 'begin'
PAST_TO_BASE['flew'] = 'fly'
PAST_TO_BASE['grew'] = 'grow'
PAST_TO_BASE['knew'] = 'know'
PAST_TO_BASE['threw'] = 'throw'
PAST_TO_BASE['wore'] = 'wear'
PAST_TO_BASE['fell'] = 'fall'
PAST_TO_BASE['hung'] = 'hang'
PAST_TO_BASE['led'] = 'lead'
PAST_TO_BASE['lay'] = 'lie'
PAST_TO_BASE['rose'] = 'rise'
PAST_TO_BASE['shook'] = 'shake'
PAST_TO_BASE['took'] = 'take'

MALE_NOUNS = {'man', 'boy', 'guy', 'male', 'father', 'brother', 'son', 'husband',
              'uncle', 'grandfather', 'grandpa', 'nephew', 'prince', 'king',
              'lord', 'sir', 'gentleman', 'daddy', 'papa', 'dad'}
FEMALE_NOUNS = {'woman', 'girl', 'lady', 'female', 'mother', 'sister', 'daughter',
                'wife', 'aunt', 'grandmother', 'grandma', 'niece', 'princess',
                'queen', 'madam', 'mommy', 'mama', 'mom'}

NEGATION_WORDS = {'no', 'not', 'never', 'neither', 'nobody', 'nothing',
                  'nowhere', 'nor', "n't", 'hardly', 'barely', 'scarcely',
                  'seldom', 'rarely'}

TEMPORAL_MARKERS_PAST = {'yesterday', 'ago', 'last', 'previously', 'before', 'earlier',
                         'once', 'formerly', 'prior'}
TEMPORAL_MARKERS_PRESENT = {'now', 'today', 'currently', 'presently', 'usually',
                            'always', 'often', 'sometimes', 'generally', 'normally'}
TEMPORAL_MARKERS_FUTURE = {'tomorrow', 'soon', 'later', 'eventually', 'upcoming'}

COMPARATIVE_WORDS = {'bigger', 'smaller', 'taller', 'shorter', 'better', 'worse',
                     'faster', 'slower', 'higher', 'lower', 'longer', 'shorter',
                     'older', 'younger', 'easier', 'harder', 'stronger', 'weaker',
                     'more', 'less', 'rather', 'further', 'farther'}

CONFUSING_PAIRS_CONTEXT = {
    'your': ("you're", lambda t: t.dep == 'poss'),
    "you're": ('your', lambda t: t.dep != 'poss' and t.pos == 'AUX'),
    'their': ('there', lambda t: t.dep == 'poss'),
    'there': ('their', lambda t: t.dep == 'poss'),
    "they're": ('their', lambda t: t.dep == 'poss'),
    'its': ("it's", lambda t: t.dep == 'poss'),
    "it's": ('its', lambda t: t.dep == 'poss'),
}

VOWEL_SOUNDS_A = set('aeiou')
SILENT_H_WORDS = {'hour', 'honest', 'honor', 'honour', 'heir', 'herb'}
Y_SOUND_U_WORDS = {'university', 'uniform', 'unit', 'united', 'universal', 'unique',
                    'universe', 'usage', 'used', 'useful', 'user', 'usual', 'usually',
                    'utter', 'urban', 'urn', 'umpire', 'union', 'universal'}

def _make_error(sentence_text, incorrect, correction, position, end_position,
                category, message, severity='error', confidence=95, error_type='grammar'):
    return {
        'type': error_type,
        'severity': severity,
        'category': category,
        'incorrect': incorrect,
        'correction': correction,
        'position': position,
        'end_position': end_position,
        'sentence': sentence_text,
        'message': message,
        'confidence': confidence,
        'pass': 'intelligent',
    }


def _is_plural_noun(token: NlpToken) -> bool:
    if token.tag == 'NNS' or token.tag == 'NNPS':
        return True
    if token.pos == 'NOUN' or token.pos == 'PROPN':
        lower = token.lower
        if lower.endswith('s') and not lower.endswith(('ss', 'us', 'is')) and len(lower) > 3:
            if lower.endswith('es') and lower[:-2] in _KNOWN_SINGULAR:
                return False
            return True
    return False

_KNOWN_SINGULAR = {'class', 'bus', 'glass', 'grass', 'dress', 'boss', 'miss',
                    'kiss', 'loss', 'cross', 'toss', 'mass', 'pass', 'bass',
                    'address', 'analysis', 'basis', 'crisis', 'diagnosis', 'emphasis',
                    'gas', 'genius', 'lens', 'minus', 'plus', 'status', 'campus',
                    'focus', 'bonus', 'census', 'circus', 'nexus', 'purse', 'nurse',
                    'verse', 'curve', 'nerve', 'serve', 'reserve', 'observe', 'deserve'}

_UNCOUNTABLE_NOUNS = {'information', 'advice', 'furniture', 'luggage', 'baggage',
                       'equipment', 'music', 'news', 'knowledge', 'research', 'evidence',
                       'feedback', 'software', 'hardware', 'progress', 'traffic',
                       'weather', 'water', 'rice', 'bread', 'milk', 'sugar', 'salt',
                       'gold', 'silver', 'iron', 'cotton', 'wood', 'paper', 'plastic',
                       'electricity', 'energy', 'power', 'money', 'work', 'homework',
                       'housework', 'travel', 'fun', 'happiness', 'sadness', 'anger',
                       'love', 'hate', 'fear', 'courage', 'patience', 'experience',
                       'education', 'health', 'wealth', 'poverty', 'freedom', 'justice',
                       'truth', 'history', 'science', 'mathematics', 'physics', 'chemistry',
                       'biology', 'music', 'art', 'literature', 'philosophy', 'religion',
                       'language', 'english', 'spanish', 'french', 'chinese', 'japanese'}


class IntelligentGrammarEngine:
    def analyze(self, text: str) -> List[Dict]:
        if not text or not text.strip():
            return []
        doc = analyze_text(text)
        all_errors = []
        for sent in doc.sentences:
            errors = []
            errors.extend(self._check_subject_verb_agreement(sent))
            errors.extend(self._check_verb_tense(sent))
            errors.extend(self._check_pronoun_case(sent))
            errors.extend(self._check_articles(sent))
            errors.extend(self._check_semantic_consistency(sent))
            errors.extend(self._check_double_negatives(sent))
            errors.extend(self._check_comparatives(sent))
            errors.extend(self._check_confusing_words(sent))
            errors.extend(self._check_capitalization(sent))
            errors.extend(self._check_punctuation(sent))
            errors.extend(self._check_word_form(sent))
            errors.extend(self._check_sentence_completeness(sent))
            all_errors.extend(errors)
        return all_errors

    def _check_subject_verb_agreement(self, sent: NlpSentence) -> List[Dict]:
        errors = []

        for token in sent.tokens:
            if token.pos not in ('VERB', 'AUX'):
                continue
            if token.dep not in ('ROOT', 'conj', 'ccomp', 'xcomp', 'advcl', 'relcl', 'aux', 'auxpass'):
                continue

            subject = find_subject(sent, token)
            if subject is None:
                continue
            if subject.pos not in ('PRON', 'NOUN', 'PROPN'):
                continue

            subj_lower = subject.lower
            verb_lower = token.lower

            if verb_lower in BE_VERBS and verb_lower not in ('be', 'been', 'being'):
                expected = self._expected_be_form(subj_lower, verb_lower, sent)
                if expected and expected != verb_lower:
                    errors.append(_make_error(
                        sent.text, token.text, expected, token.idx, token.idx_end,
                        'Subject-Verb Agreement',
                        f'The subject "{subject.text}" requires "{expected}", not "{verb_lower}".',
                        confidence=95
                    ))
                elif subject.pos in ('NOUN', 'PROPN'):
                    is_pl = _is_plural_noun(subject)
                    if is_pl and verb_lower in ('is', 'was'):
                        expected_be = 'are' if verb_lower == 'is' else 'were'
                        errors.append(_make_error(
                            sent.text, token.text, expected_be, token.idx, token.idx_end,
                            'Subject-Verb Agreement',
                            f'The subject "{subject.text}" is plural and requires "{expected_be}", not "{verb_lower}".',
                            confidence=80
                        ))
                    elif not is_pl and verb_lower in ('are', 'were'):
                        expected_be = 'is' if verb_lower == 'are' else 'was'
                        errors.append(_make_error(
                            sent.text, token.text, expected_be, token.idx, token.idx_end,
                            'Subject-Verb Agreement',
                            f'The subject "{subject.text}" is singular and requires "{expected_be}", not "{verb_lower}".',
                            confidence=80
                        ))

            elif verb_lower in HAVE_VERBS:
                expected = PRONOUN_HAVE.get(subj_lower)
                if expected and expected != verb_lower:
                    errors.append(_make_error(
                        sent.text, token.text, expected, token.idx, token.idx_end,
                        'Subject-Verb Agreement',
                        f'The subject "{subject.text}" requires "{expected}", not "{verb_lower}".',
                        confidence=95
                    ))
                elif subject.pos in ('NOUN', 'PROPN') and not _is_plural_noun(subject):
                    if verb_lower == 'have':
                        errors.append(_make_error(
                            sent.text, token.text, 'has', token.idx, token.idx_end,
                            'Subject-Verb Agreement',
                            f'The subject "{subject.text}" is singular and requires "has", not "have".',
                            confidence=85
                        ))
                elif subject.pos in ('NOUN', 'PROPN') and _is_plural_noun(subject):
                    if verb_lower == 'has':
                        errors.append(_make_error(
                            sent.text, token.text, 'have', token.idx, token.idx_end,
                            'Subject-Verb Agreement',
                            f'The subject "{subject.text}" is plural and requires "have", not "has".',
                            confidence=85
                        ))

            elif verb_lower in DO_VERBS and token.dep not in ('aux', 'auxpass'):
                expected = PRONOUN_DO.get(subj_lower)
                if expected and expected != verb_lower:
                    errors.append(_make_error(
                        sent.text, token.text, expected, token.idx, token.idx_end,
                        'Subject-Verb Agreement',
                        f'The subject "{subject.text}" requires "{expected}", not "{verb_lower}".',
                        confidence=95
                    ))
                elif subject.pos in ('NOUN', 'PROPN') and not _is_plural_noun(subject):
                    if verb_lower == 'do':
                        errors.append(_make_error(
                            sent.text, token.text, 'does', token.idx, token.idx_end,
                            'Subject-Verb Agreement',
                            f'The subject "{subject.text}" is singular and requires "does", not "do".',
                            confidence=85
                        ))
                elif subject.pos in ('NOUN', 'PROPN') and _is_plural_noun(subject):
                    if verb_lower == 'does':
                        errors.append(_make_error(
                            sent.text, token.text, 'do', token.idx, token.idx_end,
                            'Subject-Verb Agreement',
                            f'The subject "{subject.text}" is plural and requires "do", not "does".',
                            confidence=85
                        ))

            elif verb_lower in ('do', 'does') and token.dep in ('aux', 'auxpass'):
                expected_do = PRONOUN_DO.get(subj_lower)
                if expected_do and expected_do != verb_lower:
                    has_neg = False
                    token_idx = sent.tokens.index(token) if token in sent.tokens else -1
                    if token_idx >= 0 and token_idx + 1 < len(sent.tokens):
                        next_t = sent.tokens[token_idx + 1]
                        if next_t.dep == 'neg' or next_t.lower in ("n't", "nt", "not"):
                            has_neg = True
                    if has_neg:
                        if expected_do == 'does' and verb_lower == 'do':
                            errors.append(_make_error(
                                sent.text, token.text, "doesn't", token.idx, token.idx_end,
                                'Subject-Verb Agreement',
                                f'The subject "{subject.text}" is third person singular. Use "doesn\'t", not "don\'t".',
                                confidence=95
                            ))
                        elif expected_do == 'do' and verb_lower == 'does':
                            errors.append(_make_error(
                                sent.text, token.text, "don't", token.idx, token.idx_end,
                                'Subject-Verb Agreement',
                                f'The subject "{subject.text}" requires "don\'t", not "doesn\'t".',
                                confidence=95
                            ))
                    else:
                        errors.append(_make_error(
                            sent.text, token.text, expected_do, token.idx, token.idx_end,
                            'Subject-Verb Agreement',
                            f'The subject "{subject.text}" requires "{expected_do}", not "{verb_lower}".',
                            confidence=95
                        ))

            elif verb_lower in MODALS and token.dep in ('aux', 'auxpass'):
                head_token = None
                for t in sent.tokens:
                    if t.idx == token.head_idx:
                        head_token = t
                        break
                if head_token and head_token.pos == 'VERB':
                    vb_lower = head_token.lower
                    base = vb_lower
                    if vb_lower.endswith('es') and len(vb_lower) > 3:
                        base = vb_lower[:-2]
                    elif vb_lower.endswith('s') and len(vb_lower) > 2:
                        base = vb_lower[:-1]
                    if base and len(base) > 1 and base not in BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS:
                        errors.append(_make_error(
                            sent.text, head_token.text, base, head_token.idx, head_token.idx_end,
                            'Subject-Verb Agreement',
                            f'After "{verb_lower}", use the base form "{base}", not "{vb_lower}".',
                            confidence=90
                        ))

            elif verb_lower not in AUX_VERBS and token.pos == 'VERB' and token.dep in ('ROOT', 'conj', 'ccomp', 'xcomp', 'advcl', 'relcl'):
                if token.tag in ('VBG', 'VBN'):
                    continue
                has_aux_for_conj = False
                for child in get_children(sent, token.idx):
                    if child.pos == 'AUX' and child.dep in ('aux', 'auxpass'):
                        has_aux_for_conj = True
                        break
                if has_aux_for_conj:
                    continue
                if subj_lower in SINGULAR_PRP:
                    if not verb_lower.endswith(('s', 'x', 'z', 'h')) and verb_lower not in ('am', 'is', 'was', 'has', 'does'):
                        base = verb_lower
                        if verb_lower.endswith('ies'):
                            base = verb_lower[:-3] + 'y'
                        elif verb_lower.endswith('es') and len(verb_lower) > 3:
                            base = verb_lower[:-2]
                        elif verb_lower.endswith('s') and len(verb_lower) > 2:
                            base = verb_lower[:-1]
                        expected = self._conjugate_3rd_singular(base)
                        if expected != verb_lower and base not in BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS:
                            errors.append(_make_error(
                                sent.text, token.text, expected, token.idx, token.idx_end,
                                'Subject-Verb Agreement',
                                f'The subject "{subject.text}" is third-person singular and requires "{expected}".',
                                confidence=90
                            ))
                elif subj_lower in PLURAL_PRP:
                    if verb_lower.endswith(('s', 'x', 'z', 'h')) and not verb_lower.endswith(('ss', 'us', 'is')):
                        base = verb_lower
                        if verb_lower.endswith('es') and len(verb_lower) > 3:
                            base = verb_lower[:-2]
                        elif verb_lower.endswith('s'):
                            base = verb_lower[:-1]
                        if base and len(base) > 1:
                            errors.append(_make_error(
                                sent.text, token.text, base, token.idx, token.idx_end,
                                'Subject-Verb Agreement',
                                f'The subject "{subject.text}" is plural and requires the base form "{base}".',
                                confidence=85
                            ))
                elif subject.pos in ('NOUN', 'PROPN'):
                    is_pl = _is_plural_noun(subject)
                    if is_pl and verb_lower.endswith(('s', 'x', 'z', 'h')) and not verb_lower.endswith(('ss', 'us', 'is')):
                        base = verb_lower
                        if verb_lower.endswith('es') and len(verb_lower) > 3:
                            base = verb_lower[:-2]
                        elif verb_lower.endswith('s'):
                            base = verb_lower[:-1]
                        if base and len(base) > 1:
                            errors.append(_make_error(
                                sent.text, token.text, base, token.idx, token.idx_end,
                                'Subject-Verb Agreement',
                                f'The subject "{subject.text}" is plural and requires the base form "{base}".',
                                confidence=80
                            ))
                    elif not is_pl and not verb_lower.endswith(('s', 'x', 'z', 'h')):
                        expected = self._conjugate_3rd_singular(verb_lower)
                        if expected != verb_lower and verb_lower not in BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS:
                            errors.append(_make_error(
                                sent.text, token.text, expected, token.idx, token.idx_end,
                                'Subject-Verb Agreement',
                                f'The subject "{subject.text}" is singular and requires "{expected}".',
                                confidence=80
                            ))

        for i, token in enumerate(sent.tokens):
            if token.lower in ('is', 'was') and token.pos == 'AUX' and token.dep in ('aux', 'auxpass'):
                pass
            if token.lower in ('there',) and token.pos == 'ADV':
                children = get_children(sent, token.idx)
                be_child = None
                for child in children:
                    if child.lower in BE_VERBS and child.pos == 'AUX':
                        be_child = child
                        break
                if be_child:
                    all_deps = get_dependents(sent, token.idx, recursive=True)
                    for dt in all_deps:
                        if dt.pos in ('NOUN', 'PROPN') and dt.dep in ('attr', 'nsubj'):
                            is_pl = _is_plural_noun(dt)
                            be_lower = be_child.lower
                            if is_pl and be_lower in ('is', 'was'):
                                expected = 'are' if be_lower == 'is' else 'were'
                                errors.append(_make_error(
                                    sent.text, be_child.text, expected, be_child.idx, be_child.idx_end,
                                    'Subject-Verb Agreement',
                                    f'After "there" with a plural noun, use "{expected}", not "{be_lower}".',
                                    confidence=80
                                ))
                            elif not is_pl and be_lower in ('are', 'were'):
                                expected = 'is' if be_lower == 'are' else 'was'
                                errors.append(_make_error(
                                    sent.text, be_child.text, expected, be_child.idx, be_child.idx_end,
                                    'Subject-Verb Agreement',
                                    f'After "there" with a singular noun, use "{expected}", not "{be_lower}".',
                                    confidence=80
                                ))
                            break

        return errors

    def _expected_be_form(self, subj_lower, current_form, sent):
        if subj_lower == 'i':
            if current_form == 'are':
                return 'am'
            elif current_form == 'is':
                return 'am'
            elif current_form == 'were':
                return 'was'
        elif subj_lower in SINGULAR_PRP:
            if current_form == 'are':
                return 'is'
            elif current_form == 'were':
                return 'was'
            elif current_form == 'am':
                return 'is'
        elif subj_lower in PLURAL_PRP:
            if current_form == 'is':
                return 'are'
            elif current_form == 'was':
                return 'were'
            elif current_form == 'am':
                return 'are'
        elif subj_lower == 'you':
            if current_form == 'is':
                return 'are'
            elif current_form == 'was':
                return 'were'
            elif current_form == 'am':
                return 'are'
        return None

    def _conjugate_3rd_singular(self, base):
        if not base or len(base) < 2:
            return base
        for b, forms in IRREGULAR_VERBS.items():
            if b == base:
                return forms.get('present_3rd', base + 's')
        if base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
            return base + 'es'
        if base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
            return base[:-1] + 'ies'
        return base + 's'

    def _check_verb_tense(self, sent: NlpSentence) -> List[Dict]:
        errors = []

        for token in sent.tokens:
            if token.lower in ('did',) and token.dep in ('aux',):
                head_token = None
                for t in sent.tokens:
                    if t.idx == token.head_idx:
                        head_token = t
                        break
                if head_token and head_token.pos == 'VERB':
                    verb_text = head_token.lower
                    if verb_text in PAST_TO_BASE:
                        base = PAST_TO_BASE[verb_text]
                        errors.append(_make_error(
                            sent.text, head_token.text, base, head_token.idx, head_token.idx_end,
                            'Verb Tense',
                            f'After "did", use the base form "{base}", not "{verb_text}".',
                            confidence=95
                        ))
                    elif verb_text.endswith('ed') and len(verb_text) > 3:
                        base = verb_text
                        if verb_text.endswith('ied'):
                            base = verb_text[:-3] + 'y'
                        elif verb_text.endswith('ed') and len(verb_text) > 4:
                            base = verb_text[:-2]
                        else:
                            base = verb_text[:-1]
                        if base and len(base) > 1:
                            errors.append(_make_error(
                                sent.text, head_token.text, base, head_token.idx, head_token.idx_end,
                                'Verb Tense',
                                f'After "did", use the base form "{base}", not "{verb_text}".',
                                confidence=90
                            ))

            if token.lower in ("didn't", "didnt"):
                head_token = None
                for t in sent.tokens:
                    if t.idx == token.head_idx:
                        head_token = t
                        break
                if head_token and head_token.pos == 'VERB':
                    verb_text = head_token.lower
                    if verb_text in PAST_TO_BASE:
                        base = PAST_TO_BASE[verb_text]
                        errors.append(_make_error(
                            sent.text, head_token.text, base, head_token.idx, head_token.idx_end,
                            'Verb Tense',
                            f'After "didn\'t", use the base form "{base}", not "{verb_text}".',
                            confidence=95
                        ))
                    elif verb_text.endswith('ed') and len(verb_text) > 3:
                        base = verb_text
                        if verb_text.endswith('ied'):
                            base = verb_text[:-3] + 'y'
                        elif verb_text.endswith('ed') and len(verb_text) > 4:
                            base = verb_text[:-2]
                        else:
                            base = verb_text[:-1]
                        if base and len(base) > 1:
                            errors.append(_make_error(
                                sent.text, head_token.text, base, head_token.idx, head_token.idx_end,
                                'Verb Tense',
                                f'After "didn\'t", use the base form "{base}", not "{verb_text}".',
                                confidence=90
                            ))

            if token.dep == 'ROOT' and token.pos in ('VERB', 'AUX'):
                chain = get_verb_chain(sent, token)
                has_temporal_past = False
                has_temporal_present = False
                for t in sent.tokens:
                    if t.lower in TEMPORAL_MARKERS_PAST:
                        has_temporal_past = True
                    elif t.lower in TEMPORAL_MARKERS_PRESENT:
                        has_temporal_present = True

                if has_temporal_past and token.pos == 'VERB':
                    if token.tag in ('VBP', 'VBZ') or (token.tag == 'VB'):
                        if token.lower in ('go', 'do', 'like', 'eat', 'play', 'run', 'walk',
                                           'talk', 'work', 'live', 'want', 'need', 'come',
                                           'take', 'give', 'make', 'know', 'think', 'say',
                                           'see', 'look', 'feel', 'try', 'ask', 'use', 'find',
                                           'tell', 'call', 'let', 'put', 'keep', 'seem', 'help',
                                           'show', 'hear', 'turn', 'start', 'stop', 'move'):
                            past = self._get_past_form(token.lower)
                            if past and past != token.lower:
                                errors.append(_make_error(
                                    sent.text, token.text, past, token.idx, token.idx_end,
                                    'Verb Tense',
                                    f'The time expression suggests past tense. Use "{past}" instead of "{token.lower}".',
                                    confidence=85, severity='warning'
                                ))

        for token in sent.tokens:
            if token.lower in ('has', 'have', 'had') and token.dep in ('aux', 'auxpass'):
                head_token = None
                for t in sent.tokens:
                    if t.idx == token.head_idx:
                        head_token = t
                        break
                if head_token and head_token.pos in ('VERB', 'AUX'):
                    vc = head_token
                    if vc.lower.endswith('ing'):
                        continue
                    if vc.lower in ALL_PAST_FORMS and vc.lower not in ALL_PAST_PART_FORMS:
                        base = PAST_TO_BASE.get(vc.lower, vc.lower)
                        expected = IRREGULAR_VERBS.get(base, {}).get('past_part', base + 'ed')
                        if expected != vc.lower:
                            errors.append(_make_error(
                                sent.text, vc.text, expected, vc.idx, vc.idx_end,
                                'Verb Tense',
                                f'After "{token.lower}", use the past participle "{expected}", not the simple past "{vc.lower}".',
                                confidence=90
                            ))

        return errors

    def _get_past_form(self, base):
        if base in IRREGULAR_VERBS:
            return IRREGULAR_VERBS[base].get('past')
        if base.endswith('e'):
            return base + 'd'
        if base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
            return base[:-1] + 'ied'
        if base.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return base + 'ed'
        return base + 'ed'

    def _check_pronoun_case(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        for i, token in enumerate(sent.tokens):
            if not token.is_alpha:
                continue

            if token.lower == 'i' and token.pos == 'PRON' and token.tag != 'NNP':
                if token.idx > 0 and token.idx < len(sent.text):
                    prev_char_idx = token.idx - 1
                    if 0 <= prev_char_idx < len(sent.text) and sent.text[prev_char_idx] == ' ':
                        if token.dep not in ('nsubj', 'nsubjpass', 'dobj', 'pobj', 'conj'):
                            pass
                        else:
                            if i > 0 and sent.tokens[i - 1].lower not in ('.', '!', '?', ',', ';', ':', '"', "'") and sent.tokens[i - 1].is_alpha:
                                pass
                            elif i == 0:
                                errors.append(_make_error(
                                    sent.text, token.text, 'I', token.idx, token.idx_end,
                                    'Capitalization',
                                    'The pronoun "I" should always be capitalized.',
                                    confidence=99
                                ))

            if token.lower in ('me', 'my', 'mine', 'myself') and token.pos == 'PRON':
                if token.dep in ('nsubj', 'nsubjpass'):
                    if i > 0 and sent.tokens[i - 1].lower in ('and', 'or', ','):
                        errors.append(_make_error(
                            sent.text, token.text,
                            'I' if token.lower == 'me' else token.text.replace('me', 'I').replace('my', 'my').replace('mine', 'mine').replace('myself', 'myself'),
                            token.idx, token.idx_end,
                            'Pronoun Case',
                            f'Use "I" (not "{token.text}") as a subject.',
                            confidence=90, severity='warning'
                        ))
                    elif i == 0:
                        replacement = 'I' if token.lower == 'me' else token.text
                        errors.append(_make_error(
                            sent.text, token.text, replacement, token.idx, token.idx_end,
                            'Pronoun Case',
                            f'Use "I" (not "{token.text}") as a subject.',
                            confidence=90
                        ))

            if token.lower in ('him', 'her', 'them') and token.pos == 'PRON':
                if token.dep in ('nsubj', 'nsubjpass'):
                    replacement_map = {'him': 'he', 'her': 'she', 'them': 'they'}
                    replacement = replacement_map.get(token.lower, token.text)
                    errors.append(_make_error(
                        sent.text, token.text, replacement, token.idx, token.idx_end,
                        'Pronoun Case',
                        f'Use "{replacement}" (not "{token.text}") as a subject.',
                        confidence=90
                    ))

            if token.lower in ('who',) and token.dep in ('dobj', 'pobj'):
                errors.append(_make_error(
                    sent.text, token.text, 'whom', token.idx, token.idx_end,
                    'Pronoun Case',
                    'Use "whom" as an object pronoun.',
                    confidence=75, severity='warning'
                ))

            if token.lower == 'who' and token.dep in ('nsubj', 'nsubjpass'):
                pass

        return errors

    def _check_articles(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        for i, token in enumerate(sent.tokens):
            if token.lower not in ('a', 'an'):
                continue
            next_word = None
            for j in range(i + 1, min(i + 4, len(sent.tokens))):
                if sent.tokens[j].is_alpha:
                    next_word = sent.tokens[j]
                    break
            if next_word is None:
                continue

            nw_lower = next_word.lower
            starts_vowel_sound = False
            starts_consonant_sound = True

            if nw_lower[0] in VOWEL_SOUNDS_A:
                starts_vowel_sound = True
                starts_consonant_sound = False
                if nw_lower in SILENT_H_WORDS:
                    starts_vowel_sound = True
                    starts_consonant_sound = False
                elif nw_lower.startswith('u') and nw_lower in Y_SOUND_U_WORDS:
                    starts_vowel_sound = False
                    starts_consonant_sound = True
                elif nw_lower.startswith('eu') or nw_lower.startswith('ew') or nw_lower.startswith('onc'):
                    starts_vowel_sound = False
                    starts_consonant_sound = True
            elif nw_lower in SILENT_H_WORDS:
                starts_vowel_sound = True
                starts_consonant_sound = False
            elif nw_lower.startswith('u') and nw_lower in Y_SOUND_U_WORDS:
                starts_vowel_sound = False
                starts_consonant_sound = True

            if token.lower == 'a' and starts_vowel_sound:
                errors.append(_make_error(
                    sent.text, token.text, 'an', token.idx, token.idx_end,
                    'Article Usage',
                    f'Use "an" before vowel sounds. "{next_word.text}" starts with a vowel sound.',
                    confidence=95
                ))
            elif token.lower == 'an' and starts_consonant_sound:
                errors.append(_make_error(
                    sent.text, token.text, 'a', token.idx, token.idx_end,
                    'Article Usage',
                    f'Use "a" before consonant sounds. "{next_word.text}" starts with a consonant sound.',
                    confidence=95
                ))

        return errors

    def _check_semantic_consistency(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        for token in sent.tokens:
            if token.lower in BE_VERBS and token.lower not in ('be', 'been', 'being'):
                subject = find_subject(sent, token)
                if not subject or subject.pos != 'PRON' or subject.lower not in ('he', 'she', 'it'):
                    continue

                children = get_children(sent, token.idx)
                noun_children = [c for c in children if c.pos in ('NOUN', 'PROPN') and c.dep in ('attr', 'dobj', 'oprd')]
                if not noun_children:
                    all_deps = get_dependents(sent, token.idx, recursive=True)
                    for dt in all_deps:
                        if dt.pos in ('NOUN', 'PROPN') and dt.dep in ('attr', 'dobj', 'oprd', 'pobj'):
                            noun_children.append(dt)
                            break

                for noun_token in noun_children:
                    noun_lower = noun_token.lower
                    if subject.lower == 'he' and noun_lower in FEMALE_NOUNS:
                        errors.append(_make_error(
                            sent.text, f'{subject.text} {token.text} {noun_token.text}',
                            f'she {token.text} {noun_token.text}',
                            subject.idx, noun_token.idx_end,
                            'Semantic Consistency',
                            f'The pronoun "{subject.text}" typically refers to a male, but "{noun_lower}" refers to a female. Consider using "she".',
                            confidence=85, severity='warning', error_type='semantic'
                        ))
                    if subject.lower == 'she' and noun_lower in MALE_NOUNS:
                        errors.append(_make_error(
                            sent.text, f'{subject.text} {token.text} {noun_token.text}',
                            f'he {token.text} {noun_token.text}',
                            subject.idx, noun_token.idx_end,
                            'Semantic Consistency',
                            f'The pronoun "{subject.text}" typically refers to a female, but "{noun_lower}" refers to a male. Consider using "he".',
                            confidence=85, severity='warning', error_type='semantic'
                        ))
                    if subject.lower == 'it' and noun_lower in (MALE_NOUNS | FEMALE_NOUNS):
                        gender = 'male' if noun_lower in MALE_NOUNS else 'female'
                        suggestion = 'he' if gender == 'male' else 'she'
                        errors.append(_make_error(
                            sent.text, f'{subject.text} {token.text} {noun_token.text}',
                            f'{suggestion} {token.text} {noun_token.text}',
                            subject.idx, noun_token.idx_end,
                            'Semantic Consistency',
                            f'"{noun_lower}" refers to a {gender} person. Consider using "{suggestion}" instead of "it".',
                            confidence=75, severity='warning', error_type='semantic'
                        ))
                    break

        return errors

    def _check_double_negatives(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        neg_count = 0
        neg_tokens = []
        for token in sent.tokens:
            if token.lower in NEGATION_WORDS or token.dep == 'neg':
                neg_count += 1
                neg_tokens.append(token)

        if neg_count >= 2:
            has_strict_neg = any(t.lower in {'no', 'nothing', 'nobody', 'nowhere', 'none', 'neither', 'nor'} for t in neg_tokens)
            if has_strict_neg:
                first_neg = neg_tokens[0]
                second_neg = neg_tokens[1] if len(neg_tokens) > 1 else neg_tokens[0]
                errors.append(_make_error(
                    sent.text, f'{first_neg.text} ... {second_neg.text}',
                    '(consider rephrasing)',
                    first_neg.idx, second_neg.idx_end,
                    'Double Negative',
                    'This sentence contains a double negative, which can be confusing. Consider rephrasing.',
                    confidence=90, severity='warning', error_type='grammar'
                ))
            else:
                for i in range(len(neg_tokens) - 1):
                    t1, t2 = neg_tokens[i], neg_tokens[i + 1]
                    if t1.lower in ('hardly', 'barely', 'scarcely', 'seldom', 'rarely'):
                        errors.append(_make_error(
                            sent.text, f'{t1.text} {t2.text}',
                            '(consider rephrasing)',
                            t1.idx, t2.idx_end,
                            'Double Negative',
                            f'"{t1.text}" already implies negation. Using "{t2.text}" creates a double negative.',
                            confidence=85, severity='warning', error_type='grammar'
                        ))

        return errors

    def _check_comparatives(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        for i, token in enumerate(sent.tokens):
            if token.lower in COMPARATIVE_WORDS or token.tag in ('JJR', 'RBR'):
                for j in range(i + 1, min(i + 3, len(sent.tokens))):
                    next_t = sent.tokens[j]
                    if next_t.is_alpha:
                        if next_t.lower == 'then':
                            errors.append(_make_error(
                                sent.text, next_t.text, 'than', next_t.idx, next_t.idx_end,
                                'Comparative',
                                'Use "than" for comparisons, not "then".',
                                confidence=90
                            ))
                        break
                    elif next_t.pos == 'PUNCT':
                        break

        return errors

    def _check_confusing_words(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        for i, token in enumerate(sent.tokens):
            lower = token.lower

            if lower in ('your', "you're"):
                next_word = None
                for j in range(i + 1, min(i + 3, len(sent.tokens))):
                    if sent.tokens[j].is_alpha:
                        next_word = sent.tokens[j]
                        break
                if next_word:
                    if lower == 'your' and next_word.pos in ('VBZ', 'VBP', 'VBG', 'VBN', 'RB'):
                        if next_word.lower in ('going', 'welcome', 'right', 'wrong') or next_word.pos == 'VBG':
                            errors.append(_make_error(
                                sent.text, token.text, "you're", token.idx, token.idx_end,
                                'Confusing Words',
                                'Use "you\'re" (you are) instead of "your".',
                                confidence=85, severity='warning', error_type='context'
                            ))
                    elif lower == "you're":
                        next_lower = next_word.lower
                        if next_word.pos in ('NN', 'NNS', 'NNP') and next_lower not in ('going', 'welcome'):
                            errors.append(_make_error(
                                sent.text, token.text, 'your', token.idx, token.idx_end,
                                'Confusing Words',
                                'Use "your" (possessive) instead of "you\'re".',
                                confidence=75, severity='warning', error_type='context'
                            ))

            if lower in ('their', 'there', "they're"):
                if lower == 'there' and i > 0:
                    prev = sent.tokens[i - 1]
                    if prev.is_alpha and prev.pos in ('VBZ', 'VBP', 'VBD', 'MD'):
                        pass
                    else:
                        next_word = None
                        for j in range(i + 1, min(i + 3, len(sent.tokens))):
                            if sent.tokens[j].is_alpha:
                                next_word = sent.tokens[j]
                                break
                        if next_word and next_word.dep == 'attr':
                            pass

            if lower == 'its' and token.dep != 'poss':
                for j in range(i + 1, min(i + 3, len(sent.tokens))):
                    nw = sent.tokens[j]
                    if nw.is_alpha:
                        if nw.dep in ('attr', 'acomp', 'dobj'):
                            errors.append(_make_error(
                                sent.text, token.text, "it's", token.idx, token.idx_end,
                                'Confusing Words',
                                'Use "it\'s" (it is) instead of "its" (possessive).',
                                confidence=75, severity='warning', error_type='context'
                            ))
                        break

        return errors

    def _check_capitalization(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        if sent.tokens:
            first_word = sent.tokens[0]
            if first_word.is_alpha and first_word.text[0].islower():
                corrected = first_word.text[0].upper() + first_word.text[1:]
                errors.append(_make_error(
                    sent.text, first_word.text, corrected, first_word.idx, first_word.idx_end,
                    'Capitalization',
                    'Sentences should begin with a capital letter.',
                    confidence=99
                ))

        for token in sent.tokens:
            if token.lower == 'i' and token.pos == 'PRON' and token.text != 'I':
                if token.tag != 'NNP':
                    errors.append(_make_error(
                        sent.text, token.text, 'I', token.idx, token.idx_end,
                        'Capitalization',
                        'The pronoun "I" should always be capitalized.',
                        confidence=99
                    ))

        return errors

    def _check_punctuation(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        if sent.tokens:
            last_non_space = None
            for t in reversed(sent.tokens):
                if not t.text.isspace():
                    last_non_space = t
                    break
            if last_non_space and last_non_space.is_alpha and last_non_space.pos not in ('PUNCT',):
                if sent.text.rstrip() and sent.text.rstrip()[-1] not in '.!?':
                    errors.append(_make_error(
                        sent.text, '', '.', len(sent.text), len(sent.text),
                        'Punctuation',
                        'Sentence should end with a period, question mark, or exclamation mark.',
                        confidence=85, severity='warning'
                    ))

        for i, token in enumerate(sent.tokens):
            if token.text == 'i' and token.pos == 'PRON':
                if i > 0 and sent.tokens[i - 1].text in ('.', '!', '?', ',', ';', ':'):
                    pass

        text = sent.text
        double_punct = re.finditer(r'([.!?])\1+', text)
        for match in double_punct:
            errors.append(_make_error(
                sent.text, match.group(), match.group()[0],
                sent.start_char + match.start(), sent.start_char + match.end(),
                'Punctuation',
                f'Use a single punctuation mark, not "{match.group()}".',
                confidence=95
            ))

        return errors

    def _check_word_form(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        for token in sent.tokens:
            lower = token.lower

            if token.dep == 'ROOT' and token.pos in ('VERB', 'AUX'):
                pass

            if lower in BE_VERBS and lower not in ('be', 'been', 'being'):
                children = get_children(sent, token.idx)
                for child in children:
                    if child.pos == 'VERB' and child.dep == 'ROOT':
                        if child.tag in ('VB', 'VBP'):
                            if child.lower not in ('agree', 'awake', 'aware', 'able', 'alive'):
                                base = child.lower
                                if base.endswith('e'):
                                    expected = base[:-1] + 'ing'
                                else:
                                    expected = base + 'ing'
                                errors.append(_make_error(
                                    sent.text, f'{token.text} {child.text}',
                                    f'{token.text} {expected}',
                                    token.idx, child.idx_end,
                                    'Verb Form',
                                    f'After "{token.lower}", use the present participle "{expected}", not "{child.text}".',
                                    confidence=85, severity='warning'
                                ))
                        break

            if lower in HAVE_VERBS and lower != 'had':
                children = get_children(sent, token.idx)
                for child in children:
                    if child.pos == 'VERB' and child.dep == 'ROOT':
                        if child.tag == 'VBD' or child.lower in PAST_TO_BASE:
                            past = child.lower
                            base = PAST_TO_BASE.get(past, past)
                            forms = IRREGULAR_VERBS.get(base, {})
                            past_part = forms.get('past_part', base + 'ed')
                            if past_part != past:
                                errors.append(_make_error(
                                    sent.text, child.text, past_part, child.idx, child.idx_end,
                                    'Verb Form',
                                    f'After "{token.lower}", use the past participle "{past_part}", not the simple past "{past}".',
                                    confidence=90
                                ))
                        break

        return errors

    def _check_sentence_completeness(self, sent: NlpSentence) -> List[Dict]:
        errors = []
        word_tokens = [t for t in sent.tokens if t.is_alpha]
        if len(word_tokens) == 0:
            return errors

        has_verb = any(t.pos in ('VERB', 'AUX') for t in word_tokens)
        has_noun = any(t.pos in ('NOUN', 'PROPN', 'PRON') for t in word_tokens)

        if len(word_tokens) >= 3 and has_noun and not has_verb:
            if not any(t.pos in ('ADJ', 'ADV') for t in word_tokens):
                errors.append(_make_error(
                    sent.text, sent.text, '', 0, len(sent.text),
                    'Sentence Structure',
                    'This sentence may be missing a verb.',
                    confidence=60, severity='info', error_type='style'
                ))

        return errors


ALL_PAST_FORMS = set()
ALL_PAST_PART_FORMS = set()
for base, forms in IRREGULAR_VERBS.items():
    ALL_PAST_FORMS.add(forms['past'])
    ALL_PAST_PART_FORMS.add(forms['past_part'])
