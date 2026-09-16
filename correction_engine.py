import re
from typing import List, Dict, Tuple
from nlp_engine import analyze_text, NlpDocument, NlpSentence, NlpToken, get_children, find_subject
from intelligent_engine import (
    BE_VERBS, HAVE_VERBS, DO_VERBS, MODALS, AUX_VERBS,
    SINGULAR_PRP, PLURAL_PRP, IRREGULAR_VERBS, PAST_TO_BASE,
    NEGATION_WORDS, TEMPORAL_MARKERS_PAST,
    MALE_NOUNS, FEMALE_NOUNS, _is_plural_noun
)


class CorrectionEngine:
    def auto_correct(self, text: str) -> str:
        if not text or not text.strip():
            return text
        corrected = text
        corrected = self._fix_contractions(corrected)
        corrected = self._fix_spelling(corrected)
        corrected = self._fix_nlp_grammar(corrected)
        corrected = self._fix_capitalization(corrected)
        corrected = self._fix_punctuation(corrected)
        corrected = self._fix_spacing(corrected)
        return corrected

    def apply_specific_corrections(self, text: str, errors: List[Dict]) -> str:
        corrected = text
        for error in errors:
            if not error.get('correction') or not error.get('incorrect'):
                continue
            incorrect = error['incorrect']
            correction = error['correction']
            if correction in ('(consider rephrasing)', '(remove)'):
                continue
            if error.get('category') in ('Verb Form', 'Verb Tense'):
                parts = incorrect.split(' ', 1)
                if len(parts) == 2:
                    corrected = corrected.replace(incorrect, correction, 1)
                else:
                    corrected = re.sub(
                        r'\b' + re.escape(incorrect) + r'\b',
                        correction, corrected, count=1, flags=re.IGNORECASE
                    )
            elif error.get('category') in ('Subject-Verb Agreement', 'Modal Verb', 'Verb Tense',
                                             'Article Usage', 'Comparative', 'Countable/Uncountable'):
                corrected = re.sub(
                    r'\b' + re.escape(incorrect) + r'\b',
                    correction, corrected, count=1, flags=re.IGNORECASE
                )
            elif error.get('category') in ('Confusing Words', 'Pronoun Case'):
                if '/' in correction:
                    correction = correction.split('/')[0]
                corrected = re.sub(
                    r'\b' + re.escape(incorrect) + r'\b',
                    correction, corrected, count=1, flags=re.IGNORECASE
                )
            elif error.get('category') == 'Semantic Consistency':
                corrected = corrected.replace(incorrect, correction, 1)
            elif error.get('category') == 'Capitalization':
                if incorrect and correction:
                    corrected = corrected.replace(incorrect, correction, 1)
                elif not incorrect and correction == '.':
                    corrected = corrected.rstrip()
                    if corrected and corrected[-1] not in '.!?':
                        corrected += '.'
            elif error.get('category') == 'Punctuation':
                if correction == '.':
                    corrected = corrected.rstrip()
                    if corrected and corrected[-1] not in '.!?':
                        corrected += '.'
                elif incorrect and correction:
                    corrected = corrected.replace(incorrect, correction, 1)
            else:
                corrected = re.sub(
                    r'\b' + re.escape(incorrect) + r'\b',
                    correction, corrected, count=1, flags=re.IGNORECASE
                )
        return corrected

    def _fix_contractions(self, text):
        contractions = [
            (r"\bdont\b", "don't"), (r"\bdoesnt\b", "doesn't"),
            (r"\bdidnt\b", "didn't"), (r"\bcant\b", "can't"),
            (r"\bwont\b", "won't"), (r"\bisnt\b", "isn't"),
            (r"\barent\b", "aren't"), (r"\bwasnt\b", "wasn't"),
            (r"\bwerent\b", "weren't"), (r"\bhasnt\b", "hasn't"),
            (r"\bhavent\b", "haven't"), (r"\bhadnt\b", "hadn't"),
            (r"\bwouldnt\b", "wouldn't"), (r"\bcouldnt\b", "couldn't"),
            (r"\bshouldnt\b", "shouldn't"), (r"\bgonna\b", "going to"),
            (r"\bwanna\b", "want to"), (r"\bgotta\b", "got to"),
        ]
        for pattern, replacement in contractions:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _fix_spelling(self, text):
        try:
            from spelling_checker import SPELLING_DICT, load_spelling_dict
            load_spelling_dict()
            words = text.split()
            result = []
            for word in words:
                clean = re.sub(r'[^\w\'-]', '', word)
                lower = clean.lower()
                if lower in SPELLING_DICT and SPELLING_DICT[lower] != lower:
                    suggestion = SPELLING_DICT[lower]
                    if clean and clean[0].isupper():
                        suggestion = suggestion.capitalize()
                    result.append(suggestion)
                else:
                    result.append(word)
            return ' '.join(result)
        except Exception:
            return text

    def _fix_nlp_grammar(self, text):
        doc = analyze_text(text)
        corrections = []

        for sent in doc.sentences:
            for token in sent.tokens:
                if token.pos == 'AUX' and token.dep in ('aux', 'auxpass'):
                    subject = find_subject(sent, token)
                    if subject is None:
                        continue

                    subj_lower = subject.lower
                    aux_lower = token.lower

                    pronoun_be = {
                        'he': ('is', 'was'), 'she': ('is', 'was'), 'it': ('is', 'was'),
                        'they': ('are', 'were'), 'we': ('are', 'were'),
                        'you': ('are', 'were'), 'i': ('am', 'was'),
                    }
                    if subj_lower in pronoun_be:
                        expected = pronoun_be[subj_lower]
                        if aux_lower == expected[0] or aux_lower == expected[1]:
                            continue
                        if aux_lower in ('are', 'were') and subj_lower in ('he', 'she', 'it'):
                            replacement = 'is' if aux_lower == 'are' else 'was'
                            token_idx = sent.tokens.index(token) if token in sent.tokens else -1
                            has_neg = False
                            neg_end = token.idx_end
                            if token_idx >= 0 and token_idx + 1 < len(sent.tokens):
                                next_t = sent.tokens[token_idx + 1]
                                if next_t.dep == 'neg' or next_t.lower in ("n't", "nt", "not"):
                                    has_neg = True
                                    neg_end = next_t.idx_end
                            if has_neg:
                                replacement = "isn't" if aux_lower == 'are' else "wasn't"
                                corrections.append((token.idx, neg_end, replacement))
                            else:
                                corrections.append((token.idx, token.idx_end, replacement))
                        elif aux_lower in ('is', 'was') and subj_lower in ('they', 'we'):
                            replacement = 'are' if aux_lower == 'is' else 'were'
                            corrections.append((token.idx, token.idx_end, replacement))
                        elif aux_lower == 'are' and subj_lower == 'i':
                            corrections.append((token.idx, token.idx_end, 'am'))
                        elif aux_lower == 'is' and subj_lower == 'i':
                            corrections.append((token.idx, token.idx_end, 'am'))
                        elif aux_lower in ('have', 'has') and subj_lower in ('he', 'she', 'it'):
                            if aux_lower == 'have':
                                corrections.append((token.idx, token.idx_end, 'has'))
                        elif aux_lower in ('have', 'has') and subj_lower in ('i', 'you', 'we', 'they'):
                            if aux_lower == 'has':
                                corrections.append((token.idx, token.idx_end, 'have'))
                        elif aux_lower in ('do', 'does'):
                            if subj_lower in ('he', 'she', 'it') and aux_lower == 'do':
                                token_idx = sent.tokens.index(token) if token in sent.tokens else -1
                                has_neg = False
                                neg_end = token.idx_end
                                if token_idx >= 0 and token_idx + 1 < len(sent.tokens):
                                    next_t = sent.tokens[token_idx + 1]
                                    if next_t.dep == 'neg' or next_t.lower in ("n't", "nt", "not"):
                                        has_neg = True
                                        neg_end = next_t.idx_end
                                if has_neg:
                                    corrections.append((token.idx, neg_end, "doesn't"))
                                else:
                                    corrections.append((token.idx, token.idx_end, 'does'))
                            elif subj_lower in ('i', 'you', 'we', 'they') and aux_lower == 'does':
                                token_idx = sent.tokens.index(token) if token in sent.tokens else -1
                                has_neg = False
                                neg_end = token.idx_end
                                if token_idx >= 0 and token_idx + 1 < len(sent.tokens):
                                    next_t = sent.tokens[token_idx + 1]
                                    if next_t.dep == 'neg' or next_t.lower in ("n't", "nt", "not"):
                                        has_neg = True
                                        neg_end = next_t.idx_end
                                if has_neg:
                                    corrections.append((token.idx, neg_end, "don't"))
                                else:
                                    corrections.append((token.idx, token.idx_end, 'do'))
                    elif subject.pos in ('NOUN', 'PROPN'):
                        is_plural_noun = _is_plural_noun(subject)
                        if aux_lower == 'is' and is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'are'))
                        elif aux_lower == 'are' and not is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'is'))
                        elif aux_lower == 'was' and is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'were'))
                        elif aux_lower == 'were' and not is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'was'))
                        elif aux_lower == 'has' and is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'have'))
                        elif aux_lower == 'have' and not is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'has'))
                        elif aux_lower == 'do' and not is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'does'))
                        elif aux_lower == 'does' and is_plural_noun:
                            corrections.append((token.idx, token.idx_end, 'do'))

                if token.dep == 'ROOT' and token.pos == 'VERB':
                    has_aux = False
                    for child in get_children(sent, token.idx):
                        if child.pos == 'AUX' and child.dep in ('aux', 'auxpass'):
                            has_aux = True
                            break
                    if has_aux:
                        continue
                    subject = find_subject(sent, token)
                    if subject and subject.pos == 'PRON':
                        subj_lower = subject.lower
                        if subj_lower in SINGULAR_PRP and token.tag not in ('VBG', 'VBN', 'VBZ'):
                            expected = self._conjugate_3rd(token.lower)
                            if expected and expected != token.lower and token.lower not in BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS:
                                corrections.append((token.idx, token.idx_end, expected))

                if token.lower in MALE_NOUNS and token.pos in ('NOUN', 'PROPN') and token.dep in ('attr', 'dobj', 'oprd'):
                    subject = find_subject(sent, token)
                    if subject is None:
                        for child in sent.tokens:
                            if child.dep in ('nsubj', 'nsubjpass') and child.pos == 'PRON':
                                subject = child
                                break
                    if subject and subject.lower in ('she', 'it'):
                        corrections.append((subject.idx, subject.idx_end, 'he'))
                if token.lower in FEMALE_NOUNS and token.pos in ('NOUN', 'PROPN') and token.dep in ('attr', 'dobj', 'oprd'):
                    subject = find_subject(sent, token)
                    if subject is None:
                        for child in sent.tokens:
                            if child.dep in ('nsubj', 'nsubjpass') and child.pos == 'PRON':
                                subject = child
                                break
                    if subject and subject.lower in ('he', 'it'):
                        corrections.append((subject.idx, subject.idx_end, 'she'))

                if token.lower == 'did' and token.dep == 'aux':
                    for t in sent.tokens:
                        if t.idx == token.head_idx and t.pos == 'VERB':
                            if t.tag == 'VBD':
                                base = PAST_TO_BASE.get(t.lower)
                                if base:
                                    corrections.append((t.idx, t.idx_end, base))
                                else:
                                    base = t.lower
                                    if base.endswith('ied'):
                                        base = base[:-3] + 'y'
                                    elif base.endswith('ed') and len(base) > 4:
                                        base = base[:-2]
                                    if base and len(base) > 1:
                                        corrections.append((t.idx, t.idx_end, base))
                            break

                if token.lower in ('has', 'have') and token.dep == 'aux':
                    for t in sent.tokens:
                        if t.idx == token.head_idx and t.pos == 'VERB':
                            if t.lower in IRREGULAR_VERBS:
                                past_part = IRREGULAR_VERBS[t.lower].get('past_part')
                                if past_part and past_part != t.lower:
                                    corrections.append((t.idx, t.idx_end, past_part))
                            elif t.tag == 'VBD' or t.lower.endswith('ed'):
                                base = PAST_TO_BASE.get(t.lower, t.lower)
                                forms = IRREGULAR_VERBS.get(base, {})
                                past_part = forms.get('past_part', base + 'ed')
                                if past_part != t.lower:
                                    corrections.append((t.idx, t.idx_end, past_part))
                            break

                if token.lower in MODALS and token.dep == 'aux':
                    for t in sent.tokens:
                        if t.idx == token.head_idx and t.pos == 'VERB':
                            if t.tag in ('VBZ', 'VBP'):
                                base = t.lower
                                if base.endswith('es') and len(base) > 3:
                                    base = base[:-2]
                                elif base.endswith('s') and len(base) > 2:
                                    base = base[:-1]
                                if base and len(base) > 1 and base not in BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS:
                                    corrections.append((t.idx, t.idx_end, base))
                            break

        corrections.sort(key=lambda x: x[0], reverse=True)
        result = text
        for start, end, replacement in corrections:
            if start < len(result) and end <= len(result):
                result = result[:start] + replacement + result[end:]

        return result

    def _conjugate_3rd(self, base):
        if base in IRREGULAR_VERBS:
            return IRREGULAR_VERBS[base].get('present_3rd', base + 's')
        if not base or len(base) < 2:
            return base
        if base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
            return base + 'es'
        if base.endswith('y') and len(base) > 1 and base[-2] not in 'aeiou':
            return base[:-1] + 'ies'
        return base + 's'

    def _fix_capitalization(self, text):
        if not text:
            return text
        result = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        return result

    def _fix_punctuation(self, text):
        text = text.strip()
        if text and text[-1] not in '.!?':
            text += '.'
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r',\s*,', ',', text)
        return text

    def _fix_spacing(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()
