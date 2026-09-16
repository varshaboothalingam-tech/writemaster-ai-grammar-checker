import re
import json
import os
from preprocessor import preprocess
from pos_tagger import tag_sentence
from spelling_checker import check_spelling
from grammar_analyzer import run_all_checks, run_intelligent_checks, get_intelligent_positions
from correction_engine import CorrectionEngine
from error_manager import deduplicate, rank_by_confidence, normalize_for_frontend


class GrammarChecker:
    def __init__(self):
        self._common_mistakes = None
        self._correction_engine = CorrectionEngine()

    def _load_common_mistakes(self):
        if self._common_mistakes is not None:
            return self._common_mistakes
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        try:
            with open(os.path.join(data_dir, 'common_mistakes.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._common_mistakes = data.get('common_grammar_mistakes', {})
        except Exception:
            self._common_mistakes = {}
        return self._common_mistakes

    def check(self, text):
        if not text or not text.strip():
            return []
        all_errors = []
        intelligent_errors = run_intelligent_checks(text)
        all_errors.extend(intelligent_errors)
        doc = preprocess(text)
        intelligent_positions = get_intelligent_positions()
        for sent in doc.sentences:
            tag_sentence(sent)
            spelling_errors = check_spelling(text, doc)
            all_errors.extend(spelling_errors)
            grammar_errors = run_all_checks(sent)
            for err in grammar_errors:
                key = (err.get('category', ''), err.get('incorrect', ''), err.get('position', 0))
                if key not in intelligent_positions:
                    all_errors.append(err)
        all_errors = deduplicate(all_errors)
        all_errors = rank_by_confidence(all_errors)
        all_errors = normalize_for_frontend(all_errors)
        return all_errors

    def auto_correct(self, text):
        if not text or not text.strip():
            return text
        corrected = self._correction_engine.auto_correct(text)
        corrected = self._review_and_fix(corrected)
        return corrected

    def _review_and_fix(self, text):
        max_iterations = 3
        for _ in range(max_iterations):
            errors = run_intelligent_checks(text)
            if not errors:
                break
            new_text = self._correction_engine.apply_specific_corrections(text, errors)
            if new_text == text:
                break
            text = new_text
        return text

    def calculate_readability(self, text):
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        words = text.split()
        word_count = len(words)
        syllable_count = sum(self._count_syllables(w) for w in words)
        if sentence_count == 0 or word_count == 0:
            return {'score': 0, 'level': 'N/A', 'grade': 0, 'sentences': 0, 'words': 0,
                    'syllables': 0, 'avg_words_sentence': 0, 'reading_time': 0,
                    'speaking_time': 0, 'complex_word_pct': 0, 'explanation': 'No text to analyze.'}
        asl = word_count / sentence_count
        asw = syllable_count / word_count
        flesch = 206.835 - 1.015 * asl - 84.6 * asw
        flesch = max(0, min(100, round(flesch, 1)))
        grade = round(0.39 * asl + 11.8 * asw - 15.59, 1)
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        complex_pct = round((complex_words / word_count) * 100, 1)
        if flesch >= 80: level = 'Easy'
        elif flesch >= 60: level = 'Standard'
        elif flesch >= 40: level = 'Difficult'
        else: level = 'Very Difficult'
        reading_time = round(word_count / 200, 1)
        speaking_time = round(word_count / 130, 1)
        return {
            'score': flesch, 'level': level, 'grade': max(0, grade),
            'sentences': sentence_count, 'words': word_count,
            'syllables': syllable_count, 'avg_words_sentence': round(asl, 1),
            'reading_time': reading_time, 'speaking_time': speaking_time,
            'complex_word_pct': complex_pct,
            'explanation': f'Text difficulty: {level}. Flesch Reading Ease: {flesch}.'
        }

    def _count_syllables(self, word):
        word = word.lower().strip()
        if len(word) <= 3: return 1
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
