import json
import os
import re
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

VOCAB_DATA = load_json('vocabulary.json')
SYNONYMS = VOCAB_DATA['synonyms']
FILLER_WORDS = VOCAB_DATA['filler_words']
WEAK_WORDS = VOCAB_DATA['weak_words']

REPHRASE_DATA = load_json('rephrase_patterns.json')
REPHRASE_PATTERNS = REPHRASE_DATA.get('rephrase_styles', {})
REPHRASE_TRANSFORMATIONS = {
    'good': ['excellent', 'superb', 'outstanding', 'remarkable', 'exceptional'],
    'bad': ['poor', 'terrible', 'subpar', 'unsatisfactory', 'inadequate'],
    'big': ['large', 'substantial', 'significant', 'considerable', 'extensive'],
    'small': ['tiny', 'minimal', 'negligible', 'insignificant', 'minor'],
    'happy': ['pleased', 'content', 'delighted', 'satisfied', 'cheerful'],
    'sad': ['unhappy', 'sorrowful', 'melancholy', 'disappointed', 'gloomy'],
    'important': ['significant', 'crucial', 'vital', 'essential', 'critical'],
    'interesting': ['fascinating', 'engaging', 'intriguing', 'captivating', 'compelling'],
    'beautiful': ['gorgeous', 'stunning', 'magnificent', 'elegant', 'exquisite'],
    'quickly': ['rapidly', 'swiftly', 'promptly', 'speedily', 'expeditiously'],
    'slowly': ['gradually', 'steadily', 'laboriously', 'leisurely', 'unhurriedly'],
    'also': ['additionally', 'furthermore', 'moreover', 'in addition', 'likewise'],
    'very': ['extremely', 'remarkably', 'exceptionally', 'extraordinarily', 'profoundly'],
    'show': ['demonstrate', 'illustrate', 'display', 'exhibit', 'reveal'],
    'help': ['assist', 'aid', 'support', 'facilitate', 'enable'],
    'start': ['begin', 'commence', 'initiate', 'launch', 'originate'],
    'end': ['conclude', 'finish', 'terminate', 'complete', 'finalize'],
    'old': ['ancient', 'aged', 'vintage', 'antique', 'time-honored'],
    'new': ['innovative', 'novel', 'cutting-edge', 'modern', 'contemporary'],
    'easy': ['simple', 'straightforward', 'effortless', 'uncomplicated', 'facile'],
    'hard': ['difficult', 'challenging', 'demanding', 'tough', 'arduous'],
    'many': ['numerous', 'several', 'multiple', 'various', 'myriad'],
    'much': ['considerable', 'substantial', 'significant', 'extensive', 'abundant'],
    'like': ['enjoy', 'appreciate', 'favor', 'prefer', 'relish'],
    'think': ['believe', 'consider', 'reflect', 'contemplate', 'ponder'],
    'want': ['desire', 'wish', 'crave', 'yearn for', 'aspire to'],
    'need': ['require', 'demand', 'necessitate', 'call for', 'warrant'],
    'use': ['utilize', 'employ', 'apply', 'leverage', 'harness'],
    'try': ['attempt', 'endeavor', 'strive', 'seek', 'aim'],
    'make': ['create', 'produce', 'build', 'construct', 'generate'],
    'get': ['obtain', 'acquire', 'receive', 'gain', 'secure'],
    'give': ['provide', 'supply', 'offer', 'present', 'deliver'],
    'take': ['grab', 'seize', 'capture', 'acquire', 'obtain'],
    'say': ['state', 'declare', 'mention', 'remark', 'indicate'],
    'go': ['proceed', 'advance', 'move', 'travel', 'head'],
    'come': ['arrive', 'approach', 'reach', 'appear', 'emerge'],
    'see': ['observe', 'notice', 'witness', 'perceive', 'detect'],
    'know': ['understand', 'comprehend', 'realize', 'recognize', 'fathom'],
    'look': ['appear', 'seem', 'examine', 'inspect', 'survey'],
    'feel': ['sense', 'perceive', 'experience', 'touch', 'handle'],
    'tell': ['inform', 'notify', 'advise', 'relay', 'communicate'],
    'ask': ['inquire', 'question', 'query', 'request', 'petition'],
    'work': ['labor', 'toil', 'strive', 'function', 'operate'],
    'play': ['engage', 'participate', 'compete', 'amuse', 'recreate'],
    'run': ['sprint', 'dash', 'jog', 'rush', 'race'],
    'walk': ['stroll', 'amble', 'saunter', 'march', 'stride'],
    'eat': ['consume', 'devour', 'dine', 'feast', 'nibble'],
    'house': ['home', 'residence', 'dwelling', 'abode', 'domicile'],
    'car': ['vehicle', 'automobile', 'sedan', 'ride'],
    'food': ['cuisine', 'meal', 'dish', 'nourishment', 'sustenance'],
    'money': ['cash', 'funds', 'currency', 'wealth', 'capital'],
    'time': ['moment', 'instant', 'era', 'period', 'duration'],
    'place': ['location', 'spot', 'site', 'position', 'area'],
    'problem': ['issue', 'difficulty', 'challenge', 'obstacle', 'dilemma'],
    'answer': ['response', 'reply', 'solution', 'retort', 'explanation'],
    'change': ['alter', 'modify', 'adjust', 'transform', 'revise'],
    'find': ['discover', 'locate', 'uncover', 'detect', 'identify'],
    'keep': ['retain', 'maintain', 'preserve', 'hold', 'sustain'],
    'move': ['relocate', 'transfer', 'shift', 'advance', 'proceed'],
    'student': ['pupil', 'learner', 'scholar', 'trainee', 'apprentice'],
    'teacher': ['instructor', 'educator', 'tutor', 'professor', 'mentor'],
    'school': ['academy', 'institution', 'educational establishment'],
    'book': ['volume', 'text', 'publication', 'tome', 'manual'],
    'write': ['compose', 'author', 'pen', 'draft', 'inscribe'],
    'buy': ['purchase', 'acquire', 'procure', 'obtain'],
    'sell': ['vend', 'trade', 'market', 'dispose of'],
    'love': ['adore', 'cherish', 'treasure', 'worship', 'relish'],
    'hate': ['despise', 'loathe', 'detest', 'abhor', 'dislike'],
    'friend': ['companion', 'acquaintance', 'colleague', 'mate', 'pal'],
}

TONE_PROFILES = {
    'professional': {
        'keywords': ['furthermore', 'moreover', 'consequently', 'therefore', 'additionally', 'henceforth', 'pursuant', 'accordingly', 'subsequently', 'notwithstanding', 'herein', 'aforementioned', 'hence', 'thus', 'ergo'],
        'formal_verbs': ['utilize', 'implement', 'facilitate', 'commence', 'terminate', 'endeavor', 'accomplish', 'procure', 'ascertain', 'elucidate', 'demonstrate', 'substantiate', 'enumerate', 'expedite', 'promulgate'],
        'avoid': ['gonna', 'wanna', 'gotta', 'dunno', 'kinda', 'sorta', 'hey', 'wow', 'yep', 'nope', 'yall'],
        'indicators': ['executive summary', 'action items', 'stakeholders', 'deliverables', 'benchmarks', 'synergy', 'leverage', 'optimize', 'scalable', 'infrastructure', 'methodology', 'initiative', 'strategic', 'tactical', 'operational'],
    },
    'academic': {
        'keywords': ['hypothesis', 'methodology', 'empirical', 'theoretical', 'paradigm', 'epistemology', 'ontology', 'praxis', 'discourse', 'juxtaposition', 'dichotomy', 'heuristic', 'pedagogical', 'andragogy', 'pragmatism'],
        'formal_verbs': ['contend', 'postulate', 'elucidate', 'delineate', 'corroborate', 'substantiate', 'refute', 'negate', 'circumscribe', 'encompass', 'transcend', 'supersede', 'predominate', 'supplant', 'circumvent'],
        'avoid': ['gonna', 'wanna', 'gotta', 'definitely', 'totally', 'basically', 'actually', 'literally', 'really', 'very'],
        'indicators': ['according to', 'evidence suggests', 'research indicates', 'findings demonstrate', 'literature review', 'methodology', 'data analysis', 'conclusion', 'implications', 'limitations'],
    },
    'simple': {
        'keywords': ['easy', 'clear', 'basic', 'simple', 'straightforward', 'plain', 'direct', 'short', 'quick', 'fast', 'help', 'use', 'make', 'get', 'start', 'try', 'check', 'look', 'see', 'find'],
        'formal_verbs': ['use', 'try', 'make', 'get', 'help', 'show', 'tell', 'give', 'take', 'keep', 'let', 'start', 'stop', 'run', 'move', 'add', 'put', 'set', 'cut', 'fix'],
        'avoid': ['furthermore', 'moreover', 'consequently', 'therefore', 'additionally', 'notwithstanding', 'aforementioned', 'subsequently', 'henceforth', 'accordingly'],
        'indicators': ['here is how', 'just do this', 'the idea is', 'in other words', 'basically', 'think of it as', 'it means that', 'so basically'],
    },
    'formal': {
        'keywords': ['furthermore', 'moreover', 'consequently', 'therefore', 'additionally', 'henceforth', 'pursuant', 'accordingly', 'subsequently', 'notwithstanding', 'herein', 'aforementioned'],
        'formal_verbs': ['utilize', 'implement', 'facilitate', 'commence', 'terminate', 'endeavor', 'accomplish', 'procure', 'ascertain', 'elucidate', 'demonstrate', 'substantiate'],
        'avoid': ['gonna', 'wanna', 'gotta', 'dunno', 'kinda', 'sorta', 'hey', 'wow', 'yep', 'nope', 'yall', 'cool', 'awesome', 'totally', 'literally', 'basically'],
        'indicators': ['it is imperative', 'it should be noted', 'with regard to', 'in accordance with', 'for the purpose of', 'in the event that', 'it is essential', 'please be advised'],
    },
    'friendly': {
        'keywords': ['hey', 'hi', 'hello', 'thanks', 'great', 'awesome', 'love', 'enjoy', 'fun', 'cool', 'nice', 'glad', 'happy', 'excited', 'amazing', 'wonderful', 'fantastic', 'brilliant'],
        'formal_verbs': ['love', 'like', 'enjoy', 'want', 'need', 'try', 'check', 'look', 'see', 'show', 'help', 'share', 'tell', 'give', 'take', 'make', 'get', 'use', 'find', 'start'],
        'avoid': ['furthermore', 'moreover', 'consequently', 'therefore', 'pursuant', 'aforementioned', 'notwithstanding', 'herein', 'subsequently', 'henceforth'],
        'indicators': ['just wanted to', 'really appreciate', 'so glad', 'that sounds great', 'hope you are', 'looking forward', 'thanks so much', 'hope this helps', "can't wait", 'so excited'],
    },
    'concise': {
        'keywords': ['brief', 'short', 'quick', 'fast', 'direct', 'key', 'main', 'core', 'essential', 'must', 'need', 'use', 'do', 'try', 'check', 'start', 'stop', 'fix', 'cut', 'trim'],
        'formal_verbs': ['use', 'try', 'make', 'get', 'start', 'stop', 'keep', 'cut', 'fix', 'run', 'move', 'add', 'put', 'set', 'help', 'show', 'tell', 'give', 'take', 'find'],
        'avoid': ['furthermore', 'moreover', 'consequently', 'therefore', 'additionally', 'in order to', 'due to the fact', 'at this point in time', 'in the event that', 'for the purpose of'],
        'indicators': ['key point:', 'main idea:', 'tldr:', 'short answer:', 'in short', 'basically', 'mainly', 'primarily', 'most importantly'],
    },
    'creative': {
        'keywords': ['imagine', 'picture', 'envision', 'create', 'craft', 'weave', 'spin', 'paint', 'sketch', 'design', 'dream', 'dreamt', 'fantasize', 'invent', 'discover', 'explore', 'journey', 'adventure', 'quest', 'odyssey'],
        'formal_verbs': ['imagine', 'create', 'paint', 'sketch', 'craft', 'weave', 'spin', 'dream', 'invent', 'discover', 'explore', 'unleash', 'ignite', 'spark', 'ignite', 'awaken', 'blossom', 'flourish', 'radiate', 'illuminate'],
        'avoid': ['standard', 'normal', 'regular', 'usual', 'typical', 'common', 'ordinary', 'basic', 'plain', 'simple', 'furthermore', 'moreover', 'consequently', 'therefore'],
        'indicators': ['picture this', 'imagine a world', 'what if', 'let me tell you a story', 'once upon a time', 'in a land', 'think of it this way', 'like a', 'as if', 'metaphorically'],
    },
    'persuasive': {
        'keywords': ['believe', 'trust', 'essential', 'critical', 'vital', 'crucial', 'must', 'need', 'should', 'have to', 'important', 'significant', 'powerful', 'compelling', 'undeniable', 'irrefutable', 'proven', 'guaranteed'],
        'formal_verbs': ['prove', 'demonstrate', 'show', 'establish', 'confirm', 'validate', 'substantiate', 'verify', 'guarantee', 'ensure', 'persuade', 'convince', 'inspire', 'motivate', 'empower', 'transform', 'revolutionize'],
        'avoid': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'somewhat', 'kind of', 'sort of', 'not sure', 'i think', 'i guess', 'i suppose'],
        'indicators': ['the evidence shows', 'studies prove', 'undeniably', 'clearly', 'obviously', 'without a doubt', 'the truth is', 'you need to', 'you must', "don't miss out", 'act now', 'limited time'],
    },
}


class AIEngine:
    def __init__(self):
        self.tone_profiles = TONE_PROFILES

    def rephrase(self, text, tone='professional'):
        text = text.strip()
        if not text:
            return ''

        tone = tone.lower()
        profile = self.tone_profiles.get(tone, self.tone_profiles['professional'])

        result = text

        for original, replacements in REPHRASE_TRANSFORMATIONS.items():
            if original.lower() in result.lower():
                replacement = random.choice(replacements)
                result = re.sub(
                    r'\b' + re.escape(original) + r'\b',
                    replacement, result, flags=re.IGNORECASE
                )

        sentences = re.split(r'(?<=[.!?])\s+', result)
        transformed = []
        for sentence in sentences:
            sent = sentence.strip()
            if not sent:
                continue
            sent = self._apply_tone_transforms(sent, tone, profile)
            transformed.append(sent)

        return ' '.join(transformed)

    def _apply_tone_transforms(self, sentence, tone, profile):
        result = sentence

        if tone in ('formal', 'professional', 'academic'):
            casual_map = {
                "don't": "do not", "can't": "cannot", "won't": "will not",
                "isn't": "is not", "aren't": "are not", "wasn't": "was not",
                "weren't": "were not", "hasn't": "has not", "haven't": "have not",
                "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
                "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
                "let's": "let us", "that's": "that is", "who's": "who is",
                "what's": "what is", "there's": "there is", "here's": "here is",
                "it's": "it is", "he's": "he is", "she's": "she is",
                "we're": "we are", "they're": "they are", "you're": "you are",
                "i'm": "I am", "i've": "I have", "we've": "we have",
                "they've": "they have", "you've": "you have",
            }
            for contraction, expansion in casual_map.items():
                result = re.sub(
                    r'\b' + re.escape(contraction) + r'\b',
                    expansion, result, flags=re.IGNORECASE
                )

        if tone == 'simple':
            formal_map = {
                'utilize': 'use', 'implement': 'set up', 'commence': 'start',
                'terminate': 'end', 'endeavor': 'try', 'accomplish': 'finish',
                'procure': 'get', 'ascertain': 'find out', 'elucidate': 'explain',
                'demonstrate': 'show', 'substantiate': 'prove', 'furthermore': 'also',
                'moreover': 'also', 'consequently': 'so', 'therefore': 'so',
                'additionally': 'also', 'notwithstanding': 'despite',
                'aforementioned': 'previously mentioned', 'subsequently': 'later',
                'henceforth': 'from now on', 'accordingly': 'so',
                'approximately': 'about', 'substantial': 'big', 'significant': 'big',
                'sufficient': 'enough', 'endeavor': 'try',
            }
            for formal, simple in formal_map.items():
                result = re.sub(
                    r'\b' + re.escape(formal) + r'\b',
                    simple, result, flags=re.IGNORECASE
                )

        if tone == 'friendly':
            result = result.replace('However,', 'But,')
            result = result.replace('Therefore,', 'So,')
            result = result.replace('Furthermore,', 'Also,')
            result = result.replace('Moreover,', 'Plus,')
            result = result.replace('Consequently,', 'As a result,')
            result = result.replace('Nevertheless,', 'Still,')
            result = result.replace('Additionally,', 'And,')
            result = result.replace('Nonetheless,', 'Even so,')

        if tone == 'concise':
            wordy_map = {
                'in order to': 'to', 'due to the fact that': 'because',
                'at this point in time': 'now', 'in the event that': 'if',
                'for the purpose of': 'to', 'with regard to': 'about',
                'in accordance with': 'following', 'on a daily basis': 'daily',
                'at the present time': 'currently', 'in the near future': 'soon',
                'a large number of': 'many', 'in the process of': 'currently',
                'it is important to note that': 'note:', 'it is worth mentioning that': 'note:',
                'the reason is because': 'because', 'each and every': 'every',
                'first and foremost': 'first', 'last but not least': 'finally',
                'in light of the fact that': 'since', 'on the grounds that': 'because',
                'with the exception of': 'except', 'in spite of the fact that': 'although',
                'a significant amount of': 'many', 'the majority of': 'most',
                'in a timely manner': 'quickly', 'until such time as': 'until',
            }
            for wordy, concise in wordy_map.items():
                result = re.sub(
                    r'\b' + re.escape(wordy) + r'\b',
                    concise, result, flags=re.IGNORECASE
                )

        if tone == 'creative':
            creative_map = {
                'said': random.choice(['exclaimed', 'declared', 'proclaimed', 'announced']),
                'good': random.choice(['excellent', 'superb', 'outstanding', 'brilliant', 'magnificent']),
                'bad': random.choice(['terrible', 'dreadful', 'abysmal', 'disastrous', 'catastrophic']),
                'big': random.choice(['enormous', 'massive', 'colossal', 'gigantic', 'monumental']),
                'small': random.choice(['tiny', 'minute', 'miniature', 'compact', 'diminutive']),
                'happy': random.choice(['elated', 'thrilled', 'overjoyed', 'ecstatic', 'euphoric']),
                'sad': random.choice(['melancholy', 'despondent', 'forlorn', 'woeful', 'sorrowful']),
                'important': random.choice(['crucial', 'vital', 'essential', 'paramount', 'imperative']),
                'interesting': random.choice(['fascinating', 'captivating', 'riveting', 'enchanting', 'mesmerizing']),
                'beautiful': random.choice(['gorgeous', 'stunning', 'breathtaking', 'exquisite', 'luminous']),
            }
            for word, replacement in creative_map.items():
                result = re.sub(
                    r'\b' + re.escape(word) + r'\b',
                    replacement, result, flags=re.IGNORECASE
                )

        if tone == 'persuasive':
            result = result.replace('might', 'will')
            result = result.replace('could', 'can')
            result = result.replace('possibly', 'certainly')
            result = result.replace('perhaps', 'undoubtedly')
            result = result.replace('maybe', 'definitely')
            result = result.replace('I think', 'I am convinced')
            result = result.replace('It seems', 'It is clear')
            result = result.replace('It appears', 'It is evident')

        return result

    def enhance_writing(self, text):
        result = text
        suggestions = []

        for weak, strong_options in WEAK_WORDS.items():
            pattern = r'\b' + re.escape(weak) + r'\b'
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                suggestions.append({
                    'type': 'vocabulary',
                    'original': weak,
                    'suggestions': strong_options[:3],
                    'message': f'Consider replacing "{weak}" with a stronger word.',
                    'position': match.start(),
                })

        for word in re.findall(r'\b[a-zA-Z]+\b', result):
            if word.lower() in FILLER_WORDS:
                suggestions.append({
                    'type': 'filler',
                    'original': word,
                    'suggestions': ['Remove'],
                    'message': f'Filler word "{word}" weakens your writing.',
                    'position': result.lower().find(word.lower()),
                })

        passive_matches = list(re.finditer(
            r'\b(is|are|was|were|been|be|being)\s+(\w+ed)\b',
            result, re.IGNORECASE
        ))
        for match in passive_matches:
            suggestions.append({
                'type': 'passive_voice',
                'original': match.group(),
                'suggestions': ['Rewrite in active voice'],
                'message': f'Passive voice detected: "{match.group()}".',
                'position': match.start(),
            })

        word_count = len(result.split())
        if word_count > 0:
            unique_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', result))
            diversity = len(unique_words) / word_count
            if diversity < 0.4:
                suggestions.append({
                    'type': 'vocabulary_diversity',
                    'original': '',
                    'suggestions': ['Use more varied vocabulary'],
                    'message': f'Word diversity is low ({diversity:.0%}). Try using synonyms.',
                    'position': 0,
                })

        return {
            'enhanced_text': result,
            'suggestions': suggestions,
            'stats': {
                'original_length': word_count,
                'unique_words': len(unique_words) if word_count > 0 else 0,
                'vocabulary_diversity': round(diversity * 100, 1) if word_count > 0 else 0,
            }
        }

    def detect_tone(self, text):
        text_lower = text.lower()
        words = set(re.findall(r'\b[a-zA-Z]+\b', text_lower))
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        word_count = len(text.split())

        tone_scores = {}

        for tone, profile in self.tone_profiles.items():
            score = 0
            max_score = 100

            keyword_matches = sum(1 for kw in profile['keywords'] if kw in text_lower)
            keyword_score = min(30, keyword_matches * 6)

            verb_matches = sum(1 for v in profile['formal_verbs'] if v in text_lower)
            verb_score = min(25, verb_matches * 5)

            avoid_matches = sum(1 for a in profile['avoid'] if a in text_lower)
            avoid_penalty = avoid_matches * 4

            indicator_matches = sum(1 for ind in profile['indicators'] if ind in text_lower)
            indicator_score = min(25, indicator_matches * 8)

            if tone in ('formal', 'professional', 'academic'):
                avg_sentence_len = word_count / max(len(sentences), 1)
                if avg_sentence_len > 15:
                    keyword_score += 5
                if not any(c in text for c in ['!', '...', ':-)', ':)']):
                    keyword_score += 5

            if tone == 'simple':
                avg_sentence_len = word_count / max(len(sentences), 1)
                if avg_sentence_len < 12:
                    keyword_score += 10

            if tone == 'friendly':
                excl_count = text.count('!')
                if excl_count > 0:
                    keyword_score += min(10, excl_count * 3)

            total = keyword_score + verb_score + indicator_score - avoid_penalty
            score = max(0, min(100, round(total / max_score * 100)))

            if score < 10:
                score = random.randint(3, 12)

            tone_scores[tone] = score

        total = sum(tone_scores.values())
        if total > 0:
            for tone in tone_scores:
                tone_scores[tone] = round(tone_scores[tone] / total * 100, 1)
        else:
            for tone in tone_scores:
                tone_scores[tone] = round(100 / len(tone_scores), 1)

        dominant = max(tone_scores, key=tone_scores.get)
        sorted_tones = sorted(tone_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'dominant_tone': dominant,
            'confidence': tone_scores[dominant],
            'tones': {t: s for t, s in sorted_tones},
            'analysis': {
                'word_count': word_count,
                'sentence_count': len(sentences),
                'avg_sentence_length': round(word_count / max(len(sentences), 1), 1),
            }
        }

    def advanced_analysis(self, text):
        text_lower = text.lower()
        issues = []

        cliches = [
            'at the end of the day', 'think outside the box', 'low hanging fruit',
            'move the needle', 'synergy', 'paradigm shift', 'it is what it is',
            'the fact of the matter', 'needless to say', 'for all intents and purposes',
            'better late than never', 'actions speak louder than words',
            'every cloud has a silver lining', 'the ball is in your court',
            'bite the bullet', 'break the ice', 'cost an arm and a leg',
            'get out of hand', 'hit the nail on the head', 'let the cat out of the bag',
            'miss the boat', 'on the ball', 'pull someones leg',
            'speak of the devil', 'the best of both worlds', 'time flies',
            'you can say that again', 'a blessing in disguise', 'beat around the bush',
            'better safe than sorry', 'blessing in disguise', 'call it a day',
            'cutting corners', 'easy as pie', 'get the ball rolling',
            'go the extra mile', 'hang in there', 'keep your chin up',
            'piece of cake', 'pull yourself together', 'so far so good',
            'take it with a grain of salt', 'the last straw', 'under the weather',
        ]
        for cliche in cliches:
            if cliche in text_lower:
                issues.append({
                    'type': 'cliche', 'severity': 'info',
                    'original': cliche,
                    'message': f'Cliche detected: "{cliche}". Consider rephrasing.',
                    'suggestions': ['Replace with original phrasing'],
                    'rule': 'Cliches',
                })

        redundancy_patterns = [
            (r'\bfree\s+gift\b', 'free gift', 'gift'),
            (r'\badvance\s+planning\b', 'advance planning', 'planning'),
            (r'\bpast\s+history\b', 'past history', 'history'),
            (r'\bunexpected\s+surprise\b', 'unexpected surprise', 'surprise'),
            (r'\bend\s+result\b', 'end result', 'result'),
            (r'\bfinal\s+outcome\b', 'final outcome', 'outcome'),
            (r'\bcompletely\s+finished\b', 'completely finished', 'finished'),
            (r'\btrue\s+facts\b', 'true facts', 'facts'),
            (r'\brevert\s+back\b', 'revert back', 'revert'),
            (r'\bbasic\s+fundamentals\b', 'basic fundamentals', 'fundamentals'),
        ]
        for pattern, original, suggestion in redundancy_patterns:
            if re.search(pattern, text_lower):
                issues.append({
                    'type': 'redundancy', 'severity': 'info',
                    'original': original,
                    'message': f'Redundant phrase: "{original}"',
                    'suggestions': [suggestion],
                    'rule': 'Redundancy',
                })

        passive_patterns = [
            r'\b(is|are|was|were|been|be|being)\s+(\w+ed)\b',
            r'\b(is|are|was|were)\s+(known|made|seen|found|given|taken|done|said|thought|told|held|kept|left|set|run|won|drawn|grown|shown|thrown|born)\b',
        ]
        for pattern in passive_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issues.append({
                    'type': 'passive_voice', 'severity': 'info',
                    'original': match.group(),
                    'message': f'Passive voice detected: "{match.group()}". Consider active voice.',
                    'suggestions': ['Rewrite in active voice'],
                    'rule': 'Voice',
                })

        word_count = len(text.split())
        if word_count > 300:
            issues.append({
                'type': 'length', 'severity': 'info',
                'original': '',
                'message': f'Text is long ({word_count} words). Consider breaking into sections.',
                'suggestions': ['Break into shorter paragraphs'],
                'rule': 'Length',
            })

        paragraphs = text.split('\n\n')
        long_paragraphs = [p for p in paragraphs if len(p.split()) > 100]
        if long_paragraphs:
            issues.append({
                'type': 'structure', 'severity': 'info',
                'original': '',
                'message': f'{len(long_paragraphs)} paragraph(s) exceed 100 words.',
                'suggestions': ['Break long paragraphs into shorter ones'],
                'rule': 'Structure',
            })

        return {'advanced_issues': issues, 'word_count': word_count}
