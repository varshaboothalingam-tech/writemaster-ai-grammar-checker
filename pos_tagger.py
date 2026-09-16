DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her',
               'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each', 'all', 'both', 'few',
               'many', 'much', 'several', 'enough', 'either', 'neither'}
PREPOSITIONS = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
                'over', 'out', 'up', 'down', 'off', 'near', 'behind', 'beyond', 'along',
                'across', 'against', 'among', 'around', 'as', 'beside', 'besides', 'beyond',
                'despite', 'down', 'except', 'inside', 'like', 'near', 'onto', 'opposite',
                'outside', 'past', 'per', 'plus', 'regarding', 'round', 'since', 'than',
                'throughout', 'till', 'toward', 'towards', 'under', 'unlike', 'until',
                'upon', 'via', 'within', 'without', 'according', 'given', 'considering',
                'concerning', 'following', 'notwithstanding'}
CONJUNCTIONS = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although',
                'though', 'while', 'whereas', 'if', 'unless', 'until', 'since', 'when',
                'whenever', 'where', 'wherever', 'whether', 'after', 'before', 'as',
                'once', 'than', 'that', 'which', 'who', 'whom', 'whose'}
PRONOUNS_SUBJECT = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'who', 'what', 'which'}
PRONOUNS_OBJECT = {'me', 'him', 'her', 'us', 'them', 'whom'}
PRONOUNS_POSSESSIVE = {'my', 'your', 'his', 'her', 'its', 'our', 'their'}
PRONOUNS_REFLEXIVE = {'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves'}
BE_VERBS = {'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
HAVE_VERBS = {'have', 'has', 'had', 'having'}
DO_VERBS = {'do', 'does', 'did'}
MODALS = {'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}
AUXILIARIES = BE_VERBS | HAVE_VERBS | DO_VERBS | MODALS
NOT_WORDS = {'not', "n't"}
NEGATIONS = {'not', "n't", 'no', 'never', 'neither', 'nor', 'nothing', 'nowhere', 'nobody',
             'no one', 'none', 'hardly', 'scarcely', 'barely'}
COMMON_ADVERBS = {'very', 'really', 'quite', 'rather', 'too', 'enough', 'almost', 'nearly',
                  'just', 'also', 'always', 'never', 'often', 'sometimes', 'usually',
                  'already', 'still', 'yet', 'only', 'even', 'ever', 'again', 'here',
                  'there', 'now', 'then', 'today', 'tomorrow', 'yesterday', 'soon',
                  'quickly', 'slowly', 'easily', 'carefully', 'well', 'badly', 'hard',
                  'fast', 'early', 'late', 'far', 'near', 'together', 'alone', 'perhaps',
                  'definitely', 'certainly', 'probably', 'possibly', 'obviously',
                  'apparently', 'clearly', 'simply', 'merely', 'actually', 'basically',
                  'fortunately', 'unfortunately', 'surprisingly', 'hopefully',
                  'suddenly', 'immediately', 'eventually', 'finally', 'recently',
                  'previously', 'currently'}
COMMON_ADJECTIVES = {'good', 'bad', 'big', 'small', 'new', 'old', 'great', 'little', 'long',
                     'short', 'high', 'low', 'large', 'young', 'important', 'different',
                     'same', 'other', 'more', 'most', 'few', 'many', 'much', 'some',
                     'next', 'last', 'first', 'second', 'third', 'every', 'own', 'real',
                     'sure', 'true', 'full', 'free', 'right', 'best', 'better', 'early',
                     'possible', 'whole', 'clear', 'easy', 'open', 'strong', 'true',
                     'wrong', 'hard', 'simple', 'fast', 'slow', 'hot', 'cold', 'warm',
                     'dark', 'light', 'white', 'black', 'red', 'blue', 'green', 'brown',
                     'happy', 'sad', 'angry', 'afraid', 'alive', 'alone', 'aware',
                     'able', 'common', 'difficult', 'final', 'foreign', 'general',
                     'natural', 'normal', 'popular', 'serious', 'special', 'standard',
                     'strange', 'useful', 'usual', 'important', 'interesting', 'beautiful',
                     'ugly', 'quiet', 'loud', 'rich', 'poor', 'sick', 'healthy', 'safe',
                     'dangerous', 'dry', 'wet', 'clean', 'dirty', 'empty', 'full',
                     'flat', 'round', 'straight', 'thick', 'thin', 'heavy', 'tight',
                     'loose', 'sharp', 'soft', 'smooth', 'rough', 'sweet', 'sour', 'bitter',
                     'fresh', 'stale', 'modern', 'ancient', 'local', 'national', 'global',
                     'public', 'private', 'political', 'personal', 'medical', 'legal',
                     'financial', 'physical', 'mental', 'social', 'cultural', 'natural',
                     'visible', 'invisible', 'audible', 'edible', 'possible', 'impossible',
                     'active', 'passive', 'aggressive', 'creative', 'conservative',
                     'liberal', 'conservative', 'positive', 'negative', 'neutral',
                     'direct', 'indirect', 'simple', 'complex', 'basic', 'advanced',
                     'formal', 'informal', 'official', 'unofficial', 'regular', 'irregular',
                     'normal', 'abnormal', 'typical', 'atypical', 'average', 'ordinary',
                     'extraordinary', 'unique', 'common', 'rare', 'frequent', 'occasional',
                     'permanent', 'temporary', 'constant', 'variable', 'stable', 'unstable',
                     'complete', 'incomplete', 'total', 'partial', 'entire', 'half',
                     'double', 'single', 'primary', 'secondary', 'main', 'major', 'minor',
                     'key', 'vital', 'essential', 'necessary', 'unnecessary', 'optional',
                     'current', 'former', 'latter', 'following', 'preceding', 'subsequent',
                     'internal', 'external', 'central', 'peripheral', 'upper', 'lower',
                     'front', 'back', 'left', 'right', 'top', 'bottom', 'inner', 'outer',
                     'domestic', 'foreign', 'tropical', 'polar', 'marine', 'terrestrial',
                     'digital', 'analog', 'electronic', 'mechanical', 'manual', 'automatic',
                     'correct', 'incorrect', 'accurate', 'inaccurate', 'valid', 'invalid',
                     'genuine', 'fake', 'authentic', 'artificial', 'real', 'imaginary',
                     'concrete', 'abstract', 'specific', 'general', 'particular', 'overall',
                     'individual', 'collective', 'separate', 'combined', 'joint', 'mutual',
                     'independent', 'dependent', 'equal', 'unequal', 'fair', 'unfair',
                     'legal', 'illegal', 'lawful', 'unlawful', 'moral', 'immoral',
                     'ethical', 'unethical', 'decent', 'indecent', 'proper', 'improper',
                     'appropriate', 'inappropriate', 'suitable', 'unsuitable', 'fit', 'unfit',
                     'ready', 'unready', 'willing', 'unwilling', 'able', 'unable',
                     'certain', 'uncertain', 'confident', 'uncertain', 'doubtful', 'sure'}
IRREGULAR_VERBS = {
    'be': {'past': 'was', 'past_part': 'been', 'present_3rd': 'is', 'present_pl': 'are', 'gerund': 'being'},
    'have': {'past': 'had', 'past_part': 'had', 'present_3rd': 'has', 'present_pl': 'have', 'gerund': 'having'},
    'do': {'past': 'did', 'past_part': 'done', 'present_3rd': 'does', 'present_pl': 'do', 'gerund': 'doing'},
    'go': {'past': 'went', 'past_part': 'gone', 'present_3rd': 'goes', 'present_pl': 'go', 'gerund': 'going'},
    'say': {'past': 'said', 'past_part': 'said', 'present_3rd': 'says', 'present_pl': 'say', 'gerund': 'saying'},
    'get': {'past': 'got', 'past_part': 'got', 'present_3rd': 'gets', 'present_pl': 'get', 'gerund': 'getting'},
    'make': {'past': 'made', 'past_part': 'made', 'present_3rd': 'makes', 'present_pl': 'make', 'gerund': 'making'},
    'know': {'past': 'knew', 'past_part': 'known', 'present_3rd': 'knows', 'present_pl': 'know', 'gerund': 'knowing'},
    'think': {'past': 'thought', 'past_part': 'thought', 'present_3rd': 'thinks', 'present_pl': 'think', 'gerund': 'thinking'},
    'come': {'past': 'came', 'past_part': 'come', 'present_3rd': 'comes', 'present_pl': 'come', 'gerund': 'coming'},
    'take': {'past': 'took', 'past_part': 'taken', 'present_3rd': 'takes', 'present_pl': 'take', 'gerund': 'taking'},
    'see': {'past': 'saw', 'past_part': 'seen', 'present_3rd': 'sees', 'present_pl': 'see', 'gerund': 'seeing'},
    'give': {'past': 'gave', 'past_part': 'given', 'present_3rd': 'gives', 'present_pl': 'give', 'gerund': 'giving'},
    'find': {'past': 'found', 'past_part': 'found', 'present_3rd': 'finds', 'present_pl': 'find', 'gerund': 'finding'},
    'tell': {'past': 'told', 'past_part': 'told', 'present_3rd': 'tells', 'present_pl': 'tell', 'gerund': 'telling'},
    'ask': {'past': 'asked', 'past_part': 'asked', 'present_3rd': 'asks', 'present_pl': 'ask', 'gerund': 'asking'},
    'work': {'past': 'worked', 'past_part': 'worked', 'present_3rd': 'works', 'present_pl': 'work', 'gerund': 'working'},
    'seem': {'past': 'seemed', 'past_part': 'seemed', 'present_3rd': 'seems', 'present_pl': 'seem', 'gerund': 'seeming'},
    'feel': {'past': 'felt', 'past_part': 'felt', 'present_3rd': 'feels', 'present_pl': 'feel', 'gerund': 'feeling'},
    'leave': {'past': 'left', 'past_part': 'left', 'present_3rd': 'leaves', 'present_pl': 'leave', 'gerund': 'leaving'},
    'call': {'past': 'called', 'past_part': 'called', 'present_3rd': 'calls', 'present_pl': 'call', 'gerund': 'calling'},
    'try': {'past': 'tried', 'past_part': 'tried', 'present_3rd': 'tries', 'present_pl': 'try', 'gerund': 'trying'},
    'keep': {'past': 'kept', 'past_part': 'kept', 'present_3rd': 'keeps', 'present_pl': 'keep', 'gerund': 'keeping'},
    'let': {'past': 'let', 'past_part': 'let', 'present_3rd': 'lets', 'present_pl': 'let', 'gerund': 'letting'},
    'begin': {'past': 'began', 'past_part': 'begun', 'present_3rd': 'begins', 'present_pl': 'begin', 'gerund': 'beginning'},
    'show': {'past': 'showed', 'past_part': 'shown', 'present_3rd': 'shows', 'present_pl': 'show', 'gerund': 'showing'},
    'hear': {'past': 'heard', 'past_part': 'heard', 'present_3rd': 'hears', 'present_pl': 'hear', 'gerund': 'hearing'},
    'play': {'past': 'played', 'past_part': 'played', 'present_3rd': 'plays', 'present_pl': 'play', 'gerund': 'playing'},
    'run': {'past': 'ran', 'past_part': 'run', 'present_3rd': 'runs', 'present_pl': 'run', 'gerund': 'running'},
    'move': {'past': 'moved', 'past_part': 'moved', 'present_3rd': 'moves', 'present_pl': 'move', 'gerund': 'moving'},
    'live': {'past': 'lived', 'past_part': 'lived', 'present_3rd': 'lives', 'present_pl': 'live', 'gerund': 'living'},
    'believe': {'past': 'believed', 'past_part': 'believed', 'present_3rd': 'believes', 'present_pl': 'believe', 'gerund': 'believing'},
    'hold': {'past': 'held', 'past_part': 'held', 'present_3rd': 'holds', 'present_pl': 'hold', 'gerund': 'holding'},
    'bring': {'past': 'brought', 'past_part': 'brought', 'present_3rd': 'brings', 'present_pl': 'bring', 'gerund': 'bringing'},
    'happen': {'past': 'happened', 'past_part': 'happened', 'present_3rd': 'happens', 'present_pl': 'happen', 'gerund': 'happening'},
    'write': {'past': 'wrote', 'past_part': 'written', 'present_3rd': 'writes', 'present_pl': 'write', 'gerund': 'writing'},
    'provide': {'past': 'provided', 'past_part': 'provided', 'present_3rd': 'provides', 'present_pl': 'provide', 'gerund': 'providing'},
    'sit': {'past': 'sat', 'past_part': 'sat', 'present_3rd': 'sits', 'present_pl': 'sit', 'gerund': 'sitting'},
    'stand': {'past': 'stood', 'past_part': 'stood', 'present_3rd': 'stands', 'present_pl': 'stand', 'gerund': 'standing'},
    'lose': {'past': 'lost', 'past_part': 'lost', 'present_3rd': 'loses', 'present_pl': 'lose', 'gerund': 'losing'},
    'pay': {'past': 'paid', 'past_part': 'paid', 'present_3rd': 'pays', 'present_pl': 'pay', 'gerund': 'paying'},
    'meet': {'past': 'met', 'past_part': 'met', 'present_3rd': 'meets', 'present_pl': 'meet', 'gerund': 'meeting'},
    'include': {'past': 'included', 'past_part': 'included', 'present_3rd': 'includes', 'present_pl': 'include', 'gerund': 'including'},
    'read': {'past': 'read', 'past_part': 'read', 'present_3rd': 'reads', 'present_pl': 'read', 'gerund': 'reading'},
    'grow': {'past': 'grew', 'past_part': 'grown', 'present_3rd': 'grows', 'present_pl': 'grow', 'gerund': 'growing'},
    'open': {'past': 'opened', 'past_part': 'opened', 'present_3rd': 'opens', 'present_pl': 'open', 'gerund': 'opening'},
    'walk': {'past': 'walked', 'past_part': 'walked', 'present_3rd': 'walks', 'present_pl': 'walk', 'gerund': 'walking'},
    'win': {'past': 'won', 'past_part': 'won', 'present_3rd': 'wins', 'present_pl': 'win', 'gerund': 'winning'},
    'teach': {'past': 'taught', 'past_part': 'taught', 'present_3rd': 'teaches', 'present_pl': 'teach', 'gerund': 'teaching'},
    'study': {'past': 'studied', 'past_part': 'studied', 'present_3rd': 'studies', 'present_pl': 'study', 'gerund': 'studying'},
    'speak': {'past': 'spoke', 'past_part': 'spoken', 'present_3rd': 'speaks', 'present_pl': 'speak', 'gerund': 'speaking'},
    'spend': {'past': 'spent', 'past_part': 'spent', 'present_3rd': 'spends', 'present_pl': 'spend', 'gerund': 'spending'},
    'fall': {'past': 'fell', 'past_part': 'fallen', 'present_3rd': 'falls', 'present_pl': 'fall', 'gerund': 'falling'},
    'cut': {'past': 'cut', 'past_part': 'cut', 'present_3rd': 'cuts', 'present_pl': 'cut', 'gerund': 'cutting'},
    'put': {'past': 'put', 'past_part': 'put', 'present_3rd': 'puts', 'present_pl': 'put', 'gerund': 'putting'},
    'set': {'past': 'set', 'past_part': 'set', 'present_3rd': 'sets', 'present_pl': 'set', 'gerund': 'setting'},
    'cost': {'past': 'cost', 'past_part': 'cost', 'present_3rd': 'costs', 'present_pl': 'cost', 'gerund': 'costing'},
    'hit': {'past': 'hit', 'past_part': 'hit', 'present_3rd': 'hits', 'present_pl': 'hit', 'gerund': 'hitting'},
    'lead': {'past': 'led', 'past_part': 'led', 'present_3rd': 'leads', 'present_pl': 'lead', 'gerund': 'leading'},
    'feed': {'past': 'fed', 'past_part': 'fed', 'present_3rd': 'feeds', 'present_pl': 'feed', 'gerund': 'feeding'},
    'catch': {'past': 'caught', 'past_part': 'caught', 'present_3rd': 'catches', 'present_pl': 'catch', 'gerund': 'catching'},
    'build': {'past': 'built', 'past_part': 'built', 'present_3rd': 'builds', 'present_pl': 'build', 'gerund': 'building'},
    'send': {'past': 'sent', 'past_part': 'sent', 'present_3rd': 'sends', 'present_pl': 'send', 'gerund': 'sending'},
    'sell': {'past': 'sold', 'past_part': 'sold', 'present_3rd': 'sells', 'present_pl': 'sell', 'gerund': 'selling'},
    'choose': {'past': 'chose', 'past_part': 'chosen', 'present_3rd': 'chooses', 'present_pl': 'choose', 'gerund': 'choosing'},
    'drive': {'past': 'drove', 'past_part': 'driven', 'present_3rd': 'drives', 'present_pl': 'drive', 'gerund': 'driving'},
    'break': {'past': 'broke', 'past_part': 'broken', 'present_3rd': 'breaks', 'present_pl': 'break', 'gerund': 'breaking'},
    'wake': {'past': 'woke', 'past_part': 'woken', 'present_3rd': 'wakes', 'present_pl': 'wake', 'gerund': 'waking'},
    'swim': {'past': 'swam', 'past_part': 'swum', 'present_3rd': 'swims', 'present_pl': 'swim', 'gerund': 'swimming'},
    'sing': {'past': 'sang', 'past_part': 'sung', 'present_3rd': 'sings', 'present_pl': 'sing', 'gerund': 'singing'},
    'drink': {'past': 'drank', 'past_part': 'drunk', 'present_3rd': 'drinks', 'present_pl': 'drink', 'gerund': 'drinking'},
    'ring': {'past': 'rang', 'past_part': 'rung', 'present_3rd': 'rings', 'present_pl': 'ring', 'gerund': 'ringing'},
    'spring': {'past': 'sprang', 'past_part': 'sprung', 'present_3rd': 'springs', 'present_pl': 'spring', 'gerund': 'springing'},
    'shrink': {'past': 'shrank', 'past_part': 'shrunk', 'present_3rd': 'shrinks', 'present_pl': 'shrink', 'gerund': 'shrinking'},
    'sink': {'past': 'sank', 'past_part': 'sunk', 'present_3rd': 'sinks', 'present_pl': 'sink', 'gerund': 'sinking'},
    'stink': {'past': 'stank', 'past_part': 'stunk', 'present_3rd': 'stinks', 'present_pl': 'stink', 'gerund': 'stinking'},
    'blow': {'past': 'blew', 'past_part': 'blown', 'present_3rd': 'blows', 'present_pl': 'blow', 'gerund': 'blowing'},
    'fly': {'past': 'flew', 'past_part': 'flown', 'present_3rd': 'flies', 'present_pl': 'fly', 'gerund': 'flying'},
    'throw': {'past': 'threw', 'past_part': 'thrown', 'present_3rd': 'throws', 'present_pl': 'throw', 'gerund': 'throwing'},
    'draw': {'past': 'drew', 'past_part': 'drawn', 'present_3rd': 'draws', 'present_pl': 'draw', 'gerund': 'drawing'},
    'freeze': {'past': 'froze', 'past_part': 'frozen', 'present_3rd': 'freezes', 'present_pl': 'freeze', 'gerund': 'freezing'},
    'shake': {'past': 'shook', 'past_part': 'shaken', 'present_3rd': 'shakes', 'present_pl': 'shake', 'gerund': 'shaking'},
    'mistake': {'past': 'mistook', 'past_part': 'mistaken', 'present_3rd': 'mistakes', 'present_pl': 'mistake', 'gerund': 'mistaking'},
    'forbid': {'past': 'forbade', 'past_part': 'forbidden', 'present_3rd': 'forbids', 'present_pl': 'forbid', 'gerund': 'forbidding'},
    'forget': {'past': 'forgot', 'past_part': 'forgotten', 'present_3rd': 'forgets', 'present_pl': 'forget', 'gerund': 'forgetting'},
    'forgive': {'past': 'forgave', 'past_part': 'forgiven', 'present_3rd': 'forgives', 'present_pl': 'forgive', 'gerund': 'forgiving'},
    'hide': {'past': 'hid', 'past_part': 'hidden', 'present_3rd': 'hides', 'present_pl': 'hide', 'gerund': 'hiding'},
    'bite': {'past': 'bit', 'past_part': 'bitten', 'present_3rd': 'bites', 'present_pl': 'bite', 'gerund': 'biting'},
    'lie': {'past': 'lay', 'past_part': 'lain', 'present_3rd': 'lies', 'present_pl': 'lie', 'gerund': 'lying'},
    'lay': {'past': 'laid', 'past_part': 'laid', 'present_3rd': 'lays', 'present_pl': 'lay', 'gerund': 'laying'},
    'rise': {'past': 'rose', 'past_part': 'risen', 'present_3rd': 'rises', 'present_pl': 'rise', 'gerund': 'rising'},
    'arise': {'past': 'arose', 'past_part': 'arisen', 'present_3rd': 'arises', 'present_pl': 'arise', 'gerund': 'arising'},
    'bear': {'past': 'bore', 'past_part': 'borne', 'present_3rd': 'bears', 'present_pl': 'bear', 'gerund': 'bearing'},
    'swear': {'past': 'swore', 'past_part': 'sworn', 'present_3rd': 'swears', 'present_pl': 'swear', 'gerund': 'swearing'},
    'tear': {'past': 'tore', 'past_part': 'torn', 'present_3rd': 'tears', 'present_pl': 'tear', 'gerund': 'tearing'},
    'wear': {'past': 'wore', 'past_part': 'worn', 'present_3rd': 'wears', 'present_pl': 'wear', 'gerund': 'wearing'},
}
PAST_TO_BASE = {}
for base, forms in IRREGULAR_VERBS.items():
    PAST_TO_BASE[forms['past']] = base
    PAST_TO_BASE[forms['past_part']] = base
COMMON_WORDS = {'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs',
                'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about',
                'and', 'but', 'or', 'nor', 'so', 'yet', 'for', 'because', 'if', 'when',
                'not', 'no', 'yes', 'very', 'really', 'quite', 'just', 'also', 'too',
                'go', 'come', 'make', 'take', 'get', 'give', 'know', 'think', 'see',
                'say', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call',
                'keep', 'let', 'begin', 'show', 'hear', 'play', 'run', 'move', 'live',
                'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand',
                'lose', 'pay', 'meet', 'include', 'read', 'grow', 'open', 'walk', 'win',
                'teach', 'study', 'speak', 'spend', 'fall', 'cut', 'put', 'set', 'cost',
                'hit', 'lead', 'feed', 'catch', 'build', 'send', 'sell', 'choose', 'drive',
                'break', 'wake', 'good', 'bad', 'big', 'small', 'new', 'old', 'great',
                'long', 'short', 'high', 'low', 'right', 'wrong', 'first', 'last',
                'like', 'want', 'need', 'use', 'find', 'tell', 'ask', 'work', 'seem',
                'day', 'time', 'year', 'way', 'thing', 'man', 'woman', 'child', 'world',
                'life', 'hand', 'part', 'place', 'case', 'week', 'company', 'system',
                'program', 'question', 'government', 'number', 'night', 'point', 'home',
                'water', 'room', 'mother', 'area', 'money', 'story', 'fact', 'month',
                'lot', 'right', 'study', 'book', 'eye', 'job', 'word', 'business',
                'issue', 'side', 'kind', 'head', 'house', 'service', 'friend', 'father',
                'power', 'hour', 'game', 'line', 'end', 'member', 'members', 'car',
                'city', 'community', 'name', 'president', 'team', 'minute', 'idea',
                'body', 'information', 'back', 'parent', 'face', 'others', 'level',
                'office', 'door', 'health', 'person', 'art', 'war', 'history', 'party',
                'result', 'change', 'morning', 'reason', 'research', 'girl', 'guy', 'moment',
                'air', 'teacher', 'force', 'education', 'dog', 'cat', 'bird', 'fish',
                'tree', 'flower', 'sun', 'moon', 'star', 'earth', 'sky', 'sea', 'river',
                'mountain', 'forest', 'city', 'town', 'village', 'road', 'street', 'school',
                'student', 'class', 'test', 'note', 'pen', 'paper', 'computer', 'phone',
                'boy', 'girl', 'men', 'women', 'children', 'baby', 'babies',
                'person', 'people', 'family', 'parents', 'children', 'brother', 'sister',
                'son', 'daughter', 'husband', 'wife', 'uncle', 'aunt', 'nephew', 'niece',
                'cousin', 'grandfather', 'grandmother', 'grandpa', 'grandma',
                'neighbor', 'enemy', 'stranger', 'president', 'king', 'queen',
                'prince', 'princess', 'hero', 'villain', 'doctor', 'nurse',
                'lawyer', 'engineer', 'scientist', 'artist', 'musician', 'actor', 'actress'}

UNCOUNTABLE_NOUNS = {'water', 'milk', 'coffee', 'tea', 'bread', 'rice', 'sugar', 'salt',
                     'meat', 'fruit', 'money', 'information', 'advice', 'news', 'music',
                     'knowledge', 'education', 'research', 'homework', 'work', 'health',
                     'happiness', 'love', 'anger', 'fear', 'hope', 'fun', 'weather',
                     'equipment', 'furniture', 'luggage', 'baggage', 'clothing', 'traffic',
                     'electricity', 'gas', 'oil', 'air', 'land', 'wood', 'paper', 'plastic',
                     'gold', 'silver', 'iron', 'copper', 'glass', 'cotton', 'wool', 'silk',
                     'flour', 'cheese', 'butter', 'cream', 'jam', 'honey', 'chocolate',
                     'wine', 'beer', 'juice', 'soup', 'sauce', 'oil', 'vinegar',
                     'space', 'time', 'weather', 'rain', 'snow', 'ice', 'fire', 'light',
                     'darkness', 'silence', 'noise', 'sound', 'music', 'speech',
                     'writing', 'reading', 'thinking', 'learning', 'teaching', 'travel',
                     'work', 'employment', 'business', 'industry', 'technology', 'science',
                     'mathematics', 'physics', 'chemistry', 'biology', 'history', 'geography',
                     'philosophy', 'psychology', 'sociology', 'economics', 'politics',
                     'art', 'music', 'literature', 'poetry', 'drama', 'film', 'photography',
                     'design', 'architecture', 'engineering', 'medicine', 'law', 'religion',
                     'faith', 'belief', 'truth', 'fact', 'evidence', 'proof', 'logic',
                     'reason', 'purpose', 'meaning', 'value', 'quality', 'quantity',
                     'progress', 'success', 'failure', 'experience', 'skill', 'ability',
                     'talent', 'strength', 'weakness', 'power', 'energy', 'force',
                     'pressure', 'stress', 'tension', 'conflict', 'peace', 'war',
                     'freedom', 'justice', 'equality', 'democracy', 'liberty',
                     'intelligence', 'wisdom', 'courage', 'patience', 'kindness',
                     'beauty', 'ugliness', 'strength', 'weakness', 'health', 'illness',
                     'disease', 'pain', 'suffering', 'pleasure', 'joy', 'sorrow',
                     'anger', 'rage', 'fear', 'terror', 'horror', 'surprise', 'shock',
                     'love', 'hate', 'jealousy', 'envy', 'pride', 'shame', 'guilt',
                     'innocence', 'guilt', 'honesty', 'dishonesty', 'loyalty', 'betrayal',
                     'trust', 'distrust', 'respect', 'disrespect', 'courtesy', 'rudeness',
                     'politeness', 'impoliteness', 'manners', 'etiquette', 'protocol',
                     'tradition', 'custom', 'culture', 'civilization', 'society', 'community'}

VERB_BASE_FORMS = {
    'eat','drink','sleep','read','write','drive','speak','teach','study','learn',
    'carry','fly','cry','try','lie','die','tie','copy','run','swim','sit',
    'buy','sell','hold','bring','build','send','spend','grow','stand','fall',
    'cut','hit','cost','catch','feed','lead','meet','put','keep','let',
    'say','see','come','get','give','make','take','know','think','find',
    'tell','ask','work','seem','feel','leave','call','show','hear','turn',
    'start','stop','move','live','begin','play','walk','talk','love','like',
    'want','need','use','open','close','help','look','change','watch','follow',
    'reach','arrive','remember','forget','believe','understand','decide','continue',
    'create','discover','produce','provide','develop','consider','suggest','explain',
    'improve','increase','include','involve','manage','offer','present','protect',
    'reduce','reflect','reject','relate','release','replace','request','respond',
    'serve','support','test','treat','trust','visit','warn','waste','wish',
    'worry','sing','dance','jump','kick','push','pull','lift','throw','catch',
    'wash','cook','clean','draw','paint','build','fix','break','choose','drive',
    'ride','climb','hunt','fish','fight','win','lose','score','plan','dream',
}

ALL_PAST_FORMS = set()
ALL_PAST_PART_FORMS = set()
for forms in IRREGULAR_VERBS.values():
    ALL_PAST_FORMS.add(forms['past'])
    ALL_PAST_PART_FORMS.add(forms['past_part'])

PLURAL_ONLY_NOUNS = {'people', 'children', 'men', 'women', 'police', 'cattle', 'scissors',
                      'pants', 'trousers', 'glasses', 'pliers', 'tweezers', 'goods',
                      'clothes', 'thanks', 'congratulations', 'earnings', 'savings',
                      'belongings', 'premises', 'headquarters',
                      'series', 'species'}

SINGULAR_ONLY_NOUNS = {'everyone', 'someone', 'anyone', 'nobody', 'somebody', 'anybody',
                        'everything', 'something', 'anything', 'nothing', 'each', 'every',
                        'neither', 'either'}

def is_verb_form(word):
    w = word.lower()
    if w in AUXILIARIES:
        return True
    if w in IRREGULAR_VERBS:
        return True
    if w in VERB_BASE_FORMS:
        return True
    if w.endswith('ing') and len(w) > 4:
        return True
    if w in ALL_PAST_FORMS:
        return True
    if w.endswith('ed') and len(w) > 3:
        return True
    return False

def is_noun_form(word):
    w = word.lower()
    if w in UNCOUNTABLE_NOUNS:
        return True
    if w in COMMON_WORDS:
        noun_like = {'day', 'time', 'year', 'way', 'thing', 'man', 'woman', 'child', 'world',
                     'life', 'hand', 'part', 'place', 'case', 'week', 'company', 'system',
                     'program', 'question', 'government', 'number', 'night', 'point', 'home',
                     'water', 'room', 'mother', 'area', 'money', 'story', 'fact', 'month',
                     'lot', 'book', 'eye', 'job', 'word', 'business', 'issue', 'side',
                     'kind', 'head', 'house', 'service', 'friend', 'father', 'power', 'hour',
                     'game', 'line', 'end', 'member', 'car', 'city', 'community', 'name',
                     'president', 'team', 'minute', 'idea', 'body', 'information', 'back',
                     'parent', 'face', 'level', 'office', 'door', 'health', 'person', 'art',
                     'war', 'history', 'party', 'result', 'change', 'morning', 'reason',
                     'research', 'girl', 'guy', 'moment', 'air', 'teacher', 'force',
                     'education', 'dog', 'cat', 'bird', 'fish', 'tree', 'flower', 'sun',
                     'moon', 'star', 'earth', 'sky', 'sea', 'river', 'mountain', 'forest',
                     'town', 'village', 'road', 'street', 'school', 'student', 'class',
                     'test', 'note', 'pen', 'paper', 'computer', 'phone',
                     'boy', 'girl', 'men', 'women', 'children', 'baby', 'person', 'people',
                     'family', 'parents', 'brother', 'sister', 'son', 'daughter', 'doctor'}
        if w in noun_like:
            return True
    for suffix in ('tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance', 'or', 'ist', 'ism', 'age'):
        if w.endswith(suffix):
            return True
    return False

def is_adjective(word):
    w = word.lower()
    if w in COMMON_ADJECTIVES:
        return True
    if w.endswith('ous') or w.endswith('ful') or w.endswith('less') or w.endswith('ive'):
        return True
    if w.endswith('able') or w.endswith('ible') or w.endswith('al') or w.endswith('ial'):
        return True
    if w.endswith('ic') or w.endswith('ical') or w.endswith('ish') or w.endswith('like'):
        return True
    if w.endswith('ly') and w not in COMMON_ADVERBS:
        return False
    return False

def is_adverb(word):
    w = word.lower()
    if w in COMMON_ADVERBS:
        return True
    if w.endswith('ly') and len(w) > 3:
        return True
    return False

NOUN_SUFFIXES = {'tion', 'sion', 'ment', 'ness', 'ity', 'ence', 'ance', 'or', 'ist', 'ism', 'age'}

def is_noun(word):
    return is_noun_form(word)

def tag_sentence(sentence):
    words = sentence.words
    if not words:
        return

    i = 0
    n = len(words)
    while i < n:
        w = words[i].lower
        if w in {'i'}:
            words[i].pos = 'PRP'
        elif w in {'me'}:
            words[i].pos = 'PRP'
        elif w in {'my', 'your', 'his', 'her', 'its', 'our', 'their'}:
            words[i].pos = 'PRP$'
        elif w in {'mine', 'yours', 'hers', 'ours', 'theirs'}:
            words[i].pos = 'PRP'
        elif w in {'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves'}:
            words[i].pos = 'PRP'
        elif w in {'he', 'she', 'it', 'you', 'we', 'they'}:
            words[i].pos = 'PRP'
        elif w in {'him', 'her', 'us', 'them'}:
            words[i].pos = 'PRP'
        elif w in {'who', 'whom', 'whose', 'which', 'what'}:
            words[i].pos = 'WP'
        elif w in {'this', 'that'}:
            words[i].pos = 'DT'
        elif w in {'these', 'those'}:
            words[i].pos = 'DT'
        elif w in {'the'}:
            words[i].pos = 'DT'
        elif w in {'a', 'an'}:
            words[i].pos = 'DT'
        elif w in {'some', 'any', 'no', 'every', 'each', 'all', 'both', 'few', 'many', 'much',
                    'several', 'enough', 'either', 'neither'}:
            words[i].pos = 'DT'
        elif w in MODALS:
            words[i].pos = 'MD'
        elif w in {'am'}:
            words[i].pos = 'VBP'
        elif w in {'is'}:
            words[i].pos = 'VBZ'
        elif w in {'are'}:
            words[i].pos = 'VBP'
        elif w in {'was'}:
            words[i].pos = 'VBD'
        elif w in {'were'}:
            words[i].pos = 'VBD'
        elif w in {'be'}:
            words[i].pos = 'VB'
        elif w in {'been'}:
            words[i].pos = 'VBN'
        elif w in {'being'}:
            words[i].pos = 'VBG'
        elif w in {'have'}:
            words[i].pos = 'VBP'
        elif w in {'has'}:
            words[i].pos = 'VBZ'
        elif w in {'had'}:
            words[i].pos = 'VBD'
        elif w in {'having'}:
            words[i].pos = 'VBG'
        elif w in {'do'}:
            words[i].pos = 'VBP'
        elif w in {'does'}:
            words[i].pos = 'VBZ'
        elif w in {'did'}:
            words[i].pos = 'VBD'
        elif w in {'not', "n't"}:
            words[i].pos = 'RB'
        elif w in PREPOSITIONS:
            words[i].pos = 'IN'
        elif w in CONJUNCTIONS:
            if w in {'and', 'but', 'or', 'nor', 'yet', 'so'}:
                words[i].pos = 'CC'
            else:
                words[i].pos = 'IN'
        elif w in COMMON_ADVERBS:
            words[i].pos = 'RB'
        elif w in {'will', 'shall'}:
            words[i].pos = 'MD'
        elif w.endswith('ly') and len(w) > 3:
            words[i].pos = 'RB'
        elif w in IRREGULAR_VERBS:
            if i + 1 < n:
                next_w = words[i + 1].lower if words[i + 1].is_word else ''
                if next_w in {'to'}:
                    words[i].pos = 'VB'
                else:
                    words[i].pos = 'VBP'
            else:
                words[i].pos = 'VBP'
        elif w in VERB_BASE_FORMS:
            prev_token = words[i-1] if i > 0 else None
            prev_pos = prev_token.pos if prev_token and prev_token.pos else ''
            if prev_pos in {'PRP', 'NN', 'NNP', 'NNS'}:
                prev_w = prev_token.lower if prev_token else ''
                if prev_w in {'i', 'you', 'we', 'they'}:
                    words[i].pos = 'VBP'
                elif prev_w in {'he', 'she', 'it'}:
                    words[i].pos = 'VB'
                else:
                    words[i].pos = 'VBP'
            elif prev_pos in {'MD', 'RB', 'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN'}:
                words[i].pos = 'VB'
            elif prev_pos in {'DT', 'PRP$'}:
                words[i].pos = 'NN'
            elif prev_pos == 'IN':
                words[i].pos = 'NN'
            else:
                words[i].pos = 'VBP'
        elif w.endswith('ing') and len(w) > 4:
            words[i].pos = 'VBG'
        elif w.endswith('ed') and len(w) > 3:
            words[i].pos = 'VBD'
        elif w.endswith('er') and len(w) > 3 and not w.endswith('ter'):
            if i + 1 < n and words[i + 1].lower in {'than'}:
                words[i].pos = 'JJR'
            else:
                words[i].pos = 'NN'
        elif w.endswith('est') and len(w) > 4:
            words[i].pos = 'JJS'
        elif w.endswith('s') and not w.endswith('ss') and len(w) > 3:
            if w in COMMON_WORDS:
                if w in {'goes', 'does', 'has', 'was', 'were', 'says', 'knows', 'thinks',
                         'sees', 'feels', 'tries', 'asks', 'uses', 'finds', 'tells', 'calls',
                         'loves', 'wants', 'needs', 'comes', 'takes', 'gives', 'makes',
                         'puts', 'keeps', 'seems', 'helps', 'shows', 'hears', 'turns',
                         'starts', 'stops', 'works', 'plays', 'walks', 'talks', 'lives',
                         'runs', 'sings', 'eats', 'drinks', 'reads', 'writes', 'drives',
                         'speaks', 'teaches', 'studies', 'carries', 'flies', 'cries',
                         'jumps', 'rocks', 'blocks', 'plays', 'stays', 'pays', 'prays'}:
                    prev_token = words[i-1] if i > 0 else None
                    prev_pos = prev_token.pos if prev_token and prev_token.pos else ''
                    if prev_pos in {'PRP', 'NN', 'NNP', 'NNS', 'VBZ', 'VBP', 'VBD', 'VB', 'MD'}:
                        prev_w = prev_token.lower if prev_token else ''
                        if prev_w in {'he', 'she', 'it'}:
                            words[i].pos = 'VBZ'
                        elif prev_w in {'i', 'you', 'we', 'they'}:
                            words[i].pos = 'VBP'
                        else:
                            words[i].pos = 'VBZ'
                    else:
                        words[i].pos = 'VBZ'
                elif w in {'boys', 'girls', 'men', 'women', 'children', 'babies',
                           'days', 'times', 'years', 'things', 'hands', 'parts',
                           'places', 'cases', 'weeks', 'companies', 'systems',
                           'programs', 'questions', 'governments', 'numbers', 'nights',
                           'points', 'homes', 'waters', 'rooms', 'areas', 'moneys',
                           'stories', 'facts', 'months', 'books', 'eyes', 'jobs',
                           'words', 'businesses', 'issues', 'sides', 'kinds', 'heads',
                           'houses', 'services', 'friends', 'fathers', 'powers', 'hours',
                           'games', 'lines', 'ends', 'members', 'cars', 'cities',
                           'communities', 'names', 'presidents', 'teams', 'minutes',
                           'ideas', 'bodies', 'offices', 'doors', 'people', 'teachers',
                           'forces', 'educations', 'dogs', 'cats', 'birds', 'fish',
                           'trees', 'flowers', 'students', 'classes', 'tests', 'notes',
                           'pens', 'papers', 'computers', 'phones', 'men', 'women',
                           'children', 'parents', 'brothers', 'sisters', 'sons',
                           'daughters', 'doctors'}:
                    words[i].pos = 'NNS'
                else:
                    words[i].pos = 'VBZ'
            elif is_noun(w[:-1]) or is_noun(w):
                words[i].pos = 'NNS'
            elif is_adjective(w):
                words[i].pos = 'JJ'
            else:
                prev_token = words[i-1] if i > 0 else None
                prev_pos = prev_token.pos if prev_token and prev_token.pos else ''
                if prev_pos in {'DT', 'PRP$', 'JJ', 'IN'}:
                    words[i].pos = 'NNS'
                else:
                    words[i].pos = 'VBZ'
        elif is_adjective(w):
            words[i].pos = 'JJ'
        elif is_adverb(w):
            words[i].pos = 'RB'
        elif is_noun(w):
            words[i].pos = 'NN'
        else:
            words[i].pos = 'NN'
        i += 1

    return words
