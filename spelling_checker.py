import re
from preprocessor import expand_contractions
from pos_tagger import COMMON_WORDS, IRREGULAR_VERBS

def _build_valid_words():
    vw = set(COMMON_WORDS)
    vw.update(['goes','does','has','was','were','been','being','does','says','knows','thinks',
               'sees','feels','tries','asks','uses','finds','tells','calls','loves','wants',
               'needs','comes','takes','gives','makes','puts','keeps','seems','helps','shows',
               'hears','turns','starts','stops','works','plays','walks','talks','lives','runs',
               'sings','eats','drinks','read','writes','drives','speaks','teaches','studies',
               'carries','flies','cries','tries','flies','dies','lies','ties','copies','runs',
               'swims','sits','sells','buys','finds','holds','brings','builds','sends','spends',
               'grows','stands','falls','cuts','hits','costs','catches','feeds','leads','meets',
               'store','stores','place','places','house','houses','room','rooms','door','doors',
               'window','windows','street','streets','road','roads','car','cars','bus','buses',
               'train','trains','shop','shops','market','markets','park','parks','garden','gardens',
               'food','drink','meal','meals','table','tables','chair','chairs','bed','beds',
               'clothes','shirt','shirts','shoe','shoes','hat','hats','coat','coats',
               'bag','bags','box','boxes','key','keys','letter','letters','name','names',
               'hand','hands','eye','eyes','face','faces','head','heads','body','bodies',
               'heart','hearts','mind','minds','voice','voices','smile','smiles',
               'sun','moon','stars','rain','snow','wind','fire','trees','flowers','grass',
               'river','lakes','ocean','sea','island','mountains','valley','forest',
               'dog','dogs','cat','cats','bird','birds','horse','horses','fish','animals',
               'horse','horses','cow','cows','pig','pigs','sheep','chickens',
               'baby','babies','boy','boys','girl','girls','man','men','woman','women',
               'child','children','people','family','friend','friends','parents','mother','father',
               'brother','sister','son','daughter','husband','wife','uncle','aunt',
               'doctor','nurse','teacher','student','students','class','classes',
               'king','queen','prince','princess','hero','villain',
               'president','officer','soldier','police','army','government',
               'church','school','college','university','library','museum',
               'hospital','hotel','restaurant','factory','office','bank',
               'story','stories','song','songs','picture','pictures','movie','movies',
               'book','books','newspaper','magazine','paper','papers',
               'map','maps','music','art','dance','game','games','sport','sports',
               'ball','balls','toy','toys','tool','tools','machine','machines',
               'glass','glasses','bottle','bottles','cup','cups','plate','plates',
               'spoon','fork','knife','knives','pot','pans',
               'clock','watch','ring','chains','jewelry','diamond',
               'money','price','cost','tax','taxes','bill','bills','bank',
               'idea','ideas','plan','plans','dream','dreams','hope','fears',
               'truth','lies','secret','secrets','answer','answers','question','questions',
               'problem','problems','solution','solutions','method','methods',
               'reason','reasons','result','results','effect','effects',
               'power','force','energy','strength','speed','size','shape',
               'color','colors','sound','silence','noise','voice','voices',
               'heat','cold','light','darkness','space','surface',
               'land','ground','field','fields','soil','sand','dust','mud',
               'stone','rocks','wood','metal','iron','gold','silver','copper',
               'plastic','rubber','cotton','silk','wool','leather',
               'oil','gas','coal','fuel','energy','electricity',
               'air','breath','smoke','steam','dust','dirt','mud',
               'blood','bone','skin','hair','teeth','nails',
               'finger','fingers','thumb','toes','arm','arms','leg','legs',
               'foot','feet','shoulder','shoulders','knee','knees',
               'back','chest','stomach','neck','hip','hips',
               'earth','world','country','countries','state','states',
               'city','cities','town','towns','village','villages',
               'north','south','east','west','middle','center','edge','corner',
               'top','bottom','front','back','side','sides','end','ends',
               'distance','direction','path','road','route','trip','journey',
               'visit','trip','flight','ride','drive','walk','step','steps',
               'every','very','many','such','than','then','also','just','well','here','there',
               'where','when','what','this','that','these','those','which','while','after',
               'before','during','since','until','about','above','below','under','over','into',
               'through','between','among','each','some','much','more','most','other','another',
               'both','few','all','any','none','every','own','same','different','first','last',
               'next','second','third','fourth','fifth','little','young','old','high','low',
               'long','short','great','good','bad','big','small','new','old','right','wrong',
               'early','late','near','far','fast','slow','hard','soft','hot','cold','warm',
               'dry','wet','light','dark','loud','quiet','clean','dirty','rich','poor','safe',
               'full','empty','open','shut','thick','thin','deep','wide','narrow','flat','round',
               'brown','black','white','red','blue','green','yellow','orange','purple','pink',
               'gray','gold','silver','brown','jumps','walks','runs','flies','swims','skips',
               'rocks','blocks','shocks','socks','locks','knocks','clocks','stocks','flocks',
               'works','barks','marks','parks','parts','starts','arts','charts','hearts','tarts',
               'plays','stays','pays','says','days','ways','rays','plays','prays','stays',
               'beautiful','wonderful','terrible','horrible','possible','comfortable',
               'important','different','interesting','dangerous','mysterious','necessary',
               'available','uncomfortable','responsible','impossible','incredible',
               'extremely','absolutely','definitely','probably','certainly','obviously',
               'happily','quickly','slowly','easily','carefully','quietly','loudly','softly',
               'usually','always','never','often','sometimes','rarely','seldom','already',
               'still','yet','even','ever','almost','quite','rather','enough','perhaps',
               'today','tomorrow','yesterday','tonight','morning','afternoon','evening',
               'monday','tuesday','wednesday','thursday','friday','saturday','sunday',
               'january','february','march','april','may','june','july','august',
               'september','october','november','december',
               'people','children','women','men','things','places','times','eyes','hands',
               'words','friends','family','parents','mothers','fathers','brothers','sisters',
               'teachers','students','doctors','officers','countries','cities','towns',
               'games','movies','books','songs','stories','pictures','questions','problems',
               'ideas','money','food','water','coffee','tea','milk','bread','rice','sugar',
               'salt','meat','fruit','apples','oranges','bananas','vegetables','flowers',
               'trees','mountains','rivers','oceans','forests','deserts','islands','beaches',
               'houses','buildings','schools','hospitals','churches','stores','markets',
               'airplanes','buses','trains','boats','bicycles','cars','trucks','taxis',
               'themselves','ourselves','himself','herself','itself','yourself','myself',
               'everyone','someone','anyone','nobody','somebody','anybody','everything',
               'something','anything','nothing','somewhere','anywhere','everywhere',
               'whatever','whoever','whenever','wherever','however',
               'towards','upon','within','without','along','across','behind','beneath',
               'beside','beyond','despite','except','inside','outside','toward',
               'might','must','shall','would','could','should','may','can','will',
               'having','making','taking','getting','going','coming','doing','being',
               'saying','thinking','seeing','knowing','feeling','finding','giving',
               'telling','working','using','asking','trying','seeming','becoming',
               'looking','living','leaving','beginning','keeping','holding','bringing',
               'happening','writing','sitting','standing','setting','reading','learning',
               'understanding','speaking','playing','running','walking','swimming','eating',
               'drinking','sleeping','dancing','singing','driving','flying','teaching',
               'studying','working','helping','showing','hearing','turning','starting',
               'stopping','moving','living','growing','falling','catching','building',
               'sending','spending','selling','buying','paying','meeting','leading',
               'feeding','cutting','putting','costing','hitting','putting',
               'believe','consider','enjoy','expect','hope','imagine','include','indicate',
               'involve','manage','notice','offer','prefer','prepare','present','produce',
               'provide','realize','receive','recognize','recommend','remember','require',
               'serve','suggest','support','suppose','understand','wonder','accept','achieve',
               'allow','apply','approve','arrange','compare','complete','concern','confirm',
               'connect','contain','continue','control','convince','cover','create','decide',
               'develop','disagree','discuss','earn','employ','establish','examine','exist',
               'explain','express','fail','fix','follow','guess','handle','identify','ignore',
               'improve','increase','inform','insist','intend','introduce','investigate',
               'join','judge','maintain','mention','observe','obtain','operate','plan',
               'predict','prepare','prevent','promise','protect','prove','raise','reach',
               'reduce','reflect','reject','relate','release','replace','report','request',
               'require','respond','restrict','reveal','review','save','seek','select','share',
               'sign','stick','strengthen','succeed','suffer','suggest','supply','suit',
               'test','translate','treat','trust','visit','volunteer','warn','waste',
               'weigh','wish','wonder','worry','write','yield',
               'angry','afraid','alone','aware','calm','certain','clever','confident',
               'curious','eager','efficient','elegant','enormous','entire','essential',
               'evident','exact','familiar','famous','favorite','final','flexible','formal',
               'frequent','friendly','funny','gentle','genuine','guilty','honest','huge',
               'ideal','illegal','immediate','impatient','incredible','independent','informal',
               'intelligent','internal','latest','legal','likely','logical','lovely','mature',
               'modern','moral','mutual','narrow','national','native','natural','negative',
               'normal','obvious','occasional','official','opposite','ordinary','patient',
               'perfect','permanent','personal','physical','plain','pleasant','polite',
               'popular','positive','powerful','precious','previous','primary','private',
               'proper','public','pure','rare','rapid','rational','raw','ready','real',
               'reasonable','regular','relative','religious','remote','responsible','rough',
               'royal','rural','severe','sharp','significant','silent','silly','slight',
               'smooth','social','solid','special','steady','strange','strict','strong',
               'substantial','successful','sudden','sufficient','suitable','superior',
               'sure','surprise','tall','tender','terrible','tight','total','tough','tremendous',
               'typical','ultimate','unique','unlikely','unusual','upset','urban','useful',
               'usual','valuable','vast','violent','visible','vital','vivid','western',
               'western','western','wooden','worthwhile',
               'across','against','among','around','before','behind','below','beneath',
               'beside','besides','between','beyond','during','inside','outside','toward',
               'towards','throughout','underneath','unlike','until','upon','within','without',
               'although','because','besides','however','moreover','nevertheless','otherwise',
               'therefore','though','unless','whereas','wherever','whenever','whoever',
               'neither','either','both','whether','if','unless','provided','supposing'])
    for base, forms in IRREGULAR_VERBS.items():
        vw.add(base)
        for form in forms.values():
            vw.add(form)
    return vw

VALID_WORDS = _build_valid_words()

EDIT_DISTANCE_CACHE = {}

def edit_distance(s1, s2):
    key = (s1, s2)
    if key in EDIT_DISTANCE_CACHE:
        return EDIT_DISTANCE_CACHE[key]
    if len(s1) > 30 or len(s2) > 30:
        return max(len(s1), len(s2))
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    result = dp[m][n]
    EDIT_DISTANCE_CACHE[key] = result
    return result

def generate_candidates(word, max_distance=2):
    candidates = set()
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    w = word.lower()
    for i in range(len(w)):
        for c in alphabet:
            candidate = w[:i] + c + w[i+1:]
            if candidate != w and candidate in COMMON_WORDS:
                candidates.add(candidate)
        for c in alphabet:
            candidate = w[:i] + c + w[i:]
            if candidate in COMMON_WORDS:
                candidates.add(candidate)
    for i in range(len(w)):
        if i + 1 < len(w):
            candidate = w[:i] + w[i+1] + w[i] + w[i+2:]
            if candidate != w and candidate in COMMON_WORDS:
                candidates.add(candidate)
    for i in range(len(w)):
        candidate = w[:i] + w[i+1:]
        if candidate and candidate in COMMON_WORDS:
            candidates.add(candidate)
    for i in range(len(w) + 1):
        for c in alphabet:
            candidate = w[:i] + c + w[i:]
            if candidate in COMMON_WORDS:
                candidates.add(candidate)
    for i in range(len(w)):
        for c in alphabet:
            candidate = w[:i] + c + w[i+1:]
            if candidate != w and candidate in COMMON_WORDS:
                candidates.add(candidate)
    return candidates

SPELLING_DICT = {}
SPELLING_LOADED = False

def load_spelling_dict():
    global SPELLING_DICT, SPELLING_LOADED
    if SPELLING_LOADED:
        return
    import json, os
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    try:
        with open(os.path.join(data_dir, 'spelling_dictionary.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        SPELLING_DICT.update(data.get('common_misspellings', {}))
    except Exception:
        pass
    try:
        with open(os.path.join(data_dir, 'common_mistakes.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data.get('common_spelling_mistakes', []):
            if 'word' in entry and 'correct' in entry:
                SPELLING_DICT[entry['word'].lower()] = entry['correct'].lower()
    except Exception:
        pass
    SPELLING_LOADED = True

CONFUSING_PAIRS = {
    'their': {'correct_context': ['there', "they're"], 'type': 'homophone'},
    'there': {'correct_context': ['their', "they're"], 'type': 'homophone'},
    "they're": {'correct_context': ['their', 'there'], 'type': 'homophone'},
    'your': {'correct_context': ["you're"], 'type': 'homophone'},
    "you're": {'correct_context': ['your'], 'type': 'homophone'},
    'its': {'correct_context': ["it's"], 'type': 'homophone'},
    "it's": {'correct_context': ['its'], 'type': 'homophone'},
    'to': {'correct_context': ['too', 'two'], 'type': 'homophone'},
    'too': {'correct_context': ['to', 'two'], 'type': 'homophone'},
    'than': {'correct_context': ['then'], 'type': 'confusing'},
    'then': {'correct_context': ['than'], 'type': 'confusing'},
    'affect': {'correct_context': ['effect'], 'type': 'confusing'},
    'effect': {'correct_context': ['affect'], 'type': 'confusing'},
    'accept': {'correct_context': ['except'], 'type': 'confusing'},
    'except': {'correct_context': ['accept'], 'type': 'confusing'},
    'loose': {'correct_context': ['lose'], 'type': 'confusing'},
    'lose': {'correct_context': ['loose'], 'type': 'confusing'},
    'passed': {'correct_context': ['past'], 'type': 'confusing'},
    'past': {'correct_context': ['passed'], 'type': 'confusing'},
    'weather': {'correct_context': ['whether'], 'type': 'confusing'},
    'whether': {'correct_context': ['weather'], 'type': 'confusing'},
    'advice': {'correct_context': ['advise'], 'type': 'confusing'},
    'advise': {'correct_context': ['advice'], 'type': 'confusing'},
    'practice': {'correct_context': ['practise'], 'type': 'confusing'},
    'practise': {'correct_context': ['practice'], 'type': 'confusing'},
    'principal': {'correct_context': ['principle'], 'type': 'confusing'},
    'principle': {'correct_context': ['principal'], 'type': 'confusing'},
    'stationary': {'correct_context': ['stationery'], 'type': 'confusing'},
    'stationery': {'correct_context': ['stationary'], 'type': 'confusing'},
    'compliment': {'correct_context': ['complement'], 'type': 'confusing'},
    'complement': {'correct_context': ['compliment'], 'type': 'confusing'},
    'emigrate': {'correct_context': ['immigrate'], 'type': 'confusing'},
    'immigrate': {'correct_context': ['emigrate'], 'type': 'confusing'},
    'fewer': {'correct_context': ['less'], 'type': 'usage'},
    'less': {'correct_context': ['fewer'], 'type': 'usage'},
    'lie': {'correct_context': ['lay'], 'type': 'confusing'},
    'lay': {'correct_context': ['lie'], 'type': 'confusing'},
}

def check_spelling(text, doc):
    errors = []
    load_spelling_dict()
    for sent in doc.sentences:
        for token in sent.words:
            w = token.lower
            if len(w) <= 2:
                continue
            if w in VALID_WORDS:
                continue
            if w in SPELLING_DICT:
                suggestion = SPELLING_DICT[w]
                if suggestion != w:
                    errors.append({
                        'type': 'spelling',
                        'severity': 'error',
                        'category': 'Spelling',
                        'incorrect': token.text,
                        'correction': suggestion,
                        'position': token.start,
                        'end_position': token.end,
                        'sentence': sent.text,
                        'message': f'"{token.text}" is misspelled. Did you mean "{suggestion}"?',
                        'confidence': 95,
                        'pass': 'spelling',
                    })
            else:
                candidates = []
                for c in generate_candidates(w, max_distance=2):
                    if c in VALID_WORDS:
                        d = edit_distance(w, c)
                        if d <= 1:
                            candidates.append((c, d))
                if len(w) <= 4:
                    candidates = []
                candidates.sort(key=lambda x: x[1])
                if candidates:
                    best = candidates[0]
                    errors.append({
                        'type': 'spelling',
                        'severity': 'warning',
                        'category': 'Spelling',
                        'incorrect': token.text,
                        'correction': best[0],
                        'position': token.start,
                        'end_position': token.end,
                        'sentence': sent.text,
                        'message': f'"{token.text}" may be misspelled. Did you mean "{best[0]}"?',
                        'confidence': max(60, 95 - best[1] * 15),
                        'pass': 'spelling',
                    })
    return errors
