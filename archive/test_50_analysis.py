"""Detailed analysis of what we miss on each sentence."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

# Expected errors per sentence (manually annotated)
expected = {
    1: [("goes", "went", "TENSE"), ("needs", "need", "SVA")],
    2: [("don't likes", "doesn't like", "DO_SUPPORT"), ("keeps", "keep", "SVA")],
    3: [("was planning", "were planning", "SVA"), ("need", "needs", "SVA")],
    4: [("is many", "are many", "SVA"), ("prefers", "prefer", "SVA")],
    5: [("makes", "make", "SVA")],
    6: [("already left", "had already left", "TENSE"), ("to waited", "to wait", "PAST_PART")],
    7: [("where was I", "where I was", "EMBEDDED_Q"), ("didn't knew", "didn't know", "DIDNT_PAST")],
    8: [("are working", "is working", "SVA"), ("provide", "provides", "SVA")],
    9: [("since five years", "for five years", "SINCE_FOR")],
    10: [("don't receive", "doesn't receive", "DO_SUPPORT"), ("employee", "employees", "ONE_OF_PLURAL")],
    11: [("nobody were", "nobody was", "SVA")],
    12: [("were very useful", "was very useful", "SVA"), ("was incorrect", "were incorrect", "SVA")],
    13: [("didn't went", "didn't go", "DIDNT_PAST")],
    14: [("would have known", "had known", "DOUBLE_MODAL"), ("will have", "would have", "TENSE")],
    15: [("Although...but", "remove but", "REDUNDANT")],
    16: [("has not submitted it", "have not submitted them", "SVA")],
    17: [("explained me", "explained to me", "PREPOSITION"), ("how does the system works", "how the system works", "INVERSION")],
    18: [("was", "were (traffic)", "SVA")],
    19: [("interested on", "interested in", "PREPOSITION")],
    20: [("works", "work (uncountable)", "UNCOUNTABLE")],
    21: [("has launched...last month", "launched", "TENSE"), ("feedbacks", "feedback", "PLURAL_UNCOUNT")],
    22: [("was aware", "were aware", "SVA")],
    23: [("Every students", "Every student", "ONE_OF_PLURAL")],
    24: [("", "", "OK - tense shift maybe")],
    25: [("am not agree", "do not agree", "STATIVE_BE")],
    26: [("from 2021", "since 2021", "SINCE_FOR")],
    27: [("recommended us", "recommended to us", "PREPOSITION"), ("food were", "food was", "SVA")],
    28: [("uses", "use", "SVA"), ("have increased", "has increased", "SVA")],
    29: [("than", "to", "COMPARISON")],
    30: [("suggested me", "suggested that I", "PREPOSITION")],
    31: [("discussing about", "discussing", "PREPOSITION")],
    32: [("is too many", "are too many", "SVA")],
    33: [("didn't told", "didn't tell", "DIDNT_PAST"), ("gets", "get", "SVA")],
    34: [("can speak", "could speak", "TENSE"), ("would helps", "would help", "MODAL"), ("am not agree", "do not agree", "STATIVE_BE")],
    35: [("have already checked", "had already checked", "TENSE")],
    36: [("informations", "information", "SPELLING")],
    37: [("hardly", "hard", "WORD_USAGE")],
    38: [("needs", "need", "SVA"), ("can starts", "can start", "MODAL"), ("starts", "start", "PARALLEL")],
    39: [("none", "any", "PRONOUN")],
    40: [("rather than to publish", "rather than publishing", "PARALLEL")],
    41: [("have responded", "has responded", "SVA")],
    42: [("didn't understood", "didn't understand", "DIDNT_PAST")],
    43: [("less", "fewer", "LESS_FEWER")],
    44: [("studied", "had studied", "MIXED_CONDITIONAL")],
    45: [],
    46: [("didn't expected", "didn't expect", "DIDNT_PAST")],
    47: [("every days", "every day", "PLURAL")],
    48: [("was making", "were making", "SVA")],
    49: [("have suggested", "has suggested", "SVA")],
    50: [("had forget", "had forgotten", "PAST_PART"), ("had to called", "had to call", "PAST_PART")],
}

sentences = [
    "Yesterday I goes to the market with my brother because we needs to buy some vegetables for dinner.",
    "She don't likes coffee, but she drinks it every morning because her colleagues keeps offering it to her.",
    "My parents was planning to visit us next weekend, but they have changed their plan because my father need to attend a meeting.",
    "There is many reasons why people prefers working from home instead of travelling to office every day.",
    "I have completed the assignment yesterday, but my manager says that I still need to makes some changes.",
    "When we reached the station, the train already left, so we had to waited for another one.",
    "He asked me where was I going, but I didn't knew how to explain the situation properly.",
    "One of my friends are working in a company which provide software solutions for small businesses.",
    "I am living in this city since five years, and I still haven't visited many important places.",
    "She is one of the best employee in our department, but she don't receive enough appreciation from the management.",
    "We discussed about the problem for almost two hours, but nobody were able to find a proper solution.",
    "The information that you provided yesterday were very useful, but some of the details was incorrect.",
    "I didn't went to the office because I was feeling very tired and I had a severe headache since morning.",
    "If I would have known about the meeting earlier, I will have prepared the presentation properly.",
    "Although he was very tired, but he continued working until the project was completed.",
    "The manager asked everyone to submit their reports before Friday, but several employees has not submitted it yet.",
    "She explained me the process very clearly, but I was still unable to understand how does the system works.",
    "There were a lot of traffic on the road, so we arrived at the airport much later than we expected.",
    "My brother is good in mathematics, but he is not very interested on learning programming.",
    "I have many works to complete today, so I probably won't be able to attend the meeting.",
    "The company has launched a new product last month and it is already receiving many positive feedbacks from customers.",
    "Neither the manager nor the employees was aware that the system had stopped working.",
    "Every students in the class were asked to submit their assignments before the end of the week.",
    "She said that she will call me when she reaches home, but she never called.",
    "I am not agree with the decision because it can create several problems in the future.",
    "He has been working here from 2021, but he is planning to leave the company next year.",
    "We went to the restaurant which you recommended us, but the food were not as good as we expected.",
    "The number of people who uses this application have increased significantly during the last few months.",
    "I prefer working from home than travelling to the office because it saves more time.",
    "She suggested me to apply for the position because she thinks I am suitable for the job.",
    "After finishing the meeting, everyone started discussing about the new project and its requirements.",
    "The website loads very slowly because there is too many images and unnecessary scripts on the page.",
    "He didn't told anyone about the problem because he was afraid that his manager might gets angry.",
    "I wish I can speak English more fluently because it would helps me communicate with international clients.",
    "By the time we arrived at the hotel, our friends have already checked in and went to their rooms.",
    "The report contains several important informations that should be reviewed before making a final decision.",
    "She works very hardly because she wants to get promoted before the end of this year.",
    "The new employees needs to complete their training before they can starts working on real projects.",
    "I was surprised to hear that he has resigned from the company without informing none of his colleagues.",
    "We should focus on improving the quality of the content rather than to publish more articles every week.",
    "The customer complained that nobody have responded to his email for more than three days.",
    "He told me that he didn't understood the instructions because they were written in a very complicated way.",
    "There are less opportunities available for fresh graduates than there was a few years ago.",
    "If she studied harder, she would have passed the examination last month.",
    "The team completed the project successfully despite facing many unexpected problems during the development process.",
    "I didn't expected the meeting to take so long, so I haven't brought anything to eat.",
    "My sister bought a new laptop because her old one was becoming slower and slower every days.",
    "The teacher asked the students why they was making so much noise during the examination.",
    "We have discussed this issue many times before, but nobody have suggested a solution that actually works.",
    "When I was reaching home yesterday, I realized that I had forget my keys at the office and had to called my colleague for help.",
]

detected_rules = []
for i, s in enumerate(sentences, 1):
    issues = p.check(s)
    detected_rules.append(set(e['rule_id'] for e in issues))

# Analyze
total_expected = 0
total_found = 0
total_missed = 0
missed_patterns = {}

for i, (exp, det) in enumerate(zip(expected.values(), detected_rules), 1):
    if not exp:
        continue
    for orig, fix, cat in exp:
        total_expected += 1
        # Check if we found something related
        found = False
        for issue in issues:
            if orig.lower() in s.lower():
                found = True
                break
        if not found:
            total_missed += 1
            if cat not in missed_patterns:
                missed_patterns[cat] = []
            missed_patterns[cat].append(f"S{i}: {orig} -> {fix}")

print(f"Expected errors: {total_expected}")
print(f"Total detected across all: 39")
print(f"\nMissed patterns by category:")
for cat, items in sorted(missed_patterns.items()):
    print(f"\n  {cat} ({len(items)}):")
    for item in items:
        print(f"    {item}")
