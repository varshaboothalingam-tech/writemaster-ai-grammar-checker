import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

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

total_issues = 0
for i, s in enumerate(sentences, 1):
    issues = p.check(s)
    total_issues += len(issues)
    if issues:
        rules = ", ".join(e['rule_id'] for e in issues)
        print(f"{i:2d}. [{len(issues)}] {rules}")
    else:
        print(f"{i:2d}. [0] ---")

print(f"\nTotal: {total_issues} issues across {len(sentences)} sentences")
