import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

text = open(r'C:\Users\hemal\OneDrive\Documents\grammar_checker\test_writing.txt', encoding='utf-8').read() if False else ""

text = """Last Monday, I was planning to go to office early because my manager have scheduled an important meeting with all the team members. I reached the bus stop at around eight o'clock, but there was too many peoples waiting for the bus. The bus finally arrived after twenty minutes, and when I entered inside, I realized that I had forgot my wallet at home. I tried calling my brother, but he didn't answered the phone because he was sleeping. By the time I reached the office, the meeting already started and everyone were waiting for me. My manager asked me why I was late, and I explained him what had happened. He was not very happy, but he said that I should be more careful in future. Yesterday, my friend and me decided to visit a new restaurant which recently opened near our house. We have heard many good reviews about the food, so we was expecting it to be very good. When we arrived at the restaurant, there were a long queue outside because the place was already crowded. After waiting for almost forty minutes, we finally got a table. The waiter gave us the menu and asked what would we like to order. My friend ordered a chicken biryani while I decided to try the special noodles. The food were served after twenty minutes, but the noodles was too spicy for me. We asked the waiter if he can bring some water, but he didn't came back for several minutes. In the end, we paid the bill and left the restaurant feeling a little disappointed. Our company has recently launched a new software product which is designed for helping small businesses manage their customer data. The development team have worked on it since almost one year, and they has spent many hours testing different features. Although the product looks very promising, there are still several problems which needs to be fixed before it can be released to the public. Some customers has reported that the application loads very slowly when they uploads large files. The developers are currently working on solving these issues, but they said that it may takes another few weeks. The marketing team have already prepared several articles and advertisements, so everybody are waiting for the final release. If the technical problems will be solved soon, the company expects that the product will becomes very popular among small businesses. Last weekend, I visited my grandparents house because they was celebrating their wedding anniversary. My cousins were also there, and everyone were helping to prepare the food. My grandmother asked me to brings some vegetables from the market, but I didn't had enough money with me. So, my cousin gave me some money and told me that I can return it later. After buying the vegetables, I came back home and realized that I had buy the wrong type of tomatoes. My grandmother laughed and said that it was not a big problem, but she asked me to be more carefully next time. In the evening, we all sat together and talked about our childhood memories. It was one of the most happiest days I have experienced in a long time. I recently started learning about SEO because I want to improve my skills and become a better digital marketer. At first, I thought SEO was very easy, but after studying it for few weeks, I realized that there are much more things to learn than I expected. There is many technical factors that can affect a websites ranking, including page speed, internal linking, structured data and backlinks. I also learned that creating good content are not enough because search engines looks at many other signals before deciding how a page should rank. Sometimes I spends several hours analyzing competitors websites, but I still don't understands why some pages performs better than others. My goal is to build a tool which can automatically identify these problems and gives useful recommendations to website owners."""

issues = p.check(text)

# Group by approximate sentence position
print(f"=== DETECTED {len(issues)} ISSUES ===\n")
for e in sorted(issues, key=lambda x: x['start']):
    # Find word context
    s = max(0, e['start'] - 20)
    fn = min(len(text), e['end'] + 20)
    ctx = text[s:fn].replace('\n', ' ')
    repl = f'\"{e["original"]}\" -> \"{e["replacement"]}\"' if e["original"] else f'INSERT \"{e["replacement"]}\"'
    print(f"  [{e['start']:4d}] {repl:50s} [{e['rule_id']:25s}] conf={e['confidence']:.2f}")
    print(f"         ...{ctx}...")

# Now list ALL expected errors the pipeline missed
print(f"\n=== EXPECTED ERRORS THE PIPELINE MISSED ===\n")
missed = [
    ("peoples", "people", "PLURAL_NOUN", "Paragraph 1: 'too many peoples'"),
    ("entered inside", "entered", "REDUNDANT_WORD", "Paragraph 1: 'entered inside' redundant"),
    ("had forgot", "had forgotten", "PAST_PARTICIPLE", "Paragraph 1: 'had forgot' needs past participle"),
    ("didn't answered", "didn't answer", "DIDNT_PAST_FORM", "Paragraph 1: 'didn't answered'"),
    ("everyone were", "everyone was", "SVA", "Paragraph 1: 'everyone were waiting'"),
    ("explained him", "explained to him", "EXPLAIN_DATIVE", "Paragraph 1: 'explained him' (detected but wrong offset)"),
    ("in future", "in the future", "MISSING_ARTICLE", "Paragraph 1: 'in future'"),
    ("we was expecting", "we were expecting", "SVA", "Paragraph 2: 'we was expecting'"),
    ("what would we like", "what we would like", "EMBEDDED_Q", "Paragraph 2: embedded question inversion"),
    ("food were served", "food was served", "SVA", "Paragraph 2: 'food were served'"),
    ("noodles was too spicy", "noodles were too spicy", "SVA", "Paragraph 2: 'noodles was'"),
    ("if he can bring", "if he could bring", "TENSE", "Paragraph 2: modal in reported speech"),
    ("didn't came back", "didn't come back", "DIDNT_PAST_FORM", "Paragraph 2: 'didn't came'"),
    ("since almost one year", "for almost one year", "SINCE_FOR_DURATION", "Paragraph 3: 'since' with duration"),
    ("they has spent", "they have spent", "SVA", "Paragraph 3: 'they has spent'"),
    ("which needs", "that need", "SVA", "Paragraph 3: 'problems which needs'"),
    ("customers has reported", "customers have reported", "SVA", "Paragraph 3: 'customers has reported'"),
    ("they uploads", "they upload", "SVA", "Paragraph 3: 'they uploads'"),
    ("it may takes", "it may take", "MODAL_WRONG_FORM", "Paragraph 3: 'may takes'"),
    ("everybody are waiting", "everybody is waiting", "SVA", "Paragraph 3: 'everybody are'"),
    ("will becomes", "will become", "MODAL_WRONG_FORM", "Paragraph 3: 'will becomes'"),
    ("grandparents house", "grandparents' house", "POSSESSIVE_APOSTROPHE", "Paragraph 4: missing possessive"),
    ("they was celebrating", "they were celebrating", "SVA", "Paragraph 4: 'they was'"),
    ("everyone were helping", "everyone was helping", "SVA", "Paragraph 4: 'everyone were'"),
    ("to brings", "to bring", "BASE_FORM", "Paragraph 4: 'to brings'"),
    ("didn't had enough", "didn't have enough", "DIDNT_PAST_FORM", "Paragraph 4: 'didn't had'"),
    ("had buy", "had bought", "PAST_PARTICIPLE", "Paragraph 4: 'had buy'"),
    ("more carefully", "more careful", "ADJ_ADV", "Paragraph 4: 'more carefully' (adjective needed)"),
    ("for few weeks", "for a few weeks", "MISSING_ARTICLE", "Paragraph 5: 'for few weeks'"),
    ("a websites ranking", "a website's ranking", "POSSESSIVE_APOSTROPHE", "Paragraph 5: missing possessive"),
    ("content are not enough", "content is not enough", "SVA", "Paragraph 5: 'content are'"),
    ("search engines looks", "search engines look", "SVA", "Paragraph 5: 'engines looks'"),
    ("competitors websites", "competitors' websites", "POSSESSIVE_APOSTROPHE", "Paragraph 5: missing possessive"),
    ("gives useful", "give useful", "SVA", "Paragraph 5: 'problems and gives'"),
]

for orig, repl, rule, ctx in missed:
    print(f"  MISSED: \"{orig}\" -> \"{repl}\" [{rule}] — {ctx}")
