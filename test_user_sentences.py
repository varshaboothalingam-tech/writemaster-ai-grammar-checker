"""Test all user sentences - check detection rate."""
import sys
sys.path.insert(0, ".")
from unified_pipeline import UnifiedPipeline

p = UnifiedPipeline()

# Each sentence has (sentence, expected_detection_count_min, description)
sentences = [
    ("Yesterday my friend and me goes to the market and buyed some apples.", 2, "pronoun_obj + SVA + past_tense"),
    ("She don't likes coffee because it make her stomach painful.", 2, "do_support + SVA"),
    ("I have went to Chennai last week and I seen many beautiful places.", 2, "wrong_pp twice"),
    ("There is many people waiting outside the shop.", 1, "existential_sva"),
    ("He have completed his work but he don't submitted the report.", 2, "SVA + do_support"),
    ("I am working here since three years.", 1, "since_for_duration"),
    ("She is more smarter than her sister.", 1, "comparative"),
    ("He is good in mathematics but bad in English.", 1, "preposition"),
    ("I didn't knew that they was coming today.", 2, "didnt_past + SVA"),
    ("The informations given by him were not correct.", 1, "uncountable_plural"),
    ("She gave me many useful advices yesterday.", 1, "uncountable_plural"),
    ("I need an umbrella because its raining outside.", 1, "its_its"),
    ("Your going to loose you're phone if you keep putting it there.", 2, "contraction + confusing"),
    ("Their going to bring they're books tomorrow.", 2, "contraction + confusing"),
    ("The company has hired new employee's for it's marketing team.", 2, "apostrophe + its"),
    ("He said that he will came tomorrow.", 1, "tense"),
    ("She asked me where was I going.", 1, "word_order"),
    ("I don't know what does he wants.", 1, "do_support"),
    ("Because I was tired. I went to sleep early.", 1, "fragment"),
    ("When I reached the station the train already left.", 1, "missing_comma"),
    ("I was walking to office when it started raining heavily and I forgot my umbrella.", 1, "preposition"),
    ("She likes reading writing and painting.", 1, "missing_comma"),
    ("My brother is tall, intelligent and he works very hard.", 1, "parallelism"),
    ("He bought a new laptop yesterday, it was very expensive.", 1, "comma_splice"),
    ("I have two sister and both of them lives in Bangalore.", 2, "plural + SVA"),
    ("Neither of the students have completed their assignment.", 1, "SVA"),
    ("Everyone should bring their own laptop.", 0, "CORRECT - skip"),
    ("Each students must submit the form before Friday.", 1, "every_singular"),
    ("There were a problem with the computer.", 1, "SVA"),
    ("The news are very surprising.", 1, "SVA"),
    ("Mathematics are difficult for many students.", 1, "SVA"),
    ("She can sings very well.", 1, "modal_wrong_form"),
    ("He must to finish the work today.", 1, "modal_wrong_form"),
    ("You should went to the doctor yesterday.", 1, "modal_wrong_form"),
    ("I am agree with your opinion.", 1, "adjective_form"),
    ("I am knowing the answer.", 1, "stative_be"),
    ("She is having two cars.", 1, "stative_be"),
    ("He explained me the problem.", 1, "explain_dative"),
    ("Please discuss about this issue with your manager.", 1, "preposition"),
    ("We reached at the airport at 6 AM.", 1, "preposition"),
    ("He entered into the room quietly.", 1, "preposition"),
    ("She married with a doctor last year.", 1, "preposition"),
    ("I am waiting your response.", 1, "missing_preposition"),
    ("He is responsible of managing the project.", 1, "preposition"),
    ("This is different than what you told me.", 1, "preposition"),
    ("She is married with a software engineer.", 1, "preposition"),
    ("He is interested to learn Python.", 1, "preposition"),
    ("I look forward to meet you.", 1, "gerund_infinitive"),
    ("She suggested me to take a break.", 1, "dative"),
    ("He made me to laugh.", 1, "causative"),
    ("I enjoyed to watch the movie.", 1, "gerund_infinitive"),
    ("I want going home now.", 1, "gerund_infinitive"),
    ("The project was completed by the team yesterday.", 0, "CORRECT - skip"),
    ("The report has been send to the manager.", 1, "wrong_pp"),
    ("The documents were not submitted on time.", 0, "CORRECT - skip"),
    ("He don't never listen to me.", 1, "double_negative"),
    ("I didn't see nobody at the office.", 1, "double_negative"),
    ("She hardly never goes outside.", 1, "double_negative"),
    ("I have never saw such a beautiful place.", 1, "wrong_pp"),
    ("He has went there many times.", 1, "wrong_pp"),
    ("She had ate before we arrived.", 1, "wrong_pp"),
    ("They was playing football when I called them.", 1, "SVA"),
    ("We were discuss the problem yesterday.", 1, "wrong_form"),
    ("He is work in this company since 2022.", 1, "tense/stative"),
    ("She have been studying English for two years.", 1, "SVA"),
    ("I will going to Chennai tomorrow.", 1, "future_form"),
    ("He will comes with us.", 1, "modal_wrong_form"),
    ("They are going visit their parents next week.", 1, "missing_to"),
    ("She can able to solve the problem.", 1, "modal + able"),
    ("He is one of the best employee in the company.", 1, "one_of_plural"),
    ("This is one of the most easiest questions.", 1, "double_superlative"),
    ("It is too much hot today.", 1, "too_much"),
    ("I have less books than my brother.", 1, "less_fewer"),
    ("There are much people in the room.", 1, "much_many"),
    ("I need few water.", 1, "uncountable"),
    ("She has a good knowledge about computers.", 1, "preposition"),
    ("He gave me an useful information.", 2, "a_an + uncountable"),
    ("I saw a elephant near the road.", 1, "a_an"),
    ("She is a honest person.", 1, "a_an"),
    ("He bought an university book.", 1, "a_an"),
    ("I went to the office on Monday morning.", 0, "CORRECT - skip"),
    ("We will meet in next week.", 1, "preposition"),
    ("She was born on 2001.", 1, "preposition"),
    ("He arrived to the office at 9 AM.", 1, "preposition"),
    ("I have been living here from five years.", 1, "since_for"),
    ("I will call you after I will finish my work.", 1, "tense"),
    ("If I will get time, I will visit you.", 1, "conditional"),
    ("If I knew about the meeting, I would attend it.", 0, "CORRECT - skip"),
    ("If she studied harder, she would have passed the exam.", 0, "CORRECT - skip"),
    ("He asked me that whether I was ready.", 1, "redundant_that"),
    ("She told that she was tired.", 1, "missing_object"),
    ("He said me that he would come later.", 1, "dative"),
    ("My manager told me to completing the report.", 1, "infinitive"),
    ("The teacher explained us the lesson.", 1, "dative"),
    ("She made a mistake because she was not careful enough.", 0, "CORRECT - skip"),
    ("The reason is because he was late.", 1, "redundant_because"),
    ("Although he was tired, but he continued working.", 1, "although_but"),
    ("Despite of being tired, she finished the work.", 1, "despite_of"),
    ("In spite he was sick, he went to work.", 1, "in_spite_of"),
    ("He likes coffee more than tea is.", 1, "extra_verb"),
    ("The book which I bought it yesterday is very interesting.", 1, "redundant_pronoun"),
    ("The person who he called me was my manager.", 1, "redundant_pronoun"),
    ("This is the place where I was born there.", 1, "redundant_adverb"),
    ("She is the girl which won the competition.", 1, "which_who"),
    ("I don't know the man who you are talking about him.", 1, "redundant_pronoun"),
    ("He speaks English very good.", 1, "adverb_form"),
    ("She sings beautiful.", 1, "adverb_form"),
    ("He runs very quick.", 1, "adverb_form"),
    ("She completed the task successful.", 1, "adverb_form"),
    ("The manager spoke angry to the employee.", 1, "adverb_form"),
    ("He drove careful because the road was wet.", 1, "adverb_form"),
    ("She did the work very perfect.", 1, "adverb_form"),
    ("I have a meeting tommorow.", 1, "spelling"),
    ("The enviroment is very important.", 1, "spelling"),
    ("This is definately the best option.", 1, "spelling"),
    ("I recieved your email yesterday.", 1, "spelling"),
    ("Please seperate these documents.", 1, "spelling"),
    ("The accomodation was expensive.", 1, "spelling"),
    ("The goverment announced a new policy.", 1, "spelling"),
    ("I definately agree with you.", 1, "spelling"),
    ("The recieve button is not working.", 1, "spelling"),
    ("She has alot of experience.", 1, "spelling"),
    ("I dont know what happend.", 2, "spelling + apostrophe"),
    ("He didnt come yesterday.", 1, "apostrophe"),
    ("Its very difficult to explain.", 1, "its_it's"),
    ("Your doing a great job.", 1, "your_you're"),
    ("Theyre waiting outside.", 1, "apostrophe"),
    ("I cant find my phone.", 1, "apostrophe"),
    ("We shouldnt ignore this problem.", 1, "apostrophe"),
    ('She said "I am tired".', 0, "CORRECT - skip"),
    ('He asked "where are you going?"', 0, "CORRECT - skip"),
    ("My friend, who lives in Chennai is a designer.", 1, "missing_comma"),
    ("However I decided to continue working.", 1, "missing_comma"),
    ("First we need to collect the data, then we need to analyse it.", 1, "semicolon"),
    ("I bought apples oranges bananas and grapes.", 1, "missing_comma"),
    ("After finishing the work he went home.", 1, "missing_comma"),
    ("Having completed the project, the manager approved it.", 0, "CORRECT - skip"),
    ("The report was prepared by John, and it was sent to Sarah.", 0, "CORRECT - skip"),
    ("She likes cooking, dancing and to sing.", 1, "parallelism"),
    ("He enjoys reading, writing and playing football.", 0, "CORRECT - skip"),
    ("The company wants employees who are hardworking, honest and have good communication skills.", 0, "CORRECT - skip"),
    ("I went to the market and I bought some vegetables and then I came back home.", 0, "CORRECT - skip"),
    ("Yesterday, I was going to the office, and I was meeting my friend, and we were discussing the project, and then we went to lunch.", 1, "comma_splice/repetitive_and"),
    ("The product which we launched last month it has received many positive reviews.", 1, "redundant_pronoun"),
    ("He told me that he will send the document yesterday.", 1, "tense"),
    ("I didn't understood why she was angry with me.", 1, "wrong_pp"),
    ("She don't wanted to attend the meeting.", 1, "do_support"),
    ("They has been working on this project since last year.", 1, "SVA"),
    ("There was several issues that needed to be fixed.", 1, "SVA"),
    ("Every employees need to follow the company policy.", 1, "every_singular"),
    ("Neither John nor his friends was available.", 1, "SVA"),
    ("The team are working on a new project.", 0, "CORRECT - skip"),
    ("The data shows that customers is leaving the platform.", 1, "SVA"),
    ("The results indicates that the strategy are not working.", 2, "SVA x2"),
    ("One of my friend have moved to Mumbai.", 2, "one_of_plural + SVA"),
    ("My parents lives in a small village.", 1, "SVA"),
    ("She usually go to work by bus.", 1, "SVA"),
    ("He rarely eat breakfast.", 1, "SVA"),
    ("I always am checking my email in the morning.", 1, "word_order"),
    ("Never I have seen such a beautiful place.", 1, "word_order"),
    ("Only after the meeting I understood the problem.", 1, "word_order"),
    ("Not only he was late, but he also forgot the documents.", 1, "word_order"),
    ("What I want is to improve my English skills.", 0, "CORRECT - skip"),
    ("What I want is improving my English skills.", 1, "gerund_infinitive"),
    ("It is important that he attends the meeting.", 0, "CORRECT - skip"),
    ("It is important that he attend the meeting.", 0, "CORRECT - skip"),
    ("I wish I was knowing the answer.", 1, "stative"),
    ("I wish I had knew about it earlier.", 1, "wrong_pp"),
]

correct_sentences = [(s, d) for s, m, d in sentences if m == 0]
error_sentences = [(s, m, d) for s, m, d in sentences if m > 0]

print(f"Total sentences: {len(sentences)}")
print(f"Error sentences: {len(error_sentences)}")
print(f"Correct sentences (should have 0 detections): {len(correct_sentences)}")
print()

# Test error sentences
detected = 0
missed = 0
missed_list = []
for sent, min_expected, desc in error_sentences:
    results = p.check(sent)
    if len(results) >= min_expected:
        detected += 1
        print(f"  DETECTED ({len(results)}): {desc}")
        print(f"    \"{sent[:80]}\"")
    elif len(results) > 0:
        detected += 1
        print(f"  PARTIAL ({len(results)}/{min_expected}): {desc}")
        print(f"    \"{sent[:80]}\"")
        for r in results:
            print(f"      {r['rule_id']}: {r['original']} -> {r['replacement']}")
    else:
        missed += 1
        missed_list.append((sent, desc))
        print(f"  MISSED: {desc}")
        print(f"    \"{sent[:80]}\"")

print()
print(f"=== ERROR DETECTION ===")
print(f"Detected: {detected}/{len(error_sentences)} ({detected/len(error_sentences)*100:.1f}%)")
print(f"Missed: {missed}/{len(error_sentences)} ({missed/len(error_sentences)*100:.1f}%)")
print()

# Test correct sentences
fps = 0
for sent, desc in correct_sentences:
    results = p.check(sent)
    if results:
        fps += 1
        print(f"  FP: {desc}: \"{sent[:80]}\"")
        for r in results:
            print(f"    {r['rule_id']}: {r['original']} -> {r['replacement']}")

print(f"\n=== CORRECT SENTENCE CHECK ===")
print(f"False positives: {fps}/{len(correct_sentences)}")

print(f"\n=== MISSED ERRORS (for fixing) ===")
for sent, desc in missed_list:
    print(f"  {desc}: \"{sent}\"")
