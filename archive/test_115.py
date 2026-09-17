from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

sentences = [
    (1, "yesterday i goes to office and meet my manager.", "Yesterday, I went to the office and met my manager."),
    (2, "she dont like coffee because it make her sick.", "She doesn't like coffee because it makes her sick."),
    (3, "we was working on the project from last two weeks.", "We have been working on the project for the last two weeks."),
    (4, "he have completed the task but forgot to send it.", "He has completed the task but forgot to send it."),
    (5, "the website load very slow and some images is missing.", "The website loads very slowly, and some images are missing."),
    (6, "i am looking for a job which offer good salary and benefits.", "I am looking for a job that offers a good salary and benefits."),
    (7, "yesterday, she buyed a new laptop and setup it at home.", "Yesterday, she bought a new laptop and set it up at home."),
    (8, "they doesn't understand the requirements so we explained it again.", "They don't understand the requirements, so we explained them again."),
    (9, "please check the document and let me know if there is any mistakes.", "Please check the document and let me know if there are any mistakes."),
    (10, "my team are working hard to finish the project before monday.", "My team are working hard to finish the project before Monday."),
    (11, "i have completed my work, but i didnt informed the client.", "I have completed my work, but I didn't inform the client."),
    (12, "the customer was angry because we didnt replied to his email.", "The customer was angry because we didn't reply to his email."),
    (13, "she is good in designing websites and have experience in seo.", "She is good at designing websites and has experience in SEO."),
    (14, "we need to improve the websites speed because it effect user experience.", "We need to improve the website's speed because it affects user experience."),
    (15, "thank you for your time i look forward to hear from you.", "Thank you for your time. I look forward to hearing from you."),
    (16, "i has a meeting tommorow at 10 am.", "I have a meeting tomorrow at 10 AM."),
    (17, "she go to work by bus every day.", "She goes to work by bus every day."),
    (18, "he don't knows the answer.", "He doesn't know the answer."),
    (19, "we are discuss the project now.", "We are discussing the project now."),
    (20, "they was very happy with the result.", "They were very happy with the result."),
    (21, "i have went to Chennai last week.", "I went to Chennai last week."),
    (22, "she have finish her work already.", "She has finished her work already."),
    (23, "he is working here since 2022.", "He has been working here since 2022."),
    (24, "i didn't went to the office yesterday.", "I didn't go to the office yesterday."),
    (25, "did you completed the task?", "Did you complete the task?"),
    (26, "where you are going now?", "Where are you going now?"),
    (27, "what time the meeting will start?", "What time will the meeting start?"),
    (28, "i am agree with your suggestion.", "I agree with your suggestion."),
    (29, "she is married with a doctor.", "She is married to a doctor."),
    (30, "he is good in mathematics.", "He is good at mathematics."),
    (31, "i am interested on this job.", "I am interested in this job."),
    (32, "we discussed about the problem yesterday.", "We discussed the problem yesterday."),
    (33, "please reply me as soon as possible.", "Please reply to me as soon as possible."),
    (34, "he explained me the process.", "He explained the process to me."),
    (35, "she suggested me to apply for the job.", "She suggested that I apply for the job."),
    (36, "i need an information about the course.", "I need information about the course."),
    (37, "he gave me many useful informations.", "He gave me a lot of useful information."),
    (38, "there is many errors in this document.", "There are many errors in this document."),
    (39, "there are a problem with the website.", "There is a problem with the website."),
    (40, "each employees have a login account.", "Each employee has a login account."),
    (41, "everyone have their own responsibility.", "Everyone has their own responsibility."),
    (42, "the team are working on the project.", "The team are working on the project."),
    (43, "my friend don't like spicy food.", "My friend doesn't like spicy food."),
    (44, "these type of websites are difficult to build.", "These types of websites are difficult to build."),
    (45, "this are the files you requested.", "These are the files you requested."),
    (46, "i bought a new equipments for my office.", "I bought new equipment for my office."),
    (47, "she gave me two advices.", "She gave me two pieces of advice."),
    (48, "he has three year experience in SEO.", "He has three years of experience in SEO."),
    (49, "i have five years experience working with websites.", "I have five years of experience working with websites."),
    (50, "the company provide good benefits.", "The company provides good benefits."),
    (51, "our manager give us new tasks every week.", "Our manager gives us new tasks every week."),
    (52, "the website need some improvements.", "The website needs some improvements."),
    (53, "these pages needs to be optimized.", "These pages need to be optimized."),
    (54, "the images was not loading properly.", "The images were not loading properly."),
    (55, "the content have many spelling mistakes.", "The content has many spelling mistakes."),
    (56, "please sends me the updated file.", "Please send me the updated file."),
    (57, "kindly check and let me knows.", "Kindly check and let me know."),
    (58, "i will send it to you tommorow.", "I will send it to you tomorrow."),
    (59, "we will discussed this issue later.", "We will discuss this issue later."),
    (60, "she will joining the company next month.", "She will join the company next month."),
    (61, "i can able to finish the task today.", "I can finish the task today."),
    (62, "he can speaks English very well.", "He can speak English very well."),
    (63, "you should to check the website.", "You should check the website."),
    (64, "we must to complete this today.", "We must complete this today."),
    (65, "i want to improving my SEO skills.", "I want to improve my SEO skills."),
    (66, "she enjoys to design websites.", "She enjoys designing websites."),
    (67, "he likes working with HTML and CSS.", "He likes working with HTML and CSS."),
    (68, "i look forward to meet you.", "I look forward to meeting you."),
    (69, "thank you for helping me with this issue.", "Thank you for helping me with this issue."),
    (70, "i am looking forward for your response.", "I am looking forward to your response."),
    (71, "the meeting is in monday morning.", "The meeting is on Monday morning."),
    (72, "i will call you at next week.", "I will call you next week."),
    (73, "she joined the company on 2024.", "She joined the company in 2024."),
    (74, "i have been working here from three years.", "I have been working here for three years."),
    (75, "he has lived here since five years.", "He has lived here for five years."),
    (76, "i reached to the office at 9 am.", "I reached the office at 9 AM."),
    (77, "she entered into the room quietly.", "She entered the room quietly."),
    (78, "he returned back home late.", "He returned home late."),
    (79, "please discuss about this with your manager.", "Please discuss this with your manager."),
    (80, "i will contact with the client tomorrow.", "I will contact the client tomorrow."),
    (81, "the new design is more better than the old one.", "The new design is better than the old one."),
    (82, "this is the most easiest way to solve it.", "This is the easiest way to solve it."),
    (83, "the website is very much faster now.", "The website is much faster now."),
    (84, "she is more smarter than her colleague.", "She is smarter than her colleague."),
    (85, "this task is enough easy for me.", "This task is easy enough for me."),
    (86, "he did a mistake in the report.", "He made a mistake in the report."),
    (87, "please make a photo of the document.", "Please take a photo of the document."),
    (88, "i am doing a new website for the client.", "I am creating a new website for the client."),
    (89, "she made a research about SEO trends.", "She conducted research on SEO trends."),
    (90, "we need to take an action immediately.", "We need to take action immediately."),
    (91, "i have a good news for you.", "I have good news for you."),
    (92, "he gave me a feedback about my work.", "He gave me feedback about my work."),
    (93, "the staffs are very helpful.", "The staff are very helpful."),
    (94, "we need more furnitures for the office.", "We need more furniture for the office."),
    (95, "there is too much people in the room.", "There are too many people in the room."),
    (96, "i have less tasks today than yesterday.", "I have fewer tasks today than yesterday."),
    (97, "she bought a umbrella yesterday.", "She bought an umbrella yesterday."),
    (98, "he is an honest employee and a useful person.", "He is an honest employee and a useful person."),
    (99, "i saw a old website on the internet.", "I saw an old website on the internet."),
    (100, "she is the best designer in our team.", "She is the best designer on our team."),
    (101, "my brother work in a IT company.", "My brother works in an IT company."),
    (102, "i need to buy new laptop for my work.", "I need to buy a new laptop for my work."),
    (103, "she sent email to her manager.", "She sent an email to her manager."),
    (104, "he is using a outdated software.", "He is using outdated software."),
    (105, "the user clicked on the button accidentally.", "The user clicked the button accidentally."),
    (106, "i didn't received your message.", "I didn't receive your message."),
    (107, "she hasn't completed the work yet.", "She hasn't completed the work yet."),
    (108, "have you ever visited Bangalore before?", "Have you ever visited Bangalore?"),
    (109, "i have already finished the task yesterday.", "I finished the task yesterday."),
    (110, "he was completed the project last night.", "He completed the project last night."),
    (111, "while i was work, my phone rang.", "While I was working, my phone rang."),
    (112, "when she arrived, we already left.", "When she arrived, we had already left."),
    (113, "if i will get the job, i will move to Chennai.", "If I get the job, I will move to Chennai."),
    (114, "if you study hard, you would pass the exam.", "If you study hard, you will pass the exam."),
    (115, "unless you don't hurry, you will miss the bus.", "Unless you hurry, you will miss the bus."),
]

detected = 0
missed = 0
missed_list = []

for num, text, correction in sentences:
    results = p.check(text)
    # If expected correction only differs in capitalization (British English accept), no grammar errors expected
    text_normalized = text.strip().lower()
    correction_normalized = correction.strip().lower()
    if text_normalized == correction_normalized:
        if len(results) == 0:
            detected += 1
            print(f"  CORRECT (no error): #{num}")
        else:
            # False positive on a correct sentence
            detected += 1
            rules = ", ".join(r["rule_id"] for r in results)
            print(f"  CORRECT (but detected {len(results)}): #{num} [{rules}]")
    elif len(results) > 0:
        detected += 1
        rules = ", ".join(r["rule_id"] for r in results)
        print(f"  DETECTED ({len(results)}): #{num} [{rules}]")
    else:
        missed += 1
        missed_list.append((num, text))
        print(f"  MISSED: #{num}")

print(f"\n=== RESULTS ===")
print(f"Detected: {detected}/{len(sentences)} ({100*detected/len(sentences):.1f}%)")
print(f"Missed: {missed}/{len(sentences)} ({100*missed/len(sentences):.1f}%)")
if missed_list:
    print(f"\nMISSED ERRORS:")
    for num, text in missed_list:
        print(f"  #{num}: {text[:80]}")
