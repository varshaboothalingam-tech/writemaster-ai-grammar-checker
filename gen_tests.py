"""Generate FP and FN test suites programmatically."""
import os

def gen_fp():
    S = []
    # Subjects
    SUBJ_SG = ["She", "He", "The cat", "My friend", "The teacher", "The dog"]
    SUBJ_PL = ["They", "We", "The dogs", "The children", "My parents", "The students"]
    SUBJ_1 = ["I"]
    SUBJ_EVERY = ["Everyone", "Nobody", "Each student", "Somebody", "Everybody"]
    BE_SG = {"she":"is","he":"is","the cat":"is","my friend":"is","the teacher":"is","the dog":"is",
             "i":"am"}
    BE_PL = {"they":"are","we":"are","the dogs":"are","the children":"are","my parents":"are",
             "the students":"are"}

    # Pattern-based correct sentences
    subjects = [
        ("I","am"),("She","is"),("He","is"),("They","are"),("We","are"),
        ("It","is"),("The cat","is"),("The dogs","are"),("The children","are"),
        ("My parents","are"),("The students","are"),("The weather","is"),
        ("The baby","is"),("My car","is"),("The door","is"),("The sky","is"),
        ("The book","is"),("The phone","is"),("The store","is"),("The music","is"),
    ]
    for subj, be in subjects:
        S.append(f"{subj} {be} going to the store.")
        S.append(f"{subj} {be} reading a book.")
        S.append(f"{subj} {be} very kind.")
        S.append(f"{subj} {be} responsible for the project.")
        S.append(f"{subj} {be} in charge of the department.")
        S.append(f"{subj} {be} expected to arrive by noon.")
        S.append(f"{subj} {be} getting better every day.")
        S.append(f"{subj} {be} planning to move.")
        S.append(f"{subj} {be} aware of the risks.")
        S.append(f"{subj} {be} capable of handling the task.")

    for subj, be in subjects:
        S.append(f"{subj} was at home yesterday.")
        S.append(f"{subj} was tired after work.")
        S.append(f"{subj} was reading a book.")
        S.append(f"{subj} was planning to leave.")
        S.append(f"{subj} was expected to arrive early.")

    # Collective nouns (both is/are are correct)
    for noun in ["team","family","committee","group","staff","audience","jury",
                 "class","government","company","police","council","board"]:
        S.append(f"The {noun} is meeting today.")
        S.append(f"The {noun} are meeting today.")
        S.append(f"The {noun} has made a decision.")
        S.append(f"The {noun} have made a decision.")

    # Have/has/had patterns
    for subj, have in [("I","have"),("She","has"),("He","has"),("They","have"),
                       ("We","have"),("The company","has"),("My parents","have")]:
        S.append(f"{subj} {have} been to Paris three times.")
        S.append(f"{subj} {have} been working here for ten years.")
        S.append(f"{subj} {have} enough time.")
        S.append(f"{subj} {have} a lot of experience.")
        S.append(f"{subj} {have} three children.")
        S.append(f"{subj} {have} to finish this report.")
        S.append(f"{subj} {have} been feeling unwell.")
        S.append(f"{subj} {have} no reason to complain.")
        S.append(f"{subj} {have} valid concerns.")
        S.append(f"{subj} {have} had enough of this.")
        S.append(f"{subj} {have} had a busy week.")
        S.append(f"{subj} {have} had great results.")
        S.append(f"{subj} {have} a beautiful garden.")
        S.append(f"{subj} {have} a lot of work to do.")
        S.append(f"{subj} {have} already finished.")

    # Modal verbs
    for subj, can in [("I","can"),("She","can"),("He","could"),("They","could"),
                      ("We","can"),("You","must"),("One","should")]:
        S.append(f"{subj} {can} swim.")
        S.append(f"{subj} {can} speak French.")
        S.append(f"{subj} {can} play the guitar.")
        S.append(f"{subj} {can} come with us.")
        S.append(f"{subj} {can} handle this.")
        S.append(f"{subj} {can} be here soon.")
        S.append(f"{subj} {can} help you with that.")
        S.append(f"{subj} {can} see the mountain from here.")
        S.append(f"{subj} {can} cook very well.")
        S.append(f"{subj} {can} remember to lock the door.")

    for subj, w in [("I","would"),("She","will"),("He","might"),("They","should"),
                    ("We","may")]:
        S.append(f"{subj} {w} like some coffee.")
        S.append(f"{subj} {w} prefer tea.")
        S.append(f"{subj} {w} be here soon.")
        S.append(f"{subj} {w} arrive by noon.")
        S.append(f"{subj} {w} enjoy the concert.")
        S.append(f"{subj} {w} benefit from the training.")
        S.append(f"{subj} {w} have forgotten.")
        S.append(f"{subj} {w} have been surprised.")
        S.append(f"{subj} {w} be tired.")
        S.append(f"{subj} {w} try harder.")

    # Prepositions
    for subj in ["I","She","He","They","We"]:
        S.append(f"{subj} am interested in learning new languages." if subj=="I"
                 else f"{subj} is interested in learning new languages." if subj in ("She","He")
                 else f"{subj} are interested in learning new languages.")
        S.append(f"{subj} am good at solving problems." if subj=="I"
                 else f"{subj} is good at solving problems." if subj in ("She","He")
                 else f"{subj} are good at solving problems.")
        S.append(f"{subj} am afraid of spiders." if subj=="I"
                 else f"{subj} is afraid of spiders." if subj in ("She","He")
                 else f"{subj} are afraid of spiders.")
    for s in ["I am looking forward to meeting you.","She is getting along with her classmates.",
              "He is getting over his illness.","They are getting into trouble.",
              "We are making progress on the project.","I am keeping track of the expenses.",
              "She is taking care of the children.","He is running into problems.",
              "They are setting up the equipment.","We are carrying out the plan.",
              "I am looking into the matter.","She is checking on the children.",
              "He is dropping off the kids.","They are filling in the form.",
              "We are picking up new skills.","I am putting on weight.",
              "She is turning down the offer.","He is breaking down the problem.",
              "They are working out at the gym.","We are settling down in our new home.",
              "I am checking in at the hotel.","She is following up on the case.",
              "He is catching up on work.","They are living up to expectations.",
              "I am looking up the information.","She is sorting out the problem.",
              "He is giving up smoking.","We are splitting up the work.",
              "She is making up for lost time.","They are tidying up the house."]:
        S.append(s)

    # Articles
    for s in ["I saw a dog in the park.","She is an engineer.","He went to the store.",
              "They bought a new car.","We live in a small town.","The sun rises in the east.",
              "A bird in the hand is worth two in the bush.","She is the best player.",
              "He is a hard worker.","I need a glass of water.","The phone is ringing.",
              "She ate an apple.","He read the book.","They visited the museum.",
              "We had a great time.","The teacher gave us homework.",
              "I will be back in an hour.","She is at the library.","He plays the piano.",
              "They went to a concert.","We need the money.","The children are playing.",
              "I saw her at a party.","He is the president.","They live in an apartment.",
              "We have a meeting today.","The dog barked loudly.","She wore a beautiful dress.",
              "He drove a truck.","They found a solution.","We need to discuss the matter.",
              "The music was wonderful.","I need to buy a present.","She is a nurse.",
              "He is an actor.","They have a beautiful house.","The car broke down.",
              "She started a new business.","He works at a hospital.","They are having a baby.",
              "We live near a park.","The weather is nice.","I need a pen.",
              "She is reading a novel.","He bought a suit.","They took a vacation.",
              "We watched a movie.","The door is open.","She has a headache.",
              "He made a mistake.","They won a prize.","We need an umbrella.",
              "The sky is cloudy.","She is an expert.","He is a teacher.",
              "They have a dog.","We had an argument.","The children need education.",
              "She is a university student.","He is an honest man."]:
        S.append(s)

    # Pronouns
    for s in ["I love my family.","She told me the truth.","He gave her a gift.",
              "They invited us to the party.","We saw them at the store.",
              "Who called you last night?","What did she say?","Which one do you prefer?",
              "Where did they go?","When did he arrive?","Why are you late?",
              "How did she do it?","He and I are going to the park.",
              "She and her sister look alike.","The dog wagged its tail.",
              "Everyone brought their own lunch.","Nobody forgot their homework.",
              "Each student submitted their assignment.","Someone left their umbrella.",
              "Anyone can bring their own tools.","Nothing matters more than family.",
              "Everything is going to be fine.","Nobody knows the answer.",
              "Everybody is welcome.","Each of them has their own room.",
              "Neither of them wants to leave.","Both of them are coming.",
              "All of them finished their work.","None of them forgot their homework.",
              "The cat cleaned itself.","The children enjoyed themselves.",
              "We amused ourselves with games.","I saw him at the library.",
              "She met them at the airport.","He told us the story.",
              "They showed me the way.","We helped her with the project.",
              "Give it to me.","Tell her the news.","Show them the picture.",
              "Call him tomorrow.","Ask us if you need help.","This is my book.",
              "That is her car.","These are our friends."]:
        S.append(s)

    # Complex/compound sentences
    for s in [
        "The cat sat on the mat while the dog slept on the floor.",
        "Although it was raining, we decided to go for a walk.",
        "She studied hard because she wanted to pass the exam.",
        "If you work hard, you will succeed.",
        "When the bell rings, the students run to the playground.",
        "The teacher explained the lesson and the students took notes.",
        "He went to the store, bought some groceries, and came home.",
        "She likes reading books, and he prefers watching movies.",
        "The more you practice, the better you become.",
        "She was tired, but she kept working.",
        "He is smart, kind, and hardworking.",
        "The book that I read last night was fascinating.",
        "The man who lives next door is a doctor.",
        "The house where I grew up has been demolished.",
        "This is the reason why I was late.",
        "I do not know what to do.",
        "She told me that she would be late.",
        "He asked if I wanted to go to the movies.",
        "The problem is that we do not have enough time.",
        "Not only did she finish first, but she also set a new record.",
        "Hardly had I arrived when the phone rang.",
        "No sooner had she sat down than the doorbell rang.",
        "It was such a cold day that we stayed inside.",
        "He is as tall as his brother.",
        "She runs faster than anyone else in the class.",
        "If I were you, I would accept the offer.",
        "Had I known about the meeting, I would have attended.",
        "She would have passed if she had studied harder.",
        "Unless you hurry, you will be late.",
        "In order to succeed, you must work hard.",
        "She is not only beautiful but also intelligent.",
        "He is both smart and hardworking.",
        "Neither the teacher nor the students were prepared.",
        "Either you come with us, or you stay here.",
        "Both the cat and the dog are sleeping.",
        "Since she moved to the city, she has made many new friends.",
        "After he graduated, he got a job at a bank.",
        "Before the movie started, we bought some popcorn.",
        "Whenever I visit my grandmother, she makes me cookies.",
        "Wherever you go, I will follow you.",
        "Whatever you decide, I will support you.",
        "The children were playing outside while their parents were cooking dinner.",
        "Although it was late, she kept working.",
        "Because it was raining, we stayed inside.",
        "Since you asked, I will tell you the truth.",
        "While I was cooking, the phone rang.",
        "After the movie, we went for dinner.",
        "Before you leave, please lock the door.",
        "Until further notice, the office will be closed.",
        "I will wait here until you come back.",
    ]:
        S.append(s)

    # Relative clauses
    for s in [
        "The book that I borrowed is overdue.",
        "The woman who lives next door is a doctor.",
        "The city where I was born is beautiful.",
        "The car which I bought last week broke down.",
        "The teacher whose class I attended was very kind.",
        "The person whom I met yesterday was very friendly.",
        "The movie that we watched last night was thrilling.",
        "The song which she sang was beautiful.",
        "The hotel where we stayed was luxurious.",
        "The project that he is working on is challenging.",
        "The student who sits next to me is very smart.",
        "The restaurant where we ate had great food.",
        "The idea that he proposed was brilliant.",
        "The teacher who taught me math retired last year.",
        "The car that she drives is red.",
        "The building where I work has twenty floors.",
        "The friend whom I trust most is my sister.",
        "The test that we took was difficult.",
        "The bridge which connects the two towns is old.",
        "The professor who lectures on Tuesdays is popular.",
        "The machine that we use every day is broken.",
        "The window that he broke has been repaired.",
        "The doctor whom I consulted was helpful.",
        "The tree that stands in our garden is very old.",
        "The airline that we flew with was excellent.",
        "The painting which hangs in the gallery is priceless.",
        "The friend that I went to school with is now a lawyer.",
        "The team that won the championship celebrated all night.",
        "The park where children play is safe.",
        "The company whose CEO resigned is in trouble.",
    ]:
        S.append(s)

    # Gerunds/infinitives
    for s in [
        "Swimming is my favorite activity.","I enjoy reading books.",
        "She loves cooking Italian food.","He hates waiting in line.",
        "They finished building the house.","I remember locking the door.",
        "She stopped smoking last year.","He admitted making a mistake.",
        "We discussed going to the movies.","I considered moving to a new city.",
        "She suggested taking a different route.","He avoided answering the question.",
        "They continued working despite the rain.","I finished eating dinner.",
        "She practiced playing the piano.","He enjoys solving puzzles.",
        "We plan to visit Italy next summer.","I want to learn a new language.",
        "She hopes to become a doctor.","He decided to change jobs.",
        "They agreed to help us.","I need to finish this report.",
        "She promised to call me back.","He offered to carry the bags.",
        "We expect to arrive by noon.","She seems to understand the problem.",
        "He appears to be happy.","They happen to know the answer.",
        "She failed to show up.","He refused to cooperate.",
        "We managed to finish on time.","She loves to sing in the shower.",
        "He likes to read before bed.","They prefer to walk instead of drive.",
        "She started to cry.","He began to realize the truth.",
        "I learned to play the guitar.","We forgot to bring the map.",
        "I remembered to lock the door.","She enjoys being outdoors.",
        "They finished eating dinner.","I stopped to take a break.",
    ]:
        S.append(s)

    # Comparatives/superlatives
    for s in [
        "She is taller than her sister.","He runs faster than I do.",
        "This book is more interesting than that one.","She is the tallest in the class.",
        "He is the best student in the school.","This is the most beautiful place I have seen.",
        "She is smarter than her brother.","He is older than he looks.",
        "She is the kindest person I know.","This is the worst movie ever.",
        "She speaks more fluently than before.","He is less experienced than she is.",
        "She works harder than anyone else.","This test is easier than the last one.",
        "She is the most talented artist.","The baby is growing bigger every day.",
        "She is far more intelligent than I am.","He is much stronger than he looks.",
        "She is the happiest person in the room.","He is more patient than I am.",
        "The older I get, the wiser I become.","She is the most beautiful woman.",
        "She is more creative than her peers.","He is less arrogant than before.",
        "She is the most popular girl in school.","He is better at math than science.",
        "She is the youngest of three sisters.","He is more athletic than his brother.",
        "The sooner, the better.","She is the most qualified candidate.",
    ]:
        S.append(s)

    # Time expressions
    for s in [
        "I have known her for five years.","She has lived here since 2015.",
        "He left three hours ago.","They have been waiting for a long time.",
        "We finished the project last week.","She will arrive in two hours.",
        "I saw him yesterday morning.","He has been sick since Monday.",
        "They have been married for twenty years.","We have been friends since college.",
        "She works out every day.","He travels abroad twice a year.",
        "We should leave by noon.","She has already finished her homework.",
        "They arrived just in time.","We will be there shortly.",
        "She called me the other day.","He has been working here since January.",
        "I will call you in a few minutes.","He finished the race in record time.",
        "They have been dating for six months.","We have lived in this house for ten years.",
        "She left the office a while ago.","He has been awake all night.",
        "They have been training for months.","I have not seen him in ages.",
        "He has been out of town for a week.","We will finish this by tomorrow.",
        "She has been learning French for two years.","He graduated ten years ago.",
    ]:
        S.append(s)

    # Conjunctions
    for s in [
        "I want to go, but I am too tired.","She likes tea and he likes coffee.",
        "We can leave now or we can wait a bit longer.","He is smart but lazy.",
        "She studied hard, so she passed the exam.","The weather was nice, yet we stayed inside.",
        "I will call you when I arrive.","She is talented and hardworking.",
        "He went to the store, and she stayed home.","We should eat before we leave.",
        "I like both cats and dogs.","Neither the rain nor the snow stopped him.",
        "Either you apologize or I will leave.","Both the cat and the dog are friendly.",
        "Although it was late, she kept working.","Because it was raining, we stayed inside.",
        "Since you asked, I will tell you the truth.","Unless you hurry, you will miss the bus.",
        "While I was cooking, the phone rang.","After the movie, we went for dinner.",
        "Before you leave, please lock the door.","Whenever I see her, she is smiling.",
        "Wherever you go, I will be there.","Whatever you say, I will believe you.",
        "He is tired, for he worked all day.","She was happy, so she smiled.",
        "He likes to read, and so does she.","I will help you if you help me.",
        "He is not only smart, but also kind.","We can go hiking, or we can go swimming.",
        "The cake was delicious, but I ate too much.","She was tired, yet she finished the race.",
        "As long as you are happy, I am happy.","Whether you agree or not, I will proceed.",
    ]:
        S.append(s)

    # Passive voice
    for s in [
        "The letter was written by my friend.","The cake was baked by my mother.",
        "The house was built in 1990.","The car was repaired by the mechanic.",
        "The report was submitted on time.","The project has been completed.",
        "The letter has been sent.","The package is being delivered today.",
        "The decision was made by the committee.","The rules must be followed.",
        "The problem can be solved easily.","The work needs to be finished.",
        "The message was delivered successfully.","The bridge was constructed last year.",
        "The taxes have been increased.","The phone was charged overnight.",
        "The agreement was signed by both parties.","The cake was decorated with flowers.",
        "The room was cleaned by the maid.","The result was announced yesterday.",
    ]:
        S.append(s)

    # Conditionals
    for s in [
        "If I had known, I would have told you.","If it rains, we will cancel the picnic.",
        "If you study hard, you will pass the exam.","If I were you, I would take the job.",
        "If we leave now, we will arrive on time.","If he were taller, he could play basketball.",
        "If I had more time, I would learn to play the piano.",
        "If the weather is nice, we will go to the beach.",
        "If I knew the answer, I would tell you.","If we work together, we can finish this.",
        "If you save money, you can buy a car.","If it snows, the schools will close.",
        "If we hurry, we can catch the last train.","If he apologizes, I will forgive him.",
        "If I were rich, I would travel the world.",
    ]:
        S.append(s)

    # Reported speech
    for s in [
        "She said that she was tired.","He told me that he had finished.",
        "They said that they were coming.","She said that she would call me.",
        "He told us that he had been working all day.",
        "She said she was going to the store.","He said that he could not come.",
        "They told me that they had left.","She said she had seen the movie.",
        "He said he would help me.","She told me she would be late.",
        "He said that he had already eaten.","They said they were planning a trip.",
        "She said that she was very happy.","He said that he had learned a lot.",
    ]:
        S.append(s)

    # Existential there
    for s in [
        "There is a cat on the mat.","There are many reasons to be happy.",
        "There was a storm last night.","There were several people at the party.",
        "There will be a meeting tomorrow.","There have been many changes.",
        "There is no reason to worry.","There are plenty of seats available.",
        "There was a time when I was afraid.","There were no survivors.",
        "There is nothing wrong with that.","There are some cookies left.",
        "There was a beautiful sunset.","There were three options to choose from.",
        "There is a problem with the system.","There are many things to do.",
        "There was an accident on the highway.","There were fifty guests at the wedding.",
        "There is no place like home.","There are plenty of fish in the sea.",
    ]:
        S.append(s)

    # Each/every/all/none
    for s in [
        "Each student has a textbook.","Every child needs love.",
        "All of the students passed the exam.","None of them were hurt.",
        "Each of the teams has a coach.","Every day is a new opportunity.",
        "All the lights were on.","None of the food was wasted.",
        "Each book on the shelf is new.","Every window was broken.",
        "All of us are going.","None of the answers were correct.",
        "Each of them has their own opinion.","Every morning she goes for a run.",
        "All the doors are locked.","None of the students forgot their homework.",
        "The number of students is increasing.","A number of students are absent.",
    ]:
        S.append(s)

    # Mass nouns
    for s in [
        "The information was helpful.","I need some advice.",
        "The furniture in the room is new.","She gave me some good feedback.",
        "The luggage is heavy.","He provided useful evidence.",
        "The music is beautiful.","There is too much traffic today.",
        "The weather is pleasant.","She needs more experience.",
        "The equipment is ready.","We have plenty of homework.",
        "The knowledge she has is impressive.","He has a lot of courage.",
        "She drank some water.","The bread is fresh.",
        "He earned a lot of money.","The glass is full.",
        "She showed great patience.","The paper is on the desk.",
    ]:
        S.append(s)

    # Short sentences
    for s in ["Yes.","No.","Please.","Thank you.","Hello.","Goodbye.","Sorry.",
              "Exactly.","Never.","Always.","I agree.","She is.","He was.",
              "They are.","We have.","It works.","Done.","Fine.","Thanks.","OK.",
              "Come in.","Go away.","Sit down.","Stand up.","Look here.",
              "Come here.","Hurry up.","Calm down.","Cheer up.","Be careful.",
              "Well done.","Of course.","Not yet.","Right now.","Just kidding.",
              "My turn.","Your fault.","My pleasure.","No problem.","Anytime.",
              "See you.","Take care.","Good luck.","God bless.","So be it."]:
        S.append(s)

    # Questions
    for s in [
        "Can you help me with this?","Where did you put my keys?",
        "What time does the movie start?","How much does this cost?",
        "Why were you late for the meeting?","Who is coming to the party?",
        "Which one should I choose?","When are you going to finish?",
        "Do you want to go for a walk?","Did she call you yesterday?",
        "Are you coming to the office tomorrow?","Have you finished your homework?",
        "Was he at the party last night?","Will you be home for dinner?",
        "Could you pass me the salt?","Would you like some coffee?",
        "Should we leave now?","May I borrow your pen?",
        "Do you know what time it is?","Can you tell me where the library is?",
    ]:
        S.append(s)

    # Imperatives
    for s in [
        "Please close the door.","Stop talking.","Come here.","Sit down.",
        "Be quiet.","Don't touch that.","Let me know when you arrive.",
        "Try to finish this today.","Remember to lock the door.",
        "Don't forget your keys.","Please pass the salt.","Open the window.",
        "Turn off the lights.","Write your name on the paper.","Listen carefully.",
        "Wait for me.","Take a deep breath.","Call me when you get home.",
        "Do your homework.","Go to bed.","Try again.","Hurry up.",
        "Think about it.","Look at this.","Help me with this.",
        "Follow the instructions.","Read the manual.","Clean your room.",
        "Feed the cat.","Water the plants.",
    ]:
        S.append(s)

    # Contractions
    for s in [
        "I'm going to the store.","She's been working hard all day.",
        "He'll be here soon.","They've already left.","We're almost there.",
        "It's raining outside.","I've been waiting for an hour.",
        "She'd rather stay home.","He can't come tonight.",
        "They won't accept the offer.","We should've left earlier.",
        "I wouldn't do that if I were you.","She hasn't finished yet.",
        "He's been feeling sick.","They're planning a trip.",
        "We've known each other for years.","It's been a long day.",
        "I'll call you later.","She's going to be fine.","They've had enough.",
    ]:
        S.append(s)

    # Semicolons/colons/dashes
    for s in [
        "I love reading; it helps me relax.","She is smart; he is lazy.",
        "We need one thing: dedication.","The answer is simple: work harder.",
        "The results were clear; the experiment was a success.",
        "I have a question; can you help me?","She was tired; however, she kept working.",
        "We went to Paris; then we drove to London.","He likes coffee; she prefers tea.",
        "We have three options; none of them are perfect.",
        "The job is difficult; it is also rewarding.",
        "I need help; can someone assist me?","The plan is simple; execute it flawlessly.",
        "She will arrive at noon; he will come at two.",
        "Life is short; make every moment count.",
        "The door was open; the house was empty.",
        "I want to go; but I can't.","She is kind; he is generous.",
        "He has one quality that sets him apart: determination.",
        "I love reading; it helps me relax.",
    ]:
        S.append(s)

    # Advanced grammar
    for s in [
        "Had I known, I would not have gone.","Never have I seen such a beautiful sunset.",
        "Not only is she beautiful, but she is also intelligent.",
        "Little did he know what was coming.",
        "Only after the meeting did I realize my mistake.",
        "In no way is this acceptable.","Here comes the bus.","There goes our plan.",
        "Never will I forget that day.","Not until tomorrow will we know.",
        "Under no circumstances should you open that door.",
        "Rarely have we encountered such a difficult problem.",
        "Seldom does he arrive on time.",
        "Scarcely had he arrived when the phone rang.",
        "At no point did he apologize.","On no account should you reveal the password.",
        "Barely had she finished speaking when the audience started clapping.",
        "So beautiful was the view that we stopped to admire it.",
        "In no case should this happen again.",
        "Well did she know the consequences.",
    ]:
        S.append(s)

    # Sentences with possessives
    for s in [
        "The teacher's desk is clean.","The students' books are on the table.",
        "My brother's car is red.","The dog's bone is buried in the garden.",
        "The children's toys are scattered.","The company's policy is clear.",
        "The women's rights movement is important.","The managers' meeting is at noon.",
        "The nurses' station is down the hall.","The players' performance was outstanding.",
        "The actresses' costumes were beautiful.","The professors' research is groundbreaking.",
        "The babies' parents are worried.","The countries' leaders met yesterday.",
        "The teams' coach gave a speech.","The factories' output has increased.",
        "The warehouses' inventory is full.","The cabinets' handles are broken.",
        "The landlords' association met.","TheMechanics' tools are organized.",
    ]:
        S.append(s)

    # Indian English patterns (still correct or acceptable)
    for s in [
        "I am having two cars.","We are having a meeting at five.",
        "They are having their lunch.","We are having a good time.",
        "I am having a wonderful day.","She is having dinner with her family.",
        "He is having his breakfast.","They are having an argument.",
        "I am having difficulty understanding this.","We are having an important discussion.",
        "They are having a party this weekend.","I am having an issue with my account.",
        "She is having a conversation with her friend.",
        "He is having a problem with his computer.",
        "We are having a celebration tonight.",
    ]:
        S.append(s)

    # Long sentences
    for s in [
        "The professor who has been teaching at the university for over twenty years announced that he would be retiring at the end of the semester.",
        "Although the weather forecast had predicted sunshine, we woke up to find that a cold front had moved in overnight.",
        "The committee spent several weeks reviewing documents before they reached a unanimous conclusion.",
        "She told me that she had been thinking about applying for the position advertised in the newspaper.",
        "The children who had been playing in the park all afternoon were very tired by the time their parents arrived.",
        "I have been considering the possibility of moving to a different city where the cost of living is lower.",
        "The restaurant that we went to last Saturday had a wonderful atmosphere and excellent food.",
        "He explained to us in great detail why he believed the project would be successful.",
        "When I was younger, I used to spend every summer at my grandparents' house in the countryside.",
        "The government announced a new policy requiring all citizens to carry identification cards.",
    ]:
        S.append(s)

    # Dedup and clean
    seen = set()
    unique = []
    for s in S:
        s = s.strip()
        if s and s not in seen and len(s) > 3:
            seen.add(s)
            unique.append(s)

    print(f"Generated {len(unique)} unique correct sentences")
    return unique


def gen_fn_categories():
    """Generate false negative test cases by category."""
    cats = {}

    cats["SVA"] = [
        ("She go to school every day.", "goes"),
        ("The dogs runs very fast.", "run"),
        ("He don't know the answer.", "doesn't"),
        ("They doesn't like ice cream.", "don't"),
        ("The list of items are on the table.", "is"),
        ("Everyone have their own opinion.", "has"),
        ("Nobody know the truth.", "knows"),
        ("The news are bad today.", "is"),
        ("Five dollars are too much.", "is"),
        ("Each of the students have a book.", "has"),
        ("Neither of them were ready.", "was"),
        ("He have been working hard.", "has"),
        ("She were at the store yesterday.", "was"),
        ("The teacher explain the lesson clearly.", "explains"),
        ("My brother play basketball every weekend.", "plays"),
        ("There is many reasons to celebrate.", "are"),
        ("The information were misleading.", "was"),
        ("Somebody have left their umbrella.", "has"),
        ("Every one of the cakes were delicious.", "was"),
        ("The police is investigating the crime.", "are"),
        ("The furniture are expensive.", "is"),
        ("The number of applicants are increasing.", "is"),
        ("Mathematics are my favorite subject.", "is"),
        ("The committee have reached a decision.", "has"),
        ("She don't understand the problem.", "doesn't"),
        ("The audience are clapping their hands.", "is"),
        ("Everyone were happy.", "was"),
        ("Nobody were interested.", "was"),
        ("Each of the boys have finished.", "has"),
        ("Half of the cake are missing.", "is"),
        ("Most of the water are gone.", "is"),
        ("All of the food were delicious.", "was"),
        ("Some of the milk have spilled.", "has"),
        ("Either the manager or the employees was wrong.", "were"),
        ("Neither the teacher nor the students was prepared.", "were"),
        ("He don't has any money.", "doesn't have"),
        ("Bread and butter are my favorite breakfast.", "is"),
        ("Tom and Jerry is a cartoon.", "are"),
        ("The family are going on vacation.", "is"),
        ("None of the above are correct.", "is"),
        ("The number of people is increasing every day.", "The number is already singular"),
        ("Me and him went to the store.", "He and I"),
        ("Her and me are going to the park.", "She and I"),
        ("Him and her went to the movies.", "He and she"),
        ("The team are winning the game.", "is"),
        ("The children was very excited.", "were"),
        ("Everybody have finished their homework.", "has"),
        ("A lot of people was waiting outside.", "were"),
        ("The pants is too long.", "are"),
        ("She don't play well.", "doesn't"),
        ("He do not know the answer.", "does not"),
    ]

    cats["DO_SUPPORT"] = [
        ("She like cats.", "likes"),
        ("He go to school.", "goes"),
        ("They comes every day.", "come"),
        ("Does she likes coffee?", "like"),
        ("Do he play basketball?", "Does"),
        ("She don't like Mondays.", "doesn't"),
        ("He don't have a car.", "doesn't"),
        ("They don't knows the answer.", "know"),
        ("Does he goes to work?", "go"),
        ("Do she have a pen?", "Does"),
        ("He don't wants to go.", "doesn't want"),
        ("She don't think so.", "doesn't"),
        ("They don't understands.", "understand"),
        ("He do goes to school.", "goes"),
        ("She do likes coffee.", "likes"),
        ("Do they goes home?", "go"),
        ("He don't needs help.", "doesn't need"),
        ("She don't has time.", "doesn't have"),
        ("They don't plays well.", "play"),
        ("He do is a good student.", "is"),
        ("She do can swim.", "can"),
        ("They do goes to school.", "go"),
        ("He don't wants to leave.", "doesn't want"),
        ("She don't believes him.", "doesn't believe"),
        ("We don't knows the answer.", "know"),
        ("He do play basketball.", "plays"),
        ("She do goes home.", "goes"),
        ("They do likes coffee.", "like"),
        ("He don't needs no help.", "doesn't need"),
        ("She don't wants any trouble.", "doesn't want"),
        ("We do goes to church.", "go"),
        ("He do has a car.", "has"),
        ("She do have a pen.", "has"),
        ("They do can swim.", "can"),
        ("He do will come.", "will"),
        ("She do should go.", "should"),
        ("They do must leave.", "must"),
        ("He do may come.", "may"),
        ("She do would like that.", "would"),
        ("They do could help.", "could"),
        ("He do might be late.", "might"),
        ("She do can sing.", "can"),
        ("They do shall we?", "shall"),
        ("He do can not swim.", "cannot"),
        ("She do will not come.", "will not"),
        ("They do should have gone.", "should have"),
        ("He do must go.", "must"),
        ("She do may be right.", "may"),
        ("They do would have helped.", "would have"),
    ]

    cats["TENSE_CONSISTENCY"] = [
        ("Yesterday I go to the store.", "went"),
        ("She walk to school yesterday.", "walked"),
        ("He is eat his lunch.", "eating"),
        ("They was play all day.", "played"),
        ("I was go to the movies.", "going"),
        ("She have finished the project.", "has"),
        ("He walk home after school.", "walked"),
        ("They arrives late every day.", "arrive"),
        ("Yesterday she cook dinner.", "cooked"),
        ("Last week I visit my parents.", "visited"),
        ("He is run every morning.", "runs"),
        ("She was study all night.", "studying"),
        ("They were go to the park.", "going"),
        ("I am went to the store.", "going"),
        ("He am finished the project.", "has"),
        ("We was have a good time.", "having"),
        ("She is called me yesterday.", "called"),
        ("He is eat lunch at noon.", "eats"),
        ("They is goes to school.", "go"),
        ("We is was at home.", "were"),
        ("I was am a student.", "was"),
        ("She were is a doctor.", "was"),
        ("He were goes home.", "went"),
        ("They was go to school.", "went"),
        ("We were is coming.", "were"),
        ("I go to the store yesterday.", "went"),
        ("She goes to school yesterday.", "went"),
        ("He come home late yesterday.", "came"),
        ("They run in the park yesterday.", "ran"),
        ("We eat lunch at noon yesterday.", "ate"),
        ("I see a movie last night.", "saw"),
        ("She take a taxi this morning.", "took"),
        ("He give her a gift yesterday.", "gave"),
        ("They buy a new car last week.", "bought"),
        ("We meet our friends yesterday.", "met"),
        ("I find my keys yesterday.", "found"),
        ("She bring her friend yesterday.", "brought"),
        ("He tell me the story yesterday.", "told"),
        ("They show us the picture yesterday.", "showed"),
        ("We choose the red one yesterday.", "chose"),
        ("I write a letter yesterday.", "wrote"),
        ("She read a book last night.", "read"),
        ("He drive to work yesterday.", "drove"),
        ("They sing a song yesterday.", "sang"),
        ("We think about it yesterday.", "thought"),
        ("I feel tired yesterday.", "felt"),
        ("She leave early yesterday.", "left"),
        ("He know the answer yesterday.", "knew"),
        ("They mean what they say yesterday.", "meant"),
    ]

    cats["DIDNT_PAST_FORM"] = [
        ("I didn't went to school.", "go"),
        ("She didn't came back.", "come"),
        ("He didn't saw the movie.", "see"),
        ("They didn't ate dinner.", "eat"),
        ("We didn't knew the answer.", "know"),
        ("I didn't had enough money.", "have"),
        ("He didn't took the test.", "take"),
        ("She didn't bought the dress.", "buy"),
        ("They didn't wrote the report.", "write"),
        ("I didn't spoke to her.", "speak"),
        ("We didn't ran fast enough.", "run"),
        ("He didn't gave me the money.", "give"),
        ("I didn't felt well.", "feel"),
        ("She didn't thought about it.", "think"),
        ("They didn't left early.", "leave"),
        ("He didn't kept his promise.", "keep"),
        ("I didn't slept well last night.", "sleep"),
        ("She didn't read the book.", "read"),
        ("We didn't drove to the airport.", "drive"),
        ("They didn't chose the right option.", "choose"),
        ("I didn't wore my coat.", "wear"),
        ("He didn't broke the window.", "break"),
        ("She didn't caught the ball.", "catch"),
        ("They didn't taught the students.", "teach"),
        ("I didn't lost my keys.", "lose"),
        ("We didn't met them before.", "meet"),
        ("He didn't spent too much.", "spend"),
        ("She didn't won the game.", "win"),
        ("I didn't told you.", "tell"),
        ("They didn't built the house.", "build"),
        ("He didn't sat down.", "sit"),
        ("I didn't stood up.", "stand"),
        ("She didn't drew the picture.", "draw"),
        ("We didn't grew tomatoes.", "grow"),
        ("They didn't sang the song.", "sing"),
        ("I didn't fell down.", "fall"),
        ("He didn't shook hands.", "shake"),
        ("She didn't bit the apple.", "bite"),
        ("I didn't hid the money.", "hide"),
        ("We didn't threw the ball.", "throw"),
        ("He didn't blew the whistle.", "blow"),
        ("She didn't froze the food.", "freeze"),
        ("They didn't chose carefully.", "choose"),
        ("I didn't slept enough.", "sleep"),
        ("He didn't stole the car.", "steal"),
        ("She didn't tore the paper.", "tear"),
        ("We didn't woke up early.", "wake"),
        ("They didn't tore the fabric.", "tear"),
        ("I didn't woke up on time.", "wake"),
        ("He didn't drove carefully.", "drive"),
    ]

    cats["ARTICLE_MISSING"] = [
        ("I went to store yesterday.", "the store"),
        ("She is student at the university.", "a student"),
        ("He is best player on the team.", "the best"),
        ("They live in small town.", "a small town"),
        ("We need to discuss matter.", "the matter"),
        ("She works at hospital.", "a hospital"),
        ("He bought new car.", "a new car"),
        ("We had great time.", "a great time"),
        ("She is teacher.", "a teacher"),
        ("He went to dentist.", "the dentist"),
        ("They visited museum.", "the museum"),
        ("We watched movie.", "a movie"),
        ("She read book.", "a book"),
        ("He plays piano.", "the piano"),
        ("They went to concert.", "a concert"),
        ("We live in apartment.", "an apartment"),
        ("She needs umbrella.", "an umbrella"),
        ("He is president of company.", "the president"),
        ("They bought house.", "a house"),
        ("We had argument.", "an argument"),
        ("She has headache.", "a headache"),
        ("He made mistake.", "a mistake"),
        ("They won prize.", "a prize"),
        ("We need to discuss it with manager.", "the manager"),
        ("She is at library.", "the library"),
        ("He works at bank.", "a bank"),
        ("They are having baby.", "a baby"),
        ("We live near park.", "a park"),
        ("She needs glass of water.", "a glass"),
        ("He drove truck.", "a truck"),
        ("They found solution.", "a solution"),
        ("We watched movie last night.", "a movie"),
        ("She wore dress.", "a dress"),
        ("He bought suit.", "a suit"),
        ("They took vacation.", "a vacation"),
        ("She started new business.", "a new business"),
        ("He is at door.", "the door"),
        ("We need to discuss it.", ""),
        ("She is in kitchen.", "the kitchen"),
        ("He went to office.", "the office"),
        ("They are in garden.", "the garden"),
        ("We had dinner.", "dinner"),
        ("She is at work.", "work"),
        ("He went to school.", "school"),
        ("They are at home.", "home"),
        ("We went to church.", "church"),
        ("She is in bed.", "bed"),
        ("He went to prison.", "prison"),
        ("They are in class.", "class"),
    ]

    cats["MODAL_WRONG_FORM"] = [
        ("She can sings beautifully.", "sing"),
        ("He will goes to the store.", "go"),
        ("They should studies harder.", "study"),
        ("We must to leave now.", "leave"),
        ("She may comes tomorrow.", "come"),
        ("He could helps you.", "help"),
        ("They would enjoys it.", "enjoy"),
        ("We should to wait.", "wait"),
        ("I can to swim.", "swim"),
        ("She will has finished.", "have"),
        ("He must to go.", "go"),
        ("They might takes a taxi.", "take"),
        ("We could to try harder.", "try"),
        ("She should to study.", "study"),
        ("He will goes home.", "go"),
        ("They can sings well.", "sing"),
        ("We may to go.", "go"),
        ("I must to finish.", "finish"),
        ("She would enjoys it.", "enjoy"),
        ("He could to help.", "help"),
        ("They should to go.", "go"),
        ("We must tries harder.", "try"),
        ("I may to come.", "come"),
        ("She will goes there.", "go"),
        ("He can to speak.", "speak"),
        ("They must to leave.", "leave"),
        ("We could goes there.", "go"),
        ("I should to check.", "check"),
        ("She may goes now.", "go"),
        ("He will helps me.", "help"),
        ("They could sings.", "sing"),
        ("We should to help.", "help"),
        ("I would to go.", "go"),
        ("She must goes.", "go"),
        ("He can helps.", "help"),
        ("They will goes.", "go"),
        ("We may to try.", "try"),
        ("I could to see.", "see"),
        ("She should goes.", "go"),
        ("He must helps.", "help"),
        ("They can goes.", "go"),
        ("We would to try.", "try"),
        ("I might to come.", "come"),
        ("She will helps.", "help"),
        ("He could goes.", "go"),
        ("They should helps.", "help"),
        ("We must goes.", "go"),
        ("I may helps.", "help"),
        ("She can goes.", "go"),
        ("He will helps.", "help"),
    ]

    cats["WRONG_PAST_PARTICIPLE"] = [
        ("She has went to the store.", "gone"),
        ("He has ate all the food.", "eaten"),
        ("They has wrote the report.", "written"),
        ("We has drove to the airport.", "driven"),
        ("I has saw that movie.", "seen"),
        ("She has took the test.", "taken"),
        ("He has gave her a gift.", "given"),
        ("They has spoke to the manager.", "spoken"),
        ("I has broke the vase.", "broken"),
        ("She has chose the blue one.", "chosen"),
        ("He has flew to London.", "flown"),
        ("We has drank all the water.", "drunk"),
        ("I has begun to understand.", "begun"),
        ("They has run out of time.", "run"),
        ("She has woke up early.", "woken"),
        ("He has froze the food.", "frozen"),
        ("We has grew the vegetables.", "grown"),
        ("I has threw the ball.", "thrown"),
        ("She has known him for years.", "known"),
        ("They has left already.", "left"),
        ("He has fell asleep.", "fallen"),
        ("We has hidden the treasure.", "hidden"),
        ("I has forgot my keys.", "forgotten"),
        ("She has bit her lip.", "bitten"),
        ("He has woke the children.", "woken"),
        ("They has tore the paper.", "torn"),
        ("We has blew out the candles.", "blown"),
        ("I has shook his hand.", "shaken"),
        ("She has stole the money.", "stolen"),
        ("He has woken up.", "woken"),
        ("They has began the project.", "begun"),
        ("We has sunk the ship.", "sunk"),
        ("I has drunk too much.", "drunk"),
        ("She has swam across the lake.", "swum"),
        ("He has rang the bell.", "rung"),
        ("They has sprang into action.", "sprung"),
        ("We has ate lunch.", "eaten"),
        ("I has went home.", "gone"),
        ("She has spoke to him.", "spoken"),
        ("He has ran a marathon.", "run"),
        ("They has gave the presentation.", "given"),
        ("We has wrote the letter.", "written"),
        ("I has drove there.", "driven"),
        ("She has took the medicine.", "taken"),
        ("He has swam in the ocean.", "swum"),
        ("They has blown the whistle.", "blown"),
        ("We has saw the results.", "seen"),
        ("I has begun to see.", "begun"),
        ("She has grew taller.", "grown"),
        ("He has froze the pipes.", "frozen"),
    ]

    cats["MISSING_APOSTROPHE"] = [
        ("The dog wagged it's tail.", "its"),
        ("She lost her dog's bone.", "dog's"),
        ("The teacher's desk is clean.", "teachers'"),
        ("My parents house is big.", "parents'"),
        ("The childrens toys are here.", "children's"),
        ("The dogs bone is buried.", "dog's"),
        ("She borrowed her friend's car.", "friends'"),
        ("The students books are on the table.", "students'"),
        ("He went to his friend's house.", "friends'"),
        ("The company's policy is clear.", ""),
        ("The women's rights are important.", ""),
        ("My brother's car is red.", ""),
        ("The cats food is on the floor.", "cat's"),
        ("The babies parents are worried.", "babies'"),
        ("She wore her mother's dress.", ""),
        ("The teams coach gave a speech.", "team's"),
        ("He is at his grandmother's house.", ""),
        ("The managers meeting is at noon.", "managers'"),
        ("The doctors office is closed.", "doctor's"),
        ("The nurses station is nearby.", "nurses'"),
        ("She left her childs toys outside.", "child's"),
        ("The mens room is upstairs.", "men's"),
        ("The womens restroom is downstairs.", "women's"),
        ("My aunts house is nearby.", "aunt's"),
        ("The horses stable is clean.", "horse's"),
        ("The birds nest is in the tree.", "bird's"),
        ("He fixed the cars engine.", "car's"),
        ("The planets orbit is elliptical.", "planet's"),
        ("She loves her husbands cooking.", "husband's"),
        ("The childs mother was worried.", "child's"),
    ]

    cats["COMMA_SPLICE"] = [
        ("I love to read, she prefers movies.", "I love to read; she prefers movies."),
        ("The cat is black, the dog is white.", "The cat is black, and the dog is white."),
        ("He went to the store, he bought some milk.", "He went to the store, and he bought some milk."),
        ("She studied hard, she passed the exam.", "She studied hard, so she passed the exam."),
        ("It was raining, we stayed inside.", "It was raining, so we stayed inside."),
        ("The movie was long, it was good.", "The movie was long, but it was good."),
        ("I wanted to go, I was too tired.", "I wanted to go, but I was too tired."),
        ("She likes coffee, he prefers tea.", "She likes coffee, and he prefers tea."),
        ("We could leave early, we chose to stay.", "We could leave early, but we chose to stay."),
        ("The results were good, the team was happy.", "The results were good, so the team was happy."),
        ("He finished the work, he went home.", "He finished the work, and he went home."),
        ("She was tired, she kept working.", "She was tired, but she kept working."),
        ("It was cold outside, we wore jackets.", "It was cold outside, so we wore jackets."),
        ("The food was ready, we sat down to eat.", "The food was ready, and we sat down to eat."),
        ("He called me, I did not answer.", "He called me, but I did not answer."),
        ("She arrived late, the meeting had started.", "She arrived late, and the meeting had started."),
        ("They practiced every day, they improved quickly.", "They practiced every day, so they improved quickly."),
        ("The project was difficult, we finished it.", "The project was difficult, but we finished it."),
        ("He is smart, he is lazy.", "He is smart, but he is lazy."),
        ("She sang beautifully, the audience clapped.", "She sang beautifully, and the audience clapped."),
    ]

    cats["PREPOSITION_COLLOCATION"] = [
        ("She is good in math.", "at"),
        ("He is afraid from spiders.", "of"),
        ("They are waiting to the bus.", "for"),
        ("I am interested for learning.", "in"),
        ("She is worried to the exam.", "about"),
        ("He is responsible to the project.", "for"),
        ("They are arguing to politics.", "about"),
        ("We are planning in the future.", "for"),
        ("I am proud in my children.", "of"),
        ("She is jealous from her sister.", "of"),
        ("He is capable for doing the job.", "of"),
        ("They are fond in outdoor activities.", "of"),
        ("We are tired in waiting.", "of"),
        ("I am familiar to this software.", "with"),
        ("She is dependent from her parents.", "on"),
        ("He is addicted in video games.", "to"),
        ("They are committed for the cause.", "to"),
        ("We are opposed in the plan.", "to"),
        ("I am used in working long hours.", "to"),
        ("She is accustomed for the cold.", "to"),
        ("He is sensitive from criticism.", "to"),
        ("I disagree in your opinion.", "with"),
        ("She specializes on pediatric medicine.", "in"),
        ("He apologized to being late.", "for"),
        ("They are grateful to your help.", "for"),
        ("We are concerned to the environment.", "about"),
        ("She is confident in her abilities.", "about"),
        ("He is looking for to meeting you.", "forward"),
        ("I am looking in the matter.", "into"),
        ("She is checking in the children.", "on"),
        ("He is running at problems.", "into"),
        ("They are setting in the equipment.", "up"),
        ("We are carrying in the plan.", "out"),
        ("I am falling in my studies.", "behind"),
        ("She is running in of patience.", "out"),
        ("He is breaking in the problem.", "down"),
        ("They are working in at the gym.", "out"),
        ("We are picking in new skills.", "up"),
        ("I am putting in weight.", "on"),
        ("She is turning in the offer.", "down"),
        ("He is giving in smoking.", "up"),
        ("They are breaking in up.", ""),
        ("We are splitting in the work.", "up"),
        ("I am checking in at.", "in"),
        ("She is following in on the case.", "up"),
        ("He is catching in on work.", "up"),
        ("They are tidying in the house.", "up"),
        ("We are clearing in the misunderstanding.", "up"),
        ("She is making in up for lost time.", ""),
    ]

    cats["DOUBLE_COMPARATIVE"] = [
        ("She is more taller than her sister.", "taller"),
        ("He is most intelligent in the class.", "the most"),
        ("This is the more better option.", "better"),
        ("She is most tallest in the school.", "the tallest"),
        ("He is more stronger than he looks.", "stronger"),
        ("This is the most easiest exam.", "easiest"),
        ("She is more prettier than her friend.", "prettier"),
        ("He is more heavier than his brother.", "heavier"),
        ("This is the most highest building.", "highest"),
        ("She is more smarter than she thinks.", "smarter"),
        ("He is less smarter than his sister.", "less smart"),
        ("This is the most fastest car.", "fastest"),
        ("She is moreolder than me.", "older"),
        ("He is moststronger than his opponent.", "stronger"),
        ("This is more better than nothing.", "better"),
        ("She is mostbeautiful in the world.", "the most"),
        ("He is morehardworking than anyone.", "hardworking"),
        ("This is the mostsimple solution.", "simplest"),
        ("She is morecareful than before.", "more careful"),
        ("He is mostpatient of all.", "the most"),
        ("This is moreeasier than I thought.", "easier"),
        ("She is mostskilled player.", "the most"),
        ("He is moregenerous than he seems.", "more generous"),
        ("This is mostdifficult question.", "the most"),
        ("She is morecreative than her peers.", "more creative"),
        ("He is mostpopular student.", "the most"),
        ("This is morecomplicated than necessary.", "more complicated"),
        ("She is mostaccomplished artist.", "the most"),
        ("He is morededicated than his colleagues.", "more dedicated"),
        ("This is mostchallenging project.", "the most"),
        ("She is moresuccessful than expected.", "more successful"),
        ("He is mosttalented musician.", "the most"),
        ("This is moreadvanced technology.", "more advanced"),
        ("She is mostknowledgeable person.", "the most"),
        ("He is moreexperienced than I am.", "more experienced"),
        ("This is mostinteresting book.", "the most"),
        ("She is morepassionate about her work.", "more passionate"),
        ("He is mostreliable team member.", "the most"),
        ("This is morepractical approach.", "more practical"),
        ("She is mostinfluential leader.", "the most"),
    ]

    cats["PARALLELISM"] = [
        ("She likes to swim, running, and biking.", "to swim, to run, and to bike"),
        ("He enjoys reading, to write, and drawing.", "reading, writing, and drawing"),
        ("They want to learn, studying, and practice.", "to learn, to study, and to practice"),
        ("We plan hiking, to swim, and fishing.", "to hike, to swim, and to fish"),
        ("She prefers to walk, jogging, and cycling.", "to walk, to jog, and to cycle"),
        ("He likes cooking, to clean, and shopping.", "cooking, cleaning, and shopping"),
        ("They enjoy to hike, camping, and fish.", "hiking, camping, and fishing"),
        ("We started to pack, cooking, and leaving.", "packing, cooking, and leaving"),
        ("She needs to study, working, and resting.", "studying, working, and resting"),
        ("He wants playing guitar, to sing, and dance.", "to play guitar, to sing, and to dance"),
        ("They plan to travel, exploring, and relaxing.", "traveling, exploring, and relaxing"),
        ("We enjoy reading books, to watch movies, and play games.", "reading books, watching movies, and playing games"),
        ("She decided to exercise, eat healthy, and sleep well.", "to exercise, to eat healthy, and to sleep well"),
        ("He likes swimming, to jog, and cycling.", "swimming, jogging, and cycling"),
        ("They prefer to cook, baking, and grilling.", "cooking, baking, and grilling"),
        ("We started learning Spanish, to speak French, and study German.", "learning Spanish, speaking French, and studying German"),
        ("She enjoys to paint, drawing, and sculpting.", "painting, drawing, and sculpting"),
        ("He prefers to read, listening, and discussing.", "reading, listening, and discussing"),
        ("They like cooking, to bake, and fry.", "cooking, baking, and frying"),
        ("We decided to run, cycling, and to swim.", "running, cycling, and swimming"),
        ("She started writing, to edit, and publishing.", "writing, editing, and publishing"),
        ("He enjoys hiking, to camp, and fish.", "hiking, camping, and fishing"),
        ("They prefer reading, to write, and discuss.", "reading, writing, and discussing"),
        ("We want to learn French, studying Spanish, and Italian.", "learning French, studying Spanish, and studying Italian"),
        ("She likes to dance, singing, and acting.", "dancing, singing, and acting"),
        ("He prefers cooking, to bake, and to grill.", "cooking, baking, and grilling"),
        ("They enjoy to hike, camping, and to fish.", "hiking, camping, and fishing"),
        ("We started to read, writing, and edit.", "reading, writing, and editing"),
        ("She likes shopping, to cook, and bake.", "shopping, cooking, and baking"),
        ("He enjoys playing tennis, to swim, and jog.", "playing tennis, swimming, and jogging"),
        ("They prefer to drive, flying, and sailing.", "driving, flying, and sailing"),
        ("We plan to move, packing, and clean.", "moving, packing, and cleaning"),
        ("She started to jog, swim, and cycle.", "jogging, swimming, and cycling"),
        ("He likes running, to cycle, and hike.", "running, cycling, and hiking"),
        ("They want learning French, to study German.", "to learn French, to study German"),
        ("We enjoy hiking, camping, and to fish.", "hiking, camping, and fishing"),
        ("She prefers reading, writing, and to edit.", "reading, writing, and editing"),
        ("He started to cook, baking, and grill.", "cooking, baking, and grilling"),
        ("They like to read, write, and draw.", "to read, to write, and to draw"),
        ("We plan hiking, to swim, and to cycle.", "to hike, to swim, and to cycle"),
    ]

    cats["RUN_ON_SENTENCE"] = [
        ("I love to read she prefers movies.", "I love to read; she prefers movies."),
        ("The cat is black the dog is white.", "The cat is black, and the dog is white."),
        ("He went to the store he bought milk.", "He went to the store, and he bought milk."),
        ("She studied hard she passed.", "She studied hard, so she passed."),
        ("It was raining we stayed inside.", "It was raining, so we stayed inside."),
        ("The movie was long it was good.", "The movie was long, but it was good."),
        ("I wanted to go I was tired.", "I wanted to go, but I was tired."),
        ("She likes coffee he prefers tea.", "She likes coffee, and he prefers tea."),
        ("He finished the work he went home.", "He finished the work, and he went home."),
        ("She was tired she kept working.", "She was tired, but she kept working."),
        ("It was cold we wore jackets.", "It was cold, so we wore jackets."),
        ("The food was ready we ate.", "The food was ready, and we ate."),
        ("He called me I did not answer.", "He called me, but I did not answer."),
        ("They practiced daily they improved.", "They practiced daily, so they improved."),
        ("The project was hard we finished.", "The project was hard, but we finished it."),
        ("He is smart he is lazy.", "He is smart, but he is lazy."),
        ("She sang the audience clapped.", "She sang, and the audience clapped."),
        ("The sun set we went home.", "The sun set, and we went home."),
        ("I am tired I need sleep.", "I am tired, and I need sleep."),
        ("She left early she had a reason.", "She left early, and she had a reason."),
    ]

    cats["MISSING_COMMA"] = [
        ("After eating dinner we went for a walk.", "After eating dinner, we went for a walk."),
        ("Before the movie started we bought popcorn.", "Before the movie started, we bought popcorn."),
        ("When the bell rang the students left.", "When the bell rang, the students left."),
        ("Although it was raining we went outside.", "Although it was raining, we went outside."),
        ("Because she was tired she went to bed.", "Because she was tired, she went to bed."),
        ("Since he was late he missed the bus.", "Since he was late, he missed the bus."),
        ("If you study hard you will pass.", "If you study hard, you will pass."),
        ("When I was young I loved to play.", "When I was young, I loved to play."),
        ("After the meeting we discussed the plan.", "After the meeting, we discussed the plan."),
        ("Before you leave lock the door.", "Before you leave, lock the door."),
        ("While she was cooking the phone rang.", "While she was cooking, the phone rang."),
        ("Unless you hurry you will be late.", "Unless you hurry, you will be late."),
        ("Whenever I visit she makes cookies.", "Whenever I visit, she makes cookies."),
        ("Wherever you go I will follow.", "Wherever you go, I will follow you."),
        ("Since you asked I will tell you.", "Since you asked, I will tell you."),
        ("Although he tried he could not win.", "Although he tried, he could not win."),
        ("Because it was cold we stayed inside.", "Because it was cold, we stayed inside."),
        ("When she arrived the party started.", "When she arrived, the party started."),
        ("If it rains we will cancel.", "If it rains, we will cancel."),
        ("After breakfast he went to work.", "After breakfast, he went to work."),
        ("Despite being tired she finished.", "Despite being tired, she finished."),
        ("However hard he tried he failed.", "However hard he tried, he failed."),
        ("Until further notice the office is closed.", "Until further notice, the office is closed."),
        ("While the others slept she studied.", "While the others slept, she studied."),
        ("Before the sun rose they left.", "Before the sun rose, they left."),
        ("As soon as he arrived the meeting started.", "As soon as he arrived, the meeting started."),
        ("Provided that everything goes well we will win.", "Provided that everything goes well, we will win."),
        ("Even though she was busy she helped.", "Even though she was busy, she helped."),
        ("In order to succeed you must work hard.", "In order to succeed, you must work hard."),
        ("Once the decision is made there is no going back.", "Once the decision is made, there is no going back."),
    ]

    cats["TO_PAST_PARTICIPLE"] = [
        ("She wants to went home.", "go"),
        ("He needs to ate more vegetables.", "eat"),
        ("They plan to swam across the lake.", "swim"),
        ("We hope to won the game.", "win"),
        ("I want to wrote a letter.", "write"),
        ("She decided to drove to work.", "drive"),
        ("He hopes to flew to Paris.", "fly"),
        ("They want to spoke to the manager.", "speak"),
        ("We need to gave them the message.", "give"),
        ("I plan to took a vacation.", "take"),
        ("She needs to chose carefully.", "choose"),
        ("He wants to broke the record.", "break"),
        ("They hope to knew the answer.", "know"),
        ("We want to saw the sunset.", "see"),
        ("I decided to threw the ball.", "throw"),
        ("She hopes to grew her business.", "grow"),
        ("He plans to froze the food.", "freeze"),
        ("They want to began the project.", "begin"),
        ("We need to rang the bell.", "ring"),
        ("I want to drank some water.", "drink"),
        ("She decided to fell asleep.", "fall"),
        ("He hopes to hid the treasure.", "hide"),
        ("They need to shook hands.", "shake"),
        ("We want to bit into the apple.", "bite"),
        ("I plan to stole the show.", "steal"),
        ("She wants to tore the paper.", "tear"),
        ("He hopes to woke up early.", "wake"),
        ("They want to swam in the ocean.", "swim"),
        ("We need to blew out the candles.", "blow"),
        ("I decided to drew a picture.", "draw"),
        ("She hopes to grew taller.", "grow"),
        ("He wants to froze the pipes.", "freeze"),
        ("They need to sunk the ship.", "sink"),
        ("We want to sprang into action.", "spring"),
        ("I plan to ran a marathon.", "run"),
        ("She hopes to wrote a book.", "write"),
        ("He wants to sang a song.", "sing"),
        ("They decide to left early.", "leave"),
        ("We want to chose the best option.", "choose"),
        ("I hope to felt better soon.", "feel"),
    ]

    cats["EXPLAIN_DATIVE"] = [
        ("She explained me the problem.", "explained the problem to me"),
        ("He told the news me.", "told me the news"),
        ("They described me the situation.", "described the situation to me"),
        ("We explained them the rules.", "explained the rules to them"),
        ("She taught him English.", "taught English to him"),
        ("He offered me the job.", "offered the job to me"),
        ("They showed us the way.", "showed the way to us"),
        ("She gave him the book.", "gave the book to him"),
        ("He handed me the papers.", "handed the papers to me"),
        ("They sent us the package.", "sent the package to us"),
        ("She read me the letter.", "read the letter to me"),
        ("He told us the story.", "told the story to us"),
        ("They paid me the money.", "paid the money to me"),
        ("She showed him the picture.", "showed the picture to him"),
        ("He gave her the keys.", "gave the keys to her"),
        ("They explained us the process.", "explained the process to us"),
        ("She taught them the lesson.", "taught the lesson to them"),
        ("He offered her the position.", "offered the position to her"),
        ("They showed me the results.", "showed the results to me"),
        ("She handed him the report.", "handed the report to him"),
        ("He told her the truth.", "told the truth to her"),
        ("They sent me the document.", "sent the document to me"),
        ("She gave them the instructions.", "gave the instructions to them"),
        ("He read her the poem.", "read the poem to her"),
        ("They offered us the deal.", "offered the deal to us"),
        ("She described him the plan.", "described the plan to him"),
        ("He explained us why.", "explained to us why"),
        ("They taught us the method.", "taught the method to us"),
        ("She told him what happened.", "told him about what happened"),
        ("He showed her how to do it.", "showed her how to do it"),
    ]

    cats["MISSING_ARTICLE"] = [
        ("In future we will do better.", "in the future"),
        ("For few weeks the shop was closed.", "for a few weeks"),
        ("She is student.", "a student"),
        ("He went to shop.", "the shop"),
        ("They live in small town.", "a small town"),
        ("We need help.", "some help"),
        ("She has headache.", "a headache"),
        ("He made mistake.", "a mistake"),
        ("They won prize.", "a prize"),
        ("We had good time.", "a good time"),
        ("She is at door.", "at the door"),
        ("He went to office.", "to the office"),
        ("They are in garden.", "in the garden"),
        ("She is best player.", "the best player"),
        ("He is president.", "the president"),
        ("They found solution.", "a solution"),
        ("We need discuss it.", "to discuss it"),
        ("She has good reason.", "a good reason"),
        ("He is teacher.", "a teacher"),
        ("They visited museum.", "the museum"),
        ("We watched movie.", "a movie"),
        ("She read book.", "a book"),
        ("He plays piano.", "the piano"),
        ("They went to concert.", "a concert"),
        ("We live in apartment.", "an apartment"),
        ("She needs umbrella.", "an umbrella"),
        ("He drove truck.", "a truck"),
        ("She wore dress.", "a dress"),
        ("He bought suit.", "a suit"),
        ("They took vacation.", "a vacation"),
    ]

    return cats


def main():
    # Generate FP test
    correct = gen_fp()

    with open("test_false_positives_comprehensive.py", "w", encoding="utf-8") as f:
        f.write('"""Comprehensive FP test: all sentences should be correct English."""\n')
        f.write('import sys, time\n')
        f.write('sys.path.insert(0, ".")\n')
        f.write('from unified_pipeline import UnifiedPipeline\n\n')
        f.write("SENTENCES = [\n")
        for s in correct:
            f.write(f"    {repr(s)},\n")
        f.write("]\n\n")
        f.write("""
def main():
    p = UnifiedPipeline()
    fps = []
    t0 = time.time()
    for i, sent in enumerate(SENTENCES):
        results = p.check(sent)
        if results:
            for r in results:
                fps.append((i, sent, r))
                print(f'FP #{len(fps)}: rule_id=\\"{r["rule_id"]}\\" conf={r["confidence"]:.2f}')
                print(f'  Text: \\"{sent[:100]}\\"')
                print(f'  Error: {r["original"]} -> {r["replacement"]}')
    elapsed = time.time() - t0
    total = len(SENTENCES)
    fp_count = len(fps)
    print(f"")
    print(f"=== FALSE POSITIVE TEST RESULTS ===")
    print(f"Total sentences tested: {total}")
    print(f"Total false positives: {fp_count}")
    print(f"FP rate: {fp_count/total*100:.2f}%")
    print(f"FP-free sentences: {total - fp_count}")
    print(f"Time: {elapsed:.1f}s ({total/elapsed:.0f} sentences/sec)")

if __name__ == "__main__":
    main()
""")
    print(f"test_false_positives_comprehensive.py: {len(correct)} sentences")

    # Generate FN test
    cats = gen_fn_categories()
    total_tests = sum(len(v) for v in cats.values())

    with open("test_categories_comprehensive.py", "w", encoding="utf-8") as f:
        f.write('"""Category-specific false negative test."""\n')
        f.write('import sys, time\n')
        f.write('sys.path.insert(0, ".")\n')
        f.write('from unified_pipeline import UnifiedPipeline\n\n')
        f.write("CATEGORIES = {\n")
        for cat, tests in cats.items():
            f.write(f'    "{cat}": [\n')
            for sent, expected in tests:
                f.write(f"        ({repr(sent)}, {repr(expected)}),\n")
            f.write("    ],\n")
        f.write("}\n\n")
        f.write("""
def main():
    p = UnifiedPipeline()
    cat_results = {}
    total_found = 0
    total_expected = 0
    t0 = time.time()
    for cat, tests in CATEGORIES.items():
        found = 0
        missed = []
        for sent, expected in tests:
            results = p.check(sent)
            rule_ids = [r["rule_id"] for r in results]
            # Check if any rule matches
            matched = False
            for r in results:
                if cat.upper() in r["rule_id"].upper() or r["rule_id"].upper() in cat.upper():
                    matched = True
                    break
                # Also check specific mappings
                if cat == "DIDNT_PAST_FORM" and r["rule_id"] == "DIDNT_PAST_FORM":
                    matched = True
                    break
                if cat == "WRONG_PAST_PARTICIPLE" and r["rule_id"] == "WRONG_PAST_PARTICIPLE":
                    matched = True
                    break
                if cat == "MODAL_WRONG_FORM" and r["rule_id"] == "MODAL_WRONG_FORM":
                    matched = True
                    break
                if cat == "RUN_ON_SENTENCE" and r["rule_id"] == "RUN_ON_SENTENCE":
                    matched = True
                    break
                if cat == "COMMA_SPLICE" and r["rule_id"] == "COMMA_SPLICE":
                    matched = True
                    break
                if cat == "DOUBLE_COMPARATIVE" and r["rule_id"] in ("DOUBLE_COMPARATIVE", "DOUBLE_SUPERLATIVE"):
                    matched = True
                    break
                if cat == "EXPLAIN_DATIVE" and r["rule_id"] == "EXPLAIN_DATIVE":
                    matched = True
                    break
                if cat == "TO_PAST_PARTICIPLE" and r["rule_id"] == "TO_PAST_PARTICIPLE":
                    matched = True
                    break
                if cat == "MISSING_ARTICLE" and r["rule_id"] in ("MISSING_ARTICLE", "MISSING_ARTICLE_PHRASES"):
                    matched = True
                    break
                if cat == "MISSING_APOSTROPHE" and r["rule_id"] in ("MISSING_APOSTROPHE", "POSSESSIVE_APOSTROPHE"):
                    matched = True
                    break
                if cat == "MISSING_COMMA" and r["rule_id"] == "MISSING_COMMA_INTRO":
                    matched = True
                    break
                if cat == "PREPOSITION_COLLOCATION" and r["rule_id"] == "PREPOSITION_COLLOCATION":
                    matched = True
                    break
                if cat == "PARALLELISM" and r["rule_id"] == "PARALLELISM":
                    matched = True
                    break
                if cat == "ARTICLE_MISSING" and r["rule_id"] in ("ARTICLE_MISSING", "ARTICLE_A_AN", "MISSING_ARTICLE"):
                    matched = True
                    break
                if cat == "DO_SUPPORT" and r["rule_id"] == "DO_SUPPORT":
                    matched = True
                    break
                if cat == "SVA" and r["rule_id"] in ("SVA", "EXISTENTIAL_THERE_SVA", "MASS_NOUN_SVA"):
                    matched = True
                    break
                if cat == "TENSE_CONSISTENCY" and r["rule_id"] == "TENSE_CONSISTENCY":
                    matched = True
                    break
            if matched:
                found += 1
            else:
                missed.append((sent, expected, rule_ids))
        cat_results[cat] = (len(tests), found, missed)
        total_found += found
        total_expected += len(tests)

    elapsed = time.time() - t0
    print(f"")
    print(f"=== CATEGORY DETECTION RESULTS ===")
    print(f"{'Category':<25} {'Found':>6} {'Total':>6} {'Rate':>7}")
    print("-" * 50)
    for cat, (total, found, _) in sorted(cat_results.items()):
        rate = found / total * 100 if total > 0 else 0
        print(f"{cat:<25} {found:>6} {total:>6} {rate:>6.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_found:>6} {total_expected:>6} {total_found/total_expected*100:>6.1f}%")
    print(f"Time: {elapsed:.1f}s")

    # Print missed errors
    print(f"")
    print(f"=== MISSED ERRORS BY CATEGORY ===")
    for cat, (total, found, missed) in sorted(cat_results.items()):
        if missed:
            print(f"\\n{cat} ({total - found} missed):")
            for sent, expected, rules in missed[:5]:
                print(f'  "{sent[:80]}..." -> expected: {expected[:30]}')
                print(f'    Detected: {rules}')
            if len(missed) > 5:
                print(f"  ... and {len(missed) - 5} more")

if __name__ == "__main__":
    main()
""")
    print(f"test_categories_comprehensive.py: {total_tests} tests across {len(cats)} categories")


if __name__ == "__main__":
    main()
