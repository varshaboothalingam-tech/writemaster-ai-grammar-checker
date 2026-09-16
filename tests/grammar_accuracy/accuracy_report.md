# Grammar Checker Accuracy Report

**Generated:** 2026-08-20 11:01:54
**Total Test Cases:** 200

## Metrics

| Metric | Value |
|--------|-------|
| True Positives (correctly flagged errors) | 33 |
| True Negatives (correctly unflagged text) | 98 |
| False Positives (incorrectly flagged text) | 10 |
| False Negatives (missed errors) | 59 |
| **Precision** | 0.767 |
| **Recall** | 0.359 |
| **F1 Score** | 0.489 |
| **False Positive Rate** | 0.093 |

## Interpretation

- **Precision** = 76.7%: Of all texts flagged as errors, 76.7% actually contain errors.
- **Recall** = 35.9%: Of all actual errors, 35.9% were detected.
- **F1** = 0.489: Harmonic mean of precision and recall.
- **FPR** = 9.3%: 9.3% of correct texts were incorrectly flagged.

## False Positives (most critical)

- **Case 23**: `I will call you when I arrive.`
- **Case 30**: `The children played happily in the backyard.`
- **Case 91**: `The cat's paw was injured.`
- **Case 92**: `The cats' toys were scattered.`
- **Case 106**: `He ain't going to like this.`
- **Case 131**: `The company's profits increased this quarter.`
- **Case 132**: `The companies' headquarters are in New York.`
- **Case 140**: `The teacher's pet is a golden retriever.`
- **Case 147**: `She works at Google.`
- **Case 181**: `The teacher said that we need to study.`

## False Negatives (missed errors)

- **Case 57**: `He went to the store yesterday and buyed some food.` (expected: GRAMMAR)
- **Case 59**: `She has went to Paris twice.` (expected: GRAMMAR)
- **Case 60**: `I seen that movie last week.` (expected: GRAMMAR)
- **Case 62**: `Their going to the party tonight.` (expected: GRAMMAR)
- **Case 63**: `I could of done better.` (expected: GRAMMAR)
- **Case 64**: `She is more better than me at math.` (expected: GRAMMAR)
- **Case 65**: `The team are going to win the game.` (expected: GRAMMAR)
- **Case 67**: `She lays on the couch all day.` (expected: GRAMMAR)
- **Case 68**: `I have less books than you.` (expected: GRAMMAR)
- **Case 71**: `She sings beautiful.` (expected: GRAMMAR)
- **Case 72**: `He did good on the test.` (expected: GRAMMAR)
- **Case 73**: `I am more hungrier than before.` (expected: GRAMMAR)
- **Case 74**: `Between you and I, this is a secret.` (expected: GRAMMAR)
- **Case 75**: `The reason is because he was late.` (expected: GRAMMAR)
- **Case 77**: `The group of students were studying.` (expected: GRAMMAR)
- **Case 80**: `The news are surprising.` (expected: GRAMMAR)
- **Case 81**: `He runned quickly to the store.` (expected: SPELLING)
- **Case 84**: `I need a informations about this topic.` (expected: GRAMMAR)
- **Case 86**: `She was to the store yesterday.` (expected: GRAMMAR)
- **Case 87**: `The team are playing well this season.` (expected: GRAMMAR)
- **Case 89**: `She is more smarter than her brother.` (expected: GRAMMAR)
- **Case 90**: `I am feeling more better today.` (expected: GRAMMAR)
- **Case 94**: `The dog wagged it's tail.` (expected: GRAMMAR)
- **Case 95**: `Your going to love this movie.` (expected: GRAMMAR)
- **Case 97**: `The teacher grading there papers.` (expected: GRAMMAR)
- **Case 101**: `I can't hardly wait for the concert.` (expected: GRAMMAR)
- **Case 102**: `She's doing good in school.` (expected: GRAMMAR)
- **Case 104**: `I seen him at the store yesterday.` (expected: GRAMMAR)
- **Case 105**: `She was real happy about the news.` (expected: GRAMMAR)
- **Case 107**: `The information were correct.` (expected: GRAMMAR)
- **Case 109**: `He was to the store and buyed some food.` (expected: GRAMMAR)
- **Case 111**: `Me and him went to the store.` (expected: GRAMMAR)
- **Case 113**: `Between you and I, this is a secret.` (expected: GRAMMAR)
- **Case 115**: `The cake was made by she.` (expected: GRAMMAR)
- **Case 118**: `She is more better than him.` (expected: GRAMMAR)
- **Case 119**: `He is the most tallest person in the class.` (expected: GRAMMAR)
- **Case 120**: `This is the bestest pizza I've ever had.` (expected: SPELLING)
- **Case 134**: `The dog wagged it's tail happily.` (expected: GRAMMAR)
- **Case 136**: `Your welcome for the help.` (expected: GRAMMAR)
- **Case 152**: `She has went to the store.` (expected: GRAMMAR)
- **Case 153**: `He was went to the store.` (expected: GRAMMAR)
- **Case 154**: `They have went to the store.` (expected: GRAMMAR)
- **Case 155**: `We will went to the store.` (expected: GRAMMAR)
- **Case 156**: `I am went to the store.` (expected: GRAMMAR)
- **Case 157**: `She had went to the store.` (expected: GRAMMAR)
- **Case 158**: `He has went to the store three times.` (expected: GRAMMAR)
- **Case 159**: `They will went to the store tomorrow.` (expected: GRAMMAR)
- **Case 160**: `I have went to the store before.` (expected: GRAMMAR)
- **Case 161**: `She is more prettier than her sister.` (expected: GRAMMAR)
- **Case 162**: `He is most tallest person in the class.` (expected: GRAMMAR)
- **Case 163**: `This is the bestest day ever.` (expected: SPELLING)
- **Case 164**: `She is the most smartest student.` (expected: GRAMMAR)
- **Case 165**: `He is very more intelligent than me.` (expected: GRAMMAR)
- **Case 166**: `The view was very much beautiful.` (expected: GRAMMAR)
- **Case 171**: `I could of gone to the store.` (expected: GRAMMAR)
- **Case 172**: `She should of called me.` (expected: GRAMMAR)
- **Case 173**: `He would of been here on time.` (expected: GRAMMAR)
- **Case 174**: `They might of gone to the park.` (expected: GRAMMAR)
- **Case 175**: `I must of forgotten my keys.` (expected: GRAMMAR)

## Detailed Results

| ID | Status | Input | Issues | Expected Category |
|----|--------|-------|--------|-------------------|
| 1 | TN | `The cat sat on the mat.` | 0 |  |
| 2 | TN | `She goes to school every day.` | 0 |  |
| 3 | TN | `I have a beautiful garden.` | 0 |  |
| 4 | TN | `He is reading a book.` | 0 |  |
| 5 | TN | `They were playing in the park.` | 0 |  |
| 6 | TN | `The weather is nice today.` | 0 |  |
| 7 | TN | `We need to finish this project.` | 0 |  |
| 8 | TN | `She bought a new car yesterday.` | 0 |  |
| 9 | TN | `The children are studying for their exams.` | 0 |  |
| 10 | TN | `I would like some water, please.` | 0 |  |
| 11 | TN | `My friend has a new phone.` | 0 |  |
| 12 | TN | `He said that it's very good.` | 0 |  |
| 13 | TN | `He doesn't know how to use all the features.` | 0 |  |
| 14 | TN | `He is learning fast.` | 0 |  |
| 15 | TN | `He told me that your phone is better than his.` | 0 |  |
| 16 | TN | `My phone works perfectly good.` | 0 |  |
| 17 | TN | `Yesterday I went to the store.` | 0 |  |
| 18 | TN | `She can speak three languages fluently.` | 0 |  |
| 19 | TN | `The movie was very entertaining.` | 0 |  |
| 20 | TN | `We should leave early tomorrow morning.` | 0 |  |
| 21 | TN | `The students were writing outside while they waited for the ` | 0 |  |
| 22 | TN | `She has been working on this project for months.` | 0 |  |
| 23 | FP | `I will call you when I arrive.` | 1 |  |
| 24 | TN | `The restaurant serves excellent Italian food.` | 0 |  |
| 25 | TN | `He has been to Paris three times.` | 0 |  |
| 26 | TN | `The concert was sold out last night.` | 0 |  |
| 27 | TN | `She decided to take a different route.` | 0 |  |
| 28 | TN | `The book I borrowed from the library was fascinating.` | 0 |  |
| 29 | TN | `He spent the afternoon reading in the garden.` | 0 |  |
| 30 | FP | `The children played happily in the backyard.` | 1 |  |
| 31 | TN | `I need to buy groceries for dinner tonight.` | 0 |  |
| 32 | TN | `She always arrives on time for meetings.` | 0 |  |
| 33 | TN | `The train departs at eight o'clock every morning.` | 0 |  |
| 34 | TN | `He finished his homework before watching TV.` | 0 |  |
| 35 | TN | `The garden was full of beautiful flowers.` | 0 |  |
| 36 | TN | `We enjoyed our vacation at the beach.` | 0 |  |
| 37 | TN | `She promised to call me later today.` | 0 |  |
| 38 | TN | `The dog chased the ball across the yard.` | 0 |  |
| 39 | TN | `I have been thinking about this problem all day.` | 0 |  |
| 40 | TN | `He said he would be here by noon.` | 0 |  |
| 41 | TN | `The students are preparing for their final exams.` | 0 |  |
| 42 | TN | `She loves to cook Italian dishes.` | 0 |  |
| 43 | TN | `The airplane landed safely at the airport.` | 0 |  |
| 44 | TN | `He works as a software engineer at a tech company.` | 0 |  |
| 45 | TN | `The library has an extensive collection of rare books.` | 0 |  |
| 46 | TN | `They celebrated their anniversary with a special dinner.` | 0 |  |
| 47 | TN | `She was disappointed by the results of the experiment.` | 0 |  |
| 48 | TN | `The mountain trail was challenging but rewarding.` | 0 |  |
| 49 | TN | `He apologized for arriving late to the meeting.` | 0 |  |
| 50 | TN | `The software update fixed several bugs in the system.` | 0 |  |
| 51 | TP | `Yesterday my friend and me goes to a large stopping mall.` | 3 | GRAMMAR |
| 52 | TP | `Yesterday my friend and me goes to a large stopping mall.` | 3 | GRAMMAR |
| 53 | TP | `Yesterday my friend and me goes to a large stopping mall.` | 3 | CONTEXTUAL_WORD_USAGE |
| 54 | TP | `I recieve alot of emails every day.` | 3 | SPELLING |
| 55 | TP | `I recieve alot of emails every day.` | 3 | SPELLING |
| 56 | TP | `She dont know the answer.` | 1 | GRAMMAR |
| 57 | FN | `He went to the store yesterday and buyed some food.` | 0 | GRAMMAR |
| 58 | TP | `The childrens were playing in the garden.` | 1 | SPELLING |
| 59 | FN | `She has went to Paris twice.` | 0 | GRAMMAR |
| 60 | FN | `I seen that movie last week.` | 0 | GRAMMAR |
| 61 | TP | `He dont have no money.` | 1 | GRAMMAR |
| 62 | FN | `Their going to the party tonight.` | 0 | GRAMMAR |
| 63 | FN | `I could of done better.` | 0 | GRAMMAR |
| 64 | FN | `She is more better than me at math.` | 0 | GRAMMAR |
| 65 | FN | `The team are going to win the game.` | 0 | GRAMMAR |
| 66 | TP | `Him and me went to the store.` | 1 | GRAMMAR |
| 67 | FN | `She lays on the couch all day.` | 0 | GRAMMAR |
| 68 | FN | `I have less books than you.` | 0 | GRAMMAR |
| 69 | TN | `The data shows that the plan is working.` | 0 |  |
| 70 | TN | `Each student must submit their assignment on time.` | 0 |  |
| 71 | FN | `She sings beautiful.` | 0 | GRAMMAR |
| 72 | FN | `He did good on the test.` | 0 | GRAMMAR |
| 73 | FN | `I am more hungrier than before.` | 0 | GRAMMAR |
| 74 | FN | `Between you and I, this is a secret.` | 0 | GRAMMAR |
| 75 | FN | `The reason is because he was late.` | 0 | GRAMMAR |
| 76 | TN | `She is one of those people who is always late.` | 0 |  |
| 77 | FN | `The group of students were studying.` | 0 | GRAMMAR |
| 78 | TP | `Neither the teacher nor the students was ready.` | 1 | GRAMMAR |
| 79 | TP | `Everyone have their own opinion.` | 1 | GRAMMAR |
| 80 | FN | `The news are surprising.` | 0 | GRAMMAR |
| 81 | FN | `He runned quickly to the store.` | 0 | SPELLING |
| 82 | TN | `She is a very good dancer.` | 0 |  |
| 83 | TP | `The informations was helpful.` | 1 | GRAMMAR |
| 84 | FN | `I need a informations about this topic.` | 0 | GRAMMAR |
| 85 | TP | `He have went to the store.` | 1 | GRAMMAR |
| 86 | FN | `She was to the store yesterday.` | 0 | GRAMMAR |
| 87 | FN | `The team are playing well this season.` | 0 | GRAMMAR |
| 88 | TN | `He did a great job on the project.` | 0 |  |
| 89 | FN | `She is more smarter than her brother.` | 0 | GRAMMAR |
| 90 | FN | `I am feeling more better today.` | 0 | GRAMMAR |
| 91 | FP | `The cat's paw was injured.` | 1 |  |
| 92 | FP | `The cats' toys were scattered.` | 1 |  |
| 93 | TP | `Its a beautiful day outside.` | 1 | GRAMMAR |
| 94 | FN | `The dog wagged it's tail.` | 0 | GRAMMAR |
| 95 | FN | `Your going to love this movie.` | 0 | GRAMMAR |
| 96 | TN | `I love your new dress.` | 0 |  |
| 97 | FN | `The teacher grading there papers.` | 0 | GRAMMAR |
| 98 | TN | `They're going to the store.` | 0 |  |
| 99 | TN | `We're going to be late.` | 0 |  |
| 100 | TP | `Its important to exercise daily.` | 1 | GRAMMAR |
| 101 | FN | `I can't hardly wait for the concert.` | 0 | GRAMMAR |
| 102 | FN | `She's doing good in school.` | 0 | GRAMMAR |
| 103 | TP | `He brung his friend to the party.` | 1 | SPELLING |
| 104 | FN | `I seen him at the store yesterday.` | 0 | GRAMMAR |
| 105 | FN | `She was real happy about the news.` | 0 | GRAMMAR |
| 106 | FP | `He ain't going to like this.` | 1 |  |
| 107 | FN | `The information were correct.` | 0 | GRAMMAR |
| 108 | TP | `She don't know what she's talking about.` | 1 | GRAMMAR |
| 109 | FN | `He was to the store and buyed some food.` | 0 | GRAMMAR |
| 110 | TN | `The news is very important.` | 0 |  |
| 111 | FN | `Me and him went to the store.` | 0 | GRAMMAR |
| 112 | TP | `Her and me are going to the park.` | 1 | GRAMMAR |
| 113 | FN | `Between you and I, this is a secret.` | 0 | GRAMMAR |
| 114 | TP | `She is taller then me.` | 1 | GRAMMAR |
| 115 | FN | `The cake was made by she.` | 0 | GRAMMAR |
| 116 | TP | `He is smarter then his brother.` | 1 | GRAMMAR |
| 117 | TN | `The book is on the table.` | 0 |  |
| 118 | FN | `She is more better than him.` | 0 | GRAMMAR |
| 119 | FN | `He is the most tallest person in the class.` | 0 | GRAMMAR |
| 120 | FN | `This is the bestest pizza I've ever had.` | 0 | SPELLING |
| 121 | TP | `She wasnt at the meeting yesterday.` | 1 | SPELLING |
| 122 | TP | `I dont have any money.` | 1 | SPELLING |
| 123 | TP | `He cant come to the party.` | 2 | SPELLING |
| 124 | TP | `She wouldnt tell me the truth.` | 2 | SPELLING |
| 125 | TP | `I didnt see him at the store.` | 1 | SPELLING |
| 126 | TP | `He isnt coming to the meeting.` | 1 | SPELLING |
| 127 | TP | `They wont be at the party.` | 1 | SPELLING |
| 128 | TP | `She hasnt finished her work yet.` | 1 | SPELLING |
| 129 | TP | `I havent seen him in a long time.` | 1 | SPELLING |
| 130 | TP | `He couldnt believe his eyes.` | 2 | SPELLING |
| 131 | FP | `The company's profits increased this quarter.` | 1 |  |
| 132 | FP | `The companies' headquarters are in New York.` | 1 |  |
| 133 | TP | `Its a beautiful day outside.` | 1 | GRAMMAR |
| 134 | FN | `The dog wagged it's tail happily.` | 0 | GRAMMAR |
| 135 | TN | `You're doing a great job!` | 0 |  |
| 136 | FN | `Your welcome for the help.` | 0 | GRAMMAR |
| 137 | TN | `The children's toys were scattered.` | 0 |  |
| 138 | TP | `The childrens toys were scattered.` | 1 | SPELLING |
| 139 | TP | `Its important to save money.` | 1 | GRAMMAR |
| 140 | FP | `The teacher's pet is a golden retriever.` | 1 |  |
| 141 | TN | `Dr. Smith is a very good doctor.` | 0 |  |
| 142 | TN | `I work at IBM.` | 0 |  |
| 143 | TN | `She lives on Main Street.` | 0 |  |
| 144 | TN | `The United States is a large country.` | 0 |  |
| 145 | TN | `He graduated from Harvard University.` | 0 |  |
| 146 | TN | `The Pacific Ocean is very deep.` | 0 |  |
| 147 | FP | `She works at Google.` | 1 |  |
| 148 | TN | `The Amazon River is very long.` | 0 |  |
| 149 | TN | `He studied at MIT.` | 0 |  |
| 150 | TN | `The Rocky Mountains are beautiful.` | 0 |  |
| 151 | TN | `I went to the store yesterday.` | 0 |  |
| 152 | FN | `She has went to the store.` | 0 | GRAMMAR |
| 153 | FN | `He was went to the store.` | 0 | GRAMMAR |
| 154 | FN | `They have went to the store.` | 0 | GRAMMAR |
| 155 | FN | `We will went to the store.` | 0 | GRAMMAR |
| 156 | FN | `I am went to the store.` | 0 | GRAMMAR |
| 157 | FN | `She had went to the store.` | 0 | GRAMMAR |
| 158 | FN | `He has went to the store three times.` | 0 | GRAMMAR |
| 159 | FN | `They will went to the store tomorrow.` | 0 | GRAMMAR |
| 160 | FN | `I have went to the store before.` | 0 | GRAMMAR |
| 161 | FN | `She is more prettier than her sister.` | 0 | GRAMMAR |
| 162 | FN | `He is most tallest person in the class.` | 0 | GRAMMAR |
| 163 | FN | `This is the bestest day ever.` | 0 | SPELLING |
| 164 | FN | `She is the most smartest student.` | 0 | GRAMMAR |
| 165 | FN | `He is very more intelligent than me.` | 0 | GRAMMAR |
| 166 | FN | `The view was very much beautiful.` | 0 | GRAMMAR |
| 167 | TN | `She sings very beautifully.` | 0 |  |
| 168 | TN | `He runs very quickly.` | 0 |  |
| 169 | TN | `The food was extremely delicious.` | 0 |  |
| 170 | TN | `She is incredibly talented.` | 0 |  |
| 171 | FN | `I could of gone to the store.` | 0 | GRAMMAR |
| 172 | FN | `She should of called me.` | 0 | GRAMMAR |
| 173 | FN | `He would of been here on time.` | 0 | GRAMMAR |
| 174 | FN | `They might of gone to the park.` | 0 | GRAMMAR |
| 175 | FN | `I must of forgotten my keys.` | 0 | GRAMMAR |
| 176 | TN | `She can have gone to the store.` | 0 |  |
| 177 | TN | `He may have gone to the store.` | 0 |  |
| 178 | TN | `They will have gone by tomorrow.` | 0 |  |
| 179 | TN | `I should have studied harder.` | 0 |  |
| 180 | TN | `She would have been here if she could.` | 0 |  |
| 181 | FP | `The teacher said that we need to study.` | 1 |  |
| 182 | TN | `She asked me if I wanted to go.` | 0 |  |
| 183 | TN | `He told me that he was tired.` | 0 |  |
| 184 | TN | `They said that the meeting was cancelled.` | 0 |  |
| 185 | TN | `We asked if they could help us.` | 0 |  |
| 186 | TN | `She said she would be here soon.` | 0 |  |
| 187 | TN | `He told her that he loved her.` | 0 |  |
| 188 | TN | `They asked us to help them move.` | 0 |  |
| 189 | TN | `We told them to come early.` | 0 |  |
| 190 | TN | `She asked him if he was coming.` | 0 |  |
| 191 | TN | `I have been waiting for two hours.` | 0 |  |
| 192 | TN | `She has been working hard all day.` | 0 |  |
| 193 | TN | `He has been studying for his exams.` | 0 |  |
| 194 | TN | `They have been playing soccer.` | 0 |  |
| 195 | TN | `We have been waiting for the bus.` | 0 |  |
| 196 | TN | `She has been reading that book for weeks.` | 0 |  |
| 197 | TN | `He has been working on this project.` | 0 |  |
| 198 | TN | `They have been living here for years.` | 0 |  |
| 199 | TN | `We have been friends since childhood.` | 0 |  |
| 200 | TN | `She has been very helpful.` | 0 |  |
