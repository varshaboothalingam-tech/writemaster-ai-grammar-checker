from new_pipeline_runner import check_text_new

tests = [
    "Yesterday my friend and me goes to a large stopping mall.",
    "The dog run fast.",
    "She go to school every day.",
    "I wanted to improve my writing.",
    "The team was many players short.",
    "She can sings well.",
    "He must to go.",
    "They should went home.",
    "Me and him went to the store.",
    "Its a nice day.",
    "Your welcome.",
    "I could of gone.",
    "She use to go there.",
    "He was suppose to come.",
    "The company was founded in 2010.",
    "I remember when we first met.",
    "He let the cat out of the bag.",
    "She hit the nail on the head.",
    "There has been a significant change.",
    "None of them are coming.",
]
for t in tests:
    issues = check_text_new(t)
    print(f'"{t[:60]}"')
    for i in issues:
        print(f'  -> {i["original_text"]} -> {i["replacement"]} (conf={i["confidence"]}, cat={i["category"]})')
    if not issues:
        print("  -> No issues")
    print()
