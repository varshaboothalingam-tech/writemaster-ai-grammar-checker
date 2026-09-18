"""Regression test: the user's 200-word messy diary paragraph must surface
all the expected errors via local rules (offline, no AI)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.master import check_master

TEXT = (
    "Last sunday my friends and I go to the park. There are many peoples there. "
    "children were enjoying theirselves. Ravi don't likes mango. he suggest to go to the zoo. "
    "everyone was very happy. I take out my phone and take pictures. my friends was playing football. "
    "The birds flyed in the sky. One bird forgetted its food and the other birds was jumping. "
    "There was a man who were feeding the birds. He tell us about the birds. He say that birds can fly. "
    "We listen carefully and enjoy watching them. We take a group photo. we will come again next week. "
    "It was the most enjoyable day we have ever spend. The childrens was very happy. "
    "I didn't wanted to leave. my mothers was watching us. The monkeys was jumping on the trees. "
    "A elephant was eating bananas. The zoo keeper feeded the animals. We seed a lion. "
    "The lion was slepping. We taked many pictures. my friends and me enjoyed the day. "
    "We hope we will visit the zoo again. It was a wonderfull day for all of us. everyone enjoyed much."
)

EXPECTED = {
    "sunday": "Sunday",
    "go": "went",
    "peoples": "people",
    "theirselves": "themselves",
    "don't": "doesn't",     # Ravi don't
    "likes": "like",
    "suggest": "suggested",
    "take": "took",       # I take out my phone
    "was": "were",        # my friends was
    "flyed": "flew",
    "forgetted": "forgot",
    "were": "was",        # a man who were
    "tell": "told",
    "say": "said",
    "take": "took",       # we take a group photo
    "will": "would",
    "spend": "spent",
    "childrens": "children",
    "taked": "took",
    "wonderfull": "wonderful",
}


def main():
    res = check_master(TEXT, use_ai=False)
    errors = res["errors"]
    by_word = {}
    for e in errors:
        by_word.setdefault(e["wrong"], []).append(e["correct"])
    missing = 0
    for wrong, corrects in EXPECTED.items():
        if wrong not in by_word:
            print(f"MISSING category '{wrong}' (wanted one of {corrects})")
            missing += 1
        elif corrects not in by_word[wrong]:
            print(f"WEAK '{wrong}': got {by_word[wrong]}, expected {corrects}")
            missing += 1
    print(f"total errors reported: {len(errors)}")
    print(f"corrected_text:\n{res['corrected_text']}")
    if missing:
        print(f"{missing} expected-fix categories missing/weak")
        raise SystemExit(1)
    print("OK: all expected fixes present")


if __name__ == "__main__":
    main()