"""Test the user's real writing text."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline

pipeline = UnifiedPipeline()

text = """My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem.

When I ask her why she selling it, she tell me because the car was too old and it didn't started properly since two years. She also say that she was not satisfy with the car because the engine make too much noise and she didn't had enough money to fix it.

My cousin she was very upset because she have to spend lot of money on the car every month. She say that she working hard but the car keep break down and she can't afford it anymore. She was also worry about the safety because the brakes was not working properly and she have a big family who she need to drive to school every day.

She ask me if I know someone who want to buy a cheap car. I tell her that my friend have a brother who was looking for a car. She was very happy to hear that and she ask me to give her the brother number. I say I will call him and let her know.

My cousin she was very relief when I tell her that my friend brother was interested in buying her car. She say that she going to give him a good price because she want to get rid of it fast. She was also happy because she can finally buy a new car that she can rely on."""

print("=" * 80)
print("USER TEXT ANALYSIS")
print("=" * 80)
print(f"\nText ({len(text)} chars):\n{text[:200]}...\n")

issues = pipeline.check(text)
print(f"Found {len(issues)} issues:\n")
for i, issue in enumerate(issues, 1):
    print(f"  {i:2d}. [{issue['rule_id']:25s}] ({issue['confidence']:.0%}) {issue['original']!r} -> {issue['replacement']!r}")
    print(f"      {issue['message']}")
print()

expected = [
    ("she was", "was", "remove redundant 'she'"),
    ("she also say", "she also says", "SVA"),
    ("she going", "she is going", "missing auxiliary"),
    ("it have", "it has", "SVA"),
    ("many problem", "many problems", "plural"),
    ("I ask", "I asked", "tense"),
    ("she tell", "she told", "tense"),
    ("it didn't started", "it didn't start", "past_participle"),
    ("since two years", "for two years", "preposition"),
    ("she have", "she has", "SVA"),
    ("lot of money", "a lot of money", "grammar"),
    ("keep break down", "keeps breaking down", "SVA+tense"),
    ("the brakes was", "the brakes were", "SVA"),
    ("she have a big family", "she has a big family", "SVA"),
    ("who want", "who wants", "SVA"),
    ("I tell", "I told", "tense"),
    ("have a brother", "has a brother", "SVA"),
    ("who was looking", "who was looking", "OK (correct)"),
    ("the brother number", "the brother's number", "possessive"),
    ("I will call", "I will call", "OK (correct)"),
    ("My cousin she was", "My cousin was", "remove redundant 'she'"),
    ("interested in buying", "interested in buying", "OK"),
    ("she going to give", "she is going to give", "missing auxiliary"),
    ("she can finally buy", "she could finally buy", "tense agreement"),
    ("she can rely on", "she could rely on", "tense agreement"),
]

print(f"Expected ~{len([e for e in expected if e[2] != 'OK (correct)'])} actual errors (excluding correct items)")
print(f"Detected: {len(issues)} issues")
