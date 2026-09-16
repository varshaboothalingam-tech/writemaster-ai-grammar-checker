"""Check which expected errors are still missing."""
import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()

text = """My cousin she was telling me about her new car and she also say that she going to sell her old car because it have many problem.

When I ask her why she selling it, she tell me because the car was too old and it didn't started properly since two years. She also say that she was not satisfy with the car because the engine make too much noise and she didn't had enough money to fix it.

My cousin she was very upset because she have to spend lot of money on the car every month. She say that she working hard but the car keep break down and she can't afford it anymore. She was also worry about the safety because the brakes was not working properly and she have a big family who she need to drive to school every day.

She ask me if I know someone who want to buy a cheap car. I tell her that my friend have a brother who was looking for a car. She was very happy to hear that and she ask me to give her the brother number. I say I will call him and let her know.

My cousin she was very relief when I tell her that my friend brother was interested in buying her car. She say that she going to give him a good price because she want to get rid of it fast. She was also happy because she can finally buy a new car that she can rely on."""

issues = p.check(text)
found_rules = set()
found_originals = set()
for e in issues:
    found_rules.add(e['rule_id'])
    found_originals.add(e['original'].lower().strip())
    print(f"  {e['rule_id']:30s} '{e['original']}' -> '{e['replacement']}' conf={e['confidence']:.0%}")

print(f"\n--- Missing checks ---")
missing = [
    ("didn't started", "didn't start", "past_participle"),
    ("since two years", "for two years", "since_for_duration"),
    ("lot of money", "a lot of money", "article_missing"),
    ("brakes was", "brakes were", "SVA"),
    ("who want", "who wants", "SVA"),
    ("I tell", "I told", "tense"),
    ("have a brother", "has a brother", "SVA"),
    ("was not satisfy", "was not satisfied", "past_participle"),
    ("she was also worry", "was also worried", "past_participle"),
    ("keep break", "keeps breaking", "SVA+tense"),
]
for orig, repl, cat in missing:
    found = any(orig.lower() in e['original'].lower() or orig.lower() in e.get('explanation','').lower() for e in issues)
    status = "FOUND" if found else "MISSING"
    print(f"  {status:8s} '{orig}' -> '{repl}' ({cat})")
