import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline, get_nlp, NLPDetector, deduplicate

p = UnifiedPipeline()
nlp = get_nlp()

t = "team have worked on it"
clean, _ = p.preprocessor.protect(t)
doc = nlp(clean)

nlp_c = p.nlp_detector.detect(clean, doc)
print(f"NLPDetector: {len(nlp_c)} candidates")
for c in nlp_c:
    print(f"  {c.rule_id}: '{c.original}' -> '{c.replacement}' subject='{c.metadata.get('subject', 'N/A')}' conf={c.raw_confidence:.2f}")

# Check what strategy 6 does for "team"
for c in nlp_c:
    suppressed, reason = p.context_engine.should_suppress(c, clean, doc)
    conf = p.confidence_engine.score(c, (suppressed, reason))
    print(f"  Context: suppressed={suppressed} reason='{reason}' conf={conf:.2f}")
    if conf >= 0.70:
        c.raw_confidence = conf

fp = p.fp_filter.filter(nlp_c, clean, doc)
print(f"After FP filter: {len(fp)}")
for c in fp:
    print(f"  {c.rule_id}: '{c.original}' -> '{c.replacement}'")

# Check if "team" triggers collective noun suppression
print(f"\nChecking Strategy 6 for 'team'...")
for c in nlp_c:
    subj = c.metadata.get("subject", "").lower()
    print(f"  Subject from metadata: '{subj}'")
    # Check if "team" is in simple_collectives
    simple_collectives = {"team", "family", "jury", "orchestra", "committee", "staff",
                "group", "class", "audience", "crowd", "government", "company"}
    if subj in simple_collectives:
        print(f"  MATCH: '{subj}' is in simple_collectives -> will be suppressed!")
        for tok in doc:
            if tok.lower_ == subj and tok.dep_ in ("nsubj", "nsubjpass"):
                has_of = any(ch.lower_ == "of" for ch in tok.children)
                print(f"  Token '{tok}' dep={tok.dep_} has_of={has_of}")
                if not has_of:
                    print(f"  -> WOULD BE FILTERED!")
