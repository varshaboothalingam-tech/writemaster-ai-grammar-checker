import sys
sys.path.insert(0, r'C:\Users\hemal\OneDrive\Documents\grammar_checker')
from unified_pipeline import UnifiedPipeline
p = UnifiedPipeline()
s = "Yesterday my friend and me goes to a large stopping mall because we waited to buy some new clothe for the winter season. The mall have a lot of shops that sells different kind of things, and we was very exciting to go inside. When we reached at the mall, we see that there was to many people waiting in line for the food court. My friend tells me that he already ate a breakfast, but I was so hungry that I wanted to ate something immediately. We was walking around the mall for almost two hours but we couldn't found the store that we was looking for. Eventually, we get tired and decides to set down on a bench near the escalator. A women who was working at the information desk come over to ask us if we need some helps. I explained her that we was looking for a particular store but we was unable to find it. She said us that the store we was searching for had already closed down since last months. We was very disappointing because we come all the way just for that store. After that, we decide to go to a restaurant in the mall but the food was very badly. My friend said that the food tasted like it was making by someone who doesn't knows how to cook properly. We both agree that we will not never come back to this mall again because our experience was very poor."
issues = p.check(s)
print(f"Found {len(issues)} issues")
for e in issues:
    print(f"  [{e['start']}:{e['end']}] \"{e['original']}\" -> \"{e['replacement']}\" [{e['rule_id']}] conf={e['confidence']}")
