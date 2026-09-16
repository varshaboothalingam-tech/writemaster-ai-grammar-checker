"""Summary of detection accuracy."""
detected = {
    1: ["SVA:goes->go", "SVA:needs->need"],
    2: ["DO_SUPPORT:don't->doesn't", "SVA:keeps->keep"],
    3: ["SVA:was->were", "SVA:need->needs"],
    4: ["SVA:prefers->prefer"],
    5: ["TENSE:says->said"],
    8: ["SVA:are->is"],
    9: ["SINCE:since->for"],
    10: ["DO_SUPPORT:don't->doesn't", "ONE_OF:employee->employees"],
    11: ["SVA:were->was", "PREP:discussed about->discussed"],
    12: ["SVA:were->was"],
    13: ["DIDNT_PAST:went->go"],
    14: ["DOUBLE_MODAL:would will->would"],
    16: ["SVA:has->have"],
    18: ["THERE_SVA:were->was"],
    19: ["ADJ_PREP:in->at"],
    22: ["SVA:was->were"],
    25: ["STATIVE_BE"],
    26: ["PARALLELISM"],
    27: ["SVA:were->was"],
    30: ["PRONOUN:me->I"],
    31: ["PREP:discussed about->discussed"],
    33: ["MODAL:gets->get", "DIDNT_PAST:told->tell"],
    34: ["MODAL:helps->help", "PRONOUN:me->I", "DOUBLE_MODAL"],
    36: ["SPELL:informations->information"],
    38: ["MODAL:starts->start", "SVA:needs->need", "PARALLELISM"],
    41: ["SVA:have->has"],
    43: ["TENSE_PAST:are->areed (wrong label)"],
    49: ["SVA:have->has"],
}

missed = {
    6: ["PAST_PART:had to waited->wait", "TENSE:already left->had already left"],
    7: ["EMBEDDED_Q:where was I->where I was", "DIDNT_PAST:didn't knew->didn't know"],
    8: ["SVA:provide->provides"],
    12: ["SVA:was incorrect->were incorrect"],
    14: ["TENSE:will have->would have"],
    15: ["REDUNDANT:Although...but"],
    16: ["SVA其它问题"],
    17: ["PREP:explained me->explained to me", "EMBEDDED_Q:how does the system works"],
    20: ["UNCOUNTABLE:works->work"],
    21: ["TENSE:has launched...last month->launched", "UNCOUNTABLE:feedbacks->feedback"],
    23: ["ONE_OF:Every students->Every student"],
    24: ["MISSING APOSTROPHE WRONG (reaches is correct)"],
    26: ["SINCE:from 2021->since 2021 (detected but wrong label)"],
    28: ["SVA:uses->use", "SVA:have increased->has increased"],
    29: ["COMPARISON:than->to (prefer...to)"],
    32: ["SVA:is too many->are too many"],
    35: ["TENSE:have checked->had checked"],
    37: ["WORD_USAGE:hardly->hard"],
    39: ["PRONOUN:none->any"],
    40: ["PARALLELISM:rather than to publish->publishing"],
    42: ["DIDNT_PAST:didn't understood->didn't understand"],
    44: ["MIXED_CONDITIONAL"],
    46: ["DIDNT_PAST:didn't expected->didn't expect"],
    47: ["PLURAL:every days->every day"],
    48: ["SVA:was making->were making"],
    50: ["PAST_PART:had forget->had forgotten", "PAST_PART:had to called->had to call"],
}

total_expected = sum(len(v) for v in {**detected, **missed}.values())
total_detected = sum(len(v) for v in detected.values())
total_missed = sum(len(v) for v in missed.values())
print(f"Expected: {total_expected}")
print(f"Detected: {total_detected}")
print(f"Missed:   {total_missed}")
print(f"Rate:     {total_detected}/{total_expected} = {total_detected/total_expected*100:.1f}%")
print(f"\nMissed by category:")
cats = {}
for s, items in missed.items():
    for item in items:
        cat = item.split(":")[0]
        cats.setdefault(cat, []).append(f"S{s}: {item}")
for cat, items in sorted(cats.items(), key=lambda x: -len(x[1])):
    print(f"  {cat}: {len(items)}")
    for item in items:
        print(f"    {item}")
