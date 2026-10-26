path = 'nids/rule_engine.py'
text = open(path, 'r', encoding='utf-8').read()

old = 'r6_min_prob: float = 0.45        # shifted from 0.35 to 0.45'
new = 'r6_min_prob: float = 0.38        # calibrated to intercept multi-model attacks (P >= 0.38)'

assert old in text, "Target not found"
text = text.replace(old, new, 1)
open(path, 'w', encoding='utf-8').write(text)
print("Successfully updated r6_min_prob in rule_engine.py")

