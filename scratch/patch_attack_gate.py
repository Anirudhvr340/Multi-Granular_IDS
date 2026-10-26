path = 'nids/attack_gate.py'
text = open(path, 'r', encoding='utf-8').read()
old = 'conf_factor = 0.5 + 0.5 * math.tanh((record.confidence - 0.5) * 4)'
new = '''binary_conf = max(record.attack_probability, 1.0 - record.attack_probability)
                conf_factor = 0.5 + 0.5 * math.tanh((binary_conf - 0.5) * 4)'''
assert old in text, 'Target not found'
text = text.replace(old, new, 1)
open(path, 'w', encoding='utf-8').write(text)
print('Successfully updated attack_gate.py')

