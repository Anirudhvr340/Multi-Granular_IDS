path = 'nids/adapters/time_window_adapter.py'
text = open(path, 'r', encoding='utf-8').read()

old = '''        top_label   = max(class_probs, key=class_probs.get)
        confidence  = class_probs[top_label]
        attack_prob = self._compute_attack_prob(class_probs, self.BENIGN_LABEL)
        is_attack   = attack_prob >= self.ATTACK_THRESHOLD'''

new = '''        top_label   = max(class_probs, key=class_probs.get)
        confidence  = class_probs[top_label]
        attack_prob = self._compute_attack_prob(class_probs, self.BENIGN_LABEL)
        if top_label == self.BENIGN_LABEL:
            max_attack_prob = max([p for lbl, p in class_probs.items() if lbl != self.BENIGN_LABEL] or [0.0])
            if max_attack_prob < self.ATTACK_THRESHOLD:
                attack_prob = min(attack_prob, 0.28)
                is_attack = False
            else:
                is_attack = attack_prob >= self.ATTACK_THRESHOLD
        else:
            is_attack = attack_prob >= self.ATTACK_THRESHOLD'''

assert old in text, "Target not found"
text = text.replace(old, new, 1)
open(path, 'w', encoding='utf-8').write(text)
print("Successfully patched time_window_adapter.py")

