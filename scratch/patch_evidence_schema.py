with open('nids/evidence_schema.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """    model_agreement_score: float = 0.0   # 0-1
    models_indicating_attack: int = 0
    models_available: int = 0"""

replacement = """    model_agreement_score: float = 0.0   # 0-1
    models_indicating_attack: int = 0
    models_available: int = 0
    model_weights: Dict[str, float] = field(default_factory=dict)"""

assert target in text, 'target not found'
text = text.replace(target, replacement, 1)

target2 = '"models_available":         self.models_available,'
replacement2 = '"models_available":         self.models_available,\n            "model_weights":            self.model_weights,'

assert target2 in text, 'target2 not found'
text = text.replace(target2, replacement2, 1)

with open('nids/evidence_schema.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Successfully added model_weights to IntegratedEvidence schema!')

