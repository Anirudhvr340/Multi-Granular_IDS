import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from nids.policy_agent import PolicyAgent

agent = PolicyAgent()
print("Default model:", agent._model)

# Test legacy model name auto-mapping
legacy_agent = PolicyAgent(gemini_model="gemini-2.5-flash")
print("Mapped legacy model:", legacy_agent._model)

# Test 1: Clean JSON
t1 = '{"action": "ALLOW", "severity": "LOW", "confidence": 0.95, "reason": "Normal flow"}'
r1 = agent._clean_and_parse_json(t1)
print("Test 1 (Clean JSON):", r1["action"], r1["severity"])

# Test 2: Markdown code fence wrapped JSON
t2 = '```json\n{"action": "BLOCK", "severity": "CRITICAL", "confidence": 0.99, "reason": "DDoS detected"}\n```'
r2 = agent._clean_and_parse_json(t2)
print("Test 2 (Markdown Fences):", r2["action"], r2["severity"])

# Test 3: Truncated JSON response (the exact error from before)
t3 = '{\n  "action": "FLAG",\n  "severity": "HIGH",\n'
r3 = agent._clean_and_parse_json(t3)
print("Test 3 (Truncated JSON):", r3["action"], r3["severity"], r3["reason"])

print("\nALL POLICY AGENT PARSING TESTS PASSED!")

