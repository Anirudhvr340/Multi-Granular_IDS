import sys
from pathlib import Path
sys.path.insert(0, '.')
import time
from nids.policy_agent import PolicyAgent
from nids.evidence_schema import IntegratedEvidence, EvidenceRecord, STATUS_OK

print("=" * 80)
print("TESTING LIVE QWEN 2.5 (7B INSTRUCT) POLICY AGENT INFERENCE")
print("=" * 80)

agent = PolicyAgent()

# Create a sample ambiguous window (e.g. 2 models attack, 2 models benign)
pkt_ev = EvidenceRecord(
    model_name="Packet 1D-CNN", granularity="packet", status=STATUS_OK,
    prediction="Infilteration", confidence=0.45, attack_probability=0.45, is_attack=True,
    class_probabilities={"Benign": 0.55, "Infilteration": 0.45},
    top_predictions=[("Benign", 0.55), ("Infilteration", 0.45)],
    original_prediction="Infilteration", normalised_family="Infiltration",
    supporting_evidence={"packets": 60}
)
flow_ev = EvidenceRecord(
    model_name="Flow XGBoost", granularity="flow", status=STATUS_OK,
    prediction="DoS attacks-Hulk", confidence=0.92, attack_probability=0.92, is_attack=True,
    class_probabilities={"Benign": 0.08, "DoS attacks-Hulk": 0.92},
    top_predictions=[("DoS attacks-Hulk", 0.92), ("Benign", 0.08)],
    original_prediction="DoS attacks-Hulk", normalised_family="DoS",
    supporting_evidence={}
)
tw_ev = EvidenceRecord(
    model_name="Time-Window XGBoost", granularity="window", status=STATUS_OK,
    prediction="Benign", confidence=0.72, attack_probability=0.28, is_attack=False,
    class_probabilities={"Benign": 0.72}, top_predictions=[("Benign", 0.72)],
    original_prediction="Benign", normalised_family=None, supporting_evidence={}
)
sess_ev = EvidenceRecord(
    model_name="Session CNN-LSTM", granularity="session", status=STATUS_OK,
    prediction="Benign", confidence=0.95, attack_probability=0.05, is_attack=False,
    class_probabilities={"Benign": 0.95}, top_predictions=[("Benign", 0.95)],
    original_prediction="Benign", normalised_family=None, supporting_evidence={}
)

integrated = IntegratedEvidence(
    packet=pkt_ev, flow=flow_ev, time_window=tw_ev, session=sess_ev,
    fused_attack_probability=0.52, fused_attack_family="DoS",
    model_agreement_score=0.50, models_indicating_attack=2, models_available=4,
    window_id="W_AMBIGUOUS_TEST"
)

ctx = {"consecutive_attack_windows": 1, "escalation_level": "MEDIUM"}

t0 = time.time()
decision = agent.decide(integrated, context_summary=ctx, rule_result=None)
elapsed = time.time() - t0

print(f"\nDecision Source:     {decision.source}")
print(f"Action:              {decision.action} [{decision.severity}]")
print(f"Attack Family:       {decision.attack_family}")
print(f"Confidence:          {decision.confidence}")
print(f"Risk Score:          {decision.risk_score}")
print(f"Elapsed Time:        {elapsed:.2f}s")
print(f"Reasoning Rationale:\n  {decision.reason}")
print(f"Recommended Actions: {decision.recommended_actions}")
print(f"\nStats: {agent.stats}")

