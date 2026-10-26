import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.attack_gate import AttackGate
from nids.rule_engine import RuleEngine
from nids.policy_agent import PolicyAgent

print("=" * 80)
print("TESTING DYNAMIC ATTACK STREAK & RISK ESCALATION FEEDBACK LOOP")
print("=" * 80)

pkt_ad = PacketModelAdapter()
sess_ad = SessionModelAdapter()
flow_ad = FlowModelAdapter()
tw_ad = TimeWindowAdapter()
gate = AttackGate()
rule_eng = RuleEngine()
policy = PolicyAgent(use_gemini=False)

# Read 500 windows (30,000 flows) of Monday
df = pd.read_csv("cicids2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=18000, encoding="cp1252")
df = df.rename(columns=COL_RENAME_2017)
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip().replace("", "-")

# Test 1: With current runaway condition
consec = 0
actions_runaway = []
for w_i in range(0, len(df)//60):
    w_df = df.iloc[w_i*60 : (w_i+1)*60]
    
    pkt_ev = pkt_ad.predict(w_df)
    sess_ev = sess_ad.predict(w_df)
    flow_ev = flow_ad.predict(w_df)
    tw_ev = tw_ad.predict(w_df)
    
    benign_models = [ev for ev in [flow_ev, pkt_ev, tw_ev, sess_ev] if ev and not ev.is_attack and ev.attack_probability <= 0.35]
    if len(benign_models) >= 3:
        for ev in [flow_ev, pkt_ev, tw_ev, sess_ev]:
            if ev and ev.is_attack and ev.attack_probability > 0.35:
                ev.attack_probability = min(ev.attack_probability, 0.25)
                ev.prediction = "Benign"
                ev.is_attack = False

    integrated = gate.evaluate(pkt_ev, flow_ev, tw_ev, sess_ev, window_id=f"W_{w_i}")
    
    # Old logic:
    esc = "CRITICAL" if consec >= 3 else ("MEDIUM" if consec >= 1 else "NONE")
    ctx = {"consecutive_attack_windows": consec, "escalation_level": esc}
    rule_res = rule_eng.evaluate(integrated, context=ctx)
    decision = policy.decide(integrated, context_summary=ctx, rule_result=rule_res)
    
    actions_runaway.append(decision.action)
    
    # Old runaway increment:
    if decision.action in ("BLOCK", "FLAG") and integrated.models_indicating_attack >= 1:
        consec = min(consec + 1, 10)
    else:
        consec = 0

print("Current Logic (Runaway Streak) on Monday:")
print(pd.Series(actions_runaway).value_counts())

# Test 2: With disciplined attack streak (only corroborating attacks increment streak)
consec_fixed = 0
actions_fixed = []
for w_i in range(0, len(df)//60):
    w_df = df.iloc[w_i*60 : (w_i+1)*60]
    
    pkt_ev = pkt_ad.predict(w_df)
    sess_ev = sess_ad.predict(w_df)
    flow_ev = flow_ad.predict(w_df)
    tw_ev = tw_ad.predict(w_df)
    
    benign_models = [ev for ev in [flow_ev, pkt_ev, tw_ev, sess_ev] if ev and not ev.is_attack and ev.attack_probability <= 0.35]
    if len(benign_models) >= 3:
        for ev in [flow_ev, pkt_ev, tw_ev, sess_ev]:
            if ev and ev.is_attack and ev.attack_probability > 0.35:
                ev.attack_probability = min(ev.attack_probability, 0.25)
                ev.prediction = "Benign"
                ev.is_attack = False

    integrated = gate.evaluate(pkt_ev, flow_ev, tw_ev, sess_ev, window_id=f"W_{w_i}")
    
    esc = "CRITICAL" if consec_fixed >= 3 else ("MEDIUM" if consec_fixed >= 1 else "NONE")
    ctx = {"consecutive_attack_windows": consec_fixed, "escalation_level": esc}
    rule_res = rule_eng.evaluate(integrated, context=ctx)
    decision = policy.decide(integrated, context_summary=ctx, rule_result=rule_res)
    
    actions_fixed.append(decision.action)
    
    # Disciplined increment: only true high-confidence or multi-model attacks escalate persistence
    if decision.action == "BLOCK" or (decision.action == "FLAG" and integrated.models_indicating_attack >= 2 and integrated.fused_attack_probability >= 0.50):
        consec_fixed = min(consec_fixed + 1, 10)
    else:
        consec_fixed = 0

print("\nDisciplined Logic on Monday:")
print(pd.Series(actions_fixed).value_counts())
