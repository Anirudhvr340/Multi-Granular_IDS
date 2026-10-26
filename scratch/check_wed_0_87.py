import sys
sys.path.insert(0, '.')
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.attack_gate import AttackGate
from nids.rule_engine import RuleEngine
from nids.policy_agent import PolicyAgent

df = pd.read_csv('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', nrows=5220, encoding='cp1252')
df = df.rename(columns=COL_RENAME_2017)
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip().replace("", "-")

for w_col in ['Init Fwd Win Byts', 'Init Bwd Win Byts']:
    if w_col in df.columns:
        df[w_col] = pd.to_numeric(df[w_col], errors='coerce').fillna(0).clip(lower=0)

durations = pd.to_numeric(df.get('Flow Duration', 1000), errors='coerce').fillna(1000).clip(lower=50) / 1e6
cum_time = [0.0]
for d in durations.values:
    cum_time.append(cum_time[-1] + float(d))
t_base = pd.Timestamp("2018-02-15 09:00:00")
df["Timestamp"] = [t_base + pd.Timedelta(seconds=float(s)) for s in cum_time[:len(df)]]
if "Protocol" not in df.columns:
    df["Protocol"] = 6.0

pkt_ad = PacketModelAdapter()
sess_ad = SessionModelAdapter()
flow_ad = FlowModelAdapter()
tw_ad = TimeWindowAdapter()
gate = AttackGate()
rule_eng = RuleEngine()
policy = PolicyAgent()

flagged_windows = []
allowed_windows = []

for w_idx in range(0, 5220, 60):
    w_num = w_idx // 60
    w_df = df.iloc[w_idx : w_idx + 60].copy()
    
    pkt_ev = pkt_ad.predict(w_df)
    sess_ev = sess_ad.predict(w_df)
    flow_ev = flow_ad.predict(w_df)
    tw_ev = tw_ad.predict(w_df)
    
    # Consensus guardrail matching dashboard
    benign_models = [ev for ev in [flow_ev, pkt_ev, tw_ev, sess_ev] if ev and not ev.is_attack and ev.attack_probability <= 0.35]
    if len(benign_models) >= 3:
        for ev in [flow_ev, pkt_ev, tw_ev, sess_ev]:
            if ev and ev.is_attack and ev.attack_probability > 0.35:
                ev.attack_probability = min(ev.attack_probability, 0.25)
                ev.prediction = "Benign"
                ev.is_attack = False

    integrated = gate.evaluate(pkt_ev, flow_ev, tw_ev, sess_ev, window_id=f"W_{w_num}")
    rule_res = rule_eng.evaluate(integrated, context={})
    decision = policy.decide(integrated, context_summary={}, rule_result=rule_res)
    
    if decision.action in ("FLAG", "BLOCK"):
        flagged_windows.append({
            "w_num": w_num,
            "action": decision.action,
            "fused": integrated.fused_attack_probability,
            "agree": integrated.model_agreement_score,
            "pkt": (pkt_ev.prediction, pkt_ev.attack_probability),
            "sess": (sess_ev.prediction, sess_ev.attack_probability),
            "flow": (flow_ev.prediction, flow_ev.attack_probability),
            "tw": (tw_ev.prediction, tw_ev.attack_probability),
            "rule": decision.rule_triggered,
            "family": decision.attack_family
        })
    else:
        allowed_windows.append(w_num)

print(f"Total Windows: {len(df)//60}")
print(f"Allowed: {len(allowed_windows)} ({len(allowed_windows)/(len(df)//60)*100:.1f}%)")
print(f"Flagged/Blocked: {len(flagged_windows)} ({len(flagged_windows)/(len(df)//60)*100:.1f}%)")
print("\nSample Flagged Windows Details:")
for fw in flagged_windows[:5]:
    print(f"W_{fw['w_num']:02d}: Action={fw['action']} (Rule={fw['rule']}), Fused={fw['fused']:.3f}, Agree={fw['agree']:.0%}, Family={fw['family']}")
    print(f"   Pkt:  {fw['pkt'][0]:<20} (P={fw['pkt'][1]:.3f})")
    print(f"   Sess: {fw['sess'][0]:<20} (P={fw['sess'][1]:.3f})")
    print(f"   Flow: {fw['flow'][0]:<20} (P={fw['flow'][1]:.3f})")
    print(f"   TW:   {fw['tw'][0]:<20} (P={fw['tw'][1]:.3f})")

