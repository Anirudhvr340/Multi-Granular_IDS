import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from nids_dashboard import COL_RENAME_2017
from nids.adapters.packet_adapter import PacketModelAdapter
from nids.adapters.session_adapter import SessionModelAdapter
from nids.adapters.flow_adapter import FlowModelAdapter
from nids.adapters.time_window_adapter import TimeWindowAdapter
from nids.attack_gate import AttackGate
from nids.rule_engine import RuleEngine
from nids.policy_agent import PolicyAgent

pkt_ad = PacketModelAdapter()
sess_ad = SessionModelAdapter()
flow_ad = FlowModelAdapter()
tw_ad = TimeWindowAdapter()
gate = AttackGate()
rule_eng = RuleEngine()
policy = PolicyAgent()

scenarios = [
    ("Tuesday FTP-Patator", "cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv", 11347),
    ("Wednesday DoS Slowloris", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv", 26700),
    ("Wednesday DoS Hulk", "cicids2017/Wednesday-workingHours.pcap_ISCX.csv", 74860),
    ("Friday DDoS", "cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", 18884),
]

for name, filepath, start_row in scenarios:
    print(f"\n================================================================================")
    print(f"EVALUATING: {name} (starting row {start_row})")
    print(f"================================================================================")
    header_df = pd.read_csv(filepath, nrows=1, encoding="cp1252")
    cols = header_df.columns
    df = pd.read_csv(
        filepath,
        skiprows=range(1, start_row + 1),
        nrows=180,
        header=0,
        names=cols,
        encoding="cp1252"
    )
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

    for w_i in range(3):
        w_df = df.iloc[w_i * 60 : (w_i + 1) * 60].copy()
        gt = w_df["Label"].mode()[0] if not w_df.empty else "Unknown"

        pkt_ev = pkt_ad.predict(w_df)
        sess_ev = sess_ad.predict(w_df)
        flow_ev = flow_ad.predict(w_df)
        tw_ev = tw_ad.predict(w_df)

        # Cross-model consensus guardrail
        non_sess_benign = sum([
            1 for ev in [flow_ev, pkt_ev, tw_ev]
            if ev and not ev.is_attack and ev.attack_probability <= 0.35
        ])
        if non_sess_benign >= 2:
            if sess_ev and sess_ev.attack_probability > 0.35:
                sess_ev.attack_probability = min(sess_ev.attack_probability, 0.28)
                sess_ev.prediction = "Benign"
                sess_ev.is_attack = False

        integrated = gate.evaluate(pkt_ev, flow_ev, tw_ev, sess_ev, window_id=f"W_{w_i}")
        rule_res = rule_eng.evaluate(integrated, context={})
        decision = policy.decide(integrated, context_summary={}, rule_result=rule_res)

        print(f"  Window {w_i:02d} | GT: {gt:<18} | Action: {decision.action:<6} [{decision.severity:<8}] "
              f"| Fused: {integrated.fused_attack_probability:.3f} | Agree: {integrated.model_agreement_score:.2f}\n"
              f"     Packet:  {pkt_ev.prediction:<20} (P={pkt_ev.attack_probability:.3f})\n"
              f"     Session: {sess_ev.prediction:<20} (P={sess_ev.attack_probability:.3f})\n"
              f"     Flow:    {flow_ev.prediction:<20} (P={flow_ev.attack_probability:.3f})\n"
              f"     TW:      {tw_ev.prediction:<20} (P={tw_ev.attack_probability:.3f})")

