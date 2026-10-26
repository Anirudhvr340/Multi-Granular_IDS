path = 'nids_dashboard.py'
text = open(path, 'r', encoding='utf-8').read()

old = '''        # Cross-model consensus guardrail for cross-dataset background traffic
        # If at least 2 of the other 3 models (Flow, Packet, Time-Window) confirm Benign traffic,
        # attenuate out-of-distribution session noise (e.g. Kerberos/LDAP domain traffic)
        non_sess_benign = sum([
            1 for ev in [flow_ev, pkt_ev, tw_ev]
            if ev and not ev.is_attack and ev.attack_probability <= 0.35
        ])
        if non_sess_benign >= 2:
            if sess_ev and sess_ev.attack_probability > 0.35:
                sess_ev.attack_probability = min(sess_ev.attack_probability, 0.28)
                sess_ev.prediction = "Benign"
                sess_ev.is_attack = False'''

new = '''        # Multi-granular Byzantine consensus guardrail (Sommer & Paxson 2010):
        # If at least 3 models confirm Benign traffic (attack_prob <= 0.35, not is_attack),
        # attenuate single-model out-of-distribution noise (e.g. Flow or Session false alarms)
        benign_models = [ev for ev in [flow_ev, pkt_ev, tw_ev, sess_ev] if ev and not ev.is_attack and ev.attack_probability <= 0.35]
        if len(benign_models) >= 3:
            for ev in [flow_ev, pkt_ev, tw_ev, sess_ev]:
                if ev and ev.is_attack and ev.attack_probability > 0.35:
                    ev.attack_probability = min(ev.attack_probability, 0.25)
                    ev.prediction = "Benign"
                    ev.is_attack = False'''

assert old in text, "Target not found"
text = text.replace(old, new, 1)
open(path, 'w', encoding='utf-8').write(text)
print("Successfully patched nids_dashboard.py consensus guardrail")

