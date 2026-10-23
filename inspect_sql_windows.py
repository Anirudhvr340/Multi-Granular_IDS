import json

with open("reports/comparison_with_qwen2.5_7b.json") as f:
    d = json.load(f)

sql_windows = [w for w in d["windows"] if "SQL" in str(w.get("ground_truth"))]
print(f"Total SQL Injection windows in benchmark: {len(sql_windows)}")

for i, w in enumerate(sql_windows):
    print(f"\n================== SQL Window {i+1}: {w['window_id']} ==================")
    print("Ground Truth:", w.get("ground_truth"))
    print("Action:", w.get("action"), "| Source:", w.get("source"))
    print("Level 1 Models:")
    for m in ["packet", "flow", "time_window", "session"]:
        ev = w["level1"].get(m, {})
        top3 = ev.get("top_predictions", [])[:3]
        print(f"  {m:<12}: pred={ev.get('prediction')}, conf={ev.get('confidence')}, P(att)={ev.get('attack_probability')}, is_attack={ev.get('is_attack')}")
        print(f"               top3={top3}")
    
    l2 = w["level2"]
    print("Level 2 Fusion:")
    print(f"  Fused P(att)={l2.get('fused_attack_probability')}, Fused Family={l2.get('fused_attack_family')}, Agreement={l2.get('model_agreement_score')}, Attacking Models={l2.get('models_indicating_attack')}")
    
    l3 = w["level3"]
    print("Level 3 Agent Decision:")
    print(f"  Action={l3.get('action')}, Severity={l3.get('severity')}, Attack Family={l3.get('attack_family')}, Confidence={l3.get('confidence')}")
    print(f"  Reason: {l3.get('reason')}")

