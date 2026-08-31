"""Fuse the flow and time-window detectors and optionally summarize alerts with Gemini."""

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIME_WINDOW = HERE / "time_window"
PROCESSED = TIME_WINDOW / "processed"

# Identifiers that should be masked if present to prevent memorizing dataset artifacts
IDENTIFIER_FEATURES = {
    "Dst Port",
    "Fwd Seg Size Min",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
}


def load_artifacts():
    flow_model = joblib.load(HERE / "xgb_flow_model.pkl")
    flow_features = joblib.load(HERE / "important_features.pkl")
    flow_labels = joblib.load(HERE / "labels.pkl").astype(str)
    window_model = joblib.load(PROCESSED / "xgb_time_window_model.joblib")
    window_labels = np.load(PROCESSED / "window_class_names.npy", allow_pickle=True).astype(str)
    window_features = np.load(PROCESSED / "window_feature_names.npy", allow_pickle=True).astype(str)
    scaler = joblib.load(PROCESSED / "time_window_scaler.joblib")
    return flow_model, flow_features, flow_labels, window_model, window_labels, window_features, scaler


def score_file(input_path, window_seconds):
    os.environ["WINDOW_SECONDS"] = str(window_seconds)
    from time_window import step0_preprocessing as preprocessing

    artifacts = load_artifacts()
    flow_model, flow_features, flow_labels, window_model, window_labels, window_feature_names, scaler = artifacts
    frame = pd.read_csv(input_path, low_memory=False)

    required_cols = ["Timestamp", "Label", "Dst Port", "Protocol", "Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Duration"]
    missing = [column for column in required_cols if column not in frame.columns]
    if missing:
        expected_file = str(Path(__file__).resolve().parent / "02-14-2018.csv")
        raise ValueError(
            f"Input file '{input_path}' does not match the CICIDS 2018 flow schema. "
            f"Missing required columns: {missing}. Use a project CSV like '{expected_file}' instead."
        )

    frame = preprocessing.clean_chunk(frame)
    if frame.empty:
        raise ValueError("The input contains no valid 2018 flow records.")
    frame["window_start"] = frame["Timestamp"].dt.floor(f"{window_seconds}s")
    rows = []
    previous_by_day = {}
    for window_start, window in frame.groupby("window_start", sort=True):
        aggregate = preprocessing.window_aggregate(window, window_start)
        day = window_start.normalize()
        previous = previous_by_day.setdefault(day, {})
        for column in ["total_flows", "total_packets", "total_bytes", "flow_rate", "packet_rate", "byte_rate"]:
            previous_value = previous.get(column, aggregate[column])
            aggregate[f"previous_{column}"] = previous_value
            aggregate[f"delta_{column}"] = aggregate[column] - previous_value
            previous[column] = aggregate[column]
        timestamps = pd.to_datetime(window["Timestamp"]).sort_values()
        inter_arrivals = timestamps.diff().dt.total_seconds().dropna()
        burstiness = float(inter_arrivals.std(ddof=0) / max(inter_arrivals.mean(), 1e-6)) if len(inter_arrivals) else 0.0
        flow_input = window.copy()
        flow_input["total_packets"] = (
            pd.to_numeric(flow_input.get("Tot Fwd Pkts", 0), errors="coerce").fillna(0)
            + pd.to_numeric(flow_input.get("Tot Bwd Pkts", 0), errors="coerce").fillna(0)
        )
        flow_input["burstiness"] = burstiness

        # Build feature vector for flow model
        values = flow_input.reindex(columns=flow_features).apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Predict per-flow probabilities
        flow_probabilities = flow_model.predict_proba(values)

        classes_int = flow_model.classes_.astype(int)
        model_labels = flow_labels[classes_int]
        benign_mask = np.array([lbl == "Benign" for lbl in model_labels])

        # Max-pool flow probabilities for attacks so a single attack flow in a window isn't diluted
        if len(flow_probabilities) > 0:
            max_probs = flow_probabilities.max(axis=0)
            mean_probs = flow_probabilities.mean(axis=0)
            attack_max = np.where(~benign_mask, max_probs, 0.0)
            max_attack_score = float(attack_max.max()) if np.any(~benign_mask) else 0.0

            if max_attack_score >= 0.40:
                flow_distribution = {}
                for idx, lbl in enumerate(model_labels):
                    if lbl == "Benign":
                        flow_distribution[lbl] = float(np.clip(1.0 - max_attack_score, 0.0, 1.0))
                    else:
                        flow_distribution[lbl] = float(attack_max[idx])
                total_dist = sum(flow_distribution.values()) or 1.0
                flow_distribution = {k: v / total_dist for k, v in flow_distribution.items()}
            else:
                flow_distribution = dict(zip(model_labels, mean_probs))
        else:
            flow_distribution = {"Benign": 1.0}

        window_vector = pd.DataFrame([aggregate]).reindex(columns=window_feature_names).fillna(0.0)
        window_probability = window_model.predict_proba(scaler.transform(window_vector))[0]
        window_distribution = dict(zip(window_labels[window_model.classes_.astype(int)], window_probability))

        common = sorted(set(flow_distribution) & set(window_distribution))
        flow_confidence = max(flow_distribution.values())
        window_confidence = max(window_distribution.values())
        flow_attack = 1.0 - flow_distribution.get("Benign", 0.0)
        window_attack = 1.0 - window_distribution.get("Benign", 0.0)
        predictions_agree = max(flow_distribution, key=flow_distribution.get) == max(window_distribution, key=window_distribution.get)

        flow_weight = float(np.clip(
            0.50 + 0.20 * (flow_confidence - window_confidence)
            + 0.10 * (flow_attack - window_attack)
            + (0.10 if predictions_agree else 0.0),
            0.25,
            0.75,
        ))
        window_weight = 1.0 - flow_weight

        fused = {
            label: flow_weight * flow_distribution.get(label, 0.0) + window_weight * window_distribution.get(label, 0.0)
            for label in common
        }
        if not fused:
            fused = {"Benign": 1.0}

        detected = max(fused, key=fused.get)
        flow_label = max(flow_distribution, key=flow_distribution.get)
        window_label = max(window_distribution, key=window_distribution.get)

        is_attack_prediction = (detected != "Benign")
        combined_attack_prob = 1.0 - fused.get("Benign", 1.0)
        flagged = is_attack_prediction or (combined_attack_prob >= 0.50)

        if is_attack_prediction:
            alert_label = detected
        elif flagged:
            alert_label = max((label for label in fused if label != "Benign"), key=fused.get, default="Suspicious")
        else:
            alert_label = "Benign"

        rows.append({
            "window_start": str(window_start),
            "window_seconds": window_seconds,
            "flows": int(len(window)),
            "flow_prediction": flow_label,
            "window_prediction": window_label,
            "detected_attack": alert_label,
            "attack_label": alert_label,
            "flagged": flagged,
            "confidence": round(float(fused[detected]), 6),
            "flow_weight": round(flow_weight, 6),
            "window_weight": round(window_weight, 6),
            "model_agreement": predictions_agree,
            "model_basis": [
                {
                    "model_name": "Flow XGBoost",
                    "prediction": flow_label,
                    "prediction_confidence": round(float(flow_distribution[flow_label]), 6),
                    "attack_probability": round(float(flow_attack), 6),
                    "support_for_detected_label": round(float(flow_distribution.get(detected, 0.0)), 6),
                    "fusion_weight": round(flow_weight, 6),
                },
                {
                    "model_name": "Time-window XGBoost",
                    "prediction": window_label,
                    "prediction_confidence": round(float(window_distribution[window_label]), 6),
                    "attack_probability": round(float(window_attack), 6),
                    "support_for_detected_label": round(float(window_distribution.get(detected, 0.0)), 6),
                    "fusion_weight": round(window_weight, 6),
                },
            ],
            "top_evidence": {
                "flow_attack_probability": round(flow_attack, 6),
                "window_attack_probability": round(window_attack, 6),
                "flow_density": round(float(aggregate["flow_density"]), 6),
                "window_occupancy": round(float(aggregate["window_occupancy"]), 6),
                "small_flow_ratio": round(float(aggregate["small_flow_ratio"]), 6),
                "syn_no_ack_ratio": round(float(aggregate["syn_no_ack_ratio"]), 6),
                "flow_duration_p90": round(float(aggregate["flow_duration_p90"]), 6),
            },
            "top_scores": [
                [label, round(float(score), 6)]
                for label, score in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:5]
            ],
        })
    return rows


def gemini_summary(alerts):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"status": "skipped", "reason": "GEMINI_API_KEY is not set"}
    payload = {
        "contents": [{"parts": [{"text": (
            "You are an evidence-fusion decision layer for an intrusion detection system. "
            "The ML models have already made the only attack predictions. You must not classify "
            "raw traffic, invent an attack type, or change a predicted attack. Choose exactly one "
            "action: ALLOW, ALERT, or BLOCK. Use BLOCK only for high-confidence, repeated alerts; "
            "use ALERT for detected attacks that need investigation. Return concise JSON with keys "
            "action, attack_types, priority, explanation, recommended_actions. The attack_types "
            "must be selected only from the supplied detected_attack values.\n" + json.dumps(
                alerts,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            )
        )}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        report = json.loads(text)
        action = str(report.get("action", "")).upper()
        allowed_attacks = {alert["detected_attack"] for alert in alerts}
        report["action"] = action if action in {"ALLOW", "ALERT", "BLOCK"} else "ALERT"
        report["attack_types"] = [
            attack for attack in report.get("attack_types", []) if attack in allowed_attacks
        ]
        return {"status": "ok", "report": report}
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        return {"status": "error", "reason": f"HTTP {error.code}: {details[:1000]}"}
    except Exception as error:
        return {"status": "error", "reason": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV file containing CICIDS2018 flow records")
    parser.add_argument("--window-seconds", type=int, default=int(os.environ.get("WINDOW_SECONDS", "30")))
    parser.add_argument("--report", default="agentic_detection_report.json")
    parser.add_argument("--no-llm", action="store_true", help="Skip Gemini and produce deterministic model output")
    args = parser.parse_args()
    try:
        windows = score_file(args.input_csv, args.window_seconds)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=None)
        raise SystemExit(1) from exc
    alerts = [window for window in windows if window["flagged"]]
    result = {
        "input": str(Path(args.input_csv).resolve()),
        "window_seconds": args.window_seconds,
        "architecture": "flow_ml + time_window_ml -> evidence_fusion -> Gemini_decision",
        "windows_scored": len(windows),
        "windows_flagged": len(alerts),
        "alerts": alerts,
    }
    if not alerts:
        result["final_decision"] = {
            "action": "ALLOW",
            "reason": "No ML model detected an attack",
            "attack_labels": [],
            "model_basis": [],
        }
        result["llm_report"] = {"status": "skipped", "reason": "No ML attack was detected"}
    elif args.no_llm:
        result["final_decision"] = {
            "action": "ALERT",
            "reason": "ML models detected an attack; Gemini was disabled",
            "attack_labels": sorted({alert["attack_label"] for alert in alerts}),
            "model_basis": "See model_basis on each alert",
        }
        result["llm_report"] = {"status": "skipped", "reason": "--no-llm was supplied"}
    else:
        # Keep the full alert evidence local; send only the highest-confidence alerts to the LLM.
        llm_alerts = sorted(alerts, key=lambda alert: alert["confidence"], reverse=True)[:100]
        result["llm_report"] = gemini_summary(llm_alerts)
        if result["llm_report"]["status"] == "ok":
            result["final_decision"] = result["llm_report"]["report"]
            result["final_decision"]["attack_labels"] = sorted({alert["attack_label"] for alert in alerts})
            result["final_decision"]["model_basis"] = "See model_basis on each alert"
        else:
            result["final_decision"] = {
                "action": "ALERT",
                "reason": "ML models detected an attack but Gemini was unavailable",
                "attack_labels": sorted({alert["attack_label"] for alert in alerts}),
                "model_basis": "See model_basis on each alert",
            }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(result, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in ("input", "window_seconds", "windows_scored", "windows_flagged")}, indent=2))
    print(f"Report saved to {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()