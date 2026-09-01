"""
=============================================================================
STANDALONE STATIC FUSION BASELINE EVALUATOR (STATIC FUSION)
=============================================================================
Evaluates the formal academic static decision fusion baselines:
  1. Kumar et al. (2024) Weighted Ensemble Averaging (WEA)
  2. Dasarathy (1997) Max-Confidence Winner-Takes-All
  3. Kittler et al. (1998) Plurality Majority Vote
against standalone Flow XGBoost models on any input CSV file.
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
FLOW_DIR = os.path.join(os.path.dirname(HERE), "flow level")

sys.path.insert(0, HERE)
from static_rule_fusion import StaticRuleFusionEngine, ALL_CLASSES


class StaticFusionEvaluator:
    def __init__(self, flow_dir=FLOW_DIR):
        print(f"[*] Loading Flow Model Artifacts from: {flow_dir}")
        self.xgb = joblib.load(os.path.join(flow_dir, "flow_xgb_base.pkl"))
        self.features = joblib.load(os.path.join(flow_dir, "flow_features.pkl"))
        self.le = joblib.load(os.path.join(flow_dir, "flow_le.pkl"))
        self.classes = list(self.le.classes_)
        
        self.engine = StaticRuleFusionEngine(
            flow_classes=self.classes,
            fixed_weights={"flow": 1.00, "time_window": 0.00, "packet": 0.00, "session": 0.00},
            confidence_threshold=0.40,
        )
        print(f"[+] Flow Model & Static Fusion Engine Loaded Successfully.")

    def _safe_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def evaluate_file(self, csv_path: str, nrows: int = 30000):
        print("\n" + "=" * 80)
        print(f"[*] EVALUATING STATIC FUSION BASELINES ON: {os.path.basename(csv_path)}")
        print("=" * 80)
        
        df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
        df.columns = df.columns.str.strip()
        df = df[df["Label"] != "Label"]
        
        df_clean = self._safe_numeric(df)
        X = df_clean.reindex(columns=self.features).fillna(0).values.astype(np.float32)
        p_flow = self.xgb.predict_proba(X)
        y_true = df["Label"].values
        
        # 1. Standalone Flow
        pred_flow_idx = np.argmax(p_flow, axis=1)
        pred_flow = self.le.inverse_transform(pred_flow_idx)
        
        # 2. Kumar et al. (2024) WEA Rule
        pred_wea, _ = self.engine.rule_weighted_sum(flow_proba=p_flow)
        
        # 3. Dasarathy (1997) Max Rule
        pred_max, _ = self.engine.rule_max_confidence(flow_proba=p_flow)
        
        # 4. Kittler (1998) Majority Vote
        pred_vote, _ = self.engine.rule_majority_vote(flow_proba=p_flow)
        
        methods = {
            "Standalone Flow Model (XGBoost)": pred_flow,
            "Kumar et al. WEA Rule (2024)": pred_wea,
            "Dasarathy Max-Confidence Rule (1997)": pred_max,
            "Kittler Majority Vote (1998)": pred_vote,
        }
        
        results = []
        for name, preds in methods.items():
            acc = accuracy_score(y_true, preds)
            w_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
            m_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            results.append({
                "Method / Rule": name,
                "Accuracy": f"{acc * 100:.2f}%",
                "Weighted F1": f"{w_f1 * 100:.2f}%",
                "Macro F1": f"{m_f1 * 100:.2f}%",
            })
            
        df_res = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("STATIC FUSION COMPARISON BENCHMARK RESULTS")
        print("=" * 80)
        print(df_res.to_string(index=False))
        print("=" * 80)
        
        print("\n" + "=" * 80)
        print("KUMAR ET AL. (2024) WEA BASELINE - DETAILED CLASSIFICATION REPORT")
        print("=" * 80)
        print(classification_report(y_true, pred_wea, digits=4, zero_division=0))


if __name__ == "__main__":
    evaluator = StaticFusionEvaluator()
    sample_file = r"F:\Day 2 - Feb 15\csv\02-15-2018.csv"
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
    if os.path.exists(sample_file):
        evaluator.evaluate_file(sample_file, nrows=30000)
    else:
        print(f"[!] Please provide a valid CSV path to evaluate.")
