"""
=============================================================================
STATIC RULE-BASED DECISION FUSION FOR MULTI-GRANULARITY IDS
=============================================================================
Theoretical Foundations:
  1. Kumar et al. (2024) - "Weighted Ensemble Averaging (WEA-DNN) for
     Heterogeneous Network Intrusion Detection" (IEEE Trans. / Sensors)
     -> Fixed-Weight Convex Probability Fusion (Soft Voting)
  2. Kittler et al. (1998) - "On Combining Classifiers" (IEEE TPAMI)
     -> Fixed-Weight Sum Rule, Product Rule, Max/Min Rule, Majority Vote
  3. Dasarathy (1997) - "Sensor Fusion Potential Exploitation"
     -> Decision-Level Information Fusion (DAI-DEO) & Static Priority Cascades
  4. Aburomman & Reaz (2016) - "Weighted Majority Rules for NIDS" (Elsevier JNCA)
     -> Static Weighted Class Voting & Fixed-Threshold Gating

This module serves as the academic baseline to benchmark against the
dynamic Agentic AI Decision Layer.
=============================================================================
"""

import os
import warnings
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("StaticRuleFusion")


# =============================================================================
# 1. CANONICAL CLASS DEFINITIONS
# =============================================================================
ALL_CLASSES = [
    "Benign",
    "Bot",
    "Brute Force -Web",
    "Brute Force -XSS",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "Infilteration",
    "SQL Injection",
    "SSH-Bruteforce",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}
N_CLASSES = len(ALL_CLASSES)


# =============================================================================
# 2. STATIC RULE FUSION ENGINE
# =============================================================================
class StaticRuleFusionEngine:
    """
    Implements classical fixed-rule decision combination algorithms.
    """

    def __init__(
        self,
        flow_classes: Optional[List[str]] = None,
        packet_classes: Optional[List[str]] = None,
        session_classes: Optional[List[str]] = None,
        fixed_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.50,
    ):
        self.flow_classes = flow_classes or ALL_CLASSES
        self.packet_classes = packet_classes or ALL_CLASSES
        self.session_classes = session_classes or ALL_CLASSES

        # Precompute index maps to union space
        self._flow_map = {i: CLASS_TO_IDX[c] for i, c in enumerate(self.flow_classes) if c in CLASS_TO_IDX}
        self._packet_map = {i: CLASS_TO_IDX[c] for i, c in enumerate(self.packet_classes) if c in CLASS_TO_IDX}
        self._session_map = {i: CLASS_TO_IDX[c] for i, c in enumerate(self.session_classes) if c in CLASS_TO_IDX}

        # Default static weights (Convex combination sum(w) = 1.0)
        self.weights = fixed_weights or {
            "flow": 0.40,
            "packet": 0.35,
            "session": 0.25,
        }
        self.confidence_threshold = confidence_threshold

    def align_to_union(self, proba: np.ndarray, model_name: str) -> np.ndarray:
        """Projects local model probabilities into the canonical N_CLASSES union space."""
        n_samples = proba.shape[0]
        union_matrix = np.zeros((n_samples, N_CLASSES), dtype=np.float64)

        if model_name == "flow":
            idx_map = self._flow_map
        elif model_name == "packet":
            idx_map = self._packet_map
        else:
            idx_map = self._session_map

        for local_idx, union_idx in idx_map.items():
            if local_idx < proba.shape[1]:
                union_matrix[:, union_idx] = proba[:, local_idx]

        # Row-wise L1 normalization
        row_sums = union_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return union_matrix / row_sums

    # -------------------------------------------------------------------------
    # RULE 1: Kittler's Fixed Weighted Sum Rule (Convex Combination)
    # Reference: Kittler et al. (1998) Section 3.1
    # -------------------------------------------------------------------------
    def rule_weighted_sum(
        self,
        p_flow: np.ndarray,
        p_packet: np.ndarray,
        p_session: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combines posterior probabilities via static weighted average:
        P_fused(x, c_j) = sum_{k} w_k * P_k(x, c_j)
        """
        u_flow = self.align_to_union(p_flow, "flow")
        u_packet = self.align_to_union(p_packet, "packet")

        if p_session is not None:
            u_session = self.align_to_union(p_session, "session")
            fused_proba = (
                self.weights["flow"] * u_flow
                + self.weights["packet"] * u_packet
                + self.weights["session"] * u_session
            )
        else:
            w_flow = self.weights["flow"] / (self.weights["flow"] + self.weights["packet"])
            w_packet = 1.0 - w_flow
            fused_proba = w_flow * u_flow + w_packet * u_packet

        pred_indices = np.argmax(fused_proba, axis=1)
        pred_labels = np.array([ALL_CLASSES[i] for i in pred_indices])
        confidences = np.max(fused_proba, axis=1)
        return pred_labels, confidences

    # -------------------------------------------------------------------------
    # RULE 2: Maximum Confidence / Winner-Takes-All Rule
    # Reference: Kittler et al. (1998) Section 3.3 & Dasarathy (1997)
    # -------------------------------------------------------------------------
    def rule_max_confidence(
        self,
        p_flow: np.ndarray,
        p_packet: np.ndarray,
        p_session: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assigns sample to the class with the highest absolute confidence across all models:
        c* = argmax_{c} [ max_{k} P_k(x, c) ]
        """
        u_flow = self.align_to_union(p_flow, "flow")
        u_packet = self.align_to_union(p_packet, "packet")

        stack = [u_flow, u_packet]
        if p_session is not None:
            stack.append(self.align_to_union(p_session, "session"))

        stacked = np.stack(stack, axis=0)  # Shape: (M_models, N_samples, N_classes)
        max_across_models = np.max(stacked, axis=0)  # Shape: (N_samples, N_classes)

        pred_indices = np.argmax(max_across_models, axis=1)
        pred_labels = np.array([ALL_CLASSES[i] for i in pred_indices])
        confidences = np.max(max_across_models, axis=1)
        return pred_labels, confidences

    # -------------------------------------------------------------------------
    # RULE 3: Static Priority Hierarchy with Threshold Gating (Expert Cascade)
    # Reference: Peddabachigari et al. (2007) & Aburomman & Reaz (2016)
    # -------------------------------------------------------------------------
    def rule_priority_cascade(
        self,
        p_flow: np.ndarray,
        p_packet: np.ndarray,
        p_session: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies a fixed domain-hierarchy rule:
          1. If Session model detects attack with confidence >= tau -> Trigger Session alert
          2. Else if Packet model detects attack with confidence >= tau -> Trigger Packet alert
          3. Else if Flow model detects attack with confidence >= tau -> Trigger Flow alert
          4. Fallback -> Flow model prediction (dominant baseline)
        """
        n_samples = p_flow.shape[0]
        u_flow = self.align_to_union(p_flow, "flow")
        u_packet = self.align_to_union(p_packet, "packet")
        u_session = self.align_to_union(p_session, "session") if p_session is not None else None

        pred_labels = []
        confidences = []

        for i in range(n_samples):
            # Session check (Index 0 is Benign, Indices 1..14 are Attacks)
            if u_session is not None:
                sess_top = np.argmax(u_session[i])
                sess_conf = u_session[i, sess_top]
                if sess_top != 0 and sess_conf >= self.confidence_threshold:
                    pred_labels.append(ALL_CLASSES[sess_top])
                    confidences.append(sess_conf)
                    continue

            # Packet check
            pkt_top = np.argmax(u_packet[i])
            pkt_conf = u_packet[i, pkt_top]
            if pkt_top != 0 and pkt_conf >= self.confidence_threshold:
                pred_labels.append(ALL_CLASSES[pkt_top])
                confidences.append(pkt_conf)
                continue

            # Flow check
            flow_top = np.argmax(u_flow[i])
            flow_conf = u_flow[i, flow_top]
            if flow_top != 0 and flow_conf >= self.confidence_threshold:
                pred_labels.append(ALL_CLASSES[flow_top])
                confidences.append(flow_conf)
                continue

            # Fallback to dominant flow prediction
            pred_labels.append(ALL_CLASSES[flow_top])
            confidences.append(flow_conf)

        return np.array(pred_labels), np.array(confidences)

    # -------------------------------------------------------------------------
    # RULE 4: Plurality / Majority Voting
    # Reference: Kittler et al. (1998) Section 3.4
    # -------------------------------------------------------------------------
    def rule_majority_vote(
        self,
        p_flow: np.ndarray,
        p_packet: np.ndarray,
        p_session: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hard majority vote across model discrete predictions.
        Ties are broken using highest average probability.
        """
        n_samples = p_flow.shape[0]
        u_flow = self.align_to_union(p_flow, "flow")
        u_packet = self.align_to_union(p_packet, "packet")
        u_session = self.align_to_union(p_session, "session") if p_session is not None else None

        pred_flow = np.argmax(u_flow, axis=1)
        pred_pkt = np.argmax(u_packet, axis=1)
        pred_sess = np.argmax(u_session, axis=1) if u_session is not None else pred_flow

        pred_labels = []
        confidences = []

        for i in range(n_samples):
            votes = [pred_flow[i], pred_pkt[i]]
            if u_session is not None:
                votes.append(pred_sess[i])

            vals, counts = np.unique(votes, return_counts=True)
            max_c = np.max(counts)

            if list(counts).count(max_c) == 1:
                # Clear majority
                winner = vals[np.argmax(counts)]
                conf = np.mean([u_flow[i, winner], u_packet[i, winner]])
            else:
                # Tie: fallback to highest average probability
                avg_p = (u_flow[i] + u_packet[i]) / 2.0
                winner = np.argmax(avg_p)
                conf = avg_p[winner]

            pred_labels.append(ALL_CLASSES[winner])
            confidences.append(conf)

        return np.array(pred_labels), np.array(confidences)


# =============================================================================
# 3. BENCHMARKING & EVALUATION HELPER
# =============================================================================
def evaluate_static_fusion_benchmarks(
    y_true: np.ndarray,
    p_flow: np.ndarray,
    p_packet: np.ndarray,
    p_session: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Evaluates all 4 static rule-based fusion methods against ground truth
    and returns a clean comparative DataFrame.
    """
    engine = StaticRuleFusionEngine()

    rules = {
        "Kittler Weighted Sum (1998)": engine.rule_weighted_sum(p_flow, p_packet, p_session),
        "Dasarathy Max-Confidence (1997)": engine.rule_max_confidence(p_flow, p_packet, p_session),
        "Peddabachigari Priority Cascade (2007)": engine.rule_priority_cascade(p_flow, p_packet, p_session),
        "Kittler Majority Vote (1998)": engine.rule_majority_vote(p_flow, p_packet, p_session),
    }

    results = []
    for name, (preds, _) in rules.items():
        acc = accuracy_score(y_true, preds)
        w_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
        m_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        results.append({
            "Fusion Method": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Weighted F1": f"{w_f1 * 100:.2f}%",
            "Macro F1": f"{m_f1 * 100:.2f}%",
        })

    df_results = pd.DataFrame(results)
    return df_results


if __name__ == "__main__":
    print("=" * 70)
    print("STATIC RULE-BASED FUSION ENGINE LOADED")
    print("Ready for benchmarking against Agentic AI Decision Layer")
    print("=" * 70)

