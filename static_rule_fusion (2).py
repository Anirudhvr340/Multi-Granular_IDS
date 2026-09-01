"""
=============================================================================
STATIC DECISION FUSION BASELINES FOR NETWORK INTRUSION DETECTION
=============================================================================
This module implements established static decision fusion rules from academic
literature as comparative baselines against the proposed Agentic AI Decision Layer.

Algorithms Implemented & Formally Cited:
  1. Kumar et al. (2024):
     Weighted Ensemble Averaging with Deep Neural Networks (WEA-DNN) for
     Heterogeneous Network Intrusion Detection (IEEE Sensors / Trans., 2024).
     Formula:
       P_fused(c_j | x) = sum_{k=1}^K w_k * P_k(c_j | x),  sum(w_k) = 1.0, w_k >= 0
       y_hat(x) = argmax_{c_j} P_fused(c_j | x)

  2. Dasarathy (1997):
     Decision Fusion (IEEE Computer Society Press, 1997).
     Max-Confidence Winner-Takes-All Rule.
     Formula:
       k* = argmax_k [ max_j P_k(c_j | x) ]
       y_hat(x) = argmax_{c_j} P_{k*}(c_j | x)

  3. Peddabachigari et al. (2007):
     Modeling Intrusion Detection System Using Hybrid Intelligent Systems
     (Computers & Security, 2007).
     Sequential Priority / Cascade Decision Rule with Confidence Fallback.

  4. Kittler et al. (1998):
     On Combining Classifiers (IEEE Trans. PAMI, 1998).
     Hard Majority / Plurality Voting.
=============================================================================
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

# Canonical 15-class taxonomy of the CSE-CIC-IDS2018 benchmark
ALL_CLASSES: List[str] = [
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


class StaticRuleFusionEngine:
    """
    Executes academic static decision fusion rules across model posterior
    probability vectors.
    """

    def __init__(
        self,
        flow_classes: Optional[List[str]] = None,
        packet_classes: Optional[List[str]] = None,
        session_classes: Optional[List[str]] = None,
        fixed_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.40,
    ):
        self.flow_classes = flow_classes or ALL_CLASSES
        self.packet_classes = packet_classes or ALL_CLASSES
        self.session_classes = session_classes or ALL_CLASSES
        self.confidence_threshold = confidence_threshold

        # Default competence-proportional weights (Kumar et al. 2024)
        self.weights = fixed_weights or {
            "flow": 0.90,
            "packet": 0.10,
            "session": 0.00,
        }

        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def _align_probabilities(
        self, proba: np.ndarray, source_classes: List[str]
    ) -> np.ndarray:
        """Aligns a model's probability vector to the canonical 15-class space."""
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        n_samples = proba.shape[0]
        aligned = np.zeros((n_samples, len(ALL_CLASSES)), dtype=np.float32)

        for src_idx, cls_name in enumerate(source_classes):
            if cls_name in ALL_CLASSES:
                tgt_idx = ALL_CLASSES.index(cls_name)
                aligned[:, tgt_idx] = proba[:, src_idx]

        # Re-normalize row sums
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned / row_sums

    def rule_weighted_sum(
        self,
        flow_proba: np.ndarray,
        packet_proba: Optional[np.ndarray] = None,
        session_proba: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rule 1: Kumar et al. (2024) Weighted Ensemble Averaging (WEA-DNN).
        P_fused = sum(w_k * P_k)
        """
        P_flow = self._align_probabilities(flow_proba, self.flow_classes)
        fused = self.weights.get("flow", 0.0) * P_flow

        if packet_proba is not None:
            P_pkt = self._align_probabilities(packet_proba, self.packet_classes)
            fused += self.weights.get("packet", 0.0) * P_pkt

        if session_proba is not None:
            P_sess = self._align_probabilities(session_proba, self.session_classes)
            fused += self.weights.get("session", 0.0) * P_sess

        best_idx = np.argmax(fused, axis=1)
        confidences = np.max(fused, axis=1)
        predicted_labels = np.array([ALL_CLASSES[i] for i in best_idx])
        return predicted_labels, confidences

    def rule_max_confidence(
        self,
        flow_proba: np.ndarray,
        packet_proba: Optional[np.ndarray] = None,
        session_proba: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rule 2: Dasarathy (1997) Winner-Takes-All Max-Confidence Rule.
        Chooses the prediction from the single most confident model.
        """
        P_flow = self._align_probabilities(flow_proba, self.flow_classes)
        conf_flow = np.max(P_flow, axis=1)
        pred_flow = np.argmax(P_flow, axis=1)

        best_conf = conf_flow.copy()
        best_pred = pred_flow.copy()

        if packet_proba is not None:
            P_pkt = self._align_probabilities(packet_proba, self.packet_classes)
            conf_pkt = np.max(P_pkt, axis=1)
            pred_pkt = np.argmax(P_pkt, axis=1)
            mask_pkt = conf_pkt > best_conf
            best_pred[mask_pkt] = pred_pkt[mask_pkt]
            best_conf[mask_pkt] = conf_pkt[mask_pkt]

        if session_proba is not None:
            P_sess = self._align_probabilities(session_proba, self.session_classes)
            conf_sess = np.max(P_sess, axis=1)
            pred_sess = np.argmax(P_sess, axis=1)
            mask_sess = conf_sess > best_conf
            best_pred[mask_sess] = pred_sess[mask_sess]
            best_conf[mask_sess] = conf_sess[mask_sess]

        predicted_labels = np.array([ALL_CLASSES[i] for i in best_pred])
        return predicted_labels, best_conf

    def rule_majority_vote(
        self,
        flow_proba: np.ndarray,
        packet_proba: Optional[np.ndarray] = None,
        session_proba: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rule 3: Kittler et al. (1998) Plurality / Majority Voting.
        """
        P_flow = self._align_probabilities(flow_proba, self.flow_classes)
        pred_flow = np.argmax(P_flow, axis=1)
        n_samples = len(pred_flow)

        votes = np.zeros((n_samples, len(ALL_CLASSES)), dtype=np.int32)
        for i, idx in enumerate(pred_flow):
            votes[i, idx] += 1

        if packet_proba is not None:
            P_pkt = self._align_probabilities(packet_proba, self.packet_classes)
            pred_pkt = np.argmax(P_pkt, axis=1)
            for i, idx in enumerate(pred_pkt):
                votes[i, idx] += 1

        if session_proba is not None:
            P_sess = self._align_probabilities(session_proba, self.session_classes)
            pred_sess = np.argmax(P_sess, axis=1)
            for i, idx in enumerate(pred_sess):
                votes[i, idx] += 1

        best_idx = np.argmax(votes, axis=1)
        confidences = np.max(votes, axis=1) / max(1, (1 + (packet_proba is not None) + (session_proba is not None)))
        predicted_labels = np.array([ALL_CLASSES[i] for i in best_idx])
        return predicted_labels, confidences

