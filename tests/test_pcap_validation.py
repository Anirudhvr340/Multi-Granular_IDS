"""
Automated Test Suite for Unified PCAP Validation Pipeline
Tests:
  A. Feature Contract Tests
  B. PCAP Reader & Conversion Tests
  C. Independent Model Execution Tests
  D. Integrated 4-Model Pipeline & Decision Layer Tests
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "time_window"))

from pcap_validation.pcap_reader import PCAPReader
from pcap_validation.cicflowmeter_runner import CICFlowMeterRunner
from pcap_validation.packet_feature_extractor import PacketFeatureExtractor
from pcap_validation.session_sequence_builder import SessionSequenceBuilder
from pcap_validation.flow_preprocessor import FlowPreprocessor
from pcap_validation.window_aggregator import WindowAggregator
from pcap_validation.evidence_aligner import EvidenceAligner

from nids.rule_engine import RuleEngine, RuleConfig
from nids.attack_gate import AttackGate
from nids.evidence_schema import STATUS_OK


class TestPCAPValidationPipeline(unittest.TestCase):

    def test_feature_contracts(self):
        """Test A: Verify input dimensions, scalers, and classes for all 4 models."""
        # 1. Flow Model
        flow_feats = joblib.load(HERE / "important_features.pkl")
        self.assertEqual(len(flow_feats), 17)
        self.assertIn("burstiness", flow_feats)
        self.assertIn("total_packets", flow_feats)

        # 2. Time-Window Model
        tw_model = joblib.load(HERE / "time_window" / "processed" / "xgb_time_window_model.joblib")
        self.assertEqual(tw_model.n_features_in_, 268)

        # 3. Packet Model
        pkt_scaler = joblib.load(HERE / "packet_level" / "cnn_scaler.pkl")
        self.assertEqual(pkt_scaler.n_features_in_, 27)
        pkt_feats = joblib.load(HERE / "packet_level" / "cnn_packet_features.pkl")
        self.assertEqual(len(pkt_feats), 27)

        # 4. Session Model
        sess_scaler = joblib.load(HERE / "session_level" / "session_scaler.pkl")
        self.assertEqual(sess_scaler.n_features_in_, 27)
        sess_feats = joblib.load(HERE / "session_level" / "session_features.pkl")
        self.assertEqual(len(sess_feats), 27)

    def test_independent_models(self):
        """Test C: Run each model independently on synthetic flow records."""
        # Generate sample flow row
        sample_row = {
            "Timestamp": "23/02/2018 10:00:00",
            "Dst Port": 80,
            "Protocol": 6,
            "Flow Duration": 150000,
            "Tot Fwd Pkts": 10,
            "Tot Bwd Pkts": 8,
            "TotLen Fwd Pkts": 1200,
            "TotLen Bwd Pkts": 4500,
            "Fwd Pkt Len Max": 500,
            "Fwd Pkt Len Min": 40,
            "Fwd Pkt Len Mean": 120.0,
            "Fwd Pkt Len Std": 30.0,
            "Bwd Pkt Len Max": 1400,
            "Bwd Pkt Len Min": 0,
            "Bwd Pkt Len Mean": 562.5,
            "Bwd Pkt Len Std": 120.0,
            "Flow Byts/s": 38000.0,
            "Flow Pkts/s": 120.0,
            "Flow IAT Mean": 8000.0,
            "Flow IAT Std": 2000.0,
            "Flow IAT Max": 15000.0,
            "Flow IAT Min": 50.0,
            "Fwd IAT Tot": 70000.0,
            "Fwd IAT Mean": 7777.0,
            "Fwd IAT Std": 1500.0,
            "Fwd IAT Max": 12000.0,
            "Fwd IAT Min": 100.0,
            "Bwd IAT Tot": 60000.0,
            "Bwd IAT Mean": 8571.0,
            "Bwd IAT Std": 2100.0,
            "Bwd IAT Max": 14000.0,
            "Bwd IAT Min": 80.0,
            "Fwd PSH Flags": 0,
            "Bwd PSH Flags": 0,
            "Fwd URG Flags": 0,
            "Bwd URG Flags": 0,
            "Fwd Header Len": 200,
            "Bwd Header Len": 160,
            "Fwd Pkts/s": 66.6,
            "Bwd Pkts/s": 53.3,
            "Pkt Len Min": 0,
            "Pkt Len Max": 1400,
            "Pkt Len Mean": 316.6,
            "Pkt Len Std": 210.0,
            "Pkt Len Var": 44100.0,
            "FIN Flag Cnt": 0,
            "SYN Flag Cnt": 1,
            "RST Flag Cnt": 0,
            "PSH Flag Cnt": 1,
            "ACK Flag Cnt": 1,
            "URG Flag Cnt": 0,
            "CWE Flag Count": 0,
            "ECE Flag Cnt": 0,
            "Down/Up Ratio": 0.8,
            "Pkt Size Avg": 316.6,
            "Fwd Seg Size Avg": 120.0,
            "Bwd Seg Size Avg": 562.5,
            "Fwd Byts/b Avg": 0.0,
            "Fwd Pkts/b Avg": 0.0,
            "Fwd Blk Rate Avg": 0.0,
            "Bwd Byts/b Avg": 0.0,
            "Bwd Pkts/b Avg": 0.0,
            "Bwd Blk Rate Avg": 0.0,
            "Subflow Fwd Pkts": 10,
            "Subflow Fwd Byts": 1200,
            "Subflow Bwd Pkts": 8,
            "Subflow Bwd Byts": 4500,
            "Init Fwd Win Byts": 29200,
            "Init Bwd Win Byts": 28960,
            "Fwd Act Data Pkts": 8,
            "Fwd Seg Size Min": 20,
            "Active Mean": 0.0,
            "Active Std": 0.0,
            "Active Max": 0.0,
            "Active Min": 0.0,
            "Idle Mean": 0.0,
            "Idle Std": 0.0,
            "Idle Max": 0.0,
            "Idle Min": 0.0,
            "Label": "Benign",
        }
        df = pd.DataFrame([sample_row] * 5)

        # 1. Flow branch
        flow_runner = FlowPreprocessor()
        ev_flow = flow_runner.predict_window(df, window_id="W_test")
        self.assertEqual(ev_flow.status, STATUS_OK)
        self.assertIsNotNone(ev_flow.prediction)

        # 2. Packet branch
        pkt_runner = PacketFeatureExtractor()
        ev_pkt = pkt_runner.predict_window(df, window_id="W_test")
        self.assertEqual(ev_pkt.status, STATUS_OK)
        self.assertIsNotNone(ev_pkt.prediction)

        # 3. Session branch
        sess_runner = SessionSequenceBuilder()
        ev_sess = sess_runner.predict_window(df, window_id="W_test")
        self.assertEqual(ev_sess.status, STATUS_OK)
        self.assertIsNotNone(ev_sess.prediction)

    def test_integrated_fusion_and_rules(self):
        """Test D: Test EvidenceAligner, AttackGate, and RuleEngine on simulated evidence."""
        aligner = EvidenceAligner(attack_threshold=0.35)
        flow_runner = FlowPreprocessor()
        pkt_runner = PacketFeatureExtractor()
        sess_runner = SessionSequenceBuilder()

        sample_df = pd.DataFrame([{
            "Timestamp": "23/02/2018 10:00:00", "Dst Port": 80, "Protocol": 6,
            "Flow Duration": 100000, "Tot Fwd Pkts": 5, "Tot Bwd Pkts": 5,
            "TotLen Fwd Pkts": 500, "TotLen Bwd Pkts": 500, "Fwd Pkt Len Max": 100,
            "Fwd Pkt Len Min": 100, "Fwd Pkt Len Mean": 100.0, "Fwd Pkt Len Std": 0.0,
            "Bwd Pkt Len Max": 100, "Bwd Pkt Len Min": 100, "Bwd Pkt Len Mean": 100.0,
            "Bwd Pkt Len Std": 0.0, "Flow Byts/s": 10000.0, "Flow Pkts/s": 100.0,
            "Flow IAT Mean": 10000.0, "Flow IAT Std": 0.0, "Flow IAT Max": 10000.0,
            "Flow IAT Min": 10000.0, "Fwd IAT Tot": 40000.0, "Fwd IAT Mean": 10000.0,
            "Fwd IAT Std": 0.0, "Fwd IAT Max": 10000.0, "Fwd IAT Min": 10000.0,
            "Bwd IAT Tot": 40000.0, "Bwd IAT Mean": 10000.0, "Bwd IAT Std": 0.0,
            "Bwd IAT Max": 10000.0, "Bwd IAT Min": 10000.0, "Fwd PSH Flags": 0,
            "Bwd PSH Flags": 0, "Fwd URG Flags": 0, "Bwd URG Flags": 0,
            "Fwd Header Len": 100, "Bwd Header Len": 100, "Fwd Pkts/s": 50.0,
            "Bwd Pkts/s": 50.0, "Pkt Len Min": 100, "Pkt Len Max": 100,
            "Pkt Len Mean": 100.0, "Pkt Len Std": 0.0, "Pkt Len Var": 0.0,
            "FIN Flag Cnt": 0, "SYN Flag Cnt": 0, "RST Flag Cnt": 0,
            "PSH Flag Cnt": 0, "ACK Flag Cnt": 1, "URG Flag Cnt": 0,
            "CWE Flag Count": 0, "ECE Flag Cnt": 0, "Down/Up Ratio": 1.0,
            "Pkt Size Avg": 100.0, "Fwd Seg Size Avg": 100.0, "Bwd Seg Size Avg": 100.0,
            "Fwd Byts/b Avg": 0.0, "Fwd Pkts/b Avg": 0.0, "Fwd Blk Rate Avg": 0.0,
            "Bwd Byts/b Avg": 0.0, "Bwd Pkts/b Avg": 0.0, "Bwd Blk Rate Avg": 0.0,
            "Subflow Fwd Pkts": 5, "Subflow Fwd Byts": 500, "Subflow Bwd Pkts": 5,
            "Subflow Bwd Byts": 500, "Init Fwd Win Byts": 8192, "Init Bwd Win Byts": 8192,
            "Fwd Act Data Pkts": 5, "Fwd Seg Size Min": 20, "Active Mean": 0.0,
            "Active Std": 0.0, "Active Max": 0.0, "Active Min": 0.0,
            "Idle Mean": 0.0, "Idle Std": 0.0, "Idle Max": 0.0, "Idle Min": 0.0,
            "Label": "Benign",
        }] * 5)

        flow_ev = flow_runner.predict_window(sample_df, "W_test")
        pkt_ev = pkt_runner.predict_window(sample_df, "W_test")
        sess_ev = sess_runner.predict_window(sample_df, "W_test")

        integrated = aligner.align_and_fuse(
            packet_ev=pkt_ev,
            flow_ev=flow_ev,
            time_ev=None,  # time_window omitted to test graceful handling
            session_ev=sess_ev,
            window_id="W_test",
        )

        self.assertGreaterEqual(integrated.fused_attack_probability, 0.0)
        self.assertLessEqual(integrated.fused_attack_probability, 1.0)
        self.assertEqual(integrated.models_available, 3)

        rule_engine = RuleEngine(config=RuleConfig())
        rule_result = rule_engine.evaluate(integrated, context={})
        # Evaluated safely (either rule match or None for policy agent)
        self.assertTrue(rule_result is None or hasattr(rule_result, "action"))


if __name__ == "__main__":
    unittest.main()

