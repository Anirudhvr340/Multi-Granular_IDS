"""
Evidence Aligner & Multi-Granular Aggregator
Aligns outputs from Packet, Flow, Time-Window, and Session branches into IntegratedEvidence.
"""

import logging
from typing import Optional, Dict, Any
from nids.attack_gate import AttackGate
from nids.evidence_schema import EvidenceRecord, IntegratedEvidence, STATUS_OK

logger = logging.getLogger(__name__)


class EvidenceAligner:
    """
    Coordinates and aligns evidence records across the 4 granular models.
    """

    def __init__(self, attack_threshold: float = 0.35):
        self.attack_gate = AttackGate(threshold=attack_threshold)
        self.stats = {
            "packet": {"invoked": 0, "success": 0, "errors": 0},
            "flow": {"invoked": 0, "success": 0, "errors": 0},
            "time_window": {"invoked": 0, "success": 0, "errors": 0},
            "session": {"invoked": 0, "success": 0, "errors": 0},
        }

    def record_status(self, model_key: str, record: Optional[EvidenceRecord]):
        self.stats[model_key]["invoked"] += 1
        if record is not None and record.status == STATUS_OK:
            self.stats[model_key]["success"] += 1
        else:
            self.stats[model_key]["errors"] += 1

    def align_and_fuse(
        self,
        packet_ev: Optional[EvidenceRecord],
        flow_ev: Optional[EvidenceRecord],
        time_ev: Optional[EvidenceRecord],
        session_ev: Optional[EvidenceRecord],
        window_id: str,
    ) -> IntegratedEvidence:
        """
        Fuses available model evidence using the confidence-weighted AttackGate.
        """
        self.record_status("packet", packet_ev)
        self.record_status("flow", flow_ev)
        self.record_status("time_window", time_ev)
        self.record_status("session", session_ev)

        integrated = self.attack_gate.evaluate(
            packet_ev=packet_ev,
            flow_ev=flow_ev,
            time_ev=time_ev,
            session_ev=session_ev,
            window_id=window_id,
        )
        return integrated

    def get_audit_summary(self) -> Dict[str, Any]:
        return dict(self.stats)

