"""
CICFlowMeter Runner & Bidirectional Flow Reassembler
Converts raw PCAP packet streams into standard CICIDS-2018 79-column flow CSVs.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from pcap_validation.pcap_reader import PCAPReader

logger = logging.getLogger(__name__)


class CICFlowMeterRunner:
    """
    Reassembles raw packets from PCAP into standard CICIDS-2018 bidirectional flows.
    """

    def __init__(self, external_executable: Optional[str] = None):
        self.external_executable = external_executable

    def convert_pcap_to_flow_df(
        self,
        pcap_path: str | Path,
        max_packets: Optional[int] = None,
        flow_timeout_sec: float = 120.0,
    ) -> pd.DataFrame:
        """
        Parses PCAP and returns a DataFrame strictly matching the CICIDS-2018 schema.
        """
        reader = PCAPReader(pcap_path, max_packets=max_packets)
        flows: Dict[Any, Dict[str, Any]] = {}

        t0 = time.perf_counter()
        pkt_count = 0

        for pkt in reader.iter_packets():
            pkt_count += 1
            src_ep = (pkt["src_ip"], pkt["src_port"])
            dst_ep = (pkt["dst_ip"], pkt["dst_port"])
            proto = pkt["protocol"]

            # Canonical bidirectional key
            if src_ep <= dst_ep:
                fwd_key = (src_ep, dst_ep, proto)
                is_forward = True
            else:
                fwd_key = (dst_ep, src_ep, proto)
                is_forward = False

            ts = pkt["timestamp"]
            pkt_len = pkt["packet_length"]
            hdr_len = pkt["header_length"]
            tcp_flags = pkt["tcp_flags"]
            win = pkt["tcp_window"]

            if fwd_key not in flows:
                flows[fwd_key] = {
                    "start_ts": ts,
                    "last_ts": ts,
                    "fwd_ts": [],
                    "bwd_ts": [],
                    "fwd_lens": [],
                    "bwd_lens": [],
                    "fwd_hdrs": 0,
                    "bwd_hdrs": 0,
                    "fwd_win": win if is_forward else 0,
                    "bwd_win": win if not is_forward else 0,
                    "dst_port": pkt["dst_port"] if is_forward else pkt["src_port"],
                    "protocol": proto,
                    "fin": 0, "syn": 0, "rst": 0, "psh": 0, "ack": 0, "urg": 0,
                    "cwe": 0, "ece": 0,
                }

            f = flows[fwd_key]
            f["last_ts"] = ts

            if is_forward:
                f["fwd_ts"].append(ts)
                f["fwd_lens"].append(pkt_len)
                f["fwd_hdrs"] += hdr_len
                if f["fwd_win"] == 0 and win > 0:
                    f["fwd_win"] = win
            else:
                f["bwd_ts"].append(ts)
                f["bwd_lens"].append(pkt_len)
                f["bwd_hdrs"] += hdr_len
                if f["bwd_win"] == 0 and win > 0:
                    f["bwd_win"] = win

            if tcp_flags & 0x01: f["fin"] += 1
            if tcp_flags & 0x02: f["syn"] += 1
            if tcp_flags & 0x04: f["rst"] += 1
            if tcp_flags & 0x08: f["psh"] += 1
            if tcp_flags & 0x10: f["ack"] += 1
            if tcp_flags & 0x20: f["urg"] += 1
            if tcp_flags & 0x40: f["ece"] += 1
            if tcp_flags & 0x80: f["cwe"] += 1

        elapsed = time.perf_counter() - t0
        logger.info("Processed %d packets into %d flows in %.2fs", pkt_count, len(flows), elapsed)

        rows = []
        for (src_ep, dst_ep, proto), f in flows.items():
            tot_fwd = len(f["fwd_lens"])
            tot_bwd = len(f["bwd_lens"])
            tot_pkts = tot_fwd + tot_bwd
            if tot_pkts == 0:
                continue

            duration_sec = max(1e-6, f["last_ts"] - f["start_ts"])
            duration_us = duration_sec * 1e6

            fwd_lens = f["fwd_lens"] if f["fwd_lens"] else [0]
            bwd_lens = f["bwd_lens"] if f["bwd_lens"] else [0]
            all_lens = f["fwd_lens"] + f["bwd_lens"]
            tot_fwd_bytes = sum(f["fwd_lens"])
            tot_bwd_bytes = sum(f["bwd_lens"])
            tot_bytes = tot_fwd_bytes + tot_bwd_bytes

            # Inter-arrival times
            all_ts = sorted(f["fwd_ts"] + f["bwd_ts"])
            iats = np.diff(all_ts) * 1e6 if len(all_ts) > 1 else np.array([0.0])
            fwd_iats = np.diff(f["fwd_ts"]) * 1e6 if len(f["fwd_ts"]) > 1 else np.array([0.0])
            bwd_iats = np.diff(f["bwd_ts"]) * 1e6 if len(f["bwd_ts"]) > 1 else np.array([0.0])

            rows.append({
                "Timestamp": datetime.fromtimestamp(f["start_ts"]).strftime("%d/%m/%Y %H:%M:%S"),
                "Dst Port": f["dst_port"],
                "Protocol": f["protocol"],
                "Flow Duration": duration_us,
                "Tot Fwd Pkts": tot_fwd,
                "Tot Bwd Pkts": tot_bwd,
                "TotLen Fwd Pkts": tot_fwd_bytes,
                "TotLen Bwd Pkts": tot_bwd_bytes,
                "Fwd Pkt Len Max": float(np.max(fwd_lens)),
                "Fwd Pkt Len Min": float(np.min(fwd_lens)),
                "Fwd Pkt Len Mean": float(np.mean(fwd_lens)),
                "Fwd Pkt Len Std": float(np.std(fwd_lens)),
                "Bwd Pkt Len Max": float(np.max(bwd_lens)),
                "Bwd Pkt Len Min": float(np.min(bwd_lens)),
                "Bwd Pkt Len Mean": float(np.mean(bwd_lens)),
                "Bwd Pkt Len Std": float(np.std(bwd_lens)),
                "Flow Byts/s": tot_bytes / duration_sec,
                "Flow Pkts/s": tot_pkts / duration_sec,
                "Flow IAT Mean": float(np.mean(iats)),
                "Flow IAT Std": float(np.std(iats)),
                "Flow IAT Max": float(np.max(iats)),
                "Flow IAT Min": float(np.min(iats)),
                "Fwd IAT Tot": float(np.sum(fwd_iats)),
                "Fwd IAT Mean": float(np.mean(fwd_iats)),
                "Fwd IAT Std": float(np.std(fwd_iats)),
                "Fwd IAT Max": float(np.max(fwd_iats)),
                "Fwd IAT Min": float(np.min(fwd_iats)),
                "Bwd IAT Tot": float(np.sum(bwd_iats)),
                "Bwd IAT Mean": float(np.mean(bwd_iats)),
                "Bwd IAT Std": float(np.std(bwd_iats)),
                "Bwd IAT Max": float(np.max(bwd_iats)),
                "Bwd IAT Min": float(np.min(bwd_iats)),
                "Fwd PSH Flags": 1 if f["psh"] > 0 else 0,
                "Bwd PSH Flags": 0,
                "Fwd URG Flags": 1 if f["urg"] > 0 else 0,
                "Bwd URG Flags": 0,
                "Fwd Header Len": f["fwd_hdrs"],
                "Bwd Header Len": f["bwd_hdrs"],
                "Fwd Pkts/s": tot_fwd / duration_sec,
                "Bwd Pkts/s": tot_bwd / duration_sec,
                "Pkt Len Min": float(np.min(all_lens)),
                "Pkt Len Max": float(np.max(all_lens)),
                "Pkt Len Mean": float(np.mean(all_lens)),
                "Pkt Len Std": float(np.std(all_lens)),
                "Pkt Len Var": float(np.var(all_lens)),
                "FIN Flag Cnt": f["fin"],
                "SYN Flag Cnt": f["syn"],
                "RST Flag Cnt": f["rst"],
                "PSH Flag Cnt": f["psh"],
                "ACK Flag Cnt": f["ack"],
                "URG Flag Cnt": f["urg"],
                "CWE Flag Count": f["cwe"],
                "ECE Flag Cnt": f["ece"],
                "Down/Up Ratio": tot_bwd / tot_fwd if tot_fwd > 0 else 0.0,
                "Pkt Size Avg": tot_bytes / tot_pkts,
                "Fwd Seg Size Avg": float(np.mean(fwd_lens)),
                "Bwd Seg Size Avg": float(np.mean(bwd_lens)),
                "Fwd Byts/b Avg": 0.0,
                "Fwd Pkts/b Avg": 0.0,
                "Fwd Blk Rate Avg": 0.0,
                "Bwd Byts/b Avg": 0.0,
                "Bwd Pkts/b Avg": 0.0,
                "Bwd Blk Rate Avg": 0.0,
                "Subflow Fwd Pkts": tot_fwd,
                "Subflow Fwd Byts": tot_fwd_bytes,
                "Subflow Bwd Pkts": tot_bwd,
                "Subflow Bwd Byts": tot_bwd_bytes,
                "Init Fwd Win Byts": f["fwd_win"],
                "Init Bwd Win Byts": f["bwd_win"],
                "Fwd Act Data Pkts": sum(1 for l in f["fwd_lens"] if l > 0),
                "Fwd Seg Size Min": float(np.min(fwd_lens)),
                "Active Mean": 0.0,
                "Active Std": 0.0,
                "Active Max": 0.0,
                "Active Min": 0.0,
                "Idle Mean": 0.0,
                "Idle Std": 0.0,
                "Idle Max": 0.0,
                "Idle Min": 0.0,
                "Label": "Benign",
                "Src IP": src_ep[0],
                "Dst IP": dst_ep[0],
                "Src Port": src_ep[1],
            })

        df = pd.DataFrame(rows)
        # Clean infinities and NaN values
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def convert_and_save(
        self,
        pcap_path: str | Path,
        output_csv_path: str | Path,
        max_packets: Optional[int] = None,
    ) -> Path:
        out_path = Path(output_csv_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.convert_pcap_to_flow_df(pcap_path, max_packets=max_packets)
        df.to_csv(out_path, index=False)
        logger.info("Saved %d flows to: %s", len(df), out_path)
        return out_path

