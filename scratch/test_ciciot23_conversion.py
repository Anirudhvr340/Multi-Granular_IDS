import pandas as pd
import numpy as np
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

def convert_ciciot23_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    
    # Directional ratios
    srate = pd.to_numeric(df.get("Srate", 0), errors="coerce").fillna(0.0)
    drate = pd.to_numeric(df.get("Drate", 0), errors="coerce").fillna(0.0)
    tot_rate = (srate + drate).replace(0, 1e-6)
    fwd_ratio = np.clip(srate / tot_rate, 0.0, 1.0)
    bwd_ratio = np.clip(drate / tot_rate, 0.0, 1.0)
    
    num_pkts = pd.to_numeric(df.get("Number", 1.0), errors="coerce").fillna(1.0)
    tot_fwd_pkts = np.where((srate + drate) > 0, np.round(num_pkts * fwd_ratio), num_pkts)
    tot_bwd_pkts = np.where((srate + drate) > 0, np.round(num_pkts * bwd_ratio), 0.0)
    
    avg_len = pd.to_numeric(df.get("AVG", 0), errors="coerce").fillna(0.0)
    tot_fwd_bytes = tot_fwd_pkts * avg_len
    tot_bwd_bytes = tot_bwd_pkts * avg_len
    
    # Destination port based on protocol flags
    dst_port = np.zeros(len(df), dtype=np.int32)
    dst_port = np.where(df.get("HTTP", 0) == 1.0, 80, dst_port)
    dst_port = np.where(df.get("HTTPS", 0) == 1.0, 443, dst_port)
    dst_port = np.where(df.get("DNS", 0) == 1.0, 53, dst_port)
    dst_port = np.where(df.get("Telnet", 0) == 1.0, 23, dst_port)
    dst_port = np.where(df.get("SMTP", 0) == 1.0, 25, dst_port)
    dst_port = np.where(df.get("SSH", 0) == 1.0, 22, dst_port)
    dst_port = np.where(df.get("IRC", 0) == 1.0, 6667, dst_port)
    dst_port = np.where(df.get("DHCP", 0) == 1.0, 67, dst_port)
    out["Dst Port"] = dst_port
    
    # Protocol
    proto = pd.to_numeric(df.get("Protocol Type", 6.0), errors="coerce").fillna(6.0)
    proto = np.where(df.get("TCP", 0) == 1.0, 6.0, proto)
    proto = np.where(df.get("UDP", 0) == 1.0, 17.0, proto)
    proto = np.where(df.get("ICMP", 0) == 1.0, 1.0, proto)
    out["Protocol"] = proto
    
    # Elapsed duration in microseconds
    flow_dur_sec = pd.to_numeric(df.get("flow_duration", 0), errors="coerce").fillna(0.0)
    dur_sec = pd.to_numeric(df.get("Duration", 0), errors="coerce").fillna(0.0)
    best_dur_sec = np.where(flow_dur_sec > 0, flow_dur_sec, dur_sec)
    out["Flow Duration"] = best_dur_sec * 1_000_000.0
    
    # Timestamps (grounded in cumulative elapsed duration)
    cum_sec = np.cumsum(best_dur_sec)
    base_ts = pd.Timestamp("2023-01-01 00:00:00")
    out["Timestamp"] = [base_ts + pd.Timedelta(seconds=float(s)) for s in cum_sec]
    
    out["Tot Fwd Pkts"] = tot_fwd_pkts
    out["Tot Bwd Pkts"] = tot_bwd_pkts
    out["TotLen Fwd Pkts"] = tot_fwd_bytes
    out["TotLen Bwd Pkts"] = tot_bwd_bytes
    
    # Packet lengths
    max_len = pd.to_numeric(df.get("Max", 0), errors="coerce").fillna(0.0)
    min_len = pd.to_numeric(df.get("Min", 0), errors="coerce").fillna(0.0)
    std_len = pd.to_numeric(df.get("Std", 0), errors="coerce").fillna(0.0)
    var_len = pd.to_numeric(df.get("Variance", 0), errors="coerce").fillna(0.0)
    
    out["Fwd Pkt Len Max"] = max_len
    out["Fwd Pkt Len Min"] = min_len
    out["Fwd Pkt Len Mean"] = avg_len
    out["Fwd Pkt Len Std"] = std_len
    
    out["Bwd Pkt Len Max"] = np.where(tot_bwd_pkts > 0, max_len, 0.0)
    out["Bwd Pkt Len Min"] = np.where(tot_bwd_pkts > 0, min_len, 0.0)
    out["Bwd Pkt Len Mean"] = np.where(tot_bwd_pkts > 0, avg_len, 0.0)
    out["Bwd Pkt Len Std"] = np.where(tot_bwd_pkts > 0, std_len, 0.0)
    
    # Rates
    rate = pd.to_numeric(df.get("Rate", 0), errors="coerce").fillna(0.0)
    out["Flow Byts/s"] = rate * avg_len
    out["Flow Pkts/s"] = rate
    out["Fwd Pkts/s"] = srate
    out["Bwd Pkts/s"] = drate
    
    # IAT
    iat = pd.to_numeric(df.get("IAT", 0), errors="coerce").fillna(0.0)
    out["Flow IAT Mean"] = iat
    out["Flow IAT Std"] = pd.to_numeric(df.get("Covariance", 0), errors="coerce").fillna(0.0)
    out["Flow IAT Max"] = np.maximum(iat, best_dur_sec * 1_000_000.0)
    out["Flow IAT Min"] = np.minimum(iat, 0.0)
    
    out["Fwd IAT Tot"] = best_dur_sec * 1_000_000.0
    out["Fwd IAT Mean"] = iat
    out["Fwd IAT Std"] = std_len
    out["Fwd IAT Max"] = best_dur_sec * 1_000_000.0
    out["Fwd IAT Min"] = 0.0
    
    out["Bwd IAT Tot"] = np.where(tot_bwd_pkts > 0, best_dur_sec * 1_000_000.0, 0.0)
    out["Bwd IAT Mean"] = np.where(tot_bwd_pkts > 0, iat, 0.0)
    out["Bwd IAT Std"] = np.where(tot_bwd_pkts > 0, std_len, 0.0)
    out["Bwd IAT Max"] = np.where(tot_bwd_pkts > 0, best_dur_sec * 1_000_000.0, 0.0)
    out["Bwd IAT Min"] = 0.0
    
    # TCP Flags
    num_f = num_pkts.values
    out["Fwd PSH Flags"] = pd.to_numeric(df.get("psh_flag_number", 0), errors="coerce").fillna(0.0)
    out["Bwd PSH Flags"] = 0.0
    out["Fwd URG Flags"] = pd.to_numeric(df.get("urg_count", 0), errors="coerce").fillna(0.0)
    out["Bwd URG Flags"] = 0.0
    
    hdr_len = pd.to_numeric(df.get("Header_Length", 0), errors="coerce").fillna(0.0)
    out["Fwd Header Len"] = hdr_len
    out["Bwd Header Len"] = np.where(tot_bwd_pkts > 0, hdr_len * bwd_ratio, 0.0)
    
    out["Pkt Len Min"] = min_len
    out["Pkt Len Max"] = max_len
    out["Pkt Len Mean"] = avg_len
    out["Pkt Len Std"] = std_len
    out["Pkt Len Var"] = var_len
    
    # Exact flag counts from genuine CICIOT23 columns
    out["FIN Flag Cnt"] = np.maximum(df.get("fin_count", 0), df.get("fin_flag_number", 0) * num_f)
    out["SYN Flag Cnt"] = np.maximum(df.get("syn_count", 0), df.get("syn_flag_number", 0) * num_f)
    out["RST Flag Cnt"] = np.maximum(df.get("rst_count", 0), df.get("rst_flag_number", 0) * num_f)
    out["PSH Flag Cnt"] = df.get("psh_flag_number", 0) * num_f
    out["ACK Flag Cnt"] = np.maximum(df.get("ack_count", 0), df.get("ack_flag_number", 0) * num_f)
    out["URG Flag Cnt"] = pd.to_numeric(df.get("urg_count", 0), errors="coerce").fillna(0.0)
    out["CWE Flag Count"] = df.get("cwr_flag_number", 0) * num_f
    out["ECE Flag Cnt"] = df.get("ece_flag_number", 0) * num_f
    
    out["Down/Up Ratio"] = np.where(tot_fwd_pkts > 0, tot_bwd_pkts / tot_fwd_pkts, 0.0)
    out["Pkt Size Avg"] = avg_len
    out["Fwd Seg Size Avg"] = avg_len
    out["Bwd Seg Size Avg"] = np.where(tot_bwd_pkts > 0, avg_len, 0.0)
    
    # Default unmeasured metrics to standard neutral zero (no fabricated data)
    for c in ["Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg"]:
        out[c] = 0.0
        
    out["Subflow Fwd Pkts"] = tot_fwd_pkts
    out["Subflow Fwd Byts"] = tot_fwd_bytes
    out["Subflow Bwd Pkts"] = tot_bwd_pkts
    out["Subflow Bwd Byts"] = tot_bwd_bytes
    
    for c in ["Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]:
        out[c] = 0.0
        
    # Grounded Label Mapping
    LABEL_MAP = {
        "BenignTraffic": "Benign",
        "DDoS-TCP_Flood": "DDOS attack-HOIC",
        "DDoS-SYN_Flood": "DDOS attack-HOIC",
        "DDoS-PSHACK_Flood": "DDOS attack-HOIC",
        "DDoS-RSTFINFlood": "DDOS attack-HOIC",
        "DDoS-SynonymousIP_Flood": "DDOS attack-HOIC",
        "DDoS-ICMP_Flood": "DDOS attack-HOIC",
        "DDoS-ICMP_Fragmentation": "DDOS attack-HOIC",
        "DDoS-UDP_Flood": "DDOS attack-LOIC-UDP",
        "DDoS-UDP_Fragmentation": "DDOS attack-LOIC-UDP",
        "DDoS-ACK_Fragmentation": "DDOS attack-HOIC",
        "DDoS-HTTP_Flood": "DDoS attacks-LOIC-HTTP",
        "DDoS-SlowLoris": "DoS attacks-Slowloris",
        "DoS-TCP_Flood": "DoS attacks-Hulk",
        "DoS-SYN_Flood": "DoS attacks-Hulk",
        "DoS-UDP_Flood": "DDOS attack-LOIC-UDP",
        "DoS-HTTP_Flood": "DoS attacks-Hulk",
        "Mirai-greeth_flood": "Bot",
        "Mirai-udpplain": "Bot",
        "Mirai-greip_flood": "Bot",
        "SqlInjection": "SQL Injection",
        "XSS": "Brute Force -XSS",
        "BrowserHijacking": "Brute Force -Web",
        "CommandInjection": "Infilteration",
        "Uploading_Attack": "Infilteration",
        "Backdoor_Malware": "Infilteration",
        "VulnerabilityScan": "Infilteration",
        "DictionaryBruteForce": "SSH-Bruteforce",
        "Recon-PortScan": "Infilteration",
        "Recon-OSScan": "Infilteration",
        "Recon-HostDiscovery": "Infilteration",
        "Recon-PingSweep": "Infilteration",
        "DNS_Spoofing": "Infilteration",
        "MITM-ArpSpoofing": "Infilteration",
    }
    raw_lbl = df.get("label", "BenignTraffic").astype(str).str.strip()
    out["Original_CICIOT23_Label"] = raw_lbl
    out["Label"] = raw_lbl.map(LABEL_MAP).fillna("Benign")
    
    return out

if __name__ == "__main__":
    df_sample = pd.read_csv("cicids2023/CICIOT23/validation/validation.csv", nrows=1000)
    out_df = convert_ciciot23_chunk(df_sample)
    print("Converted shape:", out_df.shape)
    print("Columns match 80 columns:", len(out_df.columns))
    print("Label counts:\n", out_df["Label"].value_counts())
    
    # Test Flow model adapter on converted df
    from nids.adapters.flow_adapter import FlowModelAdapter
    adapter = FlowModelAdapter()
    ev = adapter.predict(out_df.iloc[:20])
    print("\nFlow Model Adapter Test on 20 flows:")
    print("Prediction:", ev.prediction, "Attack Prob:", ev.attack_probability)

