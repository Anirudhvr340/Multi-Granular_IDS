with open('convert_ciciot23_to_cicids2018.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """    # 10. TCP Flags
    out["FIN Flag Cnt"] = pd.to_numeric(df.get("fin_count", df.get("fin_flag_number", 0.0)), errors="coerce").fillna(0.0)
    out["SYN Flag Cnt"] = pd.to_numeric(df.get("syn_count", df.get("syn_flag_number", 0.0)), errors="coerce").fillna(0.0)
    out["RST Flag Cnt"] = pd.to_numeric(df.get("rst_count", df.get("rst_flag_number", 0.0)), errors="coerce").fillna(0.0)
    out["PSH Flag Cnt"] = pd.to_numeric(df.get("psh_flag_number", 0.0), errors="coerce").fillna(0.0)
    out["ACK Flag Cnt"] = pd.to_numeric(df.get("ack_count", df.get("ack_flag_number", 0.0)), errors="coerce").fillna(0.0)
    out["URG Flag Cnt"] = pd.to_numeric(df.get("urg_count", 0.0), errors="coerce").fillna(0.0)
    out["ECE Flag Cnt"] = pd.to_numeric(df.get("ece_flag_number", 0.0), errors="coerce").fillna(0.0)
    out["CWE Flag Count"] = pd.to_numeric(df.get("cwr_flag_number", 0.0), errors="coerce").fillna(0.0)
    
    out["Fwd PSH Flags"] = np.where(fwd_ratio > 0, out["PSH Flag Cnt"], 0.0)
    out["Bwd PSH Flags"] = np.where(bwd_ratio > 0, out["PSH Flag Cnt"], 0.0)
    out["Fwd URG Flags"] = np.where((fwd_ratio > 0) & (out["URG Flag Cnt"] > 0), 1.0, 0.0)
    out["Bwd URG Flags"] = np.where((bwd_ratio > 0) & (out["URG Flag Cnt"] > 0), 1.0, 0.0)
    
    # 11. Headers
    hdr_len = pd.to_numeric(df.get("Header_Length", 0.0), errors="coerce").fillna(0.0)
    out["Fwd Header Len"] = hdr_len * fwd_ratio
    out["Bwd Header Len"] = hdr_len * bwd_ratio"""

replacement = """    # 10. TCP Flags (Prefer per-flow binary flags over stream aggregate counters)
    out["FIN Flag Cnt"] = pd.to_numeric(df.get("fin_flag_number", df.get("fin_count", 0.0)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["SYN Flag Cnt"] = pd.to_numeric(df.get("syn_flag_number", df.get("syn_count", 0.0)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["RST Flag Cnt"] = pd.to_numeric(df.get("rst_flag_number", df.get("rst_count", 0.0)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["PSH Flag Cnt"] = pd.to_numeric(df.get("psh_flag_number", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["ACK Flag Cnt"] = pd.to_numeric(df.get("ack_flag_number", df.get("ack_count", 0.0)), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["URG Flag Cnt"] = 0.0  # CICIOT2023 has streaming urg_count; flow-level URG flag is unasserted
    out["ECE Flag Cnt"] = pd.to_numeric(df.get("ece_flag_number", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["CWE Flag Count"] = pd.to_numeric(df.get("cwr_flag_number", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    
    out["Fwd PSH Flags"] = np.where(fwd_ratio > 0, out["PSH Flag Cnt"], 0.0)
    out["Bwd PSH Flags"] = np.where(bwd_ratio > 0, out["PSH Flag Cnt"], 0.0)
    out["Fwd URG Flags"] = 0.0
    out["Bwd URG Flags"] = 0.0
    
    # 11. Headers (Physically grounded: IPv4+TCP max header is ~60 bytes/packet)
    hdr_len = pd.to_numeric(df.get("Header_Length", 0.0), errors="coerce").fillna(0.0)
    max_phys_hdr = np.maximum(num_pkts * 60.0, 20.0)
    hdr_len = np.minimum(hdr_len, max_phys_hdr)
    out["Fwd Header Len"] = hdr_len * fwd_ratio
    out["Bwd Header Len"] = hdr_len * bwd_ratio"""

# Normalize CRLF
text_norm = text.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')
replacement_norm = replacement.replace('\r\n', '\n')

assert target_norm in text_norm, "Target not found in text!"
updated = text_norm.replace(target_norm, replacement_norm, 1)

# Write back preserving CRLF
with open('convert_ciciot23_to_cicids2018.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(updated.replace('\n', '\r\n'))

print("Updated convert_ciciot23_to_cicids2018.py successfully!")

