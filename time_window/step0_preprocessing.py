import warnings
warnings.filterwarnings("ignore")
import os
import glob
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==============================
# CONFIG
# ==============================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATTERN = os.path.join(ROOT_DIR, "0*-*-2018.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS", "5"))
CHUNK_SIZE = 100_000
MAX_ROWS_PER_FILE = 500_000
USECOLS = [
    "Timestamp",
    "Label",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Duration",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Pkt Len Mean",
    "Active Mean",
    "Idle Mean",
    "Dst Port",
    "Protocol",
    "SYN Flag Cnt",
    "FIN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_numeric_columns(df):
    return [c for c in df.columns if c not in ["Timestamp", "Label", "Protocol"]]


def clean_chunk(chunk):
    chunk = chunk.copy()
    chunk = chunk.drop_duplicates()
    chunk["Timestamp"] = pd.to_datetime(
        chunk["Timestamp"], errors="coerce", dayfirst=True, format="mixed"
    )
    chunk = chunk.loc[chunk["Timestamp"].dt.year == 2018]
    chunk["Label"] = chunk["Label"].astype(str).str.strip()
    chunk = chunk.dropna(subset=["Timestamp", "Label"])
    chunk = chunk.loc[chunk["Label"] != ""]

    chunk["Timestamp"] = chunk["Timestamp"].astype("datetime64[ns]")

    numeric_cols = get_numeric_columns(chunk)
    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
    chunk[numeric_cols] = chunk[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return chunk


def window_aggregate(window_df, window_start):
    total_flows = len(window_df)
    total_fwd_pkts = float(window_df.get("Tot Fwd Pkts", pd.Series([0])).astype(float).sum())
    total_bwd_pkts = float(window_df.get("Tot Bwd Pkts", pd.Series([0])).astype(float).sum())
    total_packets = total_fwd_pkts + total_bwd_pkts
    total_fwd_bytes = float(window_df.get("TotLen Fwd Pkts", pd.Series([0])).astype(float).sum())
    total_bwd_bytes = float(window_df.get("TotLen Bwd Pkts", pd.Series([0])).astype(float).sum())
    total_bytes = total_fwd_bytes + total_bwd_bytes

    def safe_mean(col):
        if col not in window_df.columns:
            return 0.0
        try:
            vals = pd.to_numeric(window_df[col], errors="coerce")
            return float(vals.mean()) if len(vals.dropna()) > 0 else 0.0
        except:
            return 0.0

    def safe_std(col):
        if col not in window_df.columns:
            return 0.0
        values = pd.to_numeric(window_df[col], errors="coerce").dropna()
        return float(values.std(ddof=0)) if len(values) > 0 else 0.0

    timestamps = pd.to_datetime(window_df["Timestamp"], errors="coerce").sort_values()
    inter_arrivals = timestamps.diff().dt.total_seconds().dropna()
    mean_inter_arrival = float(inter_arrivals.mean()) if len(inter_arrivals) else 0.0
    std_inter_arrival = float(inter_arrivals.std(ddof=0)) if len(inter_arrivals) else 0.0
    burstiness = std_inter_arrival / max(mean_inter_arrival, 1e-6)

    def quantile(col, q):
        if col not in window_df.columns:
            return 0.0
        values = pd.to_numeric(window_df[col], errors="coerce").dropna()
        return float(values.quantile(q)) if len(values) else 0.0

    flow_packets = (
        pd.to_numeric(window_df.get("Tot Fwd Pkts", 0), errors="coerce").fillna(0)
        + pd.to_numeric(window_df.get("Tot Bwd Pkts", 0), errors="coerce").fillna(0)
    )
    flow_bytes = (
        pd.to_numeric(window_df.get("TotLen Fwd Pkts", 0), errors="coerce").fillna(0)
        + pd.to_numeric(window_df.get("TotLen Bwd Pkts", 0), errors="coerce").fillna(0)
    )
    flow_duration = pd.to_numeric(window_df.get("Flow Duration", 0), errors="coerce").fillna(0)
    syn = pd.to_numeric(window_df.get("SYN Flag Cnt", 0), errors="coerce").fillna(0)
    ack = pd.to_numeric(window_df.get("ACK Flag Cnt", 0), errors="coerce").fillna(0)
    rst = pd.to_numeric(window_df.get("RST Flag Cnt", 0), errors="coerce").fillna(0)
    flow_count = max(total_flows, 1)
    active_span_seconds = max(
        float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds())
        if len(timestamps) else 0.0,
        0.0,
    )

    label_counts = window_df["Label"].value_counts()
    attack_count = int((window_df["Label"] != "Benign").sum())
    attack_labels = window_df.loc[window_df["Label"] != "Benign", "Label"].value_counts()
    # Require a meaningful attack ratio to avoid labeling benign-dominant windows
    # as attacks due to a single stray flow.  Threshold: >= 10 % of flows.
    attack_ratio_in_window = attack_count / max(len(window_df), 1)
    if len(attack_labels) and attack_ratio_in_window >= 0.10:
        majority_label = attack_labels.idxmax()
    else:
        majority_label = "Benign"

    return {
        "window_start": window_start,
        "total_flows": total_flows,
        "flow_rate": total_flows / WINDOW_SECONDS,
        "total_packets": total_packets,
        "packet_rate": total_packets / WINDOW_SECONDS,
        "total_bytes": total_bytes,
        "byte_rate": total_bytes / WINDOW_SECONDS,
        "total_fwd_pkts": total_fwd_pkts,
        "total_bwd_pkts": total_bwd_pkts,
        "total_fwd_bytes": total_fwd_bytes,
        "total_bwd_bytes": total_bwd_bytes,
        "fwd_bwd_pkt_ratio": total_fwd_pkts / max(total_bwd_pkts, 1.0),
        "fwd_bwd_byte_ratio": total_fwd_bytes / max(total_bwd_bytes, 1.0),
        "mean_flow_duration": safe_mean("Flow Duration"),
        "mean_flow_bytes": safe_mean("Flow Byts/s"),
        "mean_flow_pkts": safe_mean("Flow Pkts/s"),
        "mean_pkt_len": safe_mean("Pkt Len Mean"),
        "mean_iat": safe_mean("Flow IAT Mean"),
        "mean_active": safe_mean("Active Mean"),
        "mean_idle": safe_mean("Idle Mean"),
        "std_flow_duration": safe_std("Flow Duration"),
        "std_flow_bytes": safe_std("Flow Byts/s"),
        "std_flow_pkts": safe_std("Flow Pkts/s"),
        "flow_duration_p50": quantile("Flow Duration", 0.50),
        "flow_duration_p90": quantile("Flow Duration", 0.90),
        "flow_duration_p99": quantile("Flow Duration", 0.99),
        "flow_packets_p50": float(flow_packets.quantile(0.50)),
        "flow_packets_p90": float(flow_packets.quantile(0.90)),
        "flow_bytes_p50": float(flow_bytes.quantile(0.50)),
        "flow_bytes_p90": float(flow_bytes.quantile(0.90)),
        "single_packet_flow_ratio": float((flow_packets <= 1).sum() / flow_count),
        "small_flow_ratio": float((flow_packets <= 4).sum() / flow_count),
        "zero_bwd_flow_ratio": float((pd.to_numeric(window_df.get("Tot Bwd Pkts", 0), errors="coerce").fillna(0) == 0).sum() / flow_count),
        "syn_no_ack_ratio": float(((syn > 0) & (ack == 0)).sum() / flow_count),
        "rst_flow_ratio": float((rst > 0).sum() / flow_count),
        "active_span_seconds": active_span_seconds,
        "window_occupancy": active_span_seconds / max(WINDOW_SECONDS, 1),
        "flow_density": total_flows / max(active_span_seconds, 1e-6),
        "mean_inter_arrival": mean_inter_arrival,
        "std_inter_arrival": std_inter_arrival,
        "burstiness": burstiness,
        "unique_dst_ports": int(window_df["Dst Port"].nunique()) if "Dst Port" in window_df.columns else 0,
        "unique_protocols": int(window_df["Protocol"].nunique()) if "Protocol" in window_df.columns else 0,
        "total_syn": float(pd.to_numeric(window_df.get("SYN Flag Cnt", pd.Series([0])), errors="coerce").sum()),
        "total_fin": float(pd.to_numeric(window_df.get("FIN Flag Cnt", pd.Series([0])), errors="coerce").sum()),
        "total_rst": float(pd.to_numeric(window_df.get("RST Flag Cnt", pd.Series([0])), errors="coerce").sum()),
        "total_psh": float(pd.to_numeric(window_df.get("PSH Flag Cnt", pd.Series([0])), errors="coerce").sum()),
        "total_ack": float(pd.to_numeric(window_df.get("ACK Flag Cnt", pd.Series([0])), errors="coerce").sum()),
        "total_urg": float(pd.to_numeric(window_df.get("URG Flag Cnt", pd.Series([0])), errors="coerce").sum()),
        "majority_label": majority_label,
        "attack_count": attack_count,
        "attack_ratio": attack_count / max(total_flows, 1),
    }


def process_chunk(chunk, pending):
    if pending is not None and not pending.empty:
        chunk = pd.concat([pending, chunk], ignore_index=True)

    if chunk.empty:
        return pd.DataFrame(), []

    chunk["Timestamp"] = pd.to_datetime(
        chunk["Timestamp"], errors="coerce", dayfirst=True, format="mixed"
    )
    chunk = chunk.loc[chunk["Timestamp"].dt.year == 2018]
    chunk = chunk.dropna(subset=["Timestamp"]).copy()
    chunk["Timestamp"] = chunk["Timestamp"].astype("datetime64[ns]")

    if chunk.empty:
        return pd.DataFrame(), []

    chunk = chunk.sort_values("Timestamp").reset_index(drop=True)
    chunk["window_start"] = chunk["Timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
    last_window_start = chunk["window_start"].iloc[-1]

    pending_mask = chunk["window_start"] == last_window_start
    pending_chunk = chunk[pending_mask].copy()
    finalize_chunk = chunk[~pending_mask].copy()

    aggregated = []
    if not finalize_chunk.empty:
        for window_start, window_df in finalize_chunk.groupby("window_start"):
            aggregated.append(window_aggregate(window_df, window_start))

    return pending_chunk, aggregated


def main():
    print("Loading CICIDS2018 CSV files in chunks...")
    files = sorted(glob.glob(DATA_PATTERN))
    print("Data files found:", files)

    window_rows = []

    for file in files:
        print(f"\nLoading file: {file}")
        file_chunks = []
        try:
            reader = pd.read_csv(
                file,
                usecols=USECOLS,
                dtype=str,
                chunksize=CHUNK_SIZE,
                nrows=MAX_ROWS_PER_FILE,
                low_memory=False,
                na_values=["", "NA", "NaN", "nan", "?"],
            )

            total_rows = 0
            for chunk in reader:
                chunk = clean_chunk(chunk)
                total_rows += len(chunk)
                if not chunk.empty:
                    file_chunks.append(chunk)
            print(f"    Rows processed: {total_rows}")
            if file_chunks:
                # Files are separate calendar days, so no window can cross files.
                combined = pd.concat(file_chunks, ignore_index=True)
                combined = combined.sort_values("Timestamp").reset_index(drop=True)
                combined["window_start"] = combined["Timestamp"].dt.floor(f"{WINDOW_SECONDS}s")
                for window_start, window_df in combined.groupby("window_start", sort=True):
                    window_rows.append(window_aggregate(window_df, window_start))
                del file_chunks, combined
                gc.collect()
        except Exception as err:
            print(f"    Skipping file due to error: {err}")
            continue

    if not window_rows:
        raise RuntimeError("No windows were generated. Check the CSV files and Timestamp parsing.")

    if not window_rows:
        raise RuntimeError("No windows were generated. Check the CSV files and Timestamp parsing.")

    features_df = pd.DataFrame(window_rows)
    features_df = features_df.sort_values("window_start").reset_index(drop=True)
    # NOTE: Lag features (previous_*, delta_*) are computed in step1_split.py
    # AFTER the train/test split to prevent temporal data leakage.
    # These are target-derived diagnostics, never model inputs.
    numeric_columns = [
        column for column in features_df.select_dtypes(include=[np.number]).columns
        if column not in ["attack_count", "attack_ratio"]
    ]
    features_df[numeric_columns] = (
        features_df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    print(f"Total windows generated: {len(features_df)}")
    print("Example feature row:")
    print(features_df.head(2).T)

    label_encoder = LabelEncoder()
    features_df["majority_label_encoded"] = label_encoder.fit_transform(features_df["majority_label"])
    np.save(os.path.join(OUTPUT_DIR, "window_class_names.npy"), label_encoder.classes_)

    np.save(os.path.join(OUTPUT_DIR, "window_features.npy"), features_df[numeric_columns].to_numpy(dtype=np.float32))
    np.save(os.path.join(OUTPUT_DIR, "window_binary_labels.npy"), np.array((features_df["attack_count"] > 0).astype(np.int64), dtype=np.int64))
    np.save(os.path.join(OUTPUT_DIR, "window_multiclass_labels.npy"), features_df["majority_label_encoded"].to_numpy(dtype=np.int64))
    np.save(os.path.join(OUTPUT_DIR, "window_times.npy"), features_df["window_start"].to_numpy(dtype="datetime64[ns]"))
    np.save(os.path.join(OUTPUT_DIR, "window_feature_names.npy"), np.array(numeric_columns, dtype=object))

    print("\nStep 0 COMPLETED: window features saved to processed/")


if __name__ == "__main__":
    main()
