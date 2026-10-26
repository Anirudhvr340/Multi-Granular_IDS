import pandas as pd
import numpy as np

print("=" * 80)
print("ANALYZING ATTACK CONCENTRATION ACROSS WEDNESDAY DATASET")
print("=" * 80)

csv_path = "cicids2017/Wednesday-workingHours.pcap_ISCX.csv"

# Scan the full file in chunks of 50,000 rows
chunksize = 50000
total_rows = 0

label_chunks = []

for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, usecols=lambda c: 'label' in c.lower(), encoding="cp1252", low_memory=False)):
    chunk.columns = chunk.columns.str.strip()
    s_labels = chunk["Label"].astype(str).str.strip()
    
    start_idx = total_rows
    end_idx = total_rows + len(chunk)
    
    for lbl, count in s_labels.value_counts().items():
        if lbl.upper() != "BENIGN":
            # Find first and last in this chunk
            sub = s_labels[s_labels == lbl]
            label_chunks.append({
                "chunk": chunk_idx,
                "label": lbl,
                "count": count,
                "first_row": sub.index[0] + total_rows,
                "last_row": sub.index[-1] + total_rows,
            })
            
    total_rows = end_idx

print(f"\nTotal Rows in Wednesday-workingHours.pcap_ISCX.csv: {total_rows:,}")

df_chunks = pd.DataFrame(label_chunks)

print("\n--- ATTACK SUMMARY & ROW RANGES ---")
summary = df_chunks.groupby("label").agg(
    total_flows=("count", "sum"),
    start_row=("first_row", "min"),
    end_row=("last_row", "max"),
).reset_index()

summary["percent_of_wednesday"] = (summary["total_flows"] / total_rows) * 100
summary = summary.sort_values(by="start_row")
print(summary.to_string(index=False))

total_attacks = summary["total_flows"].sum()
total_benign = total_rows - total_attacks
print(f"\nTotal Attacks: {total_attacks:,} ({(total_attacks/total_rows)*100:.2f}%)")
print(f"Total Benign:  {total_benign:,} ({(total_benign/total_rows)*100:.2f}%)")

# Detailed Row Sections (in 10% blocks)
print("\n--- TIMELINE CONCENTRATION (10% FILE BLOCKS) ---")
block_size = total_rows // 10
for b in range(10):
    b_start = b * block_size
    b_end = (b + 1) * block_size if b < 9 else total_rows
    
    # Overlap with chunks
    b_attacks = 0
    attack_breakdown = {}
    for _, row in df_chunks.iterrows():
        # Check overlap
        overlap_start = max(b_start, row["first_row"])
        overlap_end = min(b_end, row["last_row"] + 1)
        if overlap_end > overlap_start:
            # Estimate count in this block
            # For exact count, proportional or direct:
            c = row["count"]
            span = max(1, row["last_row"] - row["first_row"] + 1)
            est = int(round(c * (overlap_end - overlap_start) / span))
            b_attacks += est
            attack_breakdown[row["label"]] = attack_breakdown.get(row["label"], 0) + est
            
    pct_att = (b_attacks / (b_end - b_start)) * 100
    top_att = ", ".join([f"{k} (~{v:,})" for k, v in attack_breakdown.items()]) if attack_breakdown else "Pure Benign"
    print(f"Block {b+1:02d} (Rows {b_start:>7,} - {b_end:>7,}): Attack Share = {pct_att:>5.1f}% | {top_att}")

