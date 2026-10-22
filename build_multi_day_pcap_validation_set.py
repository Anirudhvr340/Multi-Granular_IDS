"""
Multi-Day PCAP Validation Set Builder
Samples packet traffic across all 10 days of CICIDS-2018 on E: drive
and constructs a unified, multi-day validation capture: multi_day_validation.pcap.
"""

import os
import glob
import time
from pathlib import Path
import dpkt

HERE = Path(__file__).resolve().parent
OUTPUT_PCAP = HERE / "multi_day_validation.pcap"


def build_multi_day_validation_pcap(packets_per_day: int = 1500) -> Path:
    print("=" * 80)
    print("BUILDING UNIFIED MULTI-DAY VALIDATION PCAP FROM E: DRIVE")
    print(f"Target Destination: {OUTPUT_PCAP}")
    print(f"Packets sampled per day: {packets_per_day}")
    print("=" * 80)

    total_written = 0
    day_summaries = []

    with open(OUTPUT_PCAP, "wb") as out_f:
        writer = dpkt.pcap.Writer(out_f)

        # Baseline start timestamp (2018-02-14 09:00:00 UTC)
        base_time = 1518600000.0

        for day in range(1, 11):
            day_pat = f"E:/Day {day} */pcap/*"
            pcap_files = glob.glob(day_pat)

            if not pcap_files:
                print(f"[!] No PCAP files found for Day {day}")
                continue

            # Prioritize attack victim hosts or representative hosts
            selected_file = None
            for cand in pcap_files:
                fname = os.path.basename(cand)
                if "69.25" in fname or "69.18" in fname or "69.15" in fname:
                    selected_file = cand
                    break
            if not selected_file:
                selected_file = pcap_files[0]

            fname = os.path.basename(selected_file)
            size_mb = os.path.getsize(selected_file) / (1024 * 1024)
            print(f"[*] Day {day:2d} -> Sampling {packets_per_day} packets from: {fname} ({size_mb:.1f} MB)...")

            day_written = 0
            with open(selected_file, "rb") as in_f:
                try:
                    reader = dpkt.pcap.Reader(in_f)
                except Exception:
                    in_f.seek(0)
                    try:
                        reader = dpkt.pcapng.Reader(in_f)
                    except Exception as e:
                        print(f"    [!] Error opening {fname}: {e}")
                        continue

                # Space out timestamps across 30-second windows
                day_offset = (day - 1) * 120.0  # 4 windows of 30s per day

                for i, (ts, buf) in enumerate(reader):
                    if day_written >= packets_per_day:
                        break

                    # Construct monotonic sequential timestamps
                    pkt_ts = base_time + day_offset + (day_written * 0.05)
                    try:
                        writer.writepkt(buf, ts=pkt_ts)
                        day_written += 1
                    except Exception:
                        continue

            print(f"    [+] Ingested {day_written} packets from Day {day}.")
            total_written += day_written
            day_summaries.append({
                "day": day,
                "source_file": fname,
                "packets_sampled": day_written,
            })

    print("\n" + "=" * 80)
    print(f"UNIFIED MULTI-DAY PCAP COMPLETE: {total_written:,} TOTAL PACKETS")
    print(f"Saved to: {OUTPUT_PCAP} ({os.path.getsize(OUTPUT_PCAP)/(1024*1024):.2f} MB)")
    print("=" * 80)
    return OUTPUT_PCAP


if __name__ == "__main__":
    build_multi_day_validation_pcap()

