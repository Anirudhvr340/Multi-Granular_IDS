import pandas as pd

for fname, name in [
    ('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', 'Wednesday DoS'),
    ('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Friday DDoS'),
    ('cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv', 'Tuesday FTP/SSH')
]:
    df = pd.read_csv(fname, usecols=lambda c: 'label' in c.lower(), nrows=100000, encoding='cp1252')
    df.columns = df.columns.str.strip()
    attack_indices = df.index[df['Label'] != 'BENIGN'].tolist()
    if attack_indices:
        print(f"{name}: First attack at index {attack_indices[0]}, total attacks in first 100k: {len(attack_indices)}")
    else:
        print(f"{name}: No attacks found in first 100k")

