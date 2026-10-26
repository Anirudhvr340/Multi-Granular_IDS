import pandas as pd

for fname, name in [
    ('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', 'Wednesday DoS'),
    ('cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Friday DDoS'),
    ('cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv', 'Tuesday FTP/SSH')
]:
    df = pd.read_csv(fname, nrows=100000, encoding='cp1252')
    df.columns = df.columns.str.strip()
    counts = df['Label'].value_counts()
    print(f"=== {name} ===")
    print(counts)

