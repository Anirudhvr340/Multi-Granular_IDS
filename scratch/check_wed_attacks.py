import pandas as pd
df = pd.read_csv('cicids2017/Wednesday-workingHours.pcap_ISCX.csv', usecols=lambda c: 'label' in c.lower(), nrows=100000, encoding='cp1252')
df.columns = df.columns.str.strip()
print("Attack counts:")
print(df['Label'].value_counts())
print("\nFirst 10 attack row indices:")
print(df.index[df['Label'] != 'BENIGN'][:10].tolist())

