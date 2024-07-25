import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_excel('C:/Users/muham/OneDrive/Masaüstü/innova/innovahackathon/innovahackathon/yapayzeka.xlsx')
data['TIME_STAMP'] = pd.to_datetime(data['TIME_STAMP'])

data = data.dropna()

Q1 = data[['DOWNLOAD', 'UPLOAD']].quantile(0.25)
Q3 = data[['DOWNLOAD', 'UPLOAD']].quantile(0.75)
IQR = Q3 - Q1
filtered_data = data[~((data[['DOWNLOAD', 'UPLOAD']] < (Q1 - 1.5 * IQR)) | (data[['DOWNLOAD', 'UPLOAD']] > (Q3 + 1.5 * IQR))).any(axis=1)]

specific_time = pd.to_datetime('2024-03-05 18:25:00')
filtered_data = filtered_data[filtered_data['TIME_STAMP'] != specific_time]

filtered_data.loc[:, 'weekday'] = filtered_data['TIME_STAMP'].dt.weekday
filtered_data.loc[:, 'hour'] = filtered_data['TIME_STAMP'].dt.hour
filtered_data.loc[:, 'minute'] = filtered_data['TIME_STAMP'].dt.minute
filtered_data.loc[:, 'weekend'] = filtered_data['weekday'].apply(lambda x: 1 if x >= 5 else 0)

bayram_gunleri = pd.to_datetime(['2024-01-01'])
filtered_data.loc[:, 'bayram'] = filtered_data['TIME_STAMP'].dt.date.isin(bayram_gunleri.date).astype(int)

# Kriterlere göre verileri filtrele
final_filtered_data = filtered_data[(filtered_data['weekday'] < 5) & 
                                    (filtered_data['hour'] == 18) & 
                                    (filtered_data['minute'] == 25) & 
                                    (filtered_data['bayram'] == 0)]

print(f"Filtrelenmiş veri sayısı: {final_filtered_data.shape[0]}")

features = ['DOWNLOAD', 'UPLOAD']
X = final_filtered_data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Local Outlier Factor 
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_scaled)

# Test verisi
test_data = pd.DataFrame({
    'TIME_STAMP': [pd.to_datetime('2024-03-05 18:25:00')],
    'DOWNLOAD': [11927.017],
    'UPLOAD': [430.7],
    'weekday': [pd.to_datetime('2024-03-05').weekday()],
    'hour': [18],
    'minute': [25],
    'weekend': [0],
    'bayram': [0]
})
X_test = test_data[features]

X_test_scaled = scaler.transform(X_test)

test_anomaly = lof.predict(X_test_scaled)
test_data['anomaly'] = test_anomaly
test_data['anomaly'] = test_data['anomaly'].apply(lambda x: 1 if x == -1 else 0)

print("Test verisi:")
print(test_data)
print(f"Test verisi anomali mi?: {'Evet' if test_data['anomaly'].values[0] == 1 else 'Hayır'}")

plt.figure(figsize=(10, 6))
plt.scatter(final_filtered_data['DOWNLOAD'], final_filtered_data['UPLOAD'], color='blue', label='Normal')
plt.scatter(test_data['DOWNLOAD'], test_data['UPLOAD'], color='red' if test_data['anomaly'].values[0] == 1 else 'green', s=100, label='Test Verisi')
plt.xlabel('DOWNLOAD')
plt.ylabel('UPLOAD')
plt.title('Anomali Tespiti')
plt.legend()
plt.show()
