import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('C:/Users/muham/OneDrive/Masaüstü/innova/innovahackathon/innovahackathon/yapayzeka.xlsx')
data['TIME_STAMP'] = pd.to_datetime(data['TIME_STAMP'])

print("Eksik veri sayısı:", data.isnull().sum().sum())
print("Eksik veri sayısı: Sütun", data.isnull().sum())

plt.figure(figsize=(12, 6))
plt.plot(data['TIME_STAMP'], data['DOWNLOAD'], label='DOWNLOAD')
plt.plot(data['TIME_STAMP'], data['UPLOAD'], label='UPLOAD', alpha=0.7)
plt.xlabel('Zaman')
plt.ylabel('Veri (MB)')
plt.title('DOWNLOAD ve UPLOAD Zaman Serisi')
plt.legend()
plt.show()
