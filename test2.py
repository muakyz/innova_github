import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import zscore

data = pd.read_excel('C:/Users/muham/OneDrive/Masaüstü/innova/innovahackathon/innovahackathon/yapayzeka.xlsx')
data['TIME_STAMP'] = pd.to_datetime(data['TIME_STAMP'])
data.set_index('TIME_STAMP', inplace=True)

data = data.dropna()

data['hour'] = data.index.hour
data['minute'] = data.index.minute
data['day'] = data.index.day

features = ['hour', 'minute', 'day']
target_download = 'DOWNLOAD'
target_upload = 'UPLOAD'

train_data = data[features + [target_download, target_upload]]

model_download = RandomForestRegressor(n_estimators=100, random_state=42)
model_upload = RandomForestRegressor(n_estimators=100, random_state=42)

model_download.fit(train_data[features], train_data[target_download])
model_upload.fit(train_data[features], train_data[target_upload])

# Gelecek 30 günü tahmin etmek için veri oluşturma
future_dates = pd.date_range(start='2024-03-05 18:25:00', periods=30, freq='D')
future_features = pd.DataFrame({
    'hour': future_dates.hour,
    'minute': future_dates.minute,
    'day': future_dates.day
}, index=future_dates)

download_forecast = model_download.predict(future_features)
upload_forecast = model_upload.predict(future_features)

forecast_df = pd.DataFrame({
    'ds': future_dates,
    'DOWNLOAD': download_forecast,
    'UPLOAD': upload_forecast
}).set_index('ds')

# Veri zamana göre sıralandı
data = data.sort_index()

comparison_start_date = '2024-02-04 18:25:00'
comparison_end_date = '2024-03-05 18:25:00'

# Tarih aralığını kontrol etme ve geçerli verilerin olduğundan emin olma
if comparison_start_date in data.index and comparison_end_date in data.index:
    comparison_df = data.loc[comparison_start_date:comparison_end_date].copy()
else:
    print("Belirtilen tarih aralığında veri bulunamadı.")
    comparison_df = data.loc[data.index.min():comparison_end_date].copy()  # Mevcut en erken tarihten alarak işlem yapma

# 5 Mart tahmin
five_march = pd.to_datetime('2024-03-05 18:25:00')
five_march_forecast = forecast_df.loc[five_march]

# 5 Mart gerçek
if five_march in data.index:
    five_march_actual = data.loc[five_march]
else:
    print("5 Mart 2024 tarihi için gerçek veri bulunamadı.")
    five_march_actual = pd.Series({'DOWNLOAD': np.nan, 'UPLOAD': np.nan})  # Eğer veri yoksa NaN değerler kullanma

# Anomali tespiti için Z-Score 
def check_anomaly_zscore(actual, forecast, data_series):
    z_scores = zscore(data_series)
    threshold = np.percentile(z_scores, 80)
    if abs((actual - forecast) / np.std(data_series)) > threshold:
        print(f"Anomaly Saptandı: Gerçek = {actual}, Tahmin = {forecast}")

# 5 Mart verisi için anomali kontrolü
check_anomaly_zscore(five_march_actual['DOWNLOAD'], five_march_forecast['DOWNLOAD'], data['DOWNLOAD'])
check_anomaly_zscore(five_march_actual['UPLOAD'], five_march_forecast['UPLOAD'], data['UPLOAD'])

# Grafiği oluşturma
plt.figure(figsize=(14, 7))

# DOWNLOAD değerlerini çizme
plt.plot(data.index, data['DOWNLOAD'], label='Gerçek DOWNLOAD', color='blue')
plt.plot(forecast_df.index, forecast_df['DOWNLOAD'], label='Tahmin DOWNLOAD', color='blue', linestyle='dashed')
plt.scatter(five_march, five_march_actual['DOWNLOAD'], color='red', label='5 Mart DOWNLOAD')

# UPLOAD değerlerini çizme
plt.plot(data.index, data['UPLOAD'], label='Gerçek UPLOAD', color='green')
plt.plot(forecast_df.index, forecast_df['UPLOAD'], label='Tahmin UPLOAD', color='green', linestyle='dashed')
plt.scatter(five_march, five_march_actual['UPLOAD'], color='yellow', label='5 Mart UPLOAD')

plt.xlabel('Zaman')
plt.ylabel('Değer')
plt.title('DOWNLOAD ve UPLOAD Tahminleri ile 5 Mart Verileri')
plt.legend()
plt.show()
