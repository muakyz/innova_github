import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('C:/Users/muham/OneDrive/Masaüstü/innova/innovahackathon/innovahackathon/data_cleaned_no_outliers.xlsx')
data['TIME_STAMP'] = pd.to_datetime(data['TIME_STAMP'])

data['hour'] = data['TIME_STAMP'].dt.hour

hourly_traffic = data.groupby('hour').agg({'DOWNLOAD': 'mean', 'UPLOAD': 'mean'}).reset_index()

peak_hours_download = hourly_traffic.loc[hourly_traffic['DOWNLOAD'].idxmax()]
peak_hours_upload = hourly_traffic.loc[hourly_traffic['UPLOAD'].idxmax()]

# Kesimlerinin en yoğun olduğu zamanı hesaplama
hourly_traffic['combined'] = hourly_traffic['DOWNLOAD'] + hourly_traffic['UPLOAD']
peak_hours_combined = hourly_traffic.loc[hourly_traffic['combined'].idxmax()]

print("Trafiğin en yoğun olduğu saat dilimi (DOWNLOAD):")
print(f"Saat: {peak_hours_download['hour']}:00, Ortalama DOWNLOAD: {peak_hours_download['DOWNLOAD']}")

print("Trafiğin en yoğun olduğu saat dilimi (UPLOAD):")
print(f"Saat: {peak_hours_upload['hour']}:00, Ortalama UPLOAD: {peak_hours_upload['UPLOAD']}")

print("Trafiğin en yoğun olduğu saat dilimi (Combined):")
print(f"Saat: {peak_hours_combined['hour']}:00, Ortalama Combined: {peak_hours_combined['combined']}")

plt.figure(figsize=(12, 6))
plt.plot(hourly_traffic['hour'], hourly_traffic['DOWNLOAD'], label='DOWNLOAD')
plt.plot(hourly_traffic['hour'], hourly_traffic['UPLOAD'], label='UPLOAD')
plt.plot(hourly_traffic['hour'], hourly_traffic['combined'], label='Combined', linestyle='--')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Traffic')
plt.title('Hourly Traffic Analysis')
plt.legend()
plt.grid(True)
plt.show()
    