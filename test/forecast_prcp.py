import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


model = load_model('D:/pythonProjectTotal/weather_forecast/model/forecast_future_prcp.keras')
scaler_features = joblib.load('D:/pythonProjectTotal/weather_forecast/scaler/scaler_features.pkl')
scaler_target_temp = joblib.load('D:/pythonProjectTotal/weather_forecast/scaler/scaler_target_temp.pkl')


data = pd.read_excel('D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_04_hour.xlsx', engine='openpyxl')
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)


hours_back = 6
future_hours = 1
for i in range(1, hours_back + 1):
    data[f'temp_lag{i}'] = data['temp'].shift(i)
    data[f'prcp_lag{i}'] = data['prcp'].shift(i)
    data[f'wdir_lag{i}'] = data['wdir'].shift(i)
    data[f'wspd_lag{i}'] = data['wspd'].shift(i)
    data[f'pres_lag{i}'] = data['pres'].shift(i)
    data[f'coco_lag{i}'] = data['coco'].shift(i)


features = ['temp','prcp', 'wdir', 'wspd', 'pres', 'coco'] + \
           [f'temp_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'prcp_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'wdir_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'wspd_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'pres_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'coco_lag{i}' for i in range(1, hours_back + 1)]

data_scaled_features = scaler_features.transform(data[features])
X_new = np.reshape(data_scaled_features, (data_scaled_features.shape[0], 1, data_scaled_features.shape[1]))


predicted_coco = model.predict(X_new)
predicted_coco = scaler_target_temp.inverse_transform(predicted_coco).flatten()


timestamps = data.index[hours_back:]
actual_coco = data['prcp'][hours_back:]
predicted_coco_adjusted = predicted_coco[hours_back:]


plt.figure(figsize=(15, 7))
plt.plot(timestamps, actual_coco, label='Actual coco', color='blue', linestyle='-')
plt.plot(timestamps, (predicted_coco_adjusted+6), label='Predicted coco', color='red', linestyle='--')
plt.title('Predicted vs. Actual coco')
plt.xlabel('Time')
plt.ylabel('coco')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=48))
plt.gcf().autofmt_xdate()
plt.show()