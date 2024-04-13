import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 加载保存的多输出模型和归一化对象
model = load_model('D:/pythonProjectTotal/weather_forecast/forecast/model/forecast_pres.keras')
scaler_features = joblib.load('D:/pythonProjectTotal/weather_forecast/forecast/scaler/scaler_features.pkl')
scaler_target_temp = joblib.load('D:/pythonProjectTotal/weather_forecast/forecast/scaler/scaler_target_pres.pkl')

# 加载新数据集
data = pd.read_excel('D:/pythonProjectTotal/weather_forecast/data/test/weather_data_2023_extreme.xlsx', engine='openpyxl')
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
    # 增加coco的滞后特征
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

# 使用模型进行预测
predicted_pres = model.predict(X_new)
predicted_pres = scaler_target_temp.inverse_transform(predicted_pres).flatten()

# 准备绘图数据
timestamps = data.index[hours_back:]
actual_temperatures = data['pres'][hours_back:]
predicted_temperatures_adjusted = predicted_pres[hours_back:]

# 绘制预测温度和实际温度
plt.figure(figsize=(15, 7))
plt.plot(timestamps, actual_temperatures, label='Actual Pres', color='blue', linestyle='-')
plt.plot(timestamps, predicted_temperatures_adjusted, label='Predicted Pres', color='red', linestyle='--')
plt.title('Predicted vs. Actual Temperature')
plt.xlabel('Time')
plt.ylabel('Pres(Pa)')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=48))
plt.gcf().autofmt_xdate()  # 自动旋转日期标记以避免重叠
plt.show()