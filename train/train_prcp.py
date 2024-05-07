import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt


files = [
    'D:/pythonProjectTotal/weather_forecast/data/weather_data_2024_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_04_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_05_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_06_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_07_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_08_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_09_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_10_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_11_hour.xlsx',
    'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_12_hour.xlsx',
]
dataframes = [pd.read_excel(file, engine='openpyxl') for file in files]
data = pd.concat(dataframes, ignore_index=True)
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

data['future_prcp'] = data['prcp'].shift(-future_hours)
data.dropna(inplace=True)

features = ['temp','prcp', 'wdir', 'wspd', 'pres', 'coco'] + \
           [f'temp_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'prcp_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'wdir_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'wspd_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'pres_lag{i}' for i in range(1, hours_back + 1)] + \
           [f'coco_lag{i}' for i in range(1, hours_back + 1)]


scaler_features = MinMaxScaler()
data_scaled_features = scaler_features.fit_transform(data[features])

scaler_target_pres = MinMaxScaler()
data_scaled_target_pres = scaler_target_pres.fit_transform(data[['future_prcp']])



X = np.reshape(data_scaled_features, (data_scaled_features.shape[0], 1, data_scaled_features.shape[1]))
y_temp = data_scaled_target_pres


X_train, X_test, y_train_pres, y_test_pres = train_test_split(X, y_temp, test_size=0.2, random_state=42)



input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm_layer = LSTM(50, activation='relu')(input_layer)
output_pres = Dense(1, name='pres_output')(lstm_layer)  # 温度预测输出


model = Model(inputs=input_layer, outputs=[output_pres])
model.compile(optimizer='adam', loss={'pres_output': 'mean_squared_error',})

history = model.fit(X_train, {'pres_output': y_train_pres}, epochs=64, batch_size=32, validation_split=0.1, verbose=2)
model.save('D:/pythonProjectTotal/weather_forecast/model/forecast_pres.keras')
joblib.dump(scaler_features, 'D:/pythonProjectTotal/weather_forecast/scaler/scaler_features.pkl')
joblib.dump(scaler_target_pres, 'D:/pythonProjectTotal/weather_forecast/scaler/scaler_target_pres.pkl')


plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

if 'accuracy' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.show()