import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf


def custom_activation(x):
    return tf.maximum(0.1 * x, x)

# 数据加载和预处理
def load_and_preprocess_data(files):
    dataframes = [pd.read_excel(file, engine='openpyxl') for file in files]
    data = pd.concat(dataframes, ignore_index=True)
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)
    return data


# 添加滞后特征
def add_lagged_features(data, features, hours_back):
    for feature in features:
        for i in range(1, hours_back + 1):
            data[f'{feature}_lag{i}'] = data[feature].shift(i)
    return data


# 构建模型
#def custom_activation(x):
   # return tf.nn.relu(x) * tf.sigmoid(x)


def build_model(input_shape, output_name, lstm_units=50, num_layers=2):
    input_layer = Input(shape=input_shape)
    if num_layers > 1:
        lstm_layer = LSTM(lstm_units, activation=custom_activation, return_sequences=True)(input_layer)
        for i in range(num_layers - 2):
            lstm_layer = LSTM(lstm_units, activation=custom_activation, return_sequences=True)(lstm_layer)
        lstm_layer = LSTM(lstm_units, activation=custom_activation)(lstm_layer)
    else:
        lstm_layer = LSTM(lstm_units, activation=custom_activation)(input_layer)

    output_layer = Dense(1, name=f'{output_name}_output')(lstm_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# 训练模型
def train_model(model, X_train, y_train, X_test, y_test, epochs=32, batch_size=16):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=1)
    return history


# 数据和文件设置
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
data = load_and_preprocess_data(files)
features = ['temp', 'prcp', 'wdir', 'wspd', 'pres', 'coco']
hours_back = 6
future_hours = 1
data = add_lagged_features(data, features, hours_back)
features = features + [f'{feature}_lag{i}' for feature in features for i in range(1, hours_back + 1)]
data['future_pres'] = data['pres'].shift(-future_hours)
data['future_temp'] = data['temp'].shift(-future_hours)
data.dropna(inplace=True)

# 数据标准化和分割


scaler_features = MinMaxScaler()
data_scaled_features = scaler_features.fit_transform(data[features])
X = np.reshape(data_scaled_features, (data_scaled_features.shape[0], 1, data_scaled_features.shape[1]))

targets = {
    'future_temp': data['temp'].shift(-future_hours),
    'future_pres': data['pres'].shift(-future_hours)
}

for target, y in targets.items():
    scaler_target = MinMaxScaler()
    y_scaled = scaler_target.fit_transform(data[[target]])
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

    # 构建和训练模型
    model = build_model(X_train.shape[1:], output_name=target, lstm_units=100)
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=64)

    # 保存模型和标准化器
    model.save(f'model/forecast_{target}.keras')
    joblib.dump(scaler_target, f'scaler/scaler_{target}.pkl')

    # 可视化结果
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss for {target}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
