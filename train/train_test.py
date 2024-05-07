import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

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

# 创建模型
def create_model(X_train, output_units, lstm_units=50):
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_layer = Bidirectional(LSTM(lstm_units, activation='relu'))(input_layer)
    outputs = [Dense(1, name=f'{output}_output')(lstm_layer) for output in output_units]
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam', loss={f'{output}_output': 'mean_squared_error' for output in output_units})
    return model

# 主程序
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
data = add_lagged_features(data, features, hours_back=6)

# 目标变量和特征
target_vars = ['temp', 'pres']
for var in target_vars:
    data[f'future_{var}'] = data[var].shift(-1)
data.dropna(inplace=True)

# 数据标准化
scaler_features = MinMaxScaler()
data_scaled_features = scaler_features.fit_transform(data[features + [f'{f}_lag{i}' for f in features for i in range(1, 7)]])

scalers_target = {var: MinMaxScaler() for var in target_vars}
data_scaled_targets = {var: scalers_target[var].fit_transform(data[[f'future_{var}']]) for var in target_vars}

# 数据分割
X = np.reshape(data_scaled_features, (data_scaled_features.shape[0], 1, data_scaled_features.shape[1]))
y = {f'{var}_output': data_scaled_targets[var] for var in target_vars}

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# 初始化用于存储训练集和测试集目标的字典
y_train = {}
y_test = {}

# 分别为每个目标进行分割
for target in y:
    y_train[target], y_test[target] = train_test_split(y[target], test_size=0.2, random_state=42)


# 创建和训练模型
model = create_model(X_train, output_units=target_vars)
history = model.fit(X_train, y_train, epochs=64, batch_size=32, validation_split=0.1, verbose=2)

# 保存模型和标准化器
model.save('model_test/forecast_model.keras')
joblib.dump(scaler_features, 'scaler_test/scaler_features.pkl')
for var in target_vars:
    joblib.dump(scalers_target[var], f'scaler_test/scaler_{var}.pkl')

# 绘制结果
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
