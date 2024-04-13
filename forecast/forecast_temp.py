import numpy as np
import pandas as pd
from keras.models import load_model
import joblib

def predict_temperature_from_excel(excel_path):
    # 加载保存的模型和归一化对象
    temp_model = load_model('model/forecast_temp.keras')
    scaler_features = joblib.load('scaler/scaler_features.pkl')
    scaler_target_temp = joblib.load('scaler/scaler_target_temp.pkl')


    data = pd.read_excel(excel_path, engine='openpyxl')
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)

    if len(data) < 6:
        raise ValueError("The provided Excel file doesn't contain enough data.")


    hours_back = 6
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


    predicted_temp = temp_model.predict(X_new)
    predicted_temp = scaler_target_temp.inverse_transform(predicted_temp).flatten()

    return predicted_temp[-1]
