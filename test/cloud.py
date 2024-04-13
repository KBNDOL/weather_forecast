import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# 假设data是包含上述数据的DataFrame
files = ['D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_04_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_05_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_06_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_07_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_08_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_09_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_10_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_11_hour.xlsx',
         'D:/pythonProjectTotal/weather_forecast/data/2023/weather_data_2023_12_hour.xlsx']
dataframes = [pd.read_excel(file, engine='openpyxl') for file in files]
data = pd.concat(dataframes, ignore_index=True)

scaler = MinMaxScaler()
data['wind_dir_cos'] = np.cos(np.radians(data['wdir']))
data['wind_dir_sin'] = np.sin(np.radians(data['wdir']))
features = ['temp', 'rhum', 'wspd','coco','wind_dir_sin','wind_dir_cos']
data[features] = scaler.fit_transform(data[features])
# 以时间窗口构建特征
def create_dataset(X, look_back=1):
    Xs = []
    for i in range(len(X)-look_back):
        v = X.iloc[i:(i+look_back)].values
        Xs.append(v)
    return np.array(Xs)

look_back = 3  # 使用3个小时的数据作为输入
X = create_dataset(data[features], look_back)

kmeans = KMeans(n_clusters=3, random_state=0).fit(X.reshape(X.shape[0], -1))

# 查看聚类结果
clusters = kmeans.labels_

temperature_data = data['temp'].iloc[look_back:]  # 调整偏移

# 为了简化展示，我们只展示前100个数据点
sample_size = 200
fig, ax = plt.subplots(figsize=(14, 7))

# 绘制温度变化，并根据聚类结果上色
for i in range(sample_size):
    ax.plot(i, temperature_data.iloc[i], 'o', color=['red', 'green', 'blue'][clusters[i]])

ax.set_title('Temperature Clusters')
ax.set_xlabel('Time')
ax.set_ylabel('Temperature (normalized)')

# 创建一个自定义图例
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='blue', lw=4)]

ax.legend(custom_lines, ['Cluster 1', 'Cluster 2', 'Cluster 3'])

plt.show()