import requests
import pandas as pd
# 替换以下变量的值为你的实际数据
userId = '712753224184J5Q1Y'
pwd = 'DeoJfLs'

# 设置其他参数
dataFormat = 'json'
interfaceId = 'getRadaFileByTimeRange'
dataCode = 'RADA_L3_MST_V3_CREF_PNG'
timeRange = '[20240407120000,20240408160000]'
staIds = '58367'
elements = 'Station_Id_C,DATETIME,FORMAT,FILE_NAME'
elements_list='Datetime,Hour,PRS,PRS_Max,PRS_Min,TEM,TEM_MAX,TEM_MIN,RHU,VAP,PRE_3h,WIN_S_MAX,WIN_D_S_Max'

# 构建完整的API URL
url = f"http://api.data.cma.cn:8090/api?userId={userId}&pwd={pwd}&dataFormat=json&interfaceId=getSurfEleByTimeRangeAndStaID&dataCode=SURF_CHN_MUL_HOR_3H&timeRange={timeRange}&staIDs={staIds}&elements=Station_Id_C,{elements_list}"

# 发起请求
response = requests.get(url)
if response.status_code == 200:
    try:
        data = response.json()
        df = pd.DataFrame(data['DS'])  # 假设数据在JSON的'DS'键下，根据实际结构调整
        # 保存DataFrame到Excel文件
        df.to_excel('data_station/weather_data.xlsx', index=False)  # 确保路径存在或者有权限写入
        print("文件已成功保存为 'data_station/weather_data.xlsx'")
    except ValueError:
        print("解析JSON时发生错误。原始响应内容是：")
        print(response.text)
else:
    print(f"请求失败，状态码：{response.status_code}")
    print("原始响应内容：")
    print(response.text)
