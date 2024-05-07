import requests
import re
import os


userId = '712744592809jfZtK'
pwd = 'utNN8xi'
timeRange = '[20240415160000,20240415160600]'
dataFormat = 'json'
interfaceId = 'getRadaFileByTimeRange'
dataCode = 'RADA_L3_MST_V3_CREF_PNG'


url = f"http://api.data.cma.cn:8090/api?userId={userId}&pwd={pwd}&dataFormat={dataFormat}&interfaceId={interfaceId}&dataCode={dataCode}&timeRange={timeRange}&elements=Station_Id_C,DATETIME,FORMAT,FILE_NAME"

response = requests.get(url)
if response.status_code == 200:
    try:
        print(response.json())
    except ValueError:
        print("解析JSON时发生错误。原始响应内容是：")
        print(response.text)
else:
    print(f"请求失败，状态码：{response.status_code}")
    print("原始响应内容：")
    print(response.text)

urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response.text)

base_directory = "downloads"

os.makedirs(base_directory, exist_ok=True)

for url in urls:
    filename = url.split('?')[0].split('/')[-1]

    filename = filename.replace("%", "_").replace("=", "_").replace("&", "_")

    if "ACCN" in filename:
        subdirectory = os.path.join(base_directory, "ACCN")
    elif "ACHN" in filename:
        subdirectory = os.path.join(base_directory, "ACHN")
    elif"AECN" in filename:
        subdirectory = os.path.join(base_directory, "AECN")
    elif "ANCN" in filename:
        subdirectory = os.path.join(base_directory, "ANCN")
    elif "ANEC" in filename:
        subdirectory = os.path.join(base_directory, "ANEC")
    elif "ANWC" in filename:
        subdirectory = os.path.join(base_directory, "ANWC")
    elif "ASCN" in filename:
        subdirectory = os.path.join(base_directory, "ASCN")
    elif "ASWC" in filename:
        subdirectory = os.path.join(base_directory, "ASWC")
    elif "CJRB" in filename:
        subdirectory = os.path.join(base_directory, "CJRB")
    elif "HHRB" in filename:
        subdirectory = os.path.join(base_directory, "HHRB")
    elif "HURB" in filename:
        subdirectory = os.path.join(base_directory, "HURB")
    elif "LORB" in filename:
        subdirectory = os.path.join(base_directory, "LORB")
    elif "SHRB" in filename:
        subdirectory = os.path.join(base_directory, "SHRB")
    elif "YLRB" in filename:
        subdirectory = os.path.join(base_directory, "YLRB")
    elif "ZHRB" in filename:
        subdirectory = os.path.join(base_directory, "ZHRB")
    else:
        subdirectory = os.path.join(base_directory, "Others")


    os.makedirs(subdirectory, exist_ok=True)


    filepath = os.path.join(subdirectory, filename)

    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"文件已保存至：{filepath}")
        else:
            print(f"下载失败，状态码：{response.status_code}")
    except Exception as e:
        print(f"处理URL时发生错误：{url}，错误信息：{e}")