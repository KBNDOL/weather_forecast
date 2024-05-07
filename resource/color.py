import cv2
import numpy as np

# 加载图像
image = cv2.imread('downloads/ACCN/Z_RADA_C_BABJ_20240415160646_P_DOR_ACCN_CREF_20240415_160000.png')

# 读取颜色数据文件
color_ranges = []
with open('color_data.txt', 'r') as file:
    for line in file:
        parts = line.split()
        hue = int(float(parts[1])/2)
        saturation = int(float(parts[2]) * 255 / 100)  # 转换百分比为0-255
        value = int(float(parts[3]) * 255 / 100)  # 转换百分比为0-255
        # 假设色调的容差为2度，饱和度和亮度的容差为10%
        lower_bound = np.array([hue - 0, max(saturation - 25, 0), max(value - 25, 0)])
        upper_bound = np.array([hue + 1, min(saturation + 25, 255), min(value + 25, 255)])
        color_ranges.append((lower_bound, upper_bound))

# 初始化一个空掩模用于存储合并后的掩模
combined_mask = None

# 转换到HSV色彩空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 循环遍历颜色范围并组合掩模
for lower_bound, upper_bound in color_ranges:
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    if combined_mask is None:
        combined_mask = mask
    else:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

# 将掩模应用到原图像
result = cv2.bitwise_and(image, image, mask=combined_mask)

# 显示结果
cv2.imshow('Filtered Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('filtered_image.png', result)
