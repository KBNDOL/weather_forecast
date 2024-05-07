import cv2
import numpy as np

# 加载图像和掩码（假设这些已经被加载和创建）
image = cv2.imread('downloads/AECN/Z_RADA_C_BABJ_20240415160647_P_DOR_AECN_CREF_20240415_160000.png')
mask = cv2.imread('filtered_black_image.png', cv2.IMREAD_GRAYSCALE)
masked_image = cv2.bitwise_and(image, image, mask=mask)

# 将非黑色像素转换为白色
white_edges = np.where(masked_image.any(axis=-1, keepdims=True), 255, 0).astype(np.uint8)


# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.imshow('White Edges', white_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('mask_image.png', white_edges)