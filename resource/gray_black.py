import cv2
import numpy as np

image = cv2.imread('filtered_balck_image.jpg')

# 将图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lower_gray = 10
upper_gray = 115

mask = cv2.inRange(gray_image, lower_gray, upper_gray)

cv2.imwrite('gray_region_mask.jpg', mask)
