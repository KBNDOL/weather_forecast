import cv2
import numpy as np


def extract_reflectivity(radar_image, mask_image):
    _, mask = cv2.threshold(mask_image, thresh=1, maxval=255, type=cv2.THRESH_BINARY)

    extracted_data = cv2.bitwise_and(radar_image, radar_image, mask=mask)
    return extracted_data


def calculate_statistics(extracted_data):
    non_zero_values = extracted_data[extracted_data > 0]

    if non_zero_values.size == 0:
        return None, None, None
    mean_val = np.mean(non_zero_values)
    max_val = np.max(non_zero_values)
    min_val = np.min(non_zero_values)

    return mean_val, max_val, min_val


radar_image = cv2.imread('filtered_image.png', cv2.IMREAD_GRAYSCALE)
mask_image = cv2.imread('filtered_balck_image.png', cv2.IMREAD_GRAYSCALE)

extracted_data = extract_reflectivity(radar_image, mask_image)

mean_reflectivity, max_reflectivity, min_reflectivity = calculate_statistics(extracted_data)

if mean_reflectivity is None:
    print("没有有效的反射率数据可用于统计。")
else:
    print(f"Mean Reflectivity: {mean_reflectivity}")
    print(f"Max Reflectivity: {max_reflectivity}")
    print(f"Min Reflectivity: {min_reflectivity}")
