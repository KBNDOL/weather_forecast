import cv2


color_radar_image = cv2.imread('filtered_image.png')

gray_radar_image = cv2.cvtColor(color_radar_image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray Radar Image', gray_radar_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('gray_radar_image.png', gray_radar_image)
