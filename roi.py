import cv2
import numpy as np

# 载入图像并进行预处理
image = cv2.imread('output/single.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 提取轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓
max_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(image, [max_contour], -1, (255, 0, 0), 2)
# 进行多边形拟合
epsilon = 0.01 * cv2.arcLength(max_contour, True)  # 设置适当的epsilon值
approx = cv2.approxPolyDP(max_contour, epsilon, True)

# 计算多边形的外接矩形，并获取宽度作为菌盖直径
x, y, w, h = cv2.boundingRect(approx)
diameter = w

print("菌盖直径：", diameter)

# 可视化结果
cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()