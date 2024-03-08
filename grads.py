import cv2
import numpy as np

# 加载图像并转换为灰度图像
img = cv2.imread('resource/open_process_to_noise.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测算法找到轮廓
edges = cv2.Canny(gray, 100, 200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓，并计算其外接矩形
largest_contour = max(contours, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(largest_contour)

# 计算轮廓的梯度
gradient_x = cv2.Sobel(gray[y:y+h, x:x+w], cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(gray[y:y+h, x:x+w], cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值和方向
gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
gradient_direction = np.arctan2(gradient_y, gradient_x)

# 设置阈值，将梯度幅值较大的点作为边缘点
threshold = 100
edge_mask = np.zeros_like(edges)
edge_mask[y:y+h, x:x+w][gradient_magnitude > threshold] = 255

# 查找目标上最清晰的分界线
split_line_y = None
for y in range(y+h-1, y, -1):
    if edge_mask[y, x:x+w].sum() == 0:
        split_line_y = y
        break

# 绘制分界线
if split_line_y is not None:
    cv2.line(img, (x, split_line_y), (x+w, split_line_y), (0, 0, 255), 2)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
else:
    print("未找到分界线")