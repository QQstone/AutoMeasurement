# 拟合弓形
import cv2

# 加载图像并转换为灰度图像
img =  cv2.imread('output/single.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测算法找到轮廓
edges = cv2.Canny(gray, 100, 200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓，并计算其外接矩形
largest_contour = max(contours, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(largest_contour)

# 确定目标上最清晰的分界线，将目标分为两部分
split_line_y = y + h // 2

# 计算菌盖宽度
cap_region = gray[y:split_line_y, x:x+w]
_, cap_thresh = cv2.threshold(cap_region, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cap_contours, _ = cv2.findContours(cap_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cap_contour = max(cap_contours, key=cv2.contourArea)
(x_cap, y_cap, w_cap, h_cap) = cv2.boundingRect(cap_contour)
cap_width = w_cap

# 输出结果
print("Cap width: ", cap_width)

# 在原始图像中绘制结果
cv2.drawContours(img, [largest_contour], -1, (0,255,0), 2)
cv2.line(img, (x, split_line_y), (x+w, split_line_y), (0, 0, 255), 2)
cv2.rectangle(img, (x_cap, y_cap), (x_cap+w_cap, y_cap+h_cap), (255,0,0), 2)
cv2.imshow('Result', img)
cv2.waitKey(0)