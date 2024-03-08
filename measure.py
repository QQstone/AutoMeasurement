import queue

import cv2
import numpy as np


class Measure:
    def __init__(self, img):
        # 比例尺
        self.scale = 1
        self.src_img = img
        self.fruit_bodies = []
        self.ruler_contour = None
        self.ruler_box = []

    def find_targets(self):
        # 查找轮廓
        img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 轮廓过滤和编号
        self.fruit_bodies = []
        self.ruler_contour = None
        rule_index = 0
        max_y_index = -1

        for index, contour in enumerate(contours):
            # 根据面积过滤轮廓
            if cv2.contourArea(contour) > 100:  # 假设面积阈值为100
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                    # 找到形心最靠下方的轮廓
                if cY > max_y_index:
                    max_y_index = cY
                    self.ruler_contour = contour
                    rule_index = index

                self.fruit_bodies.append([contour, cX, cY])

        # list 选择排序
        del self.fruit_bodies[rule_index]
        self.fruit_bodies.sort(key=lambda item: item[1], reverse=False)
        # # 在形心处标记编号
        for index, item in enumerate(self.fruit_bodies):
            cv2.putText(self.src_img, str(index+1), (item[1], item[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # 计算rule轮廓的宽度
        rect = cv2.minAreaRect(self.ruler_contour)
        box = cv2.boxPoints(rect)  # 获取矩形的四个角点
        box = np.int0(box)  # 将坐标转换为整数
        # 平方根
        rule_width = int(np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2))
        # 在图像上绘制矩形
        cv2.drawContours(self.src_img, [box], 0, (0, 255, 0), 2)
        cv2.putText(self.src_img, f'Width: {rule_width}', (box[2][0], box[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return self.src_img

    def sperate_cap_by_curvature(self):
        img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
        img = cv2.blur(img, (5, 5))
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 选择最大的轮廓
        contour = max(contours, key=cv2.contourArea)
        # 采样
        sample_size = 100
        interval = max(1, len(contour) // sample_size)
        contour = contour[::interval]
        # 计算曲率，即角度变化除以相邻点之间的距离
        curvatures = self.getCurvature(contour)

        # for index in range(len(curvatures)):
        #     if index%10 ==0:
        #         text = "{:.2f}".format(curvatures[index]*100)
        #         cv2.putText(src_img, text, (contour[index][0][0], contour[index][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                 (255, 0, 0), 1)
        # cv2.imshow('Image with split contours', src_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 找到曲率绝对值最大两个负曲率对应的点
        #max_neg = 0
        max_pos = 0
        corners_index = queue.deque(maxlen=2)
        for index in range(len(curvatures)):
            val = curvatures[index]
            if val > 0 and val > max_pos:
                corners_index.append(index)
                max_pos = val
            # if val < 0 and val < max_neg:
            #     corners_index.append(index)
            #     max_neg = val

        if len(corners_index) < 2:
            #max_neg = 0
            max_pos = 0
            for index in range(corners_index[0], len(curvatures)-1):
                val = curvatures[index]
                if val > 0 and val > max_pos:
                    corners_index.append(index)
                    max_pos = val
                # if val < 0 and val < max_neg:
                #     corners_index.append(index)
                #     max_neg = val

        index_list = list(corners_index)
        # contour被分割成两个部分
        cv2.putText(img, 'o', (contour[index_list[0]][0][0], contour[index_list[0]][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)
        cv2.putText(img, 'o', (contour[index_list[1]][0][0], contour[index_list[1]][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)
        cv2.imshow('Image with split contours', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        index_list.sort()
        part1 = np.concatenate((contour[index_list[1]:], contour[:index_list[0]], contour[index_list[1]:index_list[1]]), axis=0)
        part2 = np.concatenate((contour[index_list[0]:index_list[1]], contour[index_list[0]:index_list[0]]), axis=0)

        image_copy = img.copy()  # 复制图像，避免在原图上绘制
        cv2.drawContours(image_copy, [part1], -1, (0, 255, 0), 2)  # 第一部分用绿色绘制
        cv2.drawContours(image_copy, [part2], -1, (0, 0, 255), 2)  # 第二部分用红色绘制

        cv2.imshow('Image with split contours', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getCurvature(self, contour):
        # 计算轮廓上的点的曲率
        curvatures = []
        gap = 1
        for index in range(len(contour)):
            point = contour[index]
            x, y = point[0]
            if index < gap:
                prev_point = contour[len(contour) - gap + index]
                next_point = contour[index + gap]

            if index >= len(contour) - gap:
                prev_point = contour[index - gap]
                next_point = contour[index - len(contour) + gap]

            if index > gap - 1 and index < len(contour) - gap:
                prev_point = contour[index - gap]
                next_point = contour[index + gap]

            # 三点线段长度
            OA = np.sqrt((x - prev_point[0][0]) ** 2 + (y - prev_point[0][1]) ** 2)
            OC = np.sqrt((x - next_point[0][0]) ** 2 + (y - next_point[0][1]) ** 2)
            AC = np.sqrt((prev_point[0][0] - next_point[0][0]) ** 2 + (prev_point[0][1] - next_point[0][1]) ** 2)

            # cosθ = (OA^2+OC^2-AC^2)/(2*OA*OC)
            cos = (OA ** 2 + OC ** 2 - AC ** 2) / (2 * OA * OC)

            # κ = 1/r = 2*sinθ/AC
            courvature = 2 * np.sqrt(1 - cos ** 2) / AC

            # d = OA·AC  (x - x_1)(y_3 - y_1) - (y - y_1)(x_3 - x_1)
            d = (x - prev_point[0][0]) * (next_point[0][1] - prev_point[0][1]) - (y - prev_point[0][1]) * (next_point[0][0] - prev_point[0][0])

            if d < 0:
                curvatures.append(-1 * courvature)
            else:
                curvatures.append(courvature)

        return curvatures

    #def sperate_cap_by_polygon(self):
