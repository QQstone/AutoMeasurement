import queue

import cv2
import numpy as np
from scipy.interpolate import make_interp_spline


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
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 选择最大的轮廓
        contour = max(contours, key=cv2.contourArea)

        # 计算轮廓的最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        mbr_rect = (x, y, x + w, y + h)

        # 从原图像中裁剪出最小外接矩形区域
        mbr_image = img[y:y + h, x:x + w]

        # 对裁剪出的图像进行中值滤波
        # 注意：中值滤波的ksize必须是大于1的奇数
        blur_core_size = 2 * (len(contour) // 40) + 1
        filtered_mbr_image = cv2.medianBlur(mbr_image, blur_core_size)

        contours, _ = cv2.findContours(filtered_mbr_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothed_contour = max(contours, key=cv2.contourArea)

        # 计算曲率，即角度变化除以相邻点之间的距离
        # 取样本80
        sample_size = 80

        # 计算步长，确保样本容量不会超过数组长度
        step_size = max(1, len(smoothed_contour) // sample_size)
        simpled_contour = smoothed_contour[::step_size]

        #在图上标记点的序号
        for index, point in enumerate(simpled_contour):
            cv2.putText(self.src_img, str(index), (point[0][0], point[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.drawContours(self.src_img, [smoothed_contour], -1, (0, 255, 0), 2)  # 第一部分用绿色绘制
        cv2.imshow('Image with split contours', self.src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        curvatures = self.getCurvature(simpled_contour)
        #curvatures = self.get_float_curvatures(smoothed_contour, 4)

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
            for index in range(corners_index[0] + 1, len(curvatures)-1):
                val = curvatures[index]
                if val > 0 and val > max_pos:
                    corners_index.append(index)
                    max_pos = val
                # if val < 0 and val < max_neg:
                #     corners_index.append(index)
                #     max_neg = val

        index_list = list(corners_index)
        index_list = [item * step_size for item in index_list]
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

    def getCurvature(self, contour, gap=1):
        # 计算轮廓上的点的曲率
        curvatures = []
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

    def get_float_curvatures(self, contour, gap):
        """
        计算轮廓的浮点数曲率。 将像素坐标转换为浮点坐标
        插值数量为 gap 轮廓总点数 变为len(contour) * (gap + 1)
        参数:
        contour - 轮廓的列表，表示一个封闭的曲线。
        gap - 切点之间间隔点数。

        返回值:
        返回一个浮点数列表，包含轮廓上每个点的曲率。
        """
        contour_len = len(contour)
        contour_float = contour.astype(np.float32)
        # 获取轮廓点的x和y坐标
        x = contour_float[:, 0, 0]
        y = contour_float[:, 0, 1]

        # 创建新的x和y坐标数组，用于存储插值后的点
        new_x = np.empty((len(x) * (gap + 1)))
        new_y = np.empty_like(new_x)

        # 使用样条插值创建新的点
        for i in range(len(x) - 1):
            x_spline = np.linspace(x[i], x[i + 1], gap, endpoint=False)  # 创建等距参数值
            spl = make_interp_spline(x, y)
            y_spline = spl(x_spline)

            new_x[i*gap] = x[i]
            new_y[i*gap] = y[i]
            new_x[i*gap+1: i*gap+gap+1] = x_spline
            new_y[i*gap+1: i*gap+gap+1] = y_spline

        # 处理最后一个点（如果需要循环回到起点）
        if x[0] != x[-1] or y[0] != y[-1]:
            x_spline = np.linspace(x[-1], x[0], gap, endpoint=False)  # 最后一个段不需要额外的点
            y_spline = make_interp_spline(x, y)(x_spline)
            new_x[-1*gap:] = x_spline
            new_y[-1*gap:] = y_spline

        # 将新的x和y坐标组合成新的轮廓点
        interpolated_contour = np.column_stack((new_x, new_y)).astype(np.int32)

        return self.getCurvature(interpolated_contour, gap)

def mark_curvatures(self, contour, curvatures):
        """
        在轮廓上标记曲率值。

        参数:
        contour - 轮廓的列表，表示一个封闭的曲线。
        curvatures - 一个浮点数列表，包含轮廓上每个点的曲率。

        返回值:
        无返回值，此函数主要用于在轮廓图上标注出各点的曲率值。
        """
        # 函数体实现标记轮廓上各点曲率值的逻辑
        for index in range(len(curvatures)):
            if index%10 ==0:
                text = "{:.2f}".format(curvatures[index]*100)
                cv2.putText(self.src_img, text, (contour[index][0][0], contour[index][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1)
        cv2.imshow('Image with split contours', self.src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()