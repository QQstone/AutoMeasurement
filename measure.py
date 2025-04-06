import math
import queue

import cv2
import numpy as np
from scipy.interpolate import make_interp_spline


class Measure:
    def __init__(self, img, debug=False):
        # 比例尺
        self.scale = 1
        self.src_img = img
        self.fruit_bodies = []
        self.sperated_area = []
        self.ruler_contour = None
        self.ruler_box = []
        self.debug = debug

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
        self.scale = rule_width
        return self.src_img

    def build_roi(self, image):
        roi_list = []
        combine_image = np.zeros(image.shape, dtype='uint8')
        for fruit_body in self.fruit_bodies:
            contour, cX, cY = fruit_body
            # 计算最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 裁剪ROI
            roi = image[y:y + h, x:x + w]
            # 清除contour外的图像
            mask = np.zeros(roi.shape[:2], dtype='uint8')
            # 绘制轮廓到掩码上，轮廓内部填充白色
            # 轮廓偏移到[x, y]
            offset_contour = contour + np.array([[-x, -y]])

            # 确保偏移后的轮廓点仍然在ROI内
            offset_contour = offset_contour.clip(0, [w - 1, h - 1])
            cv2.drawContours(mask, [offset_contour], -1, (255), thickness=cv2.FILLED)
            # 使用掩码来清除contour外的图像部分，保留contour内的灰度图像
            roi = cv2.bitwise_and(roi, roi, mask=mask)
            roi_list.append([roi, x, y])
            #cv2.imwrite(f'./output/roi/{cX}_{cY}.jpg', roi)

            # ...
            # 将分割结果放回原图对应位置
            # 将分割结果放回原图对应位置
            combine_image[y:y + h, x:x + w] = roi
        #cv2.imwrite(f'./output/roi/COMBINE.jpg', combine_image)
        edge_image = self.find_edges(combine_image)
        if self.debug:
            cv2.imshow('Image with split contours', edge_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        for index in range(len(roi_list)):
            roi, x, y = roi_list[index]
            edge_roi = edge_image[y:y + roi.shape[0], x:x + roi.shape[1]]
            # cv2.imshow('Image with split contours', edge_roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cap_area, stipe_area = self.sperate_cap_by_circle(edge_roi, roi)
            combine_cap_area = self.axis_offset(cap_area, x, y)
            combine_stipe_area = self.axis_offset(stipe_area, x, y)
            self.sperated_area.append([combine_cap_area, combine_stipe_area])

    def axis_offset(self, positions, offset_x, offset_y):
        for i in range(len(positions)):
            positions[i][0] += offset_x  # x坐标加上offsetX
            positions[i][1] += offset_y  # y坐标加上offsetY
        return positions

    def find_edges(self, combie_image):
        # 转换为灰度图
        img = cv2.cvtColor(combie_image, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理，由于combie_image中非目标区域为黑色(0)，所以只需要简单的阈值处理
        _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        
        # 找到外轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建边缘图像
        edges = np.zeros_like(img)
        
        # 绘制所有外轮廓
        cv2.drawContours(edges, contours, -1, 255, 1)
        
        return edges



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
        if self.debug:
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
        if self.debug:
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

    def sperate_cap_by_circle(self, edge, roi):
        # 找到轮廓
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 选择最大的轮廓
        outer_contour = max(contours, key=cv2.contourArea)
        
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(outer_contour)
        
        # 创建掩码图像
        mask = np.zeros(edge.shape, dtype=np.uint8)
        cv2.drawContours(mask, [outer_contour], -1, 255, -1)
        
        # 使用形态学操作提取骨架
        skeleton = np.zeros_like(mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        
        while not done:
            eroded = cv2.erode(mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            mask = eroded.copy()
            
            if cv2.countNonZero(mask) == 0:
                done = True
        
        # 找到骨架的端点
        kernel = np.uint8([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])
        filtered = cv2.filter2D(skeleton, -1, kernel)
        endpoints = np.where(filtered == 11)
        endpoints = list(zip(endpoints[1], endpoints[0]))  # 转换为(x,y)格式
        
        if len(endpoints) >= 2:
            # 找到最远的两个端点
            max_dist = 0
            start_point = None
            end_point = None
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    dist = np.sqrt((endpoints[i][0] - endpoints[j][0])**2 + 
                                 (endpoints[i][1] - endpoints[j][1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        start_point = endpoints[i]
                        end_point = endpoints[j]
            
            # 计算主方向
            vx = end_point[0] - start_point[0]
            vy = end_point[1] - start_point[1]
            length = np.sqrt(vx*vx + vy*vy)
            if length > 0:
                vx /= length
                vy /= length
        else:
            # 如果找不到足够的端点，使用原来的方法
            vx, vy, x0, y0 = cv2.fitLine(outer_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # 将轮廓点转换为更易于处理的格式
        contour_points = outer_contour.reshape(-1, 2)
        
        # 计算轮廓上每个点到主方向线的距离
        distances = []
        for point in contour_points:
            # 点到直线的距离公式：|ax + by + c|/sqrt(a^2 + b^2)
            # 其中直线方程为：vy*x - vx*y + (vx*y0 - vy*x0) = 0
            if 'x0' in locals() and 'y0' in locals():
                a, b, c = vy, -vx, vx*y0 - vy*x0
            else:
                # 使用起点和终点计算直线方程
                a = vy
                b = -vx
                c = vx*start_point[1] - vy*start_point[0]
            
            dist = abs(a*point[0] + b*point[1] + c) / math.sqrt(a*a + b*b)
            distances.append(dist)
        
        # 找到距离最大的点，这通常是菌盖和菌柄的分界点
        max_dist_idx = np.argmax(distances)
        split_point = tuple(contour_points[max_dist_idx])
        
        # 在图像上标记分割点
        cv2.putText(roi, 'o', split_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # 如果debug模式开启，显示骨架
        if self.debug:
            debug_img = roi.copy()
            debug_img[skeleton > 0] = [255, 255, 255]  # 用白色显示骨架
            cv2.imshow('Skeleton', debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # 计算垂直方向
        perpendicular_slope = -vx/vy if vy != 0 else float('inf')
        
        # 通过分割点的垂直线
        if perpendicular_slope == float('inf'):
            # 垂直线的情况
            def point_side_of_line(point):
                return 1 if point[0] > split_point[0] else -1
        else:
            # 计算过分割点的垂直线方程: y - y1 = m(x - x1)
            intercept = split_point[1] - perpendicular_slope * split_point[0]
            def point_side_of_line(point):
                return 1 if point[1] > perpendicular_slope * point[0] + intercept else -1
        
        # 将轮廓点分为两组
        points_left = []
        points_right = []
        for point in contour_points:
            side = point_side_of_line(point)
            if side == 1:
                points_left.append(point)
            else:
                points_right.append(point)
        
        # 如果任一组为空，使用水平分割线作为备选
        if len(points_left) == 0 or len(points_right) == 0:
            points_left = []
            points_right = []
            for point in contour_points:
                if point[1] < split_point[1]:
                    points_left.append(point)
                else:
                    points_right.append(point)
        
        # 确保两组都有点
        if len(points_left) > 0 and len(points_right) > 0:
            # 计算凸包
            hull_left = cv2.convexHull(np.array(points_left))
            hull_right = cv2.convexHull(np.array(points_right))
            
            # 计算最小外接矩形
            bbox_left = cv2.boxPoints(cv2.minAreaRect(hull_left)).round().astype(int)
            bbox_right = cv2.boxPoints(cv2.minAreaRect(hull_right)).round().astype(int)
            
            # 在图像上绘制结果
            cv2.drawContours(roi, [bbox_left], -1, (255, 0, 0), thickness=2)
            cv2.drawContours(roi, [bbox_right], -1, (0, 0, 255), thickness=2)
            
            # 如果debug模式开启，显示边界框
            if self.debug:
                debug_img = roi.copy()

                cv2.imshow('sperate result', debug_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return bbox_left, bbox_right
        else:
            # 如果分割失败，返回整个轮廓的边界框作为两个部分
            bbox = cv2.boxPoints(cv2.minAreaRect(outer_contour)).round().astype(int)
            mid_idx = len(bbox) // 2
            return bbox[:mid_idx], bbox[mid_idx:]

    def calculate_cos_of_angle(self, x, y, cx, cy, centerx, centery):
        # 计算向量A和B
        vector_A = (cx - x, cy - y)
        vector_B = (centerx - x, centery - y)

        # 计算点积
        dot_product = vector_A[0] * vector_B[0] + vector_A[1] * vector_B[1]

        # 计算向量A和B的模长
        magnitude_A = math.sqrt(vector_A[0] ** 2 + vector_A[1] ** 2)
        magnitude_B = math.sqrt(vector_B[0] ** 2 + vector_B[1] ** 2)

        # 计算夹角的余弦值
        cos_theta = dot_product / (magnitude_A * magnitude_B)

        return cos_theta

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
        if self.debug:
            cv2.imshow('Image with split contours', self.src_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()