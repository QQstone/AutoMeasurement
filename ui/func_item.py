import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton
import cv2
from flags import *


class MyItem(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem, self).__init__(name, parent=parent)
        # self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(80, 60))  # size

    def get_params(self):
        protected = [v for v in dir(self) if v.startswith('_') and not v.startswith('__')]
        param = {}
        for v in protected:
            param[v.replace('_', '', 1)] = self.__getattribute__(v)
        return param

    def update_params(self, param):
        for k, v in param.items():
            if '_' + k in dir(self):
                self.__setattr__('_' + k, int(v))


class GrayingItem(MyItem):
    def __init__(self, parent=None):
        super(GrayingItem, self).__init__(' 灰度化 ', parent=parent)

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class FilterItem(MyItem):

    def __init__(self, parent=None):
        super().__init__('平滑处理', parent=parent)
        self._ksize = 3
        self._kind = MEDIAN_FILTER
        self._sigmax = 0

    def __call__(self, img):
        if self._kind == MEAN_FILTER:
            img = cv2.blur(img, (self._ksize, self._ksize))
        elif self._kind == GAUSSIAN_FILTER:
            img = cv2.GaussianBlur(img, (self._ksize, self._ksize), self._sigmax)
        elif self._kind == MEDIAN_FILTER:
            img = cv2.medianBlur(img, self._ksize)
        return img


class MorphItem(MyItem):
    def __init__(self, parent=None):
        super().__init__(' 形态学 ', parent=parent)
        self._ksize = 3
        self._op = ERODE_MORPH_OP
        self._kshape = RECT_MORPH_SHAPE

    def __call__(self, img):
        op = MORPH_OP[self._op]
        kshape = MORPH_SHAPE[self._kshape]
        kernal = cv2.getStructuringElement(kshape, (self._ksize, self._ksize))
        img = cv2.morphologyEx(img, self._op, kernal)
        return img


class GradItem(MyItem):

    def __init__(self, parent=None):
        super().__init__('图像梯度', parent=parent)
        self._kind = SOBEL_GRAD
        self._ksize = 3
        self._dx = 1
        self._dy = 0

    def __call__(self, img):
        if self._dx == 0 and self._dy == 0 and self._kind != LAPLACIAN_GRAD:
            self.setBackground(QColor(255, 0, 0))
            self.setText('图像梯度 （无效: dx与dy不同时为0）')
        else:
            self.setBackground(QColor(200, 200, 200))
            self.setText('图像梯度')
            if self._kind == SOBEL_GRAD:
                img = cv2.Sobel(img, -1, self._dx, self._dy, self._ksize)
            elif self._kind == SCHARR_GRAD:
                img = cv2.Scharr(img, -1, self._dx, self._dy)
            elif self._kind == LAPLACIAN_GRAD:
                img = cv2.Laplacian(img, -1)
        return img


class ThresholdItem(MyItem):
    def __init__(self, parent=None):
        super().__init__('阈值处理', parent=parent)
        self._thresh = 127
        self._maxval = 255
        self._method = BINARY_THRESH_METHOD

    def __call__(self, img):
        method = THRESH_METHOD[self._method]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, self._thresh, self._maxval, method)[1]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class EdgeItem(MyItem):
    def __init__(self, parent=None):
        super(EdgeItem, self).__init__('边缘检测', parent=parent)
        self._thresh1 = 20
        self._thresh2 = 100

    def __call__(self, img):
        img = cv2.Canny(img, threshold1=self._thresh1, threshold2=self._thresh2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class ContourItem(MyItem):
    def __init__(self, parent=None):
        super(ContourItem, self).__init__('轮廓检测', parent=parent)
        self._mode = TREE_CONTOUR_MODE
        self._method = SIMPLE_CONTOUR_METHOD
        self._bbox = NORMAL_CONTOUR

    def __call__(self, img):
        mode = CONTOUR_MODE[self._mode]
        method = CONTOUR_METHOD[self._method]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnts, _ = cv2.findContours(img, mode, method)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self._bbox == RECT_CONTOUR:
            bboxs = [cv2.boundingRect(cnt) for cnt in cnts]
            print(bboxs)
            for x, y, w, h in bboxs:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        elif self._bbox == MINRECT_CONTOUR:
            bboxs = [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))) for cnt in cnts]
            img = cv2.drawContours(img, bboxs, -1, (255, 0, 0), thickness=2)
        elif self._bbox == MINCIRCLE_CONTOUR:
            circles = [cv2.minEnclosingCircle(cnt) for cnt in cnts]
            print(circles)
            for (x, y), r in circles:
                img = cv2.circle(img, (int(x), int(y)), int(r), (255, 0, 0), thickness=2)
        elif self._bbox == NORMAL_CONTOUR:
            img = cv2.drawContours(img, cnts, -1, (255, 0, 0), thickness=2)

        return img


class EqualizeItem(MyItem):
    def __init__(self, parent=None):
        super().__init__(' 均衡化 ', parent=parent)
        self._blue = True
        self._green = True
        self._red = True

    def __call__(self, img):
        b, g, r = cv2.split(img)
        if self._blue:
            b = cv2.equalizeHist(b)
        if self._green:
            g = cv2.equalizeHist(g)
        if self._red:
            r = cv2.equalizeHist(r)
        return cv2.merge((b, g, r))


class HoughLineItem(MyItem):
    def __init__(self, parent=None):
        super(HoughLineItem, self).__init__('直线检测', parent=parent)
        self._rho = 1
        self._theta = np.pi / 180
        self._thresh = 10
        self._min_length = 20
        self._max_gap = 5

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(img, self._rho, self._theta, self._thresh, minLineLength=self._min_length,
                                maxLineGap=self._max_gap)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if lines is None: return img
        for line in lines:
            for x1, y1, x2, y2 in line:
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        return img


class LightItem(MyItem):
    def __init__(self, parent=None):
        super(LightItem, self).__init__('亮度调节', parent=parent)
        self._alpha = 1
        self._beta = 0

    def __call__(self, img):
        blank = np.zeros(img.shape, img.dtype)
        img = cv2.addWeighted(img, self._alpha, blank, 1 - self._alpha, self._beta)
        return img


class GammaItem(MyItem):
    def __init__(self, parent=None):
        super(GammaItem, self).__init__('伽马校正', parent=parent)
        self._gamma = 1

    def __call__(self, img):
        gamma_table = [np.power(x / 255.0, self._gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)


class FrequencyFilterItem(MyItem):
    def __init__(self, parent=None):
        super().__init__('频域滤波', parent=parent)
        self._filter_type = 0  # 0: 低通, 1: 高通, 2: 带通
        self._radius = 30      # 截止频率/半径
        self._width = 10       # 带通滤波的带宽
        self._order = 2        # Butterworth滤波器阶数
    
    def create_butterworth_filter(self, shape, radius, order, highpass=False):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        
        # 使用numpy的meshgrid替代ogrid，提高网格计算效率
        x = np.arange(cols) - center_col
        y = np.arange(rows) - center_row
        X, Y = np.meshgrid(x, y)
        d = np.sqrt(X*X + Y*Y)
        
        # Butterworth滤波器
        if highpass:
            h = 1 / (1 + (radius / (d + 1e-6)) ** (2*order))
        else:
            h = 1 / (1 + (d / radius) ** (2*order))
            
        return h
    
    def create_band_filter(self, shape, center_radius, width, order):
        rows, cols = shape
        center_row, center_col = rows // 2, cols // 2
        
        # 使用numpy的meshgrid
        x = np.arange(cols) - center_col
        y = np.arange(rows) - center_row
        X, Y = np.meshgrid(x, y)
        d = np.sqrt(X*X + Y*Y)
        
        # 带通滤波器 = 低通 - 高通
        h_low = 1 / (1 + (d / (center_radius + width/2)) ** (2*order))
        h_high = 1 / (1 + ((center_radius - width/2) / (d + 1e-6)) ** (2*order))
        
        return h_low - h_high
    
    def apply_frequency_filter(self, img, h):
        # 转换为浮点数
        img_float = img.astype(np.float32)
        
        # 对每个通道进行FFT和滤波
        channels = []
        for channel in cv2.split(img_float):
            # 使用numpy的FFT，比cv2.dft更快
            # 添加填充以减少频谱泄漏
            padded = np.pad(channel, ((0, 0), (0, 0)), mode='reflect')
            
            # 快速傅里叶变换
            f = np.fft.fft2(padded)
            fshift = np.fft.fftshift(f)
            
            # 应用滤波器
            filtered = fshift * h
            
            # 逆变换
            f_ishift = np.fft.ifftshift(filtered)
            img_back = np.real(np.fft.ifft2(f_ishift))
            
            # 去除填充
            img_back = img_back[:channel.shape[0], :channel.shape[1]]
            
            # 归一化到0-255范围
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            channels.append(img_back)
        
        return cv2.merge(channels)
    
    def optimize_fft_size(self, size):
        """计算最优的FFT大小"""
        return 2 ** np.ceil(np.log2(size)).astype(int)
    
    def __call__(self, img):
        # 获取最优FFT尺寸
        rows, cols = img.shape[:2]
        optimal_rows = self.optimize_fft_size(rows)
        optimal_cols = self.optimize_fft_size(cols)
        
        # 使用反射填充而不是零填充，减少边缘效应
        padded = cv2.copyMakeBorder(img, 
                                   0, optimal_rows - rows, 
                                   0, optimal_cols - cols,
                                   cv2.BORDER_REFLECT)
        
        # 创建滤波器
        if self._filter_type == 0:  # 低通
            h = self.create_butterworth_filter(padded.shape[:2], 
                                            self._radius, 
                                            self._order, 
                                            False)
        elif self._filter_type == 1:  # 高通
            h = self.create_butterworth_filter(padded.shape[:2], 
                                            self._radius, 
                                            self._order, 
                                            True)
        else:  # 带通
            h = self.create_band_filter(padded.shape[:2], 
                                      self._radius, 
                                      self._width, 
                                      self._order)
        
        # 应用滤波器
        filtered = self.apply_frequency_filter(padded, h)
        
        # 裁剪回原始尺寸
        filtered = filtered[:rows, :cols]
        
        return filtered

    def visualize_spectrum(self, img):
        """可视化频谱（用于调试）"""
        img_float = img.astype(np.float32)
        channels = []
        
        for channel in cv2.split(img_float):
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            magnitude_spectrum = np.clip(magnitude_spectrum, 0, 255).astype(np.uint8)
            channels.append(magnitude_spectrum)
            
        return cv2.merge(channels)


func_items = [
    GrayingItem,
    FilterItem,
    EqualizeItem,
    MorphItem,
    GradItem,
    ThresholdItem,
    EdgeItem,
    ContourItem,
    HoughLineItem,
    LightItem,
    GammaItem,
    FrequencyFilterItem
]