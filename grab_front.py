
import cv2
import numpy as np

# 载入原图像和背景图像
def grab_front(img):

    # 预处理
    # ...

    # 分割前景与背景
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, img.shape[1], img.shape[0])  # 指定矩形初始区域
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 提取前景
    foreground_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = img * foreground_mask[:, :, np.newaxis]

    # 后处理
    # ...
    
    # 显示结果
    return foreground
