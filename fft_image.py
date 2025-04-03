import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft_image(image_path):
    # 读取图像并转换为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # 获取图像尺寸
    rows, cols = img.shape
    
    # 计算最优DFT尺寸
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    
    # 扩展图像到最优尺寸
    padded = cv2.copyMakeBorder(img, 0, nrows-rows, 0, ncols-cols, cv2.BORDER_CONSTANT, value=0)
    
    # 执行FFT
    dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # 计算幅度谱
    magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    
    # 显示结果
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) != 2:
    #     print("Usage: python fft_image.py <image_path>")
    #     sys.exit(1)
    
    fft_image('resource/img1.jpg') 