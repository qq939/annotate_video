import cv2
import numpy as np

# 读取帧12的图片
img = cv2.imread('temp_data/frames/frame_000012.jpg')

if img is None:
    print('无法读取图片!')
else:
    print(f'帧12图片尺寸: {img.shape}')
    print(f'图片尺寸: {img.shape[1]}x{img.shape[0]}')
    
    # 检查点(437,552)处的像素
    y, x = 552, 437
    if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
        pixel = img[y, x]
        print(f'点({x},{y})的像素值: {pixel}')
        
        # 检查该点附近是否有非黑色像素
        for dy in range(-50, 51, 10):
            for dx in range(-50, 51, 10):
                ny, nx = y + dy, x + dx
                if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                    pixel = img[ny, nx]
                    if pixel.sum() > 30:
                        print(f'点({nx},{ny}): {pixel}')
