import sys
sys.path.insert(0, '/Users/jimjiang/Downloads/biaozhu')

from PyQt5.QtWidgets import QApplication
app = QApplication(sys.argv)

import tempfile
output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

from post_annotate import PostAnnotatorWindow

window = PostAnnotatorWindow(output_path)

# 加载第12帧
frame, annotations = window.load_frame_data(12)

print(f'加载了{len(annotations)}个标注')
print()

# 渲染
annotated_frame = window.apply_threshold_to_masks(frame, annotations.copy(), window.conf_threshold)

import cv2
import numpy as np

# 保存渲染后的图片
cv2.imwrite('/tmp/frame_12_annotated.jpg', annotated_frame)
print('已保存渲染后的图片到 /tmp/frame_12_annotated.jpg')

# 检查点(437,552)处是否有mask
y, x = 552, 437
if 0 <= y < annotated_frame.shape[0] and 0 <= x < annotated_frame.shape[1]:
    pixel_original = frame[y, x]
    pixel_annotated = annotated_frame[y, x]
    print(f'点({x},{y})像素:')
    print(f'  原图: {pixel_original}')
    print(f'  渲染后: {pixel_annotated}')
    
    diff = np.abs(pixel_annotated.astype(int) - pixel_original.astype(int)).sum()
    if diff > 30:
        print(f'  该点有mask叠加! 差异={diff}')
    else:
        print(f'  该点无mask叠加')

import os
os.unlink(output_path)
app.quit()
