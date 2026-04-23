import sys
sys.path.insert(0, '/Users/jimjiang/Downloads/biaozhu')

from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

import tempfile

# 创建临时文件
output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

from post_annotate import PostAnnotatorWindow

window = PostAnnotatorWindow(output_path)

print(f'labels_dir: {window.labels_dir}')
print(f'frames_dir: {window.frames_dir}')
print()

# 读取第12帧的数据
frame, annotations = window.load_frame_data(12)

print(f'帧12共加载了{len(annotations)}个标注')

# 用户点击的坐标
click_x, click_y = 437, 552

import cv2
import numpy as np

print(f'用户点击点: ({click_x}, {click_y})')
print()

found = False
for ann in annotations:
    track_id = ann.get('track_id', ann['id'])
    bbox = ann['bbox']
    polygon = ann['segmentation'][0]
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    
    result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
    in_poly = result >= 0
    
    if track_id == 0 or in_poly:
        print(f'track_id={track_id}: 包含={in_poly}, 结果={result:.2f}')
        print(f'  边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}')
        if track_id == 0:
            print('  ← 用户点击的标注!')
        print()
        
    if in_poly:
        found = True

if not found:
    print('⚠️ 该点不在任何mask内！')

import os
os.unlink(output_path)
app.quit()
