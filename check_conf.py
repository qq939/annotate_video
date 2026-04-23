import sys
sys.path.insert(0, '/Users/jimjiang/Downloads/biaozhu')

from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

import tempfile

output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

from post_annotate import PostAnnotatorWindow

window = PostAnnotatorWindow(output_path)

# 读取第12帧的数据
frame, annotations = window.load_frame_data(12)

print(f'当前置信度阈值: {window.conf_threshold}')
print()

# 用户点击的坐标
click_x, click_y = 437, 552

import cv2
import numpy as np

print(f'用户点击点: ({click_x}, {click_y})')
print()

# 检查track_id=0的标注
for ann in annotations:
    track_id = ann.get('track_id', ann['id'])
    if track_id == 0:
        conf = ann.get('confidence', 1.0)
        print(f'track_id=0:')
        print(f'  置信度: {conf:.3f}')
        print(f'  是否高于阈值: {conf >= window.conf_threshold}')
        
        bbox = ann['bbox']
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        
        result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
        print(f'  点包含检测: 结果={result:.2f}')

import os
os.unlink(output_path)
app.quit()
