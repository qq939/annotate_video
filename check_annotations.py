import json
import cv2
import numpy as np
from pathlib import Path

# 读取annotations.json
with open('temp_data/annotations.json') as f:
    coco = json.load(f)

print('annotations.json中image_id=12的标注:')
for ann in coco['annotations']:
    if ann.get('image_id') == 12:
        track_id = ann.get('track_id', ann['id'])
        conf = ann.get('confidence', 1.0)
        bbox = ann['bbox']
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        
        # 用户点击的坐标
        click_x, click_y = 437, 552
        result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
        
        print(f'track_id={track_id}: conf={conf:.3f}, 包含={result>=0}, 边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}')

print()
print('frame_000012.json中的track_id=0:')
with open('temp_data/labels/frame_000012.json') as f:
    labels = json.load(f)

for ann in labels:
    if ann.get('track_id', ann.get('id')) == 0:
        conf = ann.get('confidence', 1.0)
        bbox = ann['bbox']
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        
        click_x, click_y = 437, 552
        result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
        
        print(f'track_id=0: conf={conf:.3f}, 包含={result>=0}, 边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}')
