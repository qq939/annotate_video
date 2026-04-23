import json
import cv2
import numpy as np

# 读取labels
with open('temp_data/labels/frame_000012.json') as f:
    labels = json.load(f)

click_x, click_y = 437, 552

print(f'检查点({click_x}, {click_y})是否在任何mask内:')
print()

found = False
for ann in labels:
    track_id = ann.get('track_id', ann.get('id'))
    conf = ann.get('confidence', 1.0)
    if conf < 0.5:
        continue
    bbox = ann['bbox']
    polygon = ann['segmentation'][0]
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    
    result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
    in_poly = result >= 0
    
    if in_poly:
        found = True
        print(f'track_id={track_id}: 包含! conf={conf:.3f}')
        print(f'  边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}')
        print()

if not found:
    print('该点不在任何mask内!')
