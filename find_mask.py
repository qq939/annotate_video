import json
import cv2
import numpy as np

# 读取帧12的标注
with open('temp_data/labels/frame_000012.json') as f:
    anns = json.load(f)

print(f'帧12共{len(anns)}个标注')
print()

# 用户点击的坐标
click_x, click_y = 437, 552

print(f'用户点击点: ({click_x}, {click_y})')
print()

found = False
for ann in anns:
    track_id = ann.get('track_id', ann['id'])
    bbox = ann['bbox']
    polygon = ann['segmentation'][0]
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    
    result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
    in_poly = result >= 0
    
    if in_poly:
        found = True
        print(f'✓ track_id={track_id}: 包含该点! 结果={result:.2f}')
        print(f'  边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}')
        print()

if not found:
    print('⚠️ 该点不在任何mask内！')
    print()
    print('检查：用户看到的画面和temp_data的数据是否一致？')
