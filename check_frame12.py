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

for ann in anns:
    track_id = ann.get('track_id', ann['id'])
    bbox = ann['bbox']
    polygon = ann['segmentation'][0]
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    
    # 检查点是否在边界框内
    in_bbox = (bbox[0] <= click_x <= bbox[0]+bbox[2] and 
               bbox[1] <= click_y <= bbox[1]+bbox[3])
    
    # 检查点是否在多边形内
    result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
    in_poly = result >= 0
    
    print(f'track_id={track_id}')
    print(f'  边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}')
    print(f'  点({click_x},{click_y}): 边界框内={in_bbox}, 多边形内={in_poly}, 结果={result:.2f}')
    
    if track_id == 0:
        print('  ← 用户点击的标注!')
    print()
