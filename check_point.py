import json
import cv2
import numpy as np

with open('temp_data/labels/frame_000001.json') as f:
    anns = json.load(f)

print(f'帧1共{len(anns)}个标注')
print(f'用户点击点: (448, 565)')
print()

found = False
for ann in anns:
    track_id = ann.get('track_id', ann['id'])
    bbox = ann['bbox']
    polygon = ann['segmentation'][0]
    pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
    
    # 检查点是否在边界框内
    in_bbox = (bbox[0] <= 448 <= bbox[0]+bbox[2] and 
               bbox[1] <= 565 <= bbox[1]+bbox[3])
    
    # 检查点是否在多边形内
    result = cv2.pointPolygonTest(pts, (448, 565), False)
    in_poly = result >= 0
    
    if in_bbox or in_poly:
        found = True
        print(f'track_id={track_id}: 边界框内={in_bbox}, 多边形内={in_poly}')
        print(f'  边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}')
        print(f'  多边形范围: x={pts[:,0].min()}-{pts[:,0].max()}, y={pts[:,1].min()}-{pts[:,1].max()}')
        print()

if not found:
    print('⚠️ 该点不在帧1的任何mask内！')
    print()
    print('检查：用户可能看错了画面，或者数据有问题')
