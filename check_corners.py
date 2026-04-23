import json
import cv2
import numpy as np

with open('temp_data/labels/frame_000001.json') as f:
    anns = json.load(f)

print('帧1的mask边界框:')
for ann in anns:
    track_id = ann.get('track_id', ann['id'])
    bbox = ann['bbox']
    conf = ann.get('confidence', 1.0)
    print(f'track_id={track_id}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}], conf={conf:.3f}')

print()
print('用户点击的四个角: (107,63), (106,418), (741,416), (744,63)')
print()

test_points = [(107,63), (106,418), (741,416), (744,63)]
for px, py in test_points:
    found = False
    for ann in anns:
        if ann.get('confidence', 1.0) < 0.5:
            continue
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        result = cv2.pointPolygonTest(pts, (float(px), float(py)), False)
        if result >= 0:
            found = True
            track_id = ann.get('track_id', ann['id'])
            print(f'点({px},{py}): 在track_id={track_id}内')
            break
    if not found:
        print(f'点({px},{py}): 不在任何mask内')
