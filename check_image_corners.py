import json
import cv2
import numpy as np

with open('temp_data/labels/frame_000001.json') as f:
    anns = json.load(f)

print('图像四个角坐标：')
corners = [(0,0), (0,719), (1279,0), (1279,719)]
for px, py in corners:
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
            print(f'({px},{py}): 在track_id={track_id}内 ✓')
            break
    if not found:
        print(f'({px},{py}): 不在任何mask内')
