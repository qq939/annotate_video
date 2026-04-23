#!/usr/bin/env python3
"""测试用户成功点击的那帧"""

import json
import cv2
import numpy as np
from pathlib import Path

def test_frame_21():
    """测试帧21"""
    labels_file = Path("/Users/jimjiang/Downloads/biaozhu/temp_data/labels/frame_000021.json")
    
    if not labels_file.exists():
        print(f"帧21的标注文件不存在")
        return
    
    with open(labels_file) as f:
        annotations = json.load(f)
    
    print(f"帧21有{len(annotations)}个标注")
    print()
    
    # 用户点击的坐标 (672, 350)
    video_w, video_h = 1280, 720
    label_w, label_h = 851, 480
    scale = min(label_w / video_w, label_h / video_h)
    display_w = int(video_w * scale)
    display_h = int(video_h * scale)
    display_x = (label_w - display_w) // 2
    display_y = (label_h - display_h) // 2
    
    test_click = (672, 350)
    x = int((test_click[0] - display_x) / scale)
    y = int((test_click[1] - display_y) / scale)
    
    print(f"用户点击: ({test_click[0]},{test_click[1]})")
    print(f"转换后视频坐标: ({x},{y})")
    print()
    
    for i, ann in enumerate(annotations):
        track_id = ann.get('track_id', ann['id'])
        conf = ann.get('confidence', 1.0)
        bbox = ann['bbox']
        
        print(f"标注{i}: track_id={track_id}, 置信度={conf:.3f}")
        print(f"  边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}")
        
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        
        result = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
        contains = result >= 0
        
        print(f"  点({x},{y}): {'✓ 包含' if contains else '✗ 不包含'} (结果={result:.2f})")
        print()

if __name__ == "__main__":
    test_frame_21()
