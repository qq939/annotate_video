#!/usr/bin/env python3
"""验证修复后的坐标转换"""

import json
import cv2
import numpy as np
from pathlib import Path

def test_fixed_conversion():
    """测试修复后的坐标转换"""
    # 视频信息
    video_w, video_h = 1280, 720
    label_w, label_h = 851, 480
    
    # 使用video尺寸计算scale
    scale = min(label_w / video_w, label_h / video_h)
    display_w = int(video_w * scale)
    display_h = int(video_h * scale)
    display_x = (label_w - display_w) // 2
    display_y = (label_h - display_h) // 2
    
    print(f"修复后的转换:")
    print(f"  video尺寸: {video_w}x{video_h}")
    print(f"  label尺寸: {label_w}x{label_h}")
    print(f"  scale: {scale:.4f}")
    print(f"  display区域: ({display_x},{display_y}) 到 ({display_x+display_w},{display_y+display_h})")
    print()
    
    # 测试点击
    test_clicks = [(107, 64), (108, 419), (743, 418), (744, 65)]
    
    for click_x, click_y in test_clicks:
        if display_x <= click_x < display_x + display_w and display_y <= click_y < display_y + display_h:
            x = int((click_x - display_x) / scale)
            y = int((click_y - display_y) / scale)
            print(f"点击({click_x},{click_y}) → 视频坐标({x},{y})")
        else:
            print(f"点击({click_x},{click_y}) → 不在display区域内")

def test_polygon_detection():
    """测试多边形检测"""
    labels_file = Path("/Users/jimjiang/Downloads/biaozhu/temp_data/labels/frame_000001.json")
    with open(labels_file) as f:
        annotations = json.load(f)
    
    # 使用修复后的转换计算
    video_w, video_h = 1280, 720
    label_w, label_h = 851, 480
    scale = min(label_w / video_w, label_h / video_h)
    display_w = int(video_w * scale)
    display_h = int(video_h * scale)
    display_x = (label_w - display_w) // 2
    display_y = (label_h - display_h) // 2
    
    # 测试点击
    test_click = (744, 65)  # 用户终端显示的右上角点击
    
    x = int((test_click[0] - display_x) / scale)
    y = int((test_click[1] - display_y) / scale)
    
    print(f"\n测试多边形检测:")
    print(f"  点击: ({test_click[0]},{test_click[1]})")
    print(f"  转换后: ({x},{y})")
    print()
    
    for i, ann in enumerate(annotations):
        conf = ann.get('confidence', 1.0)
        if conf < 0.5:
            continue
        
        track_id = ann.get('track_id', ann['id'])
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        
        result = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
        contains = result >= 0
        
        print(f"  标注{i} track_id={track_id}: {'✓ 包含' if contains else '✗ 不包含'} (结果={result:.2f})")

if __name__ == "__main__":
    print("=" * 60)
    print("验证修复后的坐标转换")
    print("=" * 60)
    test_fixed_conversion()
    test_polygon_detection()
