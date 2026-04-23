#!/usr/bin/env python3
"""测试坐标转换"""

import json
import cv2
import numpy as np
from pathlib import Path

def test_point_in_polygon():
    """测试点是否在多边形内"""
    # 读取测试帧的标注
    labels_file = Path("/Users/jimjiang/Downloads/biaozhu/temp_data/labels/frame_000001.json")
    with open(labels_file) as f:
        annotations = json.load(f)
    
    print(f"帧1有{len(annotations)}个标注")
    
    # 测试点 (365, 624) - 这是第一个标注的起点
    test_x, test_y = 365, 624
    
    for i, ann in enumerate(annotations):
        conf = ann.get('confidence', 1.0)
        if conf < 0.5:
            continue
            
        polygon = ann['segmentation'][0]
        track_id = ann.get('track_id', ann['id'])
        
        # 转换为numpy数组
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        
        print(f"\n标注{i}: track_id={track_id}, 置信度={conf:.3f}")
        print(f"  多边形点数: {len(pts)}")
        print(f"  边界框: {ann['bbox']}")
        print(f"  前3个点: {pts[:3].tolist()}")
        print(f"  后3个点: {pts[-3:].tolist()}")
        
        # 测试点是否在多边形内
        result = cv2.pointPolygonTest(pts, (float(test_x), float(test_y)), False)
        contains = result >= 0
        
        print(f"  测试点({test_x},{test_y}): 结果={result:.2f}, 包含={contains}")
        
        if contains:
            print(f"  ✓ 找到包含该点的标注!")

def test_manual_polygon():
    """手动创建一个多边形并测试"""
    # 创建一个矩形多边形
    pts = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.int32)
    
    test_points = [
        (150, 150, "中心点"),
        (50, 50, "外部左上"),
        (250, 250, "外部右下"),
        (100, 100, "顶点"),
    ]
    
    print("\n手动测试多边形:")
    print(f"多边形: {pts.tolist()}")
    
    for x, y, desc in test_points:
        result = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
        contains = result >= 0
        print(f"  {desc} ({x},{y}): 结果={result:.2f}, 包含={contains}")

def test_video_coordinates():
    """测试视频坐标转换"""
    # 视频尺寸
    video_w, video_h = 1280, 720
    
    # 假设显示尺寸
    label_w, label_h = 851, 480
    
    # pixmap尺寸（视频缩放到640x360显示）
    pixmap_w, pixmap_h = 640, 360
    
    # 计算缩放比例
    scale = min(label_w / pixmap_w, label_h / pixmap_h)
    print(f"\n缩放计算:")
    print(f"  label尺寸: {label_w}x{label_h}")
    print(f"  pixmap尺寸: {pixmap_w}x{pixmap_h}")
    print(f"  scale: {scale:.4f}")
    
    # display区域
    display_w = int(pixmap_w * scale)
    display_h = int(pixmap_h * scale)
    display_x = (label_w - display_w) // 2
    display_y = (label_h - display_h) // 2
    
    print(f"  display区域: ({display_x},{display_y}) to ({display_x+display_w},{display_y+display_h})")
    
    # 测试点击坐标
    click_x, click_y = 107, 64
    
    print(f"\n点击坐标: ({click_x},{click_y})")
    
    # 检查是否在display内
    if display_x <= click_x < display_x + display_w and display_y <= click_y < display_y + display_h:
        print("  ✓ 点击在display区域内")
        
        # 转换为pixmap坐标
        pixmap_x = int((click_x - display_x) / scale)
        pixmap_y = int((click_y - display_y) / scale)
        print(f"  pixmap坐标: ({pixmap_x},{pixmap_y})")
        
        # 转换为视频坐标
        video_x = int(pixmap_x * video_w / pixmap_w)
        video_y = int(pixmap_y * video_h / pixmap_h)
        print(f"  视频坐标: ({video_x},{video_y})")
    else:
        print("  ✗ 点击不在display区域内")

if __name__ == "__main__":
    print("=" * 60)
    print("测试坐标转换")
    print("=" * 60)
    
    test_manual_polygon()
    test_point_in_polygon()
    test_video_coordinates()
