#!/usr/bin/env python3
"""详细测试坐标转换"""

import json
import cv2
import numpy as np
from pathlib import Path

def test_frame_annotations():
    """测试帧1的所有标注"""
    labels_file = Path("/Users/jimjiang/Downloads/biaozhu/temp_data/labels/frame_000001.json")
    with open(labels_file) as f:
        annotations = json.load(f)
    
    print(f"帧1的标注（共{len(annotations)}个）:")
    print()
    
    for i, ann in enumerate(annotations):
        track_id = ann.get('track_id', ann['id'])
        conf = ann.get('confidence', 1.0)
        bbox = ann['bbox']
        polygon = ann['segmentation'][0]
        
        print(f"标注{i}: track_id={track_id}, 置信度={conf:.3f}")
        print(f"  边界框: x={bbox[0]:.0f}-{bbox[0]+bbox[2]:.0f}, y={bbox[1]:.0f}-{bbox[1]+bbox[3]:.0f}")
        print(f"  多边形范围: x=[{min(polygon[::2]):.0f}-{max(polygon[::2]):.0f}], y=[{min(polygon[1::2]):.0f}-{max(polygon[1::2]):.0f}]")
        
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        print(f"  顶点数: {len(pts)}")
        print()

def test_conversion():
    """测试坐标转换"""
    # 视频信息
    video_w, video_h = 1280, 720
    
    # frames目录中的图片尺寸
    import os
    frame_path = Path("/Users/jimjiang/Downloads/biaozhu/temp_data/frames/frame_000001.jpg")
    img = cv2.imread(str(frame_path))
    if img is not None:
        frame_h, frame_w = img.shape[:2]
        print(f"帧图片尺寸: {frame_w}x{frame_h}")
    else:
        print("无法读取帧图片")
        return
    
    print(f"视频尺寸(video_info): {video_w}x{video_h}")
    print()
    
    # 检查是否一致
    if frame_w != video_w or frame_h != video_h:
        print(f"⚠️ 警告：帧图片尺寸与video_info不一致！")
        print(f"   帧图片: {frame_w}x{frame_h}")
        print(f"   video_info: {video_w}x{video_h}")
        print()
    
    # 用户终端显示的转换
    print("用户点击坐标转换示例:")
    examples = [
        ("点击坐标", 107, 64),
        ("点击坐标", 108, 419),
        ("点击坐标", 743, 418),
    ]
    
    # 假设label尺寸
    label_w, label_h = 851, 480
    pixmap_w, pixmap_h = frame_w, frame_h  # 帧图片尺寸
    
    scale = min(label_w / pixmap_w, label_h / pixmap_h)
    display_w = int(pixmap_w * scale)
    display_h = int(pixmap_h * scale)
    display_x = (label_w - display_w) // 2
    display_y = (label_h - display_h) // 2
    
    print(f"label尺寸: {label_w}x{label_h}")
    print(f"pixmap尺寸: {pixmap_w}x{pixmap_h}")
    print(f"scale: {scale:.4f}")
    print(f"display区域: ({display_x},{display_y}) 到 ({display_x+display_w},{display_y+display_h})")
    print()
    
    for desc, click_x, click_y in examples:
        if display_x <= click_x < display_x + display_w and display_y <= click_y < display_y + display_h:
            pixmap_x = int((click_x - display_x) / scale)
            pixmap_y = int((click_y - display_y) / scale)
            
            video_x = int(pixmap_x * video_w / pixmap_w)
            video_y = int(pixmap_y * video_h / pixmap_h)
            
            print(f"{desc} ({click_x},{click_y}):")
            print(f"  → pixmap: ({pixmap_x},{pixmap_y})")
            print(f"  → video: ({video_x},{video_y})")
            print()
        else:
            print(f"{desc} ({click_x},{click_y}): 不在display区域内")

def test_actual_clicks():
    """测试用户实际点击的坐标"""
    labels_file = Path("/Users/jimjiang/Downloads/biaozhu/temp_data/labels/frame_000001.json")
    with open(labels_file) as f:
        annotations = json.load(f)
    
    # 用户点击的坐标
    clicks = [
        (223, 285),  # 来自用户终端
    ]
    
    print("测试用户点击坐标:")
    print()
    
    for click_x, click_y in clicks:
        print(f"点击: ({click_x},{click_y})")
        
        for i, ann in enumerate(annotations):
            track_id = ann.get('track_id', ann['id'])
            conf = ann.get('confidence', 1.0)
            if conf < 0.5:
                continue
                
            polygon = ann['segmentation'][0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            
            result = cv2.pointPolygonTest(pts, (float(click_x), float(click_y)), False)
            contains = result >= 0
            
            print(f"  标注{i} track_id={track_id}: {'包含' if contains else '不包含'} (结果={result:.2f})")

if __name__ == "__main__":
    print("=" * 70)
    print("详细坐标测试")
    print("=" * 70)
    print()
    
    test_frame_annotations()
    test_conversion()
    test_actual_clicks()
