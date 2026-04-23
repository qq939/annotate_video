#!/usr/bin/env python3
"""自动测试点击功能"""

import sys
import os
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent

def test_click_detection():
    """测试点击检测"""
    print("=" * 70)
    print("自动测试点击检测功能")
    print("=" * 70)
    
    app = QApplication(sys.argv)
    
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    try:
        from post_annotate import PostAnnotatorWindow
        
        window = PostAnnotatorWindow(output_path)
        window.show()
        
        print(f"\n视频信息: {window.video_info['width']}x{window.video_info['height']}")
        print(f"标签尺寸: {window.image_label.width()}x{window.image_label.height()}")
        
        # 加载第21帧（用户成功点击的那帧）
        window.current_frame_idx = 21
        window.update_display()
        
        # 读取帧21的标注
        labels_file = window.labels_dir / "frame_000021.json"
        import json
        with open(labels_file) as f:
            annotations = json.load(f)
        
        print(f"\n帧21有{len(annotations)}个标注")
        
        # 找到所有置信度>=0.5的标注
        valid_annotations = [ann for ann in annotations if ann.get('confidence', 1.0) >= 0.5]
        print(f"置信度>=0.5的标注: {len(valid_annotations)}个")
        
        # 在每个有效标注的边界框中心创建点击事件
        for i, ann in enumerate(valid_annotations):
            bbox = ann['bbox']
            track_id = ann.get('track_id', ann['id'])
            conf = ann.get('confidence', 1.0)
            
            # 计算边界框中心点（视频坐标）
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            
            print(f"\n测试标注{i}: track_id={track_id}, 置信度={conf:.3f}")
            print(f"  边界框中心(视频坐标): ({center_x:.0f}, {center_y:.0f})")
            
            # 计算在label上的显示位置
            label_w = window.image_label.width()
            label_h = window.image_label.height()
            video_w = window.video_info['width']
            video_h = window.video_info['height']
            
            scale = min(label_w / video_w, label_h / video_h)
            display_w = int(video_w * scale)
            display_h = int(video_h * scale)
            offset_x = (label_w - display_w) // 2
            offset_y = (label_h - display_h) // 2
            
            # 转换为label上的点击位置
            display_x = int(offset_x + center_x * scale)
            display_y = int(offset_y + center_y * scale)
            
            print(f"  计算的显示位置: ({display_x}, {display_y})")
            
            # 创建鼠标点击事件
            event = QMouseEvent(
                QMouseEvent.MouseButtonPress,
                QPoint(display_x, display_y),
                Qt.LeftButton,
                Qt.LeftButton,
                Qt.NoModifier
            )
            
            # 模拟点击
            window.image_label.mousePressEvent(event)
            
            # 检查是否添加了删除点
            if window.del_points:
                last_del = window.del_points[-1]
                print(f"  ✓ 点击成功! 删除的track_ids: {last_del.get('track_ids', [])}")
            else:
                print(f"  ✗ 点击失败!")
                
            # 清理
            window.del_points.clear()
            
    finally:
        app.quit()
        if os.path.exists(output_path):
            os.unlink(output_path)

if __name__ == "__main__":
    test_click_detection()
