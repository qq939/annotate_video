#!/usr/bin/env python3
"""直接测试点击功能"""

import sys
sys.path.insert(0, '/Users/jimjiang/Downloads/biaozhu')

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent
import tempfile
import json
import cv2
import numpy as np

def test_simple():
    app = QApplication(sys.argv)
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    from post_annotate import PostAnnotatorWindow
    
    window = PostAnnotatorWindow(output_path)
    window.show()
    
    print(f"视频尺寸: {window.video_info['width']}x{window.video_info['height']}")
    print(f"标签尺寸: {window.image_label.width()}x{window.image_label.height()}")
    print()
    
    # 直接模拟点击，发送视频坐标
    test_points = [
        (353, 595),  # track_id=0的边界框中心
        (240, 103),  # track_id=1的边界框中心
    ]
    
    for click_x, click_y in test_points:
        print(f"点击视频坐标: ({click_x}, {click_y})")
        
        # 模拟点击
        event = QMouseEvent(
            QMouseEvent.MouseButtonPress,
            QPoint(click_x, click_y),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier
        )
        window.image_label.mousePressEvent(event)
        
        if window.del_points:
            last = window.del_points[-1]
            print(f"  ✓ 成功! track_ids: {last.get('track_ids', [])}")
        else:
            print(f"  ✗ 失败")
        
        # 清空继续测试
        window.del_points.clear()
        window.del_track_id_list.clear()
        print()
    
    import os
    os.unlink(output_path)
    app.quit()

if __name__ == "__main__":
    test_simple()
