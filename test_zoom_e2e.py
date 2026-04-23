import unittest
import sys
import os
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QLabel

class TestZoomE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

    def test_zoom_level_initialization(self):
        """端到端测试：缩放级别初始化"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            self.assertEqual(window.zoom_level, 100, "缩放级别应该初始化为100")
            print(f"✓ 缩放级别初始化正确: {window.zoom_level}%")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_zoom_level_change(self):
        """端到端测试：缩放级别变化"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            window.on_zoom_change(150)
            self.assertEqual(window.zoom_level, 150, "缩放级别应该更新为150")
            self.assertIn("150%", window.zoom_label.text(), "标签应该显示150%")
            print(f"✓ 缩放级别更新正确: {window.zoom_level}%")
            
            window.on_zoom_change(50)
            self.assertEqual(window.zoom_level, 50, "缩放级别应该更新为50")
            print(f"✓ 缩放级别更新正确: {window.zoom_level}%")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_zoom_slider_range(self):
        """端到端测试：缩放滑块范围"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            min_val = window.zoom_slider.minimum()
            max_val = window.zoom_slider.maximum()
            
            self.assertEqual(min_val, 25, "最小值应该是25%")
            self.assertEqual(max_val, 200, "最大值应该是200%")
            print(f"✓ 缩放滑块范围正确: {min_val}% - {max_val}%")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

if __name__ == '__main__':
    unittest.main(verbosity=2)
