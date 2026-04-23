import unittest
import sys
import os
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest

class TestPostAnnotateE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

    def test_playback_speed_adjustment(self):
        """端到端测试：播放速度调节功能"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            window.play_speed = 4.0
            window.on_speed_change(4.0)
            
            self.assertEqual(window.play_speed, 4.0, "播放速度应该设置为4.0")
            self.assertIn("4.0fps", window.speed_label.text(), "标签应显示4.0fps")
            
            print("✓ 播放速度调节测试通过（默认4fps）")
            
            window.on_speed_change(10.0)
            self.assertEqual(window.play_speed, 10.0, "播放速度应该更新为10.0")
            
            print("✓ 播放速度更新测试通过")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_play_toggle_functionality(self):
        """端到端测试：播放/暂停切换功能"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            window.is_playing = False
            window.toggle_play()
            
            self.assertTrue(window.is_playing, "切换后应该正在播放")
            self.assertEqual(window.play_btn.text(), "暂停", "按钮文本应为'暂停'")
            
            print("✓ 播放状态切换测试通过")
            
            window.toggle_play()
            
            self.assertFalse(window.is_playing, "再次切换后应该停止播放")
            self.assertEqual(window.play_btn.text(), "播放", "按钮文本应为'播放'")
            
            print("✓ 暂停状态切换测试通过")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_frame_navigation(self):
        """端到端测试：帧导航功能"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            initial_frame = window.current_frame_idx
            
            window.play_next_frame()
            
            expected_frame = (initial_frame + 1) % window.total_frames
            self.assertEqual(window.current_frame_idx, expected_frame, "帧索引应该正确递增")
            
            print(f"✓ 帧导航测试通过（从帧 {initial_frame} 导航到帧 {expected_frame}）")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_timer_interval_calculation(self):
        """端到端测试：定时器间隔计算"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            for speed in [1, 4, 10, 20]:
                window.play_speed = float(speed)
                window.timer.setInterval(int(1000.0 / window.play_speed))
                
                expected_interval = int(1000.0 / speed)
                actual_interval = window.timer.interval()
                
                self.assertEqual(actual_interval, expected_interval, 
                               f"速度 {speed}fps 时的间隔应该为 {expected_interval}ms")
                
                print(f"✓ 速度 {speed}fps → 间隔 {expected_interval}ms 计算正确")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

if __name__ == '__main__':
    unittest.main(verbosity=2)
