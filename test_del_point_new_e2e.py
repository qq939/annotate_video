import unittest
import sys
import os
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QLabel

class TestDelPointNewE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

    def test_del_point_data_structure(self):
        """端到端测试：新的删除点数据结构"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            del_info = {
                'x': 100,
                'y': 200,
                'frame_idx': 5,
                'track_id': 123,
                'ann_id': None,
                'shortcut': 'F6',
                'idx': 0
            }
            window.del_points.append(del_info)
            
            point = window.del_points[0]
            self.assertEqual(point['x'], 100)
            self.assertEqual(point['y'], 200)
            self.assertEqual(point['frame_idx'], 5)
            self.assertEqual(point['track_id'], 123)
            self.assertEqual(point['shortcut'], 'F6')
            
            print(f"✓ 新删除点数据结构正确: {point}")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_add_del_point_ui(self):
        """端到端测试：添加删除点到UI"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            del_info = {
                'idx': 0,
                'x': 100,
                'y': 200,
                'frame_idx': 5,
                'track_id': 123,
                'ann_id': None,
                'shortcut': 'F6'
            }
            
            window.add_del_point_ui(del_info)
            
            layout_count = window.del_points_layout.count()
            self.assertEqual(layout_count, 1, "应该有1个UI条目")
            
            print(f"✓ UI条目添加成功，当前有{layout_count}个条目")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_remove_del_point(self):
        """端到端测试：移除删除点"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            window.del_track_id_list = [123]
            
            del_info = {
                'idx': 0,
                'x': 100,
                'y': 200,
                'frame_idx': 5,
                'track_id': 123,
                'ann_id': None,
                'shortcut': 'F6'
            }
            window.del_points.append(del_info)
            window.add_del_point_ui(del_info)
            
            self.assertEqual(len(window.del_points), 1)
            self.assertEqual(len(window.del_track_id_list), 1)
            
            window.remove_del_point(0)
            
            self.assertEqual(len(window.del_points), 0)
            self.assertEqual(len(window.del_track_id_list), 0)
            
            print("✓ 删除点移除功能正常")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_update_del_count_label(self):
        """端到端测试：更新删除计数标签"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            window.del_track_id_list = [1, 2, 3]
            window.update_del_count_label()
            
            count_label = window.findChild(QLabel, "del_count_label")
            self.assertIsNotNone(count_label, "应该找到计数标签")
            
            text = count_label.text()
            self.assertIn("3", text, "应该显示3个标注")
            
            print(f"✓ 计数标签更新: {text}")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_clear_all_del_points(self):
        """端到端测试：清空所有删除点"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            for i in range(3):
                del_info = {
                    'idx': i,
                    'x': 100 + i*10,
                    'y': 200 + i*10,
                    'frame_idx': i,
                    'track_id': i + 1,
                    'ann_id': None,
                    'shortcut': f'F{i+1}'
                }
                window.del_points.append(del_info)
                window.del_track_id_list.append(i + 1)
                window.add_del_point_ui(del_info)
            
            self.assertEqual(len(window.del_points), 3)
            self.assertEqual(len(window.del_track_id_list), 3)
            
            window.clear_del_points()
            
            self.assertEqual(len(window.del_points), 0)
            self.assertEqual(len(window.del_track_id_list), 0)
            self.assertEqual(window.del_points_layout.count(), 0)
            
            print("✓ 清空所有删除点功能正常")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

if __name__ == '__main__':
    unittest.main(verbosity=2)
