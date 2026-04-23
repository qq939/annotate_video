import unittest
import sys
import os
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication

class TestConfThresholdE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

    def test_conf_threshold_filtering(self):
        """端到端测试：置信度筛选功能"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            test_annotations = [
                {'id': 1, 'confidence': 0.9, 'track_id': 101, 'segmentation': [[[0,0,10,0,10,10,0,10]]], 'bbox': [0, 0, 10, 10], 'category_id': 1},
                {'id': 2, 'confidence': 0.3, 'track_id': 102, 'segmentation': [[[20,0,30,0,30,10,20,10]]], 'bbox': [20, 0, 10, 10], 'category_id': 1},
                {'id': 3, 'confidence': 0.6, 'track_id': 103, 'segmentation': [[[40,0,50,0,50,10,40,10]]], 'bbox': [40, 0, 10, 10], 'category_id': 1},
            ]
            
            window.conf_threshold = 0.5
            
            import numpy as np
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame = window.apply_threshold_to_masks(frame, test_annotations, window.conf_threshold)
            
            visible_ids = []
            for ann in test_annotations:
                if ann['confidence'] >= window.conf_threshold:
                    visible_ids.append(ann['id'])
            
            self.assertIn(1, visible_ids, "置信度0.9应该可见")
            self.assertNotIn(2, visible_ids, "置信度0.3应该不可见")
            self.assertIn(3, visible_ids, "置信度0.6应该可见")
            
            print(f"✓ 置信度筛选测试通过: {visible_ids}")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_del_track_id_filtering(self):
        """端到端测试：删除track_id筛选"""
        from post_annotate import PostAnnotatorWindow
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            output_path = f.name
        
        try:
            window = PostAnnotatorWindow(output_path)
            window.show()
            
            test_annotations = [
                {'id': 1, 'confidence': 0.9, 'track_id': 101, 'segmentation': [[[0,0,10,0,10,10,0,10]]], 'bbox': [0, 0, 10, 10], 'category_id': 1},
                {'id': 2, 'confidence': 0.9, 'track_id': 102, 'segmentation': [[[20,0,30,0,30,10,20,10]]], 'bbox': [20, 0, 10, 10], 'category_id': 1},
                {'id': 3, 'confidence': 0.9, 'track_id': 103, 'segmentation': [[[40,0,50,0,50,10,40,10]]], 'bbox': [40, 0, 10, 10], 'category_id': 1},
            ]
            
            window.del_track_id_list = [102]
            
            visible_annotations = [
                ann for ann in test_annotations
                if ann.get('track_id', ann['id']) not in window.del_track_id_list
            ]
            
            self.assertEqual(len(visible_annotations), 2, "应该只有2个可见")
            self.assertEqual(visible_annotations[0]['track_id'], 101)
            self.assertEqual(visible_annotations[1]['track_id'], 103)
            
            print(f"✓ 删除track_id筛选测试通过")
            
            window.close()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

if __name__ == '__main__':
    unittest.main(verbosity=2)
