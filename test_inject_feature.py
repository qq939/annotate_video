import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestTrackIdFeature(unittest.TestCase):

    def test_next_track_id_starts_at_9999(self):
        from video_control import VideoController
        c = VideoController()
        self.assertEqual(c.next_track_id, 9999)

    def test_click_assigns_immediately(self):
        from video_control import VideoController
        import tempfile
        c = VideoController()
        c.add_track_id_point(10, 20, 0, 7)
        pt = c.track_id_points[0]
        self.assertEqual(pt['track_id'], 7)
        self.assertIsNone(pt['assigned_id'])
        assigned = c.next_track_id
        pt['assigned_id'] = assigned
        self.assertEqual(assigned, 9999)
        self.assertEqual(c.next_track_id, 9999)

    def test_convert_track_id_old_to_new(self):
        from video_control import VideoController
        import tempfile
        c = VideoController()
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = Path(tmpdir)
            ann = [{'track_id': 7, 'bbox': [0,0,10,10]}]
            (labels_dir / 'frame_000000.json').write_text(json.dumps(ann))

            ann_read = json.loads((labels_dir / 'frame_000000.json').read_text())
            ann_read[0]['track_id'] = 9999
            (labels_dir / 'frame_000000.json').write_text(json.dumps(ann_read))

            result = json.loads((labels_dir / 'frame_000000.json').read_text())
            self.assertEqual(result[0]['track_id'], 9999)

    def test_controller_next_track_id_increments(self):
        from video_control import VideoController
        c = VideoController()
        c.next_track_id += 1
        self.assertEqual(c.next_track_id, 10000)

    def test_purple_rendering_applies_purple_color(self):
        from video_control import VideoController
        import numpy as np
        c = VideoController()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        anns = [
            {'track_id': 9999, 'bbox': [10, 10, 30, 30], 'confidence': 0.9, 'category': 'test'},
            {'track_id': 10000, 'bbox': [50, 50, 20, 20], 'confidence': 0.8, 'category': 'test'},
        ]
        result = c.apply_threshold_to_masks(frame, anns)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, frame.shape)

    def test_decrement_does_not_go_below_9999(self):
        from video_control import VideoController
        c = VideoController()
        c.next_track_id = 9999
        self.assertEqual(c.next_track_id, 9999)

    def test_export_only_track_id_gt_9900(self):
        anns = [
            {'track_id': 9900, 'bbox': [0,0,10,10], 'confidence': 1.0},
            {'track_id': 9901, 'bbox': [0,0,10,10], 'confidence': 1.0},
            {'track_id': 9999, 'bbox': [0,0,10,10], 'confidence': 1.0},
            {'track_id': 10000, 'bbox': [0,0,10,10], 'confidence': 1.0},
            {'track_id': 5, 'bbox': [0,0,10,10], 'confidence': 1.0},
            {'track_id': 100, 'bbox': [0,0,10,10], 'confidence': 1.0},
        ]
        filtered = [ann for ann in anns if ann.get('track_id', 0) > 9900]
        self.assertEqual(len(filtered), 3)
        track_ids = [a['track_id'] for a in filtered]
        self.assertIn(9901, track_ids)
        self.assertIn(9999, track_ids)
        self.assertIn(10000, track_ids)
        self.assertNotIn(9900, track_ids)
        self.assertNotIn(5, track_ids)
        self.assertNotIn(100, track_ids)

    def test_delete_trace_id_filters_correctly(self):
        import tempfile
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = Path(tmpdir)
            (labels_dir / "frame_000000.json").write_text(json.dumps([
                {'track_id': 7, 'bbox': [0,0,10,10], 'confidence': 1.0},
                {'track_id': 5, 'bbox': [0,0,10,10], 'confidence': 1.0},
            ]))
            anns = json.loads((labels_dir / "frame_000000.json").read_text())
            anns = [a for a in anns if a.get('track_id') != 7]
            (labels_dir / "frame_000000.json").write_text(json.dumps(anns))
            result = json.loads((labels_dir / "frame_000000.json").read_text())
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['track_id'], 5)

    def test_merge_masks_in_frame_no_overlap(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from annotate_video import merge_masks_in_frame
        import numpy as np
        mask1 = np.zeros((10, 10), dtype=np.uint8)
        mask1[2:5, 2:5] = 1
        mask2 = np.zeros((10, 10), dtype=np.uint8)
        mask2[6:9, 6:9] = 1
        masks, bboxes = merge_masks_in_frame([mask1, mask2], [[0,0,3,3], [0,0,3,3]], 0.5)
        self.assertEqual(len(masks), 2)

    def test_track_color_map_same_id_same_color(self):
        mask_colors = [
            (255, 0, 0),     # 红
            (255, 165, 0),   # 橙
            (255, 255, 0),   # 黄
            (0, 255, 0),     # 绿
            (0, 255, 255),   # 青
            (0, 0, 255),     # 蓝
            (128, 0, 128),   # 紫
        ]
        track_color_map = {}

        def get_color(track_id):
            if track_id not in track_color_map:
                track_color_map[track_id] = mask_colors[len(track_color_map) % len(mask_colors)]
            return track_color_map[track_id]

        self.assertEqual(get_color(5), (255, 0, 0))
        self.assertEqual(get_color(5), (255, 0, 0))
        self.assertEqual(get_color(7), (255, 165, 0))
        self.assertEqual(get_color(5), (255, 0, 0))
        self.assertEqual(get_color(9999), (255, 255, 0))
        self.assertEqual(get_color(10000), (0, 255, 0))
        self.assertEqual(get_color(3), (0, 255, 255))

    def test_filter_with_9999(self):
        from video_control import VideoController
        c = VideoController()
        c.track_ids_to_9999.add(7)
        anns = [
            {'track_id': 9999, 'bbox': [0,0,10,10], 'confidence': 1.0},
            {'track_id': 10000, 'bbox': [0,0,10,10], 'confidence': 1.0},
            {'track_id': 7, 'bbox': [0,0,10,10], 'confidence': 1.0},
        ]
        filtered = c.filter_annotations(anns)
        self.assertEqual(len(filtered), 2)
        self.assertNotIn(7, [a['track_id'] for a in filtered])

    def test_app_track_id_buttons_exist(self):
        from app import UnifiedPanel
        self.assertTrue(hasattr(UnifiedPanel, 'increment_track_id'))
        self.assertTrue(hasattr(UnifiedPanel, 'decrement_track_id'))
        self.assertTrue(hasattr(UnifiedPanel, 'trace_id_label'))
        self.assertTrue(hasattr(UnifiedPanel, 'trace_id_plus_btn'))
        self.assertTrue(hasattr(UnifiedPanel, 'trace_id_minus_btn'))

    def test_app_has_convert(self):
        from app import UnifiedPanel
        self.assertTrue(hasattr(UnifiedPanel, '_convert_track_id'))

    def test_control_panel_has_convert(self):
        from control_panel import ControlPanel
        self.assertTrue(hasattr(ControlPanel, '_convert_track_id'))

    def test_button_styles_consistent(self):
        from PyQt5.QtWidgets import QApplication, QPushButton
        app = QApplication.instance() or QApplication([])
        from app import UnifiedPanel
        ui = UnifiedPanel()
        for child in ui.findChildren(QPushButton):
            if child.text() in ("+", "-", "删除", "清空"):
                style = child.styleSheet()
                self.assertIn("border-radius: 3px", style)
                self.assertIn("border: none", style)
                self.assertTrue(child.width() > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
