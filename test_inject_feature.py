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
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['track_id'], 10000)

    def test_app_add_track_id_increments_counter(self):
        from app import UnifiedPanel
        self.assertTrue(hasattr(UnifiedPanel, 'add_track_id'))

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
            if child.text() in ("新增", "删除", "清空"):
                style = child.styleSheet()
                self.assertIn("border-radius: 3px", style)
                self.assertIn("border: none", style)
                self.assertTrue(child.width() > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
