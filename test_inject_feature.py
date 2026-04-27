import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestTrackIdFeature(unittest.TestCase):

    def test_track_id_point_has_assigned_id(self):
        from video_control import VideoController
        c = VideoController()
        c.add_track_id_point(10, 20, 5, 7)
        pt = c.track_id_points[0]
        self.assertEqual(pt['track_id'], 7)
        self.assertIsNone(pt['assigned_id'])

    def test_assign_next_track_id(self):
        from video_control import VideoController
        import tempfile
        c = VideoController()
        c.add_track_id_point(10, 20, 0, 7)
        c.add_track_id_point(30, 40, 1, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = Path(tmpdir)
            ann1 = [{'track_id': 7, 'bbox': [0,0,10,10]}]
            ann2 = [{'track_id': 7, 'bbox': [0,0,10,10]}]
            ann3 = [{'track_id': 8, 'bbox': [0,0,10,10]}]
            (labels_dir / 'frame_000000.json').write_text(json.dumps(ann1))
            (labels_dir / 'frame_000001.json').write_text(json.dumps(ann2))
            (labels_dir / 'frame_000002.json').write_text(json.dumps(ann3))

            new_id = c.assign_next_track_id(0, labels_dir)
            self.assertEqual(new_id, 9999)
            self.assertEqual(c.track_id_points[0]['assigned_id'], 9999)

            data1 = json.loads((labels_dir / 'frame_000000.json').read_text())
            data2 = json.loads((labels_dir / 'frame_000001.json').read_text())
            data3 = json.loads((labels_dir / 'frame_000002.json').read_text())
            self.assertEqual(data1[0]['track_id'], 9999)
            self.assertEqual(data2[0]['track_id'], 9999)
            self.assertEqual(data3[0]['track_id'], 8)

    def test_revert_track_id(self):
        from video_control import VideoController
        import tempfile
        c = VideoController()
        c.add_track_id_point(10, 20, 0, 7)

        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = Path(tmpdir)
            ann1 = [{'track_id': 7, 'bbox': [0,0,10,10]}]
            (labels_dir / 'frame_000000.json').write_text(json.dumps(ann1))

            c.assign_next_track_id(0, labels_dir)
            data_before = json.loads((labels_dir / 'frame_000000.json').read_text())
            self.assertEqual(data_before[0]['track_id'], 9999)

            c.revert_track_id(0, labels_dir)
            data_after = json.loads((labels_dir / 'frame_000000.json').read_text())
            self.assertEqual(data_after[0]['track_id'], 7)
            self.assertIsNone(c.track_id_points[0]['assigned_id'])

    def test_purple_render_threshold(self):
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

    def test_app_has_add_track_id(self):
        from app import UnifiedPanel
        self.assertTrue(hasattr(UnifiedPanel, 'add_track_id'))
        self.assertTrue(callable(UnifiedPanel.add_track_id))

    def test_app_has_refresh_list(self):
        from app import UnifiedPanel
        self.assertTrue(hasattr(UnifiedPanel, '_refresh_track_id_list'))

    def test_control_panel_has_refresh_list(self):
        from control_panel import ControlPanel
        self.assertTrue(hasattr(ControlPanel, '_refresh_list'))

    def test_controller_has_assign_and_revert(self):
        from video_control import VideoController
        c = VideoController()
        self.assertTrue(callable(c.assign_next_track_id))
        self.assertTrue(callable(c.revert_track_id))


if __name__ == "__main__":
    unittest.main(verbosity=2)
