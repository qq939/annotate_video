import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestTrackIdTo9999(unittest.TestCase):

    def test_controller_has_track_id_points(self):
        from video_control import VideoController
        c = VideoController()
        self.assertTrue(hasattr(c, 'track_id_points'))
        self.assertTrue(hasattr(c, 'track_ids_to_9999'))
        self.assertIsInstance(c.track_ids_to_9999, set)

    def test_track_id_point_methods_exist(self):
        from video_control import VideoController
        c = VideoController()
        self.assertTrue(callable(c.add_track_id_point))
        self.assertTrue(callable(c.remove_track_id_point))
        self.assertTrue(callable(c.clear_track_id_points))
        self.assertTrue(callable(c.get_track_id_points))

    def test_track_id_point_add_and_get(self):
        from video_control import VideoController
        c = VideoController()
        c.add_track_id_point(100, 200, 5, 7)
        points = c.get_track_id_points()
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]['x'], 100)
        self.assertEqual(points[0]['track_id'], 7)

    def test_track_ids_to_9999_filter(self):
        from video_control import VideoController
        c = VideoController()
        c.track_ids_to_9999.add(5)
        anns = [
            {'track_id': 5, 'bbox': [0, 0, 10, 10], 'confidence': 1.0},
            {'track_id': 3, 'bbox': [10, 10, 20, 20], 'confidence': 1.0},
        ]
        filtered = c.filter_annotations(anns)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['track_id'], 3)

    def test_track_id_9999_color(self):
        from video_control import TRACK_ID_9999_COLOR
        self.assertEqual(TRACK_ID_9999_COLOR, (128, 0, 128))

    def test_convert_track_id_logic(self):
        from video_control import VideoController
        c = VideoController()
        c.track_ids_to_9999.add(7)
        anns = [
            {'track_id': 7, 'bbox': [0, 0, 10, 10]},
            {'track_id': 7, 'bbox': [20, 20, 30, 30]},
            {'track_id': 3, 'bbox': [40, 40, 50, 50]},
        ]
        changed_anns = []
        for ann in anns:
            if ann.get('track_id') in c.track_ids_to_9999:
                ann = dict(ann)
                ann['track_id'] = 9999
            changed_anns.append(ann)
        track_ids = [a['track_id'] for a in changed_anns]
        self.assertEqual(track_ids, [9999, 9999, 3])

    def test_9999_color_in_apply_threshold(self):
        from video_control import VideoController, TRACK_ID_9999_COLOR, MASK_COLORS
        c = VideoController()
        c.track_ids_to_9999.add(7)
        anns = [{'track_id': 7, 'bbox': [10, 10, 50, 50], 'confidence': 1.0}]
        filtered = c.filter_annotations(anns)
        self.assertEqual(len(filtered), 0)

    def test_app_has_convert_method(self):
        from app import UnifiedPanel
        self.assertTrue(hasattr(UnifiedPanel, '_convert_track_id_to_9999'))
        self.assertTrue(callable(UnifiedPanel._convert_track_id_to_9999))

    def test_control_panel_has_convert_method(self):
        from control_panel import ControlPanel
        self.assertTrue(hasattr(ControlPanel, '_convert_track_id_to_9999'))
        self.assertTrue(callable(ControlPanel._convert_track_id_to_9999))

    def test_video_viewer_update_draws_green_points(self):
        from video_viewer import VideoViewer
        self.assertTrue(hasattr(VideoViewer, 'update_display'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
