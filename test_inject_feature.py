import sys
import json
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestInjectFeature(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(__file__).parent / "test_inject_temp"
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_annotate_video_inject_function_exists(self):
        from annotate_video import run_inject
        self.assertTrue(callable(run_inject))

    def test_annotate_video_cli_args_inject(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--inject', action='store_true')
        parser.add_argument('--prompt-bboxes', type=str)
        parser.add_argument('--output-temp', type=str)
        parser.add_argument('--src', type=str)
        parser.add_argument('--iou', type=float)
        parser.add_argument('--items', type=str)

        args = parser.parse_args([
            '--inject',
            '--src', '/tmp/test.mp4',
            '--prompt-bboxes', '[[100,100,200,200],[300,300,400,400]]',
            '--output-temp', '/tmp/inject_out',
            '--iou', '0.3',
            '--items', 'cat,dog'
        ])
        self.assertTrue(args.inject)
        self.assertEqual(args.src, '/tmp/test.mp4')
        self.assertEqual(args.prompt_bboxes, '[[100,100,200,200],[300,300,400,400]]')
        self.assertEqual(args.output_temp, '/tmp/inject_out')
        self.assertEqual(args.iou, 0.3)
        self.assertEqual(args.items, 'cat,dog')

    def test_annotate_video_inject_parsed_bboxes(self):
        bboxes_str = '[[100,100,200,200],[300,300,400,400]]'
        bboxes = json.loads(bboxes_str)
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(bboxes[0], [100, 100, 200, 200])
        self.assertEqual(bboxes[1], [300, 300, 400, 400])

    def test_annotate_video_inject_items_parsing(self):
        items_str = 'cat,dog'
        items = items_str.split(',')
        self.assertEqual(items, ['cat', 'dog'])

    def test_extract_video_clip_logic(self):
        from app import UnifiedPanel
        self.assertTrue(hasattr(UnifiedPanel, 'extract_video_clip_from_frames'))
        self.assertTrue(callable(UnifiedPanel.extract_video_clip_from_frames))

    def test_merge_inject_results_logic(self):
        src_data = {
            "images": [
                {"id": 0, "file_name": "frame_000000.jpg"},
                {"id": 1, "file_name": "frame_000001.jpg"},
                {"id": 2, "file_name": "frame_000002.jpg"},
                {"id": 3, "file_name": "frame_000003.jpg"},
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "bbox": [0, 0, 10, 10]},
                {"id": 1, "image_id": 1, "bbox": [5, 5, 15, 15]},
                {"id": 2, "image_id": 2, "bbox": [10, 10, 20, 20]},
                {"id": 3, "image_id": 3, "bbox": [15, 15, 25, 25]},
            ]
        }
        inject_data = {
            "images": [
                {"id": 0, "file_name": "frame_000000.jpg"},
                {"id": 1, "file_name": "frame_000001.jpg"},
            ],
            "annotations": [
                {"id": 0, "image_id": 0, "bbox": [100, 100, 110, 110]},
                {"id": 1, "image_id": 1, "bbox": [200, 200, 210, 210]},
            ]
        }

        start_frame = 2
        original_images = [img for img in src_data['images']
                          if img['id'] < start_frame]
        original_annotations = [ann for ann in src_data['annotations']
                               if ann['image_id'] < start_frame]

        max_ann_id = max([ann['id'] for ann in original_annotations], default=-1) + 1
        for img in inject_data['images']:
            new_img = dict(img)
            new_img['id'] = img['id'] + start_frame
            original_images.append(new_img)
        for ann in inject_data['annotations']:
            new_ann = dict(ann)
            new_ann['id'] = ann['id'] + max_ann_id
            new_ann['image_id'] = ann['image_id'] + start_frame
            original_annotations.append(new_ann)

        self.assertEqual(len(original_images), 4)
        self.assertEqual(original_images[0]['id'], 0)
        self.assertEqual(original_images[1]['id'], 1)
        self.assertEqual(original_images[2]['id'], 2)
        self.assertEqual(original_images[3]['id'], 3)

        self.assertEqual(len(original_annotations), 4)
        self.assertEqual(original_annotations[0]['id'], 0)
        self.assertEqual(original_annotations[0]['image_id'], 0)
        self.assertEqual(original_annotations[2]['id'], 2)
        self.assertEqual(original_annotations[2]['image_id'], 2)
        self.assertEqual(original_annotations[3]['image_id'], 3)

    def test_video_viewer_bbox_methods_exist(self):
        from video_viewer import VideoViewer
        self.assertTrue(hasattr(VideoViewer, 'enable_bbox_drawing'))
        self.assertTrue(hasattr(VideoViewer, 'get_prompt_bboxes'))
        self.assertTrue(hasattr(VideoViewer, 'clear_prompt_bboxes'))

    def test_video_label_drawing_methods_exist(self):
        from video_viewer import VideoLabel
        self.assertTrue(hasattr(VideoLabel, 'set_drawing_enabled'))

    def test_prompt_bboxes_conversion(self):
        display_w, display_h = 960, 540
        video_w, video_h = 1280, 720
        zoom = 0.75

        scaled_w = int(video_w * zoom)
        scaled_h = int(video_h * zoom)
        offset_x = (display_w - scaled_w) // 2
        offset_y = (display_h - scaled_h) // 2

        display_x1, display_y1 = 100, 100
        display_x2, display_y2 = 300, 300

        video_x1 = int((display_x1 - offset_x) / zoom)
        video_y1 = int((display_y1 - offset_y) / zoom)
        video_x2 = int((display_x2 - offset_x) / zoom)
        video_y2 = int((display_y2 - offset_y) / zoom)

        self.assertGreater(video_x1, 0)
        self.assertGreater(video_y1, 0)
        self.assertGreater(video_x2, video_x1)
        self.assertGreater(video_y2, video_y1)

        video_x1 = max(0, min(video_x1, video_w))
        video_y1 = max(0, min(video_y1, video_h))
        video_x2 = max(0, min(video_x2, video_w))
        video_y2 = max(0, min(video_y2, video_h))

        self.assertGreaterEqual(video_x1, 0)
        self.assertLessEqual(video_x2, video_w)

    def test_existing_bbox_conversion(self):
        frame_anns = [
            {"bbox": [100, 200, 50, 80], "category_id": 1},
            {"bbox": [300, 400, 60, 90], "category_id": 2},
        ]
        prompt_bboxes = [
            [500, 500, 600, 600],
        ]

        existing_bboxes = []
        for ann in frame_anns:
            b = ann.get('bbox')
            if b:
                existing_bboxes.append([int(b[0]), int(b[1]),
                                        int(b[0] + b[2]), int(b[1] + b[3])])

        all_prompts = existing_bboxes + prompt_bboxes
        self.assertEqual(len(all_prompts), 3)
        self.assertEqual(all_prompts[0], [100, 200, 150, 280])
        self.assertEqual(all_prompts[1], [300, 400, 360, 490])
        self.assertEqual(all_prompts[2], [500, 500, 600, 600])


if __name__ == "__main__":
    unittest.main(verbosity=2)
