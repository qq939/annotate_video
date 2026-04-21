import unittest
import cv2
import numpy as np
from pathlib import Path
import json

class TestPostAnnotate(unittest.TestCase):
    def test_coco_annotation_format(self):
        """测试COCO格式标注功能"""
        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        annotations_path = temp_data / "annotations.json"
        if not annotations_path.exists():
            self.skipTest("COCO标注文件不存在，跳过测试")

        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        self.assertIn('info', coco_data, "COCO数据应该包含info字段")
        self.assertIn('images', coco_data, "COCO数据应该包含images字段")
        self.assertIn('annotations', coco_data, "COCO数据应该包含annotations字段")
        self.assertIn('categories', coco_data, "COCO数据应该包含categories字段")

        print(f"✓ COCO格式标注测试通过")
        print(f"  总帧数: {len(coco_data['images'])}")
        print(f"  总标注数: {len(coco_data['annotations'])}")

    def test_video_info_consistency(self):
        """测试视频参数一致性"""
        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        annotations_path = temp_data / "annotations.json"
        if not annotations_path.exists():
            self.skipTest("COCO标注文件不存在，跳过测试")

        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        info = coco_data['info']
        self.assertIn('fps', info, "视频信息应该包含fps")
        self.assertIn('width', info, "视频信息应该包含width")
        self.assertIn('height', info, "视频信息应该包含height")
        self.assertIn('fourcc', info, "视频信息应该包含fourcc")

        print(f"✓ 视频参数一致性测试通过")
        print(f"  尺寸: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']}")
        print(f"  编码: {info['fourcc']}")

    def test_frame_labels_format(self):
        """测试帧标注格式"""
        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        labels_dir = temp_data / "labels"
        if not labels_dir.exists():
            self.skipTest("labels目录不存在，跳过测试")

        label_files = list(labels_dir.glob("frame_*.json"))
        if len(label_files) == 0:
            self.skipTest("没有帧标注文件，跳过测试")

        with open(label_files[0], 'r') as f:
            frame_labels = json.load(f)

        if len(frame_labels) > 0:
            ann = frame_labels[0]
            self.assertIn('bbox', ann, "标注应该包含bbox")
            self.assertIn('segmentation', ann, "标注应该包含segmentation")
            self.assertIn('confidence', ann, "标注应该包含confidence")
            self.assertIn('id', ann, "标注应该包含id")
            self.assertIn('image_id', ann, "标注应该包含image_id")

        print(f"✓ 帧标注格式测试通过")
        print(f"  标注文件数: {len(label_files)}")
        print(f"  第一个标注包含bbox和segmentation")

if __name__ == '__main__':
    unittest.main(verbosity=2)
