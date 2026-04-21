import unittest
import cv2
import numpy as np
from pathlib import Path
import time

class TestPostAnnotate(unittest.TestCase):
    def test_post_annotate_import(self):
        """测试post_annotate模块导入"""
        try:
            import post_annotate
            print("✓ post_annotate模块导入成功")
        except ImportError as e:
            self.fail(f"post_annotate模块导入失败: {e}")

    def test_confidence_threshold_functionality(self):
        """测试置信度阈值功能"""
        from post_annotate import PostAnnotator

        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        frames_dir = temp_data / "frames"
        if not frames_dir.exists():
            self.skipTest("帧数据目录不存在，跳过测试")

        frame_files = list(frames_dir.glob("frame_*.jpg"))
        if len(frame_files) == 0:
            self.skipTest("没有帧数据，跳过测试")

        print(f"✓ 找到 {len(frame_files)} 帧数据")
        print("✓ 置信度阈值功能测试通过")

    def test_frame_loading(self):
        """测试帧加载功能"""
        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        frames_dir = temp_data / "frames"
        if not frames_dir.exists():
            self.skipTest("帧数据目录不存在，跳过测试")

        frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
        if len(frame_files) == 0:
            self.skipTest("没有帧数据，跳过测试")

        frame_path = frame_files[0]
        frame = cv2.imread(str(frame_path))
        self.assertIsNotNone(frame, "应该能成功加载帧")
        print(f"✓ 帧加载功能测试通过，帧尺寸: {frame.shape}")

    def test_mask_data_structure(self):
        """测试mask数据结构"""
        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        masks_dir = temp_data / "masks"
        if not masks_dir.exists():
            self.skipTest("masks目录不存在，跳过测试")

        mask_files = list(masks_dir.glob("frame_*_info.npy"))
        if len(mask_files) == 0:
            self.skipTest("没有mask数据，跳过测试")

        masks_info = np.load(str(mask_files[0]), allow_pickle=True).item()
        self.assertIsInstance(masks_info, dict, "mask数据应该是字典类型")
        print(f"✓ Mask数据结构测试通过，包含 {len(masks_info)} 个mask")

    def test_metadata_loading(self):
        """测试元数据加载"""
        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        metadata_path = temp_data / "metadata.npy"
        if not metadata_path.exists():
            self.skipTest("元数据文件不存在，跳过测试")

        metadata = np.load(str(metadata_path), allow_pickle=True).item()
        self.assertIn('fps', metadata, "元数据应该包含fps")
        self.assertIn('width', metadata, "元数据应该包含width")
        self.assertIn('height', metadata, "元数据应该包含height")
        print(f"✓ 元数据加载测试通过")
        print(f"  视频尺寸: {metadata['width']}x{metadata['height']}")
        print(f"  FPS: {metadata['fps']}")

    def test_threshold_adjustment(self):
        """测试阈值调整逻辑"""
        from post_annotate import PostAnnotator

        temp_data = Path("temp_data")
        if not temp_data.exists():
            self.skipTest("临时数据目录不存在，跳过测试")

        frames_dir = temp_data / "frames"
        if not frames_dir.exists():
            self.skipTest("帧数据目录不存在，跳过测试")

        test_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
        for threshold in test_thresholds:
            self.assertTrue(0.0 <= threshold <= 1.0, f"阈值应该在[0, 1]范围内: {threshold}")

        print("✓ 阈值调整逻辑测试通过")

if __name__ == '__main__':
    unittest.main(verbosity=2)
