# global参数
TEMP_DATA_DIR = "temp_data"  # 第8行：临时数据目录
DEFAULT_CONF_THRESHOLD = 0.5  # 第9行：默认置信度阈值
WINDOW_NAME = "后处理预览"  # 第10行：窗口名称

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import threading
import time

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """在图像上绘制中文文本（使用UTF-8编码）"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", font_size)
        except:
            font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class PostAnnotator:
    def __init__(self, output_video_path):
        self.output_video_path = output_video_path
        self.temp_data_path = Path(TEMP_DATA_DIR)
        self.conf_threshold = DEFAULT_CONF_THRESHOLD

        if not self.temp_data_path.exists():
            print(f"错误：临时数据目录不存在: {self.temp_data_path}")
            sys.exit(1)

        with open(self.temp_data_path / 'annotations.json', 'r') as f:
            self.coco_data = json.load(f)

        self.video_info = self.coco_data['info']
        self.frames_dir = self.temp_data_path / "frames"
        self.labels_dir = self.temp_data_path / "labels"

        self.frame_files = sorted(list(self.frames_dir.glob("frame_*.jpg")))
        self.total_frames = len(self.frame_files)

        if self.total_frames == 0:
            print("错误：没有找到帧数据")
            sys.exit(1)

        print(f"加载了 {self.total_frames} 帧数据")
        print(f"视频信息: {self.video_info['width']}x{self.video_info['height']}, FPS: {self.video_info['fps']}")
        print(f"总标注数: {len(self.coco_data['annotations'])}")

        self.current_frame_idx = 0
        self.is_playing = False
        self.play_thread = None

        self.annotation_map = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotation_map:
                self.annotation_map[img_id] = []
            self.annotation_map[img_id].append(ann)

        cv2.namedWindow(WINDOW_NAME)
        cv2.createTrackbar('进度条', WINDOW_NAME, 0, self.total_frames - 1, self.on_trackbar_change)
        cv2.createTrackbar('置信度阈值', WINDOW_NAME, int(DEFAULT_CONF_THRESHOLD * 100), 100, self.on_conf_change)

        self.update_display()

    def on_trackbar_change(self, pos):
        self.current_frame_idx = pos
        self.is_playing = False
        self.update_display()

    def on_conf_change(self, pos):
        self.conf_threshold = pos / 100.0
        self.update_display()

    def load_frame_data(self, idx):
        frame_path = self.frames_dir / f"frame_{idx:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            frame = np.zeros((self.video_info['height'], self.video_info['width'], 3), dtype=np.uint8)

        label_path = self.labels_dir / f"frame_{idx:06d}.json"
        frame_annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                frame_annotations = json.load(f)

        return frame, frame_annotations

    def apply_threshold_to_masks(self, frame, annotations, threshold):
        result_frame = frame.copy()

        if not annotations:
            return result_frame

        height, width = frame.shape[:2]
        mask_colors = [
            (255, 0, 0),      # 蓝色
            (0, 255, 0),      # 绿色
            (0, 0, 255),      # 红色
            (255, 255, 0),    # 青色
            (255, 0, 255),    # 紫色
            (0, 255, 255),    # 黄色
            (255, 128, 0),    # 橙色
            (128, 0, 255),    # 紫红色
        ]

        for ann in annotations:
            if ann.get('confidence', 1.0) < threshold:
                continue

            polygon = ann['segmentation'][0]
            bbox = ann['bbox']
            category_id = ann['category_id']

            color = mask_colors[category_id % len(mask_colors)]

            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)

            overlay = result_frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.4, result_frame, 0.6, 0, result_frame)

            cv2.polylines(result_frame, [pts], True, color, 2)

            x, y, w, h = bbox
            label = f"ID:{ann['id']} {ann.get('confidence', 0):.2f}"
            cv2.putText(result_frame, label, (int(x), int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result_frame

    def update_display(self):
        frame, annotations = self.load_frame_data(self.current_frame_idx)
        annotated_frame = self.apply_threshold_to_masks(frame, annotations, self.conf_threshold)

        info_text = f"帧: {self.current_frame_idx + 1}/{self.total_frames} | 置信度阈值: {self.conf_threshold:.2f}"
        annotated_frame = put_chinese_text(annotated_frame, info_text, (10, 30),
                                        font_size=16, color=(255, 255, 255))

        visible_count = sum(1 for ann in annotations if ann.get('confidence', 1.0) >= self.conf_threshold)
        count_text = f"可见标注数: {visible_count}/{len(annotations)}"
        annotated_frame = put_chinese_text(annotated_frame, count_text, (10, 55),
                                        font_size=14, color=(200, 200, 200))

        instructions = [
            "操作说明:",
            "1. 拖动进度条或按空格键播放/暂停",
            "2. 调整置信度阈值滑块实时生效",
            "3. 按 'e' 导出视频",
            "4. 按 'q' 退出"
        ]
        for i, text in enumerate(instructions):
            annotated_frame = put_chinese_text(annotated_frame, text, (10, 80 + i * 25),
                                            font_size=14, color=(200, 200, 200))

        cv2.imshow(WINDOW_NAME, annotated_frame)
        cv2.setTrackbarPos('进度条', WINDOW_NAME, self.current_frame_idx)
        cv2.setTrackbarPos('置信度阈值', WINDOW_NAME, int(self.conf_threshold * 100))

    def play_video(self):
        while self.is_playing:
            time.sleep(1.0 / self.video_info['fps'])

            if self.is_playing:
                self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
                self.update_display()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing and (self.play_thread is None or not self.play_thread.is_alive()):
            self.play_thread = threading.Thread(target=self.play_video, daemon=True)
            self.play_thread.start()

    def export_video(self):
        print("\n正在导出视频...")
        print(f"使用置信度阈值: {self.conf_threshold:.2f}")

        output_path = Path(self.output_video_path)
        fourcc = cv2.VideoWriter_fourcc(*self.video_info['fourcc'])
        fps = self.video_info['fps']
        width = self.video_info['width']
        height = self.video_info['height']

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for i in range(self.total_frames):
            if i % 30 == 0:
                print(f"正在导出帧: {i}/{self.total_frames}")

            frame, annotations = self.load_frame_data(i)
            annotated_frame = self.apply_threshold_to_masks(frame, annotations, self.conf_threshold)
            out.write(annotated_frame)

        out.release()
        print(f"✓ 视频导出成功: {output_path}")
        print(f"✓ 使用置信度阈值: {self.conf_threshold:.2f}")

    def run(self):
        print("\n" + "=" * 50)
        print("后处理预览程序")
        print("=" * 50)
        print(f"总帧数: {self.total_frames}")
        print(f"置信度阈值: {self.conf_threshold:.2f}")
        print("=" * 50)

        while True:
            key = cv2.waitKey(100) & 0xFF

            if key == ord('q'):
                print("\n退出程序")
                break
            elif key == ord(' '):
                self.toggle_play()
            elif key == ord('e'):
                self.export_video()
            elif key == ord('+') or key == ord('='):
                self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
                self.update_display()
            elif key == ord('-'):
                self.conf_threshold = max(0.0, self.conf_threshold - 0.05)
                self.update_display()
            elif key == ord('j'):
                self.current_frame_idx = max(0, self.current_frame_idx - 10)
                self.update_display()
            elif key == ord('k'):
                self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 10)
                self.update_display()
            elif key == ord('g'):
                self.current_frame_idx = 0
                self.update_display()
            elif key == ord('G'):
                self.current_frame_idx = self.total_frames - 1
                self.update_display()

        cv2.destroyAllWindows()

def main():
    if len(sys.argv) > 1:
        output_video_path = sys.argv[1]
    else:
        print("用法: python3 post_annotate.py <输出视频路径>")
        print("将使用临时数据目录中的数据进行处理")
        output_video_path = "dst/output_annotated.mp4"

    app = PostAnnotator(output_video_path)
    app.run()

if __name__ == "__main__":
    main()
