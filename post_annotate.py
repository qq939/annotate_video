# global参数
TEMP_DATA_DIR = "temp_data"  # 第8行：临时数据目录
DEFAULT_CONF_THRESHOLD = 0.5  # 第9行：默认置信度阈值
WINDOW_NAME = "后处理预览"  # 第10行：窗口名称

import cv2
import numpy as np
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

        self.metadata = np.load(self.temp_data_path / 'metadata.npy', allow_pickle=True).item()
        self.frames_dir = self.temp_data_path / "frames"
        self.masks_dir = self.temp_data_path / "masks"

        self.frame_files = sorted(list(self.frames_dir.glob("frame_*.jpg")))
        self.total_frames = len(self.frame_files)

        if self.total_frames == 0:
            print("错误：没有找到帧数据")
            sys.exit(1)

        print(f"加载了 {self.total_frames} 帧数据")
        print(f"视频信息: {self.metadata['width']}x{self.metadata['height']}, FPS: {self.metadata['fps']}")

        self.current_frame_idx = 0
        self.is_playing = False
        self.play_thread = None

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
            frame = np.zeros((self.metadata['height'], self.metadata['width'], 3), dtype=np.uint8)

        masks_info_path = self.masks_dir / f"frame_{idx:06d}_info.npy"
        masks_data = {}
        if masks_info_path.exists():
            masks_data = np.load(str(masks_info_path), allow_pickle=True).item()

        logits_path = self.masks_dir / f"frame_{idx:06d}_logits.npy"
        logits = None
        if logits_path.exists():
            logits = np.load(str(logits_path))

        return frame, masks_data, logits

    def apply_threshold_to_masks(self, frame, masks_data, logits, threshold):
        result_frame = frame.copy()

        if logits is None or len(logits) == 0:
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

        if isinstance(logits, np.ndarray) and len(logits) > 0:
            if logits.ndim == 3:
                for i in range(min(len(logits), len(mask_colors))):
                    mask = logits[i]
                    if mask.ndim > 2:
                        mask = mask[0]

                    mask_binary = (mask > threshold).astype(np.uint8) * 255

                    if mask_binary.shape != (height, width):
                        mask_binary = cv2.resize(mask_binary, (width, height))

                    color = mask_colors[i % len(mask_colors)]
                    colored_mask = np.zeros_like(frame)
                    colored_mask[:] = color

                    mask_bool = mask_binary > 0
                    result_frame[mask_bool] = cv2.addWeighted(
                        frame[mask_bool], 0.3,
                        colored_mask[mask_bool], 0.7, 0
                    )

                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        cv2.drawContours(result_frame, contours, -1, color, 2)

                        for contour in contours:
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.putText(result_frame, f"ID:{i}", (cx-20, cy),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result_frame

    def update_display(self):
        frame, masks_data, logits = self.load_frame_data(self.current_frame_idx)
        annotated_frame = self.apply_threshold_to_masks(frame, masks_data, logits, self.conf_threshold)

        info_text = f"帧: {self.current_frame_idx + 1}/{self.total_frames} | 置信度阈值: {self.conf_threshold:.2f}"
        annotated_frame = put_chinese_text(annotated_frame, info_text, (10, 30),
                                        font_size=16, color=(255, 255, 255))

        instructions = [
            "操作说明:",
            "1. 拖动进度条或按空格键播放/暂停",
            "2. 调整置信度阈值滑块",
            "3. 按 'e' 导出视频",
            "4. 按 'q' 退出"
        ]
        for i, text in enumerate(instructions):
            annotated_frame = put_chinese_text(annotated_frame, text, (10, 60 + i * 25),
                                            font_size=14, color=(200, 200, 200))

        cv2.imshow(WINDOW_NAME, annotated_frame)
        cv2.setTrackbarPos('进度条', WINDOW_NAME, self.current_frame_idx)
        cv2.setTrackbarPos('置信度阈值', WINDOW_NAME, int(self.conf_threshold * 100))

    def play_video(self):
        while self.is_playing:
            time.sleep(1.0 / self.metadata['fps'])

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
        fourcc = cv2.VideoWriter_fourcc(*self.metadata['fourcc'])
        fps = self.metadata['fps']
        width = self.metadata['width']
        height = self.metadata['height']

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for i in range(self.total_frames):
            if i % 30 == 0:
                print(f"正在导出帧: {i}/{self.total_frames}")

            frame, masks_data, logits = self.load_frame_data(i)
            annotated_frame = self.apply_threshold_to_masks(frame, masks_data, logits, self.conf_threshold)
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
