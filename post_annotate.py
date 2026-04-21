# global参数
TEMP_DATA_DIR = "temp_data"  # 第8行：临时数据目录
DEFAULT_CONF_THRESHOLD = 0.5  # 第9行：默认置信度阈值

import cv2
import numpy as np
import json
import sys
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class PostAnnotatorWindow(QMainWindow):
    def __init__(self, output_video_path):
        super().__init__()
        self.output_video_path = output_video_path
        self.temp_data_path = Path(TEMP_DATA_DIR)
        self.conf_threshold = DEFAULT_CONF_THRESHOLD

        if not self.temp_data_path.exists():
            print(f"错误：临时数据目录不存在: {self.temp_data_path}")
            sys.exit(1)

        annotations_file = self.temp_data_path / "annotations.json"
        with open(str(annotations_file), 'r') as f:
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

        self.current_frame_idx = 0
        self.is_playing = False
        self.play_speed = 0.5

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)

        self.update_display()

    def init_ui(self):
        self.setWindowTitle('后处理预览 - 置信度阈值调整')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)

        info_layout = QHBoxLayout()
        self.info_label = QLabel(f"帧: 1/{self.total_frames}")
        info_layout.addWidget(self.info_label)
        self.count_label = QLabel(f"可见标注数: 0/0")
        info_layout.addWidget(self.count_label)
        self.threshold_label = QLabel(f"置信度阈值: {self.conf_threshold:.2f}")
        info_layout.addWidget(self.threshold_label)
        layout.addLayout(info_layout)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(DEFAULT_CONF_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)
        layout.addWidget(QLabel("置信度阈值:"))
        layout.addWidget(self.threshold_slider)

        button_layout = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        button_layout.addWidget(self.play_btn)

        self.export_btn = QPushButton("导出视频")
        self.export_btn.clicked.connect(self.export_video)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

    def on_threshold_change(self, value):
        self.conf_threshold = value / 100.0
        self.threshold_label.setText(f"置信度阈值: {self.conf_threshold:.2f}")
        self.update_display()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("暂停")
            interval = int(1000.0 / (self.video_info['fps'] * self.play_speed))
            self.timer.start(interval)
        else:
            self.play_btn.setText("播放")
            self.timer.stop()

    def play_next_frame(self):
        self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
        self.update_display()

    def load_frame_data(self, idx):
        frame_path = str(self.frames_dir / f"frame_{idx:06d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            frame = np.zeros((self.video_info['height'], self.video_info['width'], 3), dtype=np.uint8)

        label_path = str(self.labels_dir / f"frame_{idx:06d}.json")
        frame_annotations = []
        if Path(label_path).exists():
            with open(label_path, 'r') as f:
                frame_annotations = json.load(f)

        return frame, frame_annotations

    def apply_threshold_to_masks(self, frame, annotations, threshold):
        result_frame = frame.copy()

        if not annotations:
            return result_frame

        mask_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
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

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        visible_count = sum(1 for ann in annotations if ann.get('confidence', 1.0) >= self.conf_threshold)
        self.info_label.setText(f"帧: {self.current_frame_idx + 1}/{self.total_frames}")
        self.count_label.setText(f"可见标注数: {visible_count}/{len(annotations)}")

    def export_video(self):
        print(f"\n正在导出视频，使用置信度阈值: {self.conf_threshold:.2f}")

        output_path = Path(self.output_video_path)
        fourcc_str = cv2.VideoWriter_fourcc(*self.video_info['fourcc'])
        fps = self.video_info['fps']
        width = self.video_info['width']
        height = self.video_info['height']

        out = cv2.VideoWriter(str(output_path), fourcc_str, fps, (width, height))

        for i in range(self.total_frames):
            if i % 30 == 0:
                print(f"正在导出帧: {i}/{self.total_frames}")

            frame, annotations = self.load_frame_data(i)
            annotated_frame = self.apply_threshold_to_masks(frame, annotations, self.conf_threshold)
            out.write(annotated_frame)

        out.release()
        print(f"✓ 视频导出成功: {output_path}")
        QMessageBox.information(self, "导出成功", f"视频已导出到:\n{output_path}")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        output_video_path = sys.argv[1]
    else:
        output_video_path = "dst/output_annotated.mp4"

    print("\n" + "=" * 50)
    print("后处理预览程序 - PyQt5版本")
    print("=" * 50)
    frames_count = len(list(Path(TEMP_DATA_DIR).glob('frames/*.jpg'))) if Path(TEMP_DATA_DIR).exists() else 0
    print(f"总帧数: {frames_count}")
    print(f"置信度阈值: {DEFAULT_CONF_THRESHOLD:.2f}")
    print("=" * 50)

    window = PostAnnotatorWindow(output_video_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
