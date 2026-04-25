#!/usr/bin/env python3
"""视频查看器 - 纯预览窗口，无缩放"""

import sys
from pathlib import Path

import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal


class VideoLabel(QLabel):
    """纯预览标签，仅显示图像，鼠标点击传递坐标"""
    point_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: black;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.point_clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)


class VideoViewer(QMainWindow):
    def __init__(self, temp_data_path, control_panel=None):
        super().__init__()
        self.temp_data_path = Path(temp_data_path)
        self.control_panel = control_panel

        with open(self.temp_data_path / "annotations.json") as f:
            self.coco_data = json.load(f)
        self.video_info = self.coco_data['info']

        self.video_width = int(self.video_info['width'])
        self.video_height = int(self.video_info['height'])

        self.labels_dir = self.temp_data_path / "labels"
        self.frames_dir = self.temp_data_path / "frames"

        self.current_frame_idx = 0
        self.total_frames = len(self.coco_data['images'])

        self.init_ui()
        self.update_display()

    def init_ui(self):
        self.setWindowTitle('视频预览')
        self.resize(self.video_width, self.video_height)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        central.setLayout(layout)

        self.image_label = VideoLabel()
        self.image_label.point_clicked.connect(self.on_click)
        layout.addWidget(self.image_label)

    def on_click(self, x, y):
        if self.control_panel:
            self.control_panel.handle_click(x, y, self.current_frame_idx)

    def load_frame_data(self, idx):
        frame_path = str(self.frames_dir / f"frame_{idx:06d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)

        label_path = str(self.labels_dir / f"frame_{idx:06d}.json")
        frame_annotations = []
        if Path(label_path).exists():
            with open(label_path) as f:
                frame_annotations = json.load(f)

        return frame, frame_annotations

    def update_display(self):
        frame, annotations = self.load_frame_data(self.current_frame_idx)

        if self.control_panel and hasattr(self.control_panel, 'filter_annotations'):
            filtered = self.control_panel.filter_annotations(annotations)
            annotated_frame = self.control_panel.apply_threshold_to_masks(
                frame, filtered, self.control_panel.conf_threshold
            )
        else:
            annotated_frame = frame

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_frame_rgb.shape
        qt_image = QImage(annotated_frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def play_next_frame(self):
        self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
        self.update_display()

    def go_to_frame(self, idx):
        self.current_frame_idx = idx % self.total_frames
        self.update_display()

    def get_current_frame(self):
        return self.current_frame_idx


def main():
    app = QApplication(sys.argv)

    from control_panel import ControlPanel
    control_panel = ControlPanel("temp_data")
    viewer = VideoViewer("temp_data", control_panel)

    control_panel.set_viewer(viewer)

    viewer.show()
    control_panel.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
