#!/usr/bin/env python3
"""视频查看器 - 纯视图层，控制逻辑由 video_control.VideoController 提供"""

import sys
from pathlib import Path

import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import pyqtSignal

from video_control import VideoController


class VideoLabel(QLabel):
    point_clicked = pyqtSignal(int, int)
    bbox_drawn = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.video_width = 1280
        self.video_height = 720
        self.drawing_enabled = False
        self._drag_start = None
        self._drag_current = None

    def set_zoom(self, factor):
        self.zoom_factor = factor

    def set_video_size(self, w, h):
        self.video_width = w
        self.video_height = h

    def set_drawing_enabled(self, enabled):
        self.drawing_enabled = enabled
        self._drag_start = None
        self._drag_current = None
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing_enabled:
            self._drag_start = (event.x(), event.y())
            self._drag_current = (event.x(), event.y())
            return
        if event.button() == Qt.LeftButton:
            self.point_clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_enabled and self._drag_start is not None:
            self._drag_current = (event.x(), event.y())
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing_enabled and self._drag_start is not None:
            x1, y1 = self._drag_start
            x2, y2 = event.x(), event.y()
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.bbox_drawn.emit(x1, y1, x2, y2)
            self._drag_start = None
            self._drag_current = None
            self.update()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap() and self.zoom_factor != 1.0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            scaled = self.pixmap().scaled(
                int(self.video_width * self.zoom_factor),
                int(self.video_height * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()
        if self.drawing_enabled and self._drag_start is not None and self._drag_current is not None:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            x1, y1 = self._drag_start
            x2, y2 = self._drag_current
            painter.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            painter.end()


class VideoViewer(QMainWindow):
    video_clicked = pyqtSignal(int, int, int)

    def __init__(self, temp_data_path, controller=None):
        super().__init__()
        self.temp_data_path = Path(temp_data_path)
        self.controller = controller  # VideoController 实例

        with open(self.temp_data_path / "annotations.json") as f:
            self.coco_data = json.load(f)
        self.video_info = self.coco_data['info']

        self.video_width = self.video_info['width']
        self.video_height = self.video_info['height']

        self.labels_dir = self.temp_data_path / "labels"
        self.frames_dir = self.temp_data_path / "frames"

        self.current_frame_idx = 0
        self.total_frames = len(self.coco_data['images'])
        self.zoom_factor = 1.0

        self.prompt_bboxes = []
        self.drawing_mode = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('视频查看器')
        self.resize(int(self.video_width * 0.8), int(self.video_height * 0.8))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        self.image_label = VideoLabel()
        self.image_label.set_zoom(self.zoom_factor)
        self.image_label.set_video_size(self.video_width, self.video_height)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.point_clicked.connect(self.on_click)
        self.image_label.bbox_drawn.connect(self.on_bbox_drawn)
        self.image_label.setStyleSheet("background: black;")
        layout.addWidget(self.image_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)

    def set_controller(self, controller):
        self.controller = controller

    def stop_playback(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

    def on_click(self, display_x, display_y):
        scaled_w = int(self.video_width * self.zoom_factor)
        scaled_h = int(self.video_height * self.zoom_factor)
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2

        if offset_x <= display_x < offset_x + scaled_w and offset_y <= display_y < offset_y + scaled_h:
            video_x = int((display_x - offset_x) / self.zoom_factor)
            video_y = int((display_y - offset_y) / self.zoom_factor)
            self.video_clicked.emit(video_x, video_y, self.current_frame_idx)

    def on_bbox_drawn(self, display_x1, display_y1, display_x2, display_y2):
        scaled_w = int(self.video_width * self.zoom_factor)
        scaled_h = int(self.video_height * self.zoom_factor)
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        offset_x = (label_w - scaled_w) // 2
        offset_y = (label_h - scaled_h) // 2

        video_x1 = int((display_x1 - offset_x) / self.zoom_factor)
        video_y1 = int((display_y1 - offset_y) / self.zoom_factor)
        video_x2 = int((display_x2 - offset_x) / self.zoom_factor)
        video_y2 = int((display_y2 - offset_y) / self.zoom_factor)

        video_x1 = max(0, min(video_x1, self.video_width))
        video_y1 = max(0, min(video_y1, self.video_height))
        video_x2 = max(0, min(video_x2, self.video_width))
        video_y2 = max(0, min(video_y2, self.video_height))

        self.prompt_bboxes.append([video_x1, video_y1, video_x2, video_y2])
        self.update_display()

    def enable_bbox_drawing(self, enabled):
        self.drawing_mode = enabled
        self.image_label.set_drawing_enabled(enabled)

    def get_prompt_bboxes(self):
        return list(self.prompt_bboxes)

    def clear_prompt_bboxes(self):
        self.prompt_bboxes = []
        self.update_display()

    def set_zoom(self, factor):
        self.zoom_factor = factor
        self.image_label.set_zoom(factor)
        self.update_display()

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

        if self.controller:
            filtered = self.controller.filter_annotations(annotations)
            annotated_frame = self.controller.apply_threshold_to_masks(frame, filtered)
        else:
            annotated_frame = frame

        if self.controller:
            for tp in self.controller.get_track_id_points():
                if tp.get('frame_idx') == self.current_frame_idx:
                    cv2.circle(annotated_frame, (tp['x'], tp['y']), 6, (0, 255, 0), -1)
                    cv2.circle(annotated_frame, (tp['x'], tp['y']), 6, (0, 0, 0), 2)

        purple = (128, 0, 128)
        purple_count = 0
        for ann in annotations:
            if ann.get('track_id') == 9999:
                purple_count += 1
                polygon = ann.get('segmentation')
                bbox = ann.get('bbox')
                if polygon:
                    pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(annotated_frame, [pts], True, purple, 2)
                if bbox:
                    x, y = int(bbox[0]), int(bbox[1])
                    w, h = int(bbox[2]), int(bbox[3])
                    conf = ann.get('confidence', 1.0)
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), purple, 2)
                    cv2.putText(annotated_frame, f"9999 {conf:.2f}", (x, max(10, y - 5)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple, 2)
        if purple_count > 0:
            print(f"[DEBUG] 帧 {self.current_frame_idx} 有 {purple_count} 个 9999 标注，绘制紫色")
        elif self.controller and self.controller.track_ids_to_9999:
            print(f"[DEBUG] track_ids_to_9999={self.controller.track_ids_to_9999} 但帧 {self.current_frame_idx} 的标注无 track_id=9999")

        for bbox in self.prompt_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"prompt {self.prompt_bboxes.index(bbox) + 1}"
            cv2.putText(annotated_frame, label, (x1, max(10, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            int(w * self.zoom_factor),
            int(h * self.zoom_factor),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

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
    controller = VideoController()
    control_panel = ControlPanel("temp_data", controller=controller)
    viewer = VideoViewer("temp_data", controller=controller)

    control_panel.set_viewer(viewer)
    viewer.show()
    control_panel.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
