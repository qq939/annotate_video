#!/usr/bin/env python3
"""视频查看器 - 预览窗口（无缩放）"""

import sys
from pathlib import Path

import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QListWidget, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter

class VideoLabel(QLabel):
    point_clicked = pyqtSignal(int, int)

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

        self.video_width = self.video_info['width']
        self.video_height = self.video_info['height']

        self.labels_dir = self.temp_data_path / "labels"
        self.frames_dir = self.temp_data_path / "frames"

        self.current_frame_idx = 0
        self.total_frames = len(self.coco_data['images'])
        self.conf_threshold = 0.5
        self.alpha = 0.5
        self.category_name = "Detect"

        self.fences = []
        self.del_points = []
        self.max_fences = 3

        self.is_playing = False
        self.is_backward = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('视频查看器')
        self.resize(int(self.video_width * 0.8), int(self.video_height * 0.8))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        self.image_label = VideoLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.point_clicked.connect(self.on_click)
        self.image_label.setStyleSheet("background: black;")
        layout.addWidget(self.image_label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)

        self.update_display()

    def on_click(self, display_x, display_y):
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        offset_x = (label_w - self.video_width) // 2
        offset_y = (label_h - self.video_height) // 2

        if offset_x <= display_x < offset_x + self.video_width and offset_y <= display_y < offset_y + self.video_height:
            video_x = display_x - offset_x
            video_y = display_y - offset_y
            self.handle_click(video_x, video_y, self.current_frame_idx)

    def handle_click(self, x, y, frame_idx):
        for fence in self.fences:
            if fence.get('mode'):
                fence['points'].append((int(x), int(y)))
                print(f"围栏点: {len(fence['points'])}")
                self.update_display()
                return

        frame, annotations = self.load_frame_data(frame_idx)
        filtered = self.filter_annotations(annotations)

        for ann in filtered:
            polygon = ann.get('segmentation')
            if not polygon:
                continue
            pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                track_id = ann.get('track_id', ann.get('id', 0))
                self.del_points.append({'x': x, 'y': y, 'frame_idx': frame_idx, 'track_id': track_id})
                print(f"删除track_id: {track_id}")
                self.update_display()
                return

        print("未找到标注")

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

    def filter_annotations(self, annotations):
        if not annotations:
            return []
        deleted_ids = set(dp['track_id'] for dp in self.del_points)
        filtered = []

        fence_pts_list = []
        for fence in self.fences:
            if len(fence['points']) >= 3:
                fence_pts_list.append(np.array(fence['points'], dtype=np.int32))

        for ann in annotations:
            track_id = ann.get('track_id', ann.get('id', 0))
            conf = ann.get('confidence', 1.0)

            if conf < self.conf_threshold:
                continue
            if track_id in deleted_ids:
                continue

            if fence_pts_list:
                bbox = ann.get('bbox')
                if bbox:
                    cx = bbox[0] + bbox[2] / 2
                    cy = bbox[1] + bbox[3] / 2
                    inside_any = False
                    for fence_pts in fence_pts_list:
                        if cv2.pointPolygonTest(fence_pts, (cx, cy), False) >= 0:
                            inside_any = True
                            break
                    if not inside_any:
                        continue

            filtered.append(ann)

        return filtered

    def apply_threshold_to_masks(self, frame, annotations):
        result_frame = frame.copy()
        mask_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        if not annotations:
            return result_frame

        for ann in annotations:
            polygon = ann.get('segmentation')
            bbox = ann.get('bbox')
            if not bbox:
                continue

            color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
            category = ann.get('category', ann.get('category_id', 0))
            conf = ann.get('confidence', 1.0)

            if polygon:
                pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                overlay = result_frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(result_frame, 1 - self.alpha, overlay, self.alpha, 0, result_frame)
                cv2.polylines(result_frame, [pts], True, (255, 255, 255), 2)

            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_frame, f"{category} {conf:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        fence_colors = [(0, 255, 0), (255, 165, 0), (255, 0, 255)]
        for i, fence in enumerate(self.fences):
            if len(fence['points']) >= 3:
                color = fence_colors[i % len(fence_colors)]
                pts = np.array(fence['points'], dtype=np.int32)
                pts_array = pts.reshape((-1, 1, 2))
                cv2.polylines(result_frame, [pts_array], True, color, 3)
                for pt in fence['points']:
                    cv2.circle(result_frame, pt, 5, color, -1)

        return result_frame

    def update_display(self):
        frame, annotations = self.load_frame_data(self.current_frame_idx)
        filtered = self.filter_annotations(annotations)
        annotated_frame = self.apply_threshold_to_masks(frame, filtered)

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def play_next(self):
        if self.is_backward:
            self.current_frame_idx = (self.current_frame_idx - 1) % self.total_frames
        else:
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
    control_panel.show()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
