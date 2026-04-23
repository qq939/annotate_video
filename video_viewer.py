#!/usr/bin/env python3
"""视频查看器 - 只显示视频画面"""

import sys
from pathlib import Path

import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
from PyQt5.QtCore import pyqtSignal

class VideoLabel(QLabel):
    """可点击的标签"""
    point_clicked = pyqtSignal(int, int)  # x, y坐标
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.point_clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)
    
    def paintEvent(self, event):
        super().paintEvent(event)

class VideoViewer(QMainWindow):
    def __init__(self, temp_data_path, control_panel=None):
        super().__init__()
        self.temp_data_path = Path(temp_data_path)
        self.control_panel = control_panel
        
        with open(self.temp_data_path / "annotations.json") as f:
            self.coco_data = json.load(f)
        self.video_info = self.coco_data['info']
        
        self.labels_dir = self.temp_data_path / "labels"
        self.frames_dir = self.temp_data_path / "frames"
        
        self.current_frame_idx = 0
        self.total_frames = len(self.coco_data['images'])
        self.conf_threshold = 0.5
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('视频查看器')
        self.setGeometry(100, 100, self.video_info['width'], self.video_info['height'])
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)
        
        self.image_label = VideoLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.point_clicked.connect(self.on_click)
        layout.addWidget(self.image_label)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
        
    def on_click(self, x, y):
        if self.control_panel:
            self.control_panel.handle_click(x, y, self.current_frame_idx)
    
    def load_frame_data(self, idx):
        frame_path = str(self.frames_dir / f"frame_{idx:06d}.jpg")
        frame = cv2.imread(frame_path)
        if frame is None:
            frame = np.zeros((self.video_info['height'], self.video_info['width'], 3), dtype=np.uint8)
        
        label_path = str(self.labels_dir / f"frame_{idx:06d}.json")
        frame_annotations = []
        if Path(label_path).exists():
            with open(label_path) as f:
                frame_annotations = json.load(f)
        
        return frame, frame_annotations
    
    def update_display(self):
        frame, annotations = self.load_frame_data(self.current_frame_idx)
        
        if self.control_panel:
            filtered_annotations = self.control_panel.filter_annotations(annotations)
            annotated_frame = self.control_panel.apply_threshold_to_masks(frame, filtered_annotations, self.conf_threshold)
        else:
            annotated_frame = frame
        
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = annotated_frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(annotated_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
