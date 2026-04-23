#!/usr/bin/env python3
"""视频查看器 - 支持缩放"""

import sys
from pathlib import Path

import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
from PyQt5.QtCore import pyqtSignal

class VideoLabel(QLabel):
    """可点击的标签"""
    point_clicked = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom_factor = 1.0
        self.video_width = 1280
        self.video_height = 720
    
    def set_zoom(self, factor):
        self.zoom_factor = factor
        self.update()
    
    def set_video_size(self, w, h):
        self.video_width = w
        self.video_height = h
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            self.point_clicked.emit(x, y)
        super().mousePressEvent(event)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.pixmap() and self.zoom_factor != 1.0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
            scaled_pixmap = self.pixmap().scaled(
                int(self.video_width * self.zoom_factor),
                int(self.video_height * self.zoom_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)

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
        
        self.zoom_factor = 1.0
        
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
        self.image_label.setStyleSheet("background: black;")
        layout.addWidget(self.image_label)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
    
    def on_click(self, display_x, display_y):
        if self.control_panel:
            scaled_w = int(self.video_width * self.zoom_factor)
            scaled_h = int(self.video_height * self.zoom_factor)
            
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            
            offset_x = (label_w - scaled_w) // 2
            offset_y = (label_h - scaled_h) // 2
            
            if offset_x <= display_x < offset_x + scaled_w and offset_y <= display_y < offset_y + scaled_h:
                video_x = int((display_x - offset_x) / self.zoom_factor)
                video_y = int((display_y - offset_y) / self.zoom_factor)
                
                self.control_panel.handle_click(video_x, video_y, self.current_frame_idx)
    
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
        
        if self.control_panel:
            filtered_annotations = self.control_panel.filter_annotations(annotations)
            annotated_frame = self.control_panel.apply_threshold_to_masks(frame, filtered_annotations, self.conf_threshold)
        else:
            annotated_frame = frame
        
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
    control_panel = ControlPanel("temp_data")
    viewer = VideoViewer("temp_data", control_panel)
    
    control_panel.set_viewer(viewer)
    
    viewer.show()
    control_panel.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
