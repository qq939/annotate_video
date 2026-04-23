#!/usr/bin/env python3
"""控制面板"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, 
                             QMessageBox, QListWidget, QListWidgetItem, QFileDialog)
from PyQt5.QtCore import Qt, QTimer

class ControlPanel(QMainWindow):
    def __init__(self, temp_data_path=None):
        super().__init__()
        self.temp_data_path = Path(temp_data_path) if temp_data_path else Path("temp_data")
        self.viewer = None
        
        with open(self.temp_data_path / "annotations.json") as f:
            self.coco_data = json.load(f)
        self.video_info = self.coco_data['info']
        
        self.total_frames = len(self.coco_data['images'])
        self.conf_threshold = 0.5
        self.del_track_id_list = []
        self.del_points = []
        
        self.init_ui()
    
    def set_viewer(self, viewer):
        self.viewer = viewer
        self.viewer.update_display()
    
    def init_ui(self):
        self.setWindowTitle('控制面板')
        self.setGeometry(100, 100, 400, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)
        
        path_layout = QHBoxLayout()
        self.path_label = QLabel(f"当前: {self.temp_data_path}")
        open_btn = QPushButton("选择文件夹")
        open_btn.clicked.connect(self.select_folder)
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(open_btn)
        layout.addLayout(path_layout)
        
        layout.addWidget(QLabel(f"总帧数: {self.total_frames}"))
        
        frame_layout = QHBoxLayout()
        self.frame_label = QLabel(f"帧: 1/{self.total_frames}")
        frame_layout.addWidget(self.frame_label)
        
        prev_btn = QPushButton("上一帧")
        prev_btn.clicked.connect(self.prev_frame)
        frame_layout.addWidget(prev_btn)
        
        next_btn = QPushButton("下一帧")
        next_btn.clicked.connect(self.next_frame)
        frame_layout.addWidget(next_btn)
        layout.addLayout(frame_layout)
        
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        conf_layout.addWidget(self.conf_slider)
        layout.addLayout(conf_layout)
        
        play_btn = QPushButton("播放")
        play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(play_btn)
        self.play_btn = play_btn
        
        export_btn = QPushButton("导出视频")
        export_btn.clicked.connect(self.export_video)
        layout.addWidget(export_btn)
        
        clear_btn = QPushButton("清空删除点")
        clear_btn.clicked.connect(self.clear_del_points)
        layout.addWidget(clear_btn)
        
        layout.addWidget(QLabel("删除列表:"))
        del_list_layout = QHBoxLayout()
        self.del_list = QListWidget()
        del_list_layout.addWidget(self.del_list)
        
        del_btn_layout = QVBoxLayout()
        remove_btn = QPushButton("🗑️")
        remove_btn.setFixedSize(40, 40)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
        """)
        remove_btn.clicked.connect(self.remove_selected_del_point)
        del_btn_layout.addWidget(remove_btn)
        del_list_layout.addLayout(del_btn_layout)
        layout.addLayout(del_list_layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据文件夹", str(self.temp_data_path))
        if folder:
            self.temp_data_path = Path(folder)
            
            with open(self.temp_data_path / "annotations.json") as f:
                self.coco_data = json.load(f)
            self.video_info = self.coco_data['info']
            self.total_frames = len(self.coco_data['images'])
            
            self.path_label.setText(f"当前: {self.temp_data_path}")
            
            if self.viewer:
                self.viewer.temp_data_path = self.temp_data_path
                self.viewer.labels_dir = self.temp_data_path / "labels"
                self.viewer.frames_dir = self.temp_data_path / "frames"
                self.viewer.update_display()
    
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("播放")
        else:
            self.timer.start(100)
            self.play_btn.setText("暂停")
    
    def play_next(self):
        if self.viewer:
            self.viewer.play_next_frame()
            self.frame_label.setText(f"帧: {self.viewer.get_current_frame()+1}/{self.total_frames}")
    
    def prev_frame(self):
        if self.viewer:
            idx = (self.viewer.get_current_frame() - 1) % self.total_frames
            self.viewer.go_to_frame(idx)
            self.frame_label.setText(f"帧: {idx+1}/{self.total_frames}")
    
    def next_frame(self):
        if self.viewer:
            idx = (self.viewer.get_current_frame() + 1) % self.total_frames
            self.viewer.go_to_frame(idx)
            self.frame_label.setText(f"帧: {idx+1}/{self.total_frames}")
    
    def on_conf_change(self, value):
        self.conf_threshold = value / 100.0
        if self.viewer:
            self.viewer.update_display()
    
    def filter_annotations(self, annotations):
        return [ann for ann in annotations 
                if ann.get('confidence', 1.0) >= self.conf_threshold
                and ann.get('track_id', ann['id']) not in self.del_track_id_list]
    
    def apply_threshold_to_masks(self, frame, annotations, threshold):
        result_frame = frame.copy()
        if not annotations:
            return result_frame
        
        mask_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        
        for ann in annotations:
            polygon = ann['segmentation'][0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            color = mask_colors[ann['category_id'] % len(mask_colors)]
            
            cv2.fillPoly(result_frame, [pts], color)
            cv2.polylines(result_frame, [pts], True, (255, 255, 255), 2)
            
            bbox = ann['bbox']
            track_id = ann.get('track_id', ann['id'])
            conf = ann.get('confidence', 1.0)
            
            x, y = int(bbox[0]), int(bbox[1])
            cv2.putText(result_frame, f"ID:{track_id} {conf:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_frame
    
    def handle_click(self, x, y, frame_idx):
        print(f"点击: ({x}, {y}), 帧{frame_idx+1}")
        
        if not self.viewer:
            return
        
        frame, annotations = self.viewer.load_frame_data(frame_idx)
        filtered = self.filter_annotations(annotations)
        
        for ann in filtered:
            polygon = ann['segmentation'][0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            result = cv2.pointPolygonTest(pts, (float(x), float(y)), False)
            
            if result >= 0:
                track_id = ann.get('track_id', ann['id'])
                if track_id not in self.del_track_id_list:
                    self.del_track_id_list.append(track_id)
                    self.del_points.append({'x': x, 'y': y, 'frame_idx': frame_idx, 'track_id': track_id})
                    self.del_list.addItem(f"帧{frame_idx+1} ({x},{y}) ID:{track_id}")
                    print(f"删除track_id: {track_id}")
                    self.viewer.update_display()
                return
        
        print("未找到标注")
    
    def remove_selected_del_point(self):
        current_row = self.del_list.currentRow()
        if current_row >= 0:
            track_id = self.del_points[current_row]['track_id']
            print(f"删除: {track_id}")
            self.del_track_id_list.remove(track_id)
            self.del_points.pop(current_row)
            self.del_list.takeItem(current_row)
            if self.viewer:
                self.viewer.update_display()
    
    def clear_del_points(self):
        self.del_track_id_list.clear()
        self.del_points.clear()
        self.del_list.clear()
        if self.viewer:
            self.viewer.update_display()
    
    def export_video(self):
        output_path = Path("temp_data_post")
        output_path.mkdir(exist_ok=True)
        
        dst_path = Path("dst/output_annotated.mp4")
        dst_path.parent.mkdir(exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(dst_path), fourcc, self.video_info['fps'], 
                            (self.video_info['width'], self.video_info['height']))
        
        labels_dir = self.temp_data_path / "labels"
        frames_dir = self.temp_data_path / "frames"
        output_labels_dir = output_path / "labels"
        output_labels_dir.mkdir(exist_ok=True)
        output_frames_dir = output_path / "frames"
        output_frames_dir.mkdir(exist_ok=True)
        
        for i in range(self.total_frames):
            frame_path = str(frames_dir / f"frame_{i:06d}.jpg")
            frame = cv2.imread(frame_path)
            
            label_path = str(labels_dir / f"frame_{i:06d}.json")
            output_frame_path = str(output_frames_dir / f"frame_{i:06d}.jpg")
            cv2.imwrite(output_frame_path, frame)
            
            output_label_path = str(output_labels_dir / f"frame_{i:06d}.json")
            if Path(label_path).exists():
                with open(label_path) as f:
                    annotations = json.load(f)
                with open(output_label_path, 'w') as f:
                    json.dump(annotations, f)
                
                filtered = self.filter_annotations(annotations)
                frame = self.apply_threshold_to_masks(frame, filtered, self.conf_threshold)
            
            out.write(frame)
        
        with open(output_path / "annotations.json", 'w') as f:
            with open(self.temp_data_path / "annotations.json") as orig:
                json.dump(json.load(orig), f, indent=2)
        
        out.release()
        QMessageBox.information(self, "完成", f"视频已导出到: {dst_path}\n数据已保存到: {output_path}")

def main():
    app = QApplication(sys.argv)
    panel = ControlPanel("temp_data")
    panel.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
