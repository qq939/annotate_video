#!/usr/bin/env python3
"""控制面板 - 三道过滤"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QMessageBox, QListWidget, QFileDialog, QLineEdit
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
        self.del_points = []
        self.fence_points = []
        self.fence_mode = False
        
        self.init_ui()
    
    def set_viewer(self, viewer):
        self.viewer = viewer
        self.viewer.update_display()
    
    def init_ui(self):
        self.setWindowTitle('控制面板')
        self.setGeometry(100, 100, 400, 700)
        
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
        
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("缩放:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_change)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        zoom_layout.addWidget(self.zoom_label)
        layout.addLayout(zoom_layout)
        
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("类别名称:"))
        self.category_input = QLineEdit("Detect")
        category_layout.addWidget(self.category_input)
        layout.addLayout(category_layout)
        
        fence_layout = QHBoxLayout()
        self.fence_btn = QPushButton("绘制围栏")
        self.fence_btn.clicked.connect(self.toggle_fence_mode)
        fence_layout.addWidget(self.fence_btn)
        
        clear_fence_btn = QPushButton("清除围栏")
        clear_fence_btn.clicked.connect(self.clear_fence)
        fence_layout.addWidget(clear_fence_btn)
        layout.addLayout(fence_layout)
        
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
    
    def toggle_fence_mode(self):
        self.fence_mode = not self.fence_mode
        if self.fence_mode:
            self.fence_btn.setText("完成围栏")
            self.fence_btn.setStyleSheet("background-color: #00ff00; color: black;")
        else:
            self.fence_btn.setText("绘制围栏")
            self.fence_btn.setStyleSheet("")
            if len(self.fence_points) >= 3:
                self.apply_fence_filter()
                if self.viewer:
                    self.viewer.update_display()
    
    def clear_fence(self):
        self.fence_points = []
        self.fence_mode = False
        self.fence_btn.setText("绘制围栏")
        self.fence_btn.setStyleSheet("")
        if self.viewer:
            self.viewer.update_display()
    
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
    
    def on_zoom_change(self, value):
        factor = value / 100.0
        self.zoom_label.setText(f"{value}%")
        if self.viewer:
            self.viewer.set_zoom(factor)
    
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
    
    def get_deleted_track_ids(self):
        return set(dp['track_id'] for dp in self.del_points)
    
    def apply_fence_filter(self):
        if len(self.fence_points) < 3:
            return
        
        labels_dir = self.temp_data_path / "labels"
        fence_pts = np.array(self.fence_points, dtype=np.int32)
        
        deleted_by_fence = set()
        for i in range(self.total_frames):
            label_path = labels_dir / f"frame_{i:06d}.json"
            if not label_path.exists():
                continue
            with open(label_path) as f:
                annotations = json.load(f)
            
            for ann in annotations:
                track_id = ann.get('track_id', ann['id'])
                bbox = ann['bbox']
                
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                x3, y3 = bbox[0], bbox[1] + bbox[3]
                x4, y4 = bbox[0] + bbox[2], bbox[1]
                
                points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                outside_fence = any(
                    cv2.pointPolygonTest(fence_pts, pt, False) < 0 
                    for pt in points
                )
                
                if outside_fence:
                    deleted_by_fence.add(track_id)
        
        for track_id in deleted_by_fence:
            if track_id not in self.get_deleted_track_ids():
                self.del_points.append({
                    'x': 0, 'y': 0, 
                    'frame_idx': 0, 
                    'track_id': track_id,
                    'shortcut': '围栏过滤'
                })
        
        self.update_del_list()
    
    def filter_annotations(self, annotations):
        if not annotations:
            return []
        deleted_ids = self.get_deleted_track_ids()
        filtered = []
        
        fence_pts = None
        if self.fence_points and len(self.fence_points) >= 3:
            fence_pts = np.array(self.fence_points, dtype=np.int32)
        
        for ann in annotations:
            track_id = ann.get('track_id', ann.get('id', 0))
            conf = ann.get('confidence', 1.0)
            
            if conf < self.conf_threshold:
                continue
            
            if track_id in deleted_ids:
                continue
            
            if fence_pts is not None:
                bbox = ann['bbox']
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                x3, y3 = bbox[0], bbox[1] + bbox[3]
                x4, y4 = bbox[0] + bbox[2], bbox[1]
                
                points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                outside_fence = any(
                    cv2.pointPolygonTest(fence_pts, pt, False) < 0 
                    for pt in points
                )
                
                if outside_fence:
                    continue
            
            filtered.append(ann)
        
        return filtered
    
    def apply_threshold_to_masks(self, frame, annotations, threshold):
        result_frame = frame.copy()
        
        mask_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for ann in annotations:
            polygon = ann['segmentation'][0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
            
            cv2.fillPoly(result_frame, [pts], color)
            cv2.polylines(result_frame, [pts], True, (255, 255, 255), 2)
            
            bbox = ann['bbox']
            track_id = ann.get('track_id', ann['id'])
            conf = ann.get('confidence', 1.0)
            
            x, y = int(bbox[0]), int(bbox[1])
            cv2.putText(result_frame, f"ID:{track_id} {conf:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if self.fence_points and len(self.fence_points) >= 3:
            fence_pts = np.array(self.fence_points, dtype=np.int32)
            pts_array = fence_pts.reshape((-1, 1, 2))
            cv2.polylines(result_frame, [pts_array], True, (0, 255, 0), 3)
            for pt in self.fence_points:
                cv2.circle(result_frame, pt, 5, (0, 255, 0), -1)
        
        return result_frame
    
    def handle_click(self, x, y, frame_idx):
        print(f"点击: ({x}, {y}), 帧{frame_idx+1}")
        
        if not self.viewer:
            return
        
        if self.fence_mode:
            self.fence_points.append((x, y))
            print(f"围栏点: {len(self.fence_points)}")
            self.viewer.update_display()
            return
        
        frame, annotations = self.viewer.load_frame_data(frame_idx)
        filtered = self.filter_annotations(annotations)
        
        for ann in filtered:
            polygon = ann['segmentation'][0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                track_id = ann.get('track_id', ann['id'])
                self.del_points.append({'x': x, 'y': y, 'frame_idx': frame_idx, 'track_id': track_id})
                self.del_list.addItem(f"帧{frame_idx+1} ({x},{y}) ID:{track_id}")
                print(f"删除track_id: {track_id}")
                self.viewer.update_display()
                return
        
        print("未找到标注")
    
    def remove_selected_del_point(self):
        current_row = self.del_list.currentRow()
        if current_row >= 0:
            self.del_points.pop(current_row)
            self.del_list.takeItem(current_row)
            if self.viewer:
                self.viewer.update_display()
    
    def update_del_list(self):
        self.del_list.clear()
        for dp in self.del_points:
            self.del_list.addItem(f"帧{dp['frame_idx']+1} ({dp['x']},{dp['y']}) ID:{dp['track_id']}")
    
    def clear_del_points(self):
        self.del_points = []
        self.del_list.clear()
        if self.viewer:
            self.viewer.update_display()
    
    def apply_masks_without_fence(self, frame, annotations, threshold):
        result_frame = frame.copy()
        overlay = frame.copy()
        
        if not annotations:
            return result_frame
        
        mask_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for ann in annotations:
            polygon = ann.get('segmentation')
            if not polygon:
                continue
            pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
            color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
            
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
            
            bbox = ann.get('bbox')
            if not bbox:
                continue
            x, y = int(bbox[0]), int(bbox[1])
            category = ann.get('category', ann.get('category_id', 0))
            conf = ann.get('confidence', 1.0)
            
            cv2.putText(overlay, f"{category} {conf:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.5, result_frame, 0.5, 0, result_frame)
        
        return result_frame
    
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
        
        all_annotations = []
        
        for i in range(self.total_frames):
            frame_path = str(frames_dir / f"frame_{i:06d}.jpg")
            frame = cv2.imread(frame_path)
            
            output_frame_path = str(output_frames_dir / f"frame_{i:06d}.jpg")
            cv2.imwrite(output_frame_path, frame)
            
            label_path = labels_dir / f"frame_{i:06d}.json"
            output_label_path = output_labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path) as f:
                    annotations = json.load(f)
                
                filtered = self.filter_annotations(annotations)
                category_name = self.category_input.text() or "Detect"
                
                frame_anns = []
                for ann in filtered:
                    ann_copy = ann.copy()
                    ann_copy['category'] = category_name
                    frame_anns.append(ann_copy)
                
                with open(output_label_path, 'w') as f:
                    json.dump(frame_anns, f)
                
                for ann in filtered:
                    ann_copy = ann.copy()
                    ann_copy['category'] = category_name
                    all_annotations.append(ann_copy)
                
                frame = self.apply_masks_without_fence(frame, frame_anns, self.conf_threshold)
            
            out.write(frame)
        
        coco_output = {
            'info': self.video_info,
            'images': [{'id': i, 'frame_idx': i} for i in range(self.total_frames)],
            'annotations': all_annotations
        }
        
        with open(output_path / "annotations.json", 'w') as f:
            json.dump(coco_output, f)
        
        out.release()
        
        import subprocess
        print(f"正在上传视频到OBS...")
        try:
            result = subprocess.run(
                ['curl', '--upload-file', str(dst_path), 'http://obs.dimond.top/output_annotated.mp4'],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print(f"上传失败: {result.stderr}")
        except Exception as e:
            print(f"上传失败: {e}")
        
        QMessageBox.information(self, "完成", 
                             f"视频已导出到: {dst_path}\n数据已保存到: {output_path}\n已上传到OBS")

def main():
    app = QApplication(sys.argv)
    panel = ControlPanel("temp_data")
    panel.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
