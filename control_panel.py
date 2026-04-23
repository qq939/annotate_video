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
        self.fences = []  # 多个围栏列表
        self.current_fence_idx = -1  # 当前绘制的围栏索引
        self.max_fences = 3
        
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
        
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(10)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.on_alpha_change)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("50%")
        alpha_layout.addWidget(self.alpha_label)
        layout.addLayout(alpha_layout)
        
        self.fence_btns = []
        self.fence_clear_btns = []
        for i in range(self.max_fences):
            fence_layout = QHBoxLayout()
            self.fence_btns.append(QPushButton(f"围栏{i+1}绘制"))
            self.fence_btns[i].clicked.connect(lambda checked, idx=i: self.toggle_fence_mode(idx))
            fence_layout.addWidget(self.fence_btns[i])
            
            self.fence_clear_btns.append(QPushButton(f"围栏{i+1}清除"))
            self.fence_clear_btns[i].clicked.connect(lambda checked, idx=i: self.clear_fence(idx))
            fence_layout.addWidget(self.fence_clear_btns[i])
            layout.addLayout(fence_layout)
        
        play_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶播放")
        self.play_btn.clicked.connect(self.toggle_play)
        play_layout.addWidget(self.play_btn)
        
        self.backward_btn = QPushButton("◀倒播")
        self.backward_btn.clicked.connect(self.toggle_backward)
        play_layout.addWidget(self.backward_btn)
        layout.addLayout(play_layout)
        
        self.is_playing = False
        self.is_backward = False
        
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
    
    def toggle_fence_mode(self, fence_idx):
        if fence_idx >= len(self.fences):
            self.fences.append({'points': [], 'mode': True})
            self.current_fence_idx = fence_idx
        else:
            fence = self.fences[fence_idx]
            fence['mode'] = not fence['mode']
            if fence['mode']:
                self.current_fence_idx = fence_idx
        
        self.update_fence_buttons()
        if self.viewer:
            self.viewer.update_display()
    
    def clear_fence(self, fence_idx):
        if fence_idx < len(self.fences):
            self.fences[fence_idx]['points'] = []
            self.fences[fence_idx]['mode'] = False
        if self.current_fence_idx == fence_idx:
            self.current_fence_idx = -1
        self.update_fence_buttons()
        if self.viewer:
            self.viewer.update_display()
    
    def update_fence_buttons(self):
        for i, btn in enumerate(self.fence_btns):
            if i < len(self.fences) and self.fences[i]['mode']:
                btn.setStyleSheet("background-color: #00ff00; color: black;")
                btn.setText(f"围栏{i+1}完成")
            else:
                btn.setStyleSheet("")
                btn.setText(f"围栏{i+1}绘制")
    
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
        self.is_backward = False
        self.backward_btn.setText("◀倒播")
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶播放")
        else:
            self.timer.start(100)
            self.is_playing = True
            self.play_btn.setText("⏸播放")
    
    def toggle_backward(self):
        self.is_playing = False
        self.play_btn.setText("▶播放")
        if self.is_backward:
            self.timer.stop()
            self.is_backward = False
            self.backward_btn.setText("◀倒播")
        else:
            self.timer.start(100)
            self.is_backward = True
            self.backward_btn.setText("⏸倒播")
    
    def play_next(self):
        if self.is_backward:
            if self.viewer:
                self.viewer.go_to_frame(self.viewer.get_current_frame() - 1)
                self.frame_label.setText(f"帧: {self.viewer.get_current_frame()+1}/{self.total_frames}")
        else:
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
    
    def on_alpha_change(self, value):
        self.alpha = value / 100.0
        self.alpha_label.setText(f"{value}%")
        if self.viewer:
            self.viewer.update_display()
    
    def get_deleted_track_ids(self):
        return set(dp['track_id'] for dp in self.del_points)
    
    def apply_fence_filter(self):
        deleted_by_fence = set()
        
        for fence in self.fences:
            if len(fence['points']) < 3:
                continue
            fence_pts = np.array(fence['points'], dtype=np.int32)
            
            for i in range(self.total_frames):
                label_path = self.temp_data_path / "labels" / f"frame_{i:06d}.json"
                if not label_path.exists():
                    continue
                with open(label_path) as f:
                    annotations = json.load(f)
                
                for ann in annotations:
                    track_id = ann.get('track_id', ann.get('id', 0))
                    bbox = ann.get('bbox')
                    if not bbox:
                        continue
                    
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
            
            # 围栏过滤：只检查bbox中心点
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
    
    def apply_threshold_to_masks(self, frame, annotations, threshold):
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
            
            # 绘制mask
            if polygon:
                pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(result_frame, [pts], color)
                cv2.polylines(result_frame, [pts], True, (255, 255, 255), 2)
            
            # 绘制bbox
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制label和置信度
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
    
    def handle_click(self, x, y, frame_idx):
        print(f"点击: ({x}, {y}), 帧{frame_idx+1}")
        
        if not self.viewer:
            return
        
        for fence in self.fences:
            if fence['mode']:
                fence['points'].append((x, y))
                print(f"围栏点: {len(fence['points'])}")
                self.viewer.update_display()
                return
        
        frame, annotations = self.viewer.load_frame_data(frame_idx)
        filtered = self.filter_annotations(annotations)
        
        for ann in filtered:
            polygon = ann.get('segmentation')
            if not polygon:
                continue
            pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                track_id = ann.get('track_id', ann.get('id', 0))
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
            bbox = ann.get('bbox')
            if not bbox:
                continue
            
            color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
            
            # 绘制mask
            if polygon:
                pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
            
            # 绘制bbox
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            category = ann.get('category', ann.get('category_id', 0))
            conf = ann.get('confidence', 1.0)
            
            cv2.putText(overlay, f"{category} {conf:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 渲染多个围栏
        fence_colors = [(0, 255, 0), (255, 165, 0), (255, 0, 255)]
        for i, fence in enumerate(self.fences):
            if len(fence['points']) >= 3:
                color = fence_colors[i % len(fence_colors)]
                pts = np.array(fence['points'], dtype=np.int32)
                pts_array = pts.reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts_array], True, color, 3)
                for pt in fence['points']:
                    cv2.circle(overlay, pt, 5, color, -1)
        
        cv2.addWeighted(overlay, 0.5, result_frame, 0.5, 0, result_frame)
    
    def export_video(self):
        output_path = Path("temp_data_post")
        output_path.mkdir(exist_ok=True)
        
        dst_path = Path("dst/output_annotated.mp4")
        dst_path.parent.mkdir(exist_ok=True)
        
        labels_dir = self.temp_data_path / "labels"
        frames_dir = self.temp_data_path / "frames"
        output_labels_dir = output_path / "labels"
        output_labels_dir.mkdir(exist_ok=True)
        output_frames_dir = output_path / "frames"
        output_frames_dir.mkdir(exist_ok=True)
        
        category_name = self.category_input.text() or "Detect"
        all_annotations = []
        
        print("步骤1: 保存到 temp_data_post...")
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
                
                frame_anns = []
                for ann in filtered:
                    ann_copy = ann.copy()
                    ann_copy['category'] = category_name
                    frame_anns.append(ann_copy)
                
                with open(output_label_path, 'w') as f:
                    json.dump(frame_anns, f)
                
                all_annotations.extend(frame_anns)
        
        coco_output = {
            'info': self.video_info,
            'images': [{'id': i, 'frame_idx': i} for i in range(self.total_frames)],
            'annotations': all_annotations
        }
        
        with open(output_path / "annotations.json", 'w') as f:
            json.dump(coco_output, f)
        
        print("步骤2: 制作 dst 视频...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(self.video_info['fps'])
        out = cv2.VideoWriter(str(dst_path), fourcc, fps, 
                            (int(self.video_info['width']), int(self.video_info['height'])))
        
        post_labels_dir = output_path / "labels"
        post_frames_dir = output_path / "frames"
        
        for i in range(self.total_frames):
            frame_path = str(post_frames_dir / f"frame_{i:06d}.jpg")
            frame = cv2.imread(frame_path)
            
            label_path = post_labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path) as f:
                    annotations = json.load(f)
                
                frame = self.render_frame_for_export(frame, annotations)
            
            out.write(frame)
        
        out.release()
        
        print("步骤3: 上传到OBS...")
        import subprocess
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
                             f"数据已保存到: {output_path}\n视频已导出到: {dst_path}\n已上传到OBS")
    
    def render_frame_for_export(self, frame, annotations):
        alpha = getattr(self, 'alpha', 0.5)
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
            bbox = ann.get('bbox')
            
            if not bbox:
                continue
            
            color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
            category = ann.get('category', ann.get('category_id', 0))
            conf = ann.get('confidence', 1.0)
            
            if polygon:
                pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
            
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            cv2.putText(overlay, f"{category} {conf:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
        return result_frame

def main():
    app = QApplication(sys.argv)
    panel = ControlPanel("temp_data")
    panel.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
