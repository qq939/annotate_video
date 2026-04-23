# global参数
TEMP_DATA_DIR = "temp_data"  # 第8行：临时数据目录
DEFAULT_CONF_THRESHOLD = 0.5  # 第9行：默认置信度阈值

import cv2
import numpy as np
import json
import sys
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QMessageBox, QFileDialog, QInputDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class ClickableLabel(QLabel):
    """可点击的标签，支持鼠标事件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.del_points = []  # 删除点列表 (x, y)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 获取点击坐标
            pixmap = self.pixmap()
            if pixmap:
                # 计算实际显示区域
                label_w = self.width()
                label_h = self.height()
                pixmap_size = pixmap.size()
                
                scale = min(label_w / pixmap_size.width(), label_h / pixmap_size.height())
                display_w = pixmap_size.width() * scale
                display_h = pixmap_size.height() * scale
                offset_x = (label_w - display_w) / 2
                offset_y = (label_h - display_h) / 2
                
                # 转换为实际像素坐标
                x = int((event.x() - offset_x) / scale)
                y = int((event.y() - offset_y) / scale)
                
                if 0 <= x < pixmap_size.width() and 0 <= y < pixmap_size.height():
                    self.del_points.append((x, y))
                    print(f"🗑️ 点击删除点: ({x}, {y})")
                    self.update()
                    
        super().mousePressEvent(event)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.del_points:
            painter = QPainter(self)
            pixmap = self.pixmap()
            if pixmap:
                label_w = self.width()
                label_h = self.height()
                pixmap_size = pixmap.size()
                scale = min(label_w / pixmap_size.width(), label_h / pixmap_size.height())
                display_w = pixmap_size.width() * scale
                display_h = pixmap_size.height() * scale
                offset_x = (label_w - display_w) / 2
                offset_y = (label_h - display_h) / 2
                
                for x, y in self.del_points:
                    # 绘制红点
                    display_x = offset_x + x * scale
                    display_y = offset_y + y * scale
                    painter.setPen(Qt.red)
                    painter.setBrush(Qt.red)
                    painter.drawEllipse(int(display_x) - 5, int(display_y) - 5, 10, 10)

class PostAnnotatorWindow(QMainWindow):
    def __init__(self, output_video_path, temp_data_path=None, del_track_id_list=None):
        super().__init__()
        self.output_video_path = output_video_path
        if temp_data_path:
            self.temp_data_path = Path(temp_data_path).resolve()
        else:
            self.temp_data_path = Path(TEMP_DATA_DIR).resolve()
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.del_track_id_list = del_track_id_list if del_track_id_list else []
        
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
        
        self.clickable_label = None  # 点击删除标签
        
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
        
        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.clickable_label = self.image_label
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
        self.threshold_slider.valueChanged.connect(self.on_conf_change)
        layout.addWidget(QLabel("置信度阈值:"))
        layout.addWidget(self.threshold_slider)
        
        button_layout = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        button_layout.addWidget(self.play_btn)
        
        self.export_btn = QPushButton("导出视频")
        self.export_btn.clicked.connect(self.export_video)
        button_layout.addWidget(self.export_btn)
        
        self.clear_del_points_btn = QPushButton("清空删除点")
        self.clear_del_points_btn.clicked.connect(self.clear_del_points)
        button_layout.addWidget(self.clear_del_points_btn)
        
        layout.addLayout(button_layout)
        
        instructions = QLabel("💡 点击视频中的误判区域添加删除点，程序会自动查找最近的标注并删除")
        layout.addWidget(instructions)
        
    def on_conf_change(self, value):
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
            track_id = ann.get('track_id', ann['id'])
            label = f"ID:{track_id}"
            cv2.putText(result_frame, label, (int(x), int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return result_frame
        
    def clear_del_points(self):
        """清空删除点"""
        self.image_label.del_points = []
        print("已清空所有删除点")
        self.update_display()
        
    def find_nearest_annotation(self, x, y):
        """查找最近的标注"""
        frame, annotations = self.load_frame_data(self.current_frame_idx)
        
        min_dist = float('inf')
        nearest_ann = None
        
        for ann in annotations:
            bbox = ann['bbox']
            ann_x, ann_y = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            dist = ((x - ann_x) ** 2 + (y - ann_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_ann = ann
                
        return nearest_ann
        
    def export_video(self):
        """导出视频前处理删除点"""
        if self.image_label.del_points:
            print("\n正在处理删除点...")
            
            for x, y in self.image_label.del_points:
                nearest_ann = self.find_nearest_annotation(x, y)
                if nearest_ann and 'track_id' in nearest_ann:
                    track_id = nearest_ann['track_id']
                    if track_id not in self.del_track_id_list:
                        self.del_track_id_list.append(track_id)
                        print(f"  添加track_id {track_id} 到删除列表")
                        
            print(f"删除列表: {self.del_track_id_list}")
            
            # 删除track_id
            if self.del_track_id_list:
                self.apply_deletions()
                
        self.do_export_video()
        
    def apply_deletions(self):
        """应用删除操作"""
        if not self.del_track_id_list:
            return
            
        print(f"\n正在删除track_ids: {self.del_track_id_list}")
        
        # 删除全局annotations
        original_count = len(self.coco_data['annotations'])
        self.coco_data['annotations'] = [
            ann for ann in self.coco_data['annotations']
            if ann.get('track_id') not in self.del_track_id_list
        ]
        deleted_count = original_count - len(self.coco_data['annotations'])
        print(f"从全局annotations中删除 {deleted_count} 条记录")
        
        # 保存全局annotations.json
        with open(str(self.temp_data_path / 'annotations.json'), 'w') as f:
            json.dump(self.coco_data, f, indent=2)
            
        # 更新每帧的labels文件
        for i in range(self.total_frames):
            label_file = self.labels_dir / f"frame_{i:06d}.json"
            if label_file.exists():
                with open(str(label_file), 'r') as f:
                    frame_labels = json.load(f)
                    
                frame_labels = [
                    lbl for lbl in frame_labels
                    if lbl.get('track_id') not in self.del_track_id_list
                ]
                
                with open(str(label_file), 'w') as f:
                    json.dump(frame_labels, f)
                    
        print("✓ 删除完成")
        
    def do_export_video(self):
        """导出视频"""
        print(f"\n正在导出视频，置信度阈值: {self.conf_threshold:.2f}")
        
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
    
    temp_data_path = None
    output_video_path = "dst/output_annotated.mp4"
    del_track_id_list = []
    
    if len(sys.argv) > 1:
        temp_data_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_video_path = sys.argv[2]
        
    if not temp_data_path:
        temp_dir = QFileDialog.getExistingDirectory(None, "选择temp_data文件夹", ".")
        if not temp_dir:
            print("未选择文件夹，程序退出")
            sys.exit(0)
        temp_data_path = temp_dir
        
    temp_path_obj = Path(temp_data_path).resolve()
    if not temp_path_obj.exists():
        print(f"错误：文件夹不存在: {temp_data_path}")
        sys.exit(1)
        
    if not (temp_path_obj / "annotations.json").exists():
        print("错误：annotations.json文件不存在")
        sys.exit(1)
        
    print(f"\n后处理预览程序")
    print(f"数据文件夹: {temp_data_path}")
    
    window = PostAnnotatorWindow(output_video_path, temp_data_path, del_track_id_list)
    window.show()
    app.exec_()
    
    # 返回删除列表供主程序使用
    return window.del_track_id_list
    
if __name__ == "__main__":
    main()
