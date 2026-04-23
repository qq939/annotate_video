# global参数
TEMP_DATA_DIR = "temp_data"  # 第8行：临时数据目录
DEFAULT_CONF_THRESHOLD = 0.5  # 第9行：默认置信度阈值

import cv2
import numpy as np
import json
import sys
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QMessageBox, QFileDialog, QInputDialog,
                             QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont

class ClickableLabel(QLabel):
    """可点击的标签，支持鼠标事件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.del_points = []  # 删除点列表 (x, y, frame_idx)
        
    def mousePressEvent(self, event):
        main_window = self.window()
        
        if event.button() == Qt.RightButton:
            if main_window:
                main_window._right_button_x = event.x()
                main_window._right_button_y = event.y()
            event.accept()
            return
            
        if event.button() == Qt.LeftButton:
            pixmap = self.pixmap()
            
            if pixmap and main_window:
                label_w = self.width()
                label_h = self.height()
                pixmap_size = pixmap.size()
                
                zoom_factor = main_window.zoom_level / 100.0
                scale = min(label_w / pixmap_size.width(), label_h / pixmap_size.height()) * zoom_factor
                display_w = pixmap_size.width() * scale
                display_h = pixmap_size.height() * scale
                offset_x = (label_w - display_w) / 2 + main_window.pan_x
                offset_y = (label_h - display_h) / 2 + main_window.pan_y
                
                x = int((event.x() - offset_x) / scale)
                y = int((event.y() - offset_y) / scale)
                
                if 0 <= x < pixmap_size.width() and 0 <= y < pixmap_size.height():
                    if hasattr(main_window, 'handle_del_point'):
                        frame_idx = main_window.current_frame_idx
                        print(f"🗑️ 点击删除点: 帧{frame_idx+1}, 坐标({x}, {y})")
                        main_window.handle_del_point(x, y, frame_idx)
                        self.update()
                    else:
                        print(f"⚠️ 无法获取主窗口，跳过处理")
                    
        super().mousePressEvent(event)
        
    def wheelEvent(self, event):
        pass
        
    def event(self, event):
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        return super().event(event)
    
    def gestureEvent(self, event):
        main_window = self.window()
        if not main_window:
            return True
            
        pinch = event.gesture(Qt.PinchGesture)
        if pinch:
            self.handlePinchGesture(pinch)
            
        pan = event.gesture(Qt.PanGesture)
        if pan:
            self.handlePanGesture(pan)
            
        return True
    
    def handlePinchGesture(self, gesture):
        main_window = self.window()
        if not main_window or not hasattr(main_window, 'zoom_level'):
            return
            
        if gesture.state() == Qt.GestureStarted:
            self._last_pinch_scale = 1.0
        elif gesture.state() == Qt.GestureUpdated:
            scale = gesture.scaleFactor()
            delta = scale / self._last_pinch_scale
            self._last_pinch_scale = scale
            
            new_zoom = main_window.zoom_level * delta
            new_zoom = max(25, min(200, new_zoom))
            main_window.on_zoom_change(int(new_zoom))
            if hasattr(main_window, 'zoom_label'):
                main_window.zoom_label.setText(f"画面缩放: {int(new_zoom)}%")
    
    def handlePanGesture(self, gesture):
        main_window = self.window()
        if not main_window or not hasattr(main_window, 'pan_x'):
            return
            
        if gesture.state() == Qt.GestureStarted:
            self._last_pan_delta = None
        elif gesture.state() == Qt.GestureUpdated:
            delta = gesture.delta()
            if self._last_pan_delta:
                dx = delta.x() - self._last_pan_delta.x()
                dy = delta.y() - self._last_pan_delta.y()
                main_window.pan_x += dx
                main_window.pan_y += dy
                self.update()
            self._last_pan_delta = delta
        
    def mouseMoveEvent(self, event):
        main_window = self.window()
        
        # 右键拖拽平移
        if event.buttons() & Qt.RightButton and hasattr(main_window, 'pan_x'):
            dx = event.x() - getattr(main_window, '_right_button_x', event.x())
            dy = event.y() - getattr(main_window, '_right_button_y', event.y())
            main_window.pan_x += dx
            main_window.pan_y += dy
            main_window._right_button_x = event.x()
            main_window._right_button_y = event.y()
            self.update()
            event.accept()
            return
            
        # 左键拖拽平移
        if event.buttons() & Qt.LeftButton and hasattr(main_window, 'pan_x'):
            if not (hasattr(main_window, '_is_dragging_del_point') and main_window._is_dragging_del_point):
                dx = event.x() - getattr(main_window, '_last_mouse_x', event.x())
                dy = event.y() - getattr(main_window, '_last_mouse_y', event.y())
                main_window.pan_x += dx
                main_window.pan_y += dy
                main_window._last_mouse_x = event.x()
                main_window._last_mouse_y = event.y()
                self.update()
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        main_window = self.window()
        if hasattr(main_window, 'del_points') and main_window.del_points:
            painter = QPainter(self)
            pixmap = self.pixmap()
            if pixmap:
                label_w = self.width()
                label_h = self.height()
                pixmap_size = pixmap.size()
                
                zoom_factor = 1.0
                if hasattr(main_window, 'zoom_level'):
                    zoom_factor = main_window.zoom_level / 100.0
                
                scale = min(label_w / pixmap_size.width(), label_h / pixmap_size.height()) * zoom_factor
                display_w = pixmap_size.width() * scale
                display_h = pixmap_size.height() * scale
                offset_x = (label_w - display_w) / 2
                offset_y = (label_h - display_h) / 2
                
                pan_x = getattr(main_window, 'pan_x', 0)
                pan_y = getattr(main_window, 'pan_y', 0)
                offset_x += pan_x
                offset_y += pan_y
                
                for del_info in main_window.del_points:
                    if isinstance(del_info, dict):
                        x = del_info['x']
                        y = del_info['y']
                        frame_idx = del_info['frame_idx']
                        shortcut = del_info.get('shortcut', f'F{frame_idx+1}')
                    else:
                        continue
                    
                    display_x = offset_x + x * scale
                    display_y = offset_y + y * scale
                    
                    painter.setPen(Qt.red)
                    painter.setBrush(Qt.red)
                    painter.drawEllipse(int(display_x) - 5, int(display_y) - 5, 10, 10)
                    painter.setPen(Qt.yellow)
                    painter.setFont(QFont("Arial", 8))
                    painter.drawText(int(display_x) + 8, int(display_y) + 3, shortcut)

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
        self.del_points = []
        self.zoom_level = 100
        self.pan_x = 0
        self.pan_y = 0
        
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
        self.setGeometry(100, 100, 1200, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, 3)
        
        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.clickable_label = self.image_label
        self.image_label.grabGesture(Qt.PinchGesture)
        self.image_label.grabGesture(Qt.PanGesture)
        self.image_label.setAttribute(Qt.WA_AcceptTouchEvents)
        left_layout.addWidget(self.image_label)
        
        info_layout = QHBoxLayout()
        self.info_label = QLabel(f"帧: 1/{self.total_frames}")
        info_layout.addWidget(self.info_label)
        self.count_label = QLabel(f"可见标注数: 0/0")
        info_layout.addWidget(self.count_label)
        self.threshold_label = QLabel(f"置信度阈值: {self.conf_threshold:.2f}")
        info_layout.addWidget(self.threshold_label)
        left_layout.addLayout(info_layout)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(int(DEFAULT_CONF_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self.on_conf_change)
        left_layout.addWidget(QLabel("置信度阈值"))
        left_layout.addWidget(self.threshold_slider)
        
        speed_layout = QHBoxLayout()
        self.play_speed = 4.0
        self.speed_label = QLabel(f"播放速度: {self.play_speed:.1f}fps")
        speed_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(4)
        self.speed_slider.valueChanged.connect(self.on_speed_change)
        speed_layout.addWidget(QLabel("播放速度"))
        speed_layout.addWidget(self.speed_slider)
        left_layout.addLayout(speed_layout)
        
        zoom_layout = QHBoxLayout()
        self.zoom_label = QLabel(f"画面缩放: {self.zoom_level}%")
        zoom_layout.addWidget(self.zoom_label)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_change)
        zoom_layout.addWidget(QLabel("画面缩放"))
        zoom_layout.addWidget(self.zoom_slider)
        left_layout.addLayout(zoom_layout)
        
        instructions = QLabel("💡 触控板双指缩放/平移 | 右键长按拖拽平移 | 左键点击添加删除点 | Ctrl+Z撤销")
        left_layout.addWidget(instructions)
        
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
        
        left_layout.addLayout(button_layout)
        
        instructions = QLabel("💡 点击视频中的误判区域添加删除点，程序会自动查找最近的标注并删除\n💡 Ctrl+Z 撤销最后一个删除点")
        left_layout.addWidget(instructions)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, 1)
        
        del_title = QLabel("🗑️ 待删除列表")
        del_title.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
        right_layout.addWidget(del_title)
        
        self.del_points_scroll = QScrollArea()
        self.del_points_scroll.setWidgetResizable(True)
        self.del_points_scroll.setMinimumWidth(250)
        self.del_points_widget = QWidget()
        self.del_points_layout = QVBoxLayout()
        self.del_points_widget.setLayout(self.del_points_layout)
        self.del_points_scroll.setWidget(self.del_points_widget)
        right_layout.addWidget(self.del_points_scroll)
        
        right_layout.addStretch()
        
        hint_label = QLabel("已删除: 0 个 track_id")
        hint_label.setObjectName("del_count_label")
        right_layout.addWidget(hint_label)
        
    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_Z:
            if self.del_points:
                self.remove_del_point(len(self.del_points) - 1)
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
            
    def on_conf_change(self, value):
        self.conf_threshold = value / 100.0
        self.threshold_label.setText(f"置信度阈值: {self.conf_threshold:.2f}")
        self.update_display()
    
    def on_speed_change(self, value):
        self.play_speed = float(value)
        self.speed_label.setText(f"播放速度: {self.play_speed:.1f}fps")
        if self.timer.isActive():
            self.timer.setInterval(int(1000.0 / self.play_speed))
        
    def on_zoom_change(self, value):
        self.zoom_level = value
        if hasattr(self, 'zoom_label'):
            self.zoom_label.setText(f"画面缩放: {self.zoom_level}%")
        self.update_display()
        
    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.timer.setInterval(int(1000.0 / self.play_speed))
        if self.is_playing:
            self.play_btn.setText("暂停")
            self.timer.start()
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
        
        zoom_factor = self.zoom_level / 100.0
        scaled_w = int(w * zoom_factor)
        scaled_h = int(h * zoom_factor)
        scaled_pixmap = pixmap.scaled(scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
        visible_count = sum(1 for ann in annotations 
                          if ann.get('confidence', 1.0) >= self.conf_threshold
                          and ann.get('track_id', ann['id']) not in self.del_track_id_list)
        deleted_count = len(annotations) - visible_count - sum(1 for ann in annotations 
                                                               if ann.get('confidence', 1.0) < self.conf_threshold)
        self.info_label.setText(f"帧: {self.current_frame_idx + 1}/{self.total_frames}")
        self.count_label.setText(f"可见标注数: {visible_count}/{len(annotations)} (已删除: {len(self.del_track_id_list)})")
        
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
            
            # 过滤已删除的track_id
            track_id = ann.get('track_id', ann['id'])
            if track_id in self.del_track_id_list:
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
    
    def handle_del_point(self, x, y, frame_idx):
        """处理删除点，实时删除track_id"""
        nearest_ann = self.find_nearest_annotation_in_frame(x, y, frame_idx)
        
        del_info = {
            'x': x,
            'y': y,
            'frame_idx': frame_idx,
            'track_id': None,
            'ann_id': None,
            'shortcut': f"F{frame_idx+1}"
        }
        
        if nearest_ann:
            if 'track_id' in nearest_ann:
                track_id = nearest_ann['track_id']
                del_info['track_id'] = track_id
                
                if track_id not in self.del_track_id_list:
                    self.del_track_id_list.append(track_id)
                    print(f"🗑️ 删除track_id: {track_id} (来自帧{frame_idx+1})")
                else:
                    print(f"⚠️ track_id {track_id} 已经在删除列表中")
            else:
                ann_id = nearest_ann['id']
                del_info['ann_id'] = ann_id
                
                if ann_id not in self.del_track_id_list:
                    self.del_track_id_list.append(ann_id)
                    print(f"🗑️ 删除标注ID: {ann_id} (来自帧{frame_idx+1})")
                else:
                    print(f"⚠️ 标注ID {ann_id} 已经在删除列表中")
        else:
            print(f"⚠️ 在帧{frame_idx+1}的坐标({x}, {y})处未找到标注")
            return
        
        del_info['idx'] = len(self.del_points)
        self.del_points.append(del_info)
        
        self.add_del_point_ui(del_info)
        
        self.update_display()
        self.update_del_count_label()
        
    def find_nearest_annotation_in_frame(self, x, y, frame_idx):
        """在指定帧中查找最近的标注"""
        frame, annotations = self.load_frame_data(frame_idx)
        
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
        
    def add_del_point_ui(self, del_info):
        """添加删除点到右侧列表UI"""
        idx = del_info['idx']
        frame_idx = del_info['frame_idx']
        x, y = del_info['x'], del_info['y']
        track_id = del_info.get('track_id')
        ann_id = del_info.get('ann_id')
        shortcut = del_info.get('shortcut', f'#{idx+1}')
        
        item_widget = QWidget()
        item_layout = QHBoxLayout()
        item_widget.setLayout(item_layout)
        
        info_label = QLabel()
        if track_id is not None:
            info_label.setText(f"{shortcut} | 帧{frame_idx+1} | ({x},{y}) | ID:{track_id}")
        else:
            info_label.setText(f"{shortcut} | 帧{frame_idx+1} | ({x},{y}) | ID:{ann_id}")
        info_label.setStyleSheet("color: #333; font-size: 11px;")
        item_layout.addWidget(info_label)
        
        del_btn = QPushButton("×")
        del_btn.setFixedSize(24, 24)
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cc0000;
            }
        """)
        del_btn.clicked.connect(lambda: self.remove_del_point(idx))
        item_layout.addWidget(del_btn)
        
        item_widget.setStyleSheet("""
            QWidget {
                background-color: #fff0f0;
                border: 1px solid #ffcccc;
                border-radius: 4px;
                padding: 5px;
                margin: 2px;
            }
        """)
        
        self.del_points_layout.addWidget(item_widget)
        
    def update_del_count_label(self):
        """更新删除计数标签"""
        count_label = self.findChild(QLabel, "del_count_label")
        if count_label:
            unique_track_ids = len(set(self.del_track_id_list))
            count_label.setText(f"已删除: {len(self.del_track_id_list)} 个标注 | {unique_track_ids} 个track_id")
        
    def remove_del_point(self, idx):
        """移除指定的删除点"""
        if 0 <= idx < len(self.del_points):
            del_info = self.del_points[idx]
            
            track_id = del_info.get('track_id')
            ann_id = del_info.get('ann_id')
            
            remove_id = track_id if track_id is not None else ann_id
            
            if remove_id in self.del_track_id_list:
                self.del_track_id_list.remove(remove_id)
                print(f"↩️ 撤销删除: {remove_id}")
            
            self.del_points.pop(idx)
            
            self.rebuild_del_points_ui()
            
            self.update_display()
            self.update_del_count_label()
            
    def rebuild_del_points_ui(self):
        """重建所有删除点UI"""
        while self.del_points_layout.count():
            item = self.del_points_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for idx, del_info in enumerate(self.del_points):
            del_info['idx'] = idx
            self.add_del_point_ui(del_info)
        
    def clear_del_points(self):
        """清空删除点和删除列表"""
        self.image_label.del_points = []
        self.del_track_id_list = []
        self.del_points = []
        
        self.rebuild_del_points_ui()
        
        print("已清空所有删除点和删除列表")
        self.update_display()
        self.update_del_count_label()
        
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
        if self.del_points:
            print(f"\n正在删除 {len(self.del_track_id_list)} 个track_id/标注...")
            print(f"删除列表: {self.del_track_id_list}")
            
            if self.del_track_id_list:
                self.apply_deletions()
                
        self.do_export_video()
        
    def apply_deletions(self):
        """应用删除操作"""
        if not self.del_track_id_list:
            return
            
        print(f"\n正在删除track_ids: {self.del_track_id_list}")
        
        def should_keep_ann(ann):
            track_id = ann.get('track_id', ann['id'])
            ann_id = ann['id']
            return track_id not in self.del_track_id_list and ann_id not in self.del_track_id_list
        
        # 删除全局annotations
        original_count = len(self.coco_data['annotations'])
        self.coco_data['annotations'] = [
            ann for ann in self.coco_data['annotations']
            if should_keep_ann(ann)
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
                    if should_keep_ann(lbl)
                ]
                
                with open(str(label_file), 'w') as f:
                    json.dump(frame_labels, f)
                    
        print("✓ 删除完成，数据已保存到temp_data")
        
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
