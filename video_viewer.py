#!/usr/bin/env python3
"""视频查看器 - 纯视图层，控制逻辑由 video_control.VideoController 提供"""

import sys
from pathlib import Path
import random

import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QPushButton, QFileDialog, QInputDialog, QListWidget, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import pyqtSignal

from video_control import VideoController

WARM_COLORS = [(180, 130, 255), (200, 100, 220), (255, 50, 200), (255, 0, 180), (220, 0, 150), (180, 0, 120), (139, 0, 100), (100, 0, 80)]
COLD_COLORS = [(100, 150, 255), (100, 200, 255), (150, 200, 255), (100, 255, 200), (150, 100, 255), (100, 200, 200), (150, 150, 255), (100, 180, 255)]
_color_cache = {}

def get_color_for_track_id(track_id):
    if track_id in _color_cache:
        return _color_cache[track_id]
    if track_id >= 1000000:
        color = WARM_COLORS[track_id % len(WARM_COLORS)]
    else:
        color = COLD_COLORS[track_id % len(COLD_COLORS)]
    _color_cache[track_id] = color
    return color


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

    def __init__(self, temp_data_path, controller=None, panel=None):
        super().__init__()
        self.temp_data_path = Path(temp_data_path)
        self.controller = controller  # VideoController 实例
        self.panel = panel  # UnifiedPanel 实例

        with open(self.temp_data_path / "annotations.json", encoding='utf-8') as f:
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
        
        # 帧删除相关
        self.del_frames = set()
        self.select_start = None
        self.ranges = []  # 待删除范围 [(start, end), ...]
        self.is_delete_mode = False
        
        # 模式切换：A=标注模式，B=标记模式
        self.is_mode_a = True  # True=模式A(标注), False=模式B(标记)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('视频查看器')
        self.resize(int(self.video_width * 0.8), int(self.video_height * 0.8))

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        # 单帧/多帧选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("修改模式:"))
        self.single_frame_radio = QRadioButton("单帧")
        self.single_frame_radio.setChecked(False)  # 默认多帧
        self.multi_frame_radio = QRadioButton("多帧")
        self.multi_frame_radio.setChecked(True)  # 默认多帧
        mode_layout.addWidget(self.single_frame_radio)
        mode_layout.addWidget(self.multi_frame_radio)
        # A/B 模式切换
        mode_layout.addWidget(QLabel("点击:"))
        self.mode_ab_btn = QPushButton("A")
        self.mode_ab_btn.setFixedWidth(30)
        self.mode_ab_btn.setStyleSheet("font-weight: bold;")
        self.mode_ab_btn.clicked.connect(self.toggle_mode_ab)
        mode_layout.addWidget(self.mode_ab_btn)
        mode_layout.addStretch()
        # 添加视频按钮
        add_btn = QPushButton("+添加视频")
        add_btn.clicked.connect(self.add_video_frames)
        mode_layout.addWidget(add_btn)
        
        # 帧删除按钮
        self.delete_mode_btn = QPushButton("帧删除")
        self.delete_mode_btn.clicked.connect(self.toggle_delete_mode)
        mode_layout.addWidget(self.delete_mode_btn)
        
        layout.addLayout(mode_layout)
        
        # 帧删除区域（默认隐藏）
        self.delete_layout = QVBoxLayout()
        self.delete_layout.setContentsMargins(0, 5, 0, 0)
        self.delete_layout.setSpacing(5)
        
        # 帧删除控制按钮
        delete_ctrl = QHBoxLayout()
        for txt, fn in [("◀◀", self.backward), ("▶", self.toggle_play), ("▶▶", self.forward), ("清空", self.clear_delete)]:
            b = QPushButton(txt)
            b.setFixedHeight(28)
            b.setStyleSheet("font-size: 12px;")
            b.clicked.connect(fn)
            delete_ctrl.addWidget(b)
        self.delete_btn = QPushButton("删除\n选中")
        self.delete_btn.setFixedWidth(70)
        self.delete_btn.setStyleSheet("font-size: 12px; background-color: #e74c3c; color: white;")
        self.delete_btn.clicked.connect(self.do_delete_frames)
        delete_ctrl.addWidget(self.delete_btn)
        delete_ctrl.addStretch()
        self.delete_layout.addLayout(delete_ctrl)
        
        # 待删除列表
        list_layout = QHBoxLayout()
        list_layout.addWidget(QLabel("待删除片段:"))
        self.delete_list = QListWidget()
        self.delete_list.setFixedHeight(80)
        self.delete_list.setStyleSheet("font-size: 12px;")
        self.delete_list.itemDoubleClicked.connect(self.delete_item)
        list_layout.addWidget(self.delete_list)
        self.delete_layout.addLayout(list_layout)
        
        # 帧删除区域默认隐藏
        self._set_delete_layout_visible(False)
        layout.addLayout(self.delete_layout)

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

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_C:
            self.undo_last_bbox()
        elif key == Qt.Key_Escape:
            self.close()
    
    def _set_delete_layout_visible(self, visible):
        """设置帧删除区域的显示/隐藏"""
        for i in range(self.delete_layout.count()):
            widget = self.delete_layout.itemAt(i)
            if widget.widget():
                widget.widget().setVisible(visible)
    
    def toggle_delete_mode(self):
        """切换帧删除模式"""
        self.is_delete_mode = not self.is_delete_mode
        if self.is_delete_mode:
            self._set_delete_layout_visible(True)
            self.delete_mode_btn.setStyleSheet("background-color: #e74c3c; color: white;")
            self.delete_mode_btn.setText("退出删除")
            self.delete_btn.setText("删除\n选中")
        else:
            self._set_delete_layout_visible(False)
            self.delete_mode_btn.setStyleSheet("")
            self.delete_mode_btn.setText("帧删除")
            self.select_start = None
    
    def toggle_mode_ab(self):
        """切换A/B模式"""
        self.is_mode_a = not self.is_mode_a
        if self.is_mode_a:
            self.mode_ab_btn.setText("A")
            self.mode_ab_btn.setStyleSheet("font-weight: bold; background-color: #3498db; color: white;")
        else:
            self.mode_ab_btn.setText("B")
            self.mode_ab_btn.setStyleSheet("font-weight: bold; background-color: #e74c3c; color: white;")
        print(f"[VideoViewer] 模式切换: {'A(标注)' if self.is_mode_a else 'B(标记)'}")
    
    def clear_delete(self):
        """清空待删除列表"""
        self.ranges = []
        self.del_frames = set()
        self.delete_list.clear()
        self.select_start = None
        self.delete_btn.setText("删除\n选中")
    
    def delete_item(self, item):
        """删除列表中的项"""
        row = self.delete_list.row(item)
        if row >= 0 and row < len(self.ranges):
            start, end = self.ranges[row]
            for i in range(start, end + 1):
                self.del_frames.discard(i)
            del self.ranges[row]
            self.delete_list.takeItem(row)
    
    def do_delete_frames(self):
        """执行删除操作"""
        if not self.ranges:
            QMessageBox.information(self, "提示", "请先选择要删除的帧范围")
            return
        
        # 收集所有待删除的帧
        delete_set = set()
        for s, e in self.ranges:
            delete_set.update(range(s, e + 1))
        
        if not delete_set:
            QMessageBox.information(self, "提示", "没有要删除的帧")
            return
        
        # 确认删除
        reply = QMessageBox.question(self, "确认", f"确定删除 {len(delete_set)} 帧吗？", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        
        # 删除帧文件和标注文件
        deleted_count = 0
        for frame_idx in sorted(delete_set, reverse=True):
            frame_path = self.frames_dir / f"frame_{frame_idx:06d}.jpg"
            label_path = self.labels_dir / f"frame_{frame_idx:06d}.json"
            if frame_path.exists():
                frame_path.unlink()
                deleted_count += 1
            if label_path.exists():
                label_path.unlink()
        
        # 重新编号：把后面的帧前移
        all_frames = sorted(self.frames_dir.glob("frame_*.jpg"))
        for new_idx, frame_path in enumerate(all_frames):
            if frame_path.stem != f"frame_{new_idx:06d}":
                new_path = self.frames_dir / f"frame_{new_idx:06d}.jpg"
                frame_path.rename(new_path)
                # 同时重命名label文件
                old_label = self.labels_dir / frame_path.stem.replace("frame_", "") + ".json"
                new_label = self.labels_dir / f"frame_{new_idx:06d}.json"
                if old_label.exists():
                    old_label.rename(new_label)
        
        # 更新总数和coco_data
        self.total_frames = len(list(self.frames_dir.glob("frame_*.jpg")))
        self.coco_data['images'] = [
            {'id': i, 'file_name': f"frame_{i:06d}.jpg", 'width': self.video_width, 'height': self.video_height, 'frame_count': i}
            for i in range(self.total_frames)
        ]
        with open(self.temp_data_path / 'annotations.json', 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, ensure_ascii=False)
        
        # 清空删除列表
        self.clear_delete()
        
        # 如果当前帧超出范围，调整到最后一帧
        if self.current_frame_idx >= self.total_frames:
            self.current_frame_idx = max(0, self.total_frames - 1)
        
        self.update_display()
        print(f"[帧删除] 已删除 {deleted_count} 帧，当前总帧数: {self.total_frames}")
        QMessageBox.information(self, "完成", f"已删除 {deleted_count} 帧\n当前总帧数: {self.total_frames}")

    def add_video_frames(self):
        """添加视频帧到当前temp_data"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择视频(支持多选)", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not file_paths:
            return
        
        print(f"[VideoViewer] 添加 {len(file_paths)} 个视频的帧...")
        
        # 获取当前最大帧号
        existing_frames = list(self.frames_dir.glob("frame_*.jpg"))
        start_idx = len(existing_frames)
        
        # 从每个视频读取帧
        for vp in file_paths:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                continue
            
            idx = start_idx
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = self.frames_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                # 创建空的label文件
                label_path = self.labels_dir / f"frame_{idx:06d}.json"
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
                idx += 1
            cap.release()
            start_idx = idx
        
        # 更新总数
        self.total_frames = len(list(self.frames_dir.glob("frame_*.jpg")))
        
        # 更新coco_data
        self.coco_data['images'] = [
            {'id': i, 'file_name': f"frame_{i:06d}.jpg", 'width': self.video_width, 'height': self.video_height, 'frame_count': i}
            for i in range(self.total_frames)
        ]
        with open(self.temp_data_path / 'annotations.json', 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, ensure_ascii=False)
        
        self.update_display()
        print(f"[VideoViewer] 添加完成，当前总帧数: {self.total_frames}")

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
        offset_x = (label_w - scaled_w) / 2
        offset_y = (label_h - scaled_h) / 2

        if offset_x <= display_x < offset_x + scaled_w and offset_y <= display_y < offset_y + scaled_h:
            video_x = int((display_x - offset_x) / self.zoom_factor)
            video_y = int((display_y - offset_y) / self.zoom_factor)
            
            # 如果是帧删除模式
            if self.is_delete_mode:
                idx = self.current_frame_idx
                display_idx = idx + 1
                if self.select_start is None:
                    self.select_start = idx
                    self.delete_btn.setText(f"选择\n帧{display_idx}")
                else:
                    start, end = min(self.select_start, idx), max(self.select_start, idx)
                    self.ranges.append((start, end))
                    self.delete_list.addItem(f"帧 {start + 1} → {end + 1} ({end - start + 1}帧)")
                    self.select_start = None
                    self.delete_btn.setText("删除\n选中")
                return
            
            # 模式B：标记模式
            if not self.is_mode_a:
                # 标记模式：标记当前帧或点击的物体
                print(f"[标记] 帧 {self.current_frame_idx + 1}, 位置 ({video_x}, {video_y})")
                return
            
            # 模式A：修改annotation的trace_id为当前ID
            panel = self.panel
            if panel and hasattr(panel, 'trace_id_input'):
                current_tid = int(panel.trace_id_input.text()) if panel.trace_id_input.text() else 1000000
                is_single = self.single_frame_radio.isChecked()
                print(f"[DEBUG] 单帧={is_single}, current_tid={current_tid}")
                frame_annotations = self._get_current_annotations()
                
                # 找出所有被点击的annotation
                clicked_anns = []
                for ann in frame_annotations:
                    bbox = ann.get('bbox', [])
                    if len(bbox) >= 4:
                        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        if x <= video_x <= x + w and y <= video_y <= y + h:
                            clicked_anns.append(ann)
                
                if len(clicked_anns) == 0:
                    # 没有点击到任何annotation
                    self.video_clicked.emit(video_x, video_y, self.current_frame_idx)
                    return
                
                if len(clicked_anns) == 1:
                    # 只有一个，直接修改
                    ann = clicked_anns[0]
                    old_tid = ann.get('track_id', 0)
                    if old_tid != current_tid:
                        if is_single:
                            self._change_trace_id_single_frame(old_tid, current_tid, video_x, video_y)
                        else:
                            self._change_trace_id_in_all_frames(old_tid, current_tid)
                    return
                
                # 多个annotation重叠，弹出选择对话框
                unique_tids = list(set(ann.get('track_id', 0) for ann in clicked_anns))
                unique_tids.sort()
                
                # 添加当前选中的trace_id作为选项
                if current_tid not in unique_tids:
                    unique_tids.append(current_tid)
                    unique_tids.sort()
                
                # 构建选项列表
                items = [f"Trace ID: {tid}" for tid in unique_tids]
                item, ok = QInputDialog.getItem(self, "选择 Trace ID", "检测到多个目标重叠，请选择要修改的 Trace ID:", items, 0, False)
                if ok and item:
                    selected_tid = int(item.split(": ")[1])
                    if is_single:
                        self._change_trace_id_single_frame(selected_tid, current_tid, video_x, video_y)
                    else:
                        self._change_trace_id_in_all_frames(selected_tid, current_tid)

    def on_bbox_drawn(self, display_x1, display_y1, display_x2, display_y2):
        scaled_w = int(self.video_width * self.zoom_factor)
        scaled_h = int(self.video_height * self.zoom_factor)
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        offset_x = (label_w - scaled_w) / 2
        offset_y = (label_h - scaled_h) / 2

        video_x1 = int((display_x1 - offset_x) / self.zoom_factor)
        video_y1 = int((display_y1 - offset_y) / self.zoom_factor)
        video_x2 = int((display_x2 - offset_x) / self.zoom_factor)
        video_y2 = int((display_y2 - offset_y) / self.zoom_factor)

        video_x1 = max(0, min(video_x1, self.video_width - 1))
        video_y1 = max(0, min(video_y1, self.video_height - 1))
        video_x2 = max(0, min(video_x2, self.video_width - 1))
        video_y2 = max(0, min(video_y2, self.video_height - 1))
        
        # 确保x1<x2, y1<y2
        if video_x1 > video_x2:
            video_x1, video_x2 = video_x2, video_x1
        if video_y1 > video_y2:
            video_y1, video_y2 = video_y2, video_y1

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

    def undo_last_bbox(self):
        if self.prompt_bboxes:
            self.prompt_bboxes.pop()
            self.update_display()
            print(f"撤销bbox，剩余: {len(self.prompt_bboxes)} 个")
        else:
            print("没有可撤销的bbox")
    
    def _get_current_annotations(self):
        """获取当前帧的annotations"""
        label_path = str(self.labels_dir / f"frame_{self.current_frame_idx:06d}.json")
        if Path(label_path).exists():
            with open(label_path) as f:
                return json.load(f)
        return []
    
    def _save_annotation(self, ann):
        """保存annotation到文件"""
        label_path = str(self.labels_dir / f"frame_{self.current_frame_idx:06d}.json")
        annotations = self._get_current_annotations()
        for i, a in enumerate(annotations):
            if a.get('bbox') == ann.get('bbox') and a.get('track_id') == ann.get('track_id', 0) - 1:
                annotations[i] = ann
                break
        else:
            for i, a in enumerate(annotations):
                if a.get('bbox') == ann.get('bbox'):
                    annotations[i] = ann
                    break
        with open(label_path, 'w') as f:
            json.dump(annotations, f)
        if self.controller and hasattr(self.controller, 'refresh_trace_id_list'):
            self.panel.refresh_trace_id_list()
    
    def _change_trace_id_in_all_frames(self, old_tid, new_tid):
        """批量修改所有帧中指定track_id的annotation"""
        undo_changes = []
        for frame_file in sorted(self.labels_dir.glob("frame_*.json")):
            try:
                with open(frame_file) as f:
                    annotations = json.load(f)
                new_anns = []
                for ann in annotations:
                    if ann.get('track_id', 0) == old_tid:
                        bbox_key = self._get_bbox_key(ann.get('bbox', []))
                        frame_idx = int(frame_file.stem.split('_')[1])
                        undo_changes.append({
                            'frame_idx': frame_idx,
                            'bbox_key': bbox_key,
                            'old_trace_id': old_tid,
                            'new_trace_id': new_tid
                        })
                        ann['track_id'] = new_tid
                    new_anns.append(ann)
                with open(frame_file, 'w') as f:
                    json.dump(new_anns, f)
            except:
                pass
        # 通知主面板
        if undo_changes and self.panel and hasattr(self.panel, 'push_undo'):
            self.panel.push_undo(undo_changes)
        if self.panel and hasattr(self.panel, 'refresh_trace_id_list'):
            self.panel.refresh_trace_id_list()
        self.update_display()
        print(f"[多帧修改] 共修改 {len(undo_changes)} 帧")
    
    def _get_bbox_key(self, bbox):
        """生成bbox的唯一键"""
        if len(bbox) >= 4:
            return f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}"
        return ""
    
    def _change_trace_id_single_frame(self, old_tid, new_tid, click_x, click_y):
        """修改当前帧中被点击的那个bbox的trace_id"""
        frame_idx = self.current_frame_idx
        frame_file = self.labels_dir / f"frame_{frame_idx:06d}.json"
        if not frame_file.exists():
            return
        
        changed = False
        try:
            with open(frame_file) as f:
                annotations = json.load(f)
            for ann in annotations:
                bbox = ann.get('bbox', [])
                if len(bbox) >= 4:
                    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    # 检查点击坐标是否在这个bbox内
                    if x <= click_x <= x + w and y <= click_y <= y + h:
                        bbox_key = self._get_bbox_key(bbox)
                        ann['track_id'] = new_tid
                        changed = True
                        break  # 只修改第一个匹配的bbox
            if changed:
                with open(frame_file, 'w') as f:
                    json.dump(annotations, f)
        except:
            return
        
        if changed and self.panel and hasattr(self.panel, 'push_undo'):
            undo_entry = {
                'frame_idx': frame_idx,
                'bbox_key': bbox_key,
                'old_trace_id': old_tid,
                'new_trace_id': new_tid
            }
            self.panel.push_undo(undo_entry)
        if self.panel and hasattr(self.panel, 'refresh_trace_id_list'):
            self.panel.refresh_trace_id_list()
        self.update_display()
        print(f"[单帧修改] 修改了1个bbox")
    
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
                    cx, cy = tp['x'], tp['y']
                    cv2.circle(annotated_frame, (cx, cy), 6, (0, 255, 0), -1)
                    cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 0), 2)
                    label = str(self.controller.track_id_points.index(tp))
                    cv2.putText(annotated_frame, label, (cx + 8, cy - 5),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # prompt_bboxes只在提示帧可见
        if self.panel and hasattr(self.panel, 'prompt_frame_idx') and self.panel.prompt_frame_idx >= 0:
            if self.current_frame_idx == self.panel.prompt_frame_idx:
                for bbox in self.prompt_bboxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"prompt {self.prompt_bboxes.index(bbox) + 1}"
                    cv2.putText(annotated_frame, label, (x1, max(10, y1 - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 固定框模式：bboxes始终可见（黄色）
        if self.panel and hasattr(self.panel, 'fixed_bbox_mode') and self.panel.fixed_bbox_mode:
            if self.current_frame_idx == getattr(self.panel, 'fixed_bbox_frame_idx', 0):
                for bbox in self.prompt_bboxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    label = f"fixed {self.prompt_bboxes.index(bbox) + 1}"
                    cv2.putText(annotated_frame, label, (x1, max(10, y1 - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
    
    def backward(self):
        """上一帧"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_display()
    
    def forward(self):
        """下一帧"""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.update_display()
    
    def toggle_play(self):
        """播放/暂停"""
        if self.timer.isActive():
            self.timer.stop()
        else:
            # 设置播放间隔，约30fps
            self.timer.start(33)


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
