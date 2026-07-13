#!/usr/bin/env python3
"""视频查看器 - 纯视图层，控制逻辑由 video_control.VideoController 提供"""

import sys
from pathlib import Path
import random

import cv2
import numpy as np
import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton)
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

        # 单帧/多帧选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("修改模式:"))
        self.single_frame_radio = QRadioButton("单帧")
        self.single_frame_radio.setChecked(False)  # 默认多帧
        self.multi_frame_radio = QRadioButton("多帧")
        self.multi_frame_radio.setChecked(True)  # 默认多帧
        mode_layout.addWidget(self.single_frame_radio)
        mode_layout.addWidget(self.multi_frame_radio)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

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
        else:
            super().keyPressEvent(event)

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
            # 单击修改annotation的trace_id为当前ID
            # controller实际上是UnifiedPanel实例
            panel = self.controller
            if panel and hasattr(panel, 'trace_id_input'):
                current_tid = int(panel.trace_id_input.text()) if panel.trace_id_input.text() else 1000000
                is_single = self.single_frame_radio.isChecked()
                print(f"[DEBUG] 单帧={is_single}, current_tid={current_tid}")
                frame_annotations = self._get_current_annotations()
                for ann in frame_annotations:
                    bbox = ann.get('bbox', [])
                    if len(bbox) >= 4:
                        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        if x <= video_x <= x + w and y <= video_y <= y + h:
                            old_tid = ann.get('track_id', 0)
                            if old_tid != current_tid:
                                if is_single:
                                    self._change_trace_id_single_frame(old_tid, current_tid, video_x, video_y)
                                else:
                                    self._change_trace_id_in_all_frames(old_tid, current_tid)
                            return
            self.video_clicked.emit(video_x, video_y, self.current_frame_idx)

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
            self.controller.refresh_trace_id_list()
    
    def _change_trace_id_in_all_frames(self, old_tid, new_tid):
        """批量修改所有帧中指定track_id的annotation"""
        undo_data = {}
        for frame_file in sorted(self.labels_dir.glob("frame_*.json")):
            try:
                with open(frame_file) as f:
                    annotations = json.load(f)
                frame_undo = {}
                new_anns = []
                for ann in annotations:
                    if ann.get('track_id', 0) == old_tid:
                        bbox_key = self._get_bbox_key(ann.get('bbox', []))
                        frame_undo[bbox_key] = old_tid
                        ann['track_id'] = new_tid
                    new_anns.append(ann)
                if frame_undo:
                    frame_idx = int(frame_file.stem.split('_')[1])
                    undo_data[frame_idx] = frame_undo
                with open(frame_file, 'w') as f:
                    json.dump(new_anns, f)
            except:
                pass
        # 通知主面板
        if undo_data and self.controller and hasattr(self.controller, 'push_undo'):
            self.controller.push_undo(undo_data)
        if self.controller and hasattr(self.controller, 'refresh_trace_id_list'):
            self.controller.refresh_trace_id_list()
        self.update_display()
        print(f"[多帧修改] 共修改 {len(undo_data)} 帧")
    
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
        
        undo_data = {frame_idx: {}}
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
                        undo_data[frame_idx][bbox_key] = old_tid
                        ann['track_id'] = new_tid
                        changed = True
                        break  # 只修改第一个匹配的bbox
            if changed:
                with open(frame_file, 'w') as f:
                    json.dump(annotations, f)
        except:
            return
        
        if changed and self.controller and hasattr(self.controller, 'push_undo'):
            self.controller.push_undo(undo_data)
        if self.controller and hasattr(self.controller, 'refresh_trace_id_list'):
            self.controller.refresh_trace_id_list()
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
        if self.controller and hasattr(self.controller, 'prompt_frame_idx') and self.controller.prompt_frame_idx >= 0:
            if self.current_frame_idx == self.controller.prompt_frame_idx:
                for bbox in self.prompt_bboxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"prompt {self.prompt_bboxes.index(bbox) + 1}"
                    cv2.putText(annotated_frame, label, (x1, max(10, y1 - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 固定框模式：bboxes始终可见（黄色）
        if self.controller and hasattr(self.controller, 'fixed_bbox_mode') and self.controller.fixed_bbox_mode:
            if self.current_frame_idx == getattr(self.controller, 'fixed_bbox_frame_idx', 0):
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
