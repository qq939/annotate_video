#!/usr/bin/env python3
"""视频标注工具 - 统一控制面板，控制逻辑委托给 video_control.VideoController"""

import sys
import random
import shutil
import cv2
import numpy as np
import json
import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QLineEdit, QFileDialog, QGroupBox, QTextEdit, QMessageBox, QListWidget, QSizePolicy, QDialog, QInputDialog, QCheckBox, QToolButton, QMenu)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.Qt import QDragEnterEvent, QDropEvent


class TrimSlider(QSlider):
    """带A-B区间高亮和竖线指示器的自定义进度条"""
    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.range_a = None  # 选择区间
        self.range_b = None
        self.delete_ranges = []  # 待删除区间列表
        self.setMinimum(0)
        self.setMaximum(100)
        
    def setDeleteRanges(self, ranges):
        self.delete_ranges = ranges
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.maximum() <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        total = self.maximum() - self.minimum()
        if total <= 0:
            return
        
        # 高亮待删除区间
        if self.delete_ranges:
            for start, end in self.delete_ranges:
                x1 = int((start - self.minimum()) / total * w)
                x2 = int((end - self.minimum()) / total * w)
                c = QColor(255, 100, 100, 150)
                painter.fillRect(x1, 0, x2 - x1, self.height(), c)
        
        # 竖线指示器
        val = self.value()
        x = int((val - self.minimum()) / total * w)
        green_color = Qt.green
        painter.setPen(green_color)
        painter.drawLine(x, 0, x, self.height())


from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QFont, QPixmap, QKeySequence
from PyQt5.QtWidgets import QShortcut

from video_control import VideoController
from annotate_video import TEMP_DATA_MID_DIR

def _patch_ultralytics_compile():
    """修复ultralytics中torch.compile的兼容性问题"""
    try:
        import ultralytics.models.sam.build_sam3 as build_file
        file_path = build_file.__file__
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'compile_mode=None_mode' in content:
            content = content.replace('compile_mode=None_mode', "compile_mode='default'")
            content = content.replace('compile_mode=compile', "compile_mode='default'")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[PATCH] build_sam3.py patched for torch.compile compatibility")
        
        import ultralytics.models.sam.sam3.vitdet as vitdet_file
        vitdet_path = vitdet_file.__file__
        with open(vitdet_path, 'r', encoding='utf-8') as f:
            vit_content = f.read()
        if 'torch.compile(self.forward' in vit_content:
            vit_content = vit_content.replace(
                'self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)',
                '# self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)  # Disabled for compatibility'
            )
            with open(vitdet_path, 'w', encoding='utf-8') as f:
                f.write(vit_content)
            print("[PATCH] vitdet.py patched for torch.compile compatibility")
    except Exception as e:
        print(f"[PATCH] Warning: Could not patch ultralytics: {e}")

_patch_ultralytics_compile()

_SAM3_SEMANTIC_PATCHED = False
def _patch_sam3_video_semantic():
    global _SAM3_SEMANTIC_PATCHED
    if _SAM3_SEMANTIC_PATCHED:
        return
    _SAM3_SEMANTIC_PATCHED = True
    import torch
    from ultralytics.utils import ops as ultralytics_ops
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    _orig = SAM3VideoSemanticPredictor.add_prompt

    def _new_add_prompt(self, frame_idx, text=None, bboxes=None, labels=None, inference_state=None):
        if bboxes is None:
            return _orig(self, frame_idx, text, bboxes, labels, inference_state)
        inference_state = inference_state or self.inference_state
        text_batch = [text] if isinstance(text, str) else (list(text) if text else [])
        n = len(text_batch)
        inference_state["text_prompt"] = text if text else None
        text_ids = torch.arange(n, device=self.device, dtype=torch.long)
        inference_state["text_ids"] = text_ids
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)
        _raw = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
        _raw = _raw[None] if _raw.ndim == 1 else _raw
        _raw = ultralytics_ops.xyxy2xywh(_raw)
        _raw[:, 0::2] /= self.batch[1][0].shape[1]
        _raw[:, 1::2] /= self.batch[1][0].shape[0]
        nb = len(_raw)
        if labels is None:
            _lbl_arr = np.ones(nb)
        else:
            _lbl_arr = np.array(labels)
        _lbl = torch.as_tensor(_lbl_arr, dtype=torch.int32, device=self.device)
        _raw = _raw.view(-1, 1, 4)
        _lbl = _lbl.view(-1, 1)
        if n > 1:
            _raw = _raw.repeat(1, n, 1)
            _lbl = _lbl.repeat(1, n)
        geometric_prompt = self._get_dummy_prompt(num_prompts=n)
        for i in range(len(_raw)):
            geometric_prompt.append_boxes(_raw[[i]], _lbl[[i]])
        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, out

    SAM3VideoSemanticPredictor.add_prompt = _new_add_prompt

BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255),
]


class RotatableBBoxEditorWidget(QWidget):
    """可旋转的bbox编辑器，支持调整倾斜角度"""
    bbox_changed = pyqtSignal(int, dict)
    editing_finished = pyqtSignal()

    def __init__(self, frame, boxes, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.boxes = boxes
        self.editing_index = -1
        self.dragging_handle = -1
        self.rotating = False
        self.last_pos = QPoint()

        h, w = frame.shape[:2]
        self.setFixedSize(w, h)
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        self.update_image = True

    def set_editing_box(self, index):
        self.editing_index = index
        self.update()

    def _get_box_corners(self, box):
        cx = (box['x1'] + box['x2']) / 2
        cy = (box['y1'] + box['y2']) / 2
        w = box['x2'] - box['x1']
        h = box['y2'] - box['y1']
        angle = box.get('angle', 0)
        import math
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        corners = [
            (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
        ]
        rotated = []
        for dx, dy in corners:
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            rotated.append((rx, ry))
        return rotated

    def _point_in_handle(self, pt, cx, cy, r=8):
        import math
        dist = math.sqrt((pt.x() - cx)**2 + (pt.y() - cy)**2)
        return dist <= r

    def _get_center(self, box):
        cx = (box['x1'] + box['x2']) / 2
        cy = (box['y1'] + box['y2']) / 2
        return cx, cy

    def mousePressEvent(self, event):
        if self.editing_index < 0 or self.editing_index >= len(self.boxes):
            return

        box = self.boxes[self.editing_index]
        corners = self._get_box_corners(box)
        cx, cy = self._get_center(box)

        import math
        for i, (px, py) in enumerate(corners):
            if self._point_in_handle(event.pos(), px, py, 12):
                self.dragging_handle = i
                self.last_pos = event.pos()
                self.rotating = False
                return

        center_x, center_y = self._get_center(box)
        if self._point_in_handle(event.pos(), center_x, center_y, 20):
            self.rotating = True
            self.last_pos = event.pos()
            return

        self.bbox_changed.emit(self.editing_index, box)
        self.editing_finished.emit()

    def mouseMoveEvent(self, event):
        if self.editing_index < 0 or self.editing_index >= len(self.boxes):
            return
        if self.dragging_handle < 0 and not self.rotating:
            return

        box = self.boxes[self.editing_index]
        corners = self._get_box_corners(box)
        cx, cy = self._get_center(box)

        import math

        if self.rotating:
            old_angle = math.atan2(self.last_pos.y() - cy, self.last_pos.x() - cx)
            new_angle = math.atan2(event.y() - cy, event.x() - cx)
            delta = math.degrees(new_angle - old_angle)
            box['angle'] = box.get('angle', 0) + delta
            self.last_pos = event.pos()
            self.update()
            return

        hx, hy = corners[self.dragging_handle]
        dx = event.x() - hx
        dy = event.y() - hy

        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)

        new_x1 = box['x1'] + dx if self.dragging_handle in [0, 3] else box['x1']
        new_y1 = box['y1'] + dy if self.dragging_handle in [0, 1] else box['y1']
        new_x2 = box['x2'] + dx if self.dragging_handle in [1, 2] else box['x2']
        new_y2 = box['y2'] + dy if self.dragging_handle in [2, 3] else box['y2']

        if new_x2 > new_x1 + 10 and new_y2 > new_y1 + 10:
            box['x1'] = max(0, new_x1)
            box['y1'] = max(0, new_y1)
            box['x2'] = min(self.width(), new_x2)
            box['y2'] = min(self.height(), new_y2)

        self.last_pos = event.pos()
        self.update()
        self.bbox_changed.emit(self.editing_index, box)

    def mouseReleaseEvent(self, event):
        self.dragging_handle = -1
        self.rotating = False
        self.bbox_changed.emit(self.editing_index, self.boxes[self.editing_index])

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.update_image:
            painter.drawImage(0, 0, self.qimage)
        else:
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
            painter.drawImage(0, 0, qimg)

        for i, box in enumerate(self.boxes):
            color = QColor(*box['color'][::-1])
            pen = QPen(color, 2)
            painter.setPen(pen)

            corners = self._get_box_corners(box)
            polygon = [QPoint(int(c[0]), int(c[1])) for c in corners]
            painter.drawPolygon(polygon)

            if i == self.editing_index:
                painter.setPen(QPen(QColor(255, 255, 0), 2))
                for px, py in corners:
                    painter.drawEllipse(QPoint(int(px), int(py)), 8, 8)

                cx, cy = self._get_center(box)
                painter.drawEllipse(QPoint(int(cx), int(cy)), 12, 12)
                painter.drawLine(QPoint(int(cx), int(cy)), QPoint(int(cx), int(cy - 40)))

            painter.setFont(QFont("Arial", 14))
            painter.drawText(box['x1'], box['y1'] - 5, f"{i + 1}")


class AnnotationImageWidget(QWidget):
    box_added = pyqtSignal()
    def __init__(self, frame, boxes, color_index, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.boxes = boxes
        self.color_index = color_index
        self.drawing = False
        self.start_point = QPoint()
        self.current_rect = QRect()
        h, w = frame.shape[:2]
        self.setFixedSize(w, h)
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.current_rect = QRect(self.start_point, self.start_point)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_rect = QRect(self.start_point, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, event.pos()).normalized()
            if rect.width() > 5 and rect.height() > 5:
                color = BOX_COLORS[self.color_index[0] % len(BOX_COLORS)]
                self.boxes.append({
                    'x1': rect.left(), 'y1': rect.top(),
                    'x2': rect.right(), 'y2': rect.bottom(),
                    'color': color
                })
                self.color_index[0] += 1
                self.box_added.emit()
            self.current_rect = QRect()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.qimage)
        for i, box in enumerate(self.boxes):
            color = QColor(*box['color'][::-1])
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(box['x1'], box['y1'], box['x2'] - box['x1'], box['y2'] - box['y1'])
            painter.setFont(QFont("Arial", 14))
            painter.drawText(box['x1'], box['y1'] - 5, f"目标 {i + 1}")
        if self.drawing and not self.current_rect.isNull():
            color = QColor(*BOX_COLORS[self.color_index[0] % len(BOX_COLORS)][::-1])
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)


class AnnotationDialog(QDialog):
    def __init__(self, video_path, parent=None, scale=1.0):
        super().__init__(parent)
        self.video_path = video_path
        self.scale = scale
        self.boxes = []
        self.color_index = [0]
        self._read_first_frame()
        self._setup_ui()
        self._setup_shortcut()

    def _read_first_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"无法读取视频首帧: {self.video_path}")
        self.frame = frame

    def _setup_ui(self):
        h, w = self.frame.shape[:2]
        if self.scale != 1.0:
            dw = max(1, int(w * self.scale))
            dh = max(1, int(h * self.scale))
            self.display_frame = cv2.resize(self.frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            self.display_frame = self.frame
        dh, dw = self.display_frame.shape[:2]
        self.setWindowTitle("视频标注")
        self.setFixedSize(dw, dh + 36)
        self.setMaximumSize(dw, dh + 36)
        self.setMinimumSize(dw, dh + 36)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        central = QWidget()
        central.setFixedSize(dw, dh)
        central.setStyleSheet("background: #111;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QWidget()
        header.setFixedSize(dw, 36)
        header.setStyleSheet("background: rgba(0,0,0,180);")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(4)

        instr_label = QLabel("框选目标 | C:撤销 | Q:退出")
        instr_label.setStyleSheet("color: #ccc; font-size: 12px;")
        header_layout.addWidget(instr_label)
        header_layout.addStretch()

        self.undo_btn = QPushButton("撤销")
        self.undo_btn.setFixedSize(60, 28)
        self.undo_btn.setStyleSheet(
            "QPushButton { background: #555; color: white; border: none; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #666; }"
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self.undo_btn.clicked.connect(self._undo_last)
        self.undo_btn.setEnabled(False)
        header_layout.addWidget(self.undo_btn)

        self.done_btn = QPushButton("✓ 完成标注")
        self.done_btn.setFixedSize(100, 28)
        self.done_btn.setStyleSheet(
            "QPushButton { background: #00CC00; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #009900; }"
        )
        self.done_btn.clicked.connect(self.accept)
        header_layout.addWidget(self.done_btn)

        main_layout.addWidget(header)

        self.img_widget = AnnotationImageWidget(self.display_frame, self.boxes, self.color_index)
        self.img_widget.setFocus()
        self.img_widget.box_added.connect(lambda: self.undo_btn.setEnabled(True))
        main_layout.addWidget(self.img_widget)

    def _setup_shortcut(self):
        QShortcut(QKeySequence("c"), self).activated.connect(self._undo_last)
        QShortcut(QKeySequence("q"), self).activated.connect(self.reject)
        QShortcut(QKeySequence("C"), self).activated.connect(self._undo_last)
        QShortcut(QKeySequence("Q"), self).activated.connect(self.reject)

    def _undo_last(self):
        if self.boxes:
            self.boxes.pop()
            self.color_index[0] = max(0, self.color_index[0] - 1)
            self.img_widget.update()
            self.undo_btn.setEnabled(len(self.boxes) > 0)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_C, Qt.Key_c):
            self._undo_last()
        elif key in (Qt.Key_Q, Qt.Key_q):
            self.reject()
        else:
            super().keyPressEvent(event)

    def get_boxes(self):
        if self.scale != 1.0:
            inv = 1.0 / self.scale
            result = []
            for b in self.boxes:
                result.append({
                    'x1': int(b['x1'] * inv), 'y1': int(b['y1'] * inv),
                    'x2': int(b['x2'] * inv), 'y2': int(b['y2'] * inv),
                    'angle': b.get('angle', 0),
                    'color': b.get('color', BOX_COLORS[0])
                })
            return result
        return self.boxes


class AngleAdjustDialog(QDialog):
    def __init__(self, frame, boxes, parent=None, scale=1.0):
        super().__init__(parent)
        self.frame = frame
        self.boxes = boxes
        self.scale = scale
        self.current_index = 0

        self._setup_ui()
        self._show_box(0)

    def _setup_ui(self):
        h, w = self.frame.shape[:2]
        if self.scale != 1.0:
            dw = max(1, int(w * self.scale))
            dh = max(1, int(h * self.scale))
            self.display_frame = cv2.resize(self.frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            self.display_frame = self.frame

        dh, dw = self.display_frame.shape[:2]

        self.setWindowTitle("调整标注框角度")
        self.setFixedSize(dw, dh + 80)
        self.setMaximumSize(dw, dh + 80)
        self.setMinimumSize(dw, dh + 80)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QWidget()
        header.setFixedSize(dw, 50)
        header.setStyleSheet("background: rgba(0,0,0,200);")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(8)

        self.info_label = QLabel(f"调整标注框 (1/{len(self.boxes)})")
        self.info_label.setStyleSheet("color: #ccc; font-size: 14px; font-weight: bold;")
        header_layout.addWidget(self.info_label)

        self.angle_label = QLabel("角度: 0")
        self.angle_label.setStyleSheet("color: #ff0; font-size: 14px;")
        self.angle_label.setFixedWidth(80)
        header_layout.addWidget(self.angle_label)

        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setMinimum(-180)
        self.angle_slider.setMaximum(180)
        self.angle_slider.setValue(0)
        self.angle_slider.setFixedWidth(200)
        self.angle_slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #555; height: 8px; background: #333; }
            QSlider::handle:horizontal { background: #00CC00; width: 18px; margin: -5px 0; }
        """)
        self.angle_slider.valueChanged.connect(self._on_angle_slider_changed)
        header_layout.addWidget(self.angle_slider)

        self.prev_btn = QPushButton("< 上一个")
        self.prev_btn.setFixedSize(80, 28)
        self.prev_btn.setStyleSheet("background: #555; color: white; border: none; border-radius: 4px;")
        self.prev_btn.clicked.connect(self._prev_box)
        header_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("下一个 >")
        self.next_btn.setFixedSize(80, 28)
        self.next_btn.setStyleSheet("background: #555; color: white; border: none; border-radius: 4px;")
        self.next_btn.clicked.connect(self._next_box)
        header_layout.addWidget(self.next_btn)

        self.done_btn = QPushButton("✓ 完成")
        self.done_btn.setFixedSize(80, 28)
        self.done_btn.setStyleSheet("background: #00CC00; color: white; border: none; border-radius: 4px; font-weight: bold;")
        self.done_btn.clicked.connect(self.accept)
        header_layout.addWidget(self.done_btn)

        main_layout.addWidget(header)

        self.editor_widget = RotatableBBoxEditorWidget(self.display_frame, self.boxes)
        self.editor_widget.set_editing_box(0)
        self.editor_widget.bbox_changed.connect(self._on_bbox_changed)
        self.editor_widget.editing_finished.connect(self._on_editing_finished)
        main_layout.addWidget(self.editor_widget)

        if len(self.boxes) <= 1:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def _show_box(self, index):
        self.current_index = index
        self.editor_widget.set_editing_box(index)
        box = self.boxes[index]
        angle = box.get('angle', 0)
        self.info_label.setText(f"调整标注框 ({index + 1}/{len(self.boxes)})")
        self.angle_label.setText(f"角度: {int(angle)}")
        self.angle_slider.blockSignals(True)
        self.angle_slider.setValue(int(angle))
        self.angle_slider.blockSignals(False)
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.boxes) - 1)

    def _on_angle_slider_changed(self, value):
        self.boxes[self.current_index]['angle'] = value
        self.angle_label.setText(f"角度: {value}")
        self.editor_widget.update()

    def _on_bbox_changed(self, index, box):
        self.boxes[index] = box

    def _on_editing_finished(self):
        pass

    def _prev_box(self):
        if self.current_index > 0:
            self._show_box(self.current_index - 1)

    def _next_box(self):
        if self.current_index < len(self.boxes) - 1:
            self._show_box(self.current_index + 1)

    def get_boxes(self):
        return self.boxes


class DragLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            file_path = url.toLocalFile()
            if file_path:
                self.setText(file_path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class TrimMidDialog(QDialog):
    """temp_data_mid帧删除对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames_dir = Path(TEMP_DATA_MID_DIR) / "frames"
        self.labels_dir = Path(TEMP_DATA_MID_DIR) / "labels"
        
        # 获取帧列表
        self.frames = sorted([f for f in self.frames_dir.glob("frame_*.jpg")])
        self.total = len(self.frames)
        if self.total == 0:
            QMessageBox.warning(self, "提示", "没有找到帧")
            return
        
        self.setWindowTitle("0、视频帧删除")
        self.setMinimumSize(1200, 900)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 视频预览
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel { background-color: #222; color: white; border: 1px solid #444; }")
        self.label.mousePressEvent = self.on_label_click
        layout.addWidget(self.label, 1)
        
        # 进度条
        self.slider = TrimSlider(self)
        self.slider.setMaximum(self.total - 1)
        self.slider.sliderMoved.connect(self.show_frame)
        layout.addWidget(self.slider)
        
        # 帧信息
        info = QHBoxLayout()
        info.addWidget(QLabel("帧:"))
        self.frame_label = QLabel("0/0")
        info.addWidget(self.frame_label)
        info.addStretch()
        layout.addLayout(info)
        
        # 控制按钮
        controls = QHBoxLayout()
        for txt, fn in [("◀◀", self.backward), ("▶", self.toggle_play), ("▶▶", self.forward), ("清空", self.clear)]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            controls.addWidget(b)
        controls.addStretch()
        layout.addLayout(controls)
        
        # 播放定时器
        self.play_timer = None
        self.is_playing = False
        
        # 待删除列表
        list_layout = QHBoxLayout()
        list_layout.addWidget(QLabel("待删除片段:"))
        self.delete_list = QListWidget()
        self.delete_list.itemDoubleClicked.connect(self.delete_item)
        list_layout.addWidget(self.delete_list)
        self.del_btn = QPushButton("删除选中")
        self.del_btn.clicked.connect(self.delete_selected)
        list_layout.addWidget(self.del_btn)
        layout.addLayout(list_layout)
        
        # 生成按钮
        gb = QPushButton("删除选中帧")
        gb.clicked.connect(self.generate)
        layout.addWidget(gb)
        
        # 初始化
        self.select_start = None
        self.ranges = []
        self.del_frames = set()
        self.show_frame(0)
    
    def show_frame(self, idx):
        if idx < 0 or idx >= self.total:
            return
        frame_path = self.frames[idx]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            return
        
        # 加载标注并渲染bbox
        label_file = self.labels_dir / f"frame_{idx:06d}.json"
        if label_file.exists():
            import json
            with open(label_file, encoding='utf-8') as f:
                annotations = json.load(f)
            overlay = frame.copy()
            for ann in annotations:
                bbox = ann.get('bbox', [])
                track_id = ann.get('track_id', 0)
                color = self._get_color(track_id)
                if bbox:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(overlay, str(track_id), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = min(1200 / w, 800 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_small = cv2.resize(frame, (new_w, new_h))
        qimg = QImage(frame_small.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))
        self.frame_label.setText(f"{idx}/{self.total}")
        self.slider.setValue(idx)
        self.slider.setDeleteRanges(self.ranges)
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_timer = QTimer()
            self.play_timer.timeout.connect(self.forward)
            self.play_timer.start(100)
        elif self.play_timer:
            self.play_timer.stop()
    
    def _get_color(self, track_id):
        """获取track_id对应的颜色"""
        COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (128, 0, 128)]
        return COLORS[track_id % len(COLORS)]
    
    def backward(self):
        self.show_frame(self.slider.value() - 1)
    
    def forward(self):
        self.show_frame(self.slider.value() + 1)
    
    def on_label_click(self, event):
        idx = self.slider.value()
        if self.select_start is None:
            self.select_start = idx
            self.del_btn.setText(f"选择帧{idx}")
        else:
            s, e = min(self.select_start, idx), max(self.select_start, idx)
            self.ranges.append((s, e))
            self.delete_list.addItem(f"帧 {s} → {e}")
            self.select_start = None
            self.del_btn.setText("删除选中")
    
    def clear(self):
        self.ranges = []
        self.del_frames = set()
        self.delete_list.clear()
        self.select_start = None
        self.del_btn.setText("删除选中")
        self.slider.setDeleteRanges([])
    
    def delete_selected(self):
        row = self.delete_list.currentRow()
        if row >= 0:
            self.delete_list.takeItem(row)
            del self.ranges[row]
            self.slider.setDeleteRanges(self.ranges)
    
    def delete_item(self, item):
        row = self.delete_list.row(item)
        self.delete_list.takeItem(row)
        del self.ranges[row]
        self.slider.setDeleteRanges(self.ranges)
    
    def generate(self):
        delete_set = set(self.del_frames)
        for s, e in self.ranges:
            delete_set.update(range(s, e + 1))
        
        if not delete_set:
            QMessageBox.warning(self, "提示", "没有要删除的帧")
            return
        
        # 删除帧文件和标注
        import os
        for idx in sorted(delete_set, reverse=True):
            frame_file = self.frames_dir / f"frame_{idx:06d}.jpg"
            if frame_file.exists():
                os.remove(str(frame_file))
            label_file = self.labels_dir / f"frame_{idx:06d}.json"
            if label_file.exists():
                os.remove(str(label_file))
        
        self.close()
    
    def open_trim_mid_dialog(self):
        pass  # 不再使用单独窗口


class TrimDialog(QDialog):
    """视频裁剪对话框"""
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        # 支持多视频路径（用 | 分隔）
        self.video_paths = video_path.split('|') if '|' in video_path else [video_path]
        self.video_path = self.video_paths[0]  # 第一个视频作为主视频
        self.parent_panel = parent  # 保存父窗口引用用于获取视频名称
        
        # 先获取视频尺寸（使用第一个视频）
        cap = cv2.VideoCapture(self.video_path)
        vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # 窗口缩小2倍，视频为主体
        win_w = int(max(1200, vw + 100) / 2)
        win_h = int((vh + 200) / 2)
        self.setMinimumSize(win_w, win_h)
        self.setWindowTitle("视频帧删除")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)
        
        # 缩放滑块
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("缩放:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(200)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addStretch()
        layout.addLayout(zoom_layout)
        
        # 视频预览（主体）
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel { background-color: #222; color: white; border: 1px solid #444; }")
        self.label.mousePressEvent = self.on_label_click
        layout.addWidget(self.label, 1)  # stretch=1 让视频占主体
        
        # 进度条
        self.slider = TrimSlider(self)
        self.slider.setFixedHeight(50)
        self.slider.setStyleSheet("font-size: 14px;")
        self.slider.sliderMoved.connect(self.seek)
        layout.addWidget(self.slider)
        
        # 帧信息
        frame_info = QHBoxLayout()
        fl = QLabel("帧:")
        fl.setStyleSheet("font-size: 14px;")
        frame_info.addWidget(fl)
        self.frame_label = QLabel("0/0")
        self.frame_label.setStyleSheet("font-size: 14px;")
        frame_info.addWidget(self.frame_label)
        frame_info.addStretch()
        layout.addLayout(frame_info)
        
        # 控制按钮
        controls = QHBoxLayout()
        for txt, fn in [("◀◀", self.backward), ("▶", self.toggle_play), ("▶▶", self.forward), ("清空", self.clear)]:
            b = QPushButton(txt)
            b.setFixedHeight(34)
            b.setStyleSheet("font-size: 14px;")
            b.clicked.connect(fn)
            controls.addWidget(b)
        # 添加视频按钮（在预览内部）
        add_btn = QPushButton("+添加视频")
        add_btn.setFixedHeight(34)
        add_btn.setStyleSheet("font-size: 14px; background-color: #4CAF50; color: white;")
        add_btn.clicked.connect(self.add_video_to_trim)
        controls.addWidget(add_btn)
        controls.addStretch()
        layout.addLayout(controls)
        
        # 待删除列表
        list_layout = QHBoxLayout()
        dl = QLabel("待删除片段:")
        dl.setStyleSheet("font-size: 14px;")
        list_layout.addWidget(dl)
        self.delete_list = QListWidget()
        self.delete_list.setFixedHeight(100)
        self.delete_list.setStyleSheet("font-size: 14px;")
        self.delete_list.itemDoubleClicked.connect(self.delete_item)
        list_layout.addWidget(self.delete_list)
        self.del_btn = QPushButton("删除\n选中")
        self.del_btn.setFixedWidth(80)
        self.del_btn.setStyleSheet("font-size: 14px;")
        self.del_btn.clicked.connect(self.delete_selected)
        list_layout.addWidget(self.del_btn)
        layout.addLayout(list_layout)
        
        # 生成按钮
        gb = QPushButton("生成裁剪视频")
        gb.setFixedHeight(40)
        gb.setStyleSheet("font-size: 16px; font-weight: bold;")
        gb.clicked.connect(self.generate)
        layout.addWidget(gb)
        
        # 初始化
        self.cap = None
        self.total = 0
        self.fps = 30
        self.is_playing = False
        self.timer = None
        self.select_start = None
        self.ranges = []
        self.del_frames = set()
        self.zoom_scale = 1.0
        self.load_video()
    
    def on_zoom_changed(self, value):
        self.zoom_scale = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self.show_frame(self.slider.value())
    
    def load_video(self):
        # 支持多视频：读取所有视频的帧
        self.video_caps = []
        self.video_frame_counts = []
        self.video_frame_starts = [0]  # 每个视频的起始帧号
        
        for vp in self.video_paths:
            cap = cv2.VideoCapture(vp)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_caps.append(cap)
                self.video_frame_counts.append(frame_count)
                self.video_frame_starts.append(self.video_frame_starts[-1] + frame_count)
            else:
                self.video_caps.append(None)
                self.video_frame_counts.append(0)
                self.video_frame_starts.append(self.video_frame_starts[-1])
        
        self.total = self.video_frame_starts[-1]  # 总帧数
        self.fps = cv2.VideoCapture(self.video_paths[0]).get(cv2.CAP_PROP_FPS) if self.video_caps[0] else 30
        self.slider.setMaximum(self.total - 1)
        self.show_frame(0)
        # 初始显示一帧以确定label尺寸
        QTimer.singleShot(100, lambda: self.show_frame(0))
        
        # 自动保存到临时视频
        QTimer.singleShot(200, self._save_temp_video)
    
    def _save_temp_video(self):
        """保存所有视频帧到temp/temp.mp4（流式写入，不占用大量内存）"""
        if hasattr(self, 'temp_video_path') and Path(self.temp_video_path).exists():
            return  # 已存在则跳过
        
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = str(temp_dir / "temp.mp4")
        
        # 获取视频信息
        first_cap = self.video_caps[0] if self.video_caps else None
        if first_cap is None:
            return
        
        w = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, self.fps, (w, h))
        
        # 流式读取并写入
        frame_count = 0
        for cap in self.video_caps:
            if cap is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"[Trim] 已保存 {frame_count} 帧...")
        
        out.release()
        
        self.temp_video_path = temp_path
        print(f"[Trim] 临时视频已保存: {temp_path}, 总帧数: {frame_count}")
    
    def show_frame(self, idx):
        # 根据帧号找到对应的视频
        video_idx = 0
        frame_in_video = idx
        for i, start in enumerate(self.video_frame_starts[:-1]):
            if start <= idx < self.video_frame_starts[i + 1]:
                video_idx = i
                frame_in_video = idx - start
                break
        
        frame = None
        if video_idx < len(self.video_caps) and self.video_caps[video_idx]:
            cap = self.video_caps[video_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
            ret, frame = cap.read()
        
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            # 适应窗口大小并应用缩放
            label_w = self.label.width()
            label_h = self.label.height()
            if label_w > 0 and label_h > 0:
                base_scale = min(label_w / w, label_h / h)
                final_scale = base_scale * self.zoom_scale
                new_w, new_h = int(w * final_scale), int(h * final_scale)
            else:
                new_w, new_h = int(w * self.zoom_scale), int(h * self.zoom_scale)
            frame_small = cv2.resize(frame, (new_w, new_h))
            qimg = QImage(frame_small.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))
        self.frame_label.setText(f"{idx}/{self.total} (视频{video_idx + 1}/{len(self.video_paths)})")
        self.slider.setValue(idx)
        # 更新高亮
        self.slider.setDeleteRanges(self.ranges)
    
    def seek(self, idx):
        self.show_frame(idx)
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.timer = QTimer()
            self.timer.timeout.connect(self.next_frame)
            interval = int(1000 / max(self.fps, 1))
            self.timer.start(interval)
        elif self.timer:
            self.timer.stop()
    
    def next_frame(self):
        idx = self.slider.value()
        if idx < self.total - 1:
            self.show_frame(idx + 1)
        else:
            self.toggle_play()
    
    def add_video_to_trim(self):
        """添加视频到当前视频后面（合并成临时视频用于预览）"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择视频(支持多选)", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not file_paths:
            return
        
        # 保存原始视频列表
        if not hasattr(self, 'original_video_paths'):
            self.original_video_paths = list(self.video_paths)
        
        # 先读取现有视频的所有帧
        print(f"[Trim] 正在读取现有视频...")
        all_frames = []
        all_fps = self.fps
        
        for i, cap in enumerate(self.video_caps):
            if cap is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame.copy())
        
        # 读取新视频的所有帧
        print(f"[Trim] 正在合并 {len(file_paths)} 个新视频...")
        for vp in file_paths:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                continue
            self.video_paths.append(vp)  # 追加到原始列表
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame.copy())
            cap.release()
        
        if not all_frames:
            print("[Trim] 无法读取视频帧")
            return
        
        # 保存临时视频用于预览
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = str(temp_dir / "temp.mp4")
        
        height, width = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, all_fps, (width, height))
        for frame in all_frames:
            out.write(frame)
        out.release()
        
        # 更新预览用变量
        for cap in self.video_caps:
            if cap:
                cap.release()
        
        self.temp_video_path = temp_path
        self.video_caps = []
        self.video_frame_counts = []
        self.video_frame_starts = [0]
        
        cap = cv2.VideoCapture(temp_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_caps.append(cap)
            self.video_frame_counts.append(frame_count)
            self.video_frame_starts.append(frame_count)
        
        self.total = self.video_frame_starts[-1]
        self.slider.setMaximum(self.total - 1)
        self.show_frame(0)
        print(f"[Trim] 视频合并完成: {temp_path}, 总帧数: {self.total}, 原始视频数: {len(self.video_paths)}")
    
    def backward(self):
        idx = self.slider.value()
        if idx > 0:
            self.show_frame(idx - 1)
    
    def forward(self):
        idx = self.slider.value()
        if idx < self.total - 1:
            self.show_frame(idx + 1)
    
    def on_label_click(self, event):
        idx = self.slider.value()  # 0-indexed
        display_idx = idx + 1  # 转为 1-indexed 显示
        if self.select_start is None:
            self.select_start = idx
            self.del_btn.setText(f"选择\n帧{display_idx}")
        else:
            start, end = min(self.select_start, idx), max(self.select_start, idx)
            self.ranges.append((start, end))
            self.delete_list.addItem(f"帧 {start + 1} → {end + 1} ({end - start + 1}帧)")
            self.select_start = None
            self.del_btn.setText("删除\n选中")
    
    def clear(self):
        self.ranges = []
        self.del_frames = set()
        self.delete_list.clear()
        self.select_start = None
        self.del_btn.setText("删除\n选中")
        self.slider.setDeleteRanges([])
    
    def delete_selected(self):
        row = self.delete_list.currentRow()
        if row >= 0:
            self.delete_list.takeItem(row)
            del self.ranges[row]
            self.slider.setDeleteRanges(self.ranges)
    
    def delete_item(self, item):
        row = self.delete_list.row(item)
        self.delete_list.takeItem(row)
        del self.ranges[row]
        self.slider.setDeleteRanges(self.ranges)
    
    def generate(self):
        import time
        delete_set = set(self.del_frames)
        for s, e in self.ranges:
            delete_set.update(range(s, e + 1))
        
        keep_ranges = []
        last = 0
        for i in range(self.total):
            if i in delete_set:
                if last < i:
                    keep_ranges.append((last, i - 1))
                last = i + 1
        if last < self.total:
            keep_ranges.append((last, self.total - 1))
        
        if not keep_ranges:
            QMessageBox.warning(self, "提示", "没有保留的片段")
            return
        
        output_files = []
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        # 使用临时视频
        temp_video_path = str(Path("temp") / "temp.mp4")
        if not Path(temp_video_path).exists():
            QMessageBox.warning(self, "错误", "临时视频不存在，请重新加载视频")
            return
        
        base_name = ""
        if self.parent_panel and hasattr(self.parent_panel, 'last_video_name'):
            base_name = self.parent_panel.last_video_name
        if not base_name:
            base_name = Path(self.video_path).stem
        
        for idx, (s, e) in enumerate(keep_ranges):
            clip_name = f"{base_name}_clip{idx + 1}_{timestamp}.mp4"
            # 保存到原视频同一个文件夹
            out_path = Path(self.video_path).parent / clip_name
            
            cap_in = cv2.VideoCapture(temp_video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_out = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
            
            for f in range(s, e + 1):
                cap_in.set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = cap_in.read()
                if ret:
                    cap_out.write(frame)
            
            cap_in.release()
            cap_out.release()
            output_files.append(str(out_path))
        
        msg = f"原视频: {self.total}帧\n删除: {len(delete_set)}帧\n保留: {len(keep_ranges)}个片段\n\n保存文件:\n"
        for f in output_files:
            msg += f"  {f}\n"
        QMessageBox.information(self, "完成", msg)
    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        super().closeEvent(event)


class UnifiedPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1.7倍缩放
        self.setStyleSheet("""
            QWidget { font-size: 14px; }
            QGroupBox { font-size: 14px; font-weight: bold; }
            QGroupBox::title { font-size: 14px; }
            QLabel { font-size: 14px; }
            QPushButton { font-size: 14px; min-height: 28px; }
            QLineEdit { font-size: 14px; min-height: 28px; }
            QSlider::groove { height: 20px; }
            QSlider::handle { width: 20px; height: 20px; margin: -10px 0; }
            QListWidget { font-size: 14px; }
            QCheckBox { font-size: 14px; }
            QSpinBox { font-size: 14px; }
            QTextEdit { font-size: 14px; }
            QComboBox { font-size: 14px; }
            QToolButton { font-size: 14px; }
        """)
        self.setGeometry(100, 100, int(500*1.7), int(650*1.7))
        self.ctrl = VideoController()
        self.temp_data_path = Path("temp_data")
        self.viewer = None
        self.video_process = None
        self.total_frames = 1
        self.current_frame_idx = 0
        self.is_playing = False
        self.is_backward = False
        self.prompt_drawing_mode = False
        self.prompt_frame_idx = -1
        self.last_video_name = ""  # 上次处理的视频名称
        self.category_layout = None  # 将在create_viewer_section中初始化
        self.category_inputs = []
        self.category_labels = []
        
        # model.json默认值
        self.default_model_id = "model_001"
        self.default_model_name = "物体检测"
        self.default_model_desc = "物体检测模型"
        
        # 回退栈：记录每次操作
        # [{'type': 'single', 'frame_idx': 10, 'bbox_key': 'x,y,w,h', 'old_trace_id': 1000, 'new_trace_id': 2000}, ...]
        # [{'type': 'multi', 'changes': [{'frame_idx': 5, 'bbox_key': '...', 'old_trace_id': 1000, 'new_trace_id': 2000}, ...]}, ...]
        self.undo_stack = []

        self.palette_colors = [
            (0, 0, 255),     # 红 (BGR)
            (0, 165, 255),   # 橙 (BGR)
            (0, 255, 255),   # 黄 (BGR)
            (0, 255, 0),     # 绿 (BGR)
            (255, 255, 0),   # 青 (BGR)
            (255, 0, 0),     # 蓝 (BGR)
            (128, 0, 128),   # 紫 (BGR)
        ]
        self.selected_color_index = random.randint(0, 6)
        # 生成1000档的track_id选项 (1000, 2000, 3000, ...)
        self.prompt_trace_id_options = list(range(1000, 1000000, 1000))

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('视频标注工具')
        self.setGeometry(100, 100, 500, 650)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        central.setLayout(main_layout)

        main_layout.addWidget(self.create_video_trim_section())
        main_layout.addWidget(self.create_annotate_section())
        main_layout.addWidget(self.create_viewer_section())
        main_layout.addWidget(self.create_save_section())

    def create_video_trim_section(self):
        """视频帧剔除模块"""
        group = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        group.setLayout(layout)
        
        # 可折叠标题栏
        header = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        self.trim_toggle_btn = QToolButton()
        self.trim_toggle_btn.setText("0. 视频帧删除 ▼")
        self.trim_toggle_btn.setStyleSheet("QToolButton { font-weight: bold; background: #333; color: white; border: none; padding: 4px; }")
        self.trim_toggle_btn.setCheckable(True)
        self.trim_toggle_btn.setChecked(True)
        self.trim_toggle_btn.toggled.connect(lambda checked: self.trim_content.setVisible(checked))
        header_layout.addWidget(self.trim_toggle_btn)
        header.setLayout(header_layout)
        layout.addWidget(header)
        
        # 可折叠内容
        self.trim_content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)
        self.trim_content.setLayout(content_layout)
        layout.addWidget(self.trim_content)
        
        # 选择视频
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("视频:"))
        self.trim_video_input = QLineEdit()
        self.trim_video_input.setFixedHeight(22)
        self.trim_video_input.setPlaceholderText("选择视频文件")
        select_layout.addWidget(self.trim_video_input)
        select_btn = QPushButton("选择")
        select_btn.setFixedWidth(50)
        select_btn.clicked.connect(self.open_trim_dialog)
        select_layout.addWidget(select_btn)
        content_layout.addLayout(select_layout)
        
        return group
    
    def open_trim_dialog(self):
        """打开视频裁剪对话框"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not file_path:
            return
        
        self.trim_video_input.setText(file_path)
        
        # 创建并显示对话框
        dialog = TrimDialog(file_path, self)
        dialog.exec_()
    
    def add_trim_video(self):
        """添加视频到裁剪对话框（支持多选）"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择视频(支持多选)", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not file_paths:
            return
        
        current = self.trim_video_input.text().strip()
        if current:
            # 追加到现有路径后面
            new_paths = current + "|" + "|".join(file_paths)
        else:
            new_paths = "|".join(file_paths)
        self.trim_video_input.setText(new_paths)
    
    def open_trim_mid_dialog(self):
        """打开temp_data_mid帧删除对话框"""
        dialog = TrimMidDialog(self)
        dialog.exec_()
    
    def create_annotate_section(self):
        group = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        group.setLayout(layout)
        
        # 可折叠标题栏
        header = QWidget()
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        self.annot_toggle_btn = QToolButton()
        self.annot_toggle_btn.setText("1. 视频标注 (annotate_video) ▼")
        self.annot_toggle_btn.setStyleSheet("QToolButton { font-weight: bold; background: #333; color: white; border: none; padding: 4px; }")
        self.annot_toggle_btn.setCheckable(True)
        self.annot_toggle_btn.setChecked(True)
        self.annot_toggle_btn.toggled.connect(lambda checked: self.annot_content.setVisible(checked))
        header_layout.addWidget(self.annot_toggle_btn)
        header.setLayout(header_layout)
        layout.addWidget(header)
        
        # 可折叠内容
        self.annot_content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)
        self.annot_content.setLayout(content_layout)
        layout.addWidget(self.annot_content)
        
        video_layout = QHBoxLayout()
        video_layout.setSpacing(4)
        video_layout.addWidget(QLabel("视频:"))
        self.video_input = DragLineEdit()
        self.video_input.setFixedHeight(22)
        video_layout.addWidget(self.video_input)
        select_btn = QPushButton("选择")
        select_btn.setFixedWidth(50)
        select_btn.clicked.connect(self.select_video)
        video_layout.addWidget(select_btn)
        content_layout.addLayout(video_layout)

        # 前处理参数行
        preprocess_layout = QHBoxLayout()
        preprocess_layout.setSpacing(4)
        preprocess_layout.addWidget(QLabel("起始"))
        self.start_time_input = QLineEdit("0")
        self.start_time_input.setFixedWidth(50)
        self.start_time_input.setFixedHeight(22)
        preprocess_layout.addWidget(self.start_time_input)
        preprocess_layout.addWidget(QLabel("秒"))
        preprocess_layout.addWidget(QLabel("取前"))
        self.max_frames_input = QLineEdit("1000")
        self.max_frames_input.setFixedWidth(60)
        self.max_frames_input.setFixedHeight(22)
        preprocess_layout.addWidget(self.max_frames_input)
        preprocess_layout.addWidget(QLabel("帧"))
        preprocess_layout.addWidget(QLabel("每隔"))
        self.skip_frames_input = QLineEdit("1")
        self.skip_frames_input.setFixedWidth(40)
        self.skip_frames_input.setFixedHeight(22)
        preprocess_layout.addWidget(self.skip_frames_input)
        preprocess_layout.addWidget(QLabel("帧取1"))
        preprocess_layout.addWidget(QLabel("缩放"))
        self.resize_ratio_input = QLineEdit("1.0")
        self.resize_ratio_input.setFixedWidth(50)
        self.resize_ratio_input.setFixedHeight(22)
        preprocess_layout.addWidget(self.resize_ratio_input)
        layout.addLayout(preprocess_layout)

        iou_layout = QHBoxLayout()
        iou_layout.setSpacing(4)
        iou_layout.addWidget(QLabel("帧IoU:"))
        self.merge_iou_input = QLineEdit("0.5")
        self.merge_iou_input.setFixedWidth(40)
        self.merge_iou_input.setFixedHeight(22)
        iou_layout.addWidget(self.merge_iou_input)
        iou_layout.addWidget(QLabel("前后IoU:"))
        self.iou_input = QLineEdit("0.02")
        self.iou_input.setFixedWidth(40)
        self.iou_input.setFixedHeight(22)
        iou_layout.addWidget(self.iou_input)
        iou_layout.addWidget(QLabel("物品:"))
        self.items_input = QLineEdit()
        self.items_input.setMinimumWidth(100)
        self.items_input.setFixedHeight(22)
        iou_layout.addWidget(self.items_input)
        layout.addLayout(iou_layout)

        # 形态学操作单独一行
        morph_layout = QHBoxLayout()
        morph_layout.setSpacing(4)
        morph_layout.addWidget(QLabel("形态:"))
        self.morph_kernel = QSlider(Qt.Horizontal)
        self.morph_kernel.setMinimum(0)
        self.morph_kernel.setMaximum(20)
        self.morph_kernel.setValue(0)
        self.morph_kernel.setFixedHeight(16)
        self.morph_kernel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.morph_kernel.valueChanged.connect(self.on_morph_kernel_changed)
        morph_layout.addWidget(self.morph_kernel)
        self.morph_label = QLabel("0")
        self.morph_label.setFixedWidth(30)
        morph_layout.addWidget(self.morph_label)
        morph_layout.addWidget(QLabel("(分离同色黏连)"))
        layout.addLayout(morph_layout)

        scale_layout = QHBoxLayout()
        scale_layout.setSpacing(4)
        scale_layout.addWidget(QLabel("缩放"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(50)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setValue(100)
        self.scale_slider.setFixedHeight(16)
        self.scale_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.scale_slider.valueChanged.connect(self.on_scale_change)
        scale_layout.addWidget(self.scale_slider)
        self.scale_label = QLabel("100%")
        self.scale_label.setFixedWidth(30)
        scale_layout.addWidget(self.scale_label)
        layout.addLayout(scale_layout)

        self.annotate_btn = QPushButton("▶ 执行标注")
        self.annotate_btn.setFixedHeight(28)
        self.annotate_btn.clicked.connect(self.run_annotate)
        layout.addWidget(self.annotate_btn)

        return group

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.video_input.setText(file_path)

    def create_temp_video(self, video_path, start_time=0, max_frames=1000, skip_frames=1, resize_ratio=1.0):
        """根据起始时间、帧数和抽帧间隔生成临时视频
        
        Args:
            resize_ratio: 缩放比例，如0.5表示缩小到一半
        
        Returns:
            tuple: (临时视频路径, fps, 总帧数) 或 None如果失败
        """
        import tempfile
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算缩放后的尺寸
        width = int(orig_width * resize_ratio)
        height = int(orig_height * resize_ratio)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 创建临时视频文件
        temp_video_dir = Path("1src")
        temp_video_dir.mkdir(exist_ok=True)
        temp_video_path = temp_video_dir / "temp_input.mp4"
        if temp_video_path.exists():
            temp_video_path.unlink()
        
        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
        
        start_frame = int(start_time * fps)
        frame_idx = 0
        output_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳过起始帧
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            
            # 检查是否在跳帧范围内
            if (frame_idx - start_frame) % skip_frames != 0:
                frame_idx += 1
                continue
            
            # 检查是否达到最大帧数
            if output_count >= max_frames:
                break
            
            # 缩放帧
            if resize_ratio != 1.0:
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
            output_count += 1
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return (str(temp_video_path), fps, output_count)

    def run_annotate(self):
        video_path = self.video_input.text()
        if not video_path:
            QMessageBox.warning(self, "错误", "请先选择视频文件")
            return

        # 获取参数
        start_time = float(self.start_time_input.text()) if self.start_time_input.text() else 0
        max_frames = int(self.max_frames_input.text()) if self.max_frames_input.text() else 1000
        skip_frames = int(self.skip_frames_input.text()) if self.skip_frames_input.text() else 1
        resize_ratio = float(self.resize_ratio_input.text()) if self.resize_ratio_input.text() else 1.0

        # 生成临时视频（包含起始时间、抽帧等处理）
        print(f"[run_annotate] 生成临时视频: 起始={start_time}秒, 取{max_frames}帧, 每隔{skip_frames}帧取1, 缩放={resize_ratio}")
        temp_result = self.create_temp_video(video_path, start_time, max_frames, skip_frames, resize_ratio)
        if temp_result is None:
            QMessageBox.warning(self, "错误", "无法打开视频文件")
            return
        
        temp_video_path, fps, temp_frame_count = temp_result
        print(f"[run_annotate] 临时视频生成完成: {temp_video_path}, {fps}fps, {temp_frame_count}帧")
        
        # 保存原视频名称用于OBS命名
        self.last_video_name = Path(video_path).stem
        
        src_video = temp_video_path
        scale = self.scale_slider.value() / 100.0
        dialog = AnnotationDialog(src_video, self, scale=scale)
        if dialog.exec_() != QDialog.Accepted:
            return
        boxes = dialog.get_boxes()
        print(f"已标注 {len(boxes)} 个目标")

        iou_val = float(self.iou_input.text() or "0.02")
        merge_iou_val = float(self.merge_iou_input.text() or "0.5")
        items_text = self.items_input.text()
        find_list = [s.strip() for s in items_text.split(',') if s.strip()]

        self.statusBar().showMessage("正在处理视频，请稍候...")
        QApplication.processEvents()

        try:
            from annotate_video import SAM_MODEL_PATH, DST_DIR, TEMP_DATA_DIR, TEMP_DATA_MID_DIR
            from annotate_video import merge_masks_in_frame, TrackManager, get_device, get_output_filename
            from annotate_video import put_chinese_text, IOU_THRESHOLD as OrigIOU, MERGE_IOU_THRESHOLD as OrigMergeIOU, FIND as OrigFIND
            import annotate_video as av_module

            av_module.IOU_THRESHOLD = iou_val
            av_module.MERGE_IOU_THRESHOLD = merge_iou_val
            av_module.FIND = find_list

            has_text = bool(find_list)
            has_bbox = bool(boxes)

            if not has_text and not has_bbox:
                QMessageBox.warning(self, "提示", "请至少框选目标或填写物品名称")
                return

            predictor_name = "SAM3VideoSemanticPredictor" if has_text else "SAM3VideoPredictor"
            print(f"正在使用 {predictor_name} 进行视频分割跟踪...")
            if has_text and has_bbox:
                for i, (t, b) in enumerate(zip(find_list, boxes)):
                    bbox_str = f"({int(b['x1'])},{int(b['y1'])},{int(b['x2'])},{int(b['y2'])})"
                    print(f"  [{i}] 文本: '{t}' | bbox: {bbox_str}")
            elif has_text:
                print(f"  文本提示词: {find_list}")
            elif has_bbox:
                bbox_str_list = [f"({int(b['x1'])},{int(b['y1'])},{int(b['x2'])},{int(b['y2'])})" for b in boxes]
                print(f"  bbox提示框: {bbox_str_list}")

            device, device_type = get_device()
            print(f"[DEBUG run_annotate] get_device() 返回: device={device}, device_type={device_type}")
            half = device_type == 'cuda'
            print(f"[DEBUG run_annotate] half={half}")
            overrides = dict(
                conf=0.25, task="segment", mode="predict",
                model=SAM_MODEL_PATH, device=device,
                half=half, save=False, verbose=False
            )
            print(f"[DEBUG run_annotate] overrides: {overrides}")
            if device_type == 'cuda':
                overrides['batch'] = 1
                overrides['stream_buffer'] = False
                print(f"[DEBUG run_annotate] CUDA优化: batch=1, stream_buffer=False")
            elif device_type == 'mps':
                overrides['half'] = True
                overrides['amp'] = True
                overrides['stream_buffer'] = True
                print(f"[DEBUG run_annotate] MPS优化: half=True, amp=True, stream_buffer=True")

            print(f"[DEBUG run_annotate] 正在初始化 {predictor_name}...")
            if predictor_name == "SAM3VideoSemanticPredictor":
                _patch_sam3_video_semantic()
                from ultralytics.models.sam import SAM3VideoSemanticPredictor
                predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            else:
                from ultralytics.models.sam import SAM3VideoPredictor
                predictor = SAM3VideoPredictor(overrides=overrides)
            print(f"[DEBUG run_annotate] {predictor_name} 初始化完成")
            print(f"[DEBUG run_annotate] predictor.device: {predictor.device}")
            print(f"[DEBUG run_annotate] predictor.model.device: {predictor.model.device if hasattr(predictor.model, 'device') else 'N/A'}")

            cap = cv2.VideoCapture(src_video)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([chr(fourcc_int & 0xFF), chr((fourcc_int >> 8) & 0xFF), chr((fourcc_int >> 16) & 0xFF), chr((fourcc_int >> 24) & 0xFF)])
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            output_filename = get_output_filename(src_video)
            output_path = Path(DST_DIR) / output_filename
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            temp_data_path = Path(TEMP_DATA_DIR)
            if temp_data_path.exists():
                shutil.rmtree(temp_data_path)
            temp_data_path.mkdir(parents=True, exist_ok=True)
            frames_dir = temp_data_path / "frames"
            labels_dir = temp_data_path / "labels"
            frames_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)

            coco_data = {
                'info': {'description': 'Video Annotation Dataset', 'video_path': src_video,
                         'fps': fps, 'width': width, 'height': height, 'fourcc': fourcc_str,
                         'FIND': find_list},
                'images': [], 'annotations': [],
                'categories': [{'id': i, 'name': f'object_{i}'} for i in range(len(boxes) if boxes else 8)]
            }

            annotation_id = [0]
            track_manager = TrackManager(iou_threshold=iou_val)

            predictor_args = {'source': src_video, 'stream': True}
            if has_bbox:
                bbox_list = [(b['x1'], b['y1'], b['x2'], b['y2']) for b in boxes]
                predictor_args['bboxes'] = bbox_list
                predictor_args['labels'] = [1] * len(boxes)
            if has_text:
                predictor_args['text'] = find_list

            print(f"[DEBUG run_annotate] predictor_args: {predictor_args}")

            import torch
            if hasattr(predictor, 'model') and hasattr(predictor.model, 'device'):
                print(f"[DEBUG run_annotate] 模型当前设备(推理前): {predictor.model.device}")
            print(f"[DEBUG run_annotate] torch.cuda.current_device(): {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")
            print(f"[DEBUG run_annotate] torch.cuda.device_count(): {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

            results = predictor(**predictor_args)
            frame_count = 0
            max_frames = int(self.max_frames_input.text()) if self.max_frames_input.text() else 1000
            start_time = float(self.start_time_input.text()) if self.start_time_input.text() else 0
            start_frame = int(start_time * fps) if start_time > 0 else 0
            print(f"[DEBUG run_annotate] 开始遍历results, 预计总帧数: {total_frames}, 起始帧: {start_frame}, 最大处理帧数: {max_frames}")

            for r in results:
                # 跳过起始帧
                if frame_count < start_frame:
                    frame_count += 1
                    continue
                if frame_count >= start_frame + max_frames:
                    print(f"[DEBUG run_annotate] 已达到最大帧数 {max_frames}，停止处理")
                    break
                if frame_count == 0:
                    if hasattr(predictor, 'model') and hasattr(predictor.model, 'device'):
                        print(f"[DEBUG run_annotate] 模型推理后设备: {predictor.model.device}")
                    print(f"[DEBUG run_annotate] 第一帧推理结果 device: {r.device if hasattr(r, 'device') else 'N/A'}")
                    if r.masks is not None and hasattr(r.masks.data, 'device'):
                        print(f"[DEBUG run_annotate] masks.device: {r.masks.data.device}")

                if frame_count % 10 == 0:
                    print(f"[DEBUG run_annotate] 已处理 {frame_count}/{total_frames} 帧, GPU内存: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB")

                orig_img = r.orig_img if hasattr(r, 'orig_img') and r.orig_img is not None else None
                if orig_img is None:
                    cap_temp = cv2.VideoCapture(src_video)
                    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret_temp, orig_img = cap_temp.read()
                    cap_temp.release()
                    if not ret_temp:
                        orig_img = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    if len(orig_img.shape) == 2:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                    elif orig_img.shape[2] == 4:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)

                cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), orig_img)

                coco_data['images'].append({
                    'id': frame_count, 'file_name': f"frame_{frame_count:06d}.jpg",
                    'width': width, 'height': height, 'frame_count': frame_count
                })

                frame_annotations = []
                debug_masks_count = 0
                debug_contours_count = 0
                debug_merged_count = 0
                debug_track_ids = []
                if hasattr(r, 'masks') and r.masks is not None:
                    masks_tensor = r.masks.data
                    if masks_tensor is not None and len(masks_tensor) > 0:
                        debug_masks_count = len(masks_tensor)
                        confs = None
                        if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'conf'):
                            confs = r.boxes.conf.cpu().numpy()
                            print(f"[DEBUG {frame_count}/{total_frames}] boxes.conf={confs.tolist()}")

                        current_masks = []
                        current_bboxes = []
                        for mask in masks_tensor:
                            mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
                            mask_binary = (mask_np > 0.5).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            debug_contours_count += len(contours)
                            for contour in contours:
                                if len(contour) >= 3:
                                    polygon = contour.squeeze().flatten().tolist()
                                    x_coords = polygon[0::2]
                                    y_coords = polygon[1::2]
                                    x_min, x_max = min(x_coords), max(x_coords)
                                    y_min, y_max = min(y_coords), max(y_coords)
                                    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                                    area = cv2.contourArea(contour)
                                    if area > 0:
                                        current_masks.append(mask_binary)
                                        current_bboxes.append(bbox)

                        if current_masks:
                            debug_merged_count = len(current_masks)
                            current_masks, current_bboxes = merge_masks_in_frame(current_masks, current_bboxes, merge_iou_val)
                            track_ids = track_manager.update(current_masks, current_bboxes, frame_count)
                            debug_track_ids = track_ids
                            after_merge_contours = 0
                            for mask in current_masks:
                                mask_binary = (mask > 0.5).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    if len(contour) >= 3:
                                        after_merge_contours += 1
                            print(f"[DEBUG {frame_count}/{total_frames}] 原始contours={debug_contours_count}, 有效polygon={debug_merged_count}, merge后={len(current_masks)}, merge后contours={after_merge_contours}, track_ids={track_ids}")
                            for idx, (mask, bbox) in enumerate(zip(current_masks, current_bboxes)):
                                mask_binary = (mask > 0.5).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    if len(contour) >= 3:
                                        polygon = contour.squeeze().flatten().tolist()
                                        area = cv2.contourArea(contour)
                                        track_id = track_ids[idx] if idx < len(track_ids) else annotation_id[0]
                                        confidence = float(confs[idx]) if confs is not None and idx < len(confs) else float(mask.max())
                                        ann = {
                                            'id': annotation_id[0], 'track_id': track_id, 'image_id': frame_count,
                                            'category_id': track_id, 'bbox': bbox, 'area': float(area),
                                            'segmentation': [polygon], 'iscrowd': 0, 'confidence': confidence
                                        }
                                        coco_data['annotations'].append(ann)
                                        frame_annotations.append(ann)
                                        annotation_id[0] += 1
                    else:
                        print(f"[DEBUG {frame_count}/{total_frames}] masks_tensor长度=0")
                else:
                    print(f"[DEBUG {frame_count}/{total_frames}] 无masks属性或masks为None")

                print(f"[DEBUG {frame_count}/{total_frames}] 帧annotations数量={len(frame_annotations)}, track_ids={debug_track_ids}")
                with open(labels_dir / f"frame_{frame_count:06d}.json", 'w', encoding='utf-8') as f:
                    json.dump(frame_annotations, f, ensure_ascii=False)

                annotated_frame = r.plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                if boxes:
                    from annotate_video import BOX_COLORS as AV_BOX_COLORS
                    for i, bbox in enumerate(boxes):
                        label = f"目标 {i + 1}"
                        x1 = int(bbox.get('x1', 0))
                        y1 = int(bbox.get('y1', 0))
                        annotated_frame_rgb = put_chinese_text(annotated_frame_rgb, label, (x1, max(10, y1 - 10)), font_size=15, color=AV_BOX_COLORS[i % len(AV_BOX_COLORS)])
                out.write(annotated_frame_rgb)
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count} 帧")
                    try:
                        import torch
                        if torch.cuda.is_available() and device_type == 'cuda':
                            torch.cuda.empty_cache()
                    except:
                        pass

            with open(temp_data_path / 'annotations.json', 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False)

            out.release()
            print(f"✓ 标注视频已保存到: {output_path}")
            print(f"✓ 共处理 {frame_count} 帧")
            print(f"✓ 临时数据已保存到: {temp_data_path}")

            self.statusBar().showMessage(f"标注完成: {DST_DIR}")
            QMessageBox.information(self, "完成", f"标注完成！\n输出目录: {DST_DIR}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("处理失败")
            QMessageBox.critical(self, "错误", f"处理失败:\n{e}")
        finally:
            av_module.IOU_THRESHOLD = OrigIOU
            av_module.MERGE_IOU_THRESHOLD = OrigMergeIOU
            av_module.FIND = OrigFIND

    def create_viewer_section(self):
        group = QGroupBox("2. 预览")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        group.setLayout(layout)

        path_layout = QHBoxLayout()
        path_layout.setSpacing(4)
        path_layout.addWidget(QLabel("数据"))
        self.path_input = QLineEdit("temp_data")
        self.path_input.setFixedHeight(22)
        path_layout.addWidget(self.path_input)
        show_btn = QPushButton("显示")
        show_btn.setFixedSize(44, 22)
        show_btn.clicked.connect(self.show_viewer)
        path_layout.addWidget(show_btn)
        redo_btn = QPushButton("选择视频")
        redo_btn.setFixedSize(60, 22)
        redo_btn.clicked.connect(self.select_data_dir)
        path_layout.addWidget(redo_btn)
        redo_copy_btn = QPushButton("重做")
        redo_copy_btn.setFixedSize(44, 22)
        redo_copy_btn.clicked.connect(self.redo_copy)
        path_layout.addWidget(redo_copy_btn)
        trim_mid_btn = QPushButton("帧删除")
        trim_mid_btn.setFixedSize(60, 22)
        trim_mid_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; border: none; border-radius: 3px; }")
        trim_mid_btn.clicked.connect(self.open_trim_mid_dialog)
        path_layout.addWidget(trim_mid_btn)
        import_labelme_btn = QPushButton("导入labelme")
        import_labelme_btn.setFixedSize(80, 22)
        import_labelme_btn.setStyleSheet("QPushButton { background-color: #17a2b8; color: white; border: none; border-radius: 3px; }")
        import_labelme_btn.clicked.connect(self.import_labelme_to_temp_data)
        path_layout.addWidget(import_labelme_btn)
        layout.addLayout(path_layout)

        self.category_layout = QVBoxLayout()
        self.category_layout.setSpacing(2)
        # 动态类别列表，根据ID映射终点生成
        self._update_category_list()
        layout.addLayout(self.category_layout)

        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(4)
        zoom_layout.addWidget(QLabel("缩放"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedHeight(16)
        self.zoom_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.zoom_slider.valueChanged.connect(self.on_zoom_change)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(30)
        zoom_layout.addWidget(self.zoom_label)
        layout.addLayout(zoom_layout)

        conf_layout = QHBoxLayout()
        conf_layout.setSpacing(4)
        conf_layout.addWidget(QLabel("置信度"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.ctrl.conf_threshold * 100))
        self.conf_slider.setFixedHeight(16)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        conf_layout.addWidget(self.conf_slider)
        layout.addLayout(conf_layout)

        # 第一行：倒播、倒帧、正帧、正播
        frame_nav_layout = QHBoxLayout()
        frame_nav_layout.setSpacing(2)
        self.backward_fast_btn = QPushButton("倒播")
        self.backward_fast_btn.setFixedHeight(24)
        self.backward_fast_btn.clicked.connect(self.toggle_backward_fast)
        frame_nav_layout.addWidget(self.backward_fast_btn)

        self.backward_btn = QPushButton("倒帧")
        self.backward_btn.setFixedHeight(24)
        self.backward_btn.clicked.connect(self.toggle_backward)
        frame_nav_layout.addWidget(self.backward_btn)

        self.next_btn = QPushButton("正帧")
        self.next_btn.setFixedHeight(24)
        self.next_btn.clicked.connect(self.toggle_play)
        frame_nav_layout.addWidget(self.next_btn)

        self.forward_fast_btn = QPushButton("正播")
        self.forward_fast_btn.setFixedHeight(24)
        self.forward_fast_btn.clicked.connect(self.toggle_play_fast)
        frame_nav_layout.addWidget(self.forward_fast_btn)
        layout.addLayout(frame_nav_layout)

        # 第二行：后向、提示帧、帧数、前向
        frame_play_layout = QHBoxLayout()
        frame_play_layout.setSpacing(4)
        self.backward_cb = QCheckBox("后向")
        self.backward_cb.setFixedHeight(24)
        self.backward_cb.setChecked(True)
        self.backward_cb.setStyleSheet("QCheckBox { font-size: 11px; }")
        frame_play_layout.addWidget(self.backward_cb)

        self.frame_label = QLabel("1/1")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setFixedHeight(24)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.frame_label.setStyleSheet("QLabel { background-color: #333; color: #fff; border-radius: 3px; font-weight: bold; font-size: 14px; padding: 0 8px; }")
        frame_play_layout.addWidget(self.frame_label)

        self.forward_cb = QCheckBox("前向")
        self.forward_cb.setFixedHeight(24)
        self.forward_cb.setChecked(True)
        self.forward_cb.setStyleSheet("QCheckBox { font-size: 11px; }")
        frame_play_layout.addWidget(self.forward_cb)
        layout.addLayout(frame_play_layout)

        # 第三行：提示帧按钮单独一行（撑满）
        prompt_layout = QHBoxLayout()
        self.prompt_btn = QPushButton("提示帧")
        self.prompt_btn.setFixedHeight(24)
        self.prompt_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.prompt_btn.setStyleSheet("QPushButton { background-color: #FFA500; color: white; border: none; border-radius: 3px; font-size: 11px; } QPushButton:hover { background-color: #FF8C00; }")
        self.prompt_btn.clicked.connect(self.toggle_prompt_mode)
        prompt_layout.addWidget(self.prompt_btn)
        layout.addLayout(prompt_layout)

        # 回退按钮单独一行（撑满）
        undo_layout = QHBoxLayout()
        undo_layout.setSpacing(4)
        self.undo_btn = QPushButton("回退")
        self.undo_btn.setFixedHeight(24)
        self.undo_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.undo_btn.setStyleSheet("QPushButton { background-color: #CC0000; color: white; border: none; border-radius: 3px; font-size: 11px; } QPushButton:hover { background-color: #990000; }")
        self.undo_btn.clicked.connect(self.show_undo_menu)
        undo_layout.addWidget(self.undo_btn)
        layout.addLayout(undo_layout)
        
        # 记录上一次执行的FIRST_ID和固定框trace_id
        self.last_prompt_first_id = None
        self.last_fixed_trace_id = None

        # Trace ID列表
        trace_list_layout = QHBoxLayout()
        trace_list_layout.setSpacing(4)

        list_area = QVBoxLayout()
        list_area.setSpacing(2)
        list_area.addWidget(QLabel("Trace ID列表"))
        self.trace_id_list = QListWidget()
        self.trace_id_list.setAlternatingRowColors(True)
        self.trace_id_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        list_area.addWidget(self.trace_id_list)
        trace_list_layout.addLayout(list_area)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        btn_red = "QPushButton { background-color: #dc3545; color: white; border: none; border-radius: 3px; font-size: 11px; } QPushButton:hover { background-color: #c82333; }"

        refresh_btn = QPushButton("刷新")
        refresh_btn.setFixedSize(44, 22)
        refresh_btn.setStyleSheet("QPushButton { background-color: #6c757d; color: white; border: none; border-radius: 3px; font-size: 11px; }")
        refresh_btn.clicked.connect(self.refresh_trace_id_list)
        btn_col.addWidget(refresh_btn)

        delete_btn = QPushButton("删除")
        delete_btn.setFixedSize(44, 22)
        delete_btn.setStyleSheet(btn_red)
        delete_btn.clicked.connect(self.remove_selected_trace_id)
        btn_col.addWidget(delete_btn)

        btn_col.addStretch()
        trace_list_layout.addLayout(btn_col)
        layout.addLayout(trace_list_layout)

        # 固定框功能
        fixed_bbox_layout = QVBoxLayout()
        fixed_bbox_layout.setSpacing(2)
        fixed_bbox_layout.addWidget(QLabel("固定框 (起始帧-终止帧):"))
        
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(QLabel("起始:"))
        self.fixed_start_input = QLineEdit("0")
        self.fixed_start_input.setFixedWidth(50)
        self.fixed_start_input.setFixedHeight(22)
        row1.addWidget(self.fixed_start_input)
        row1.addWidget(QLabel("终止:"))
        self.fixed_end_input = QLineEdit("-1")
        self.fixed_end_input.setFixedWidth(50)
        self.fixed_end_input.setFixedHeight(22)
        row1.addWidget(self.fixed_end_input)
        self.fixed_edit_btn = QPushButton("编辑固定框")
        self.fixed_edit_btn.setFixedHeight(24)
        self.fixed_edit_btn.setStyleSheet("QPushButton { background-color: #ffc107; color: black; border: none; border-radius: 3px; }")
        self.fixed_edit_btn.clicked.connect(self.toggle_fixed_bbox_mode)
        row1.addWidget(self.fixed_edit_btn)
        fixed_bbox_layout.addLayout(row1)
        
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self.fixed_bbox_btn = QPushButton("执行固定框")
        self.fixed_bbox_btn.setFixedHeight(24)
        self.fixed_bbox_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.fixed_bbox_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; border: none; border-radius: 3px; }")
        self.fixed_bbox_btn.clicked.connect(self.apply_fixed_bbox)
        row2.addWidget(self.fixed_bbox_btn)
        fixed_bbox_layout.addLayout(row2)
        
        layout.addLayout(fixed_bbox_layout)

        assign_layout = QHBoxLayout()
        assign_layout.setSpacing(4)
        assign_layout.addWidget(QLabel("当前ID:"))
        dec_btn = QPushButton("-")
        dec_btn.setFixedSize(28, 28)
        dec_btn.setStyleSheet("QPushButton { background-color: #dc3545; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; } QPushButton:hover { background-color: #c82333; }")
        dec_btn.clicked.connect(self.decrement_trace_id)
        assign_layout.addWidget(dec_btn)

        self.trace_id_input = QLineEdit(str(self.ctrl.next_track_id))
        self.trace_id_input.setFixedHeight(28)
        self.trace_id_input.setFixedWidth(80)
        self.trace_id_input.setAlignment(Qt.AlignCenter)
        self.trace_id_input.setStyleSheet("QLineEdit { background-color: #2c3e50; color: #ecf0f1; border: 1px solid #2c3e50; border-radius: 4px; padding: 0 8px; font-weight: bold; font-size: 13px; }")
        self.trace_id_input.textChanged.connect(self.on_trace_id_input_changed)
        assign_layout.addWidget(self.trace_id_input)

        inc_btn = QPushButton("+")
        inc_btn.setFixedSize(28, 28)
        inc_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; } QPushButton:hover { background-color: #218838; }")
        inc_btn.clicked.connect(self.increment_trace_id)
        assign_layout.addWidget(inc_btn)
        assign_layout.addStretch()
        layout.addLayout(assign_layout)

        self.export_btn = QPushButton("📦 导出到 temp_data_post")
        self.export_btn.setFixedHeight(26)
        self.export_btn.clicked.connect(self.export_to_temp_data_post)
        layout.addWidget(self.export_btn)

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.on_frame_play)

        return group

    def create_save_section(self):
        group = QGroupBox("3. 导出视频 (save)")
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        group.setLayout(layout)

        input_dir_name_layout = QHBoxLayout()
        input_dir_name_layout.setSpacing(4)
        input_dir_name_layout.addWidget(QLabel("输入:"))
        self.save_input_dir = QLineEdit("temp_data_post")
        self.save_input_dir.setFixedWidth(100)
        self.save_input_dir.setFixedHeight(22)
        input_dir_name_layout.addWidget(self.save_input_dir)
        browse_btn = QPushButton("选择")
        browse_btn.setFixedHeight(22)
        browse_btn.clicked.connect(self.select_save_input_dir)
        input_dir_name_layout.addWidget(browse_btn)
        input_dir_name_layout.addWidget(QLabel("前"))
        self.export_frame_limit = QLineEdit("-1")
        self.export_frame_limit.setFixedWidth(50)
        self.export_frame_limit.setFixedHeight(22)
        input_dir_name_layout.addWidget(self.export_frame_limit)
        input_dir_name_layout.addWidget(QLabel("帧"))
        input_dir_name_layout.addWidget(QLabel("名称:"))
        self.save_output_name = QLineEdit("1dst.mp4")
        self.save_output_name.setFixedWidth(80)
        self.save_output_name.setFixedHeight(22)
        input_dir_name_layout.addWidget(self.save_output_name)
        input_dir_name_layout.addWidget(QLabel("标题:"))
        self.labelme_title = QLineEdit("frame")
        self.labelme_title.setFixedWidth(60)
        self.labelme_title.setFixedHeight(22)
        input_dir_name_layout.addWidget(self.labelme_title)
        input_dir_name_layout.addWidget(QLabel("位数:"))
        self.labelme_digit = QLineEdit("6")
        self.labelme_digit.setFixedWidth(30)
        self.labelme_digit.setFixedHeight(22)
        input_dir_name_layout.addWidget(self.labelme_digit)
        layout.addLayout(input_dir_name_layout)

        save_alpha_layout = QHBoxLayout()
        save_alpha_layout.setSpacing(4)
        save_alpha_layout.addWidget(QLabel("透明度:"))
        self.save_alpha_slider = QSlider(Qt.Horizontal)
        self.save_alpha_slider.setMinimum(10)
        self.save_alpha_slider.setMaximum(100)
        self.save_alpha_slider.setValue(int(self.ctrl.alpha * 100))
        self.save_alpha_slider.setFixedHeight(16)
        self.save_alpha_slider.valueChanged.connect(self.on_save_alpha_change)
        save_alpha_layout.addWidget(self.save_alpha_slider)
        self.save_alpha_label = QLabel(f"{int(self.ctrl.alpha * 100)}%")
        save_alpha_layout.addWidget(self.save_alpha_label)
        layout.addLayout(save_alpha_layout)

        self.color_btn_layout = QHBoxLayout()
        self.color_btn_layout.setSpacing(2)
        self.color_btn_layout.addWidget(QLabel("颜色:"))
        self.color_btns = []
        self.color_styles = []
        for idx, (b_val, g_val, r_val) in enumerate(self.palette_colors):
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            color = f"rgb({r_val},{g_val},{b_val})"
            active_color = "border: 2px solid #FFD700;" if idx == self.selected_color_index else ""
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; border-radius: 3px; {active_color} }}"
                f"QPushButton:selected {{ border: 2px solid #FFD700; }}"
            )
            btn.clicked.connect(lambda _, i=idx: self.on_color_select(i))
            self.color_btns.append(btn)
            self.color_btn_layout.addWidget(btn)
        layout.addLayout(self.color_btn_layout)

        self.render_segment_check = QCheckBox("只展示bbox")
        self.render_segment_check.setChecked(False)
        self.render_segment_check.setStyleSheet("QCheckBox { font-size: 11px; }")
        layout.addWidget(self.render_segment_check)

        self.train_model_check = QCheckBox("训练YOLO模型")
        self.train_model_check.setChecked(True)
        self.train_model_check.setStyleSheet("QCheckBox { font-size: 11px; }")
        layout.addWidget(self.train_model_check)

        # 训练参数
        train_params_layout = QHBoxLayout()
        train_params_layout.setSpacing(4)
        train_params_layout.addWidget(QLabel("ID:"))
        self.train_id_input = QLineEdit(self.default_model_id)
        self.train_id_input.setFixedWidth(100)
        self.train_id_input.setFixedHeight(22)
        train_params_layout.addWidget(self.train_id_input)
        train_params_layout.addWidget(QLabel("描述:"))
        self.train_desc_input = QLineEdit(self.default_model_desc)
        self.train_desc_input.setFixedWidth(100)
        self.train_desc_input.setFixedHeight(22)
        train_params_layout.addWidget(self.train_desc_input)
        train_params_layout.addWidget(QLabel("Epoch:"))
        self.train_epochs_input = QLineEdit("30")
        self.train_epochs_input.setFixedWidth(40)
        self.train_epochs_input.setFixedHeight(22)
        train_params_layout.addWidget(self.train_epochs_input)
        self.train_resume_check = QCheckBox("继续训练")
        self.train_resume_check.setChecked(False)
        self.train_resume_check.setStyleSheet("QCheckBox { font-size: 11px; }")
        train_params_layout.addWidget(self.train_resume_check)
        train_params_layout.addStretch()
        layout.addLayout(train_params_layout)

        trail_layout = QHBoxLayout()
        trail_layout.setSpacing(4)
        self.trail_check = QCheckBox("粒子效果")
        self.trail_check.setChecked(False)
        self.trail_check.setStyleSheet("QCheckBox { font-size: 11px; }")
        trail_layout.addWidget(self.trail_check)
        self.latex_check = QCheckBox("白色乳胶漆")
        self.latex_check.setChecked(False)
        self.latex_check.setStyleSheet("QCheckBox { font-size: 11px; }")
        trail_layout.addWidget(self.latex_check)
        self.trail_line_check = QCheckBox("轨迹")
        self.trail_line_check.setChecked(False)
        self.trail_line_check.setStyleSheet("QCheckBox { font-size: 11px; }")
        trail_layout.addWidget(self.trail_line_check)
        trail_layout.addWidget(QLabel("时间:"))
        self.trail_duration = QLineEdit("500")
        self.trail_duration.setFixedWidth(40)
        self.trail_duration.setFixedHeight(22)
        trail_layout.addWidget(self.trail_duration)
        trail_layout.addWidget(QLabel("ms"))
        trail_layout.addStretch()
        layout.addLayout(trail_layout)

        self.save_btn = QPushButton("💾 保存视频并上传OBS")
        self.save_btn.setFixedHeight(28)
        self.save_btn.clicked.connect(self.run_save)
        layout.addWidget(self.save_btn)

        return group

    def on_zoom_change(self, value):
        self.zoom_label.setText(f"{value}%")

    def on_scale_change(self, value):
        self.scale_label.setText(f"{value}%")

    def on_conf_change(self, value):
        self.ctrl.conf_threshold = value / 100.0
        if self.viewer:
            self.viewer.update_display()
    
    def on_morph_kernel_changed(self, value):
        self.morph_label.setText(str(value))
        self.ctrl.morph_kernel = value
        if self.viewer:
            self.viewer.update_display()

    def on_alpha_change(self, value):
        self.ctrl.alpha = value / 100.0
        self.alpha_label.setText(f"{value}%")
        if self.viewer:
            self.viewer.update_display()

    def on_save_alpha_change(self, value):
        self.ctrl.alpha = value / 100.0
        self.save_alpha_label.setText(f"{value}%")

    def _update_color_btn_styles(self):
        for idx, btn in enumerate(self.color_btns):
            b_val, g_val, r_val = self.palette_colors[idx]
            color = f"rgb({r_val},{g_val},{b_val})"
            border = "border: 3px solid #FFD700;" if idx == self.selected_color_index else "border: 1px solid #555555;"
            btn.setStyleSheet(f"QPushButton {{ background-color: {color}; border-radius: 4px; {border} }}")

    def on_color_select(self, idx):
        self.selected_color_index = idx
        self._update_color_btn_styles()

    def prev_frame(self):
        if self.viewer:
            idx = (self.viewer.get_current_frame() - 1) % self.total_frames
            self.viewer.go_to_frame(idx)
            self.frame_label.setText(f"{idx+1}/{self.total_frames}")

    def next_frame(self):
        if self.viewer:
            idx = (self.viewer.get_current_frame() + 1) % self.total_frames
            self.viewer.go_to_frame(idx)
            self.frame_label.setText(f"{idx+1}/{self.total_frames}")

    def on_frame_play(self):
        if not self.viewer:
            return
        if self.is_backward:
            idx = (self.viewer.get_current_frame() - 1) % self.total_frames
            self.viewer.go_to_frame(idx)
        else:
            self.viewer.play_next_frame()
        self.frame_label.setText(f"{self.viewer.get_current_frame()+1}/{self.total_frames}")

    def toggle_fence(self, idx):
        self.ctrl.toggle_fence_mode(idx)
        for i, btn in enumerate(self.fence_btns):
            if i < len(self.ctrl.fences) and self.ctrl.fences[i].get('mode', False):
                btn.setStyleSheet("background-color: #00ff00; color: black;")
                btn.setText(f"围栏{i+1}完成")
            else:
                btn.setStyleSheet("")
                btn.setText(f"围栏{i+1}")
        if self.viewer:
            self.viewer.update_display()

    def clear_fence(self, idx):
        self.ctrl.clear_fence(idx)
        self.fence_btns[idx].setStyleSheet("")
        self.fence_btns[idx].setText(f"围栏{idx+1}")
        if self.viewer:
            self.viewer.update_display()

    def toggle_play_fast(self):
        self.is_backward = False
        self.backward_btn.setText("倒帧")
        self.backward_fast_btn.setText("倒播")
        if self.is_playing and not self.is_backward and self.play_timer.interval() == 100:
            self.play_timer.stop()
            self.is_playing = False
            self.next_btn.setText("正帧")
            self.forward_fast_btn.setText("正播")
        else:
            self.play_timer.stop()
            self.play_timer.start(100)
            self.is_playing = True
            self.is_backward = False
            self.next_btn.setText("正帧")
            self.forward_fast_btn.setText("■正播")

    def toggle_play(self):
        self.is_backward = False
        self.backward_btn.setText("倒帧")
        self.backward_fast_btn.setText("倒播")
        if self.is_playing and not self.is_backward and self.play_timer.interval() == 1000:
            self.play_timer.stop()
            self.is_playing = False
            self.next_btn.setText("正帧")
            self.forward_fast_btn.setText("正播")
        else:
            self.play_timer.stop()
            self.play_timer.start(1000)
            self.is_playing = True
            self.is_backward = False
            self.next_btn.setText("▶正帧")
            self.forward_fast_btn.setText("正播")

    def toggle_backward(self):
        self.is_playing = False
        self.next_btn.setText("正帧")
        self.forward_fast_btn.setText("正播")
        if self.is_backward and self.play_timer.interval() == 1000:
            self.play_timer.stop()
            self.is_backward = False
            self.backward_btn.setText("倒帧")
            self.backward_fast_btn.setText("倒播")
        else:
            self.play_timer.stop()
            self.play_timer.start(1000)
            self.is_backward = True
            self.backward_btn.setText("▶倒帧")
            self.backward_fast_btn.setText("倒播")

    def toggle_backward_fast(self):
        self.is_playing = False
        self.next_btn.setText("正帧")
        self.forward_fast_btn.setText("正播")
        if self.is_backward and self.play_timer.interval() == 100:
            self.play_timer.stop()
            self.is_backward = False
            self.backward_btn.setText("倒帧")
            self.backward_fast_btn.setText("倒播")
        else:
            self.play_timer.stop()
            self.play_timer.start(100)
            self.is_backward = True
            self.backward_btn.setText("倒帧")
            self.backward_fast_btn.setText("■倒播")

    def toggle_prompt_mode(self):
        if not self.viewer:
            QMessageBox.warning(self, "错误", "请先 Show 打开预览")
            return
        if not self.prompt_drawing_mode:
            self.prompt_drawing_mode = True
            self.prompt_frame_idx = self.viewer.get_current_frame()
            self.prompt_btn.setText("执行提示帧")
            self.prompt_btn.setStyleSheet("QPushButton { background-color: #00CC00; color: white; border: none; border-radius: 3px; } QPushButton:hover { background-color: #009900; }")
            self.viewer.enable_bbox_drawing(True)
            self.viewer.clear_prompt_bboxes()
            print(f"提示帧模式：在帧 {self.prompt_frame_idx + 1} 上绘制 Bbox")
        else:
            self.prompt_btn.setEnabled(False)
            self.prompt_btn.setText("处理中...")
            self.viewer.enable_bbox_drawing(False)
            self.prompt_drawing_mode = False
            self.do_bidirectional_inject()
    
    def toggle_fixed_bbox_mode(self):
        """切换固定框编辑模式"""
        if not self.viewer:
            QMessageBox.warning(self, "错误", "请先显示预览")
            return
        if self.prompt_drawing_mode:
            QMessageBox.warning(self, "提示", "请先退出提示帧模式")
            return
        if not hasattr(self, 'fixed_bbox_mode') or not self.fixed_bbox_mode:
            self.fixed_bbox_mode = True
            self.fixed_bbox_frame_idx = self.viewer.get_current_frame()
            self.fixed_edit_btn.setText("退出编辑")
            self.viewer.enable_bbox_drawing(True)
            self.viewer.clear_prompt_bboxes()
            print(f"固定框编辑模式：在帧 {self.fixed_bbox_frame_idx + 1} 上绘制 Bbox")
        else:
            self.fixed_bbox_mode = False
            self.fixed_edit_btn.setText("编辑固定框")
            self.viewer.enable_bbox_drawing(False)
    
    def _do_semantic_with_tracking(self, items_text):
        """纯语义模式：语义分割后前向/后向追踪合并"""
        prompt_idx = self.prompt_frame_idx
        total = self.total_frames
        temp_mid = Path(TEMP_DATA_MID_DIR)
        mid_frames_dir = temp_mid / "frames"
        mid_labels_dir = temp_mid / "labels"
        
        print(f"[纯语义+追踪] 开始处理...")
        
        try:
            from annotate_video import merge_masks_in_frame, TrackManager, get_device, SAM_MODEL_PATH
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
            
            _patch_sam3_video_semantic()
            
            device, device_type = get_device()
            half = device_type == 'cuda'
            overrides = dict(
                conf=0.25, task="segment", mode="predict",
                model=SAM_MODEL_PATH, device=device,
                half=half, save=False, verbose=False
            )
            if device_type == 'cuda':
                overrides['batch'] = 1
                overrides['stream_buffer'] = False
            elif device_type == 'mps':
                overrides['half'] = True
                overrides['amp'] = True
                overrides['stream_buffer'] = True
            
            predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            
            # 收集 occupied_bands 和 FIRST_ID
            src_annotations_file = temp_mid / "annotations.json"
            occupied_bands = set()
            if src_annotations_file.exists():
                with open(src_annotations_file, encoding='utf-8') as f:
                    coco = json.load(f)
                for ann in coco.get('annotations', []):
                    tid = ann.get('track_id', 0)
                    occupied_bands.add((tid // 1000) * 1000)
            FIRST_ID = 1000
            for band in range(1000, 1000000, 1000):
                if band not in occupied_bands:
                    FIRST_ID = band
                    break
            else:
                QMessageBox.warning(self, "错误", "所有 track_id 档位都已被占用")
                self.reset_prompt_btn()
                return
            
            print(f"[纯语义+追踪] FIRST_ID={FIRST_ID}")
            
            # 纯语义模式：用文本提示在提示帧生成bboxes
            prompt_frame_path = mid_frames_dir / f"frame_{prompt_idx:06d}.jpg"
            results = predictor(source=str(prompt_frame_path), text=items_text)
            
            # 从结果提取bboxes
            prompt_bboxes = []
            for r in results:
                masks = r.masks
                if masks is not None:
                    masks_np = masks.data.cpu().numpy() if hasattr(masks, 'data') else np.array(masks)
                    for mask in masks_np:
                        mask_binary = (mask > 0.5).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if len(cnt) >= 3:
                                xs = cnt.squeeze()[0::2].tolist()
                                ys = cnt.squeeze()[1::2].tolist()
                                x1, x2 = min(xs), max(xs)
                                y1, y2 = min(ys), max(ys)
                                # 转换为xyxy格式
                                prompt_bboxes.append([x1, y1, x2, y2])
            
            if not prompt_bboxes:
                QMessageBox.warning(self, "提示", "未检测到分割结果")
                self.reset_prompt_btn()
                return
            
            print(f"[纯语义+追踪] 检测到 {len(prompt_bboxes)} 个目标，使用bboxes进行追踪")
            
            # 使用语义+追踪模式处理（复用现有逻辑）
            is_semantic = True
            
            # 执行前向追踪
            if self.forward_cb.isChecked():
                forward_start = prompt_idx + 1
                print(f"[前向] 帧 {forward_start} → {total-1}")
                self._process_semantic_clip(forward_start, total, True, prompt_bboxes, items_text, FIRST_ID)
            
            # 执行后向追踪
            if self.backward_cb.isChecked():
                print(f"[后向] 帧 0 → {prompt_idx-1}")
                self._process_semantic_clip(0, prompt_idx, False, prompt_bboxes, items_text, FIRST_ID)
            
            self.reset_prompt_btn()
            self.viewer.update_display()
            QMessageBox.information(self, "完成", "语义+追踪完成")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"语义+追踪失败:\n{e}")
            self.reset_prompt_btn()
    
    def _process_semantic_clip(self, start_frame, end_frame, forward, prompt_bboxes, items_text, FIRST_ID):
        """处理语义分割的clip，使用bboxes追踪"""
        from annotate_video import merge_masks_in_frame, TrackManager, get_device, SAM_MODEL_PATH
        
        direction = "向前" if forward else "向后"
        temp_mid = Path(TEMP_DATA_MID_DIR)
        mid_frames_dir = temp_mid / "frames"
        
        if start_frame >= end_frame:
            return
        
        temp_frames = Path("temp_inject") / ("forward" if forward else "backward")
        temp_frames.mkdir(parents=True, exist_ok=True)
        
        # 复制帧
        frame_count = end_frame - start_frame
        if forward:
            for i in range(start_frame, end_frame):
                src = mid_frames_dir / f"frame_{i:06d}.jpg"
                dst = temp_frames / f"frame_{i - start_frame:06d}.jpg"
                if src.exists():
                    shutil.copy2(src, dst)
        else:
            for rev_idx, i in enumerate(range(end_frame - 1, start_frame - 1, -1)):
                src = mid_frames_dir / f"frame_{i:06d}.jpg"
                dst = temp_frames / f"frame_{rev_idx:06d}.jpg"
                if src.exists():
                    shutil.copy2(src, dst)
        
        # 生成clip
        clip_path = str(temp_frames / "clip.mp4")
        sample = cv2.imread(str(temp_frames / "frame_000000.jpg"))
        height, width = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, 30, (width, height))
        for i in range(frame_count):
            frame = cv2.imread(str(temp_frames / f"frame_{i:06d}.jpg"))
            out.write(frame)
        out.release()
        
        # 加载模型
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        _patch_sam3_video_semantic()
        device, device_type = get_device()
        overrides = dict(
            conf=0.25, task="segment", mode="predict",
            model=SAM_MODEL_PATH, device=device,
            half=device_type == 'cuda', save=False, verbose=False
        )
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        
        # 使用bboxes追踪
        print(f"[{direction}] 使用bboxes追踪: {len(prompt_bboxes)} 个目标")
        results = predictor(source=clip_path, stream=True, bboxes=prompt_bboxes, labels=[1]*len(prompt_bboxes), text=items_text)
        
        # 处理结果
        manager = TrackManager(iou_threshold=float(self.iou_input.text() or "0.02"))
        manager.next_track_id = FIRST_ID
        merge_iou_val = float(self.merge_iou_input.text() or "0.5")
        
        frame_anns_list = []
        for r in results:
            masks = r.masks
            if masks is None:
                frame_anns_list.append([])
                continue
            
            masks_np = masks.data.cpu().numpy() if hasattr(masks, 'data') else np.array(masks)
            cur_masks, cur_bboxes = [], []
            for mask in masks_np:
                mask_binary = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) >= 3:
                        poly = cnt.squeeze().flatten().tolist()
                        xs, ys = poly[0::2], poly[1::2]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        bb = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        area = cv2.contourArea(cnt)
                        if area > 0:
                            cur_masks.append(mask_binary)
                            cur_bboxes.append(bb)
            
            if cur_masks:
                cur_masks, cur_bboxes = merge_masks_in_frame(cur_masks, cur_bboxes, merge_iou_val)
                track_ids = manager.update(cur_masks, cur_bboxes, 0)
                frame_anns = []
                for idx, (m, bb) in enumerate(zip(cur_masks, cur_bboxes)):
                    tid = track_ids[idx] if idx < len(track_ids) else FIRST_ID
                    if tid is None:
                        # 被舍弃的检测
                        continue
                    ann = {
                        'track_id': tid,
                        'bbox': bb,
                        'category': items_text,
                        'trace_id_list': [tid]
                    }
                    frame_anns.append(ann)
                frame_anns_list.append(frame_anns)
            else:
                frame_anns_list.append([])
        
        # 保存结果
        src_labels_dir = temp_mid / "labels"
        for i, frame_anns in enumerate(frame_anns_list):
            if forward:
                orig_idx = start_frame + i
            else:
                orig_idx = end_frame - 1 - i
            
            label_file = src_labels_dir / f"frame_{orig_idx:06d}.json"
            existing = []
            if label_file.exists():
                with open(label_file, encoding='utf-8') as f:
                    existing = json.load(f)
            merged = existing + frame_anns
            with open(label_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False)
        
        print(f"[{direction}] 完成")
    
    def extract_video_clip_from_frames(self, frames_dir, start_idx, total_frames, output_path, fps=30):
        sample = cv2.imread(str(frames_dir / f"frame_{start_idx:06d}.jpg"))
        if sample is None:
            raise ValueError(f"无法读取起始帧: frame_{start_idx:06d}.jpg")
        height, width = sample.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_count = 0
        for i in range(start_idx, total_frames):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            if not frame_path.exists():
                break
            frame = cv2.imread(str(frame_path))
            if frame is None:
                break
            out.write(frame)
            frame_count += 1
        out.release()
        print(f"视频片段已从帧目录提取: {output_path} ({frame_count} 帧)")
        return frame_count

    def do_bidirectional_inject(self):
        prompt_bboxes = self.viewer.get_prompt_bboxes()
        items_text = self.items_input.text().strip()
        
        # 根据输入决定模式
        # 1. 语义+追踪：items有内容 且 prompt_bboxes有内容
        # 2. 纯语义：items有内容 且 prompt_bboxes为空
        # 3. 纯追踪：items为空 且 prompt_bboxes有内容
        has_items = bool(items_text)
        has_bboxes = bool(prompt_bboxes)
        
        if not has_items and not has_bboxes:
            QMessageBox.warning(self, "错误", "请先绘制 Bbox 或填写物品名称")
            self.reset_prompt_btn()
            return
        
        prompt_idx = self.prompt_frame_idx
        total = self.total_frames

        temp_mid = Path(TEMP_DATA_MID_DIR)
        mid_frames_dir = temp_mid / "frames"
        mid_labels_dir = temp_mid / "labels"
        mid_annotations_file = temp_mid / "annotations.json"

        src_frames_dir = self.temp_data_path / "frames"
        src_labels_dir = temp_mid / "labels"
        src_annotations_file = temp_mid / "annotations.json"

        self.prompt_btn.setText("正在处理...")
        QApplication.processEvents()
        
        # 纯语义模式：先用文本分割得到bboxes，再前向+后向追踪
        if has_items and not has_bboxes:
            try:
                from ultralytics.models.sam import SAM3SemanticPredictor
                from annotate_video import get_device, SAM_MODEL_PATH
                device, device_type = get_device()
                overrides = dict(
                    conf=0.25, task="segment", mode="predict",
                    model=SAM_MODEL_PATH, device=device,
                    half=device_type == 'cuda', save=False, verbose=False
                )
                predictor = SAM3SemanticPredictor(overrides=overrides)
                
                # 在提示帧上做文本分割得到bboxes
                prompt_frame_path = mid_frames_dir / f"frame_{prompt_idx:06d}.jpg"
                # 参考现有代码：用 text 参数
                predictor_args = {
                    'source': str(prompt_frame_path),
                }
                if items_text:
                    predictor_args['text'] = items_text
                results = list(predictor(**predictor_args))
                
                prompt_bboxes = []
                for r in results:
                    if r.masks is not None:
                        masks_np = r.masks.data.cpu().numpy() if hasattr(r.masks, 'data') else np.array(r.masks)
                        for mask in masks_np:
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in contours:
                                if len(cnt) >= 3:
                                    xs = cnt.squeeze()[0::2].tolist()
                                    ys = cnt.squeeze()[1::2].tolist()
                                    x1, x2 = min(xs), max(xs)
                                    y1, y2 = min(ys), max(ys)
                                    prompt_bboxes.append([x1, y1, x2, y2])
                
                if not prompt_bboxes:
                    QMessageBox.warning(self, "提示", "未检测到分割结果")
                    self.reset_prompt_btn()
                    return
                
                # 保存提示帧标注
                label_file = src_labels_dir / f"frame_{prompt_idx:06d}.json"
                existing = []
                if label_file.exists():
                    with open(label_file, encoding='utf-8') as f:
                        existing = json.load(f)
                prompt_anns = []
                for i, pb in enumerate(prompt_bboxes):
                    x1, y1, x2, y2 = pb
                    prompt_anns.append({
                        'id': i + 1, 'track_id': 0, 'image_id': prompt_idx,
                        'category_id': 0, 'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': float((x2 - x1) * (y2 - y1)),
                        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]],
                        'iscrowd': 0, 'confidence': 1.0,
                        'category': items_text, 'trace_id_list': [0]
                    })
                merged = existing + prompt_anns
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(merged, f, ensure_ascii=False)
                
                # 提示帧本身用语义分割结果，但追踪时用bboxes
                # 设置提示帧的bboxes供后续process_clip使用
                self.viewer.set_prompt_bboxes([[pb[0], pb[1], pb[2], pb[3]] for pb in prompt_bboxes])
                prompt_bboxes = [[pb[0], pb[1], pb[2], pb[3]] for pb in prompt_bboxes]
                
                # 用语义+追踪模式继续（items_text有值，prompt_bboxes有值）
                is_semantic = True
                has_bboxes = True  # 已有bboxes
                
                print(f"[提示帧] 语义+追踪: 文本={items_text}, bboxes={len(prompt_bboxes)}")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "错误", f"语义分割失败:\n{e}")
                self.reset_prompt_btn()
                return

        try:
            from annotate_video import merge_masks_in_frame, TrackManager, get_device, SAM_MODEL_PATH, put_chinese_text
            
            # 根据模式选择预测器
            is_semantic = has_items and has_bboxes
            if is_semantic:
                # 语义+追踪模式
                from ultralytics.models.sam import SAM3VideoSemanticPredictor
                _patch_sam3_video_semantic()
                print(f"[提示帧] 模式: 语义+追踪, 物品: {items_text}")
            else:
                # 纯追踪模式
                from ultralytics.models.sam import SAM3VideoPredictor
                print(f"[提示帧] 模式: 纯追踪")

            device, device_type = get_device()
            half = device_type == 'cuda'
            overrides = dict(
                conf=0.25, task="segment", mode="predict",
                model=SAM_MODEL_PATH, device=device,
                half=half, save=False, verbose=False
            )
            if device_type == 'cuda':
                overrides['batch'] = 1
                overrides['stream_buffer'] = False
            elif device_type == 'mps':
                overrides['half'] = True
                overrides['amp'] = True
                overrides['stream_buffer'] = True

            # 根据模式初始化对应的预测器
            if is_semantic:
                predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            else:
                predictor = SAM3VideoPredictor(overrides=overrides)

            sample_frame = cv2.imread(str(mid_frames_dir / f"frame_{0:06d}.jpg"))
            height, width = sample_frame.shape[:2]

            occupied_bands = set()
            if src_annotations_file.exists():
                with open(src_annotations_file, encoding='utf-8') as f:
                    coco = json.load(f)
                for ann in coco.get('annotations', []):
                    tid = ann.get('track_id', 0)
                    occupied_bands.add((tid // 1000) * 1000)  # 1000档
            # 遍历所有1000档，检查是否被占用
            for band in range(1000, 1000000, 1000):
                if band not in occupied_bands:
                    FIRST_ID = band
                    break
            else:
                QMessageBox.warning(self, "错误", "所有 track_id 档位都已被占用")
                self.reset_prompt_btn()
                return
            available_options = [FIRST_ID]
            FIRST_ID = available_options[0]
            # 记录本次执行的FIRST_ID用于回退
            self.last_prompt_first_id = FIRST_ID
            device_str = "GPU" if device_type == 'cuda' else ("MPS" if device_type == 'mps' else "CPU")
            print(f"=== 双向标注开始 === 提示帧: {prompt_idx}, 总帧数: {total}, 设备: [{device_str}], 前向={self.forward_cb.isChecked()}, 后向={self.backward_cb.isChecked()}, FIRST_ID={FIRST_ID}")
            forward_annotations = []
            backward_annotations = []

            def process_clip(start_frame, end_frame, forward=True, prompt_bboxes=None):
                direction = "向前" if forward else "向后"
                print(f"\n[DEBUG {direction}] === 进入 process_clip ===")
                print(f"[DEBUG {direction}] start_frame={start_frame}, end_frame={end_frame}, 总帧数={end_frame - start_frame}, 设备=[{device_str}]")

                if start_frame >= end_frame:
                    print(f"[DEBUG {direction}] start_frame >= end_frame, 直接返回空列表")
                    return []

                temp_frames = Path("temp_inject") / ("forward" if forward else "backward")
                temp_frames.mkdir(parents=True, exist_ok=True)

                frame_count = end_frame - start_frame
                print(f"[DEBUG {direction}] 正在复制 {frame_count} 帧到临时目录...")
                if forward:
                    for i in range(start_frame, end_frame):
                        src = mid_frames_dir / f"frame_{i:06d}.jpg"
                        dst = temp_frames / f"frame_{i - start_frame:06d}.jpg"
                        if src.exists():
                            shutil.copy2(src, dst)
                        else:
                            print(f"[DEBUG {direction}] ⚠️ 帧文件不存在: {src}")
                else:
                    for rev_idx, i in enumerate(range(end_frame - 1, start_frame - 1, -1)):
                        src = mid_frames_dir / f"frame_{i:06d}.jpg"
                        dst = temp_frames / f"frame_{rev_idx:06d}.jpg"
                        if src.exists():
                            shutil.copy2(src, dst)
                        else:
                            print(f"[DEBUG {direction}] ⚠️ 帧文件不存在: {src}")
                print(f"[DEBUG {direction}] ✓ 帧复制完成: {frame_count} 帧")

                clip_path = str(temp_frames / "clip.mp4")
                print(f"[DEBUG {direction}] 正在生成视频片段: {clip_path}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps_cap = 30
                out = cv2.VideoWriter(clip_path, fourcc, fps_cap, (width, height))
                frames_written = 0
                if forward:
                    for i in range(start_frame, end_frame):
                        frame = cv2.imread(str(mid_frames_dir / f"frame_{i:06d}.jpg"))
                        if frame is not None:
                            out.write(frame)
                            frames_written += 1
                else:
                    for rev_idx, i in enumerate(range(end_frame - 1, start_frame - 1, -1)):
                        frame = cv2.imread(str(mid_frames_dir / f"frame_{i:06d}.jpg"))
                        if frame is not None:
                            out.write(frame)
                            frames_written += 1
                out.release()
                print(f"[DEBUG {direction}] ✓ 视频片段生成完成: {frames_written} 帧")
                cap_check = cv2.VideoCapture(clip_path)
                actual_clip_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_check.release()
                print(f"[DEBUG {direction}] clip文件实际帧数: {actual_clip_frames}, expected: {end_frame - start_frame}")

                print(f"[DEBUG {direction}] 正在加载 predictor 处理...")
                print(f"[DEBUG {direction}] prompt_bboxes={prompt_bboxes}, is_semantic={is_semantic}")
                if is_semantic:
                    # 语义模式：使用文本提示
                    results = predictor(source=clip_path, stream=True, bboxes=prompt_bboxes, labels=[1]*len(prompt_bboxes), text=items_text)
                elif prompt_bboxes:
                    # 纯追踪模式：使用bbox提示
                    results = predictor(source=clip_path, stream=True, bboxes=prompt_bboxes, labels=[1]*len(prompt_bboxes))
                else:
                    # 无提示：纯追踪
                    results = predictor(source=clip_path, stream=True)
                    print(f"[DEBUG {direction}] ⚠️ 无提示，使用无提示模式")
                manager = TrackManager(iou_threshold=float(self.iou_input.text() or "0.02"))
                manager.next_track_id = FIRST_ID
                ann_id = FIRST_ID
                merge_iou_val = float(self.merge_iou_input.text() or "0.5")
                print(f"[DEBUG {direction}] TrackManager 初始化: next_track_id={manager.next_track_id}, iou={manager.iou_threshold}")

                result_anns = []
                frame_idx = 0
                total_results = 0

                print(f"[DEBUG {direction}] 开始遍历 predictor 结果...")
                for r in results:
                    total_results += 1
                    print(f"[DEBUG {direction}] [帧{total_results}] 收到结果对象")

                    orig_img = r.orig_img if hasattr(r, 'orig_img') and r.orig_img is not None else None
                    if orig_img is None:
                        cap_t = cv2.VideoCapture(clip_path)
                        cap_t.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret_t, orig_img = cap_t.read()
                        cap_t.release()
                        if not ret_t:
                            orig_img = np.zeros((height, width, 3), dtype=np.uint8)
                            print(f"[DEBUG {direction}] [帧{total_results}] ⚠️ cap fallback 也失败，使用空白图")
                    else:
                        print(f"[DEBUG {direction}] [帧{total_results}] 使用 orig_img")

                    if len(orig_img.shape) == 2:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                    elif orig_img.shape[2] == 4:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(str(temp_frames / f"frame_{frame_idx:06d}.jpg"), orig_img)

                    frame_anns = []
                    debug_masks_count = 0
                    debug_contours_count = 0
                    debug_merged_count = 0
                    debug_track_ids = []

                    has_masks = hasattr(r, 'masks') and r.masks is not None
                    print(f"[DEBUG {direction}] [帧{total_results}] has_masks={has_masks}")

                    if has_masks:
                        masks_tensor = r.masks.data
                        has_tensor = masks_tensor is not None and len(masks_tensor) > 0
                        print(f"[DEBUG {direction}] [帧{total_results}] masks_tensor: {has_tensor}, len={len(masks_tensor) if has_tensor else 0}")

                        if has_tensor:
                            debug_masks_count = len(masks_tensor)
                            confs = None
                            if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'conf'):
                                confs = r.boxes.conf.cpu().numpy()
                                print(f"[DEBUG {direction}] [帧{total_results}] boxes.conf={confs.tolist()}")

                            cur_masks = []
                            cur_bboxes = []
                            for mask in masks_tensor:
                                mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
                                mask_binary = (mask_np > 0.5).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                debug_contours_count += len(contours)
                                for cnt in contours:
                                    if len(cnt) >= 3:
                                        poly = cnt.squeeze().flatten().tolist()
                                        xs = poly[0::2]
                                        ys = poly[1::2]
                                        x1, x2 = min(xs), max(xs)
                                        y1, y2 = min(ys), max(ys)
                                        bb = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                                        area = cv2.contourArea(cnt)
                                        if area > 0:
                                            cur_masks.append(mask_binary)
                                            cur_bboxes.append(bb)

                            debug_merged_count = len(cur_masks)
                            print(f"[DEBUG {direction}] [帧{total_results}] masks={debug_masks_count}, contours={debug_contours_count}, 有效polygon={debug_merged_count}")

                            if cur_masks:
                                cur_masks, cur_bboxes = merge_masks_in_frame(cur_masks, cur_bboxes, merge_iou_val)
                                track_ids = manager.update(cur_masks, cur_bboxes, frame_idx)
                                debug_track_ids = track_ids
                                print(f"[DEBUG {direction}] [帧{total_results}] merge后={len(cur_masks)}, track_ids={track_ids}")

                                for idx, (m, bb) in enumerate(zip(cur_masks, cur_bboxes)):
                                    m_bin = (m > 0.5).astype(np.uint8)
                                    cnts2, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    for cnt2 in cnts2:
                                        if len(cnt2) >= 3:
                                            poly2 = cnt2.squeeze().flatten().tolist()
                                            area2 = cv2.contourArea(cnt2)
                                            tid = track_ids[idx] if idx < len(track_ids) else ann_id
                                            conf = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                                            if forward:
                                                img_id = frame_idx + start_frame
                                            else:
                                                img_id = end_frame - 1 - frame_idx
                                            ann = {
                                                'id': ann_id, 'track_id': tid, 'image_id': img_id,
                                                'category_id': tid, 'bbox': bb, 'area': float(area2),
                                                'segmentation': [poly2], 'iscrowd': 0, 'confidence': conf,
                                                'category': 'Detect', 'trace_id_list': [tid]
                                            }
                                            result_anns.append(ann)
                                            frame_anns.append(ann)
                                            ann_id += 1
                                print(f"[DEBUG {direction}] [帧{total_results}] 本帧标注数={len(frame_anns)}, 累计={len(result_anns)}")
                            else:
                                print(f"[DEBUG {direction}] [帧{total_results}] ⚠️ cur_masks为空，跳过")
                        else:
                            print(f"[DEBUG {direction}] [帧{total_results}] ⚠️ masks_tensor为空或长度=0")
                    else:
                        print(f"[DEBUG {direction}] [帧{total_results}] ⚠️ 无masks属性或masks为None")

                    if forward:
                        orig_frame_idx = frame_idx + start_frame
                    else:
                        orig_frame_idx = end_frame - 1 - frame_idx
                    clip_frames = end_frame - start_frame
                    if orig_frame_idx >= total:
                        print(f"[DEBUG {direction}] [帧{frame_idx + 1}/{clip_frames}] ⚠️ orig_frame_idx={orig_frame_idx} >= total={total}，跳过")
                        frame_idx += 1
                        continue
                    print(f"[DEBUG {direction}] [帧{frame_idx + 1}/{clip_frames}] clip_frame={frame_idx} → 原帧{orig_frame_idx}, 新增标注数={len(frame_anns)}")
                    label_file = src_labels_dir / f"frame_{orig_frame_idx:06d}.json"
                    existing_anns = []
                    if label_file.exists():
                        with open(label_file, encoding='utf-8') as f:
                            existing_anns = json.load(f)
                        print(f"[DEBUG {direction}] [帧{total_results}] 已存在标注{len(existing_anns)}条，追加新标注")
                    merged_anns = existing_anns + frame_anns
                    with open(label_file, 'w', encoding='utf-8') as f:
                        json.dump(merged_anns, f, ensure_ascii=False)
                    print(f"[DEBUG {direction}] [帧{total_results}] 保存label文件: frame_{orig_frame_idx:06d}.json, 保留{len(existing_anns)}+新增{len(frame_anns)}=合计{len(merged_anns)}")
                    frame_idx += 1

                print(f"[DEBUG {direction}] === process_clip 完成 ===")
                print(f"[DEBUG {direction}] 总results数={total_results}, 总frame_idx={frame_idx}, 总annotations={len(result_anns)}, clip帧数={end_frame - start_frame}")
                if total_results != (end_frame - start_frame):
                    print(f"[DEBUG {direction}] ⚠️ 警告: predictor返回{total_results}个结果，但clip有{end_frame - start_frame}帧，可能有帧对齐问题！")
                print(f"[DEBUG {direction}] id范围: {FIRST_ID} ~ {ann_id - 1}")
                return result_anns

            print(f"=== 双向标注开始 === 提示帧: {prompt_idx}, 总帧数: {total}, 设备: [{device_str}], 前向={self.forward_cb.isChecked()}, 后向={self.backward_cb.isChecked()}, FIRST_ID={FIRST_ID}")
            forward_anns = []
            backward_anns = []
            prompt_frame_anns = []
            prompt_ann_id = FIRST_ID
            
            # 保存提示帧本身的bbox和segmentation
            prompt_frame_file = src_labels_dir / f"frame_{prompt_idx:06d}.json"
            if prompt_bboxes:
                prompt_anns = []
                for pb in prompt_bboxes:
                    x1, y1, x2, y2 = pb
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    # 创建简单的polygon矩形
                    poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                    area = float((x2 - x1) * (y2 - y1))
                    ann = {
                        'id': prompt_ann_id, 'track_id': FIRST_ID, 'image_id': prompt_idx,
                        'category_id': 0, 'bbox': bbox, 'area': area,
                        'segmentation': [poly], 'iscrowd': 0, 'confidence': 1.0,
                        'category': 'Detect', 'trace_id_list': [FIRST_ID]
                    }
                    prompt_anns.append(ann)
                    prompt_ann_id += 1
                # 保存到label文件
                existing = []
                if prompt_frame_file.exists():
                    with open(prompt_frame_file, encoding='utf-8') as f:
                        existing = json.load(f)
                merged = existing + prompt_anns
                with open(prompt_frame_file, 'w', encoding='utf-8') as f:
                    json.dump(merged, f, ensure_ascii=False)
                prompt_frame_anns = prompt_anns
                print(f"[提示帧] 已保存 {len(prompt_anns)} 个标注到 frame_{prompt_idx:06d}.json")
            
            if self.forward_cb.isChecked():
                forward_start = prompt_idx + 1
                print(f"[1/2] 向前标注: 帧 {forward_start} → {total-1} (共 {total - forward_start} 帧)")
                forward_anns = process_clip(forward_start, total, forward=True, prompt_bboxes=prompt_bboxes)

            if self.backward_cb.isChecked():
                print(f"\n[2/2] 向后标注: 帧 0 → {prompt_idx-1} (共 {prompt_idx} 帧)")
                backward_anns = process_clip(0, prompt_idx, forward=False, prompt_bboxes=prompt_bboxes)

            all_new_anns = backward_anns + forward_anns + prompt_frame_anns
            print(f"\n[DEBUG 汇总] 向后标注={len(backward_anns)}, 向前标注={len(forward_anns)}, 提示帧标注={len(prompt_frame_anns)}, 合计={len(all_new_anns)}")

            if not all_new_anns:
                QMessageBox.warning(self, "提示", "未检测到任何分割结果")
                self.reset_prompt_btn()
                return

            if src_annotations_file.exists():
                with open(src_annotations_file, encoding='utf-8') as f:
                    coco = json.load(f)
                print(f"[DEBUG 汇总] 现有coco: 已有annotations={len(coco.get('annotations', []))}")
            else:
                coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}
                print(f"[DEBUG 汇总] annotations.json 不存在，创建新的coco结构")

            max_img_id = max([img['id'] for img in coco.get('images', [])], default=-1)
            max_ann_id = max([ann['id'] for ann in coco.get('annotations', [])], default=FIRST_ID - 1)
            max_track_id = max([ann['track_id'] for ann in coco.get('annotations', [])], default=FIRST_ID - 1)
            print(f"[DEBUG 汇总] 现有max_ann_id={max_ann_id}, max_track_id={max_track_id}")

            # 检查与现有标注重叠的IoU阈值
            overlap_threshold = float(self.merge_iou_input.text()) if self.merge_iou_input.text() else 0.5
            
            new_anns_count = 0
            skipped_count = 0
            for ann in all_new_anns:
                # 检查是否与现有标注重叠
                new_bbox = ann.get('bbox', [])
                skip = False
                for existing in coco['annotations']:
                    existing_bbox = existing.get('bbox', [])
                    if new_bbox and existing_bbox and existing.get('image_id') == ann.get('image_id'):
                        # 计算IoU
                        x1 = max(new_bbox[0], existing_bbox[0])
                        y1 = max(new_bbox[1], existing_bbox[1])
                        x2 = min(new_bbox[0] + new_bbox[2], existing_bbox[0] + existing_bbox[2])
                        y2 = min(new_bbox[1] + new_bbox[3], existing_bbox[1] + existing_bbox[3])
                        inter = max(0, x2 - x1) * max(0, y2 - y1)
                        area1 = new_bbox[2] * new_bbox[3]
                        area2 = existing_bbox[2] * existing_bbox[3]
                        union = area1 + area2 - inter
                        iou = inter / union if union > 0 else 0
                        if iou > overlap_threshold:
                            print(f"[DEBUG 汇总] 跳过重叠标注: frame={ann.get('image_id')}, iou={iou:.3f}")
                            skip = True
                            break
                
                if skip:
                    skipped_count += 1
                    continue
                
                new_ann = dict(ann)
                max_ann_id += 1
                new_ann['id'] = max_ann_id
                new_ann['track_id'] = max_track_id + 1
                new_ann['category_id'] = new_ann['track_id']
                max_track_id = new_ann['track_id']
                coco['annotations'].append(new_ann)
                new_anns_count += 1

            print(f"[DEBUG 汇总] 追加 {new_anns_count} 条标注, 最终track_id范围: {FIRST_ID}~{max_track_id}")

            with open(src_annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco, f, ensure_ascii=False)
            print(f"[DEBUG 汇总] ✓ annotations.json 已写入")

            # 更新labels目录下的帧标注文件
            temp_mid = Path(TEMP_DATA_MID_DIR)
            labels_dir = temp_mid / "labels"
            labels_dir.mkdir(exist_ok=True)
            for ann in coco['annotations']:
                frame_idx = ann.get('image_id', 0)
                label_file = labels_dir / f"frame_{frame_idx:06d}.json"
                frame_anns = []
                if label_file.exists():
                    with open(label_file, encoding='utf-8') as f:
                        frame_anns = json.load(f)
                
                # 检查bbox是否与现有标注重叠
                new_bbox = ann.get('bbox', [])
                skip = False
                for existing in frame_anns:
                    existing_bbox = existing.get('bbox', [])
                    if new_bbox and existing_bbox:
                        # 计算IoU
                        x1 = max(new_bbox[0], existing_bbox[0])
                        y1 = max(new_bbox[1], existing_bbox[1])
                        x2 = min(new_bbox[0] + new_bbox[2], existing_bbox[0] + existing_bbox[2])
                        y2 = min(new_bbox[1] + new_bbox[3], existing_bbox[1] + existing_bbox[3])
                        inter = max(0, x2 - x1) * max(0, y2 - y1)
                        area1 = new_bbox[2] * new_bbox[3]
                        area2 = existing_bbox[2] * existing_bbox[3]
                        union = area1 + area2 - inter
                        iou = inter / union if union > 0 else 0
                        if iou > overlap_threshold:
                            skip = True
                            break
                
                if not skip:
                    frame_anns.append(ann)
                    with open(label_file, 'w', encoding='utf-8') as f:
                        json.dump(frame_anns, f, ensure_ascii=False)
            print(f"[DEBUG 汇总] ✓ labels目录已更新")

            shutil.rmtree(Path("temp_inject"), ignore_errors=True)

            print(f"=== 双向标注完成 === 新增标注: {len(all_new_anns)}, 总标注: {len(coco['annotations'])}")
            self.statusBar().showMessage(f"双向标注完成: +{len(all_new_anns)} 条")
            QMessageBox.information(self, "完成", f"双向标注完成！\n新增标注: {len(all_new_anns)} 条\n追加到 temp_data")
            if self.viewer:
                self.viewer.coco_data = coco
                self.viewer.total_frames = total
                self.viewer.go_to_frame(prompt_idx)
                self.frame_label.setText(f"{prompt_idx + 1}/{total}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("双向标注失败")
            QMessageBox.critical(self, "错误", f"双向标注失败:\n{e}")
        finally:
            self.reset_prompt_btn()

    def push_undo(self, undo_data):
        """记录回退信息到栈"""
        if not undo_data:
            return
        # 统一格式：单条记录的列表
        if isinstance(undo_data, dict):
            if 'changes' in undo_data:
                # 多帧格式
                entry = {'type': 'multi', 'changes': undo_data['changes']}
            else:
                # 单帧格式
                entry = undo_data
            self.undo_stack.append(entry)
        elif isinstance(undo_data, list):
            # 已经是列表格式
            if len(undo_data) == 1:
                self.undo_stack.append(undo_data[0])
            else:
                # 多帧记录
                self.undo_stack.append({'type': 'multi', 'changes': undo_data})
        self.refresh_trace_id_list()
        print(f"[Undo] 记录回退")
    
    def pop_undo(self):
        """从栈中弹出回退信息并执行回退"""
        if not self.undo_stack:
            return None
        return self.undo_stack.pop()
    
    def undo_last_prompt(self):
        """回退上一次的trace_id修改"""
        if not self.undo_stack:
            QMessageBox.warning(self, "提示", "没有可回退的操作")
            return
        entry = self.undo_stack.pop()
        self._do_undo(entry)
        self.refresh_trace_id_list()
    
    def _do_undo(self, entry):
        """执行回退"""
        labels_dir = Path(TEMP_DATA_MID_DIR) / "labels"
        if entry.get('type') == 'multi':
            changes = entry.get('changes', [])
        else:
            changes = [entry]
        
        restored = 0
        for change in changes:
            frame_idx = change.get('frame_idx')
            bbox_key = change.get('bbox_key')
            old_tid = change.get('old_trace_id')
            frame_file = labels_dir / f"frame_{frame_idx:06d}.json"
            if not frame_file.exists():
                continue
            try:
                with open(frame_file, encoding='utf-8') as f:
                        anns = json.load(f)
                if old_tid == -1:
                    # 新增的bbox，删除它
                    new_anns = []
                    for ann in anns:
                        bbox = ann.get('bbox', [])
                        if len(bbox) >= 4:
                            key = f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}"
                            if key == bbox_key:
                                restored += 1
                                continue  # 删除
                        new_anns.append(ann)
                    anns = new_anns
                else:
                    # 恢复旧值
                    for ann in anns:
                        bbox = ann.get('bbox', [])
                        if len(bbox) >= 4:
                            key = f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}"
                            if key == bbox_key:
                                ann['track_id'] = old_tid
                                restored += 1
                                break
                with open(frame_file, 'w', encoding='utf-8') as f:
                    json.dump(anns, f, ensure_ascii=False)
            except:
                pass
        
        if self.viewer:
            self.viewer.update_display()
        print(f"[Undo] 已回退 {restored} 个标注")

    def reset_prompt_btn(self):
        self.prompt_drawing_mode = False
        self.prompt_frame_idx = -1
        self.prompt_btn.setEnabled(True)
        self.prompt_btn.setText("设为提示帧")
        self.prompt_btn.setStyleSheet("QPushButton { background-color: #FFA500; color: white; border: none; border-radius: 3px; } QPushButton:hover { background-color: #FF8C00; }")
        if self.viewer:
            self.viewer.enable_bbox_drawing(False)

    def show_undo_menu(self):
        """显示回退菜单"""
        if not self.undo_stack:
            QMessageBox.information(self, "提示", "没有可回退的操作")
            return
        
        menu = QMenu(self)
        for i, entry in enumerate(self.undo_stack):
            if entry.get('type') == 'multi':
                frame_count = len(entry.get('changes', []))
                label = f"回退步骤{i+1}: 多帧{frame_count}个"
            else:
                frame_idx = entry.get('frame_idx', '?')
                old_id = entry.get('old_trace_id', '?')
                new_id = entry.get('new_trace_id', '?')
                label = f"回退步骤{i+1}: 帧{frame_idx} {old_id}→{new_id}"
            action = menu.addAction(label)
            action.triggered.connect(lambda checked, idx=i: self.undo_by_index(idx))
        
        menu.exec_(self.undo_btn.mapToGlobal(self.undo_btn.rect().bottomLeft()))
    
    def undo_by_index(self, idx):
        """按索引回退"""
        if idx < 0 or idx >= len(self.undo_stack):
            return
        entry = self.undo_stack.pop(idx)
        self._do_undo(entry)
        self.refresh_trace_id_list()

    def _get_trace_id_mappings_file(self):
        return Path(TEMP_DATA_MID_DIR) / "trace_id_changes.json"

    def _save_trace_id_mappings(self):
        mappings_file = self._get_trace_id_mappings_file()
        mappings_file.parent.mkdir(parents=True, exist_ok=True)
        mappings = []
        for i in range(self.trace_id_list.count()):
            text = self.trace_id_list.item(i).text()
            mappings.append(text)
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False)
        print(f"已保存 trace_id_mappings 到 {mappings_file}")
        self._update_category_list()

    def _load_trace_id_mappings(self):
        """从temp_data_mid扫描所有trace_id"""
        self.trace_id_list.clear()
        labels_dir = Path(TEMP_DATA_MID_DIR) / "labels"
        if labels_dir.exists():
            tid_frames = {}
            for f in labels_dir.glob("*.json"):
                try:
                    with open(f, encoding='utf-8') as fp:
                        data = json.load(fp)
                        for ann in data:
                            tid = ann.get('track_id', 0)
                            if tid >= 1000000:
                                tid_frames[tid] = tid_frames.get(tid, 0) + 1
                except:
                    pass
            for tid in sorted(tid_frames.keys()):
                self.trace_id_list.addItem(f"ID: {tid} ({tid_frames[tid]}帧)")
            print(f"已加载 trace_id: {len(tid_frames)} 个")
        self._update_category_list()
    
    def _update_category_list(self):
        """根据trace_id_list更新类别列表"""
        target_ids = []
        if hasattr(self, 'trace_id_list') and self.trace_id_list:
            for i in range(self.trace_id_list.count()):
                text = self.trace_id_list.item(i).text()
                try:
                    tid = int(text.split(":")[1].split("(")[0].strip())
                    target_ids.append(tid)
                except:
                    pass
        if not target_ids:
            target_ids = [1000000]
        
        # 更新palette_colors数量与类别数量一致
        self.palette_colors = []
        for _ in target_ids:
            # 生成随机BGR颜色
            b = random.randint(50, 255)
            g = random.randint(50, 255)
            r = random.randint(50, 255)
            self.palette_colors.append((b, g, r))
        
        # 清除旧UI
        for label in self.category_labels:
            label.setParent(None)
        for inp in self.category_inputs:
            inp.setParent(None)
        self.category_labels.clear()
        self.category_inputs.clear()
        
        # 查找category_layout并清空
        for i in reversed(range(self.category_layout.count())):
            widget = self.category_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        # 添加标题
        title = QLabel("类别名称 (trace_id → 类别):")
        self.category_layout.addWidget(title)
        
        # 添加新的类别行
        for idx, tid in enumerate(target_ids):
            row = QHBoxLayout()
            row.setSpacing(2)
            label = QLabel(f"{tid}:")
            label.setFixedWidth(80)
            row.addWidget(label)
            inp = QLineEdit("Detect")
            inp.setFixedHeight(20)
            row.addWidget(inp)
            self.category_labels.append(label)
            self.category_inputs.append(inp)
            self.category_layout.addLayout(row)
        
        # 重建颜色按钮（仅当布局已初始化时）
        if hasattr(self, 'color_btn_layout'):
            self._rebuild_color_buttons()
    
    def _rebuild_color_buttons(self):
        """重建颜色按钮以匹配类别数量"""
        if not hasattr(self, 'color_btn_layout'):
            return
        # 找到并清空color_btn_layout
        for i in reversed(range(self.color_btn_layout.count())):
            widget = self.color_btn_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        self.color_btns = []
        self.color_styles = []
        for idx, (b_val, g_val, r_val) in enumerate(self.palette_colors):
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            color = f"rgb({r_val},{g_val},{b_val})"
            active_color = "border: 2px solid #FFD700;" if idx == self.selected_color_index else ""
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; border-radius: 3px; {active_color} }}"
                f"QPushButton:selected {{ border: 2px solid #FFD700; }}"
            )
            btn.clicked.connect(lambda _, i=idx: self.on_color_select(i))
            self.color_btns.append(btn)
            self.color_btn_layout.addWidget(btn)
    
    def _apply_single_mapping_to_mid(self, from_id, to_id):
        temp_mid = Path(TEMP_DATA_MID_DIR)
        labels_dir = temp_mid / "labels"
        annotations_file = temp_mid / "annotations.json"
        if not labels_dir.exists():
            return
        count = 0
        for label_file in sorted(labels_dir.glob("frame_*.json")):
            with open(label_file, encoding='utf-8') as f:
                frame_anns = json.load(f)
            changed = False
            for ann in frame_anns:
                if ann.get('track_id') == from_id:
                    ann['track_id'] = to_id
                    ann['trace_id_list'] = [to_id]
                    changed = True
            if changed:
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(frame_anns, f, ensure_ascii=False)
                count += 1
        if annotations_file.exists():
            with open(annotations_file, encoding='utf-8') as f:
                coco = json.load(f)
            for ann in coco.get('annotations', []):
                if ann.get('track_id') == from_id:
                    ann['track_id'] = to_id
                    ann['trace_id_list'] = [to_id]
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco, f, ensure_ascii=False)
        print(f"恢复映射: {from_id} → {to_id}, 影响 {count} 帧")

    def _apply_trace_id_mappings_to_mid(self):
        temp_mid = Path(TEMP_DATA_MID_DIR)
        frames_dir = temp_mid / "frames"
        labels_dir = temp_mid / "labels"
        annotations_file = temp_mid / "annotations.json"
        history_file = temp_mid / "trace_id_history.json"

        mappings = []
        for i in range(self.trace_id_list.count()):
            text = self.trace_id_list.item(i).text()
            parts = text.replace("ID:", "").split("→")
            if len(parts) == 2:
                try:
                    old_id = int(parts[0].strip())
                    new_id = int(parts[1].strip())
                    mappings.append((old_id, new_id))
                except ValueError:
                    pass

        if not mappings:
            print("没有有效的映射规则")
            return

        history_records = [{'old_id': old_id, 'new_id': new_id} for old_id, new_id in mappings]

        converted_count = 0
        for label_file in sorted(labels_dir.glob("frame_*.json")):
            with open(label_file, encoding='utf-8') as f:
                frame_anns = json.load(f)
            changed = False
            for ann in frame_anns:
                for old_id, new_id in mappings:
                    if ann.get('track_id') == old_id:
                        ann['track_id'] = new_id
                        trace_list = ann.get('trace_id_list', [])
                        if not trace_list or trace_list[-1] != new_id:
                            trace_list.append(new_id)
                            ann['trace_id_list'] = trace_list
                        changed = True
            if changed:
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(frame_anns, f, ensure_ascii=False)
                converted_count += 1

        if annotations_file.exists():
            with open(annotations_file, encoding='utf-8') as f:
                coco = json.load(f)
            changed = False
            for ann in coco.get('annotations', []):
                for old_id, new_id in mappings:
                    if ann.get('track_id') == old_id:
                        ann['track_id'] = new_id
                        trace_list = ann.get('trace_id_list', [])
                        if not trace_list or trace_list[-1] != new_id:
                            trace_list.append(new_id)
                            ann['trace_id_list'] = trace_list
                        changed = True
            if changed:
                with open(annotations_file, 'w', encoding='utf-8') as f:
                    json.dump(coco, f, ensure_ascii=False)

        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_records, f, ensure_ascii=False)

        print(f"应用 trace_id 映射: {len(mappings)} 条规则, 影响 {converted_count} 帧")

    def add_trace_id_mapping(self):
        text, ok = QInputDialog.getText(self, "添加ID映射", "输入映射 (如: 100 → 500)")
        if ok and text:
            text = text.strip()
            self.trace_id_list.addItem(f"ID: {text}")
            self._save_trace_id_mappings()
            if self.viewer:
                self.viewer.update_display()

    def remove_selected_trace_id(self):
        """删除选中的trace_id"""
        row = self.trace_id_list.currentRow()
        if row < 0:
            return
        item = self.trace_id_list.item(row)
        # 解析 trace_id: 1000000 (123帧)
        text = item.text()
        try:
            tid = int(text.split(":")[1].split("(")[0].strip())
        except:
            return
        
        labels_dir = Path(TEMP_DATA_MID_DIR) / "labels"
        deleted = 0
        for f in sorted(labels_dir.glob("*.json")):
            try:
                with open(f, encoding='utf-8') as fp:
                    anns = json.load(fp)
                new_anns = [a for a in anns if a.get('track_id', 0) != tid]
                if len(new_anns) != len(anns):
                    with open(f, 'w', encoding='utf-8') as fp:
                        json.dump(new_anns, fp, ensure_ascii=False)
                    deleted += len(anns) - len(new_anns)
            except:
                pass
        
        self.trace_id_list.takeItem(row)
        if self.viewer:
            self.viewer.update_display()
        QMessageBox.information(self, "完成", f"已删除 trace_id={tid}，共 {deleted} 个标注")

    def clear_trace_id_mappings(self):
        self.trace_id_list.clear()
        self._save_trace_id_mappings()
        if self.viewer:
            self.viewer.update_display()

    def on_trace_id_double_clicked(self, item):
        old_text = item.text()
        text, ok = QInputDialog.getText(self, "修改ID映射", "输入新映射", text=old_text.replace("ID: ", ""))
        if ok and text:
            item.setText(f"ID: {text.strip()}")
            self._save_trace_id_mappings()
            if self.viewer:
                self.viewer.update_display()

    def _revert_trace_id_mapping(self):
        temp_mid = Path(TEMP_DATA_MID_DIR)
        labels_dir = temp_mid / "labels"
        annotations_file = temp_mid / "annotations.json"
        history_file = temp_mid / "trace_id_history.json"

        if not history_file.exists():
            print("没有可撤回的映射记录")
            return

        with open(history_file, encoding='utf-8') as f:
                    history_records = json.load(f)

        if not history_records:
            print("没有可撤回的映射记录")
            return

        converted_count = 0
        for label_file in sorted(labels_dir.glob("frame_*.json")):
            with open(label_file, encoding='utf-8') as f:
                frame_anns = json.load(f)
            changed = False
            for ann in frame_anns:
                trace_list = ann.get('trace_id_list', [])
                if len(trace_list) >= 2 and trace_list[-1] == history_records[-1]['new_id']:
                    ann['track_id'] = trace_list[-2]
                    ann['trace_id_list'] = trace_list[:-1]
                    changed = True
                elif len(trace_list) == 1 and trace_list[0] == history_records[-1]['new_id']:
                    ann['track_id'] = history_records[-1]['old_id']
                    ann['trace_id_list'] = [history_records[-1]['old_id']]
                    changed = True
            if changed:
                with open(label_file, 'w', encoding='utf-8') as f:
                    json.dump(frame_anns, f, ensure_ascii=False)
                converted_count += 1

        if annotations_file.exists():
            with open(annotations_file, encoding='utf-8') as f:
                coco = json.load(f)
            changed = False
            for ann in coco.get('annotations', []):
                trace_list = ann.get('trace_id_list', [])
                if len(trace_list) >= 2 and trace_list[-1] == history_records[-1]['new_id']:
                    ann['track_id'] = trace_list[-2]
                    ann['trace_id_list'] = trace_list[:-1]
                    changed = True
                elif len(trace_list) == 1 and trace_list[0] == history_records[-1]['new_id']:
                    ann['track_id'] = history_records[-1]['old_id']
                    ann['trace_id_list'] = [history_records[-1]['old_id']]
                    changed = True
            if changed:
                with open(annotations_file, 'w', encoding='utf-8') as f:
                    json.dump(coco, f, ensure_ascii=False)

        history_file.unlink()
        self.trace_id_list.clear()
        self._load_trace_id_mappings()
        print(f"已撤回 trace_id 映射: {len(history_records)} 条规则, 影响 {converted_count} 帧")

    def on_trace_id_input_changed(self, text):
        try:
            self.ctrl.next_track_id = int(text)
        except ValueError:
            pass

    def decrement_trace_id(self):
        if self.ctrl.next_track_id > 0:
            self.ctrl.next_track_id -= 1
            self.trace_id_input.setText(str(self.ctrl.next_track_id))

    def increment_trace_id(self):
        self.ctrl.next_track_id += 1
        self.trace_id_input.setText(str(self.ctrl.next_track_id))

    def modify_selected_trace_mapping(self):
        item = self.trace_id_list.currentItem()
        if item:
            self.on_trace_id_double_clicked(item)

    def handle_viewer_click(self, video_x, video_y, frame_idx):
        if self.ctrl.fence_mode_active():
            for i, fence in enumerate(self.ctrl.fences):
                if fence.get('mode', False):
                    self.ctrl.add_fence_point(i, (video_x, video_y))
                    break
            if self.viewer:
                self.viewer.update_display()
            return

        if not self.viewer:
            return

        if self.is_playing:
            self.is_playing = False
            self.play_timer.stop()
            self.next_btn.setText("正帧")
            self.forward_fast_btn.setText("正播")
            if self.viewer:
                self.viewer.stop_playback()

        _, annotations = self.viewer.load_frame_data(frame_idx)
        filtered = self.ctrl.filter_annotations(annotations)

        all_found = []
        for ann in filtered:
            polygon = ann.get('segmentation')
            if not polygon:
                continue
            pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
            if cv2.pointPolygonTest(pts, (float(video_x), float(video_y)), False) >= 0:
                all_found.append(ann)

        if not all_found:
            return

        chosen = all_found[0]
        if len(all_found) > 1:
            items = [f"track_id={ann.get('track_id', ann.get('id', 0))}" for ann in all_found]
            item, ok = QInputDialog.getItem(self, "选择标注", "多个标注重叠，请选择一个", items, 0, False)
            if not ok:
                return
            idx = items.index(item)
            chosen = all_found[idx]

        old_id = chosen.get('track_id', 0)
        new_id = self.ctrl.next_track_id

        self._apply_single_mapping_to_mid(old_id, new_id)

        self.trace_id_list.addItem(f"ID: {old_id} → {new_id}")
        self._save_trace_id_mappings()
        self.viewer.current_frame_idx = frame_idx
        self.viewer.update_display()

    def select_data_dir(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件(支持多选)", ".",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV);;所有文件 (*)"
        )
        if paths:
            self._extract_video_to_temp_data([Path(p) for p in paths])

    def _extract_video_to_temp_data(self, video_paths=None):
        if video_paths is None:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "选择视频文件(支持多选)", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV);;所有文件 (*)"
            )
            if not paths:
                return
            video_paths = [Path(p) for p in paths]
        
        # 如果是单个路径，转为列表
        if isinstance(video_paths, Path):
            video_paths = [video_paths]

        # 选择视频时删除temp_data和temp_data_mid
        import shutil
        temp_data = Path("temp_data")
        temp_data_mid = Path("temp_data_mid")
        if temp_data.exists():
            shutil.rmtree(temp_data)
        if temp_data_mid.exists():
            shutil.rmtree(temp_data_mid)

        temp_dir = self.temp_data_path = Path(self.path_input.text() or "temp_data")
        frames_dir = temp_dir / "frames"
        labels_dir = temp_dir / "labels"

        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        self.statusBar().showMessage("正在切帧，请稍候...")
        QApplication.processEvents()

        try:
            # 获取参数
            start_time = float(self.start_time_input.text()) if self.start_time_input.text() else 0
            max_frames = int(self.max_frames_input.text()) if self.max_frames_input.text() else 1000
            skip_frames = int(self.skip_frames_input.text()) if self.skip_frames_input.text() else 1
            resize_ratio = float(self.resize_ratio_input.text()) if self.resize_ratio_input.text() else 1.0
            
            # 处理多个视频：合并成一个大视频
            all_frames = []
            all_fps = None
            
            for i, video_path in enumerate(video_paths):
                print(f"[DEBUG] 处理视频 {i+1}/{len(video_paths)}: {video_path}")
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"[ERROR] 无法打开视频: {video_path}")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                if all_fps is None:
                    all_fps = fps
                
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if resize_ratio != 1.0:
                        frame = cv2.resize(frame, (int(frame.shape[1] * resize_ratio), int(frame.shape[0] * resize_ratio)))
                    all_frames.append(frame.copy())
                    frame_idx += 1
                    if frame_idx >= max_frames:
                        break
                cap.release()
                
                # 如果设置了跳过帧，每隔skip_frames取1帧
                if skip_frames > 1:
                    all_frames = all_frames[::skip_frames]
                
                # 更新max_frames控制每个视频的帧数
                if max_frames < 10000:
                    max_frames = 10000  # 下一个视频不受限制
            
            if not all_frames:
                QMessageBox.warning(self, "错误", "无法读取任何视频帧")
                return
            
            print(f"[DEBUG] 合并后总帧数: {len(all_frames)}, FPS: {all_fps}")
            
            # 保存帧到文件
            frame_count = len(all_frames)
            for count, frame in enumerate(all_frames):
                cv2.imwrite(str(frames_dir / f"frame_{count:06d}.jpg"), frame)
                if count % 100 == 0:
                    self.statusBar().showMessage(f"正在保存帧... {count} / {frame_count}")
                    QApplication.processEvents()

            # 获取第一帧的尺寸
            height, width = all_frames[0].shape[:2]
            
            # 获取第一个视频的FPS
            first_cap = cv2.VideoCapture(str(video_paths[0]))
            fps = first_cap.get(cv2.CAP_PROP_FPS)
            first_cap.release()

            coco_data = {
                'info': {
                    'description': 'Video Annotation Dataset',
                    'video_path': str(video_paths[0]),
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'videos': [str(vp) for vp in video_paths],
                    'FIND': []
                },
                'images': [
                    {'id': i, 'file_name': f"frame_{i:06d}.jpg", 'width': width, 'height': height, 'frame_count': i}
                    for i in range(frame_count)
                ],
                'annotations': [],
                'categories': []
            }

            with open(temp_dir / 'annotations.json', 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False)

            for i in range(frame_count):
                with open(labels_dir / f"frame_{i:06d}.json", 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)

            self.total_frames = frame_count
            self.path_input.setText(str(temp_dir))
            self.statusBar().showMessage(f"切帧完成: {frame_count} 帧")
            print(f"[DEBUG] 视频切帧完成: {frame_count} 帧, 保存到 {temp_dir}")
            QMessageBox.information(self, "完成", f"视频切帧完成！\n共 {frame_count} 帧\n保存到: {temp_dir}")

            if self.viewer:
                self.viewer.coco_data = coco_data
                self.viewer.total_frames = frame_count
                self.viewer.go_to_frame(0)
                self.frame_label.setText(f"1/{frame_count}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("切帧失败")
            QMessageBox.critical(self, "错误", f"切帧失败:\n{e}")

    def import_labelme_to_temp_data(self):
        """选择labelme格式文件夹，转换为COCO格式覆盖temp_data"""
        labelme_dir = QFileDialog.getExistingDirectory(self, "选择labelme格式文件夹")
        if not labelme_dir:
            return

        labelme_path = Path(labelme_dir)
        temp_data = Path("temp_data")

        # 收集所有json和图片
        json_files = sorted(labelme_path.glob("*.json"))
        if not json_files:
            QMessageBox.warning(self, "错误", "选择的文件夹中没有.json文件")
            return

        print(f"[labelme import] 找到 {len(json_files)} 个JSON文件")

        # 清空或创建temp_data目录
        if temp_data.exists():
            import shutil
            shutil.rmtree(temp_data)
        temp_data.mkdir(parents=True)
        frames_dir = temp_data / "frames"
        labels_dir = temp_data / "labels"
        frames_dir.mkdir()
        labels_dir.mkdir()

        all_annotations = []
        ann_id = 0
        all_images = []

        for json_file in json_files:
            with open(json_file, encoding='utf-8') as f:
                        data = json.load(f)

            # 获取图片文件名
            image_name = data.get('imagePath', '')
            if not image_name:
                continue

            # 复制图片
            src_img = labelme_path / image_name
            if not src_img.exists():
                print(f"[WARN] 图片不存在: {src_img}")
                continue

            # 从文件名提取帧索引
            try:
                frame_idx = int(image_name.replace('frame_', '').replace('.jpg', ''))
            except:
                print(f"[WARN] 无法解析帧索引: {image_name}")
                continue

            dst_img = frames_dir / f"frame_{frame_idx:06d}.jpg"
            shutil.copy2(src_img, dst_img)

            # 获取图片尺寸
            img = cv2.imread(str(dst_img))
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]

            # 添加images记录
            all_images.append({'id': frame_idx, 'frame_idx': frame_idx})

            # 解析shapes
            shapes = data.get('shapes', [])
            frame_anns = []
            for shape in shapes:
                label = shape.get('label', 'Unknown')
                points = shape.get('points', [])
                shape_type = shape.get('shape_type', '')

                if not points:
                    continue

                if shape_type == 'rectangle' and len(points) == 4:
                    # rectangle格式：4个角点
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    w, h = x2 - x1, y2 - y1
                    bbox = [float(x1), float(y1), float(w), float(h)]
                    # polygon用4个角点
                    polygon = []
                    for p in points:
                        polygon.extend([p[0], p[1]])
                elif len(points) >= 3:
                    # polygon格式
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    w, h = x2 - x1, y2 - y1
                    bbox = [float(x1), float(y1), float(w), float(h)]
                    polygon = []
                    for p in points:
                        polygon.extend([p[0], p[1]])
                else:
                    continue

                area = float(w * h)
                ann = {
                    'id': ann_id,
                    'track_id': 1000,  # 默认track_id
                    'image_id': frame_idx,
                    'category_id': 0,
                    'bbox': bbox,
                    'area': area,
                    'segmentation': [polygon] if polygon else [],
                    'iscrowd': 0,
                    'confidence': 1.0,
                    'category': label
                }
                frame_anns.append(ann)
                all_annotations.append(ann)
                ann_id += 1

            # 保存帧标注
            with open(labels_dir / f"frame_{frame_idx:06d}.json", 'w', encoding='utf-8') as f:
                json.dump(frame_anns, f, ensure_ascii=False)

        # 保存annotations.json
        coco_data = {
            'info': {'description': 'Imported from labelme', 'fps': 30, 'width': orig_w, 'height': orig_h},
            'images': all_images,
            'annotations': all_annotations,
            'categories': [{'id': 0, 'name': 'Detect'}]
        }
        with open(temp_data / "annotations.json", 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False)

        print(f"[labelme import] 完成: {len(all_images)} 帧, {len(all_annotations)} 标注")
        self.path_input.setText("temp_data")
        QMessageBox.information(self, "完成", f"已导入labelme格式到temp_data\n\n{len(all_images)} 帧, {len(all_annotations)} 标注")

    def show_viewer(self):
        from video_viewer import VideoViewer
        # 直接使用temp_data_mid作为预览路径
        temp_mid = Path(TEMP_DATA_MID_DIR)
        if not temp_mid.exists():
            QMessageBox.warning(self, "错误", "temp_data_mid 目录不存在")
            return
        ann_file = temp_mid / "annotations.json"
        if not ann_file.exists():
            QMessageBox.warning(self, "错误", "temp_data_mid/annotations.json 不存在")
            return

        try:
            with open(ann_file, encoding='utf-8') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError:
            QMessageBox.critical(self, "错误", f"annotations.json 文件损坏!")
            return
        self.total_frames = len(coco_data.get('images', []))
        self.temp_data_path = temp_mid
        viewer_path = str(temp_mid)

        self._load_trace_id_mappings()
        self._apply_trace_id_mappings_to_mid()

        self.viewer = VideoViewer(viewer_path, controller=self.ctrl, panel=self)
        self.viewer.video_clicked.connect(self.handle_viewer_click)
        zoom_factor = self.zoom_slider.value() / 100.0
        self.viewer.set_zoom(zoom_factor)
        geo = self.geometry()
        self.viewer.move(geo.right(), geo.top())
        self.viewer.show()
        self.viewer.update_display()

        self.frame_label.setText(f"1/{self.total_frames}")

    def select_save_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入目录", ".")
        if folder:
            self.save_input_dir.setText(folder)

    def _get_category_for_track_id(self, track_id):
        # track_id - 1000000 直接作为类别列表索引
        if track_id >= 1000000:
            idx = track_id - 1000000
            if idx < len(self.category_inputs):
                name = self.category_inputs[idx].text() or "Detect"
                return (idx, name)
        return (0, self.ctrl.category_name)
    
    def redo_copy(self):
        """从选择的文件夹复制覆盖temp_data和temp_data_mid"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", ".")
        if not folder:
            return
        src = Path(folder)
        if not src.exists():
            QMessageBox.warning(self, "错误", "选择的文件夹不存在")
            return
        
        # 如果选择的是temp_data，只覆盖temp_data_mid
        is_temp_data = src.resolve() == Path("temp_data").resolve()
        
        # 复制到temp_data_mid
        dst = Path(TEMP_DATA_MID_DIR)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        
        # 如果不是temp_data，同时复制到temp_data
        if not is_temp_data:
            dst_temp = Path("temp_data")
            if dst_temp.exists():
                shutil.rmtree(dst_temp)
            shutil.copytree(src, dst_temp)
            QMessageBox.information(self, "完成", f"已复制 {src.name} 到 temp_data 和 temp_data_mid")
        else:
            QMessageBox.information(self, "完成", f"已复制 {src.name} 到 temp_data_mid")
        
        if self.viewer:
            self.viewer.update_display()
    
    def apply_fixed_bbox(self):
        """在指定帧范围添加固定框"""
        if not self.viewer:
            QMessageBox.warning(self, "错误", "请先显示预览")
            return
        prompt_bboxes = self.viewer.get_prompt_bboxes()
        if not prompt_bboxes:
            QMessageBox.warning(self, "错误", "请先在预览中画框")
            return
        bbox = prompt_bboxes[0]  # 使用第一个框
        trace_id = int(self.trace_id_input.text())
        
        # 获取起始帧和终止帧
        try:
            start_frame = int(self.fixed_start_input.text() or "0")
            end_frame = int(self.fixed_end_input.text() if self.fixed_end_input.text() != "-1" else "-1")
        except ValueError:
            QMessageBox.warning(self, "错误", "帧号必须是整数")
            return
        
        labels_dir = Path(TEMP_DATA_MID_DIR) / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        total_frames = self.total_frames
        if end_frame == -1 or end_frame >= total_frames:
            end_frame = total_frames - 1
        
        print(f"[固定框] 用户输入: {start_frame} - {end_frame}, 总帧数: {total_frames}")
        
        # 转换为coco格式: [x, y, w, h]
        coco_bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), 
                     abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1])]
        bbox_key = f"{int(coco_bbox[0])},{int(coco_bbox[1])},{int(coco_bbox[2])},{int(coco_bbox[3])}"
        
        # 记录回退信息（用户输入是1-indexed，frame文件是0-indexed）
        undo_changes = []
        added = 0
        print(f"[固定框] 实际标注帧范围: {start_frame} 到 {end_frame} (转为0-indexed: {start_frame-1} 到 {end_frame-1})")
        for i in range(start_frame - 1, end_frame):
            frame_file = labels_dir / f"frame_{i:06d}.json"
            print(f"[固定框] 正在标注帧 {i}")
            old_trace_id = None
            if frame_file.exists():
                with open(frame_file, encoding='utf-8') as fp:
                    anns = json.load(fp)
                # 检查是否已有相同的bbox
                for ann in anns:
                    ann_bbox = ann.get('bbox', [])
                    if len(ann_bbox) >= 4 and ann_bbox == coco_bbox:
                        old_trace_id = ann.get('track_id', 0)
                        break
            else:
                anns = []
            
            if old_trace_id is not None:
                # 已有bbox，更新trace_id
                for ann in anns:
                    if ann.get('bbox') == coco_bbox:
                        ann['track_id'] = trace_id
                        undo_changes.append({
                            'frame_idx': i,
                            'bbox_key': bbox_key,
                            'old_trace_id': old_trace_id,
                            'new_trace_id': trace_id
                        })
                        break
            else:
                # 新增bbox
                undo_changes.append({
                    'frame_idx': i,
                    'bbox_key': bbox_key,
                    'old_trace_id': -1,  # -1表示新增
                    'new_trace_id': trace_id
                })
                anns.append({
                    'bbox': coco_bbox,
                    'track_id': trace_id,
                    'segmentation': [[
                        coco_bbox[0], coco_bbox[1],
                        coco_bbox[0] + coco_bbox[2], coco_bbox[1],
                        coco_bbox[0] + coco_bbox[2], coco_bbox[1] + coco_bbox[3],
                        coco_bbox[0], coco_bbox[1] + coco_bbox[3]
                    ]],
                    'category': 'Fixed',
                    'confidence': 1.0
                })
                added += 1
            
            with open(frame_file, 'w', encoding='utf-8') as fp:
                json.dump(anns, fp, ensure_ascii=False)
        
        # 记录到回退栈
        if undo_changes:
            self.push_undo(undo_changes)
        
        self.viewer.clear_prompt_bboxes()
        self.viewer.enable_bbox_drawing(False)
        self.fixed_bbox_mode = False
        self.fixed_edit_btn.setText("编辑固定框")
        self.viewer.update_display()
        self.refresh_trace_id_list()
        print(f"[固定框] 已添加 {len(undo_changes)} 个框")
    
    def refresh_trace_id_list(self):
        """刷新Trace ID列表"""
        self._load_trace_id_mappings()
    
    def export_to_temp_data_post(self):
        data_dir = Path(TEMP_DATA_MID_DIR)
        if not data_dir.exists():
            QMessageBox.warning(self, "错误", "temp_data_mid 目录不存在，请先点击显示")
            return

        annotations_file = data_dir / "annotations.json"
        if not annotations_file.exists():
            QMessageBox.warning(self, "错误", "temp_data_mid/annotations.json 不存在")
            return

        with open(annotations_file, encoding='utf-8') as f:
                coco_data = json.load(f)

        video_info = coco_data.get('info', {})
        total_frames = len(coco_data.get('images', []))
        if total_frames == 0:
            QMessageBox.warning(self, "错误", "没有帧数据")
            return
        
        # 读取帧数限制
        frame_limit_str = self.export_frame_limit.text().strip()
        frame_limit = -1
        if frame_limit_str and frame_limit_str != "-1":
            try:
                frame_limit = int(frame_limit_str)
                if frame_limit <= 0:
                    frame_limit = -1
                elif frame_limit > total_frames:
                    frame_limit = total_frames
            except:
                frame_limit = -1
        export_frames = total_frames if frame_limit == -1 else frame_limit
        print(f"[导出] 总帧数: {total_frames}, 导出帧数: {export_frames}")

        labels_dir = data_dir / "labels"
        frames_dir = data_dir / "frames"
        
        # 检查帧是否连续，如果不连续则重新编号
        existing_frames = sorted([int(f.stem.replace('frame_', '')) for f in frames_dir.glob("frame_*.jpg")])
        if not existing_frames:
            QMessageBox.warning(self, "错误", "没有帧文件")
            return
        
        print(f"[导出] 检测到帧: {existing_frames[:5]}...{existing_frames[-5:] if len(existing_frames) > 5 else ''} (共{len(existing_frames)}个)")
        
        # 检查是否连续
        is_sequential = all(existing_frames[i] == existing_frames[i-1] + 1 for i in range(1, len(existing_frames)))
        
        if not is_sequential:
            print(f"[导出] 检测到不连续的帧，正在重新编号...")
            # 重新编号：按顺序重命名
            new_idx = 0
            for old_idx in existing_frames:
                old_frame = frames_dir / f"frame_{old_idx:06d}.jpg"
                old_label = labels_dir / f"frame_{old_idx:06d}.json"
                new_frame = frames_dir / f"frame_{new_idx:06d}.jpg"
                new_label = labels_dir / f"frame_{new_idx:06d}.json"
                if old_frame.exists():
                    old_frame.rename(new_frame)
                if old_label.exists():
                    old_label.rename(new_label)
                new_idx += 1
            
            print(f"[导出] 重新编号完成: {new_idx} 帧")
            existing_frames = list(range(new_idx))
            total_frames = new_idx
            export_frames = min(export_frames, total_frames)
            
            # 更新 annotations.json 中的 images 列表
            coco_data['images'] = [{'id': i, 'frame_idx': i} for i in range(total_frames)]
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump(coco_data, f, ensure_ascii=False)
            print(f"[导出] annotations.json 已更新")

        # 清空temp_data_post
        import shutil
        output_path = Path("temp_data_post")
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        output_labels_dir = output_path / "labels"
        output_labels_dir.mkdir(exist_ok=True)
        output_frames_dir = output_path / "frames"
        output_frames_dir.mkdir(exist_ok=True)

        cat_id_set = set()

        print("步骤1: 保存到 temp_data_post...")
        for i in range(export_frames):
            frame_path = str(frames_dir / f"frame_{i:06d}.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[导出] 警告: 无法读取帧 {i}")
                continue

            output_frame_path = str(output_frames_dir / f"frame_{i:06d}.jpg")
            cv2.imwrite(output_frame_path, frame)

            label_path = labels_dir / f"frame_{i:06d}.json"
            output_label_path = output_labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path, encoding='utf-8') as f:
                        annotations = json.load(f)

                filtered = self.ctrl.filter_annotations(annotations)
                track_id_filtered = [ann for ann in filtered if ann.get('track_id', 0) > 999998]

                # 根据bbox去重（完全相同的bbox只保留一个）
                seen_bboxes = set()
                frame_anns = []
                for ann in track_id_filtered:
                    bbox = tuple(ann.get('bbox', []))
                    if bbox in seen_bboxes:
                        continue
                    seen_bboxes.add(bbox)
                    tid = ann.get('track_id', 0)
                    cat_id, cat_name = self._get_category_for_track_id(tid)
                    cat_id_set.add((cat_id, cat_name))
                    ann_copy = ann.copy()
                    ann_copy['category_id'] = cat_id
                    ann_copy['category'] = cat_name
                    frame_anns.append(ann_copy)

                with open(output_label_path, 'w', encoding='utf-8') as f:
                    json.dump(frame_anns, f, ensure_ascii=False)

        all_annotations = []
        all_seen_ids = set()  # 用于全局去重: (image_id, track_id)
        for i in range(export_frames):
            label_path = labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path, encoding='utf-8') as f:
                        annotations = json.load(f)
                filtered = self.ctrl.filter_annotations(annotations)
                for ann in filtered:
                    tid = ann.get('track_id', 0)
                    if tid > 999998:
                        unique_key = (i, tid)
                        if unique_key in all_seen_ids:
                            continue
                        all_seen_ids.add(unique_key)
                        cat_id, cat_name = self._get_category_for_track_id(tid)
                        cat_id_set.add((cat_id, cat_name))
                        ann_copy = ann.copy()
                        ann_copy['category_id'] = cat_id
                        ann_copy['category'] = cat_name
                        all_annotations.append(ann_copy)
        categories_list = [{'id': cid, 'name': cname} for cid, cname in sorted(cat_id_set, key=lambda x: x[0])]
        coco_output = {
            'info': video_info,
            'images': [{'id': i, 'frame_idx': i} for i in range(export_frames)],
            'annotations': all_annotations,
            'categories': categories_list
        }

        with open(output_path / "annotations.json", 'w', encoding='utf-8') as f:
            json.dump(coco_output, f, ensure_ascii=False)

        print(f"导出完成: {output_path}")
        QMessageBox.information(self, "完成", f"数据已保存到 {output_path}\n\n帧数: {export_frames}")

    def _export_to_labelme(self, input_dir):
        """将temp_data_post转换为labelme格式"""
        import shutil
        input_path = Path(input_dir)
        output_path = Path("label_x_label_me")

        # 获取title和digit参数
        title = self.labelme_title.text() or "frame"
        digit = int(self.labelme_digit.text()) if self.labelme_digit.text() else 6
        frame_pattern = title + "_{:0" + str(digit) + "d}"

        # 如果已有label_x_label_me，先删掉
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)

        frames_dir = input_path / "frames"
        labels_dir = input_path / "labels"

        # 获取所有帧
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        print(f"[labelme] 正在转换 {len(frame_files)} 帧到labelme格式... (title={title}, digit={digit})")

        converted_count = 0
        for frame_file in frame_files:
            frame_idx = int(frame_file.stem.split('_')[1])
            label_file = labels_dir / f"frame_{frame_idx:06d}.json"

            # 读取图片（不resize）
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]

            # 保存图片（使用配置的命名格式）
            new_filename = f"{frame_pattern.format(frame_idx)}.jpg"
            img_path = output_path / new_filename
            cv2.imwrite(str(img_path), frame)

            # 生成labelme JSON
            shapes = []
            if label_file.exists():
                with open(label_file, encoding='utf-8') as f:
                    annotations = json.load(f)

                # IoU去重：根据traceId和IoU过滤
                iou_threshold = float(self.merge_iou_input.text()) if self.merge_iou_input.text() else 0.5
                filtered_anns = []
                
                for i, ann in enumerate(annotations):
                    bbox = ann.get('bbox', [])
                    if not bbox:
                        continue
                    
                    track_id = ann.get('track_id', 0)
                    should_keep = True
                    
                    for j, existing in enumerate(filtered_anns):
                        existing_bbox = existing.get('bbox', [])
                        if not existing_bbox:
                            continue
                        
                        # 计算IoU
                        x1 = max(bbox[0], existing_bbox[0])
                        y1 = max(bbox[1], existing_bbox[1])
                        x2 = min(bbox[0] + bbox[2], existing_bbox[0] + existing_bbox[2])
                        y2 = min(bbox[1] + bbox[3], existing_bbox[1] + existing_bbox[3])
                        inter = max(0, x2 - x1) * max(0, y2 - y1)
                        area1 = bbox[2] * bbox[3]
                        area2 = existing_bbox[2] * existing_bbox[3]
                        union = area1 + area2 - inter
                        iou = inter / union if union > 0 else 0
                        
                        if iou > iou_threshold:
                            # IoU超过阈值，保留traceId大的
                            existing_tid = existing.get('track_id', 0)
                            if track_id > existing_tid:
                                # 用当前替换已有的
                                filtered_anns[j] = ann
                                should_keep = False
                                break
                            else:
                                # 丢弃当前的
                                should_keep = False
                                break
                    
                    if should_keep:
                        filtered_anns.append(ann)
                
                for ann in filtered_anns:
                    bbox = ann.get('bbox', [])
                    if not bbox:
                        continue

                    label = ann.get('category', 'Unknown')

                    # 将bbox转换为四个点（不scale）
                    x1, y1, w, h = bbox
                    x2 = x1 + w
                    y2 = y1 + h

                    shapes.append({
                        "label": label,
                        "score": None,
                        "points": [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ],
                        "group_id": None,
                        "description": "",
                        "difficult": False,
                        "shape_type": "rectangle",
                        "flags": {},
                        "attributes": {},
                        "kie_linking": []
                    })

            labelme_data = {
                "version": "4.0.0-beta.5",
                "flags": {},
                "checked": False,
                "shapes": shapes,
                "imagePath": new_filename,
                "imageData": None,
                "imageHeight": orig_h,
                "imageWidth": orig_w,
                "description": ""
            }

            # 保存JSON（使用配置的命名格式）
            json_filename = f"{frame_pattern.format(frame_idx)}.json"
            json_path = output_path / json_filename
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)

            converted_count += 1

        print(f"[labelme] 转换完成: {converted_count} 张图片")

    def run_save(self):
        input_dir = self.save_input_dir.text() or "temp_data_post"
        output_name = self.save_output_name.text() or "1dst.mp4"

        output_path = Path("1dst") / output_name
        output_path.parent.mkdir(exist_ok=True)

        input_path = Path(input_dir)
        annotations_path = input_path / "annotations.json"

        if not annotations_path.exists():
            QMessageBox.warning(self, "错误", f"找不到 {annotations_path}")
            return

        with open(annotations_path, encoding='utf-8') as f:
            coco_data = json.load(f)

        video_info = coco_data.get('info', {})
        total_frames = len(coco_data.get('images', []))

        if total_frames == 0:
            QMessageBox.warning(self, "错误", "没有找到帧数据")
            return

        width = int(video_info.get('width', 1280))
        height = int(video_info.get('height', 720))
        fps = int(video_info.get('fps', 30))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        labels_dir = input_path / "labels"
        frames_dir = input_path / "frames"

        # 粒子效果初始化
        enable_particle = self.trail_check.isChecked()
        enable_latex = self.latex_check.isChecked()
        enable_trail_line = self.trail_line_check.isChecked()
        fade_duration_ms = float(self.trail_duration.text()) if self.trail_duration.text() else 500
        fade_frames = int(fade_duration_ms / 1000 * fps)  # 消散帧数
        particle_history = []  # 保存历史帧的重叠粒子位置
        trail_history = {}  # 保存历史轨迹 track_id -> [(frame_idx, x, y), ...]

        print(f"正在生成视频: {output_path}")
        print(f"[DEBUG run_save] 粒子效果: {'开启' if enable_particle else '关闭'}, 白色乳胶漆: {'开启' if enable_latex else '关闭'}, 轨迹: {'开启' if enable_trail_line else '关闭'}, 消散时间: {fade_duration_ms}ms ({fade_frames}帧)")

        written = 0
        for i in range(total_frames):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            # 跳过不存在的帧（被删除的帧）
            if not frame_path.exists():
                continue
            frame = cv2.imread(str(frame_path))

            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            label_path = labels_dir / f"frame_{i:06d}.json"
            annotations = []
            if label_path.exists():
                with open(label_path, encoding='utf-8') as f:
                        annotations = json.load(f)

                result_frame = frame.copy()
                overlay = frame.copy()

                current_track_positions = {}  # 当前帧的track_id -> (cx, cy, color)

                for ann in annotations:
                    polygon = ann.get('segmentation')
                    bbox = ann.get('bbox')

                    if not bbox:
                        continue

                    cat_name = ann.get('category', 'Unknown')
                    conf = ann.get('confidence', 1.0)
                    track_id = ann.get('track_id', 0)
                    if track_id == 1000:
                        color = self.palette_colors[self.selected_color_index]
                    else:
                        n_colors = len(self.palette_colors) - 1
                        color_idx_in_remapped = track_id % n_colors
                        selected_idx = self.selected_color_index
                        if color_idx_in_remapped < selected_idx:
                            color = self.palette_colors[color_idx_in_remapped]
                        else:
                            color = self.palette_colors[color_idx_in_remapped + 1]

                    if polygon and not self.render_segment_check.isChecked():
                        pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)

                    x, y = int(bbox[0]), int(bbox[1])
                    w, h = int(bbox[2]), int(bbox[3])
                    cx, cy = x + w // 2, y + h // 2  # 中心点

                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(overlay, f"{cat_name} {conf:.2f}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    current_track_positions[track_id] = (cx, cy, color)

                # 收集所有contours和bbox
                all_contours = []
                all_bboxes = []
                all_track_ids = []
                for ann in annotations:
                    polygon = ann.get('segmentation')
                    bbox = ann.get('bbox', [])
                    track_id = ann.get('track_id', 0)
                    if polygon and len(polygon[0]) >= 6 and bbox:
                        try:
                            pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                            all_contours.append(pts)
                            all_bboxes.append(bbox)
                            all_track_ids.append(track_id)
                        except:
                            pass
                
                # 检测接触面
                track_ids_with_particles = set()
                if len(all_contours) >= 2:
                    for a_idx in range(len(all_contours)):
                        for b_idx in range(a_idx + 1, len(all_contours)):
                            mask_a = np.zeros((height, width), dtype=np.uint8)
                            mask_b = np.zeros((height, width), dtype=np.uint8)
                            cv2.fillPoly(mask_a, [all_contours[a_idx]], 255)
                            cv2.fillPoly(mask_b, [all_contours[b_idx]], 255)
                            intersection = cv2.bitwise_and(mask_a, mask_b)
                            if cv2.countNonZero(intersection) > 0:
                                m_a = cv2.moments(all_contours[a_idx])
                                m_b = cv2.moments(all_contours[b_idx])
                                if m_a["m00"] > 0 and m_b["m00"] > 0:
                                    if m_a["m01"] / m_a["m00"] > m_b["m01"] / m_b["m00"]:
                                        track_ids_with_particles.add(all_track_ids[a_idx])
                                    else:
                                        track_ids_with_particles.add(all_track_ids[b_idx])

                # 粒子效果
                if enable_particle and track_ids_with_particles:
                    # 随机颜色粒子，闪烁效果用帧索引作为种子
                    np.random.seed(i)  # 固定随机种子保证可复现
                    def random_color():
                        return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                    
                    # 闪烁：每3帧闪烁一次
                    show_particles = (i % 3 == 0)
                    
                    # 记录有粒子的track_id
                    track_ids_with_particles = set()
                    
                    # 收集所有多边形和bbox
                    all_contours = []
                    all_bboxes = []
                    all_track_ids = []
                    for ann in annotations:
                        polygon = ann.get('segmentation')
                        bbox = ann.get('bbox', [])
                        track_id = ann.get('track_id', 0)
                        if polygon and len(polygon[0]) >= 6 and bbox:
                            try:
                                pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                                all_contours.append(pts)
                                all_bboxes.append(bbox)
                                all_track_ids.append(track_id)
                            except:
                                pass
                    
                    # 当前帧粒子: (x, y, track_id, offset_x, offset_y)
                    # offset是粒子相对于生成时bbox左上角的偏移量
                    current_particles = []
                    
                    # 建立track_id到bbox的映射
                    tid_to_bbox = {tid: bbox for tid, bbox in zip(all_track_ids, all_bboxes)}
                    
                    # 检测接触面 - 在segmentation重叠处生成粒子，根据当前帧判断哪个在下方
                    if len(all_contours) >= 2:
                        for a_idx in range(len(all_contours)):
                            for b_idx in range(a_idx + 1, len(all_contours)):
                                pts_a, bbox_a, tid_a = all_contours[a_idx], all_bboxes[a_idx], all_track_ids[a_idx]
                                pts_b, bbox_b, tid_b = all_contours[b_idx], all_bboxes[b_idx], all_track_ids[b_idx]
                                
                                # 创建mask用于计算segmentation交集
                                mask_a = np.zeros((height, width), dtype=np.uint8)
                                mask_b = np.zeros((height, width), dtype=np.uint8)
                                cv2.fillPoly(mask_a, [pts_a], 255)
                                cv2.fillPoly(mask_b, [pts_b], 255)
                                
                                # segmentation交集
                                intersection_mask = cv2.bitwise_and(mask_a, mask_b)
                                
                                if cv2.countNonZero(intersection_mask) > 0:
                                    # 找到交集区域的轮廓
                                    intersection_contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    
                                    # 根据当前帧判断哪个在下方
                                    m_a = cv2.moments(pts_a)
                                    m_b = cv2.moments(pts_b)
                                    if m_a["m00"] > 0 and m_b["m00"] > 0:
                                        cy_a = m_a["m01"] / m_a["m00"]
                                        cy_b = m_b["m01"] / m_b["m00"]
                                        # y值大的在下方
                                        if cy_a > cy_b:
                                            bottom_idx, bottom_bbox, bottom_tid = a_idx, bbox_a, tid_a
                                        else:
                                            bottom_idx, bottom_bbox, bottom_tid = b_idx, bbox_b, tid_b
                                    else:
                                        bottom_idx, bottom_bbox, bottom_tid = a_idx, bbox_a, tid_a
                                    
                                    # 在当前判断的下方bbox内生成粒子
                                    bx, by, bw, bh = bottom_bbox
                                    bx1, by1, bx2, by2 = int(bx), int(by), int(bx + bw), int(by + bh)
                                    
                                    # 在交集区域轮廓内生成粒子
                                    for cnt in intersection_contours:
                                        for pt in cnt:
                                            px, py = int(pt[0][0]), int(pt[0][1])
                                            # 确保在下方bbox内
                                            if bx1 <= px <= bx2 and by1 <= py <= by2:
                                                particle_x = np.random.randint(max(bx1, px - 2), min(bx2, px + 3))
                                                particle_y = np.random.randint(max(by1, py - 2), min(by2, py + 3))
                                                # 计算相对偏移量
                                                offset_x = particle_x - bx1
                                                offset_y = particle_y - by1
                                                # 保存粒子：位置、track_id、偏移量、颜色、大小(1或2px)
                                                particle_size = 1 if np.random.randint(0, 2) == 0 else 2
                                                current_particles.append((particle_x, particle_y, bottom_tid, offset_x, offset_y, random_color(), particle_size))
                                                track_ids_with_particles.add(bottom_tid)
                    
                    # 添加当前帧粒子到历史
                    particle_history.append(current_particles)
                    
                    # 裁剪历史
                    if len(particle_history) > fade_frames:
                        particle_history = particle_history[-fade_frames:]

                cv2.addWeighted(overlay, self.ctrl.alpha, result_frame, 1 - self.ctrl.alpha, 0, result_frame)
                
                # 轨迹效果在addWeighted之后立即绘制（不被覆盖）
                if enable_trail_line:
                    # 只给最小的track_id用
                    min_tid = float('inf')
                    min_tid_color = None
                    for ann in annotations:
                        tid = ann.get('track_id', 0)
                        if tid < min_tid:
                            min_tid = tid
                            # 获取颜色
                            if tid == 1000:
                                min_tid_color = self.palette_colors[self.selected_color_index]
                            else:
                                n_colors = len(self.palette_colors) - 1
                                color_idx = tid % n_colors
                                selected_idx = self.selected_color_index
                                if color_idx < selected_idx:
                                    min_tid_color = self.palette_colors[color_idx]
                                else:
                                    min_tid_color = self.palette_colors[color_idx + 1]
                    # 清除其他tid的历史，只保留当前最小tid
                    for tid in list(trail_history.keys()):
                        if tid != min_tid:
                            del trail_history[tid]
                    # 记录和绘制轨迹
                    for ann in annotations:
                        bbox = ann.get('bbox', [])
                        track_id = ann.get('track_id', 0)
                        if bbox and track_id == min_tid:
                            cx = int(bbox[0] + bbox[2] / 2)
                            cy = int(bbox[1] + bbox[3] / 2)
                            if track_id not in trail_history:
                                trail_history[track_id] = []
                            trail_history[track_id].append((cx, cy))
                            if len(trail_history[track_id]) > fade_frames:
                                trail_history[track_id] = trail_history[track_id][-fade_frames:]
                    # 绘制轨迹线条
                    if min_tid in trail_history and len(trail_history[min_tid]) >= 2:
                        positions = trail_history[min_tid]
                        color = min_tid_color
                        for j in range(len(positions) - 1):
                            thickness = 2
                            cv2.line(result_frame, positions[j], positions[j+1], color, thickness)
                
                # 效果都在addWeighted之后绘制
                if enable_particle or enable_latex:
                    # 收集contours
                    all_contours = []
                    all_bboxes = []
                    all_track_ids = []
                    for ann in annotations:
                        polygon = ann.get('segmentation')
                        bbox = ann.get('bbox', [])
                        track_id = ann.get('track_id', 0)
                        if polygon and len(polygon[0]) >= 6 and bbox:
                            try:
                                pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                                all_contours.append(pts)
                                all_bboxes.append(bbox)
                                all_track_ids.append(track_id)
                            except:
                                pass
                    
                    # 检测segmentation交集
                    track_ids_with_particles = set()
                    tid_to_bbox = {}
                    if len(all_contours) >= 2:
                        for a_idx in range(len(all_contours)):
                            for b_idx in range(a_idx + 1, len(all_contours)):
                                mask_a = np.zeros((height, width), dtype=np.uint8)
                                mask_b = np.zeros((height, width), dtype=np.uint8)
                                cv2.fillPoly(mask_a, [all_contours[a_idx]], 255)
                                cv2.fillPoly(mask_b, [all_contours[b_idx]], 255)
                                intersection = cv2.bitwise_and(mask_a, mask_b)
                                if cv2.countNonZero(intersection) > 0:
                                    m_a = cv2.moments(all_contours[a_idx])
                                    m_b = cv2.moments(all_contours[b_idx])
                                    if m_a["m00"] > 0 and m_b["m00"] > 0:
                                        if m_a["m01"] / m_a["m00"] > m_b["m01"] / m_b["m00"]:
                                            track_ids_with_particles.add(all_track_ids[a_idx])
                                            tid_to_bbox[all_track_ids[a_idx]] = all_bboxes[a_idx]
                                        else:
                                            track_ids_with_particles.add(all_track_ids[b_idx])
                                            tid_to_bbox[all_track_ids[b_idx]] = all_bboxes[b_idx]
                    
                    # 粒子效果
                    if enable_particle and track_ids_with_particles:
                        current_particles = []
                        for a_idx in range(len(all_contours)):
                            for b_idx in range(a_idx + 1, len(all_contours)):
                                mask_a = np.zeros((height, width), dtype=np.uint8)
                                mask_b = np.zeros((height, width), dtype=np.uint8)
                                cv2.fillPoly(mask_a, [all_contours[a_idx]], 255)
                                cv2.fillPoly(mask_b, [all_contours[b_idx]], 255)
                                intersection = cv2.bitwise_and(mask_a, mask_b)
                                if cv2.countNonZero(intersection) > 0:
                                    # 找下方
                                    m_a = cv2.moments(all_contours[a_idx])
                                    m_b = cv2.moments(all_contours[b_idx])
                                    if m_a["m00"] > 0 and m_b["m00"] > 0:
                                        if m_a["m01"] / m_a["m00"] > m_b["m01"] / m_b["m00"]:
                                            bottom_tid = all_track_ids[a_idx]
                                            bottom_bbox = all_bboxes[a_idx]
                                        else:
                                            bottom_tid = all_track_ids[b_idx]
                                            bottom_bbox = all_bboxes[b_idx]
                                    else:
                                        bottom_tid = all_track_ids[a_idx]
                                        bottom_bbox = all_bboxes[a_idx]
                                    
                                    # 在交集处画粒子
                                    intersection_binary = intersection > 0
                                    ys, xs = np.where(intersection_binary)
                                    for x, y in zip(xs[::5], ys[::5]):
                                        # 灰白1px粒子
                                        cv2.circle(result_frame, (int(x), int(y)), 1, (0, 255, 0), -1)
                                        current_particles.append((int(x), int(y), bottom_tid))
                        
                        particle_history.append(current_particles)
                        if len(particle_history) > fade_frames:
                            particle_history = particle_history[-fade_frames:]
                        
                        # 画历史粒子
                        for fo, particles in enumerate(particle_history):
                            alpha = 1.0 - (fo / max(len(particle_history), 1) * 0.7)
                            for p in particles:
                                if len(p) >= 3:
                                    px, py, p_tid = p[0], p[1], p[2]
                                    if p_tid in tid_to_bbox:
                                        bbox = tid_to_bbox[p_tid]
                                        bx, by, bw, bh = [int(v) for v in bbox]
                                        if bx <= px <= bx + bw and by <= py <= by + bh:
                                            gray = int(255 * alpha)
                                            cv2.circle(result_frame, (px, py), 1, (gray, gray, gray), -1)
                    
                    # 白色乳胶漆效果 - 只覆盖下位bbox的segmentation区域
                    if enable_latex and track_ids_with_particles:
                        for a_idx, ann in enumerate(annotations):
                            if all_track_ids[a_idx] in track_ids_with_particles:
                                polygon = ann.get('segmentation')
                                if polygon and len(polygon[0]) >= 6:
                                    try:
                                        pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                                        # 创建白色mask
                                        mask_white = np.zeros((height, width), dtype=np.uint8)
                                        cv2.fillPoly(mask_white, [pts], 255)
                                        # 只在segmentation区域填白色
                                        result_frame[mask_white > 0] = 255
                                    except:
                                        pass
                
                frame = result_frame

            out.write(frame)
            written += 1

        out.release()
        print(f"视频已保存: {output_path} ({written}/{total_frames}帧)")

        # 转换为labelme格式
        try:
            self._export_to_labelme(input_path)
        except Exception as e:
            import traceback
            print(f"[ERROR] 导出labelme格式失败: {e}")
            traceback.print_exc()

        # 如果勾选了继续训练，跳过上传视频和labelme压缩包
        if self.train_resume_check.isChecked():
            print("[YOLO] 继续训练模式，跳过视频和labelme上传")
            if self.train_model_check.isChecked():
                print(f"[YOLO] 开始训练模型...")
                try:
                    # 继续训练时重新从temp_data_post导出并增广
                    self._export_to_labelme(Path("temp_data_post"))
                    self._train_yolo_model(Path("label_x_label_me"))
                except Exception as e:
                    print(f"[YOLO] 训练失败: {e}")
                    import traceback
                    traceback.print_exc()
            return

        # 上传标注视频
        # OBS文件名添加时间戳，使用原视频名称
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_name = self.last_video_name or output_name.rsplit('.', 1)[0]
        ext = output_name.split('.')[-1] if '.' in output_name else 'mp4'
        obs_filename = f"{video_name}_{timestamp}.{ext}"
        obs_url = f"http://obs.dimond.top/{obs_filename}"

        print("正在上传到OBS...")
        print(f"[OBS] 上传文件名: {obs_filename}")
        try:
            result = subprocess.run(
                ['curl', '--upload-file', str(output_path), obs_url],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"上传成功! OBS地址: {obs_url}")
            else:
                print(f"上传失败: {result.stderr}")
        except Exception as e:
            print(f"上传失败: {e}")
        
        # 保存原视频到temp文件夹（命名为ID_raw.mp4）
        train_id = self.train_id_input.text() or self.default_model_id
        temp_dataset_dir = Path("temp") / f"{train_id}_dataset"
        temp_dataset_dir.mkdir(parents=True, exist_ok=True)
        raw_video_path = temp_dataset_dir / f"{train_id}_raw.mp4"
        
        # 尝试从coco_data获取原始视频列表，合并保存
        raw_frames = []
        raw_fps = None
        raw_width = None
        raw_height = None
        videos = coco_data.get('info', {}).get('videos', [])
        if videos:
            for vp in videos:
                cap = cv2.VideoCapture(vp)
                if not cap.isOpened():
                    continue
                if raw_fps is None:
                    raw_fps = cap.get(cv2.CAP_PROP_FPS)
                    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    raw_frames.append(frame.copy())
                cap.release()
        
        if raw_frames and raw_fps:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_raw = cv2.VideoWriter(str(raw_video_path), fourcc, raw_fps, (raw_width, raw_height))
            for frame in raw_frames:
                out_raw.write(frame)
            out_raw.release()
            print(f"[保存] 原视频已保存到: {raw_video_path}")
        else:
            print(f"[保存] 无法获取原视频")

        # 复制标注视频到temp文件夹
        annotated_video_path = temp_dataset_dir / f"{train_id}_annotated.mp4"
        shutil.copy2(output_path, annotated_video_path)
        print(f"[保存] 标注视频已保存到: {annotated_video_path}")

        # 复制label_x_label_me到temp文件夹
        import zipfile
        labelme_dir = Path("label_x_label_me")
        if labelme_dir.exists():
            # 保存dataset到temp
            temp_labelme_dir = temp_dataset_dir / "labelme"
            if temp_labelme_dir.exists():
                shutil.rmtree(temp_labelme_dir)
            shutil.copytree(labelme_dir, temp_labelme_dir)
            print(f"[保存] labelme数据已保存到: {temp_labelme_dir}")
            
            # 训练YOLO模型
            if self.train_model_check.isChecked():
                print(f"[YOLO] 开始训练模型...")
                try:
                    self._train_yolo_model(labelme_dir)
                except Exception as e:
                    print(f"[YOLO] 训练失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 复制训练好的model到temp文件夹
            model_output_dir = temp_dataset_dir / "model"
            if Path("runs/detect/yolo_runs/train/weights/best.onnx").exists():
                model_output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path("runs/detect/yolo_runs/train/weights/best.onnx"), model_output_dir / "best.onnx")
                shutil.copy2(Path("runs/detect/yolo_runs/train/weights/model.json"), model_output_dir / "model.json")
                print(f"[保存] 模型已保存到: {model_output_dir}")
            
            print(f"完成!\n标注视频: {obs_url}\n数据保存在: {temp_dataset_dir}")
        else:
            print(f"标注视频已上传!\nOBS地址: {obs_url}")

    def _train_yolo_model(self, labelme_dir):
        """训练YOLO模型"""
        import yaml
        import random
        import shutil
        
        labelme_dir = Path(labelme_dir)
        output_dir = Path("yolo_dataset")
        yolo_project = Path("yolo_runs")
        
        # 清理旧数据
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        # 提取类别名
        class_names = []
        for json_file in labelme_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for shape in data.get('shapes', []):
                        if shape.get('label') not in class_names:
                            class_names.append(shape.get('label'))
            except:
                pass
        
        if not class_names:
            print("[YOLO] 未检测到类别")
            return
        
        print(f"[YOLO] 检测到类别: {class_names}")
        
        # 创建dataset.yaml
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        yaml_content = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
nc: {len(class_names)}
names: {class_names}
"""
        yaml_path = output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        # 获取所有图片文件
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_files.extend(labelme_dir.glob(ext))
        img_files = list(set(img_files))  # 去重
        random.shuffle(img_files)
        
        if not img_files:
            print("[YOLO] 未找到图片文件")
            return
        
        # 统计每个类别的帧数
        class_counts = {c: 0 for c in class_names}
        img_with_class = {c: [] for c in class_names}
        for img_file in img_files:
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    labels = set(shape.get('label') for shape in data.get('shapes', []))
                    for label in labels:
                        if label in class_counts:
                            class_counts[label] += 1
                            img_with_class[label].append(img_file)
                except:
                    pass
        
        # 数据增广：每个类别增广到 >= 原始总帧数 * 0.2
        original_total = len(img_files)  # 原始总帧数
        target_count = int(original_total * 0.2)  # 每个类别最少要有原始总帧数的0.2倍
        print(f"[YOLO] 原始总帧数: {original_total}")
        print(f"[YOLO] 每个类别目标帧数: {target_count} (原始*0.2)")
        
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        
        print(f"[YOLO] 类别帧数统计: {class_counts}")
        print(f"[YOLO] 数据增广目标: 每个类别 >= {target_count} 帧")
        
        # 使用albumentations做专业数据增广
        try:
            import albumentations as A
            transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=10, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
                A.ElasticTransform(alpha=30, sigma=5, p=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            has_albumentations = True
        except ImportError:
            print("[YOLO] 警告: 未安装albumentations，使用简单翻转")
            has_albumentations = False
        
        def augment_with_bbox(img_path, json_path, aug_idx, target_dir):
            """使用albumentations增广，返回新的json数据（使用标准labelme格式）"""
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            h, w = img.shape[:2]
            bboxes = []
            labels = []
            original_shapes = []
            
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                if len(points) >= 4:
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, y_min = min(x_coords), min(y_coords)
                    x_max, y_max = max(x_coords), max(y_coords)
                    bboxes.append([x_min, y_min, x_max, y_max])
                    labels.append(shape.get('label', ''))
                    original_shapes.append(shape)
            
            new_name = f"{Path(img_path).stem}_aug{aug_idx}{Path(img_path).suffix}"
            new_img_path = target_dir / new_name
            
            if has_albumentations and bboxes:
                transformed = transform(image=img, bboxes=bboxes, class_labels=labels)
                cv2.imwrite(str(new_img_path), transformed['image'])
                
                new_shapes = []
                for bbox, label, orig_shape in zip(transformed['bboxes'], transformed['class_labels'], original_shapes):
                    x_min, y_min, x_max, y_max = bbox
                    new_shapes.append({
                        "label": label,
                        "score": orig_shape.get('score'),
                        "points": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                        "group_id": orig_shape.get('group_id'),
                        "description": orig_shape.get('description', ''),
                        "difficult": orig_shape.get('difficult', False),
                        "shape_type": "rectangle",
                        "flags": orig_shape.get('flags', {}),
                        "attributes": orig_shape.get('attributes', {}),
                        "kie_linking": orig_shape.get('kie_linking', [])
                    })
                
                new_data = {
                    "version": "4.0.0-beta.5",
                    "flags": {},
                    "checked": False,
                    "shapes": new_shapes,
                    "imagePath": new_name,
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w,
                    "description": ""
                }
            else:
                # 简单翻转
                flipped = cv2.flip(img, 1)
                cv2.imwrite(str(new_img_path), flipped)
                
                new_shapes = []
                for bbox, label, orig_shape in zip(bboxes, labels, original_shapes):
                    x_min, y_min, x_max, y_max = bbox
                    new_x_min = w - x_max
                    new_x_max = w - x_min
                    new_shapes.append({
                        "label": label,
                        "score": orig_shape.get('score'),
                        "points": [[new_x_min, y_min], [new_x_max, y_min], [new_x_max, y_max], [new_x_min, y_max]],
                        "group_id": orig_shape.get('group_id'),
                        "description": orig_shape.get('description', ''),
                        "difficult": orig_shape.get('difficult', False),
                        "shape_type": "rectangle",
                        "flags": orig_shape.get('flags', {}),
                        "attributes": orig_shape.get('attributes', {}),
                        "kie_linking": orig_shape.get('kie_linking', [])
                    })
                
                new_data = {
                    "version": "4.0.0-beta.5",
                    "flags": {},
                    "checked": False,
                    "shapes": new_shapes,
                    "imagePath": new_name,
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w,
                    "description": ""
                }
            
            return new_data, new_img_path
        
        # 对每个类别增广到 target_count（原始总帧数*0.2）
        aug_idx = 0
        for class_name in class_names:
            current_count = class_counts[class_name]
            if current_count < target_count:
                copies_needed = target_count - current_count
                source_imgs = img_with_class[class_name]
                if source_imgs:
                    print(f"[YOLO] 增广 {class_name}: {current_count} -> {target_count}")
                    for i in range(copies_needed):
                        src = source_imgs[i % len(source_imgs)]
                        json_file = src.with_suffix('.json')
                        if json_file.exists():
                            result = augment_with_bbox(src, json_file, aug_idx, labelme_dir)
                            if result:
                                new_data, new_img_path = result
                                img_files.append(new_img_path)
                                new_json_path = new_img_path.with_suffix('.json')
                                with open(new_json_path, 'w', encoding='utf-8') as f:
                                    json.dump(new_data, f, ensure_ascii=False)
                                aug_idx += 1
        
        random.shuffle(img_files)
        
        # 输出最终增广后的图片数量和分类别统计
        print(f"[YOLO] 增广后总图片数: {len(img_files)}")
        # 分类别统计
        final_counts = {c: 0 for c in class_names}
        for img_file in img_files:
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    labels = set(shape.get('label') for shape in data.get('shapes', []))
                    for label in labels:
                        if label in final_counts:
                            final_counts[label] += 1
                except:
                    pass
        print(f"[YOLO] 分类别帧数: {final_counts}")
        
        # 划分训练集和验证集 (8:2)
        split_idx = int(len(img_files) * 0.8)
        train_files = img_files[:split_idx]
        val_files = img_files[split_idx:]
        
        def convert_to_yolo(json_path, class_names):
            """将labelme JSON转换为YOLO格式"""
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_w = data.get('imageWidth', 640)
            img_h = data.get('imageHeight', 480)
            
            yolo_lines = []
            for shape in data.get('shapes', []):
                label = shape.get('label')
                if label not in class_names:
                    continue
                class_id = class_names.index(label)
                points = shape['points']
                
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                x_center = (x_min + x_max) / 2 / img_w
                y_center = (y_min + y_max) / 2 / img_h
                width = (x_max - x_min) / img_w
                height = (y_max - y_min) / img_h
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            return yolo_lines
        
        # 处理训练集
        for img_file in train_files:
            json_file = img_file.with_suffix('.json')
            shutil.copy(img_file, output_dir / "images" / "train" / img_file.name)
            if json_file.exists():
                yolo_lines = convert_to_yolo(json_file, class_names)
                with open(output_dir / "labels" / "train" / f"{img_file.stem}.txt", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
        
        # 处理验证集
        for img_file in val_files:
            json_file = img_file.with_suffix('.json')
            shutil.copy(img_file, output_dir / "images" / "val" / img_file.name)
            if json_file.exists():
                yolo_lines = convert_to_yolo(json_file, class_names)
                with open(output_dir / "labels" / "val" / f"{img_file.stem}.txt", 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
        
        print(f"[YOLO] 训练集: {len(train_files)}, 验证集: {len(val_files)}")
        
        # 训练输出目录
        yolo_runs_dir = Path("runs/detect/yolo_runs")
        resume = self.train_resume_check.isChecked()
        
        # 如果继续训练，查找编号最大的train文件夹
        if resume:
            # 查找所有train*文件夹
            train_dirs = list(yolo_runs_dir.glob("train*"))
            train_dirs = [d for d in train_dirs if d.is_dir()]
            # 过滤出有weights目录的
            train_dirs = [d for d in train_dirs if (d / "weights").exists()]
            if train_dirs:
                # 按编号排序，train-1-2 > train-1 > train
                def get_train_num(p):
                    name = p.name
                    if name == "train":
                        return 0
                    # 取train-后面的数字，取最后一个作为主编号
                    suffix = name.replace("train", "")
                    nums = [int(x) for x in suffix.split("-") if x.isdigit()]
                    return nums[-1] if nums else 0
                train_dirs.sort(key=get_train_num, reverse=True)
                train_dir = train_dirs[0]
                print(f"[YOLO] 继续训练，使用: {train_dir.name} (编号最大)")
            else:
                print("[YOLO] 未找到可继续训练的模型，请取消勾选继续训练")
                return
        else:
            # 清理旧的train文件夹，统一用train
            for td in yolo_runs_dir.glob("train*"):
                if td.is_dir():
                    shutil.rmtree(td)
            train_dir = yolo_runs_dir / "train"
            print(f"[YOLO] 新训练文件夹: {train_dir.name}")
        
        # 如果继续训练，从已有model.json读取ID、名称、描述
        prev_model_json = train_dir / "model.json"
        if resume and prev_model_json.exists():
            try:
                with open(prev_model_json, encoding='utf-8') as f:
                    prev_info = json.load(f)
                self.train_id_input.setText(prev_info.get('id', ''))
                self.train_desc_input.setText(prev_info.get('description', ''))
                print(f"[YOLO] 从上次的model.json读取: id={prev_info.get('id')}")
            except:
                pass
        
        # 训练模型
        epochs_input = self.train_epochs_input.text() if hasattr(self, 'train_epochs_input') else "30"
        epochs = int(epochs_input) if epochs_input else 30
        print(f"[YOLO] 开始训练... epochs={epochs}")
        from ultralytics import YOLO
        
        if resume:
            # 从已有权重加载作为预训练权重，生成新的train-N文件夹
            print("[YOLO] 从已有权重加载...")
            print(f"[YOLO] epochs={epochs}")
            best_pt = train_dir / "weights" / "best.pt"
            last_pt = train_dir / "weights" / "last.pt"
            best_onnx = train_dir / "weights" / "best.onnx"
            
            if best_pt.exists():
                # 有best.pt，加载作为预训练权重
                model = YOLO(str(best_pt))
                model.train(
                    data=yaml_path.as_posix(),
                    epochs=epochs,
                    imgsz=640,
                    batch=8,
                    device=0,
                    workers=0,
                    project=yolo_project.as_posix(),
                    name=train_dir.name,
                    resume=False
                )
            elif last_pt.exists():
                # 有last.pt，加载作为预训练权重
                model = YOLO(str(last_pt))
                model.train(
                    data=yaml_path.as_posix(),
                    epochs=epochs,
                    imgsz=640,
                    batch=8,
                    device=0,
                    workers=0,
                    project=yolo_project.as_posix(),
                    name=train_dir.name,
                    resume=False
                )
            elif best_onnx.exists():
                # 只有onnx，从onnx加载
                print("[YOLO] 只有best.onnx，从ONNX加载...")
                model = YOLO(str(best_onnx))
                model.train(
                    data=yaml_path.as_posix(),
                    epochs=epochs,
                    imgsz=640,
                    batch=8,
                    device=0,
                    workers=0,
                    project=yolo_project.as_posix(),
                    name=train_dir.name
                )
            else:
                print("[YOLO] 未找到best.pt、last.pt或best.onnx")
                return
        else:
            # 全新训练
            model = YOLO("yolo11m.pt")
            model.train(
                data=yaml_path.as_posix(),
                epochs=epochs,
                imgsz=640,
                batch=8,
                device=0,
                workers=0,
                project=yolo_project.as_posix(),
                name=train_dir.name,
                patience=10,
                cache="ram"
            )
        
        # 导出ONNX
        best_model = train_dir / "weights" / "best.pt"
        print(f"[YOLO] 检查模型路径: {best_model.resolve()}")
        print(f"[YOLO] 路径存在: {best_model.exists()}")
        if best_model.exists():
            model = YOLO(str(best_model))
            model.export(format="onnx")
            
            # 创建model.json
            train_id = self.train_id_input.text() or self.default_model_id
            train_desc = self.train_desc_input.text() or self.default_model_desc
            model_json = {
                "id": train_id,
                "displayname": train_id,  # displayname同id
                "description": train_desc,
                "model_path": f"{train_id}_train/best.onnx",
                "classes": class_names,
                "nc": len(class_names),
                "input_size": [640, 640]
            }
            
            # 保存model.json到weights文件夹
            weights_dir = best_model.parent
            weights_json = weights_dir / "model.json"
            with open(weights_json, 'w', encoding='utf-8') as f:
                json.dump(model_json, f, ensure_ascii=False, indent=2)
            
            # 整体拷贝runs/detect/yolo_runs文件夹到1dst/{ID}_train
            upload_dir = Path("1dst") / f"{train_id}_train"
            if upload_dir.exists():
                shutil.rmtree(upload_dir)
            shutil.copytree(yolo_runs_dir, upload_dir)
            print(f"[YOLO] 已拷贝runs文件夹到 {upload_dir}")
            
            # 压缩上传整个文件夹
            import zipfile
            zip_filename = f"{train_id}_train.zip"
            zip_path = Path("1dst") / zip_filename
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in upload_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(upload_dir.parent))
            print(f"[ZIP] 正在上传模型压缩包...")
            zip_url = f"http://obs.dimond.top/{zip_filename}"
            result = subprocess.run(['curl', '--upload-file', str(zip_path), zip_url], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[YOLO] 模型上传成功: {zip_url}")
            else:
                print(f"[YOLO] 模型上传失败: {result.stderr}")
            
            print(f"[YOLO] 训练完成!")
            print(f"[YOLO] 模型ID: {train_id}")
        else:
            print("[YOLO] 未找到训练好的模型")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="视频标注工具 - 统一控制面板")
    parser.add_argument('--src', type=str, default=None, help='视频文件路径')
    parser.add_argument('--iou', type=float, default=None, help='IoU阈值')
    parser.add_argument('--merge-iou', type=float, default=None, help='当前帧IoU阈值')
    parser.add_argument('--items', type=str, default=None, help='物品列表，逗号分隔')
    args = parser.parse_args()

    if args.src:
        cmd = [sys.executable, 'annotate_video.py',
               '--src', args.src]
        if args.iou is not None:
            cmd.extend(['--iou', str(args.iou)])
        if args.merge_iou is not None:
            cmd.extend(['--merge-iou', str(args.merge_iou)])
        if args.items:
            cmd.extend(['--items', args.items])
        subprocess.Popen(cmd, cwd=str(Path.cwd()))
        return

    app = QApplication(sys.argv)
    panel = UnifiedPanel()
    panel.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
