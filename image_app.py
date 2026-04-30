#!/usr/bin/env python3
"""图片标注工具 - 使用 SAM3 进行图片分割标注，支持 bbox 和文本提示"""

import sys
import shutil
import random
import cv2
import numpy as np
import json
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QFileDialog, QGroupBox, QMessageBox, QDialog, QShortcut, QScrollArea, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QFont, QKeySequence, QPalette

from video_control import VideoController

BOX_COLORS = [
    (0, 0, 255),    # 红 BGR
    (0, 165, 255),  # 橙 BGR
    (0, 255, 255),  # 黄 BGR
    (0, 255, 0),    # 绿 BGR
    (255, 255, 0),  # 青 BGR
    (255, 0, 0),    # 蓝 BGR
    (255, 0, 128),  # 紫 BGR
]

SRC_IMAGES_DIR = "1src/image"
TEMP_DATA_IMAGE_DIR = "temp_data_image"
DST_IMAGES_DIR = "1dst/image"
SAM_MODEL_PATH = "sam3.pt"
IOU_THRESHOLD = 0.5
MERGE_IOU_THRESHOLD = 0.5

def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def _match_annotation_to_box(ann_bbox, boxes):
    best_iou, best_idx = 0, 0
    for i, box in enumerate(boxes):
        iou = _bbox_iou(ann_bbox, box)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_idx, best_iou


def _convert_heic_to_jpg(heic_path, jpg_path):
    from PIL import Image
    import pillow_heif
    heif_file = pillow_heif.read_heif(str(heic_path))
    img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
    img.save(jpg_path, "JPEG")


def _filter_by_confidence(annotations, threshold):
    return [a for a in annotations if a.get('confidence', 1.0) >= threshold]


def _render_filtered_image(img, annotations, find_list, threshold):
    filtered = _filter_by_confidence(annotations, threshold)
    result = img.copy()
    for ann in filtered:
        cat_idx = ann['category_id']
        color = ann.get('color', BOX_COLORS[cat_idx % len(BOX_COLORS)])
        b = ann['bbox']
        conf = ann.get('confidence', 1.0)
        cat_name = find_list[cat_idx] if cat_idx < len(find_list) else f"obj{cat_idx}"

        overlay = result.copy()
        seg = ann.get('segmentation', [])
        if seg:
            pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(result, 0.75, overlay, 0.25, 0, result)

        cv2.rectangle(result, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), color, 1)

        label = f"{cat_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, 0.5, 1)
        tx, ty = int(b[0]), max(14, int(b[1]))
        cv2.rectangle(result, (tx, ty - th - baseline), (tx + tw, ty), (0, 0, 0), -1)
        cv2.putText(result, label, (tx, ty - baseline), font, 0.5, (255, 255, 255), 1)
    return result


def _load_temp_annotations(temp_dir):
    temp_path = Path(temp_dir)
    with open(temp_path / 'annotations.json', 'r') as f:
        coco_data = json.load(f)
    labels_path = list((temp_path / 'labels').glob('*.json'))
    labels_data = []
    if labels_path:
        with open(labels_path[0], 'r') as f:
            labels_data = json.load(f)
    frame_path = list((temp_path / 'frames').glob('*.jpg'))
    if not frame_path:
        frame_path = list((temp_path / 'frames').glob('*.png'))
    orig_img = None
    if frame_path:
        orig_img = cv2.imread(str(frame_path[0]))
    return coco_data, labels_data, orig_img


class ImageAnnotationWidget(QWidget):
    box_added = pyqtSignal()

    def __init__(self, image, boxes, color_index, category_names=None, parent=None):
        super().__init__(parent)
        self.image = image
        self.orig_h, self.orig_w = image.shape[:2]
        self.boxes = boxes
        self.color_index = color_index
        self.category_names = category_names or []
        self.drawing = False
        self.start_point = QPoint()
        self.current_rect = QRect()

        self.setFixedSize(self.orig_w, self.orig_h)
        self.setMinimumSize(self.orig_w, self.orig_h)
        self.setFocusPolicy(Qt.StrongFocus)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb.data, self.orig_w, self.orig_h, self.orig_w * 3, QImage.Format_RGB888)

    def mousePressEvent(self, event):
        self.setFocus()
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
            if self.current_rect.width() > 5 and self.current_rect.height() > 5:
                color = BOX_COLORS[self.color_index[0] % len(BOX_COLORS)]
                self.boxes.append({
                    'x1': self.current_rect.left(), 'y1': self.current_rect.top(),
                    'x2': self.current_rect.right(), 'y2': self.current_rect.bottom(),
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
            painter.setFont(QFont("Arial", 12))
            label = self.category_names[i] if i < len(self.category_names) else f"目标{i+1}"
            painter.drawText(box['x1'], box['y1'] - 3, label)
        if self.drawing and not self.current_rect.isNull():
            color = QColor(*BOX_COLORS[self.color_index[0] % len(BOX_COLORS)][::-1])
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)


class ImageAnnotationDialog(QDialog):
    def __init__(self, image_path, parent=None, start_color_idx=0, category_names=None):
        super().__init__(parent)
        self.image_path = image_path
        self.boxes = []
        self.color_index = [start_color_idx]
        self.category_names = category_names or []
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        self._setup_ui()
        self._setup_shortcut()

    def _setup_ui(self):
        self.img_widget = ImageAnnotationWidget(self.image, self.boxes, self.color_index, self.category_names)
        iw = self.img_widget.orig_w
        ih = self.img_widget.orig_h
        screen = QApplication.primaryScreen().geometry()
        sw, sh = screen.width() - 80, screen.height() - 120
        dw = min(iw, sw)
        dh = min(ih, sh)
        self.setWindowTitle(f"图片标注 - {Path(self.image_path).name}")
        self.setFixedSize(dw, dh + 36)
        self.setMinimumSize(dw, dh + 36)
        self.setMaximumSize(dw, dh + 36)
        self.move(screen.x() + (screen.width() - dw) // 2, screen.y() + (screen.height() - dh - 36) // 2)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setModal(True)

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

        scroll_area = QScrollArea()
        scroll_area.setFixedSize(dw, dh)
        scroll_area.setWidget(self.img_widget)
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.horizontalScrollBar().setHidden(True)
        scroll_area.verticalScrollBar().setHidden(True)
        scroll_area.installEventFilter(self)
        self.scroll_area = scroll_area
        self.img_widget.setFocus()
        self.img_widget.box_added.connect(lambda: self.undo_btn.setEnabled(True))
        main_layout.addWidget(scroll_area)

    def _setup_shortcut(self):
        QShortcut(QKeySequence("c"), self).activated.connect(self._undo_last)
        QShortcut(QKeySequence("C"), self).activated.connect(self._undo_last)
        QShortcut(QKeySequence("q"), self).activated.connect(self.reject)
        QShortcut(QKeySequence("Q"), self).activated.connect(self.reject)

    def _undo_last(self):
        if self.boxes:
            self.boxes.pop()
            self.color_index[0] = max(0, self.color_index[0] - 1)
            self.img_widget.update()
            self.undo_btn.setEnabled(len(self.boxes) > 0)

    def _select_color(self, idx):
        self.color_index[0] = idx
        self.img_widget.color_index[0] = idx
        self.img_widget.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_C, Qt.Key_c):
            self._undo_last()
        elif key in (Qt.Key_Q, Qt.Key_q):
            self.reject()
        else:
            super().keyPressEvent(event)

    def get_boxes(self):
        return [(b['x1'], b['y1'], b['x2'], b['y2']) for b in self.boxes]

    def get_boxes_and_colors(self):
        return [(b['x1'], b['y1'], b['x2'], b['y2'], b['color']) for b in self.boxes]


class ConfidenceFilterWidget(QWidget):
    def __init__(self, image, annotations, find_list, threshold, parent=None):
        super().__init__(parent)
        self.orig_img = image
        self.annotations = annotations
        self.find_list = find_list
        self.threshold = threshold
        self.orig_h, self.orig_w = image.shape[:2]

        self.setFixedSize(self.orig_w, self.orig_h)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb.data, self.orig_w, self.orig_h, self.orig_w * 3, QImage.Format_RGB888)
        self._update_render()

    def _update_render(self):
        rendered = _render_filtered_image(self.orig_img, self.annotations, self.find_list, self.threshold)
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb.data, self.orig_w, self.orig_h, self.orig_w * 3, QImage.Format_RGB888)
        self.update()

    def set_threshold(self, th):
        self.threshold = th
        self._update_render()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.qimage)


class ConfidenceFilterDialog(QDialog):
    def __init__(self, temp_dir, parent=None):
        super().__init__(parent)
        self.temp_dir = temp_dir
        self.coco_data, self.labels_data, self.orig_img = _load_temp_annotations(temp_dir)
        if self.orig_img is None:
            raise ValueError("无法加载临时目录中的图片")
        self.annotations = self.labels_data if self.labels_data else self.coco_data.get('annotations', [])
        self.find_list = self.coco_data.get('info', {}).get('FIND', [])
        self._setup_ui()
        self._setup_shortcut()
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)
        self.setModal(True)

    def _setup_ui(self):
        screen = QApplication.primaryScreen().geometry()
        sw, sh = screen.width() - 80, screen.height() - 160
        dw = min(self.orig_w, sw)
        dh = min(self.orig_h, sh)
        self.setWindowTitle(f"后处理 - 置信度筛选")
        self.setFixedSize(dw, dh + 80)
        self.move(screen.x() + (screen.width() - dw) // 2, screen.y() + (screen.height() - dh - 80) // 2)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QWidget()
        header.setFixedSize(dw, 40)
        header.setStyleSheet("background: rgba(0,0,0,180);")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(8)

        header_layout.addWidget(QLabel("置信度阈值:"))
        self.slider = QSlider(Qt.Horizontal)
        confs = [a.get('confidence', 1.0) for a in self.annotations]
        self.min_conf = min(confs) if confs else 0.0
        self.max_conf = max(confs) if confs else 1.0
        self.slider.setRange(0, 100)
        self.slider.setValue(int(self.min_conf * 100))
        self.slider.setFixedWidth(300)
        self.slider.valueChanged.connect(self._on_threshold_changed)
        header_layout.addWidget(self.slider)

        self.threshold_label = QLabel(f"{self.min_conf:.2f}")
        self.threshold_label.setStyleSheet("color: #ccc; font-size: 12px; min-width: 40px;")
        header_layout.addWidget(self.threshold_label)

        header_layout.addWidget(QLabel("数量:"))
        self.count_label = QLabel()
        self.count_label.setStyleSheet("color: #0f0; font-size: 12px; font-weight: bold;")
        header_layout.addWidget(self.count_label)
        header_layout.addStretch()

        self.save_btn = QPushButton("✓ 确认保存")
        self.save_btn.setFixedSize(100, 28)
        self.save_btn.setStyleSheet(
            "QPushButton { background: #00CC00; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #009900; }"
        )
        self.save_btn.clicked.connect(self.accept)
        header_layout.addWidget(self.save_btn)

        main_layout.addWidget(header)

        self.img_widget = ConfidenceFilterWidget(self.orig_img, self.annotations, self.find_list, self.min_conf)
        scroll_area = QScrollArea()
        scroll_area.setFixedSize(dw, dh)
        scroll_area.setWidget(self.img_widget)
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignCenter)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.horizontalScrollBar().setHidden(True)
        scroll_area.verticalScrollBar().setHidden(True)
        scroll_area.installEventFilter(self)
        self.scroll_area = scroll_area
        main_layout.addWidget(scroll_area)

        self._update_count()

    def _on_threshold_changed(self, val):
        th = val / 100.0
        self.threshold_label.setText(f"{th:.2f}")
        self.img_widget.set_threshold(th)
        self._update_count()

    def _update_count(self):
        filtered = _filter_by_confidence(self.annotations, self.img_widget.threshold)
        self.count_label.setText(f"{len(filtered)} / {len(self.annotations)}")

    def _setup_shortcut(self):
        QShortcut(QKeySequence("q"), self).activated.connect(self.reject)
        QShortcut(QKeySequence("Q"), self).activated.connect(self.reject)
        QShortcut(QKeySequence("Return"), self).activated.connect(self.accept)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_q):
            self.reject()
        else:
            super().keyPressEvent(event)


class ImageAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片标注工具")
        self.setGeometry(100, 100, 400, 300)
        self.ctrl = VideoController()
        self.selected_image_path = ""
        self.setAcceptDrops(True)
        self._setup_ui()

    def _select_color(self, idx):
        self.selected_color_idx[0] = idx
        for i, btn in enumerate(self.color_btns):
            qc = btn.palette().color(QPalette.Button)
            border = "2px solid white" if i == idx else "2px solid transparent"
            r, g, b = qc.red(), qc.green(), qc.blue()
            text_color = "black" if r > 200 or g > 200 else "white"
            btn.setStyleSheet(
                f"QPushButton {{ background: rgb({r},{g},{b}); color: {text_color}; "
                f"border: {border}; border-radius: 4px; }}"
                f"QPushButton:hover {{ border: 2px solid white; }}"
            )

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("图片标注工具")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(QLabel("文本提示词（多个用逗号分隔）:"))
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("如: nozzle, needle")
        layout.addWidget(self.text_input)

        self.use_semantic_cb = QCheckBox("使用语义推理模型")
        self.use_semantic_cb.setChecked(False)
        layout.addWidget(self.use_semantic_cb)

        iou_layout = QHBoxLayout()
        iou_layout.setSpacing(8)
        iou_layout.addWidget(QLabel("前后IoU:"))
        self.iou_input = QLineEdit("0.5")
        self.iou_input.setFixedWidth(60)
        iou_layout.addWidget(self.iou_input)
        iou_layout.addWidget(QLabel("帧IoU:"))
        self.merge_iou_input = QLineEdit("0.5")
        self.merge_iou_input.setFixedWidth(60)
        iou_layout.addWidget(self.merge_iou_input)
        iou_layout.addStretch()
        layout.addLayout(iou_layout)

        color_layout = QHBoxLayout()
        color_layout.setSpacing(6)
        color_layout.addWidget(QLabel("颜色:"))
        self.color_btns = []
        color_names = ["红", "橙", "黄", "绿", "青", "蓝", "紫"]
        self.selected_color_idx = [random.randint(0, len(color_names) - 1)]
        color_qcolors = [
            QColor(255, 0, 0),    # 红 RGB (BOX_COLORS: B=255,G=0,R=0)
            QColor(255, 165, 0),  # 橙 RGB
            QColor(255, 255, 0),  # 黄 RGB
            QColor(0, 255, 0),    # 绿 RGB
            QColor(0, 255, 255),  # 青 RGB
            QColor(0, 0, 255),    # 蓝 RGB
            QColor(255, 0, 128),  # 紫 RGB
        ]
        for i, (name, qc) in enumerate(zip(color_names, color_qcolors)):
            btn = QPushButton(name)
            btn.setFixedSize(36, 24)
            btn.setStyleSheet(
                f"QPushButton {{ background: rgb({qc.red()},{qc.green()},{qc.blue()}); "
                f"color: {'black' if qc.red() > 200 or qc.green() > 200 else 'white'}; "
                f"border: 2px solid {'white' if i == 0 else 'transparent'}; border-radius: 4px; }}"
                f"QPushButton:hover {{ border: 2px solid white; }}"
            )
            btn.clicked.connect(lambda _, idx=i: self._select_color(idx))
            color_layout.addWidget(btn)
            self.color_btns.append(btn)
        self._select_color(self.selected_color_idx[0])
        color_layout.addStretch()
        layout.addLayout(color_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        select_btn = QPushButton("选择图片")
        select_btn.clicked.connect(self.select_image)
        btn_layout.addWidget(select_btn)

        self.run_btn = QPushButton("执行标注")
        self.run_btn.clicked.connect(self.run_annotate)
        btn_layout.addWidget(self.run_btn)
        layout.addLayout(btn_layout)

        self.statusLabel = QLabel("就绪")
        self.statusLabel.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.statusLabel)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = Path(url.toLocalFile()).suffix.lower()
                    if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic'):
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                self.selected_image_path = path
                self.statusLabel.setText(f"已选: {Path(path).name}")
                return

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", ".",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.heic);;所有文件 (*)"
        )
        if path:
            self.selected_image_path = path
            self.statusLabel.setText(f"已选: {Path(path).name}")

    def run_annotate(self):
        image_path = self.selected_image_path
        if not image_path:
            QMessageBox.warning(self, "错误", "请先选择图片")
            return
        if not Path(image_path).exists():
            QMessageBox.warning(self, "错误", f"图片不存在: {image_path}")
            return

        src_dir = Path(SRC_IMAGES_DIR)
        src_dir.mkdir(parents=True, exist_ok=True)
        src_path = Path(image_path)
        if src_path.suffix.lower() == '.heic':
            image_name = src_path.stem + '.jpg'
            dst_image = src_dir / image_name
            if dst_image.exists():
                dst_image.unlink()
            _convert_heic_to_jpg(src_path, dst_image)
            print(f"HEIC已转换并拷贝到{src_dir}: {dst_image}")
        else:
            image_name = src_path.name
            dst_image = src_dir / image_name
            if src_path.resolve() != dst_image.resolve():
                if dst_image.exists():
                    dst_image.unlink()
                shutil.copy2(src_path, dst_image)
                print(f"已拷贝到{src_dir}: {dst_image}")
        src_image = str(dst_image)

        find_list = [s.strip() for s in self.text_input.text().split(',') if s.strip()]
        dialog = ImageAnnotationDialog(src_image, self, self.selected_color_idx[0], find_list)
        if dialog.exec_() != QDialog.Accepted:
            return
        boxes_colors = dialog.get_boxes_and_colors()
        boxes = [(b[0], b[1], b[2], b[3]) for b in boxes_colors]
        box_colors = {i: b[4] for i, b in enumerate(boxes_colors)}

        iou_val = float(self.iou_input.text() or "0.5")
        merge_iou_val = float(self.merge_iou_input.text() or "0.5")

        has_text = bool(find_list)
        has_bbox = bool(boxes)

        if not has_text and not has_bbox:
            QMessageBox.warning(self, "提示", "请至少框选目标或填写文本提示词")
            return

        self.statusLabel.setText("正在处理，请稍候...")
        QApplication.processEvents()

        try:
            from annotate_video import merge_masks_in_frame

            use_semantic = self.use_semantic_cb.isChecked()
            predictor_name = "SAM3SemanticPredictor" if use_semantic else "SAM3Predictor"
            print(f"正在使用 {predictor_name} 进行图片分割...")
            if find_list:
                for i, t in enumerate(find_list):
                    print(f"  [{i}] category: '{t}'")
            if boxes:
                print(f"  bboxes: {[tuple(int(x) for x in b) for b in boxes]}")

            import torch
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            overrides = dict(
                conf=0.25, task="segment", mode="predict",
                model=SAM_MODEL_PATH, device=device,
                half=False, save=False, verbose=False
            )

            all_masks = []
            all_confs = []

            if has_bbox:
                center_points = np.array([[
                    float((b[0] + b[2]) / 2),
                    float((b[1] + b[3]) / 2)
                ] for b in boxes], dtype=np.float32)
                print(f"  bbox中心点(points): {center_points.tolist()}")
                if use_semantic:
                    from ultralytics.models.sam import SAM3SemanticPredictor
                    predictor = SAM3SemanticPredictor(overrides=overrides)
                    results = predictor(source=src_image, bboxes=boxes, points=center_points, labels=[1] * len(boxes), text=find_list)
                else:
                    from ultralytics.models.sam import SAM3Predictor
                    predictor = SAM3Predictor(overrides=overrides)
                    results = predictor(source=src_image, bboxes=boxes, points=center_points, labels=[1] * len(boxes))
                r = list(results)[0] if hasattr(results, '__iter__') else results
                if hasattr(r, 'masks') and r.masks is not None:
                    all_masks.append(r.masks.data)
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        all_confs.append(r.boxes.conf.cpu().numpy())
            else:
                from ultralytics.models.sam import SAM3SemanticPredictor
                predictor = SAM3SemanticPredictor(overrides=overrides)
                if find_list:
                    results = predictor(source=src_image, text=find_list)
                else:
                    results = predictor(source=src_image)
                r = list(results)[0] if hasattr(results, '__iter__') else results
                if hasattr(r, 'masks') and r.masks is not None:
                    all_masks.append(r.masks.data)

            img_h, img_w = r.orig_img.shape[:2] if hasattr(r, 'orig_img') and r.orig_img is not None else cv2.imread(src_image).shape[:2]
            if img_h == 0 or img_w == 0:
                img_h, img_w = cv2.imread(src_image).shape[:2]

            temp_data_path = Path(TEMP_DATA_IMAGE_DIR)
            if temp_data_path.exists():
                shutil.rmtree(temp_data_path)
            temp_data_path.mkdir(parents=True, exist_ok=True)
            frames_dir = temp_data_path / "frames"
            frames_dir.mkdir(exist_ok=True)
            labels_dir = temp_data_path / "labels"
            labels_dir.mkdir(exist_ok=True)
            shutil.copy2(src_image, frames_dir / "frame_000000.jpg")

            dst_dir = Path(DST_IMAGES_DIR)
            dst_dir.mkdir(parents=True, exist_ok=True)

            coco_data = {
                'info': {
                    'description': 'Image Annotation Dataset',
                    'image_path': src_image,
                    'width': img_w, 'height': img_h,
                    'FIND': find_list
                },
                'images': [{
                    'id': 0, 'file_name': image_name,
                    'width': img_w, 'height': img_h
                }],
                'annotations': [],
                'categories': (
                    [{'id': i, 'name': name} for i, name in enumerate(find_list)]
                    if find_list else
                    [{'id': 0, 'name': 'object'}]
                )
            }

            frame_annotations = []
            annotation_id = [0]

            if all_masks:
                import torch
                combined = torch.cat(all_masks, dim=0)
                confs = np.concatenate(all_confs) if all_confs else None
                print(f"[DEBUG] 总masks={len(combined)}, confs={confs.tolist() if confs is not None else None}")

                current_masks = []
                current_bboxes = []
                contours_total = 0
                for mask in combined:
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
                    mask_binary = (mask_np > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_total += len(contours)
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
                    print(f"[DEBUG] 原始contours={contours_total}, 有效polygon={len(current_masks)}")
                    current_masks, current_bboxes = merge_masks_in_frame(current_masks, current_bboxes, merge_iou_val)
                    print(f"[DEBUG] merge后={len(current_masks)}")

                    merge_contours = 0
                    for mask in current_masks:
                        mb = (mask > 0.5).astype(np.uint8)
                        cs, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        merge_contours += len(cs)
                    print(f"[DEBUG] merge后contours={merge_contours}")

                    for idx, (mask, bbox) in enumerate(zip(current_masks, current_bboxes)):
                        mb = (mask > 0.5).astype(np.uint8)
                        cs, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in cs:
                            if len(contour) >= 3:
                                polygon = contour.squeeze().flatten().tolist()
                                area = cv2.contourArea(contour)
                                track_id = annotation_id[0]
                                if find_list:
                                    cat_idx = idx % len(find_list)
                                else:
                                    cat_idx = 0
                                confidence = float(confs[idx]) if confs is not None and idx < len(confs) else float(mask.max())
                                ann_bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                                matched_box_idx, match_iou = _match_annotation_to_box(ann_bbox_xyxy, boxes)
                                ann_color = box_colors.get(matched_box_idx, BOX_COLORS[0])
                                ann = {
                                    'id': annotation_id[0], 'track_id': track_id, 'image_id': 0,
                                    'category_id': cat_idx, 'bbox': bbox, 'area': float(area),
                                    'segmentation': [polygon], 'iscrowd': 0, 'confidence': confidence,
                                    'color': ann_color
                                }
                                coco_data['annotations'].append(ann)
                                frame_annotations.append(ann)
                                annotation_id[0] += 1
                else:
                    print("[DEBUG] 无有效polygon")

            print(f"[DEBUG] 图片annotations数量={len(frame_annotations)}")

            with open(labels_dir / "frame_000000.json", 'w') as f:
                json.dump(frame_annotations, f)
            with open(temp_data_path / 'annotations.json', 'w') as f:
                json.dump(coco_data, f)

            filter_dialog = ConfidenceFilterDialog(str(temp_data_path), self)
            if filter_dialog.exec_() != QDialog.Accepted:
                self.statusLabel.setText("已取消")
                return

            threshold = filter_dialog.img_widget.threshold
            final_annotations = _filter_by_confidence(frame_annotations, threshold)

            with open(labels_dir / "frame_000000.json", 'w') as f:
                json.dump(final_annotations, f)
            coco_data['annotations'] = final_annotations
            with open(temp_data_path / 'annotations.json', 'w') as f:
                json.dump(coco_data, f)

            orig_img = cv2.imread(src_image)
            annotated_img = _render_filtered_image(orig_img, final_annotations, find_list, 0.0)

            output_path = dst_dir / image_name
            if output_path.exists():
                output_path.unlink()
            cv2.imwrite(str(output_path), annotated_img)
            print(f"✓ 标注图片已保存到: {output_path}")
            print(f"✓ 中间结果已保存到: {temp_data_path}")

            self.statusLabel.setText(f"完成: {image_name}")
            QMessageBox.information(self, "完成", f"标注完成！\n输出: {output_path}\n临时: {temp_data_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusLabel.setText("处理失败")
            QMessageBox.critical(self, "错误", f"处理失败:\n{e}")


def main():
    app = QApplication(sys.argv)
    w = ImageAnnotatorApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
