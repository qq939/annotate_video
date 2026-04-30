#!/usr/bin/env python3
"""图片标注工具 - 使用 SAM3 进行图片分割标注，支持 bbox 和文本提示"""

import sys
import shutil
import cv2
import numpy as np
import json
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QLineEdit, QFileDialog, QGroupBox, QMessageBox, QDialog, QShortcut)
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

SRC_IMAGES_DIR = "src/images"
TEMP_DATA_IMAGE_DIR = "temp_data_image"
DST_IMAGES_DIR = "1dst/image"
SAM_MODEL_PATH = "sam3.pt"
IOU_THRESHOLD = 0.5
MERGE_IOU_THRESHOLD = 0.5


class ImageAnnotationWidget(QWidget):
    box_added = pyqtSignal()

    def __init__(self, image, boxes, color_index, parent=None):
        super().__init__(parent)
        self.image = image
        self.orig_h, self.orig_w = image.shape[:2]
        self.boxes = boxes
        self.color_index = color_index
        self.drawing = False
        self.start_point = QPoint()
        self.current_rect = QRect()

        screen = QApplication.primaryScreen().geometry()
        max_w = screen.width() - 40
        max_h = screen.height() - 100
        scale_w = max_w / self.orig_w if self.orig_w > max_w else 1.0
        scale_h = max_h / self.orig_h if self.orig_h > max_h else 1.0
        self.scale = min(scale_w, scale_h)
        self.display_w = int(self.orig_w * self.scale)
        self.display_h = int(self.orig_h * self.scale)

        self.setFixedSize(self.display_w, self.display_h)
        self.setMinimumSize(self.display_w, self.display_h)
        self.setMaximumSize(self.display_w, self.display_h)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.qimage = QImage(rgb.data, self.orig_w, self.orig_h, self.orig_w * 3, QImage.Format_RGB888)

    def _to_orig(self, pt):
        return QPoint(int(pt.x() / self.scale), int(pt.y() / self.scale))

    def _rect_to_orig(self, r):
        return QRect(self._to_orig(r.topLeft()), self._to_orig(r.bottomRight()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = self._to_orig(event.pos())
            self.current_rect = QRect(self.start_point, self.start_point)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.current_rect = QRect(self.start_point, self._to_orig(event.pos())).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self._to_orig(event.pos())).normalized()
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
        scaled = self.qimage.scaled(self.display_w, self.display_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawImage(0, 0, scaled)
        for i, box in enumerate(self.boxes):
            color = QColor(*box['color'][::-1])
            pen = QPen(color, 2)
            painter.setPen(pen)
            x1 = int(box['x1'] * self.scale)
            y1 = int(box['y1'] * self.scale)
            w = int((box['x2'] - box['x1']) * self.scale)
            h = int((box['y2'] - box['y1']) * self.scale)
            painter.drawRect(x1, y1, w, h)
            font_size = max(10, int(14 * self.scale))
            painter.setFont(QFont("Arial", font_size))
            painter.drawText(x1, y1 - 5, f"目标 {i + 1}")
        if self.drawing and not self.current_rect.isNull():
            color = QColor(*BOX_COLORS[self.color_index[0] % len(BOX_COLORS)][::-1])
            pen = QPen(color, 2)
            painter.setPen(pen)
            rx = int(self.current_rect.left() * self.scale)
            ry = int(self.current_rect.top() * self.scale)
            rw = int(self.current_rect.width() * self.scale)
            rh = int(self.current_rect.height() * self.scale)
            painter.drawRect(rx, ry, rw, rh)


class ImageAnnotationDialog(QDialog):
    def __init__(self, image_path, parent=None, start_color_idx=0):
        super().__init__(parent)
        self.image_path = image_path
        self.boxes = []
        self.color_index = [start_color_idx]
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        self._setup_ui()
        self._setup_shortcut()

    def _setup_ui(self):
        self.img_widget = ImageAnnotationWidget(self.image, self.boxes, self.color_index)
        dw = self.img_widget.display_w
        dh = self.img_widget.display_h
        self.setWindowTitle(f"图片标注 - {Path(self.image_path).name}")
        self.setFixedSize(dw, dh + 36)
        self.setMinimumSize(dw, dh + 36)
        self.setMaximumSize(dw, dh + 36)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

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
        main_layout.addWidget(self.img_widget)
        self.img_widget.setFocus()
        self.img_widget.box_added.connect(lambda: self.undo_btn.setEnabled(True))

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


class ImageAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片标注工具")
        self.setGeometry(100, 100, 400, 300)
        self.ctrl = VideoController()
        self.selected_image_path = ""
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
        self.selected_color_idx = [0]
        color_names = ["红", "橙", "黄", "绿", "青", "蓝", "紫"]
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

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", ".",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)"
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
        image_name = Path(image_path).name
        dst_image = src_dir / image_name
        if Path(image_path).resolve() != dst_image.resolve():
            shutil.copy2(image_path, dst_image)
            print(f"已拷贝到{src_dir}: {dst_image}")
        src_image = str(dst_image)

        dialog = ImageAnnotationDialog(src_image, self, self.selected_color_idx[0])
        if dialog.exec_() != QDialog.Accepted:
            return
        boxes = dialog.get_boxes()

        iou_val = float(self.iou_input.text() or "0.5")
        merge_iou_val = float(self.merge_iou_input.text() or "0.5")
        find_list = [s.strip() for s in self.text_input.text().split(',') if s.strip()]

        has_text = bool(find_list)
        has_bbox = bool(boxes)

        if not has_text and not has_bbox:
            QMessageBox.warning(self, "提示", "请至少框选目标或填写文本提示词")
            return

        self.statusLabel.setText("正在处理，请稍候...")
        QApplication.processEvents()

        try:
            from annotate_video import merge_masks_in_frame

            print(f"正在使用 SAM3Predictor 进行图片分割...")
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
                from ultralytics.models.sam import SAM3Predictor
                predictor = SAM3Predictor(overrides=overrides)
                results = predictor(source=src_image, bboxes=boxes, labels=[1] * len(boxes))
                r = list(results)[0] if hasattr(results, '__iter__') else results
                if hasattr(r, 'masks') and r.masks is not None:
                    all_masks.append(r.masks.data)
                    if hasattr(r, 'boxes') and r.boxes is not None:
                        all_confs.append(r.boxes.conf.cpu().numpy())
            else:
                from ultralytics.models.sam import SAM3SemanticPredictor
                predictor = SAM3SemanticPredictor(overrides=overrides)
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
                                ann = {
                                    'id': annotation_id[0], 'track_id': track_id, 'image_id': 0,
                                    'category_id': cat_idx, 'bbox': bbox, 'area': float(area),
                                    'segmentation': [polygon], 'iscrowd': 0, 'confidence': confidence
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

            orig_img = cv2.imread(src_image)
            annotated_img = orig_img.copy()

            if frame_annotations:
                for ann in frame_annotations:
                    track_id = ann['track_id']
                    color = BOX_COLORS[track_id % len(BOX_COLORS)]
                    b = ann['bbox']
                    conf = ann.get('confidence', 1.0)

                    overlay = annotated_img.copy()
                    seg = ann.get('segmentation', [])
                    if seg:
                        pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.addWeighted(annotated_img, 0.75, overlay, 0.25, 0, annotated_img)

                    cv2.rectangle(annotated_img,
                        (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])),
                        color, 1)

                    label = f"id{track_id} {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), baseline = cv2.getTextSize(label, font, 0.5, 1)
                    tx, ty = int(b[0]), max(14, int(b[1]))
                    cv2.rectangle(annotated_img, (tx, ty - th - baseline), (tx + tw, ty), (0, 0, 0), -1)
                    cv2.putText(annotated_img, label, (tx, ty - baseline), font, 0.5, (255, 255, 255), 1)

            output_path = dst_dir / image_name
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
