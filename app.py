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

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QLineEdit, QFileDialog, QGroupBox, QTextEdit, QMessageBox, QListWidget, QSizePolicy, QDialog)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
from PyQt5.Qt import QDragEnterEvent, QDropEvent
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QFont, QPixmap, QKeySequence
from PyQt5.QtWidgets import QShortcut

from video_control import VideoController


BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255),
]


class AnnotationImageWidget(QWidget):
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
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
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
        self.setWindowTitle("视频标注")
        self.setFixedSize(w, h)
        self.setMaximumSize(w, h)
        self.setMinimumSize(w, h)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)

        central = QWidget()
        central.setFixedSize(w, h)
        central.setStyleSheet("background: #111;")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        header = QWidget()
        header.setFixedSize(w, 36)
        header.setStyleSheet("background: rgba(0,0,0,180);")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(4)

        instr_label = QLabel("操作：框选目标 | C:撤销 | Q:退出")
        instr_label.setStyleSheet("color: #ccc; font-size: 12px;")
        instr_label.setFixedWidth(320)
        header_layout.addWidget(instr_label)
        header_layout.addStretch()

        self.done_btn = QPushButton("✓ 完成标注")
        self.done_btn.setFixedSize(100, 28)
        self.done_btn.setStyleSheet(
            "QPushButton { background: #00CC00; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #009900; }"
        )
        self.done_btn.clicked.connect(self.accept)
        header_layout.addWidget(self.done_btn)

        main_layout.addWidget(header)

        self.img_widget = AnnotationImageWidget(self.frame, self.boxes, self.color_index)
        self.img_widget.setFocus()
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


class UnifiedPanel(QMainWindow):
    def __init__(self):
        super().__init__()
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

        main_layout.addWidget(self.create_annotate_section())
        main_layout.addWidget(self.create_viewer_section())
        main_layout.addWidget(self.create_save_section())

    def create_annotate_section(self):
        group = QGroupBox("1. 视频标注 (annotate_video)")
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        group.setLayout(layout)

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
        layout.addLayout(video_layout)

        iou_layout = QHBoxLayout()
        iou_layout.setSpacing(4)
        iou_layout.addWidget(QLabel("帧IoU:"))
        self.merge_iou_input = QLineEdit("0.5")
        self.merge_iou_input.setFixedWidth(40)
        self.merge_iou_input.setFixedHeight(22)
        iou_layout.addWidget(self.merge_iou_input)
        iou_layout.addWidget(QLabel("前后IoU:"))
        self.iou_input = QLineEdit("0.5")
        self.iou_input.setFixedWidth(40)
        self.iou_input.setFixedHeight(22)
        iou_layout.addWidget(self.iou_input)
        iou_layout.addWidget(QLabel("物品:"))
        self.items_input = QLineEdit()
        self.items_input.setMinimumWidth(100)
        self.items_input.setFixedHeight(22)
        iou_layout.addWidget(self.items_input)
        layout.addLayout(iou_layout)

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

    def run_annotate(self):
        video_path = self.video_input.text()
        if not video_path:
            QMessageBox.warning(self, "错误", "请先选择视频文件")
            return

        src_dir = Path("1src")
        src_dir.mkdir(exist_ok=True)
        video_name = Path(video_path).name

        if Path(video_path).parent.resolve() != src_dir.resolve():
            dst_video = src_dir / video_name
            if dst_video.exists():
                dst_video.unlink()
            shutil.copy2(video_path, dst_video)
            print(f"已拷贝到src: {dst_video}")

        src_video = str(src_dir / video_name)
        dialog = AnnotationDialog(src_video, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        boxes = dialog.get_boxes()

        iou_val = float(self.iou_input.text() or "0.5")
        merge_iou_val = float(self.merge_iou_input.text() or "0.5")
        items_text = self.items_input.text()
        find_list = [s.strip() for s in items_text.split(',') if s.strip()]

        self.statusBar().showMessage("正在处理视频，请稍候...")
        QApplication.processEvents()

        try:
            from annotate_video import SAM_MODEL_PATH, DST_DIR, TEMP_DATA_DIR
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
            if has_text:
                print(f"  文本提示词: {find_list}")
            if has_bbox:
                print(f"  bbox提示框: {boxes}")

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

            if predictor_name == "SAM3VideoSemanticPredictor":
                from ultralytics.models.sam import SAM3VideoSemanticPredictor
                predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            else:
                from ultralytics.models.sam import SAM3VideoPredictor
                predictor = SAM3VideoPredictor(overrides=overrides)

            cap = cv2.VideoCapture(src_video)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([chr(fourcc_int & 0xFF), chr((fourcc_int >> 8) & 0xFF), chr((fourcc_int >> 16) & 0xFF), chr((fourcc_int >> 24) & 0xFF)])
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
                predictor_args['bboxes'] = boxes
                predictor_args['labels'] = [1] * len(boxes)
            if has_text:
                predictor_args['text'] = find_list

            results = predictor(**predictor_args)
            frame_count = 0
            print("正在生成标注视频...")

            for r in results:
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
                            print(f"[DEBUG 帧{frame_count}] boxes.conf={confs.tolist()}")

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
                            print(f"[DEBUG 帧{frame_count}] 原始contours={debug_contours_count}, 有效polygon={debug_merged_count}, merge后={len(current_masks)}, track_ids={track_ids}")
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
                        print(f"[DEBUG 帧{frame_count}] masks_tensor长度=0")
                else:
                    print(f"[DEBUG 帧{frame_count}] 无masks属性或masks为None")

                print(f"[DEBUG 帧{frame_count}] 帧annotations数量={len(frame_annotations)}, track_ids={debug_track_ids}")
                with open(labels_dir / f"frame_{frame_count:06d}.json", 'w') as f:
                    json.dump(frame_annotations, f)

                annotated_frame = r.plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                if boxes:
                    from annotate_video import BOX_COLORS as AV_BOX_COLORS
                    for i, bbox in enumerate(boxes):
                        label = f"目标 {i + 1}"
                        annotated_frame_rgb = put_chinese_text(annotated_frame_rgb, label, (bbox[0], max(10, bbox[1] - 10)), font_size=15, color=AV_BOX_COLORS[i % len(AV_BOX_COLORS)])
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

            with open(temp_data_path / 'annotations.json', 'w') as f:
                json.dump(coco_data, f)

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
        open_btn = QPushButton("选择")
        open_btn.setFixedSize(44, 22)
        open_btn.clicked.connect(self.select_data_dir)
        path_layout.addWidget(open_btn)
        show_btn = QPushButton("显示")
        show_btn.setFixedSize(44, 22)
        show_btn.clicked.connect(self.show_viewer)
        path_layout.addWidget(show_btn)
        layout.addLayout(path_layout)

        category_layout = QVBoxLayout()
        category_layout.setSpacing(2)
        category_layout.addWidget(QLabel("类别名称 (trace_id → 类别):"))
        self.category_inputs = []
        for tid in range(1000000, 1000004):
            row = QHBoxLayout()
            row.setSpacing(2)
            label = QLabel(f"{tid}:")
            label.setFixedWidth(70)
            row.addWidget(label)
            inp = QLineEdit("Detect")
            inp.setFixedHeight(20)
            row.addWidget(inp)
            self.category_inputs.append(inp)
            category_layout.addLayout(row)
        layout.addLayout(category_layout)

        zoom_layout = QHBoxLayout()
        zoom_layout.setSpacing(4)
        zoom_layout.addWidget(QLabel("缩放"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(50)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedHeight(16)
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

        frame_nav_play_layout = QHBoxLayout()
        frame_nav_play_layout.setSpacing(2)
        self.backward_fast_btn = QPushButton("倒播")
        self.backward_fast_btn.setFixedHeight(24)
        self.backward_fast_btn.clicked.connect(self.toggle_backward_fast)
        frame_nav_play_layout.addWidget(self.backward_fast_btn)

        self.backward_btn = QPushButton("倒帧")
        self.backward_btn.setFixedHeight(24)
        self.backward_btn.clicked.connect(self.toggle_backward)
        frame_nav_play_layout.addWidget(self.backward_btn)

        self.prompt_btn = QPushButton("提示帧")
        self.prompt_btn.setFixedSize(50, 24)
        self.prompt_btn.setStyleSheet("QPushButton { background-color: #FFA500; color: white; border: none; border-radius: 3px; font-size: 11px; } QPushButton:hover { background-color: #FF8C00; }")
        self.prompt_btn.clicked.connect(self.toggle_prompt_mode)
        frame_nav_play_layout.addWidget(self.prompt_btn)

        self.frame_label = QLabel("1/1")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setFixedSize(50, 24)
        self.frame_label.setStyleSheet("QLabel { background-color: #333; color: #fff; border-radius: 3px; font-weight: bold; font-size: 11px; }")
        frame_nav_play_layout.addWidget(self.frame_label)

        self.next_btn = QPushButton("正帧")
        self.next_btn.setFixedHeight(24)
        self.next_btn.clicked.connect(self.toggle_play)
        frame_nav_play_layout.addWidget(self.next_btn)

        self.forward_fast_btn = QPushButton("正播")
        self.forward_fast_btn.setFixedHeight(24)
        self.forward_fast_btn.clicked.connect(self.toggle_play_fast)
        frame_nav_play_layout.addWidget(self.forward_fast_btn)
        layout.addLayout(frame_nav_play_layout)

        delete_trace_layout = QHBoxLayout()
        delete_trace_layout.setSpacing(4)
        delete_trace_layout.addWidget(QLabel("删除trace"))
        self.delete_trace_input = QLineEdit()
        self.delete_trace_input.setPlaceholderText("输入")
        self.delete_trace_input.setFixedHeight(24)
        delete_trace_layout.addWidget(self.delete_trace_input)
        delete_trace_btn = QPushButton("删除")
        delete_trace_btn.setFixedSize(40, 24)
        delete_trace_btn.setStyleSheet("QPushButton { background-color: #FF4444; color: white; border: none; border-radius: 3px; } QPushButton:hover { background-color: #CC0000; }")
        delete_trace_btn.clicked.connect(self.delete_trace_id)
        delete_trace_layout.addWidget(delete_trace_btn)
        layout.addLayout(delete_trace_layout)

        del_layout = QHBoxLayout()
        del_layout.setSpacing(4)
        del_layout.addWidget(QLabel("绿点列表"))
        self.track_id_list = QListWidget()
        del_layout.addWidget(self.track_id_list)

        del_btn_layout = QVBoxLayout()
        del_btn_layout.setSpacing(2)

        trace_id_ctrl_layout = QHBoxLayout()
        trace_id_ctrl_layout.setSpacing(2)
        self.trace_id_minus_btn = QPushButton("-")
        self.trace_id_minus_btn.setFixedSize(24, 24)
        self.trace_id_minus_btn.setStyleSheet("QPushButton { background-color: #FFA500; color: white; border: none; border-radius: 3px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #FF8C00; }")
        self.trace_id_minus_btn.clicked.connect(self.decrement_track_id)
        trace_id_ctrl_layout.addWidget(self.trace_id_minus_btn)

        self.trace_id_label = QLabel(str(self.ctrl.next_track_id))
        self.trace_id_label.setAlignment(Qt.AlignCenter)
        self.trace_id_label.setFixedHeight(24)
        self.trace_id_label.setStyleSheet("QLabel { background-color: #222; color: #ccc; border: 1px solid #555; border-radius: 3px; font-size: 11px; font-weight: bold; padding: 0 6px; }")
        trace_id_ctrl_layout.addWidget(self.trace_id_label)

        self.trace_id_plus_btn = QPushButton("+")
        self.trace_id_plus_btn.setFixedSize(24, 24)
        self.trace_id_plus_btn.setStyleSheet("QPushButton { background-color: #00CC00; color: white; border: none; border-radius: 3px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #009900; }")
        self.trace_id_plus_btn.clicked.connect(self.increment_track_id)
        trace_id_ctrl_layout.addWidget(self.trace_id_plus_btn)

        del_btn_layout.addLayout(trace_id_ctrl_layout)

        track_btn_row = QHBoxLayout()
        track_btn_row.setSpacing(2)
        remove_btn = QPushButton("删除")
        remove_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        remove_btn.setFixedHeight(24)
        remove_btn.setStyleSheet("QPushButton { background-color: #FF4444; color: white; border: none; border-radius: 3px; } QPushButton:hover { background-color: #CC0000; }")
        remove_btn.clicked.connect(self.remove_selected_track_id)
        track_btn_row.addWidget(remove_btn)

        clear_del_btn = QPushButton("清空")
        clear_del_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        clear_del_btn.setFixedHeight(24)
        clear_del_btn.setStyleSheet("QPushButton { background-color: #555555; color: white; border: none; border-radius: 3px; } QPushButton:hover { background-color: #333333; }")
        clear_del_btn.clicked.connect(self.clear_track_id)
        track_btn_row.addWidget(clear_del_btn)
        del_btn_layout.addLayout(track_btn_row)

        del_layout.addLayout(del_btn_layout)
        layout.addLayout(del_layout)

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
        input_dir_name_layout.addWidget(QLabel("名称:"))
        self.save_output_name = QLineEdit("1dst.mp4")
        self.save_output_name.setFixedWidth(80)
        self.save_output_name.setFixedHeight(22)
        input_dir_name_layout.addWidget(self.save_output_name)
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

        color_btn_layout = QHBoxLayout()
        color_btn_layout.setSpacing(2)
        color_btn_layout.addWidget(QLabel("颜色:"))
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
            color_btn_layout.addWidget(btn)
        layout.addLayout(color_btn_layout)

        self.save_btn = QPushButton("💾 保存视频并上传OBS")
        self.save_btn.setFixedHeight(28)
        self.save_btn.clicked.connect(self.run_save)
        layout.addWidget(self.save_btn)

        return group

    def on_zoom_change(self, value):
        if self.viewer:
            factor = value / 100.0
            self.zoom_label.setText(f"{value}%")
            self.viewer.set_zoom(factor)

    def on_conf_change(self, value):
        self.ctrl.conf_threshold = value / 100.0
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
        if not prompt_bboxes:
            QMessageBox.warning(self, "错误", "请先绘制至少一个 Bbox")
            self.reset_prompt_btn()
            return

        prompt_idx = self.prompt_frame_idx
        total = self.total_frames
        frames_dir = self.temp_data_path / "frames"
        labels_dir = self.temp_data_path / "labels"
        annotations_file = self.temp_data_path / "annotations.json"

        self.prompt_btn.setText("正在处理...")
        QApplication.processEvents()

        try:
            from annotate_video import merge_masks_in_frame, TrackManager, get_device, SAM_MODEL_PATH, put_chinese_text
            from ultralytics.models.sam import SAM3VideoPredictor

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

            predictor = SAM3VideoPredictor(overrides=overrides)
            print(f"使用 SAM3VideoPredictor 进行双向标注，track_id 起始: 50000")

            sample_frame = cv2.imread(str(frames_dir / f"frame_{0:06d}.jpg"))
            height, width = sample_frame.shape[:2]

            FIRST_ID = 50000
            forward_annotations = []
            backward_annotations = []

            def process_clip(start_frame, end_frame, forward=True, prompt_bboxes=None):
                direction = "向前" if forward else "向后"
                print(f"\n[DEBUG {direction}] === 进入 process_clip ===")
                print(f"[DEBUG {direction}] start_frame={start_frame}, end_frame={end_frame}, 总帧数={end_frame - start_frame}")

                if start_frame >= end_frame:
                    print(f"[DEBUG {direction}] start_frame >= end_frame, 直接返回空列表")
                    return []

                temp_frames = Path("temp_inject") / ("forward" if forward else "backward")
                temp_frames.mkdir(parents=True, exist_ok=True)

                frame_count = end_frame - start_frame
                print(f"[DEBUG {direction}] 正在复制 {frame_count} 帧到临时目录...")
                for i in range(start_frame, end_frame):
                    src = frames_dir / f"frame_{i:06d}.jpg"
                    dst = temp_frames / f"frame_{i - start_frame:06d}.jpg"
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
                for i in range(start_frame, end_frame):
                    frame = cv2.imread(str(frames_dir / f"frame_{i:06d}.jpg"))
                    if frame is not None:
                        out.write(frame)
                        frames_written += 1
                out.release()
                print(f"[DEBUG {direction}] ✓ 视频片段生成完成: {frames_written} 帧")
                cap_check = cv2.VideoCapture(clip_path)
                actual_clip_frames = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_check.release()
                print(f"[DEBUG {direction}] clip文件实际帧数: {actual_clip_frames}, expected: {end_frame - start_frame}")

                print(f"[DEBUG {direction}] 正在加载 SAM3VideoPredictor 处理...")
                print(f"[DEBUG {direction}] prompt_bboxes={prompt_bboxes}")
                if prompt_bboxes:
                    results = predictor(source=clip_path, stream=True, bboxes=prompt_bboxes, labels=[1]*len(prompt_bboxes))
                else:
                    results = predictor(source=clip_path, stream=True)
                    print(f"[DEBUG {direction}] ⚠️ prompt_bboxes为空，无法进行SAM3VideoPredictor分割！")
                manager = TrackManager(iou_threshold=float(self.iou_input.text() or "0.5"))
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
                                            ann = {
                                                'id': ann_id, 'track_id': tid, 'image_id': frame_idx + start_frame,
                                                'category_id': tid, 'bbox': bb, 'area': float(area2),
                                                'segmentation': [poly2], 'iscrowd': 0, 'confidence': conf,
                                                'category': 'Detect'
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

                    orig_frame_idx = frame_idx + start_frame
                    if orig_frame_idx >= total:
                        print(f"[DEBUG {direction}] [帧{total_results}] ⚠️ orig_frame_idx={orig_frame_idx} >= total={total}，跳过")
                        frame_idx += 1
                        continue
                    print(f"[DEBUG {direction}] [帧{total_results}] clip_frame={frame_idx} → 原帧{orig_frame_idx}, 新增标注数={len(frame_anns)}")
                    label_file = labels_dir / f"frame_{orig_frame_idx:06d}.json"
                    existing_anns = []
                    if label_file.exists():
                        with open(label_file) as f:
                            existing_anns = json.load(f)
                        print(f"[DEBUG {direction}] [帧{total_results}] 已存在标注{len(existing_anns)}条，追加新标注")
                    merged_anns = existing_anns + frame_anns
                    with open(label_file, 'w') as f:
                        json.dump(merged_anns, f)
                    print(f"[DEBUG {direction}] [帧{total_results}] 保存label文件: frame_{orig_frame_idx:06d}.json, 原有{len(existing_anns)}+新增{len(frame_anns)}=合计{len(merged_anns)}")
                    frame_idx += 1

                print(f"[DEBUG {direction}] === process_clip 完成 ===")
                print(f"[DEBUG {direction}] 总results数={total_results}, 总frame_idx={frame_idx}, 总annotations={len(result_anns)}, clip帧数={end_frame - start_frame}")
                if total_results != (end_frame - start_frame):
                    print(f"[DEBUG {direction}] ⚠️ 警告: predictor返回{total_results}个结果，但clip有{end_frame - start_frame}帧，可能有帧对齐问题！")
                print(f"[DEBUG {direction}] id范围: {FIRST_ID} ~ {ann_id - 1}")
                return result_anns

            print(f"=== 双向标注开始 === 提示帧: {prompt_idx}, 总帧数: {total}, prompt_bboxes数量: {len(prompt_bboxes)}")
            forward_start = prompt_idx + 1
            print(f"[1/2] 向前标注: 帧 {forward_start} → {total-1} (共 {total - forward_start} 帧)")
            forward_anns = process_clip(forward_start, total, forward=True, prompt_bboxes=prompt_bboxes)

            print(f"\n[2/2] 向后标注: 帧 0 → {prompt_idx-1} (共 {prompt_idx} 帧)")
            backward_anns = process_clip(0, prompt_idx, forward=False, prompt_bboxes=prompt_bboxes)

            all_new_anns = backward_anns + forward_anns
            print(f"\n[DEBUG 汇总] 向后标注={len(backward_anns)}, 向前标注={len(forward_anns)}, 合计={len(all_new_anns)}")

            if not all_new_anns:
                QMessageBox.warning(self, "提示", "未检测到任何分割结果")
                self.reset_prompt_btn()
                return

            if annotations_file.exists():
                with open(annotations_file) as f:
                    coco = json.load(f)
                print(f"[DEBUG 汇总] 现有coco: 已有annotations={len(coco.get('annotations', []))}")
            else:
                coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}
                print(f"[DEBUG 汇总] annotations.json 不存在，创建新的coco结构")

            max_img_id = max([img['id'] for img in coco.get('images', [])], default=-1)
            max_ann_id = max([ann['id'] for ann in coco.get('annotations', [])], default=FIRST_ID - 1)
            max_track_id = max([ann['track_id'] for ann in coco.get('annotations', [])], default=FIRST_ID - 1)
            print(f"[DEBUG 汇总] 现有max_ann_id={max_ann_id}, max_track_id={max_track_id}")

            new_anns_count = 0
            for ann in all_new_anns:
                new_ann = dict(ann)
                max_ann_id += 1
                new_ann['id'] = max_ann_id
                new_ann['track_id'] = max_track_id + 1
                new_ann['category_id'] = new_ann['track_id']
                max_track_id = new_ann['track_id']
                coco['annotations'].append(new_ann)
                new_anns_count += 1

            print(f"[DEBUG 汇总] 追加 {new_anns_count} 条标注, 最终track_id范围: {FIRST_ID}~{max_track_id}")

            with open(annotations_file, 'w') as f:
                json.dump(coco, f)
            print(f"[DEBUG 汇总] ✓ annotations.json 已写入")
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

    def reset_prompt_btn(self):
        self.prompt_drawing_mode = False
        self.prompt_frame_idx = -1
        self.prompt_btn.setEnabled(True)
        self.prompt_btn.setText("设为提示帧")
        self.prompt_btn.setStyleSheet("QPushButton { background-color: #FFA500; color: white; border: none; border-radius: 3px; } QPushButton:hover { background-color: #FF8C00; }")
        if self.viewer:
            self.viewer.enable_bbox_drawing(False)

    def remove_selected_track_id(self):
        row = self.track_id_list.currentRow()
        if row >= 0:
            self.ctrl.remove_track_id_point(row)
            self.track_id_list.takeItem(row)
            if self.viewer:
                self.viewer.update_display()

    def clear_track_id(self):
        for pt in self.ctrl.track_id_points:
            if pt['assigned_id'] is not None:
                self._convert_track_id(pt['assigned_id'], pt['track_id'])
                self.ctrl.assigned_to_original.pop(pt['assigned_id'], None)
                self.ctrl.track_ids_to_9999.discard(pt['track_id'])
        self.ctrl.clear_track_id_points()
        self.track_id_list.clear()
        if self.viewer:
            self.viewer.update_display()

    def increment_track_id(self):
        self.ctrl.next_track_id += 1
        self.trace_id_label.setText(str(self.ctrl.next_track_id))
        print(f"next_track_id 递增 → {self.ctrl.next_track_id}")

    def decrement_track_id(self):
        if self.ctrl.next_track_id > 1000000:
            self.ctrl.next_track_id -= 1
            self.trace_id_label.setText(str(self.ctrl.next_track_id))
            print(f"next_track_id 递减 → {self.ctrl.next_track_id}")

    def delete_trace_id(self):
        trace_id_text = self.delete_trace_input.text().strip()
        if not trace_id_text:
            QMessageBox.warning(self, "错误", "请输入要删除的 trace id")
            return
        try:
            trace_id = int(trace_id_text)
        except ValueError:
            QMessageBox.warning(self, "错误", "trace id 必须是整数")
            return

        labels_dir = self.temp_data_path / "labels"
        annotations_file = self.temp_data_path / "annotations.json"
        if not labels_dir.exists():
            QMessageBox.warning(self, "错误", "labels 目录不存在")
            return

        frame_count = 0
        for label_file in sorted(labels_dir.glob("frame_*.json")):
            with open(label_file) as f:
                anns = json.load(f)
            original_len = len(anns)
            anns = [ann for ann in anns if ann.get('track_id') != trace_id]
            if len(anns) < original_len:
                with open(label_file, 'w') as f:
                    json.dump(anns, f)
                frame_count += 1

        if annotations_file.exists():
            with open(annotations_file) as f:
                coco = json.load(f)
            original_len = len(coco.get('annotations', []))
            coco['annotations'] = [ann for ann in coco.get('annotations', []) if ann.get('track_id') != trace_id]
            if len(coco['annotations']) < original_len:
                with open(annotations_file, 'w') as f:
                    json.dump(coco, f)

        QMessageBox.information(self, "完成", f"已从 {frame_count} 帧中删除 trace_id={trace_id}")
        if self.viewer:
            self.viewer.update_display()

    def remove_selected_track_id(self):
        row = self.track_id_list.currentRow()
        if row < 0:
            return
        pt = self.ctrl.track_id_points[row]
        if pt['assigned_id'] is not None:
            self._convert_track_id(pt['assigned_id'], pt['track_id'])
            self.ctrl.assigned_to_original.pop(pt['assigned_id'], None)
            self.ctrl.track_ids_to_9999.discard(pt['track_id'])
        self.ctrl.remove_track_id_point(row)
        self.track_id_list.takeItem(row)
        self._refresh_track_id_list()
        if self.viewer:
            self.viewer.update_display()

    def _refresh_track_id_list(self):
        self.track_id_list.clear()
        for i, pt in enumerate(self.ctrl.track_id_points):
            if pt['assigned_id'] is not None:
                self.track_id_list.addItem(
                    f"绿点 {i} 帧{pt['frame_idx']+1} ({pt['x']},{pt['y']}) ID:{pt['track_id']}→{pt['assigned_id']}"
                )
            else:
                self.track_id_list.addItem(
                    f"绿点 {i} 帧{pt['frame_idx']+1} ({pt['x']},{pt['y']}) ID:{pt['track_id']}"
                )

    def _convert_track_id(self, old_id, new_id):
        labels_dir = self.temp_data_path / "labels"
        converted_count = 0
        for label_file in sorted(labels_dir.glob("frame_*.json")):
            with open(label_file) as f:
                frame_anns = json.load(f)
            changed = False
            for ann in frame_anns:
                if ann.get('track_id') == old_id:
                    ann['track_id'] = new_id
                    changed = True
            if changed:
                with open(label_file, 'w') as f:
                    json.dump(frame_anns, f)
                converted_count += 1
        print(f"已将 track_id={old_id} → {new_id}，共影响 {converted_count} 帧")

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
        found = self.ctrl.find_annotation_at(filtered, video_x, video_y)

        if found is not None:
            track_id = found.get('track_id', found.get('id', 0))
            assigned_id = self.ctrl.next_track_id
            self.ctrl.add_track_id_point(video_x, video_y, frame_idx, track_id)
            pt = self.ctrl.track_id_points[-1]
            pt['assigned_id'] = assigned_id
            self.ctrl.track_ids_to_9999.add(track_id)
            self.ctrl.assigned_to_original[assigned_id] = track_id
            self._convert_track_id(track_id, assigned_id)
            idx = len(self.ctrl.track_id_points) - 1
            self.track_id_list.addItem(f"绿点 {idx} 帧{frame_idx+1} ({video_x},{video_y}) ID:{track_id}→{assigned_id}")
            self.viewer.update_display()

    def select_data_dir(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据目录或视频", ".",
            "所有文件 (*);;目录"
        )
        if not path:
            return
        p = Path(path)
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.mp4', '.flv', '.wmv', '.webm'}
        if p.suffix.lower() in video_exts:
            self._extract_video_to_temp_data(p)
        else:
            self.path_input.setText(str(p))
            self.temp_data_path = p

    def _extract_video_to_temp_data(self, video_path=None):
        if video_path is None:
            video_path_selected, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV);;所有文件 (*)"
            )
            if not video_path_selected:
                return
            video_path = video_path_selected

        temp_dir = self.temp_data_path = Path(self.path_input.text() or "temp_data")
        frames_dir = temp_dir / "frames"
        labels_dir = temp_dir / "labels"

        if frames_dir.exists():
            import shutil
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        self.statusBar().showMessage("正在切帧，请稍候...")
        QApplication.processEvents()

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "错误", f"无法打开视频: {video_path}")
                return

            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([chr(fourcc_int & 0xFF), chr((fourcc_int >> 8) & 0xFF), chr((fourcc_int >> 16) & 0xFF), chr((fourcc_int >> 24) & 0xFF)])
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), frame)
                frame_count += 1
                if frame_count % 100 == 0:
                    self.statusBar().showMessage(f"正在切帧... {frame_count} 帧")
                    QApplication.processEvents()
            cap.release()

            coco_data = {
                'info': {
                    'description': 'Video Annotation Dataset',
                    'video_path': video_path,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'fourcc': fourcc_str,
                    'FIND': []
                },
                'images': [
                    {'id': i, 'file_name': f"frame_{i:06d}.jpg", 'width': width, 'height': height, 'frame_count': i}
                    for i in range(frame_count)
                ],
                'annotations': [],
                'categories': []
            }

            with open(temp_dir / 'annotations.json', 'w') as f:
                json.dump(coco_data, f)

            for i in range(frame_count):
                with open(labels_dir / f"frame_{i:06d}.json", 'w') as f:
                    json.dump([], f)

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

    def show_viewer(self):
        from video_viewer import VideoViewer
        self.temp_data_path = Path(self.path_input.text())
        if not self.temp_data_path.exists():
            QMessageBox.warning(self, "错误", "数据目录不存在")
            return
        if not (self.temp_data_path / "annotations.json").exists():
            QMessageBox.warning(self, "错误", "annotations.json 不存在")
            return

        with open(self.temp_data_path / "annotations.json") as f:
            coco_data = json.load(f)
        self.total_frames = len(coco_data.get('images', []))

        self.viewer = VideoViewer(str(self.temp_data_path), controller=self.ctrl)
        self.viewer.video_clicked.connect(self.handle_viewer_click)
        geo = self.geometry()
        self.viewer.move(geo.right(), geo.top())
        self.viewer.show()
        self.viewer.update_display()

        self.frame_label.setText(f"帧: 1/{self.total_frames}")

    def select_save_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入目录", ".")
        if folder:
            self.save_input_dir.setText(folder)

    def _get_category_for_track_id(self, track_id):
        if 1000000 <= track_id <= 1000003:
            idx = track_id - 1000000
            name = self.category_inputs[idx].text() or "Detect"
            return (idx, name)
        elif track_id >= 1000000:
            return (track_id - 1000000, "Detect")
        return (0, self.ctrl.category_name)

    def export_to_temp_data_post(self):
        data_dir = self.temp_data_path
        if not data_dir.exists():
            QMessageBox.warning(self, "错误", "数据目录不存在")
            return

        annotations_file = data_dir / "annotations.json"
        if not annotations_file.exists():
            QMessageBox.warning(self, "错误", "annotations.json 不存在")
            return

        with open(annotations_file) as f:
            coco_data = json.load(f)

        video_info = coco_data.get('info', {})
        total_frames = len(coco_data.get('images', []))
        if total_frames == 0:
            QMessageBox.warning(self, "错误", "没有帧数据")
            return

        output_path = Path("temp_data_post")
        output_path.mkdir(exist_ok=True)
        output_labels_dir = output_path / "labels"
        output_labels_dir.mkdir(exist_ok=True)
        output_frames_dir = output_path / "frames"
        output_frames_dir.mkdir(exist_ok=True)

        labels_dir = data_dir / "labels"
        frames_dir = data_dir / "frames"
        cat_id_set = set()

        print("步骤1: 保存到 temp_data_post...")
        for i in range(total_frames):
            frame_path = str(frames_dir / f"frame_{i:06d}.jpg")
            frame = cv2.imread(frame_path)

            output_frame_path = str(output_frames_dir / f"frame_{i:06d}.jpg")
            cv2.imwrite(output_frame_path, frame)

            label_path = labels_dir / f"frame_{i:06d}.json"
            output_label_path = output_labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path) as f:
                    annotations = json.load(f)

                filtered = self.ctrl.filter_annotations(annotations)
                track_id_filtered = [ann for ann in filtered if ann.get('track_id', 0) > 999998]

                frame_anns = []
                for ann in track_id_filtered:
                    tid = ann.get('track_id', 0)
                    cat_id, cat_name = self._get_category_for_track_id(tid)
                    cat_id_set.add((cat_id, cat_name))
                    ann_copy = ann.copy()
                    ann_copy['category_id'] = cat_id
                    ann_copy['category'] = cat_name
                    frame_anns.append(ann_copy)

                with open(output_label_path, 'w') as f:
                    json.dump(frame_anns, f)

        all_annotations = []
        for i in range(total_frames):
            label_path = labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path) as f:
                    annotations = json.load(f)
                filtered = self.ctrl.filter_annotations(annotations)
                for ann in filtered:
                    if ann.get('track_id', 0) > 999998:
                        tid = ann.get('track_id', 0)
                        cat_id, cat_name = self._get_category_for_track_id(tid)
                        cat_id_set.add((cat_id, cat_name))
                        ann_copy = ann.copy()
                        ann_copy['category_id'] = cat_id
                        ann_copy['category'] = cat_name
                        all_annotations.append(ann_copy)
        categories_list = [{'id': cid, 'name': cname} for cid, cname in sorted(cat_id_set, key=lambda x: x[0])]
        coco_output = {
            'info': video_info,
            'images': [{'id': i, 'frame_idx': i} for i in range(total_frames)],
            'annotations': all_annotations,
            'categories': categories_list
        }

        with open(output_path / "annotations.json", 'w') as f:
            json.dump(coco_output, f)

        QMessageBox.information(self, "完成", f"数据已保存到 {output_path}")
        print(f"导出完成: {output_path}")

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

        with open(annotations_path) as f:
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

        print(f"正在生成视频: {output_path}")
        print(f"[DEBUG run_save] 已选颜色索引={self.selected_color_index}, 颜色={self.palette_colors[self.selected_color_index]}, 调色板长度={len(self.palette_colors)}")

        for i in range(total_frames):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            frame = cv2.imread(str(frame_path))

            if frame is None:
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            label_path = labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path) as f:
                    annotations = json.load(f)

                result_frame = frame.copy()
                overlay = frame.copy()

                for ann in annotations:
                    polygon = ann.get('segmentation')
                    bbox = ann.get('bbox')

                    if not bbox:
                        continue

                    cat_name = ann.get('category', 'Unknown')
                    conf = ann.get('confidence', 1.0)
                    track_id = ann.get('track_id', 0)
                    if track_id == 1000000:
                        color = self.palette_colors[self.selected_color_index]
                    else:
                        n_colors = len(self.palette_colors) - 1
                        color_idx_in_remapped = track_id % n_colors
                        selected_idx = self.selected_color_index
                        if color_idx_in_remapped < selected_idx:
                            color = self.palette_colors[color_idx_in_remapped]
                        else:
                            color = self.palette_colors[color_idx_in_remapped + 1]

                    if polygon:
                        pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)

                    x, y = int(bbox[0]), int(bbox[1])
                    w, h = int(bbox[2]), int(bbox[3])
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(overlay, f"{cat_name} {conf:.2f}", (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.addWeighted(overlay, self.ctrl.alpha, result_frame, 1 - self.ctrl.alpha, 0, result_frame)
                frame = result_frame

            out.write(frame)

        out.release()
        print(f"视频已保存: {output_path}")

        print("正在上传到OBS...")
        try:
            result = subprocess.run(
                ['curl', '--upload-file', str(output_path), f'http://obs.dimond.top/{output_name}'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("上传成功!")
                QMessageBox.information(self, "完成", f"视频已保存并上传!")
            else:
                print(f"上传失败: {result.stderr}")
                QMessageBox.warning(self, "部分完成", f"视频已保存，但上传失败")
        except Exception as e:
            print(f"上传失败: {e}")
            QMessageBox.warning(self, "错误", str(e))


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
