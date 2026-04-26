#!/usr/bin/env python3
"""视频标注工具 - 统一控制面板，控制逻辑委托给 video_control.VideoController"""

import sys
import shutil
import cv2
import numpy as np
import json
import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QLineEdit, QFileDialog, QGroupBox, QTextEdit, QMessageBox, QListWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.Qt import QDragEnterEvent, QDropEvent

from video_control import VideoController


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
        self.inject_process = None
        self.inject_timer = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('视频标注工具')
        self.setGeometry(100, 100, 500, 800)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        main_layout.addWidget(self.create_annotate_section())
        main_layout.addWidget(self.create_viewer_section())
        main_layout.addWidget(self.create_save_section())

    def create_annotate_section(self):
        group = QGroupBox("1. 视频标注 (annotate_video)")
        layout = QVBoxLayout()
        group.setLayout(layout)

        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("视频文件:"))
        self.video_input = DragLineEdit()
        video_layout.addWidget(self.video_input)
        select_btn = QPushButton("选择视频")
        select_btn.clicked.connect(self.select_video)
        video_layout.addWidget(select_btn)
        layout.addLayout(video_layout)

        iou_items_layout = QHBoxLayout()
        iou_items_layout.addWidget(QLabel("IoU:"))
        self.iou_input = QLineEdit("0.5")
        self.iou_input.setFixedWidth(60)
        iou_items_layout.addWidget(self.iou_input)
        iou_items_layout.addWidget(QLabel("物品:"))
        self.items_input = QLineEdit()
        self.items_input.setMinimumWidth(150)
        iou_items_layout.addWidget(self.items_input)
        layout.addLayout(iou_items_layout)

        self.annotate_btn = QPushButton("▶ 执行标注")
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

        src_dir = Path("src")
        src_dir.mkdir(exist_ok=True)

        video_name = Path(video_path).name

        if Path(video_path).parent.resolve() == src_dir.resolve():
            print(f"视频已在src目录，无需拷贝")
        else:
            dst_video = src_dir / video_name
            shutil.copy2(video_path, dst_video)
            print(f"已拷贝到src: {dst_video}")

        iou = self.iou_input.text() or "0.5"
        items_text = self.items_input.text()

        src_video = src_dir / video_name
        cmd = [sys.executable, 'annotate_video.py',
               '--iou', str(iou),
               '--src', str(src_video),
               '--items', items_text]

        sys.stderr.write(f"[DEBUG app] cmd={cmd}\n")
        sys.stderr.flush()

        self.video_process = subprocess.Popen(cmd, cwd=str(Path.cwd()))

    def create_viewer_section(self):
        group = QGroupBox("2. 预览 (video_viewer)")
        layout = QVBoxLayout()
        group.setLayout(layout)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("数据目录:"))
        self.path_input = QLineEdit("temp_data")
        path_layout.addWidget(self.path_input)
        open_btn = QPushButton("选择")
        open_btn.clicked.connect(self.select_data_dir)
        path_layout.addWidget(open_btn)
        show_btn = QPushButton("Show")
        show_btn.clicked.connect(self.show_viewer)
        path_layout.addWidget(show_btn)
        layout.addLayout(path_layout)

        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("类别名称:"))
        self.category_input = QLineEdit(self.ctrl.category_name)
        category_layout.addWidget(self.category_input)
        layout.addLayout(category_layout)

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

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(self.ctrl.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        conf_layout.addWidget(self.conf_slider)
        layout.addLayout(conf_layout)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(10)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(int(self.ctrl.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.on_alpha_change)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel(f"{int(self.ctrl.alpha * 100)}%")
        alpha_layout.addWidget(self.alpha_label)
        layout.addLayout(alpha_layout)

        frame_nav_play_layout = QHBoxLayout()

        self.backward_btn = QPushButton("◀")
        self.backward_btn.setFixedWidth(40)
        self.backward_btn.clicked.connect(self.toggle_backward)
        frame_nav_play_layout.addWidget(self.backward_btn)

        prev_btn = QPushButton("◀帧")
        prev_btn.setFixedWidth(50)
        prev_btn.clicked.connect(self.prev_frame)
        frame_nav_play_layout.addWidget(prev_btn)

        self.prompt_btn = QPushButton("设为提示帧")
        self.prompt_btn.setFixedWidth(80)
        self.prompt_btn.setStyleSheet("QPushButton { background-color: #FFA500; color: white; border: none; border-radius: 3px; } QPushButton:hover { background-color: #FF8C00; }")
        self.prompt_btn.clicked.connect(self.toggle_prompt_mode)
        frame_nav_play_layout.addWidget(self.prompt_btn)

        self.frame_label = QLabel("1/1")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setFixedWidth(50)
        frame_nav_play_layout.addWidget(self.frame_label)

        next_btn = QPushButton("帧▶")
        next_btn.setFixedWidth(50)
        next_btn.clicked.connect(self.next_frame)
        frame_nav_play_layout.addWidget(next_btn)

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.toggle_play)
        frame_nav_play_layout.addWidget(self.play_btn)

        layout.addLayout(frame_nav_play_layout)

        fence_group_layout = QHBoxLayout()
        self.fence_btns = []
        self.fence_clear_btns = []
        for i in range(3):
            fence_btn = QPushButton(f"围栏{i+1}")
            fence_btn.clicked.connect(lambda checked, idx=i: self.toggle_fence(idx))
            self.fence_btns.append(fence_btn)
            fence_group_layout.addWidget(fence_btn)

            clear_btn = QPushButton("清")
            clear_btn.setFixedWidth(30)
            clear_btn.clicked.connect(lambda checked, idx=i: self.clear_fence(idx))
            self.fence_clear_btns.append(clear_btn)
            fence_group_layout.addWidget(clear_btn)
        layout.addLayout(fence_group_layout)

        del_layout = QHBoxLayout()
        del_layout.addWidget(QLabel("绿点列表:"))
        self.track_id_list = QListWidget()
        del_layout.addWidget(self.track_id_list)

        del_btn_layout = QVBoxLayout()
        remove_btn = QPushButton("🗑️")
        remove_btn.setFixedSize(40, 40)
        remove_btn.setStyleSheet("QPushButton { background-color: #ff4444; color: white; border: none; border-radius: 5px; font-size: 20px; } QPushButton:hover { background-color: #cc0000; }")
        remove_btn.clicked.connect(self.remove_selected_track_id)
        del_btn_layout.addWidget(remove_btn)

        clear_del_btn = QPushButton("清空")
        clear_del_btn.clicked.connect(self.clear_track_id)
        del_btn_layout.addWidget(clear_del_btn)

        del_layout.addLayout(del_btn_layout)
        layout.addLayout(del_layout)

        self.export_btn = QPushButton("📦 导出到 temp_data_post")
        self.export_btn.clicked.connect(self.export_to_temp_data_post)
        layout.addWidget(self.export_btn)

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.on_frame_play)

        return group

    def create_save_section(self):
        group = QGroupBox("3. 导出视频 (save)")
        layout = QVBoxLayout()
        group.setLayout(layout)

        input_dir_name_layout = QHBoxLayout()
        input_dir_name_layout.addWidget(QLabel("输入:"))
        self.save_input_dir = QLineEdit("temp_data_post")
        self.save_input_dir.setFixedWidth(120)
        input_dir_name_layout.addWidget(self.save_input_dir)
        browse_btn = QPushButton("选择")
        browse_btn.clicked.connect(self.select_save_input_dir)
        input_dir_name_layout.addWidget(browse_btn)
        input_dir_name_layout.addWidget(QLabel("名称:"))
        self.save_output_name = QLineEdit("dst.mp4")
        self.save_output_name.setFixedWidth(100)
        input_dir_name_layout.addWidget(self.save_output_name)
        layout.addLayout(input_dir_name_layout)

        save_alpha_layout = QHBoxLayout()
        save_alpha_layout.addWidget(QLabel("透明度:"))
        self.save_alpha_slider = QSlider(Qt.Horizontal)
        self.save_alpha_slider.setMinimum(10)
        self.save_alpha_slider.setMaximum(100)
        self.save_alpha_slider.setValue(int(self.ctrl.alpha * 100))
        self.save_alpha_slider.valueChanged.connect(self.on_save_alpha_change)
        save_alpha_layout.addWidget(self.save_alpha_slider)
        self.save_alpha_label = QLabel(f"{int(self.ctrl.alpha * 100)}%")
        save_alpha_layout.addWidget(self.save_alpha_label)
        layout.addLayout(save_alpha_layout)

        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("标签:"))
        self.save_category = QLineEdit(self.ctrl.category_name)
        category_layout.addWidget(self.save_category)
        layout.addLayout(category_layout)

        self.save_btn = QPushButton("💾 保存视频并上传OBS")
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

    def toggle_play(self):
        self.is_backward = False
        self.backward_btn.setText("◀")
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶")
        else:
            self.play_timer.start(100)
            self.is_playing = True
            self.play_btn.setText("⏸")

    def toggle_backward(self):
        self.is_playing = False
        self.play_btn.setText("▶")
        if self.is_backward:
            self.play_timer.stop()
            self.is_backward = False
            self.backward_btn.setText("◀")
        else:
            self.play_timer.start(100)
            self.is_backward = True
            self.backward_btn.setText("⏸")

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
            self.do_inject()

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

    def do_inject(self):
        prompt_bboxes = self.viewer.get_prompt_bboxes()
        if not prompt_bboxes:
            QMessageBox.warning(self, "错误", "请先绘制至少一个 Bbox")
            self.reset_prompt_btn()
            return

        labels_path = self.temp_data_path / "labels" / f"frame_{self.prompt_frame_idx:06d}.json"
        existing_bboxes = []
        if labels_path.exists():
            with open(labels_path) as f:
                frame_anns = json.load(f)
            for ann in frame_anns:
                b = ann.get('bbox')
                if b:
                    existing_bboxes.append([int(b[0]), int(b[1]),
                                            int(b[0] + b[2]), int(b[1] + b[3])])

        all_prompts = existing_bboxes + prompt_bboxes
        if not all_prompts:
            QMessageBox.warning(self, "错误", "该帧没有现有 bbox 且未绘制新 bbox")
            self.reset_prompt_btn()
            return

        inject_temp_dir = Path("temp_inject")
        if inject_temp_dir.exists():
            import shutil
            shutil.rmtree(inject_temp_dir)
        inject_temp_dir.mkdir(parents=True, exist_ok=True)

        clip_path = str(inject_temp_dir / "clip.mp4")
        try:
            self.extract_video_clip_from_frames(
                self.temp_data_path / "frames",
                self.prompt_frame_idx,
                self.total_frames,
                clip_path
            )
        except Exception as e:
            QMessageBox.warning(self, "错误", f"提取视频片段失败: {e}")
            self.reset_prompt_btn()
            return

        inject_temp_data = str(inject_temp_dir / "temp_data")
        cmd = [sys.executable, 'annotate_video.py',
               '--inject',
               '--src', clip_path,
               '--iou', self.iou_input.text() or "0.5",
               '--prompt-bboxes', json.dumps(all_prompts),
               '--output-temp', inject_temp_data]
        items_text = self.items_input.text()
        if items_text:
            cmd.extend(['--items', items_text])

        print(f"启动注入进程: {cmd}")
        self.inject_process = subprocess.Popen(cmd, cwd=str(Path.cwd()))
        self.inject_timer = QTimer()
        self.inject_timer.timeout.connect(self.check_inject_done)
        self.inject_timer.start(2000)

    def check_inject_done(self):
        if self.inject_process is None:
            return
        ret = self.inject_process.poll()
        if ret is not None:
            self.inject_timer.stop()
            if ret == 0:
                self.merge_inject_results()
            else:
                QMessageBox.warning(self, "错误", f"注入进程退出码: {ret}")
                self.reset_prompt_btn()

    def merge_inject_results(self):
        inject_temp_dir = Path("temp_inject")
        inject_data_dir = inject_temp_dir / "temp_data"
        if not inject_data_dir.exists():
            QMessageBox.warning(self, "错误", "注入结果目录不存在")
            self.reset_prompt_btn()
            return

        inject_frames_dir = inject_data_dir / "frames"
        inject_labels_dir = inject_data_dir / "labels"
        inject_annotations_file = inject_data_dir / "annotations.json"

        if not inject_annotations_file.exists():
            QMessageBox.warning(self, "错误", "注入结果 annotations.json 不存在")
            self.reset_prompt_btn()
            return

        frames_dir = self.temp_data_path / "frames"
        labels_dir = self.temp_data_path / "labels"
        annotations_file = self.temp_data_path / "annotations.json"

        for i in range(self.prompt_frame_idx, self.total_frames):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            label_path = labels_dir / f"frame_{i:06d}.json"
            if frame_path.exists():
                frame_path.unlink()
            if label_path.exists():
                label_path.unlink()

        inject_total = len(list(inject_frames_dir.glob("frame_*.jpg")))
        for i in range(inject_total):
            src_frame = inject_frames_dir / f"frame_{i:06d}.jpg"
            src_label = inject_labels_dir / f"frame_{i:06d}.json"
            dst_frame = frames_dir / f"frame_{i + self.prompt_frame_idx:06d}.jpg"
            dst_label = labels_dir / f"frame_{i + self.prompt_frame_idx:06d}.json"
            if src_frame.exists():
                import shutil
                shutil.copy2(src_frame, dst_frame)
            if src_label.exists():
                shutil.copy2(src_label, dst_label)

        with open(annotations_file) as f:
            original_coco = json.load(f)
        with open(inject_annotations_file) as f:
            inject_coco = json.load(f)

        original_images = [img for img in original_coco.get('images', [])
                          if img['id'] < self.prompt_frame_idx]
        original_annotations = [ann for ann in original_coco.get('annotations', [])
                               if ann['image_id'] < self.prompt_frame_idx]

        offset = self.prompt_frame_idx
        max_ann_id = max([ann['id'] for ann in original_annotations], default=-1) + 1
        for img in inject_coco.get('images', []):
            new_img = dict(img)
            new_img['id'] = img['id'] + offset
            original_images.append(new_img)
        for ann in inject_coco.get('annotations', []):
            new_ann = dict(ann)
            new_ann['id'] = ann['id'] + max_ann_id
            new_ann['image_id'] = ann['image_id'] + offset
            original_annotations.append(new_ann)

        original_coco['images'] = original_images
        original_coco['annotations'] = original_annotations
        with open(annotations_file, 'w') as f:
            json.dump(original_coco, f)

        self.total_frames = len(original_images)

        if self.viewer:
            self.viewer.coco_data = original_coco
            self.viewer.total_frames = self.total_frames
            self.viewer.go_to_frame(self.prompt_frame_idx)
            self.frame_label.setText(f"{self.prompt_frame_idx + 1}/{self.total_frames}")
            self.viewer.clear_prompt_bboxes()

        import shutil
        shutil.rmtree(inject_temp_dir)
        print(f"注入完成！从帧 {self.prompt_frame_idx + 1} 开始已覆盖 {inject_total} 帧")
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
        self.ctrl.clear_track_id_points()
        self.track_id_list.clear()
        if self.viewer:
            self.viewer.update_display()

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
        _, annotations = self.viewer.load_frame_data(frame_idx)
        filtered = self.ctrl.filter_annotations(annotations)
        found = self.ctrl.find_annotation_at(filtered, video_x, video_y)

        if found is not None:
            track_id = found.get('track_id', found.get('id', 0))
            self.ctrl.add_track_id_point(video_x, video_y, frame_idx, track_id)
            self.track_id_list.addItem(f"绿点 {len(self.ctrl.track_id_points)} 帧{frame_idx+1} ({video_x},{video_y}) ID:{track_id}→9999")
            self._convert_track_id_to_9999(track_id)
            self.viewer.update_display()

    def _convert_track_id_to_9999(self, old_track_id):
        self.ctrl.track_ids_to_9999.add(old_track_id)
        labels_dir = self.temp_data_path / "labels"
        converted_count = 0
        for label_file in sorted(labels_dir.glob("frame_*.json")):
            with open(label_file) as f:
                frame_anns = json.load(f)
            changed = False
            for ann in frame_anns:
                if ann.get('track_id') == old_track_id:
                    ann['track_id'] = 9999
                    changed = True
            if changed:
                with open(label_file, 'w') as f:
                    json.dump(frame_anns, f)
                converted_count += 1
        print(f"已将 track_id={old_track_id} → 9999，共影响 {converted_count} 帧")

    def select_data_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据目录", ".")
        if folder:
            self.path_input.setText(folder)
            self.temp_data_path = Path(folder)

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
        self.viewer.show()
        self.viewer.update_display()

        self.frame_label.setText(f"帧: 1/{self.total_frames}")

    def select_save_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入目录", ".")
        if folder:
            self.save_input_dir.setText(folder)

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

        category_name = self.category_input.text() or self.ctrl.category_name
        self.ctrl.category_name = category_name

        labels_dir = data_dir / "labels"
        frames_dir = data_dir / "frames"

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

                frame_anns = []
                for ann in filtered:
                    ann_copy = ann.copy()
                    ann_copy['category'] = category_name
                    frame_anns.append(ann_copy)

                with open(output_label_path, 'w') as f:
                    json.dump(frame_anns, f)

        all_annotations = self.ctrl.export_filtered_annotations(total_frames, labels_dir, category_name)
        coco_output = {
            'info': video_info,
            'images': [{'id': i, 'frame_idx': i} for i in range(total_frames)],
            'annotations': all_annotations
        }

        with open(output_path / "annotations.json", 'w') as f:
            json.dump(coco_output, f)

        QMessageBox.information(self, "完成", f"数据已保存到 {output_path}")
        print(f"导出完成: {output_path}")

    def run_save(self):
        input_dir = self.save_input_dir.text() or "temp_data_post"
        output_name = self.save_output_name.text() or "dst.mp4"
        category = self.save_category.text() or self.ctrl.category_name

        output_path = Path("dst") / output_name
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
                    cat = ann.get('category', category)
                    conf = ann.get('confidence', 1.0)

                    if polygon:
                        pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(overlay, [pts], color)
                        cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)

                    x, y = int(bbox[0]), int(bbox[1])
                    w, h = int(bbox[2]), int(bbox[3])
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(overlay, f"{cat} {conf:.2f}", (x, y - 5),
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
    parser.add_argument('--items', type=str, default=None, help='物品列表，逗号分隔')
    args = parser.parse_args()

    if args.src:
        cmd = [sys.executable, 'annotate_video.py',
               '--src', args.src]
        if args.iou is not None:
            cmd.extend(['--iou', str(args.iou)])
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
