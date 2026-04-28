#!/usr/bin/env python3
"""控制面板 - 三道过滤，控制逻辑委托给 video_control.VideoController"""

import sys
import cv2
import numpy as np
import json
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QMessageBox, QListWidget, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt, QTimer

from video_control import VideoController


class ControlPanel(QMainWindow):
    def __init__(self, temp_data_path=None, controller=None):
        super().__init__()
        self.temp_data_path = Path(temp_data_path) if temp_data_path else Path("temp_data")
        self.ctrl = controller if controller is not None else VideoController()
        self.viewer = None

        with open(self.temp_data_path / "annotations.json") as f:
            self.coco_data = json.load(f)
        self.video_info = self.coco_data['info']

        self.total_frames = len(self.coco_data['images'])
        self.max_fences = 3

        self.init_ui()

    def set_viewer(self, viewer):
        self.viewer = viewer
        self.viewer.video_clicked.connect(self.handle_click)
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
        self.conf_slider.setValue(int(self.ctrl.conf_threshold * 100))
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
        self.category_input = QLineEdit(self.ctrl.category_name)
        category_layout.addWidget(self.category_input)
        layout.addLayout(category_layout)

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

        clear_btn = QPushButton("清空绿点")
        clear_btn.clicked.connect(self.clear_track_id_points)
        layout.addWidget(clear_btn)

        layout.addWidget(QLabel("绿点列表:"))
        del_list_layout = QHBoxLayout()
        self.track_id_list = QListWidget()
        del_list_layout.addWidget(self.track_id_list)

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
        remove_btn.clicked.connect(self.remove_selected_track_id_point)
        del_btn_layout.addWidget(remove_btn)
        del_list_layout.addLayout(del_btn_layout)
        layout.addLayout(del_list_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)

    def toggle_fence_mode(self, fence_idx):
        self.ctrl.toggle_fence_mode(fence_idx)
        self.update_fence_buttons()
        if self.viewer:
            self.viewer.update_display()

    def clear_fence(self, fence_idx):
        self.ctrl.clear_fence(fence_idx)
        self.update_fence_buttons()
        if self.viewer:
            self.viewer.update_display()

    def update_fence_buttons(self):
        for i, btn in enumerate(self.fence_btns):
            if i < len(self.ctrl.fences) and self.ctrl.fences[i].get('mode', False):
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

    def on_conf_change(self, value):
        self.ctrl.conf_threshold = value / 100.0
        if self.viewer:
            self.viewer.update_display()

    def on_alpha_change(self, value):
        self.ctrl.alpha = value / 100.0
        self.alpha_label.setText(f"{value}%")
        if self.viewer:
            self.viewer.update_display()

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
        if not self.viewer:
            return
        if self.is_backward:
            idx = (self.viewer.get_current_frame() - 1) % self.total_frames
            self.viewer.go_to_frame(idx)
        else:
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

    def handle_click(self, video_x, video_y, frame_idx):
        print(f"点击: ({video_x}, {video_y}), 帧{frame_idx+1}")

        if self.ctrl.fence_mode_active():
            for i, fence in enumerate(self.ctrl.fences):
                if fence.get('mode', False):
                    self.ctrl.add_fence_point(i, (video_x, video_y))
                    print(f"围栏{i+1}点: {len(fence['points'])}")
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
            assigned_id = self.ctrl.next_track_id
            self.ctrl.add_track_id_point(video_x, video_y, frame_idx, track_id)
            pt = self.ctrl.track_id_points[-1]
            pt['assigned_id'] = assigned_id
            self.ctrl.track_ids_to_9999.add(track_id)
            self.ctrl.assigned_to_original[assigned_id] = track_id
            self._convert_track_id(track_id, assigned_id)
            idx = len(self.ctrl.track_id_points) - 1
            self.track_id_list.addItem(f"绿点 {idx} 帧{frame_idx+1} ({video_x},{video_y}) ID:{track_id}→{assigned_id}")
            if self.viewer:
                self.viewer.update_display()
        else:
            print("未找到标注")

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

    def remove_selected_track_id_point(self):
        current_row = self.track_id_list.currentRow()
        if current_row >= 0:
            pt = self.ctrl.track_id_points[current_row]
            if pt['assigned_id'] is not None:
                self._convert_track_id(pt['assigned_id'], pt['track_id'])
                self.ctrl.assigned_to_original.pop(pt['assigned_id'], None)
            self.ctrl.remove_track_id_point(current_row)
            self.track_id_list.takeItem(current_row)
            self._refresh_list()
            if self.viewer:
                self.viewer.update_display()

    def _refresh_list(self):
        self.track_id_list.clear()
        for i, pt in enumerate(self.ctrl.track_id_points):
            if pt.get('assigned_id') is not None:
                self.track_id_list.addItem(
                    f"绿点 {i} 帧{pt['frame_idx']+1} ({pt['x']},{pt['y']}) ID:{pt['track_id']}→{pt['assigned_id']}"
                )
            else:
                self.track_id_list.addItem(
                    f"绿点 {i} 帧{pt['frame_idx']+1} ({pt['x']},{pt['y']}) ID:{pt['track_id']}"
                )

    def update_del_list(self):
        self._refresh_list()

    def clear_track_id_points(self):
        for pt in self.ctrl.track_id_points:
            if pt.get('assigned_id') is not None:
                self._convert_track_id(pt['assigned_id'], pt['track_id'])
                self.ctrl.assigned_to_original.pop(pt['assigned_id'], None)
        self.ctrl.clear_track_id_points()
        self.track_id_list.clear()
        if self.viewer:
            self.viewer.update_display()

    def filter_annotations(self, annotations):
        return self.ctrl.filter_annotations(annotations)

    def apply_threshold_to_masks(self, frame, annotations, threshold=None):
        return self.ctrl.apply_threshold_to_masks(frame, annotations)

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

        category_name = self.category_input.text() or self.ctrl.category_name
        self.ctrl.category_name = category_name

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

                filtered = self.ctrl.filter_annotations(annotations)

                frame_anns = []
                for ann in filtered:
                    ann_copy = ann.copy()
                    ann_copy['category'] = category_name
                    frame_anns.append(ann_copy)

                with open(output_label_path, 'w') as f:
                    json.dump(frame_anns, f)

        all_annotations = self.ctrl.export_filtered_annotations(self.total_frames, labels_dir, category_name)
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
        return self.ctrl.apply_threshold_to_masks(frame, annotations)


def main():
    app = QApplication(sys.argv)
    panel = ControlPanel("temp_data")
    panel.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
