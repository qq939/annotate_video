#!/usr/bin/env python3
"""视频标注工具 - 统一控制面板"""

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
        self.temp_data_path = Path("temp_data")
        self.viewer = None
        self.video_process = None
        self.alpha = 0.5
        self.conf_threshold = 0.5
        
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
        
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU阈值:"))
        self.iou_input = QLineEdit("0.5")
        iou_layout.addWidget(self.iou_input)
        layout.addLayout(iou_layout)
        
        items_layout = QHBoxLayout()
        items_layout.addWidget(QLabel("物品(逗号分隔):"))
        self.items_input = QLineEdit()
        items_layout.addWidget(self.items_input)
        layout.addLayout(items_layout)
        
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

        self.frame_label = QLabel("帧: 1/1")
        layout.addWidget(self.frame_label)

        frame_nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◀上一帧")
        prev_btn.clicked.connect(self.prev_frame)
        frame_nav_layout.addWidget(prev_btn)
        next_btn = QPushButton("下一帧▶")
        next_btn.clicked.connect(self.next_frame)
        frame_nav_layout.addWidget(next_btn)
        layout.addLayout(frame_nav_layout)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self.on_conf_change)
        conf_layout.addWidget(self.conf_slider)
        layout.addLayout(conf_layout)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(10)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.on_alpha_change)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("50%")
        alpha_layout.addWidget(self.alpha_label)
        layout.addLayout(alpha_layout)

        self.fence_btns = []
        self.fence_clear_btns = []
        for i in range(3):
            fence_layout = QHBoxLayout()
            self.fence_btns.append(QPushButton(f"围栏{i+1}绘制"))
            self.fence_btns[i].clicked.connect(lambda checked, idx=i: self.toggle_fence_mode(idx))
            fence_layout.addWidget(self.fence_btns[i])
            self.fence_clear_btns.append(QPushButton("清除"))
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
        self.fences = []
        self.del_points = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next)

        del_layout = QHBoxLayout()
        del_layout.addWidget(QLabel("删除列表:"))
        self.del_list = QListWidget()
        del_layout.addWidget(self.del_list)
        remove_btn = QPushButton("🗑️")
        remove_btn.setFixedSize(40, 40)
        remove_btn.setStyleSheet("QPushButton { background-color: #ff4444; color: white; border: none; border-radius: 5px; font-size: 20px; } QPushButton:hover { background-color: #cc0000; }")
        remove_btn.clicked.connect(self.remove_selected_del_point)
        del_layout.addWidget(remove_btn)
        layout.addLayout(del_layout)

        export_btn = QPushButton("📦 导出到 temp_data_post")
        export_btn.clicked.connect(self.export_to_temp_data_post)
        layout.addWidget(export_btn)

        return group
    
    def create_save_section(self):
        group = QGroupBox("3. 导出视频 (save)")
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入目录:"))
        self.save_input_dir = QLineEdit("temp_data_post")
        input_layout.addWidget(self.save_input_dir)
        browse_btn = QPushButton("选择")
        browse_btn.clicked.connect(self.select_save_input_dir)
        input_layout.addWidget(browse_btn)
        layout.addLayout(input_layout)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("视频名称:"))
        self.save_output_name = QLineEdit("dst.mp4")
        output_layout.addWidget(self.save_output_name)
        layout.addLayout(output_layout)
        
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("透明度:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(10)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.on_alpha_change)
        alpha_layout.addWidget(self.alpha_slider)
        self.alpha_label = QLabel("50%")
        alpha_layout.addWidget(self.alpha_label)
        layout.addLayout(alpha_layout)
        
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("标签:"))
        self.save_category = QLineEdit("Detect")
        category_layout.addWidget(self.save_category)
        layout.addLayout(category_layout)
        
        self.save_btn = QPushButton("💾 保存视频并上传OBS")
        self.save_btn.clicked.connect(self.run_save)
        layout.addWidget(self.save_btn)
        
        return group
    
    def select_data_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据目录", ".")
        if folder:
            self.path_input.setText(folder)

    def show_viewer(self):
        from video_viewer import VideoViewer
        self.temp_data_path = Path(self.path_input.text())
        if not self.temp_data_path.exists():
            QMessageBox.warning(self, "错误", "数据目录不存在")
            return
        if not (self.temp_data_path / "annotations.json").exists():
            QMessageBox.warning(self, "错误", "annotations.json 不存在")
            return
        self.viewer = VideoViewer(str(self.temp_data_path))
        self.viewer.show()

    def on_conf_change(self, value):
        self.conf_threshold = value / 100.0
        if self.viewer:
            self.viewer.conf_threshold = self.conf_threshold
            self.viewer.update_display()

    def toggle_fence_mode(self, fence_idx):
        if fence_idx >= len(self.fences):
            self.fences.append({'points': [], 'mode': True})
        else:
            fence = self.fences[fence_idx]
            fence['mode'] = not fence['mode']
        self.update_fence_buttons()
        if self.viewer:
            self.viewer.fences = self.fences
            self.viewer.update_display()

    def clear_fence(self, fence_idx):
        if fence_idx < len(self.fences):
            self.fences[fence_idx] = {'points': [], 'mode': False}
        self.update_fence_buttons()
        if self.viewer:
            self.viewer.fences = self.fences
            self.viewer.update_display()

    def update_fence_buttons(self):
        for i, btn in enumerate(self.fence_btns):
            if i < len(self.fences) and self.fences[i]['mode']:
                btn.setStyleSheet("background-color: #00ff00; color: black;")
                btn.setText(f"围栏{i+1}完成")
            else:
                btn.setStyleSheet("")
                btn.setText(f"围栏{i+1}绘制")

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
        if self.viewer:
            self.viewer.is_backward = self.is_backward
            self.viewer.play_next()
            self.frame_label.setText(f"帧: {self.viewer.get_current_frame()+1}/{self.viewer.total_frames}")

    def prev_frame(self):
        if self.viewer:
            idx = (self.viewer.get_current_frame() - 1) % self.viewer.total_frames
            self.viewer.go_to_frame(idx)
            self.frame_label.setText(f"帧: {idx+1}/{self.viewer.total_frames}")

    def next_frame(self):
        if self.viewer:
            idx = (self.viewer.get_current_frame() + 1) % self.viewer.total_frames
            self.viewer.go_to_frame(idx)
            self.frame_label.setText(f"帧: {idx+1}/{self.viewer.total_frames}")

    def remove_selected_del_point(self):
        for item in self.del_list.selectedItems():
            row = self.del_list.row(item)
            self.del_list.takeItem(row)
            if row < len(self.del_points):
                self.del_points.pop(row)
        if self.viewer:
            self.viewer.del_points = self.del_points
            self.viewer.update_display()
    
    def select_save_input_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输入目录", ".")
        if folder:
            self.save_input_dir.setText(folder)

    def on_alpha_change(self, value):
        self.alpha = value / 100.0
        self.alpha_label.setText(f"{value}%")
        if self.viewer:
            self.viewer.alpha = self.alpha
            self.viewer.update_display()
    
    def export_to_temp_data_post(self):
        print("导出到 temp_data_post...")
        QMessageBox.information(self, "提示", "请在终端运行: python video_viewer.py 进行导出")
    
    def run_save(self):
        input_dir = self.save_input_dir.text() or "temp_data_post"
        output_name = self.save_output_name.text() or "dst.mp4"
        alpha = self.alpha
        category = self.save_category.text() or "Detect"
        
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
        
        mask_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
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
                
                cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
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
        # 命令行模式：跳过GUI，直接执行标注
        cmd = [sys.executable, 'annotate_video.py',
               '--src', args.src]
        if args.iou is not None:
            cmd.extend(['--iou', str(args.iou)])
        if args.items:
            cmd.extend(['--items', args.items])
        subprocess.Popen(cmd, cwd=str(Path.cwd()))
        return

    # GUI模式
    app = QApplication(sys.argv)
    panel = UnifiedPanel()
    panel.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
