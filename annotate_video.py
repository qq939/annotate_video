# global参数
FIND = []  # 第1行：文本提示词列表，运行时由用户输入，用于SAM3语义分割
SRC_DIR = "src"  # 第31行：视频源目录
DST_DIR = "dst"  # 第67行：输出视频目录
TEMP_DATA_DIR = "temp_data"  # 第8行：临时数据目录，用于保存每帧画面和COCO格式标注
IOU_THRESHOLD = 0.5  # 第10行：IoU阈值，默认0.5，用于目标跟踪匹配
WINDOW_NAME = "视频标注工具"  # 第37行：窗口名称
SAM_MODEL_PATH = "sam3.pt"  # SAM模型路径（可下载sam_b.pt或sam3.pt）
BOX_COLORS = [  # 第55行：标注框颜色列表
    (255, 0, 0),      # 蓝色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 红色
    (255, 255, 0),    # 青色
    (255, 0, 255),    # 紫色
    (0, 255, 255),    # 黄色
    (255, 128, 0),    # 橙色
    (128, 0, 255),    # 紫红色
]

import cv2
import numpy as np
import subprocess
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """在图像上绘制中文文本（使用UTF-8编码）"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", font_size)
        except:
            font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def calculate_bbox_iou(bbox1, bbox2):
    """计算两个bbox的IoU"""
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2

    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def calculate_mask_iou(mask1, mask2):
    """计算两个mask的IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)

class TrackManager:
    """跟踪管理器，用于保持目标ID在不同帧间的一致性"""
    def __init__(self, iou_threshold=0.5):
        self.track_id_counter = 0
        self.tracked_objects = {}  # track_id -> {mask, bbox, last_seen}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold

    def update(self, masks, bboxes, frame_idx):
        """更新跟踪，返回每个mask对应的track_id"""
        if len(masks) == 0:
            return []

        track_ids = []

        if len(self.tracked_objects) == 0:
            for i in range(len(masks)):
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracked_objects[track_id] = {
                    'mask': masks[i],
                    'bbox': bboxes[i] if i < len(bboxes) else None,
                    'last_seen': frame_idx
                }
                track_ids.append(track_id)
            return track_ids

        if len(bboxes) < len(masks):
            bboxes = bboxes + [None] * (len(masks) - len(bboxes))

        for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
            best_iou = 0
            best_track_id = None

            for track_id, obj in self.tracked_objects.items():
                if obj['mask'] is not None:
                    iou = calculate_mask_iou(mask, obj['mask'])
                elif obj['bbox'] is not None and bbox is not None:
                    iou = calculate_bbox_iou(bbox, obj['bbox'])
                else:
                    iou = 0

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou >= self.iou_threshold:
                track_ids.append(best_track_id)
                self.tracked_objects[best_track_id] = {
                    'mask': mask,
                    'bbox': bbox,
                    'last_seen': frame_idx
                }
            else:
                track_id = self.next_track_id
                self.next_track_id += 1
                track_ids.append(track_id)
                self.tracked_objects[track_id] = {
                    'mask': mask,
                    'bbox': bbox,
                    'last_seen': frame_idx
                }

        for track_id in list(self.tracked_objects.keys()):
            if self.tracked_objects[track_id]['last_seen'] < frame_idx - 30:
                del self.tracked_objects[track_id]

        return track_ids

def upload_to_obs(file_path: str):
    """上传文件到OBS云存储"""
    obs_url = f"http://obs.dimond.top/{Path(file_path).name}"
    try:
        print(f"正在上传文件到OBS: {obs_url}")
        result = subprocess.run(
            ["curl", "--upload-file", file_path, obs_url],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print(f"✓ 文件上传成功: {obs_url}")
            return True
        else:
            print(f"✗ 文件上传失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 文件上传超时")
        return False
    except Exception as e:
        print(f"✗ 上传出错: {e}")
        return False

def get_video_extension(video_path: str) -> str:
    """获取原视频的扩展名"""
    return Path(video_path).suffix.lower()

def get_output_filename(video_path: str) -> str:
    """生成输出文件名，避免双点"""
    video_name = Path(video_path).stem
    video_ext = Path(video_path).suffix.lower()

    video_name = video_name.replace("..", "_")

    while "__" in video_name:
        video_name = video_name.replace("__", "_")

    if video_name.endswith("_"):
        video_name = video_name[:-1]

    return f"{video_name}_annotated{video_ext}"

def get_device():
    """自动检测并返回可用的计算设备"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ 检测到NVIDIA GPU: {gpu_name}")
            # CUDA优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            return '0', 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ 检测到Apple MPS GPU")
            # MPS优化
            torch.backends.mps.empty_cache()
            return 'mps', 'mps'
        else:
            print("✓ 未检测到GPU，使用CPU")
            return 'cpu', 'cpu'
    except Exception as e:
        print(f"⚠ GPU检测失败: {e}，使用CPU")
        return 'cpu', 'cpu'

@dataclass
class AnnotationBox:
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int]
    mask: np.ndarray = None

    def normalize(self):
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    def to_bbox_mask(self, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, y1 = max(0, self.x1), max(0, self.y1)
        x2, y2 = min(width, self.x2), min(height, self.y2)
        mask[y1:y2, x1:x2] = 255
        return mask

    def apply_mask_to_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = None) -> np.ndarray:
        if self.mask is not None:
            mask = self.mask
        else:
            mask = self.to_bbox_mask(frame.shape[0], frame.shape[1])

        if color is None:
            color = self.color
        colored_mask = np.zeros_like(frame)
        colored_mask[:] = color
        frame_with_box = frame.copy()
        mask_bool = mask > 0
        frame_with_box[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 0.5,
            colored_mask[mask_bool], 0.5, 0
        )
        return frame_with_box

    def apply_sam_mask_to_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = None) -> np.ndarray:
        if self.mask is None:
            return self.apply_mask_to_frame(frame, color)

        if color is None:
            color = self.color

        mask = self.mask
        colored_mask = np.zeros_like(frame)
        colored_mask[:] = color
        frame_with_mask = frame.copy()
        mask_bool = mask > 0
        frame_with_mask[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 0.3,
            colored_mask[mask_bool], 0.7, 0
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_with_mask, contours, -1, color, 2)

        return frame_with_mask

class VideoAnnotator:
    def __init__(self, video_path: str, output_dir: str):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("无法读取视频帧")

        self.boxes: List[AnnotationBox] = []
        self.current_box: AnnotationBox = None
        self.drawing = False
        self.start_point = None
        self.color_index = 0
        self.button_clicked = False

        self.window_name = WINDOW_NAME
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, self)

    def mouse_callback(self, event, x, y, flags, param):
        annotator = param

        if event == cv2.EVENT_LBUTTONDOWN:
            annotator.drawing = True
            annotator.start_point = (x, y)
            color = BOX_COLORS[annotator.color_index % len(BOX_COLORS)]
            annotator.current_box = AnnotationBox(x, y, x, y, color)

        elif event == cv2.EVENT_MOUSEMOVE:
            if annotator.drawing and annotator.current_box:
                annotator.current_box.x2 = x
                annotator.current_box.y2 = y

        elif event == cv2.EVENT_LBUTTONUP:
            button_x1 = annotator.frame.shape[1] - 150
            button_x2 = button_x1 + 130
            button_y1 = 30
            button_y2 = button_y1 + 40

            if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
                annotator.button_clicked = True
                print("点击了完成按钮")
            elif annotator.drawing and annotator.current_box:
                annotator.current_box.x2 = x
                annotator.current_box.y2 = y
                annotator.current_box.normalize()
                annotator.boxes.append(annotator.current_box)
                annotator.color_index += 1
                annotator.drawing = False
                annotator.current_box = None
            else:
                annotator.drawing = False
                annotator.current_box = None

    def draw_boxes(self, frame: np.ndarray) -> np.ndarray:
        display_frame = frame.copy()

        for box in self.boxes:
            cv2.rectangle(display_frame,
                         (box.x1, box.y1),
                         (box.x2, box.y2),
                         box.color, 2)
            label = f"目标 {self.boxes.index(box) + 1}"
            display_frame = put_chinese_text(display_frame, label,
                                            (box.x1, box.y1 - 10),
                                            font_size=15, color=box.color)

        if self.current_box and self.drawing:
            cv2.rectangle(display_frame,
                         (self.current_box.x1, self.current_box.y1),
                         (self.current_box.x2, self.current_box.y2),
                         self.current_box.color, 2)

        return display_frame

    def add_complete_button(self, frame: np.ndarray) -> np.ndarray:
        button_text = "完成标注"
        button_pos = (frame.shape[1] - 150, 30)
        button_size = (130, 40)

        cv2.rectangle(frame,
                     (button_pos[0], button_pos[1]),
                     (button_pos[0] + button_size[0], button_pos[1] + button_size[1]),
                     (0, 255, 0), -1)
        frame = put_chinese_text(frame, button_text,
                                (button_pos[0] + 15, button_pos[1] + 8),
                                font_size=18, color=(255, 255, 255))

        return frame

    def show_instructions(self, frame: np.ndarray) -> np.ndarray:
        instructions = [
            "操作说明:",
            "1. 鼠标左键框选目标",
            "2. 可框选多个目标",
            "3. 按 'c' 撤销最后一个框",
            "4. 按 'q' 退出",
            "5. 点击绿色按钮完成标注"
        ]

        for i, text in enumerate(instructions):
            frame = put_chinese_text(frame, text,
                                    (10, 30 + i * 25),
                                    font_size=16, color=(255, 255, 255))

        return frame

    def launch_control_panel(self, output_path):
        print("\n" + "=" * 50)
        print("正在启动控制面板...")
        print("=" * 50)
        try:
            import sys as _sys
            script_dir = Path(_sys.argv[0] if _sys.argv else __file__).parent.resolve()
            venv_python = script_dir / '.venv' / 'bin' / 'python'
            control_panel_script = script_dir / 'control_panel.py'
            
            if venv_python.exists():
                subprocess.Popen([str(venv_python), str(control_panel_script)])
                print("✓ 控制面板已启动（使用虚拟环境）")
            else:
                subprocess.Popen(['python3', str(control_panel_script)])
                print("✓ 控制面板已启动（使用系统Python）")
        except Exception as e:
            print(f"✗ 启动控制面板失败: {e}")
            print("您可以稍后手动运行: .venv/bin/python video_viewer.py")
        print("=" * 50)

    def run(self):
        while True:
            display_frame = self.draw_boxes(self.frame)
            display_frame = self.add_complete_button(display_frame)
            display_frame = self.show_instructions(display_frame)

            cv2.imshow(self.window_name, display_frame)

            if self.button_clicked:
                self.process_video()
                break

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('c'):
                if self.boxes:
                    removed = self.boxes.pop()
                    self.color_index = max(0, self.color_index - 1)
                    print(f"已撤销: {removed}")
                else:
                    print("没有可撤销的标注框")

        cv2.destroyAllWindows()
        self.cap.release()

    def process_video(self):
        if not self.boxes and not FIND:
            print("错误：文字和标注框至少要有一个！")
            print("请重新运行程序并添加物品名称或绘制标注框")
            return

        bboxes = [[box.x1, box.y1, box.x2, box.y2] for box in self.boxes] if self.boxes else None

        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
            print("正在加载SAM3视频分割模型...")

            device, device_type = get_device()

            # 根据设备类型优化
            half = device_type == 'cuda'
            
            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=SAM_MODEL_PATH,
                device=device,
                half=half,
                save=False,
                verbose=False
            )
            # 根据设备类型优化
            if device_type == 'cuda':
                # CUDA优化：启用batch和stream
                overrides['batch'] = 1
                overrides['stream'] = False
                overrides['stream_buffer'] = False
            elif device_type == 'mps':
                # MPS优化：启用半精度
                overrides['half'] = True
                overrides['amp'] = True
            
            predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            print(f"SAM3视频模型加载成功: {SAM_MODEL_PATH}")

            print(f"正在使用SAM3进行视频实例分割跟踪...")
            if FIND:
                print(f"文本提示词: {FIND}")
            else:
                print("未提供文本提示词，将使用边界框进行分割")
            print(f"将跟踪 {len(self.boxes)} 个目标实例")

            output_filename = get_output_filename(self.video_path)
            output_path = self.output_dir / output_filename

            fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([
                chr(fourcc_int & 0xFF),
                chr((fourcc_int >> 8) & 0xFF),
                chr((fourcc_int >> 16) & 0xFF),
                chr((fourcc_int >> 24) & 0xFF)
            ])
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            temp_data_path = Path(TEMP_DATA_DIR)
            if temp_data_path.exists():
                import shutil
                shutil.rmtree(temp_data_path)
            temp_data_path.mkdir(parents=True, exist_ok=True)
            frames_dir = temp_data_path / "frames"
            labels_dir = temp_data_path / "labels"
            frames_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)

            coco_data = {
                'info': {
                    'description': 'Video Annotation Dataset',
                    'video_path': self.video_path,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'fourcc': fourcc_str,
                    'FIND': FIND if FIND else []
                },
                'images': [],
                'annotations': [],
                'categories': [{'id': i, 'name': f'object_{i}'} for i in range(len(self.boxes) if self.boxes else 8)]
            }

            annotation_id = 0

            track_manager = TrackManager(iou_threshold=0.5)
            print("✓ 已启用记忆跟踪功能")

            if bboxes:
                predictor_args = {
                    'source': self.video_path,
                    'bboxes': bboxes,
                    'labels': [1] * len(bboxes),
                    'stream': True
                }
                if FIND:
                    predictor_args['text'] = FIND
                results = predictor(**predictor_args)
            else:
                predictor_args = {
                    'source': self.video_path,
                    'stream': True
                }
                if FIND:
                    predictor_args['text'] = FIND
                results = predictor(**predictor_args)

            frame_count = 0
            print("正在生成标注视频...")
            for r in results:
                orig_img = r.orig_img if hasattr(r, 'orig_img') else None
                if orig_img is None and hasattr(r, 'orig_shape'):
                    orig_shape = r.orig_shape
                    cap_temp = cv2.VideoCapture(self.video_path)
                    cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret_temp, orig_img = cap_temp.read()
                    cap_temp.release()
                    if not ret_temp:
                        orig_img = np.zeros((height, width, 3), dtype=np.uint8)

                if orig_img is not None:
                    if len(orig_img.shape) == 2:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                    elif orig_img.shape[2] == 4:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), orig_img)

                image_info = {
                    'id': frame_count,
                    'file_name': f"frame_{frame_count:06d}.jpg",
                    'width': width,
                    'height': height,
                    'frame_count': frame_count
                }
                coco_data['images'].append(image_info)

                frame_annotations = []
                if hasattr(r, 'masks') and r.masks is not None:
                    masks_tensor = r.masks.data
                    if masks_tensor is not None and len(masks_tensor) > 0:
                        masks_array = masks_tensor.cpu().numpy()

                        confs = None
                        if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'conf'):
                            confs = r.boxes.conf.cpu().numpy()

                        current_masks = []
                        current_bboxes = []
                        for mask in masks_array:
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            for contour in contours:
                                if len(contour) >= 3:
                                    polygon = contour.squeeze().flatten().tolist()
                                    x_coords = polygon[0::2]
                                    y_coords = polygon[1::2]
                                    x_min, x_max = min(x_coords), max(x_coords)
                                    y_min, y_max = min(y_coords), max(y_coords)

                                    bbox = [float(x_min), float(y_min),
                                           float(x_max - x_min), float(y_max - y_min)]

                                    area = cv2.contourArea(contour)

                                    if area > 0:
                                        current_masks.append(mask_binary)
                                        current_bboxes.append(bbox)

                        if current_masks:
                            track_ids = track_manager.update(current_masks, current_bboxes, frame_count)

                            for idx, (mask, bbox) in enumerate(zip(current_masks, current_bboxes)):
                                mask_binary = (mask > 0.5).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                                for contour in contours:
                                    if len(contour) >= 3:
                                        polygon = contour.squeeze().flatten().tolist()
                                        area = cv2.contourArea(contour)

                                        track_id = track_ids[idx] if idx < len(track_ids) else annotation_id

                                        confidence = float(confs[idx]) if confs is not None and idx < len(confs) else float(mask.max())

                                        ann = {
                                            'id': annotation_id,
                                            'track_id': track_id,
                                            'image_id': frame_count,
                                            'category_id': track_id,
                                            'bbox': bbox,
                                            'area': float(area),
                                            'segmentation': [polygon],
                                            'iscrowd': 0,
                                            'confidence': confidence
                                        }
                                        coco_data['annotations'].append(ann)
                                        frame_annotations.append(ann)
                                        annotation_id += 1

                with open(labels_dir / f"frame_{frame_count:06d}.json", 'w') as f:
                    json.dump(frame_annotations, f)

                annotated_frame = r.plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                if self.boxes:
                    for i, box in enumerate(self.boxes):
                        label = f"目标 {i + 1}"
                        annotated_frame_rgb = put_chinese_text(
                            annotated_frame_rgb,
                            label,
                            (box.x1, max(10, box.y1 - 10)),
                            font_size=15,
                            color=box.color
                        )

                out.write(annotated_frame_rgb)
                frame_count += 1
                
                # GPU内存优化：定期清理
                if frame_count % 30 == 0:
                    print(f"已处理 {frame_count} 帧")
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                
                with open(temp_data_path / 'annotations.json', 'w') as f:
                    json.dump(coco_data, f)

            out.release()
            print(f"✓ 标注视频已保存到: {output_path}")
            print(f"✓ 共处理 {frame_count} 帧")
            print(f"✓ 标注了 {len(self.boxes) if self.boxes else 0} 个目标区域")
            print(f"✓ COCO格式标注已保存到: {temp_data_path / 'annotations.json'}")
            print(f"✓ 临时数据已保存到: {temp_data_path}")
            upload_to_obs(str(output_path))

            self.launch_control_panel(output_path)

        except ImportError as e:
            print(f"SAM3VideoPredictor导入失败: {e}")
            print("正在回退到SAM图片分割模式...")

            try:
                from ultralytics import SAM
                print("正在加载SAM模型...")
                device, device_type = get_device()
                sam_model = SAM(SAM_MODEL_PATH)
                sam_model.to(device)
                print(f"SAM模型加载成功: {SAM_MODEL_PATH} (device: {device})")

                print("正在使用SAM模型进行智能分割...")
                print("注意: SAM分割可能需要一些时间，请耐心等待...")

                for i, box in enumerate(self.boxes):
                    print(f"正在分割目标 {i+1}/{len(self.boxes)}...")
                    bbox = [box.x1, box.y1, box.x2, box.y2]

                    try:
                        results = sam_model(self.frame, bboxes=[bbox], verbose=False)

                        if results and results[0].masks is not None:
                            mask = results[0].masks.data[0].cpu().numpy()
                            mask = (mask * 255).astype(np.uint8)
                            box.mask = mask
                            print(f"  ✓ 目标 {i+1} 分割完成")
                        else:
                            print(f"  ⚠ 目标 {i+1} SAM未检测到掩码，使用矩形框")
                    except Exception as e:
                        print(f"  ✗ 目标 {i+1} 分割失败: {e}")
                        print(f"  → 使用矩形框替代")

                output_filename = get_output_filename(self.video_path)
                output_path = self.output_dir / output_filename

                fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = ''.join([
                    chr(fourcc_int & 0xFF),
                    chr((fourcc_int >> 8) & 0xFF),
                    chr((fourcc_int >> 16) & 0xFF),
                    chr((fourcc_int >> 24) & 0xFF)
                ])
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0

                print("正在生成标注视频...")
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    annotated_frame = frame.copy()
                    if self.boxes:
                        for box in self.boxes:
                            if box.mask is not None:
                                annotated_frame = box.apply_sam_mask_to_frame(annotated_frame)
                            else:
                                annotated_frame = box.apply_mask_to_frame(annotated_frame)
                                cv2.rectangle(annotated_frame,
                                            (box.x1, box.y1),
                                            (box.x2, box.y2),
                                            box.color, 2)

                            label = f"目标 {self.boxes.index(box) + 1}"
                            annotated_frame = put_chinese_text(annotated_frame, label,
                                                            (box.x1, box.y1 - 10),
                                                            font_size=15, color=box.color)

                    out.write(annotated_frame)
                    frame_count += 1

                    if frame_count % 30 == 0:
                        print(f"已处理 {frame_count} 帧")

                out.release()
                print(f"✓ 标注视频已保存到: {output_path}")
                print(f"✓ 共处理 {frame_count} 帧")
                print(f"✓ 标注了 {len(self.boxes) if self.boxes else 0} 个目标区域")
                print(f"✓ 临时数据已保存到: {temp_data_path}")
                upload_to_obs(str(output_path))

                self.launch_control_panel(output_path)

            except Exception as e:
                print(f"SAM模型加载失败: {e}")
                print("将使用简单的矩形框标注")

                output_filename = get_output_filename(self.video_path)
                output_path = self.output_dir / output_filename

                fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = ''.join([
                    chr(fourcc_int & 0xFF),
                    chr((fourcc_int >> 8) & 0xFF),
                    chr((fourcc_int >> 16) & 0xFF),
                    chr((fourcc_int >> 24) & 0xFF)
                ])
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0

                print("正在生成标注视频...")
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    annotated_frame = frame.copy()
                    if self.boxes:
                        for box in self.boxes:
                            annotated_frame = box.apply_mask_to_frame(annotated_frame)
                            cv2.rectangle(annotated_frame,
                                        (box.x1, box.y1),
                                        (box.x2, box.y2),
                                        box.color, 2)

                            label = f"目标 {self.boxes.index(box) + 1}"
                            annotated_frame = put_chinese_text(annotated_frame, label,
                                                            (box.x1, box.y1 - 10),
                                                            font_size=15, color=box.color)

                    out.write(annotated_frame)
                    frame_count += 1

                    if frame_count % 30 == 0:
                        print(f"已处理 {frame_count} 帧")

                out.release()
                print(f"✓ 标注视频已保存到: {output_path}")
                print(f"✓ 共处理 {frame_count} 帧")
                print(f"✓ 标注了 {len(self.boxes) if self.boxes else 0} 个目标区域")
                upload_to_obs(str(output_path))

                self.launch_control_panel(output_path)

def main():
    global FIND, IOU_THRESHOLD
    FIND = []

    print("=" * 50)
    print("视频标注工具 - SAM3实例分割")
    print("=" * 50)
    
    default_iou = IOU_THRESHOLD
    iou_input = input("IoU阈值（默认{0}）：".format(default_iou)).strip()
    if iou_input:
        try:
            IOU_THRESHOLD = float(iou_input)
            print("✓ IoU阈值设置为: {0}".format(IOU_THRESHOLD))
        except ValueError:
            print("无效输入，使用默认值: {0}".format(IOU_THRESHOLD))
    
    print("\n请输入要查找的物品名称（可输入多个）：")
    print("输入 'done' 表示完成输入")
    print("-" * 50)

    while True:
        item = input("物品名称: ").strip()
        if not item:
            print("物品名称不能为空，请重新输入")
            continue
        if item.lower() == 'done':
            break
        if item.lower() == 'skip':
            FIND = []
            print("  ⚠ 已跳过文本提示词输入")
            break
        if item not in FIND:
            FIND.append(item)
            print(f"  ✓ 已添加: {item}")
        else:
            print(f"  ⚠ 已存在: {item}")

    print("-" * 50)
    if FIND:
        print(f"已添加 {len(FIND)} 个物品: {', '.join(FIND)}")
    else:
        print("未添加物品（将跳过文本提示）")
    print("=" * 50)

    video_files = list(Path(SRC_DIR).glob("*.mp4"))

    if not video_files:
        print(f"\n在 {SRC_DIR} 目录下没有找到视频文件")
        print("请将视频文件放入 src 目录")
        return

    print("\n找到以下视频文件:")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {video_file.name}")

    if len(video_files) == 1:
        video_path = str(video_files[0])
    else:
        choice = input("\n请选择要标注的视频编号: ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                video_path = str(video_files[idx])
            else:
                print("无效的选择")
                return
        except ValueError:
            print("请输入有效的数字")
            return

    print(f"\n开始标注: {video_path}")
    annotator = VideoAnnotator(video_path, DST_DIR)
    annotator.run()

if __name__ == "__main__":
    main()
