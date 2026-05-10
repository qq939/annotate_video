#!/usr/bin/env python3
"""app_utils.py - app.py和post_annotate.py共用的业务逻辑模块"""

import cv2
import numpy as np
import json
import shutil
import torch
from pathlib import Path


# ==================== 目录常量 ====================
TEMP_DATA_DIR = Path("temp_data")
TEMP_DATA_MID_DIR = Path("temp_data_mid")
TEMP_DATA_POST_DIR = Path("temp_data_post")


# ==================== SAM相关 ====================
_SAM3_SEMANTIC_PATCHED = False

def patch_sam3_video_semantic():
    """打补丁SAM3VideoSemanticPredictor支持bboxes提示"""
    global _SAM3_SEMANTIC_PATCHED
    if _SAM3_SEMANTIC_PATCHED:
        return
    _SAM3_SEMANTIC_PATCHED = True
    
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
        _lbl_arr = np.ones(nb) if labels is None else np.array(labels)
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
        return frame_idx, self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)

    SAM3VideoSemanticPredictor.add_prompt = _new_add_prompt


def get_device():
    """获取设备类型"""
    from annotate_video import get_device as _get_device
    return _get_device()


def get_sam_overrides(device='auto', conf=0.25, model_path=None):
    """获取SAM预测器的overrides参数"""
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    half = device == 'cuda'
    overrides = {
        'conf': conf, 'task': "segment", 'mode': "predict",
        'model': model_path, 'device': device, 'half': half, 
        'save': False, 'verbose': False
    }
    if device == 'cuda':
        overrides['batch'] = 1
        overrides['stream_buffer'] = False
    elif device == 'mps':
        overrides['half'] = True
        overrides['amp'] = True
        overrides['stream_buffer'] = True
    return overrides, device


# ==================== track_id管理 ====================
def first_available_track_id(coco_data, base_start=10000):
    """在coco数据中找到可用的起始track_id"""
    if not coco_data or not coco_data.get('annotations'):
        return base_start
    occupied = set()
    for ann in coco_data.get('annotations', []):
        occupied.add((ann.get('track_id', 0) // 10000) * 10000)
    opts = list(range(base_start, 501000, 10000))
    for o in opts:
        if (o // 10000) * 10000 not in occupied:
            return o
    return opts[-1] if opts else base_start


def apply_single_mapping_to_mid(from_id, to_id, temp_mid_dir=None):
    """将单个track_id映射应用到temp_data_mid"""
    temp_mid = Path(temp_mid_dir) if temp_mid_dir else TEMP_DATA_MID_DIR
    labels_dir = temp_mid / "labels"
    annotations_file = temp_mid / "annotations.json"
    if not labels_dir.exists():
        return 0
    
    count = 0
    for label_file in sorted(labels_dir.glob("frame_*.json")):
        with open(label_file) as f:
            frame_anns = json.load(f)
        changed = False
        for ann in frame_anns:
            if ann.get('track_id') == from_id:
                ann['track_id'] = to_id
                changed = True
        if changed:
            with open(label_file, 'w') as f:
                json.dump(frame_anns, f)
            count += 1
    
    if annotations_file.exists():
        with open(annotations_file) as f:
            coco = json.load(f)
        for ann in coco.get('annotations', []):
            if ann.get('track_id') == from_id:
                ann['track_id'] = to_id
        with open(annotations_file, 'w') as f:
            json.dump(coco, f)
    
    return count


def apply_trace_id_mappings(mappings, temp_mid_dir=None):
    """将track_id映射列表应用到temp_data_mid
    
    Args:
        mappings: [(old_id, new_id), ...] 列表
    
    Returns:
        影响的帧数
    """
    temp_mid = Path(temp_mid_dir) if temp_mid_dir else TEMP_DATA_MID_DIR
    labels_dir = temp_mid / "labels"
    annotations_file = temp_mid / "annotations.json"
    
    if not mappings or not labels_dir.exists():
        return 0
    
    converted_count = 0
    for label_file in sorted(labels_dir.glob("frame_*.json")):
        with open(label_file) as f:
            frame_anns = json.load(f)
        changed = False
        for ann in frame_anns:
            for old_id, new_id in mappings:
                if ann.get('track_id') == old_id:
                    ann['track_id'] = new_id
                    changed = True
        if changed:
            with open(label_file, 'w') as f:
                json.dump(frame_anns, f)
            converted_count += 1
    
    if annotations_file.exists():
        with open(annotations_file) as f:
            coco = json.load(f)
        changed = False
        for ann in coco.get('annotations', []):
            for old_id, new_id in mappings:
                if ann.get('track_id') == old_id:
                    ann['track_id'] = new_id
                    changed = True
        if changed:
            with open(annotations_file, 'w') as f:
                json.dump(coco, f)
    
    return converted_count


# ==================== 提示帧/双向标注 ====================
def run_prompt_frame(frame_path, bboxes=None, find_list=None, overrides=None, first_id=1000000):
    """对单帧执行提示标注，返回标注列表"""
    from annotate_video import merge_masks_in_frame
    
    if overrides is None:
        overrides, _ = get_sam_overrides()
    
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return [], first_id
    h, w = frame.shape[:2]
    
    use_semantic = bool(find_list)
    
    if bboxes and not use_semantic:
        from ultralytics.models.sam import SAM3SemanticPredictor
        predictor = SAM3SemanticPredictor(overrides=overrides)
        pred_args = {'source': str(frame_path), 'bboxes': bboxes, 'labels': [1] * len(bboxes)}
        if find_list:
            pred_args['text'] = find_list
        results = predictor(**pred_args)
    elif use_semantic:
        patch_sam3_video_semantic()
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        clip_path = str(frame_path) + '_clip.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, 30, (w, h))
        out.write(frame)
        out.release()
        pred_args = {'source': clip_path, 'stream': True}
        if bboxes:
            pred_args['bboxes'] = bboxes
            pred_args['labels'] = [1] * len(bboxes)
        pred_args['text'] = find_list
        results = predictor(**pred_args)
    else:
        from ultralytics.models.sam import SAM3Predictor
        predictor = SAM3Predictor(overrides=overrides)
        pred_args = {'source': str(frame_path)}
        results = predictor(**pred_args)
    
    annotations = []
    for r in results:
        masks_attr = getattr(r, 'masks', None)
        if masks_attr is None or masks_attr.data is None:
            continue
        
        mt = masks_attr.data
        confs = None
        boxes_attr = getattr(r, 'boxes', None)
        if boxes_attr is not None and hasattr(boxes_attr, 'conf'):
            confs = boxes_attr.conf.cpu().numpy()
        
        cm, cb = [], []
        for m in mt:
            mn = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m)
            if mn.shape[-2:] != (h, w):
                mn = cv2.resize(mn.astype(np.float32), (w, h))
            mb = (mn > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    poly = cnt.squeeze().flatten().tolist()
                    xs, ys = poly[0::2], poly[1::2]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    area = cv2.contourArea(cnt)
                    if area > 0:
                        cm.append(mb)
                        cb.append(bbox)
        
        if cm:
            cm, cb = merge_masks_in_frame(cm, cb, 0.5)
            for idx, (m, b) in enumerate(zip(cm, cb)):
                track_id = first_id + idx
                conf_val = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
                
                m_uint8 = None
                if m.dtype == np.uint8:
                    m_uint8 = m
                elif m.max() <= 1:
                    m_uint8 = (m * 255).astype(np.uint8)
                else:
                    m_uint8 = m.astype(np.uint8)
                
                contours, _ = cv2.findContours(m_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) >= 3:
                        area = cv2.contourArea(cnt)
                        if area > 0:
                            ann = {
                                'id': idx + 1,
                                'track_id': track_id,
                                'image_id': 0,
                                'category_id': track_id,
                                'bbox': b,
                                'area': float(area),
                                'segmentation': [cnt.squeeze().flatten().tolist()],
                                'iscrowd': 0,
                                'confidence': conf_val
                            }
                            annotations.append(ann)
                            break
    
    return annotations, first_id + len(annotations)


# ==================== 删除track_id相关 ====================
def find_annotations_containing_point(x, y, annotations, conf_threshold=0.5):
    """在标注列表中查找包含点击位置的所有标注"""
    containing = []
    for ann in annotations:
        conf = ann.get('confidence', 1.0)
        if conf < conf_threshold:
            continue
        polygon = ann.get('segmentation')
        if not polygon:
            continue
        pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
        if len(pts) < 3:
            continue
        if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
            containing.append(ann)
    return containing


def mark_track_ids_deleted(track_ids, temp_data_dir=None):
    """将指定track_id标记为删除(改为9999)
    
    Args:
        track_ids: 要删除的track_id列表
        temp_data_dir: 数据目录，默认为TEMP_DATA_MID_DIR
    
    Returns:
        影响的帧数
    """
    temp_dir = Path(temp_data_dir) if temp_data_dir else TEMP_DATA_MID_DIR
    labels_dir = temp_dir / "labels"
    annotations_file = temp_dir / "annotations.json"
    
    if not labels_dir.exists():
        return 0
    
    track_ids_set = set(track_ids)
    count = 0
    
    for label_file in sorted(labels_dir.glob("frame_*.json")):
        with open(label_file) as f:
            frame_anns = json.load(f)
        changed = False
        for ann in frame_anns:
            if ann.get('track_id') in track_ids_set:
                ann['track_id'] = 9999
                changed = True
        if changed:
            with open(label_file, 'w') as f:
                json.dump(frame_anns, f)
            count += 1
    
    if annotations_file.exists():
        with open(annotations_file) as f:
            coco = json.load(f)
        changed = False
        for ann in coco.get('annotations', []):
            if ann.get('track_id') in track_ids_set:
                ann['track_id'] = 9999
                changed = True
        if changed:
            with open(annotations_file, 'w') as f:
                json.dump(coco, f)
    
    return count


# ==================== 保存标注 ====================
def save_frame_annotations(frame_idx, annotations, labels_dir, annotations_file):
    """保存帧标注到文件"""
    if not annotations:
        return
    
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    existing = []
    lf = labels_dir / f"frame_{frame_idx:06d}.json"
    if lf.exists():
        with open(lf) as f:
            existing = json.load(f)
    
    for ann in annotations:
        ann['image_id'] = frame_idx
    
    existing.extend(annotations)
    with open(lf, 'w') as f:
        json.dump(existing, f)
    
    if annotations_file and Path(annotations_file).exists():
        with open(annotations_file, 'r') as f:
            coco = json.load(f)
        coco['annotations'].extend(annotations)
        with open(annotations_file, 'w') as f:
            json.dump(coco, f)


def load_frame_annotations(frame_idx, labels_dir):
    """加载指定帧的标注"""
    labels_dir = Path(labels_dir)
    lf = labels_dir / f"frame_{frame_idx:06d}.json"
    if lf.exists():
        with open(lf) as f:
            return json.load(f)
    return []


# ==================== 复制目录 ====================
def copy_temp_data(src_dir, dst_dir, apply_mappings=None):
    """复制temp_data到目标目录，可选应用track_id映射
    
    Args:
        src_dir: 源目录
        dst_dir: 目标目录
        apply_mappings: 可选的映射列表 [(old_id, new_id), ...]
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    
    if apply_mappings:
        apply_trace_id_mappings(apply_mappings)


# ==================== 导出相关 ====================
def get_category_for_track_id(track_id, category_mappings=None):
    """根据track_id获取类别名称"""
    if category_mappings is None:
        category_mappings = ['Detect'] * 8
    
    if track_id >= 1000000:
        return category_mappings[0] if category_mappings else 'Detect'
    
    categories = {
        1: 'Pedestrian', 2: 'Cyclist', 3: 'Car',
        4: 'Van', 5: 'Truck', 6: 'Tram', 7: 'Tricycle', 8: 'Other'
    }
    return categories.get(track_id, category_mappings[-1] if category_mappings else 'Detect')


def export_to_temp_data_post(cat_maps=None, del_track_id_list=None, temp_mid_dir=None):
    """导出到temp_data_post
    
    Args:
        cat_maps: 类别映射列表
        del_track_id_list: 要删除的track_id列表
        temp_mid_dir: 源目录，默认为TEMP_DATA_MID_DIR
    """
    temp_mid = Path(temp_mid_dir) if temp_mid_dir else TEMP_DATA_MID_DIR
    if cat_maps is None:
        cat_maps = ['Detect'] * 8
    
    af = temp_mid / "annotations.json"
    if not af.exists():
        return False, "annotations.json不存在"
    
    with open(af) as f:
        coco = json.load(f)
    
    out = TEMP_DATA_POST_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    old = out / "labels"
    ofram = out / "frames"
    old.mkdir(parents=True)
    ofram.mkdir(parents=True)
    
    ld = temp_mid / "labels"
    fd = temp_mid / "frames"
    
    deleted = set(del_track_id_list) if del_track_id_list else set()
    deleted.add(9999)
    
    exported_count = 0
    for frame_file in sorted(fd.glob("frame_*.jpg")):
        shutil.copy2(frame_file, ofram / frame_file.name)
    
    for label_file in sorted(ld.glob("frame_*.json")):
        with open(label_file) as f:
            fa = json.load(f)
        
        filtered = [a for a in fa if a.get('track_id', 0) not in deleted and a.get('track_id', 0) < 1000000]
        
        for ann in filtered:
            ann['category_id'] = cat_maps.index(get_category_for_track_id(ann.get('track_id', 0), cat_maps)) + 1
        
        with open(old / label_file.name, 'w') as f:
            json.dump(filtered, f)
        exported_count += len(filtered)
    
    coco['annotations'] = [a for a in coco.get('annotations', []) 
                           if a.get('track_id', 0) not in deleted and a.get('track_id', 0) < 1000000]
    for ann in coco['annotations']:
        ann['category_id'] = cat_maps.index(get_category_for_track_id(ann.get('track_id', 0), cat_maps)) + 1
    
    with open(out / "annotations.json", 'w') as f:
        json.dump(coco, f)
    
    return True, f"导出完成，{exported_count}条标注"


# ==================== 视频处理 ====================
def extract_video_clip_from_frames(frames_dir, start_idx, total_frames, output_path, fps=30):
    """从帧目录提取视频片段"""
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    
    sample = cv2.imread(str(frames_dir / f"frame_{start_idx:06d}.jpg"))
    if sample is None:
        return False, f"无法读取起始帧: frame_{start_idx:06d}.jpg"
    
    height, width = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
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
    return True, f"视频片段已提取: {output_path} ({frame_count}帧)"


# ==================== 双向标注 ====================
def process_clip_for_bidirectional(start_frame, end_frame, forward, prompt_bboxes, 
                                   mid_frames_dir, src_labels_dir, temp_inject_dir,
                                   predictor, width, height, first_id,
                                   iou_threshold=0.5, merge_iou_threshold=0.5):
    """处理视频片段进行双向标注（从app.py的process_clip复制）"""
    from annotate_video import merge_masks_in_frame, TrackManager
    
    direction = "向前" if forward else "向后"
    print(f"\n[DEBUG {direction}] === 进入 process_clip ===")
    print(f"[DEBUG {direction}] start_frame={start_frame}, end_frame={end_frame}, 总帧数={end_frame - start_frame}")

    if start_frame >= end_frame:
        print(f"[DEBUG {direction}] start_frame >= end_frame, 直接返回空列表")
        return [], 0

    temp_frames = temp_inject_dir / ("forward" if forward else "backward")
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

    print(f"[DEBUG {direction}] 正在加载 SAM3VideoPredictor 处理...")
    print(f"[DEBUG {direction}] prompt_bboxes={prompt_bboxes}")
    
    if prompt_bboxes:
        results = predictor(source=clip_path, stream=True, bboxes=prompt_bboxes, labels=[1]*len(prompt_bboxes))
    else:
        results = predictor(source=clip_path, stream=True)
        print(f"[DEBUG {direction}] ⚠️ prompt_bboxes为空，无法进行SAM3VideoPredictor分割！")
    
    manager = TrackManager(iou_threshold=iou_threshold)
    manager.next_track_id = first_id
    ann_id = first_id

    result_anns = []
    frame_idx = 0
    total = end_frame

    print(f"[DEBUG {direction}] 开始遍历 predictor 结果...")
    for r in results:
        orig_img = r.orig_img if hasattr(r, 'orig_img') and r.orig_img is not None else None
        if orig_img is None:
            cap_t = cv2.VideoCapture(clip_path)
            cap_t.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret_t, orig_img = cap_t.read()
            cap_t.release()
            if not ret_t:
                orig_img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            if len(orig_img.shape) == 2:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            elif orig_img.shape[2] == 4:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(temp_frames / f"frame_{frame_idx:06d}.jpg"), orig_img)

        frame_anns = []
        has_masks = hasattr(r, 'masks') and r.masks is not None

        if has_masks:
            masks_tensor = r.masks.data
            if masks_tensor is not None and len(masks_tensor) > 0:
                confs = None
                if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'conf'):
                    confs = r.boxes.conf.cpu().numpy()

                cur_masks = []
                cur_bboxes = []
                for mask in masks_tensor:
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
                    mask_binary = (mask_np > 0.5).astype(np.uint8)
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
                    cur_masks, cur_bboxes = merge_masks_in_frame(cur_masks, cur_bboxes, merge_iou_threshold)
                    track_ids = manager.update(cur_masks, cur_bboxes, frame_idx)

                    for idx, (m, bb) in enumerate(zip(cur_masks, cur_bboxes)):
                        m_bin = (m > 0.5).astype(np.uint8)
                        cnts2, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt2 in cnts2:
                            if len(cnt2) >= 3:
                                poly2 = cnt2.squeeze().flatten().tolist()
                                area2 = cv2.contourArea(cnt2)
                                tid = track_ids[idx] if idx < len(track_ids) else ann_id
                                conf = float(confs[idx]) if confs is not None and idx < len(confs) else 1.0
                                if forward:
                                    img_id = frame_idx + start_frame
                                else:
                                    img_id = end_frame - 1 - frame_idx
                                ann = {
                                    'id': ann_id, 'track_id': tid, 'image_id': img_id,
                                    'category_id': tid, 'bbox': bb, 'area': float(area2),
                                    'segmentation': [poly2], 'iscrowd': 0, 'confidence': conf,
                                    'category': 'Detect'
                                }
                                result_anns.append(ann)
                                frame_anns.append(ann)
                                ann_id += 1

        if forward:
            orig_frame_idx = frame_idx + start_frame
        else:
            orig_frame_idx = end_frame - 1 - frame_idx
        
        if orig_frame_idx >= total:
            frame_idx += 1
            continue
        
        label_file = src_labels_dir / f"frame_{orig_frame_idx:06d}.json"
        existing_anns = []
        if label_file.exists():
            with open(label_file) as f:
                existing_anns = json.load(f)
        merged_anns = existing_anns + frame_anns
        with open(label_file, 'w') as f:
            json.dump(merged_anns, f)
        
        frame_idx += 1

    print(f"[DEBUG {direction}] === process_clip 完成 ===")
    return result_anns, ann_id


def run_bidirectional_inject(prompt_idx, total_frames, bboxes, forward_enabled=True, backward_enabled=True,
                              iou_threshold=0.5, merge_iou_threshold=0.5, first_id=1000000,
                              temp_mid_dir=None):
    """执行双向标注（从app.py的do_bidirectional_inject复制）"""
    from annotate_video import get_device, SAM_MODEL_PATH
    from pathlib import Path as P
    
    temp_mid = P(temp_mid_dir) if temp_mid_dir else TEMP_DATA_MID_DIR
    mid_frames_dir = temp_mid / "frames"
    mid_labels_dir = temp_mid / "labels"
    mid_annotations_file = temp_mid / "annotations.json"
    
    device, device_type = get_device()
    half = device_type == 'cuda'
    overrides = {
        'conf': 0.25, 'task': "segment", 'mode': "predict",
        'model': SAM_MODEL_PATH, 'device': device, 'half': half, 
        'save': False, 'verbose': False
    }
    if device_type == 'cuda':
        overrides['batch'] = 1
        overrides['stream_buffer'] = False
    elif device_type == 'mps':
        overrides['half'] = True
        overrides['amp'] = True
        overrides['stream_buffer'] = True

    from ultralytics.models.sam import SAM3VideoPredictor
    predictor = SAM3VideoPredictor(overrides=overrides)

    sample_frame = cv2.imread(str(mid_frames_dir / f"frame_{0:06d}.jpg"))
    if sample_frame is None:
        return False, "无法读取帧"
    height, width = sample_frame.shape[:2]

    print(f"=== 双向标注开始 === 提示帧: {prompt_idx}, 总帧数: {total_frames}, 前向={forward_enabled}, 后向={backward_enabled}, FIRST_ID={first_id}")

    temp_inject = P("temp_inject")
    temp_inject.mkdir(exist_ok=True)

    forward_anns = []
    backward_anns = []

    if forward_enabled:
        forward_anns, new_first_id = process_clip_for_bidirectional(
            prompt_idx, total_frames, True, bboxes,
            mid_frames_dir, mid_labels_dir, temp_inject,
            predictor, width, height, first_id,
            iou_threshold, merge_iou_threshold
        )
        first_id = new_first_id

    if backward_enabled:
        backward_anns, _ = process_clip_for_bidirectional(
            0, prompt_idx, False, bboxes,
            mid_frames_dir, mid_labels_dir, temp_inject,
            predictor, width, height, first_id,
            iou_threshold, merge_iou_threshold
        )

    return True, f"双向标注完成，前向{len(forward_anns)}条，后向{len(backward_anns)}条"


# ==================== 视频标注（用于SSE） ====================
def get_predictor_args(video_path, bboxes=None, find_list=None, device='auto', model_path=None):
    """获取预测器参数（从app.py的annotate endpoint复制）"""
    from annotate_video import get_device
    
    if device == 'auto':
        device, _ = get_device()
    
    device_type = device if isinstance(device, str) else 'cpu'
    half = device_type == 'cuda'
    overrides = {
        'conf': 0.25, 'task': "segment", "mode": "predict",
        'model': model_path, 'device': device, 'half': half,
        'save': False, 'verbose': False
    }
    if device_type == 'cuda':
        overrides['batch'] = 1
        overrides['stream_buffer'] = False
    elif device_type == 'mps':
        overrides['half'] = True
        overrides['amp'] = True
        overrides['stream_buffer'] = True
    
    use_semantic = bool(find_list)
    if use_semantic:
        try:
            patch_sam3_video_semantic()
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
            predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        except Exception:
            from ultralytics.models.sam import SAM3VideoPredictor
            predictor = SAM3VideoPredictor(overrides=overrides)
            use_semantic = False
    else:
        from ultralytics.models.sam import SAM3VideoPredictor
        predictor = SAM3VideoPredictor(overrides=overrides)
    
    if bboxes:
        source = video_path
        pred_args = {'source': source, 'stream': True, 'bboxes': bboxes, 'labels': [1] * len(bboxes)}
    else:
        source = video_path
        pred_args = {'source': source, 'stream': True}
        if find_list:
            pred_args['text'] = find_list
    
    return predictor, pred_args, overrides


def process_annotation_stream(source, predictor_args, iou=0.5, merge_iou=0.5, height=720, width=1280, yield_func=None):
    """处理视频标注，返回结果迭代器
    
    Args:
        source: 视频路径
        predictor_args: 预测器参数
        iou: IOU阈值
        merge_iou: 合并IOU阈值
        height, width: 视频尺寸
        yield_func: 可选的yield回调函数(current_frame, total, msg)
    
    Returns:
        (coco_data, frame_count): 标注数据和帧数
    """
    from annotate_video import merge_masks_in_frame, TrackManager
    from pathlib import Path as P
    
    track_manager = TrackManager(iou_threshold=iou)
    ann_id_counter = [0]
    
    cap_cnt = cv2.VideoCapture(source)
    total = int(cap_cnt.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_cnt.release()
    
    coco_data = {
        'info': {},
        'images': [], 'annotations': [], 'categories': []
    }
    
    predictor = predictor_args.get('predictor')
    args = {k: v for k, v in predictor_args.items() if k != 'predictor'}
    results = predictor(**args)
    
    frame_count = 0
    for r in results:
        orig_img = getattr(r, 'orig_img', None)
        if orig_img is None:
            ct = cv2.VideoCapture(source)
            ct.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            _, orig_img = ct.read()
            ct.release()
        if orig_img is None:
            orig_img = np.zeros((height, width, 3), dtype=np.uint8)
        elif len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        elif orig_img.shape[2] == 4:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
        
        coco_data['images'].append({
            'id': frame_count,
            'file_name': f"frame_{frame_count:06d}.jpg",
            'width': width, 'height': height
        })
        
        frame_anns = []
        masks_attr = getattr(r, 'masks', None)
        if masks_attr is not None and masks_attr.data is not None:
            mt = masks_attr.data
            confs = None
            boxes_attr = getattr(r, 'boxes', None)
            if boxes_attr is not None and hasattr(boxes_attr, 'conf'):
                confs = boxes_attr.conf.cpu().numpy()
            
            curr_masks = []
            curr_boxes = []
            for m in mt:
                mn = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m)
                if mn.shape[-2:] != (height, width):
                    mn = cv2.resize(mn.astype(np.float32), (width, height))
                mb = (mn > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) >= 3:
                        poly = cnt.squeeze().flatten().tolist()
                        xs, ys = poly[0::2], poly[1::2]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        area = cv2.contourArea(cnt)
                        if area > 0:
                            curr_masks.append(mb)
                            curr_boxes.append(bbox)
            
            if curr_masks:
                curr_masks, curr_boxes = merge_masks_in_frame(curr_masks, curr_boxes, merge_iou)
                tids = track_manager.update(curr_masks, curr_boxes, frame_count)
                for idx, (m, b) in enumerate(zip(curr_masks, curr_boxes)):
                    mb = (m > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if len(cnt) >= 3:
                            poly = cnt.squeeze().flatten().tolist()
                            area = cv2.contourArea(cnt)
                            tid = tids[idx] if idx < len(tids) else ann_id_counter[0]
                            conf = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                            ann = {
                                'id': ann_id_counter[0], 'track_id': tid, 'image_id': frame_count,
                                'category_id': tid, 'bbox': b, 'area': float(area),
                                'segmentation': [poly], 'iscrowd': 0, 'confidence': conf
                            }
                            coco_data['annotations'].append(ann)
                            frame_anns.append(ann)
                            ann_id_counter[0] += 1
        
        frame_count += 1
        debug_contours = 0
        if masks_attr is not None and masks_attr.data is not None:
            mt = masks_attr.data
            for m in mt:
                mn = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m)
                mb = (mn > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                debug_contours += len(contours)
        
        if yield_func:
            yield_func(frame_count, total, f'帧 {frame_count}/{total}: contours={debug_contours}, annotations={len(frame_anns)}')
    
    return coco_data, frame_count


def run_video_annotate(src_video, bboxes, find_list, overrides, use_semantic, iou, merge_iou, src_video_dir, temp_data_dir, yield_func=None):
    """执行视频标注（SSE版，从app.py的annotate endpoint移植）
    
    Returns:
        (coco_data, frame_count)
    """
    from annotate_video import merge_masks_in_frame, TrackManager
    
    if temp_data_dir.exists():
        shutil.rmtree(temp_data_dir)
    temp_data_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = temp_data_dir / "frames"
    labels_dir = temp_data_dir / "labels"
    frames_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(src_video)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join([chr((fourcc_int >> 24) & 0xFF), chr((fourcc_int >> 16) & 0xFF), chr((fourcc_int >> 8) & 0xFF), chr(fourcc_int & 0xFF)])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if use_semantic:
        try:
            patch_sam3_video_semantic()
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
            predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        except Exception:
            from ultralytics.models.sam import SAM3VideoPredictor
            predictor = SAM3VideoPredictor(overrides=overrides)
            use_semantic = False
    else:
        from ultralytics.models.sam import SAM3VideoPredictor
        predictor = SAM3VideoPredictor(overrides=overrides)

    if bboxes:
        src_dst = src_video_dir / "input_source.mp4"
        shutil.copy2(src_video, str(src_dst))
        source = str(src_dst)
        predictor_args = {'source': source, 'stream': True, 'bboxes': bboxes, 'labels': [1] * len(bboxes)}
    else:
        source = src_video
        predictor_args = {'source': source, 'stream': True}
        if find_list:
            predictor_args['text'] = find_list

    track_manager = TrackManager(iou_threshold=iou)
    ann_id_counter = [0]

    cap_cnt = cv2.VideoCapture(source)
    total = int(cap_cnt.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_cnt.release()

    coco_data = {'info': {'fps': fps, 'width': width, 'height': height, 'fourcc': fourcc_str, 'FIND': find_list}, 'images': [], 'annotations': [], 'categories': []}
    results = predictor(**predictor_args)

    frame_count = 0
    for r in results:
        orig_img = getattr(r, 'orig_img', None)
        if orig_img is None:
            ct = cv2.VideoCapture(source)
            ct.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            _, orig_img = ct.read()
            ct.release()
        if orig_img is None:
            orig_img = np.zeros((height, width, 3), dtype=np.uint8)
        elif len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        elif orig_img.shape[2] == 4:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)

        cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), orig_img)
        coco_data['images'].append({'id': frame_count, 'file_name': f"frame_{frame_count:06d}.jpg", 'width': width, 'height': height})

        frame_anns = []
        masks_attr = getattr(r, 'masks', None)
        if masks_attr is not None and masks_attr.data is not None:
            mt = masks_attr.data
            confs = None
            boxes_attr = getattr(r, 'boxes', None)
            if boxes_attr is not None and hasattr(boxes_attr, 'conf'):
                confs = boxes_attr.conf.cpu().numpy()

            curr_masks = []
            curr_boxes = []
            for m in mt:
                mn = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m)
                if mn.shape[-2:] != (height, width):
                    mn = cv2.resize(mn.astype(np.float32), (width, height))
                mb = (mn > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if len(cnt) >= 3:
                        poly = cnt.squeeze().flatten().tolist()
                        xs, ys = poly[0::2], poly[1::2]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        area = cv2.contourArea(cnt)
                        if area > 0:
                            curr_masks.append(mb)
                            curr_boxes.append(bbox)

            if curr_masks:
                curr_masks, curr_boxes = merge_masks_in_frame(curr_masks, curr_boxes, merge_iou)
                tids = track_manager.update(curr_masks, curr_boxes, frame_count)
                for idx, (m, b) in enumerate(zip(curr_masks, curr_boxes)):
                    mb = (m > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if len(cnt) >= 3:
                            poly = cnt.squeeze().flatten().tolist()
                            area = cv2.contourArea(cnt)
                            tid = tids[idx] if idx < len(tids) else ann_id_counter[0]
                            conf = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                            ann = {'id': ann_id_counter[0], 'track_id': tid, 'image_id': frame_count, 'category_id': tid, 'bbox': b, 'area': float(area), 'segmentation': [poly], 'iscrowd': 0, 'confidence': conf}
                            coco_data['annotations'].append(ann)
                            frame_anns.append(ann)
                            ann_id_counter[0] += 1

        with open(labels_dir / f"frame_{frame_count:06d}.json", 'w') as f_out:
            json.dump(frame_anns, f_out)

        frame_count += 1
        debug_contours = 0
        if masks_attr is not None and masks_attr.data is not None:
            mt = masks_attr.data
            for m in mt:
                mn = m.cpu().numpy() if hasattr(m, 'cpu') else np.array(m)
                mb = (mn > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                debug_contours += len(contours)
        
        if yield_func:
            yield_func(frame_count, total, f'帧 {frame_count}/{total}: contours={debug_contours}, annotations={len(frame_anns)}')

    with open(temp_data_dir / 'annotations.json', 'w') as f_out:
        json.dump(coco_data, f_out)

    return coco_data, frame_count
