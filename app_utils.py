#!/usr/bin/env python3
"""app_utils.py - app.py和web_app.py共用的业务逻辑"""

import cv2
import numpy as np
import json
from pathlib import Path


def get_sam_overrides(device='auto', conf=0.25, model_path=None):
    """获取SAM预测器的overrides参数"""
    if device == 'auto':
        import torch
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


def _first_available_track_id(coco_data, base_start=10000):
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


def run_prompt_frame(frame_path, bboxes=None, find_list=None, overrides=None, first_id=1000000):
    """对单帧执行提示标注，返回标注列表
    
    Args:
        frame_path: 帧图片路径
        bboxes: 边界框列表 [[x1,y1,x2,y2], ...]
        find_list: 文本提示列表
        overrides: SAM预测器参数
        first_id: 起始track_id
    
    Returns:
        (annotations, first_id): 标注列表和新first_id
    """
    from annotate_video import merge_masks_in_frame, _patch_sam3_video_semantic
    
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
        results = predictor(**pred_args)
    elif use_semantic:
        from annotate_video import _patch_sam3_video_semantic
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        _patch_sam3_video_semantic()
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
                conf_val = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max()) if hasattr(m, 'max') else 1.0
                ann = {
                    'id': idx + 1,
                    'track_id': track_id,
                    'image_id': 0,
                    'category_id': track_id,
                    'bbox': b,
                    'area': float(cv2.contourArea(m)),
                    'segmentation': [poly],
                    'iscrowd': 0,
                    'confidence': conf_val
                }
                annotations.append(ann)
    
    return annotations, first_id + len(annotations)


def save_frame_annotations(frame_idx, annotations, labels_dir, annotations_file):
    """保存帧标注到文件
    
    Args:
        frame_idx: 帧索引
        annotations: 标注列表
        labels_dir: 标签目录
        annotations_file: annotations.json路径
    """
    if not annotations:
        return
    
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    existing = []
    lf = labels_dir / f"frame_{frame_idx:06d}.json"
    if lf.exists():
        with open(lf) as f:
            existing = json.load(f)
    
    existing.extend(annotations)
    with open(lf, 'w') as f:
        json.dump(existing, f)
    
    for ann in annotations:
        ann['image_id'] = frame_idx
    
    with open(annotations_file, 'r') as f:
        coco = json.load(f)
    coco['annotations'].extend(annotations)
    with open(annotations_file, 'w') as f:
        json.dump(coco, f)
