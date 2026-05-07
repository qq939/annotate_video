#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web标注工具 - 视频标注，与app.py功能同步"""

import os
import sys
import shutil
import random
import json
import cv2
import numpy as np
from pathlib import Path
from threading import Thread
import base64
import io

from flask import Flask, render_template, request, jsonify, send_from_directory, Response

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

TEMP_DATA_DIR = Path("temp_data")
TEMP_DATA_MID_DIR = Path("temp_data_mid")
TEMP_DATA_POST_DIR = Path("temp_data_post")
SRC_VIDEO_DIR = Path("1src")
DST_VIDEO_DIR = Path("1dst")
SAM_MODEL_PATH = "sam3.pt"

for _d in [TEMP_DATA_DIR, TEMP_DATA_MID_DIR, TEMP_DATA_POST_DIR, SRC_VIDEO_DIR, DST_VIDEO_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

WARM_COLORS = [(180, 130, 255), (200, 100, 220), (255, 50, 200), (255, 0, 180),
               (220, 0, 150), (180, 0, 120), (139, 0, 100), (100, 0, 80)]
COLD_COLORS = [(100, 150, 255), (100, 200, 255), (150, 200, 255), (100, 255, 200),
               (150, 100, 255), (100, 200, 200), (150, 150, 255), (100, 180, 255)]

def get_color_for_track_id(track_id):
    if track_id >= 1000000:
        return WARM_COLORS[track_id % len(WARM_COLORS)]
    return COLD_COLORS[track_id % len(COLD_COLORS)]


def _put_chinese(img, text, pos, font_size=20, color=(255, 255, 255)):
    try:
        from PIL import Image, ImageDraw, ImageFont
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(img_pil)
        font = None
        for font_path in [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        draw.text(pos, text, font=font, fill=color)
        result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return result
    except Exception as e:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, color, 1)
        return img


@app.route('/')
def index():
    return render_template('web_app.html')


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': '没有选择文件'}), 400
    fname = Path(f.filename).name
    dst = SRC_VIDEO_DIR / fname
    if dst.exists():
        dst.unlink()
    f.save(str(dst))
    cap = cv2.VideoCapture(str(dst))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'error': '无法读取视频'}), 400
    h, w = frame.shape[:2]
    _, buf = cv2.imencode('.jpg', frame)
    thumb = base64.b64encode(buf).decode()
    return jsonify({'filename': fname, 'width': w, 'height': h, 'thumb': thumb})


@app.route('/api/extract_video', methods=['POST'])
def extract_video():
    data = request.json
    video_name = data.get('video_name')
    src_video = str(SRC_VIDEO_DIR / video_name)
    temp_data = TEMP_DATA_DIR
    if temp_data.exists():
        shutil.rmtree(temp_data)
    temp_data.mkdir(parents=True, exist_ok=True)
    frames_dir = temp_data / "frames"
    labels_dir = temp_data / "labels"
    frames_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(src_video)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join([chr(fourcc_int >> 24 & 0xFF), chr(fourcc_int >> 16 & 0xFF),
                          chr(fourcc_int >> 8 & 0xFF), chr(fourcc_int & 0xFF)])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    images = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), frame)
        images.append({'id': frame_count, 'file_name': f"frame_{frame_count:06d}.jpg",
                       'width': width, 'height': height, 'frame_count': frame_count})
        frame_count += 1
    cap.release()
    coco_data = {
        'info': {'description': 'Video', 'video_path': src_video, 'fps': fps,
                 'width': width, 'height': height, 'fourcc': fourcc_str},
        'images': images, 'annotations': [],
        'categories': [{'id': i, 'name': f'object_{i}'} for i in range(8)]
    }
    with open(temp_data / 'annotations.json', 'w') as f:
        json.dump(coco_data, f)
    return jsonify({'frames': frame_count, 'width': width, 'height': height})


@app.route('/api/annotate_stream', methods=['POST'])
def annotate_stream():
    data = request.json
    video_name = data.get('video_name')
    boxes = data.get('boxes', [])
    items_text = data.get('items', '')
    iou_val = float(data.get('iou', '0.5'))
    merge_iou_val = float(data.get('merge_iou', '0.5'))
    find_list = [s.strip() for s in items_text.split(',') if s.strip()]
    has_text = bool(find_list)
    has_bbox = bool(boxes)

    def generate():
        yield f"data: {json.dumps({'type': 'start', 'msg': '开始处理...'})}\n\n"
        try:
            import torch
            src_video = str(SRC_VIDEO_DIR / video_name)
            temp_data = TEMP_DATA_DIR
            if temp_data.exists():
                shutil.rmtree(temp_data)
            temp_data.mkdir(parents=True, exist_ok=True)
            frames_dir = temp_data / "frames"
            labels_dir = temp_data / "labels"
            frames_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            cap = cv2.VideoCapture(src_video)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([chr(fourcc_int >> 24 & 0xFF), chr(fourcc_int >> 16 & 0xFF),
                                  chr(fourcc_int >> 8 & 0xFF), chr(fourcc_int & 0xFF)])
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            predictor_name = "SAM3VideoSemanticPredictor" if has_text else "SAM3VideoPredictor"
            device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
            half = device == 'cuda'
            overrides = dict(conf=0.25, task="segment", mode="predict", model=SAM_MODEL_PATH,
                           device=device, half=half, save=False, verbose=False)
            if device == 'cuda':
                overrides['batch'] = 1
                overrides['stream_buffer'] = False
            elif device == 'mps':
                overrides['half'] = True
                overrides['amp'] = True
                overrides['stream_buffer'] = True
            if has_text:
                try:
                    from app import _patch_sam3_video_semantic
                    _patch_sam3_video_semantic()
                except Exception:
                    pass
            if predictor_name == "SAM3VideoSemanticPredictor":
                from ultralytics.models.sam import SAM3VideoSemanticPredictor
                predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            else:
                from ultralytics.models.sam import SAM3VideoPredictor
                predictor = SAM3VideoPredictor(overrides=overrides)
            predictor_args = {'source': src_video, 'stream': True}
            if has_bbox:
                predictor_args['bboxes'] = boxes
                predictor_args['labels'] = [1] * len(boxes)
            if has_text:
                predictor_args['text'] = find_list
            from annotate_video import merge_masks_in_frame, TrackManager
            track_manager = TrackManager(iou_threshold=iou_val)
            annotation_id = [0]
            coco_data = {'info': {'fps': fps, 'width': width, 'height': height, 'fourcc': fourcc_str, 'FIND': find_list},
                        'images': [], 'annotations': [],
                        'categories': [{'id': i, 'name': f'object_{i}'} for i in range(8)]}
            results = predictor(**predictator_args)
            cap_cnt = cv2.VideoCapture(src_video)
            total_frames = int(cap_cnt.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_cnt.release()
            frame_count = 0
            for r in results:
                orig_img = r.orig_img if hasattr(r, 'orig_img') and r.orig_img is not None else None
                if orig_img is None:
                    cap_t = cv2.VideoCapture(src_video)
                    cap_t.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret_t, orig_img = cap_t.read()
                    cap_t.release()
                    if not ret_t:
                        orig_img = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    if len(orig_img.shape) == 2:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
                    elif orig_img.shape[2] == 4:
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), orig_img)
                coco_data['images'].append({'id': frame_count, 'file_name': f"frame_{frame_count:06d}.jpg",
                                          'width': width, 'height': height, 'frame_count': frame_count})
                frame_annotations = []
                if hasattr(r, 'masks') and r.masks is not None:
                    masks_tensor = r.masks.data
                    if masks_tensor is not None and len(masks_tensor) > 0:
                        confs = None
                        if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'conf'):
                            confs = r.boxes.conf.cpu().numpy()
                        current_masks = []
                        current_bboxes = []
                        for mask in masks_tensor:
                            mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
                            mh, mw = mask_np.shape[-2:]
                            if mh != height or mw != width:
                                mask_np = cv2.resize(mask_np.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
                            mask_binary = (mask_np > 0.5).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for contour in contours:
                                if len(contour) >= 3:
                                    polygon = contour.squeeze().flatten().tolist()
                                    x_coords, y_coords = polygon[0::2], polygon[1::2]
                                    x_min, x_max = min(x_coords), max(x_coords)
                                    y_min, y_max = min(y_coords), max(y_coords)
                                    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                                    area = cv2.contourArea(contour)
                                    if area > 0:
                                        current_masks.append(mask_binary)
                                        current_bboxes.append(bbox)
                        if current_masks:
                            current_masks, current_bboxes = merge_masks_in_frame(current_masks, current_bboxes, merge_iou_val)
                            track_ids = track_manager.update(current_masks, current_bboxes, frame_count)
                            for idx, (mask, bbox) in enumerate(zip(current_masks, current_bboxes)):
                                mask_binary = (mask > 0.5).astype(np.uint8)
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for contour in contours:
                                    if len(contour) >= 3:
                                        polygon = contour.squeeze().flatten().tolist()
                                        area = cv2.contourArea(contour)
                                        track_id = track_ids[idx] if idx < len(track_ids) else annotation_id[0]
                                        confidence = float(confs[idx]) if confs is not None and idx < len(confs) else float(mask.max())
                                        ann = {'id': annotation_id[0], 'track_id': track_id, 'image_id': frame_count,
                                              'category_id': track_id, 'bbox': bbox, 'area': float(area),
                                              'segmentation': [polygon], 'iscrowd': 0, 'confidence': confidence}
                                        coco_data['annotations'].append(ann)
                                        frame_annotations.append(ann)
                                        annotation_id[0] += 1
                with open(labels_dir / f"frame_{frame_count:06d}.json", 'w') as f:
                    json.dump(frame_annotations, f)
                frame_count += 1
                pct = int(frame_count / max(total_frames, 1) * 100)
                yield f"data: {json.dumps({'type': 'progress', 'frame': frame_count, 'total': total_frames, 'percent': pct})}\n\n"
            with open(temp_data / 'annotations.json', 'w') as f:
                json.dump(coco_data, f)
            yield f"data: {json.dumps({'type': 'done', 'frames': frame_count})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/show_viewer', methods=['POST'])
def show_viewer():
    data = request.json or {}
    temp_data = TEMP_DATA_DIR
    if not temp_data.exists():
        return jsonify({'error': 'temp_data 不存在'}), 400
    if TEMP_DATA_MID_DIR.exists():
        shutil.rmtree(TEMP_DATA_MID_DIR)
    shutil.copytree(temp_data, TEMP_DATA_MID_DIR)
    category_mappings = data.get('category_mappings', [])
    trace_id_changes = data.get('trace_id_changes', [])
    if trace_id_changes:
        mappings_file = TEMP_DATA_MID_DIR / "trace_id_changes.json"
        with open(mappings_file, 'w') as f:
            json.dump(trace_id_changes, f)
    if trace_id_changes:
        _apply_trace_id_mappings()
    return jsonify({'msg': '已加载到 temp_data_mid', 'frames': len(list((TEMP_DATA_MID_DIR / 'frames').glob('*.jpg')))})


def _apply_trace_id_mappings():
    mappings_file = TEMP_DATA_MID_DIR / "trace_id_changes.json"
    if not mappings_file.exists():
        return
    with open(mappings_file) as f:
        mappings = json.load(f)
    mappings_list = []
    for m in mappings:
        parts = m.replace("ID:", "").split("→")
        if len(parts) == 2:
            try:
                old_id = int(parts[0].strip())
                new_id = int(parts[1].strip())
                mappings_list.append((old_id, new_id))
            except ValueError:
                pass
    labels_dir = TEMP_DATA_MID_DIR / "labels"
    annotations_file = TEMP_DATA_MID_DIR / "annotations.json"
    for label_file in sorted(labels_dir.glob("frame_*.json")):
        with open(label_file) as f:
            frame_anns = json.load(f)
        changed = False
        for ann in frame_anns:
            for old_id, new_id in mappings_list:
                if ann.get('track_id') == old_id:
                    ann['track_id'] = new_id
                    changed = True
        if changed:
            with open(label_file, 'w') as f:
                json.dump(frame_anns, f)
    if annotations_file.exists():
        with open(annotations_file) as f:
            coco = json.load(f)
        changed = False
        for ann in coco.get('annotations', []):
            for old_id, new_id in mappings_list:
                if ann.get('track_id') == old_id:
                    ann['track_id'] = new_id
                    changed = True
        if changed:
            with open(annotations_file, 'w') as f:
                json.dump(coco, f)


@app.route('/api/get_frame/<int:frame_idx>')
def get_frame(frame_idx):
    frames_dir = TEMP_DATA_MID_DIR / "frames"
    labels_dir = TEMP_DATA_MID_DIR / "labels"
    frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
    if not frame_path.exists():
        return jsonify({'error': '帧不存在'}), 404
    frame = cv2.imread(str(frame_path))
    h, w = frame.shape[:2]
    label_path = labels_dir / f"frame_{frame_idx:06d}.json"
    annotations = []
    if label_path.exists():
        with open(label_path) as f:
            annotations = json.load(f)
    result_img = frame.copy()
    for ann in annotations:
        track_id = ann.get('track_id', 0)
        if track_id == 0:
            continue
        color = get_color_for_track_id(track_id)
        polygon = ann.get('segmentation')
        if polygon:
            pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(result_img, [pts], True, color, 2)
            M = cv2.moments(pts)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(result_img, str(track_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    _, buf = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


@app.route('/api/get_video_info')
def get_video_info():
    frames_dir = TEMP_DATA_MID_DIR / "frames"
    annotations_file = TEMP_DATA_MID_DIR / "annotations.json"
    if not annotations_file.exists():
        return jsonify({'total_frames': 0})
    with open(annotations_file) as f:
        coco = json.load(f)
    total_frames = len(list(frames_dir.glob('*.jpg')))
    return jsonify({'total_frames': total_frames, 'coco': coco})


@app.route('/api/assign_trace_id', methods=['POST'])
def assign_trace_id():
    data = request.json
    frame_idx = data.get('frame_idx')
    old_track_id = data.get('old_track_id')
    new_track_id = data.get('new_track_id')
    labels_dir = TEMP_DATA_MID_DIR / "labels"
    label_file = labels_dir / f"frame_{frame_idx:06d}.json"
    if not label_file.exists():
        return jsonify({'error': 'label不存在'}), 400
    with open(label_file) as f:
        frame_anns = json.load(f)
    for ann in frame_anns:
        if ann.get('track_id') == old_track_id:
            ann['track_id'] = new_track_id
            break
    with open(label_file, 'w') as f:
        json.dump(frame_anns, f)
    return jsonify({'msg': '已赋值'})


@app.route('/api/bidirectional_annotate', methods=['POST'])
def bidirectional_annotate():
    data = request.json
    prompt_frame_idx = data.get('prompt_frame_idx')
    prompt_bboxes = data.get('prompt_bboxes', [])
    forward_enabled = data.get('forward_enabled', True)
    backward_enabled = data.get('backward_enabled', True)
    iou_val = float(data.get('iou', '0.5'))
    merge_iou_val = float(data.get('merge_iou', '0.5'))
    try:
        import torch
        temp_mid = TEMP_DATA_MID_DIR
        mid_frames_dir = temp_mid / "frames"
        mid_labels_dir = temp_mid / "labels"
        mid_annotations_file = temp_mid / "annotations.json"
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        half = device == 'cuda'
        overrides = dict(conf=0.25, task="segment", mode="predict", model=SAM_MODEL_PATH,
                        device=device, half=half, save=False, verbose=False)
        if device == 'cuda':
            overrides['batch'] = 1
            overrides['stream_buffer'] = False
        elif device == 'mps':
            overrides['half'] = True
            overrides['amp'] = True
            overrides['stream_buffer'] = True
        from ultralytics.models.sam import SAM3VideoPredictor
        predictor = SAM3VideoPredictor(overrides=overrides)
        sample_frame = cv2.imread(str(mid_frames_dir / f"frame_{0:06d}.jpg"))
        height, width = sample_frame.shape[:2]
        occupied_bands = set()
        if mid_annotations_file.exists():
            with open(mid_annotations_file) as f:
                coco = json.load(f)
            for ann in coco.get('annotations', []):
                tid = ann.get('track_id', 0)
                occupied_bands.add((tid // 10000) * 10000)
        options = list(range(10000, 501000, 10000))
        available = [opt for opt in options if (opt // 10000) * 10000 not in occupied_bands]
        if not available:
            return jsonify({'error': '所有track_id选项都已被占用'}), 400
        FIRST_ID = available[0]
        from annotate_video import merge_masks_in_frame, TrackManager
        manager = TrackManager(iou_threshold=iou_val)
        manager.next_track_id = FIRST_ID
        total = len(list(mid_frames_dir.glob('*.jpg')))
        all_new_anns = []
        if forward_enabled:
            start = prompt_frame_idx + 1
            if start < total:
                result = _process_clip(mid_frames_dir, mid_labels_dir, predictor, manager, merge_iou_val,
                                      start, total, True, prompt_bboxes, height, width)
                all_new_anns.extend(result)
        if backward_enabled:
            if prompt_frame_idx > 0:
                result = _process_clip(mid_frames_dir, mid_labels_dir, predictor, manager, merge_iou_val,
                                      0, prompt_frame_idx, False, prompt_bboxes, height, width)
                all_new_anns.extend(result)
        for ann in all_new_anns:
            ann['category_id'] = ann['track_id']
        if mid_annotations_file.exists():
            with open(mid_annotations_file) as f:
                coco = json.load(f)
        else:
            coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}
        max_ann_id = max([a['id'] for a in coco.get('annotations', [])], default=0)
        max_track_id = max([a['track_id'] for a in coco.get('annotations', [])], default=FIRST_ID - 1)
        for ann in all_new_anns:
            max_ann_id += 1
            max_track_id += 1
            ann['id'] = max_ann_id
            ann['track_id'] = max_track_id
            ann['category_id'] = max_track_id
            coco['annotations'].append(ann)
        with open(mid_annotations_file, 'w') as f:
            json.dump(coco, f)
        return jsonify({'msg': f'双向标注完成，新增{len(all_new_anns)}条', 'FIRST_ID': FIRST_ID})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _process_clip(mid_frames_dir, mid_labels_dir, predictor, manager, merge_iou_val,
                  start_frame, end_frame, forward, prompt_bboxes, height, width):
    temp_frames = Path("temp_inject")
    temp_frames.mkdir(exist_ok=True)
    frame_count = end_frame - start_frame
    if forward:
        for i in range(start_frame, end_frame):
            src = mid_frames_dir / f"frame_{i:06d}.jpg"
            dst = temp_frames / f"frame_{i - start_frame:06d}.jpg"
            shutil.copy2(src, dst)
    else:
        for rev_idx, i in enumerate(range(end_frame - 1, start_frame - 1, -1)):
            src = mid_frames_dir / f"frame_{i:06d}.jpg"
            dst = temp_frames / f"frame_{rev_idx:06d}.jpg"
            shutil.copy2(src, dst)
    clip_path = str(temp_frames / "clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, 30, (width, height))
    for i in range(frame_count):
        frame = cv2.imread(str(temp_frames / f"frame_{i:06d}.jpg"))
        out.write(frame)
    out.release()
    results = predictor(source=clip_path, stream=True, bboxes=prompt_bboxes, labels=[1]*len(prompt_bboxes))
    all_anns = []
    ann_id = manager.next_track_id
    for r in results:
        orig_img = r.orig_img if hasattr(r, 'orig_img') and r.orig_img is not None else None
        if orig_img is None:
            orig_img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            if len(orig_img.shape) == 2:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            elif orig_img.shape[2] == 4:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
        frame_anns = []
        if hasattr(r, 'masks') and r.masks is not None:
            masks_tensor = r.masks.data
            if masks_tensor is not None and len(masks_tensor) > 0:
                confs = None
                if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'conf'):
                    confs = r.boxes.conf.cpu().numpy()
                current_masks = []
                current_bboxes = []
                for mask in masks_tensor:
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
                    mh, mw = mask_np.shape[-2:]
                    if mh != height or mw != width:
                        mask_np = cv2.resize(mask_np.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
                    mask_binary = (mask_np > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) >= 3:
                            polygon = contour.squeeze().flatten().tolist()
                            x_coords, y_coords = polygon[0::2], polygon[1::2]
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                            area = cv2.contourArea(contour)
                            if area > 0:
                                current_masks.append(mask_binary)
                                current_bboxes.append(bbox)
                if current_masks:
                    current_masks, current_bboxes = merge_masks_in_frame(current_masks, current_bboxes, merge_iou_val)
                    track_ids = manager.update(current_masks, current_bboxes, 0)
                    for idx, (mask, bbox) in enumerate(zip(current_masks, current_bboxes)):
                        mask_binary = (mask > 0.5).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) >= 3:
                                polygon = contour.squeeze().flatten().tolist()
                                area = cv2.contourArea(contour)
                                track_id = track_ids[idx] if idx < len(track_ids) else ann_id
                                confidence = float(confs[idx]) if confs is not None and idx < len(confs) else float(mask.max())
                                ann = {'id': ann_id, 'track_id': track_id, 'image_id': 0,
                                      'bbox': bbox, 'area': float(area), 'segmentation': [polygon],
                                      'iscrowd': 0, 'confidence': confidence}
                                all_anns.append(ann)
                                frame_anns.append(ann)
                                ann_id += 1
        shutil.rmtree(temp_frames, ignore_errors=True)
    return all_anns


@app.route('/api/export_to_post', methods=['POST'])
def export_to_post():
    data = request.json or {}
    category_mappings = data.get('category_mappings', [])
    try:
        data_dir = TEMP_DATA_MID_DIR
        annotations_file = data_dir / "annotations.json"
        if not data_dir.exists():
            return jsonify({'error': 'temp_data_mid不存在'}), 400
        with open(annotations_file) as f:
            coco_data = json.load(f)
        total_frames = len(coco_data.get('images', []))
        output_path = TEMP_DATA_POST_DIR
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        output_labels_dir = output_path / "labels"
        output_frames_dir = output_path / "frames"
        output_labels_dir.mkdir(exist_ok=True)
        output_frames_dir.mkdir(exist_ok=True)
        labels_dir = data_dir / "labels"
        frames_dir = data_dir / "frames"
        for i in range(total_frames):
            frame_path = frames_dir / f"frame_{i:06d}.jpg"
            if frame_path.exists():
                shutil.copy2(frame_path, output_frames_dir / f"frame_{i:06d}.jpg")
            label_path = labels_dir / f"frame_{i:06d}.json"
            output_label_path = output_labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path) as f:
                    annotations = json.load(f)
                track_id_filtered = [ann for ann in annotations if ann.get('track_id', 0) >= 999999]
                frame_anns = []
                for ann in track_id_filtered:
                    tid = ann.get('track_id', 0)
                    if 1000000 <= tid <= 1000007:
                        idx = tid - 1000000
                        name = category_mappings[idx] if idx < len(category_mappings) else "Detect"
                        ann_copy = dict(ann)
                        ann_copy['category_id'] = idx
                        ann_copy['category'] = name
                        frame_anns.append(ann_copy)
                    elif tid >= 1000000:
                        ann_copy = dict(ann)
                        ann_copy['category_id'] = tid - 1000000
                        ann_copy['category'] = "Detect"
                        frame_anns.append(ann_copy)
                with open(output_label_path, 'w') as f:
                    json.dump(frame_anns, f)
        with open(output_path / 'annotations.json', 'w') as f:
            json.dump(coco_data, f)
        return jsonify({'msg': f'导出完成，{total_frames}帧'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_trace_id_changes', methods=['POST'])
def save_trace_id_changes():
    data = request.json
    mappings = data.get('mappings', [])
    mappings_file = TEMP_DATA_MID_DIR / "trace_id_changes.json"
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f)
    _apply_trace_id_mappings()
    return jsonify({'msg': '已保存并应用映射'})


@app.route('/api/load_trace_id_changes')
def load_trace_id_changes():
    mappings_file = TEMP_DATA_MID_DIR / "trace_id_changes.json"
    if mappings_file.exists():
        with open(mappings_file) as f:
            mappings = json.load(f)
        return jsonify({'mappings': mappings})
    return jsonify({'mappings': []})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)
