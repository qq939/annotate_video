#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web标注工具 - 视频和图片标注，基于 app.py 和 image_app.py 的逻辑"""

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

BOX_COLORS = [
    (0, 0, 255),    # 红 BGR
    (0, 165, 255),  # 橙 BGR
    (0, 255, 255),  # 黄 BGR
    (0, 255, 0),    # 绿 BGR
    (255, 255, 0),  # 青 BGR
    (255, 0, 0),    # 蓝 BGR
    (255, 0, 128),  # 紫 BGR
]


def _put_chinese(img, text, pos, font_size=20, color=(255, 255, 255)):
    """在图像上绘制中文文本，UTF-8编码，尝试多个字体"""
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
        print(f"[WARN] 中文渲染失败，使用ASCII: {e}")
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 20.0, color, 1)
        return img

SRC_VIDEO_DIR = Path("1src")
SRC_IMAGES_DIR = Path("1src/image")
TEMP_DATA_DIR = Path("temp_data")
TEMP_DATA_IMAGE_DIR = Path("temp_data_image")
DST_VIDEO_DIR = Path("1dst")
DST_IMAGES_DIR = Path("1dst/image")
SAM_MODEL_PATH = "sam3.pt"
IOU_THRESHOLD = 0.5
MERGE_IOU_THRESHOLD = 0.5

for _d in [SRC_VIDEO_DIR, SRC_IMAGES_DIR, TEMP_DATA_DIR, TEMP_DATA_IMAGE_DIR, DST_VIDEO_DIR, DST_IMAGES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


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
    return jsonify({
        'filename': fname,
        'width': w, 'height': h,
        'thumb': thumb
    })


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': '没有选择文件'}), 400
    fname = Path(f.filename).name
    dst = SRC_IMAGES_DIR / fname
    if dst.exists():
        dst.unlink()
    f.save(str(dst))
    img = cv2.imread(str(dst))
    if img is None:
        return jsonify({'error': '无法读取图片'}), 400
    h, w = img.shape[:2]
    _, buf = cv2.imencode('.jpg', img)
    thumb = base64.b64encode(buf).decode()
    return jsonify({
        'filename': fname,
        'width': w, 'height': h,
        'thumb': thumb
    })


@app.route('/api/run_video_annotate_stream', methods=['GET', 'POST'])
def run_video_annotate_stream():
    if request.method == 'GET':
        import json as _json
        data = {}
        for k in ['video_name', 'boxes', 'items', 'iou', 'merge_iou']:
            v = request.args.get(k)
            if v is not None:
                try:
                    data[k] = _json.loads(v)
                except Exception:
                    data[k] = v
    else:
        data = request.json or {}
    video_name = data.get('video_name')
    boxes = data.get('boxes', [])
    items_text = data.get('items', '')
    iou_val = float(data.get('iou', '0.5'))
    merge_iou_val = float(data.get('merge_iou', '0.5'))

    def generate():
        yield f"data: {json.dumps({'type':'start','msg':'开始处理...'})}\n\n"

        try:
            import torch
            src_video = str(SRC_VIDEO_DIR / video_name)
            temp_data_path = TEMP_DATA_DIR
            if temp_data_path.exists():
                shutil.rmtree(temp_data_path)
            temp_data_path.mkdir(parents=True, exist_ok=True)
            frames_dir = temp_data_path / "frames"
            labels_dir = temp_data_path / "labels"
            frames_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)

            cap = cv2.VideoCapture(src_video)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([chr(fourcc_int & 0xFF), chr((fourcc_int >> 8) & 0xFF), chr((fourcc_int >> 16) & 0xFF), chr((fourcc_int >> 24) & 0xFF)])
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            output_name = f"annotated_{Path(video_name).stem}.mp4"
            output_path = DST_VIDEO_DIR / output_name
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            find_list = [s.strip() for s in items_text.split(',') if s.strip()]
            has_text = bool(find_list)
            has_bbox = bool(boxes)
            predictor_name = "SAM3VideoSemanticPredictor" if has_text else "SAM3VideoPredictor"

            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            device_type = 'mps' if device == 'mps' else ('cuda' if torch.cuda.is_available() else 'cpu')
            half = device_type == 'cuda'
            overrides = dict(conf=0.25, task="segment", mode="predict", model=SAM_MODEL_PATH, device=device, half=half, save=False, verbose=False)
            if device_type == 'cuda':
                overrides['batch'] = 1
                overrides['stream_buffer'] = False
            elif device_type == 'mps':
                overrides['half'] = True
                overrides['amp'] = True
                overrides['stream_buffer'] = True

            try:
                from app import _patch_sam3_video_semantic
                if predictor_name == "SAM3VideoSemanticPredictor":
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
            coco_data = {
                'info': {'description': 'Video', 'video_path': src_video,
                         'fps': fps, 'width': width, 'height': height, 'fourcc': fourcc_str, 'FIND': find_list},
                'images': [], 'annotations': [],
                'categories': [{'id': i, 'name': f'object_{i}'} for i in range(8)]
            }

            results = predictor(**predictor_args)
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
                coco_data['images'].append({'id': frame_count, 'file_name': f"frame_{frame_count:06d}.jpg", 'width': width, 'height': height, 'frame_count': frame_count})

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

                annotated_frame = r.plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                if boxes:
                    for i, bbox in enumerate(boxes):
                        label = f"目标 {i + 1}"
                        color = BOX_COLORS[i % len(BOX_COLORS)]
                        x, y = int(bbox[0]), max(10, int(bbox[1]) - 10)
                        annotated_frame_rgb = _put_chinese(annotated_frame_rgb, label, (x, y), font_size=15, color=color)
                out.write(annotated_frame_rgb)
                frame_count += 1

                pct = int(frame_count / max(total_frames, 1) * 100)
                yield f"data: {json.dumps({'type':'progress','frame':frame_count,'total':total_frames,'percent':pct})}\n\n"

            with open(temp_data_path / 'annotations.json', 'w') as f:
                json.dump(coco_data, f)
            out.release()

            yield f"data: {json.dumps({'type':'done','output': f'/api/dst_video/{output_name}', 'frames': frame_count})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type':'error','msg': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/run_video_annotate', methods=['POST'])
def run_video_annotate():
    data = request.json
    video_name = data.get('video_name')
    boxes = data.get('boxes', [])
    items_text = data.get('items', '')
    iou_val = float(data.get('iou', '0.5'))
    merge_iou_val = float(data.get('merge_iou', '0.5'))

    find_list = [s.strip() for s in items_text.split(',') if s.strip()]
    has_text = bool(find_list)
    has_bbox = bool(boxes)

    if not has_text and not has_bbox:
        return jsonify({'error': '请至少框选目标或填写物品名称'}), 400

    src_video = str(SRC_VIDEO_DIR / video_name)
    temp_data_path = TEMP_DATA_DIR
    if temp_data_path.exists():
        shutil.rmtree(temp_data_path)
    temp_data_path.mkdir(parents=True, exist_ok=True)
    frames_dir = temp_data_path / "frames"
    labels_dir = temp_data_path / "labels"
    frames_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(src_video)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = ''.join([chr(fourcc_int & 0xFF), chr((fourcc_int >> 8) & 0xFF), chr((fourcc_int >> 16) & 0xFF), chr((fourcc_int >> 24) & 0xFF)])
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_name = f"annotated_{Path(video_name).stem}.mp4"
    output_path = DST_VIDEO_DIR / output_name
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    predictor_name = "SAM3VideoSemanticPredictor" if has_text else "SAM3VideoPredictor"
    print(f"[Video] 使用 {predictor_name}, boxes={boxes}, find={find_list}")

    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device_type = 'mps' if device == 'mps' else ('cuda' if torch.cuda.is_available() else 'cpu')
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

    try:
        from app import _patch_sam3_video_semantic
        if predictor_name == "SAM3VideoSemanticPredictor":
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
    coco_data = {
        'info': {'description': 'Video Annotation Dataset', 'video_path': src_video,
                 'fps': fps, 'width': width, 'height': height, 'fourcc': fourcc_str,
                 'FIND': find_list},
        'images': [], 'annotations': [],
        'categories': [{'id': i, 'name': f'object_{i}'} for i in range(8)]
    }

    results = predictor(**predictor_args)
    frame_count = 0
    total_frames = int(cv2.VideoCapture(src_video).get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.VideoCapture(src_video).release()

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
        coco_data['images'].append({
            'id': frame_count, 'file_name': f"frame_{frame_count:06d}.jpg",
            'width': width, 'height': height, 'frame_count': frame_count
        })

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
                                ann = {
                                    'id': annotation_id[0], 'track_id': track_id, 'image_id': frame_count,
                                    'category_id': track_id, 'bbox': bbox, 'area': float(area),
                                    'segmentation': [polygon], 'iscrowd': 0, 'confidence': confidence
                                }
                                coco_data['annotations'].append(ann)
                                frame_annotations.append(ann)
                                annotation_id[0] += 1

        with open(labels_dir / f"frame_{frame_count:06d}.json", 'w') as f:
            json.dump(frame_annotations, f)

        annotated_frame = r.plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        if boxes:
            for i, bbox in enumerate(boxes):
                label = f"目标 {i + 1}"
                color = BOX_COLORS[i % len(BOX_COLORS)]
                x, y = int(bbox[0]), max(10, int(bbox[1]) - 10)
                annotated_frame_rgb = _put_chinese(annotated_frame_rgb, label, (x, y), font_size=15, color=color)
        out.write(annotated_frame_rgb)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"[Video] 已处理 {frame_count}/{total_frames} 帧")

    with open(temp_data_path / 'annotations.json', 'w') as f:
        json.dump(coco_data, f)
    out.release()
    print(f"[Video] 完成: {output_path}")
    return jsonify({
        'success': True,
        'output': f'/api/dst_video/{output_name}',
        'temp_data': str(temp_data_path),
        'frames': frame_count
    })


@app.route('/api/run_image_annotate', methods=['POST'])
def run_image_annotate():
    data = request.json
    image_name = data.get('image_name')
    boxes = data.get('boxes', [])
    box_colors_data = data.get('box_colors', [])
    items_text = data.get('items', '')
    use_semantic = data.get('use_semantic', False)
    iou_val = float(data.get('iou', '0.5'))
    merge_iou_val = float(data.get('merge_iou', '0.5'))
    selected_color_idx = int(data.get('selected_color_idx', 0))

    find_list = [s.strip() for s in items_text.split(',') if s.strip()]
    has_text = bool(find_list)
    has_bbox = bool(boxes)

    if not has_text and not has_bbox:
        return jsonify({'error': '请至少框选目标或填写文本提示词'}), 400

    src_image = str(SRC_IMAGES_DIR / image_name)

    category_to_color = {}
    category_to_color[0] = BOX_COLORS[selected_color_idx % len(BOX_COLORS)]
    for i, color_bgr in enumerate(box_colors_data):
        cat_idx = i % len(find_list) if find_list else 0
        if cat_idx not in category_to_color:
            category_to_color[cat_idx] = tuple(color_bgr)

    temp_data_path = TEMP_DATA_IMAGE_DIR
    if temp_data_path.exists():
        shutil.rmtree(temp_data_path)
    temp_data_path.mkdir(parents=True, exist_ok=True)
    frames_dir = temp_data_path / "frames"
    labels_dir = temp_data_path / "labels"
    frames_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    shutil.copy2(src_image, str(frames_dir / "frame_000000.jpg"))

    img_orig = cv2.imread(src_image)
    img_h, img_w = img_orig.shape[:2]
    predictor_name = "SAM3SemanticPredictor" if use_semantic else "SAM3Predictor"
    print(f"[Image] 使用 {predictor_name}, boxes={boxes}, find={find_list}")

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
            float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)
        ] for b in boxes], dtype=np.float32)
        if use_semantic:
            from ultralytics.models.sam import SAM3SemanticPredictor
            predictor = SAM3SemanticPredictor(overrides=overrides)
            results = predictor(source=src_image, bboxes=boxes, points=center_points,
                               labels=[1] * len(boxes), text=find_list)
        else:
            from ultralytics.models.sam import SAM3Predictor
            predictor = SAM3Predictor(overrides=overrides)
            results = predictor(source=src_image, bboxes=boxes, points=center_points,
                               labels=[1] * len(boxes))
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

    if img_h == 0 or img_w == 0:
        img_h, img_w = img_orig.shape[:2]

    coco_data = {
        'info': {'description': 'Image Annotation Dataset', 'image_path': src_image,
                 'width': img_w, 'height': img_h, 'FIND': find_list},
        'images': [{'id': 0, 'file_name': image_name, 'width': img_w, 'height': img_h}],
        'annotations': [],
        'categories': (
            [{'id': i, 'name': name} for i, name in enumerate(find_list)]
            if find_list else [{'id': 0, 'name': 'object'}]
        )
    }
    frame_annotations = []
    annotation_id = [0]

    if all_masks:
        import torch
        combined = torch.cat(all_masks, dim=0)
        confs = np.concatenate(all_confs) if all_confs else None
        from annotate_video import merge_masks_in_frame

        current_masks = []
        current_bboxes = []
        for mask in combined:
            mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
            mh, mw = mask_np.shape[-2:]
            if mh != img_h or mw != img_w:
                mask_np = cv2.resize(mask_np.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_np > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            current_masks, current_bboxes = merge_masks_in_frame(current_masks, current_bboxes, merge_iou_val)
            for idx, (mask, bbox) in enumerate(zip(current_masks, current_bboxes)):
                mask_binary = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        polygon = contour.squeeze().flatten().tolist()
                        area = cv2.contourArea(contour)
                        if find_list:
                            cat_idx = idx % len(find_list)
                        else:
                            cat_idx = 0
                        confidence = float(confs[idx]) if confs is not None and idx < len(confs) else float(mask.max())
                        ann_color = category_to_color.get(cat_idx, BOX_COLORS[0])
                        ann = {
                            'id': annotation_id[0], 'track_id': annotation_id[0], 'image_id': 0,
                            'category_id': cat_idx, 'bbox': bbox, 'area': float(area),
                            'segmentation': [polygon], 'iscrowd': 0, 'confidence': confidence,
                            'color': ann_color
                        }
                        coco_data['annotations'].append(ann)
                        frame_annotations.append(ann)
                        annotation_id[0] += 1

    with open(labels_dir / "frame_000000.json", 'w') as f:
        json.dump(frame_annotations, f)
    with open(temp_data_path / 'annotations.json', 'w') as f:
        json.dump(coco_data, f)

    dst_dir = DST_IMAGES_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)
    output_path = dst_dir / image_name
    if output_path.exists():
        output_path.unlink()

    annotated_img = img_orig.copy()
    for ann in frame_annotations:
        cat_idx = ann['category_id']
        color = ann.get('color', BOX_COLORS[cat_idx % len(BOX_COLORS)])
        b = ann['bbox']
        conf = ann.get('confidence', 1.0)
        cat_name = find_list[cat_idx] if cat_idx < len(find_list) else f"obj{cat_idx}"

        overlay = annotated_img.copy()
        seg = ann.get('segmentation', [])
        if seg:
            pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(annotated_img, 0.75, overlay, 0.25, 0, annotated_img)

        cv2.rectangle(annotated_img, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), color, 1)
        label = f"{cat_name} {conf:.2f}"
        tx, ty = int(b[0]), max(14, int(b[1]))
        annotated_img = _put_chinese(annotated_img, label, (tx, ty - 12), font_size=14, color=(255, 255, 255))

    cv2.imwrite(str(output_path), annotated_img)
    print(f"[Image] 完成: {output_path}")

    return jsonify({
        'success': True,
        'output': f'/api/dst_image/{image_name}',
        'temp_data': str(temp_data_path),
        'annotations': frame_annotations
    })


@app.route('/api/dst_video/<path:filename>')
def dst_video(filename):
    return send_from_directory(DST_VIDEO_DIR, filename)


@app.route('/api/dst_image/<path:filename>')
def dst_image(filename):
    return send_from_directory(DST_IMAGES_DIR, filename)


@app.route('/api/temp_image_frame')
def temp_image_frame():
    frame_path = TEMP_DATA_IMAGE_DIR / "frames"
    files = list(frame_path.glob("*.jpg")) or list(frame_path.glob("*.png"))
    if not files:
        return jsonify({'error': '无帧'}), 404
    img = cv2.imread(str(files[0]))
    _, buf = cv2.imencode('.jpg', img)
    thumb = base64.b64encode(buf).decode()
    return jsonify({'thumb': thumb})


@app.route('/api/video_frames')
def video_frames():
    data_dir = request.args.get('dir', 'temp_data')
    data_path = Path(data_dir)
    if not data_path.exists():
        return jsonify({'error': '目录不存在'}), 400
    frames_dir = data_path / 'frames'
    labels_dir = data_path / 'labels'
    if not frames_dir.exists():
        return jsonify({'error': 'frames目录不存在'}), 400
    frame_files = sorted(frames_dir.glob('*.jpg'))
    total = len(frame_files)
    if total == 0:
        frame_files = sorted(frames_dir.glob('*.png'))
        total = len(frame_files)
    ann_file = data_path / 'annotations.json'
    conf_threshold = float(request.args.get('conf', 0.0))
    category_map = {}
    if ann_file.exists():
        with open(ann_file) as f:
            coco = json.load(f)
            for ann in coco.get('annotations', []):
                tid = ann.get('track_id', 0)
                cat = ann.get('category_id', 0)
                if tid not in category_map:
                    category_map[tid] = cat
            next_tid = max([0] + [a.get('track_id', 0) for a in coco.get('annotations', [])]) + 1
    else:
        coco = None
        next_tid = 1000000
    return jsonify({
        'total': total,
        'fps': coco.get('info', {}).get('fps', 30) if coco else 30,
        'width': coco.get('info', {}).get('width', 0) if coco else 0,
        'height': coco.get('info', {}).get('height', 0) if coco else 0,
        'next_track_id': next_tid,
        'conf_threshold': conf_threshold,
        'category_map': category_map
    })


@app.route('/api/video_frame')
def video_frame():
    data_dir = request.args.get('dir', 'temp_data')
    frame_idx = int(request.args.get('frame', 0))
    conf_th = float(request.args.get('conf', 0.0))
    zoom = float(request.args.get('zoom', 1.0))
    alpha = float(request.args.get('alpha', 0.3))
    category_names_raw = request.args.get('categories', '')
    category_names = [s.strip() for s in category_names_raw.split(',') if s.strip()]
    data_path = Path(data_dir)
    frame_path = data_path / 'frames'
    label_path = data_path / 'labels'
    frame_files = sorted(frame_path.glob('*.jpg'))
    if not frame_files:
        frame_files = sorted(frame_path.glob('*.png'))
    if not frame_files:
        return jsonify({'error': '无帧'}), 400
    if frame_idx >= len(frame_files):
        frame_idx = len(frame_files) - 1
    img = cv2.imread(str(frame_files[frame_idx]))
    if img is None:
        return jsonify({'error': '无法读取帧'}), 400
    h, w = img.shape[:2]
    label_file = label_path / f"frame_{frame_idx:06d}.json"
    anns = []
    if label_file.exists():
        with open(label_file) as f:
            anns = [a for a in json.load(f) if a.get('confidence', 1.0) >= conf_th]
    rendered = img.copy()
    for ann in anns:
        cat_idx = ann.get('category_id', 0)
        color = BOX_COLORS[cat_idx % len(BOX_COLORS)]
        seg = ann.get('segmentation', [])
        if seg:
            pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0, rendered)
        b = ann.get('bbox', [0, 0, 0, 0])
        bx, by, bw, bh = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(rendered, (bx, by), (bx + bw, by + bh), color, 1)
        cat_name = category_names[cat_idx] if cat_idx < len(category_names) else f"t{ann.get('track_id',0)}"
        conf = ann.get('confidence', 1.0)
        label = f"{cat_name} {conf:.2f}"
        rendered = _put_chinese(rendered, label, (bx, max(14, by)), font_size=14, color=(255, 255, 255))
    disp_w = int(w * zoom)
    disp_h = int(h * zoom)
    if zoom != 1.0:
        rendered = cv2.resize(rendered, (disp_w, disp_h))
    _, buf = cv2.imencode('.jpg', rendered)
    return jsonify({
        'thumb': base64.b64encode(buf).decode(),
        'frame': frame_idx,
        'total': len(frame_files),
        'w': disp_w, 'h': disp_h,
        'ann_count': len(anns)
    })


@app.route('/api/video_delete_trace', methods=['POST'])
def video_delete_trace():
    data = request.json or {}
    data_dir = data.get('dir', 'temp_data')
    trace_id = int(data.get('trace_id', 0))
    data_path = Path(data_dir)
    labels_dir = data_path / 'labels'
    ann_file = data_path / 'annotations.json'
    for lf in labels_dir.glob('*.json'):
        with open(lf) as f:
            anns = json.load(f)
        changed = False
        new_anns = []
        for a in anns:
            if a.get('track_id') == trace_id:
                changed = True
            else:
                new_anns.append(a)
        if changed:
            with open(lf, 'w') as f:
                json.dump(new_anns, f)
    if ann_file.exists():
        with open(ann_file) as f:
            coco = json.load(f)
        new_anns_coco = [a for a in coco.get('annotations', []) if a.get('track_id') != trace_id]
        if len(new_anns_coco) != len(coco.get('annotations', [])):
            coco['annotations'] = new_anns_coco
            with open(ann_file, 'w') as f:
                json.dump(coco, f)
    return jsonify({'success': True, 'trace_id': trace_id})


@app.route('/api/video_set_track_id', methods=['POST'])
def video_set_track_id():
    data = request.json or {}
    next_id = int(data.get('next_track_id', 1000000))
    return jsonify({'next_track_id': next_id})


@app.route('/api/video_categories', methods=['POST'])
def video_categories():
    data = request.json or {}
    categories = data.get('categories', {})
    return jsonify({'categories': categories})


@app.route('/api/video_export_post', methods=['POST'])
def video_export_post():
    data = request.json or {}
    src_dir = data.get('src_dir', 'temp_data')
    dst_dir = data.get('dst_dir', 'temp_data_post')
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    return jsonify({'success': True, 'dst': dst_dir})


@app.route('/api/video_save', methods=['POST'])
def video_save():
    data = request.json or {}
    input_dir = data.get('input_dir', 'temp_data_post')
    output_name = data.get('output_name', 'output.mp4')
    alpha = float(data.get('alpha', 0.3))
    conf_th = float(data.get('conf', 0.0))
    categories_raw = data.get('categories', '')
    category_names = [s.strip() for s in categories_raw.split(',') if s.strip()]
    from save import render_frames_to_video
    try:
        out_path = render_frames_to_video(input_dir, DST_VIDEO_DIR / output_name, alpha, conf_th, category_names)
        return jsonify({'success': True, 'output': f'/api/dst_video/{output_name}'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/video_prompt_frame', methods=['POST'])
def video_prompt_frame():
    data = request.json or {}
    frame_idx = int(data.get('frame_idx', 0))
    boxes = data.get('boxes', [])
    data_dir = data.get('dir', 'temp_data')
    iou_val = float(data.get('iou', 0.5))
    merge_iou_val = float(data.get('merge_iou', 0.5))
    items_text = data.get('items', '')
    find_list = [s.strip() for s in items_text.split(',') if s.strip()]
    data_path = Path(data_dir)
    frames_dir = data_path / 'frames'
    ann_file = data_path / 'annotations.json'
    cap = cv2.VideoCapture(str(list(frames_dir.glob('*.jpg'))[0]))
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if ann_file.exists():
        with open(ann_file) as f:
            coco = json.load(f)
        fps = coco.get('info', {}).get('fps', 30)
    try:
        from post_annotate import do_bidirectional_annotate
        do_bidirectional_annotate(str(data_path), frame_idx, boxes, iou_val, merge_iou_val, find_list)
        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/video_postprocess', methods=['POST'])
def video_postprocess():
    data = request.json or {}
    src_dir = data.get('src_dir', 'temp_data')
    conf_th = float(data.get('conf', 0.0))
    data_path = Path(src_dir)
    labels_dir = data_path / 'labels'
    ann_file = data_path / 'annotations.json'
    total_removed = 0
    for lf in labels_dir.glob('*.json'):
        with open(lf) as f:
            anns = json.load(f)
        before = len(anns)
        anns = [a for a in anns if a.get('confidence', 1.0) >= conf_th]
        total_removed += before - len(anns)
        with open(lf, 'w') as f:
            json.dump(anns, f)
    if ann_file.exists():
        with open(ann_file) as f:
            coco = json.load(f)
        coco['annotations'] = [a for a in coco.get('annotations', []) if a.get('confidence', 1.0) >= conf_th]
        with open(ann_file, 'w') as f:
            json.dump(coco, f)
    return jsonify({'success': True, 'removed': total_removed})


if __name__ == '__main__':
    print("=" * 60)
    print("Web标注工具启动中: http://0.0.0.0:8081")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)
