#!/usr/bin/env python3
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

    from annotate_video import merge_masks_in_frame, TrackManager, put_chinese_text
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
                annotated_frame_rgb = put_chinese_text(annotated_frame_rgb, label, (x, y), font_size=15, color=color)
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
        from annotate_video import merge_masks_in_frame, put_chinese_text

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
        annotated_img = put_chinese_text(annotated_img, label, (tx, ty - 12), font_size=14, color=(255, 255, 255))

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


if __name__ == '__main__':
    print("=" * 60)
    print("Web标注工具启动中: http://0.0.0.0:8081")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)
