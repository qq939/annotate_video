#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web标注工具 - 只保留路由和UI，业务逻辑在app_utils.py"""

import os
import sys
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
import base64
import subprocess
import traceback

from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

# --- 路径常量 ---
TEMP_DATA_DIR = Path("temp_data")
TEMP_DATA_MID_DIR = Path("temp_data_mid")
TEMP_DATA_POST_DIR = Path("temp_data_post")
SRC_VIDEO_DIR = Path("1src")
DST_VIDEO_DIR = Path("1dst")
SAM_MODEL_PATH = "sam3.pt"

for _d in [TEMP_DATA_DIR, TEMP_DATA_MID_DIR, TEMP_DATA_POST_DIR, SRC_VIDEO_DIR, DST_VIDEO_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# --- 颜色常量 ---
PALETTE_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
    (255, 255, 0), (255, 0, 255), (0, 128, 255), (255, 128, 0),
    (128, 0, 255), (0, 255, 128), (255, 0, 128), (128, 255, 0),
    (0, 128, 128), (128, 0, 128), (128, 128, 0), (64, 64, 255),
]
BOX_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# --- 全局状态 ---
state = {'total_frames': 0, 'video_name': '', 'mappings': [], 'fence_mode': False, 'fences': []}


def get_color_for_track_id(tid):
    """根据track_id获取颜色（工具函数）"""
    if tid >= 1000000:
        return PALETTE_COLORS[0]
    idx = (tid % (len(PALETTE_COLORS) - 1)) + 1
    return PALETTE_COLORS[idx % len(PALETTE_COLORS)]


# ==================== UI路由 ====================
@app.route('/')
def index():
    return render_template('web_app.html')


# ==================== 视频标注（SSE实时流） ====================
@app.route('/api/start_annotate', methods=['POST'])
def start_annotate():
    """SSE方式执行视频标注"""
    data = request.json or {}
    video_name = data.get('video_name', '')
    bboxes = data.get('bboxes', [])
    items_text = data.get('items', '')
    find_list = [s.strip() for s in items_text.split(',') if s.strip()]
    iou = float(data.get('iou', 0.5))
    merge_iou = float(data.get('merge_iou', 0.5))

    def generate():
        yield 'data: ' + json.dumps({'type': 'start', 'msg': '开始处理...'}) + '\n\n'
        try:
            from app_utils import get_sam_overrides, patch_sam3_video_semantic, TEMP_DATA_DIR
            
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
            fourcc_str = ''.join([chr((fourcc_int >> 24) & 0xFF), chr((fourcc_int >> 16) & 0xFF), chr((fourcc_int >> 8) & 0xFF), chr(fourcc_int & 0xFF)])
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            overrides, device = get_sam_overrides(device='auto', model_path=SAM_MODEL_PATH)
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
                src_dst = SRC_VIDEO_DIR / "input_source.mp4"
                shutil.copy2(src_video, str(src_dst))
                source = str(src_dst)
                predictor_args = {'source': source, 'stream': True, 'bboxes': bboxes, 'labels': [1] * len(bboxes)}
            else:
                source = src_video
                predictor_args = {'source': source, 'stream': True}
                if find_list:
                    predictor_args['text'] = find_list

            yield 'data: ' + json.dumps({'type': 'progress', 'msg': '加载模型...'}) + '\n\n'

            from annotate_video import merge_masks_in_frame, TrackManager
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
                yield 'data: ' + json.dumps({'type': 'progress', 'frame': frame_count, 'total': total, 'percent': int(frame_count / max(total, 1) * 100), 'msg': f'帧 {frame_count}/{total}: contours={debug_contours}, annotations={len(frame_anns)}'}) + '\n\n'

            with open(temp_data / 'annotations.json', 'w') as f_out:
                json.dump(coco_data, f_out)

            state['total_frames'] = frame_count
            yield 'data: ' + json.dumps({'type': 'done', 'frames': frame_count}) + '\n\n'
        except Exception as e:
            traceback.print_exc()
            yield 'data: ' + json.dumps({'type': 'error', 'msg': str(e)}) + '\n\n'

    return Response(generate(), mimetype='text/event-stream')


# ==================== 查看器 ====================
@app.route('/api/show_viewer', methods=['POST'])
def show_viewer():
    """将temp_data拷贝到temp_data_mid，应用trace_id变更"""
    from app_utils import copy_temp_data, apply_trace_id_mappings, TEMP_DATA_DIR, TEMP_DATA_MID_DIR
    
    if not TEMP_DATA_DIR.exists():
        return jsonify({'error': 'temp_data 不存在'}), 400

    data = request.json or {}
    mappings = data.get('trace_id_changes', [])
    
    copy_temp_data(TEMP_DATA_DIR, TEMP_DATA_MID_DIR)
    
    if mappings:
        ml = []
        for m in mappings:
            parts = m.replace("ID:", "").split("→")
            if len(parts) == 2:
                try:
                    ml.append((int(parts[0].strip()), int(parts[1].strip())))
                except Exception:
                    pass
        if ml:
            apply_trace_id_mappings(ml)
        state['mappings'] = mappings

    frames = sorted((TEMP_DATA_MID_DIR / 'frames').glob('*.jpg'))
    return jsonify({'frames': len(frames), 'total': len(frames)})


# ==================== 帧渲染 ====================
@app.route('/api/get_frame/<int:idx>')
def get_frame(idx):
    fd = TEMP_DATA_MID_DIR / "frames"
    ld = TEMP_DATA_MID_DIR / "labels"
    fp = fd / f"frame_{idx:06d}.jpg"
    if not fp.exists():
        return jsonify({'error': '帧不存在'}), 404

    frame = cv2.imread(str(fp))
    if frame is None:
        return jsonify({'error': '无法读取帧'}), 500
    res = frame.copy()
    h, w = frame.shape[:2]
    lp = ld / f"frame_{idx:06d}.json"
    frame_anns = []
    if lp.exists():
        with open(lp) as f:
            frame_anns = json.load(f)
        for a in frame_anns:
            tid = a.get('track_id', 0)
            if tid == 0:
                continue
            color = get_color_for_track_id(tid)
            seg = a.get('segmentation')
            if seg:
                pts = np.array(seg[0], dtype=np.int32).reshape(-1, 2)
                cv2.polylines(res, [pts], True, color, 2)
                M = cv2.moments(pts)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(res, str(tid), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            bbox = a.get('bbox')
            if bbox:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(res, (x, y), (x + bw, y + bh), color, 1)

    _, buf = cv2.imencode('.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({'image': base64.b64encode(buf).decode(), 'frame': idx, 'width': w, 'height': h, 'annotations': frame_anns})


@app.route('/api/get_video_info')
def get_video_info_route():
    af = TEMP_DATA_MID_DIR / "annotations.json"
    if not af.exists():
        return jsonify({'total': 0, 'annotations': []})
    frames = sorted((TEMP_DATA_MID_DIR / 'frames').glob('*.jpg'))
    with open(af) as f:
        coco = json.load(f)
    return jsonify({'total': len(frames), 'annotations': coco.get('annotations', [])})


# ==================== 双向标注 ====================
@app.route('/api/bidirectional', methods=['POST'])
def bidirectional():
    from app_utils import run_bidirectional_inject, first_available_track_id, TEMP_DATA_MID_DIR
    
    data = request.json
    prompt_idx = data.get('prompt_frame_idx', 0)
    bboxes = data.get('bboxes', [])
    forward_en = data.get('forward_enabled', True)
    backward_en = data.get('backward_enabled', True)
    iou = float(data.get('iou', 0.5))
    merge_iou = float(data.get('merge_iou', 0.5))

    try:
        ma = TEMP_DATA_MID_DIR / "annotations.json"
        if ma.exists():
            with open(ma) as f:
                coco = json.load(f)
        else:
            coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}
        
        first_id = first_available_track_id(coco, 1000000)
        total = len(list((TEMP_DATA_MID_DIR / "frames").glob('*.jpg')))
        
        success, msg = run_bidirectional_inject(prompt_idx=prompt_idx, total_frames=total, bboxes=bboxes, forward_enabled=forward_en, backward_enabled=backward_en, iou_threshold=iou, merge_iou_threshold=merge_iou, first_id=first_id, temp_mid_dir=TEMP_DATA_MID_DIR)
        
        return jsonify({'msg': msg, 'FIRST_ID': first_id}) if success else jsonify({'error': msg}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== 提示帧标注（单帧） ====================
@app.route('/api/prompt_frame', methods=['POST'])
def prompt_frame():
    from app_utils import run_prompt_frame, save_frame_annotations, first_available_track_id, get_sam_overrides, TEMP_DATA_MID_DIR
    
    data = request.json
    prompt_idx = data.get('prompt_frame_idx', 0)
    bboxes = data.get('bboxes', [])
    items_text = data.get('items', '')
    find_list = [s.strip() for s in items_text.split(',') if s.strip()]

    try:
        mf = TEMP_DATA_MID_DIR / "frames"
        ml_dir = TEMP_DATA_MID_DIR / "labels"
        ma = TEMP_DATA_MID_DIR / "annotations.json"

        sf_path = mf / f"frame_{prompt_idx:06d}.jpg"
        if not sf_path.exists():
            return jsonify({'error': f'提示帧 {prompt_idx} 不存在'}), 400

        if ma.exists():
            with open(ma) as f:
                coco = json.load(f)
        else:
            coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}
        
        first_id = first_available_track_id(coco, 1000000)
        overrides, device = get_sam_overrides(device='auto', model_path=SAM_MODEL_PATH)
        annotations, new_first_id = run_prompt_frame(sf_path, bboxes, find_list, overrides, first_id)
        
        if annotations:
            save_frame_annotations(prompt_idx, annotations, ml_dir, ma)
        
        return jsonify({'msg': f'提示帧完成，新增{len(annotations)}条', 'count': len(annotations), 'FIRST_ID': first_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== 导出 ====================
@app.route('/api/export', methods=['POST'])
def export_route():
    from app_utils import export_to_temp_data_post, TEMP_DATA_MID_DIR
    
    data = request.json or {}
    cat_maps = data.get('category_mappings', ['Detect'] * 8)

    if not TEMP_DATA_MID_DIR.exists():
        return jsonify({'error': 'temp_data_mid 不存在'}), 400

    success, msg = export_to_temp_data_post(cat_maps)
    return jsonify({'msg': msg})


# ==================== Trace ID 管理 ====================
@app.route('/api/save_mappings', methods=['POST'])
def save_mappings():
    from app_utils import apply_trace_id_mappings, TEMP_DATA_MID_DIR
    
    data = request.json
    mappings = data.get('mappings', [])
    
    mappings_file = TEMP_DATA_MID_DIR / "trace_id_changes.json"
    mappings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f)

    ml = []
    for m in mappings:
        parts = m.replace("ID:", "").split("→")
        if len(parts) == 2:
            try:
                ml.append((int(parts[0].strip()), int(parts[1].strip())))
            except Exception:
                pass
    if ml:
        apply_trace_id_mappings(ml)
    
    state['mappings'] = mappings
    return jsonify({'msg': '已保存'})


@app.route('/api/load_mappings')
def load_mappings():
    from app_utils import TEMP_DATA_MID_DIR
    
    mf = TEMP_DATA_MID_DIR / "trace_id_changes.json"
    if mf.exists():
        with open(mf) as f:
            return jsonify({'mappings': json.load(f)})
    return jsonify({'mappings': []})


@app.route('/api/delete_track_id', methods=['POST'])
def delete_track_id():
    from app_utils import mark_track_ids_deleted
    
    data = request.json
    track_id = int(data.get('track_id', 0))
    if track_id <= 0:
        return jsonify({'error': '无效track_id'}), 400

    count = mark_track_ids_deleted([track_id])
    return jsonify({'msg': f'已删除 track_id={track_id}，影响 {count} 帧'})


# ==================== 保存视频 + OBS上传 ====================
@app.route('/api/save_video', methods=['POST'])
def save_video():
    from app_utils import TEMP_DATA_POST_DIR, DST_VIDEO_DIR
    
    data = request.json or {}
    alpha = float(data.get('alpha', 0.5))
    color_index = int(data.get('color_index', 0))
    category_name = data.get('category', 'Detect')

    input_path = TEMP_DATA_POST_DIR
    output_path = DST_VIDEO_DIR / "output.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        return jsonify({'error': 'temp_data_post 不存在，请先导出'}), 400

    annotations_path = input_path / "annotations.json"
    if not annotations_path.exists():
        return jsonify({'error': 'annotations.json 不存在'}), 400

    with open(annotations_path) as f:
        coco_data = json.load(f)

    video_info = coco_data.get('info', {})
    total_frames = len(coco_data.get('images', []))
    if total_frames == 0:
        return jsonify({'error': '没有帧数据'}), 400

    width = int(video_info.get('width', 1280))
    height = int(video_info.get('height', 720))
    fps = int(video_info.get('fps', 30))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    labels_dir = input_path / "labels"
    frames_dir = input_path / "frames"

    for i in range(total_frames):
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        label_path = labels_dir / f"frame_{i:06d}.json"
        if label_path.exists():
            with open(label_path) as f:
                annotations = json.load(f)

            overlay = frame.copy()
            for ann in annotations:
                polygon = ann.get('segmentation')
                bbox = ann.get('bbox')
                if not bbox:
                    continue

                tid = ann.get('track_id', 0)
                if tid == 1000000:
                    color = PALETTE_COLORS[color_index]
                else:
                    palette_idx = (tid % (len(PALETTE_COLORS) - 1)) + 1
                    idx = (palette_idx + color_index) % len(PALETTE_COLORS)
                    color = PALETTE_COLORS[idx]
                cat = ann.get('category', category_name)
                conf = ann.get('confidence', 1.0)

                if polygon:
                    pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)

                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(overlay, f"{cat} {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        out.write(frame)

    out.release()

    upload_result = ""
    try:
        result = subprocess.run(['curl', '--upload-file', str(output_path), 'http://obs.dimond.top/dst.mp4'], capture_output=True, text=True, timeout=120)
        upload_result = "上传成功" if result.returncode == 0 else f"上传失败: {result.stderr[:200]}"
    except Exception as e:
        upload_result = f"上传失败: {str(e)}"

    return jsonify({'msg': f'视频已保存: {output_path}', 'path': str(output_path), 'upload': upload_result})


# ==================== 围栏管理 ====================
@app.route('/api/fence/state', methods=['GET'])
def get_fence_state():
    return jsonify({'active': state.get('fence_mode', False), 'fences': state.get('fences', [])})


@app.route('/api/fence/toggle', methods=['POST'])
def toggle_fence():
    state['fence_mode'] = not state.get('fence_mode', False)
    return jsonify({'active': state['fence_mode']})


@app.route('/api/fence/clear', methods=['POST'])
def clear_fence():
    state['fences'] = []
    state['fence_mode'] = False
    return jsonify({'msg': '已清空围栏'})


@app.route('/api/fence/add_point', methods=['POST'])
def add_fence_point():
    data = request.json
    idx = int(data.get('fence_idx', 0))
    x = float(data.get('x', 0))
    y = float(data.get('y', 0))
    fences = state.get('fences', [])
    while len(fences) <= idx:
        fences.append({'points': [], 'mode': True})
    fences[idx]['points'].append([x, y])
    state['fences'] = fences
    return jsonify({'fences': fences})


# ==================== 获取帧标注（用于点击检测） ====================
@app.route('/api/frame_annotations/<int:idx>')
def get_frame_annotations(idx):
    from app_utils import load_frame_annotations, TEMP_DATA_MID_DIR
    
    ld = TEMP_DATA_MID_DIR / "labels"
    anns = load_frame_annotations(idx, ld)
    return jsonify({'annotations': anns})


# ==================== Debug日志 ====================
@app.route('/api/debug_log', methods=['GET', 'POST'])
def debug_log():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / "debug.log"
    
    if request.method == 'GET':
        if not log_file.exists():
            log_file.write_text('')
        return jsonify({'path': str(log_file)})
    else:
        data = request.json
        action = data.get('action', '')
        if action == 'append':
            msg = data.get('msg', '') + '\n'
            with open(log_file, 'a') as f:
                f.write(msg)
            if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:
                log_file.write_text('')
        return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)