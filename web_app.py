#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web标注工具 - 完整移植app.py功能，左控制面板+右可视化面板"""

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

# --- 全局路径 ---
TEMP_DATA_DIR = Path("temp_data")
TEMP_DATA_MID_DIR = Path("temp_data_mid")
TEMP_DATA_POST_DIR = Path("temp_data_post")
SRC_VIDEO_DIR = Path("1src")
DST_VIDEO_DIR = Path("1dst")
SAM_MODEL_PATH = "sam3.pt"

for _d in [TEMP_DATA_DIR, TEMP_DATA_MID_DIR, TEMP_DATA_POST_DIR, SRC_VIDEO_DIR, DST_VIDEO_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# --- 颜色常量 ---
WARM_COLORS = [
    (180, 130, 255), (200, 100, 220), (255, 50, 200), (255, 0, 180),
    (220, 0, 150), (180, 0, 120), (139, 0, 100), (100, 0, 80)
]
COLD_COLORS = [
    (100, 150, 255), (100, 200, 255), (150, 200, 255), (100, 255, 200),
    (150, 100, 255), (100, 200, 200), (150, 150, 255), (100, 180, 255)
]
BOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 0, 255),
]
PALETTE_COLORS = [
    (0, 0, 255),     # 红 (BGR)
    (0, 165, 255),   # 橙 (BGR)
    (0, 255, 255),   # 黄 (BGR)
    (0, 255, 0),     # 绿 (BGR)
    (255, 255, 0),   # 青 (BGR)
    (255, 0, 0),     # 蓝 (BGR)
    (128, 0, 128),   # 紫 (BGR)
]

# --- 全局状态 ---
_SAM3_SEMANTIC_PATCHED = False
state = {
    'current_frame': 0,
    'total_frames': 0,
    'video_info': None,
    'fences': [],
    'fence_mode': False,
    'mappings': [],
    'next_track_id': 1000000,
    'deleted_ids': set(),
    'selected_color': 0,
    'palette_colors': PALETTE_COLORS,
}


def get_color_for_track_id(track_id):
    if track_id >= 1000000:
        return WARM_COLORS[(track_id - 1000000) % len(WARM_COLORS)]
    return COLD_COLORS[track_id % len(COLD_COLORS)]


def _patch_sam3_video_semantic():
    global _SAM3_SEMANTIC_PATCHED
    if _SAM3_SEMANTIC_PATCHED:
        return
    _SAM3_SEMANTIC_PATCHED = True
    import torch
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
        if labels is None:
            _lbl_arr = np.ones(nb)
        else:
            _lbl_arr = np.array(labels)
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
        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, out

    SAM3VideoSemanticPredictor.add_prompt = _new_add_prompt


def _get_extract_dir(video_path):
    """获取与视频同名的提取帧目录"""
    return SRC_VIDEO_DIR / Path(video_path).stem


def _extract_video_to_frames(video_path, extract_dir=None):
    """将视频拆帧到指定目录，返回 (帧数, 宽, 高, fps)"""
    if extract_dir is None:
        extract_dir = _get_extract_dir(video_path)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(extract_dir / f"frame_{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    return total, width, height, fps


def _get_coco_frame_id(path):
    """从 frame_000000.jpg 路径提取帧序号"""
    try:
        return int(Path(path).stem.split('_')[1])
    except (IndexError, ValueError):
        return 0


def _first_available_track_id(coco_data, base_start=10000):
    """在coco数据中找到可用的起始track_id"""
    occupied = set()
    for ann in coco_data.get('annotations', []):
        occupied.add((ann.get('track_id', 0) // 10000) * 10000)
    opts = list(range(base_start, 501000, 10000))
    for o in opts:
        if (o // 10000) * 10000 not in occupied:
            return o
    return max(opts)


# ==================== 页面路由 ====================


@app.route('/')
def index():
    return render_template('web_app.html')


# ==================== 视频上传 ====================


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    f = request.files['video']
    if not f.filename:
        return jsonify({'error': '没有选择文件'}), 400

    fname = Path(f.filename).name
    dst = SRC_VIDEO_DIR / fname
    if dst.exists():
        dst.unlink()
    f.save(str(dst))

    cap = cv2.VideoCapture(str(dst))
    ret, frame = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ret:
        return jsonify({'error': '无法读取视频'}), 400

    state['video_info'] = {
        'filename': fname,
        'width': width,
        'height': height,
        'total': total,
        'fps': fps,
    }
    state['total_frames'] = total

    return jsonify({
        'filename': fname,
        'width': width,
        'height': height,
        'total': total,
        'fps': fps,
    })


@app.route('/api/get_video_first_frame')
def get_video_first_frame():
    """获取上传视频的第一帧（用于标注模态框预览）"""
    info = state.get('video_info')
    if not info or not info.get('filename'):
        return jsonify({'error': '没有视频'}), 400

    src = SRC_VIDEO_DIR / info['filename']
    if not src.exists():
        return jsonify({'error': '视频文件不存在'}), 404

    import base64 as b64
    cap = cv2.VideoCapture(str(src))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({'error': '无法读取视频首帧'}), 400

    # 缩放到600宽显示
    h, w = frame.shape[:2]
    scale = 600 / w if w > 600 else 1.0
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({'image': b64.b64encode(buf).decode(), 'width': frame.shape[1], 'height': frame.shape[0]})


# ==================== 执行标注 (SSE) ====================


@app.route('/api/run_annotate', methods=['POST'])
def run_annotate():
    data = request.json
    video_name = data.get('video_name', '')
    merge_iou = float(data.get('merge_iou', 0.5))
    iou = float(data.get('iou', 0.5))
    items_text = data.get('items', '')
    bboxes = data.get('bboxes', [])
    find_list = [s.strip() for s in items_text.split(',') if s.strip()]

    def generate():
        yield 'data: ' + json.dumps({'type': 'start', 'msg': '开始处理...'}) + '\n\n'
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
            fourcc_str = ''.join([
                chr((fourcc_int >> 24) & 0xFF),
                chr((fourcc_int >> 16) & 0xFF),
                chr((fourcc_int >> 8) & 0xFF),
                chr(fourcc_int & 0xFF)
            ])
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 如果用户绘制了bbox，需要把视频拷贝到1src/下（与app.py一致）
            src_dst = SRC_VIDEO_DIR / "input_source.mp4"
            cap.release()
            if bboxes or find_list:
                shutil.copy2(src_video, str(src_dst))

            device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
            half = device == 'cuda'
            overrides = {
                'conf': 0.25, 'task': "segment", 'mode': "predict",
                'model': SAM_MODEL_PATH, 'device': device, 'half': half, 'save': False, 'verbose': False
            }
            if device == 'cuda':
                overrides['batch'] = 1
                overrides['stream_buffer'] = False
            elif device == 'mps':
                overrides['half'] = True
                overrides['amp'] = True
                overrides['stream_buffer'] = True

            yield 'data: ' + json.dumps({'type': 'progress', 'msg': '加载模型...'}) + '\n\n'

            use_semantic = bool(find_list)
            if use_semantic:
                try:
                    _patch_sam3_video_semantic()
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
                # 有bbox时用视频路径 + bboxes作为提示
                source = str(src_dst)
                predictor_args = {'source': source, 'stream': True, 'bboxes': bboxes, 'labels': [1] * len(bboxes)}
            else:
                source = src_video
                predictor_args = {'source': source, 'stream': True}
                if find_list:
                    predictor_args['text'] = find_list

            from annotate_video import merge_masks_in_frame, TrackManager
            track_manager = TrackManager(iou_threshold=iou)
            ann_id_counter = [0]

            cap_cnt = cv2.VideoCapture(source)
            total = int(cap_cnt.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_cnt.release()

            coco_data = {
                'info': {
                    'fps': fps, 'width': width, 'height': height,
                    'fourcc': fourcc_str, 'FIND': find_list
                },
                'images': [], 'annotations': [], 'categories': []
            }

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

                with open(labels_dir / f"frame_{frame_count:06d}.json", 'w') as f_out:
                    json.dump(frame_anns, f_out)

                frame_count += 1
                pct = int(frame_count / max(total, 1) * 100)
                yield 'data: ' + json.dumps({
                    'type': 'progress', 'frame': frame_count, 'total': total, 'percent': pct
                }) + '\n\n'

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
    temp_data = TEMP_DATA_DIR
    if not temp_data.exists():
        return jsonify({'error': 'temp_data 不存在'}), 400

    if TEMP_DATA_MID_DIR.exists():
        shutil.rmtree(TEMP_DATA_MID_DIR)
    shutil.copytree(temp_data, TEMP_DATA_MID_DIR)

    data = request.json or {}
    mappings = data.get('trace_id_changes', [])
    if mappings:
        _apply_mappings(mappings)
        state['mappings'] = mappings

    frames = sorted((TEMP_DATA_MID_DIR / 'frames').glob('*.jpg'))
    return jsonify({'frames': len(frames), 'total': len(frames)})


def _apply_mappings(mappings):
    """将trace_id变更应用到temp_data_mid的标签和annotations.json"""
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

    if not ml:
        return

    ld = TEMP_DATA_MID_DIR / "labels"
    for lf in sorted(ld.glob("frame_*.json")):
        with open(lf) as f:
            fa = json.load(f)
        changed = False
        for a in fa:
            for o, n in ml:
                if a.get('track_id') == o:
                    a['track_id'] = n
                    changed = True
        if changed:
            with open(lf, 'w') as f:
                json.dump(fa, f)

    af = TEMP_DATA_MID_DIR / "annotations.json"
    if af.exists():
        with open(af) as f:
            coco = json.load(f)
        changed = False
        for a in coco.get('annotations', []):
            for o, n in ml:
                if a.get('track_id') == o:
                    a['track_id'] = n
                    changed = True
        if changed:
            with open(af, 'w') as f:
                json.dump(coco, f)


# ==================== 帧渲染 ====================


@app.route('/api/get_frame/<int:idx>')
def get_frame(idx):
    fd = TEMP_DATA_MID_DIR / "frames"
    ld = TEMP_DATA_MID_DIR / "labels"
    fp = fd / f"frame_{idx:06d}.jpg"
    if not fp.exists():
        return jsonify({'error': '帧不存在'}), 404

    frame = cv2.imread(str(fp))
    res = frame.copy()
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
            # 也绘制bbox
            bbox = a.get('bbox')
            if bbox:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(res, (x, y), (x + w, y + h), color, 1)

    _, buf = cv2.imencode('.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({'image': base64.b64encode(buf).decode(), 'frame': idx, 'annotations': frame_anns})


@app.route('/api/get_video_info')
def get_video_info_route():
    af = TEMP_DATA_MID_DIR / "annotations.json"
    if not af.exists():
        return jsonify({'total': 0, 'annotations': []})
    frames = sorted((TEMP_DATA_MID_DIR / 'frames').glob('*.jpg'))
    with open(af) as f:
        coco = json.load(f)
    return jsonify({
        'total': len(frames),
        'annotations': coco.get('annotations', [])
    })


# ==================== 双向标注 ====================


@app.route('/api/bidirectional', methods=['POST'])
def bidirectional():
    data = request.json
    prompt_idx = data.get('prompt_frame_idx', 0)
    bboxes = data.get('bboxes', [])
    forward_en = data.get('forward_enabled', True)
    backward_en = data.get('backward_enabled', True)
    iou = float(data.get('iou', 0.5))
    merge_iou = float(data.get('merge_iou', 0.5))

    try:
        import torch
        mf = TEMP_DATA_MID_DIR / "frames"
        ml_dir = TEMP_DATA_MID_DIR / "labels"
        ma = TEMP_DATA_MID_DIR / "annotations.json"

        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        half = device == 'cuda'
        overrides = {
            'conf': 0.25, 'task': "segment", 'mode': "predict",
            'model': SAM_MODEL_PATH, 'device': device, 'half': half, 'save': False, 'verbose': False
        }
        if device == 'cuda':
            overrides['batch'] = 1
            overrides['stream_buffer'] = False
        elif device == 'mps':
            overrides['half'] = True
            overrides['amp'] = True
            overrides['stream_buffer'] = True

        sf = cv2.imread(str(mf / "frame_000000.jpg"))
        h, w = sf.shape[:2]

        # 查找可用起始track_id
        occupied = set()
        if ma.exists():
            with open(ma) as f:
                coco = json.load(f)
            for ann in coco.get('annotations', []):
                occupied.add((ann.get('track_id', 0) // 10000) * 10000)
        opts = list(range(10000, 501000, 10000))
        avail = [o for o in opts if (o // 10000) * 10000 not in occupied]
        if not avail:
            return jsonify({'error': '所有track_id选项都已被占用'}), 400
        first_id = avail[0]

        from annotate_video import merge_masks_in_frame, TrackManager
        mgr = TrackManager(iou_threshold=iou)
        mgr.next_track_id = first_id

        total = len(list(mf.glob('*.jpg')))
        all_new_anns = []

        if forward_en and prompt_idx + 1 < total:
            result = _process_clip(mf, ml_dir, ma, predictor_overrides=overrides, mgr=mgr,
                                   merge_iou=merge_iou, start=prompt_idx + 1, end=total,
                                   fwd=True, bboxes=bboxes, h=h, w=w)
            all_new_anns.extend(result)

        if backward_en and prompt_idx > 0:
            result = _process_clip(mf, ml_dir, ma, predictor_overrides=overrides, mgr=mgr,
                                   merge_iou=merge_iou, start=0, end=prompt_idx,
                                   fwd=False, bboxes=bboxes, h=h, w=w)
            all_new_anns.extend(result)

        if ma.exists():
            with open(ma) as f:
                coco = json.load(f)
        else:
            coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}

        max_aid = max([a['id'] for a in coco.get('annotations', [])], default=0)
        for ann in all_new_anns:
            max_aid += 1
            ann['id'] = max_aid
            ann['category_id'] = ann['track_id']
            coco['annotations'].append(ann)

        with open(ma, 'w') as f:
            json.dump(coco, f)

        # 同时写入对应帧的标签文件
        for ann in all_new_anns:
            img_id = ann.get('image_id', 0)
            lf = ml_dir / f"frame_{img_id:06d}.json"
            existing = []
            if lf.exists():
                with open(lf) as f:
                    existing = json.load(f)
            existing.append(ann)
            with open(lf, 'w') as f:
                json.dump(existing, f)

        return jsonify({'msg': f'双向标注完成，新增{len(all_new_anns)}条', 'FIRST_ID': first_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _process_clip(mf, ml_dir, ma, predictor_overrides, mgr, merge_iou,
                  start, end, fwd, bboxes, h, w):
    """处理一段视频片段，返回新增的annotations列表"""
    tp = Path("temp_clip")
    if tp.exists():
        shutil.rmtree(tp)
    tp.mkdir(exist_ok=True)
    fc = end - start

    try:
        if fwd:
            for i in range(start, end):
                shutil.copy2(mf / f"frame_{i:06d}.jpg", tp / f"frame_{i - start:06d}.jpg")
        else:
            for ri, i in enumerate(range(end - 1, start - 1, -1)):
                shutil.copy2(mf / f"frame_{i:06d}.jpg", tp / f"frame_{ri:06d}.jpg")

        cp = str(tp / "clip.mp4")
        vw = cv2.VideoWriter(cp, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        for i in range(fc):
            vw.write(cv2.imread(str(tp / f"frame_{i:06d}.jpg")))
        vw.release()

        from ultralytics.models.sam import SAM3VideoPredictor
        predictor = SAM3VideoPredictor(overrides=predictor_overrides)

        from annotate_video import merge_masks_in_frame
        pred_args = {'source': cp, 'stream': True}
        if bboxes:
            pred_args['bboxes'] = bboxes
            pred_args['labels'] = [1] * len(bboxes)
        results = predictor(**pred_args)

        all_new_anns = []
        aid = mgr.next_track_id

        for r in results:
            orig = getattr(r, 'orig_img', None) or np.zeros((h, w, 3), dtype=np.uint8)
            if len(orig.shape) == 2:
                orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
            elif orig.shape[2] == 4:
                orig = cv2.cvtColor(orig, cv2.COLOR_BGRA2BGR)

            masks_attr = getattr(r, 'masks', None)
            if masks_attr is not None and masks_attr.data is not None:
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
                    cm, cb = merge_masks_in_frame(cm, cb, merge_iou)
                    tids = mgr.update(cm, cb, 0)
                    for idx, (m, b) in enumerate(zip(cm, cb)):
                        mb = (m > 0.5).astype(np.uint8)
                        contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if len(cnt) >= 3:
                                poly = cnt.squeeze().flatten().tolist()
                                area = cv2.contourArea(cnt)
                                tid = tids[idx] if idx < len(tids) else aid
                                conf_val = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                                ann = {
                                    'id': aid, 'track_id': tid, 'image_id': 0,
                                    'bbox': b, 'area': float(area),
                                    'segmentation': [poly], 'iscrowd': 0,
                                    'confidence': conf_val
                                }
                                all_new_anns.append(ann)
                                aid += 1

        # 将image_id映射回原始帧号
        for ann in all_new_anns:
            raw_idx = ann['image_id']
            if fwd:
                ann['image_id'] = start + raw_idx
            else:
                ann['image_id'] = end - 1 - raw_idx

        return all_new_anns
    finally:
        if tp.exists():
            shutil.rmtree(tp, ignore_errors=True)


# ==================== 提示帧标注（单帧） ====================


@app.route('/api/prompt_frame', methods=['POST'])
def prompt_frame():
    """对指定帧执行单帧提示标注（使用bbox或文本）"""
    data = request.json
    prompt_idx = data.get('prompt_frame_idx', 0)
    bboxes = data.get('bboxes', [])
    items_text = data.get('items', '')
    find_list = [s.strip() for s in items_text.split(',') if s.strip()]

    try:
        import torch
        mf = TEMP_DATA_MID_DIR / "frames"
        ml_dir = TEMP_DATA_MID_DIR / "labels"
        ma = TEMP_DATA_MID_DIR / "annotations.json"

        sf_path = mf / f"frame_{prompt_idx:06d}.jpg"
        if not sf_path.exists():
            return jsonify({'error': f'提示帧 {prompt_idx} 不存在'}), 400

        sf = cv2.imread(str(sf_path))
        h, w = sf.shape[:2]

        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        half = device == 'cuda'
        overrides = {
            'conf': 0.25, 'task': "segment", 'mode': "predict",
            'model': SAM_MODEL_PATH, 'device': device, 'half': half, 'save': False, 'verbose': False
        }
        if device == 'cuda':
            overrides['batch'] = 1
            overrides['stream_buffer'] = False
        elif device == 'mps':
            overrides['half'] = True
            overrides['amp'] = True
            overrides['stream_buffer'] = True

        use_semantic = bool(find_list)
        if use_semantic:
            try:
                _patch_sam3_video_semantic()
                from ultralytics.models.sam import SAM3VideoSemanticPredictor
                predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            except Exception:
                from ultralytics.models.sam import SAM3VideoPredictor
                predictor = SAM3VideoPredictor(overrides=overrides)
                use_semantic = False
        else:
            from ultralytics.models.sam import SAM3VideoPredictor
            predictor = SAM3VideoPredictor(overrides=overrides)

        # 查找可用track_id
        if ma.exists():
            with open(ma) as f:
                coco = json.load(f)
        else:
            coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}

        first_id = _first_available_track_id(coco, 1000000)

        from annotate_video import merge_masks_in_frame
        pred_args = {'source': str(sf_path), 'stream': True}
        if bboxes:
            pred_args['bboxes'] = bboxes
            pred_args['labels'] = [1] * len(bboxes)
        if find_list:
            pred_args['text'] = find_list

        results = predictor(**pred_args)
        from annotate_video import merge_masks_in_frame

        all_anns = []
        for r in results:
            masks_attr = getattr(r, 'masks', None)
            if masks_attr is not None and masks_attr.data is not None:
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
                        conf_val = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                        ann = {
                            'id': len(coco.get('annotations', [])) + idx + 1,
                            'track_id': track_id,
                            'image_id': prompt_idx,
                            'category_id': track_id,
                            'bbox': b,
                            'area': float(cv2.contourArea(m.astype(np.uint8))),
                            'segmentation': [poly],
                            'iscrowd': 0,
                            'confidence': conf_val
                        }
                        all_anns.append(ann)

        # 写入coco
        for ann in all_anns:
            coco['annotations'].append(ann)
        with open(ma, 'w') as f:
            json.dump(coco, f)

        # 写入帧标签
        if all_anns:
            lf = ml_dir / f"frame_{prompt_idx:06d}.json"
            existing = []
            if lf.exists():
                with open(lf) as f:
                    existing = json.load(f)
            existing.extend(all_anns)
            with open(lf, 'w') as f:
                json.dump(existing, f)

        return jsonify({'msg': f'提示帧完成，新增{len(all_anns)}条', 'count': len(all_anns), 'FIRST_ID': first_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==================== 导出 ====================


@app.route('/api/export', methods=['POST'])
def export_route():
    """导出标注到temp_data_post，带类别映射和track_id过滤(track_id>999998)"""
    data = request.json or {}
    cat_maps = data.get('category_mappings', ['Detect'] * 8)

    if not TEMP_DATA_MID_DIR.exists():
        return jsonify({'error': 'temp_data_mid 不存在'}), 400

    af = TEMP_DATA_MID_DIR / "annotations.json"
    with open(af) as f:
        coco = json.load(f)

    total = len(coco.get('images', []))
    out = TEMP_DATA_POST_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    old = out / "labels"
    ofram = out / "frames"
    old.mkdir(parents=True)
    ofram.mkdir(parents=True)

    ld = TEMP_DATA_MID_DIR / "labels"
    fd = TEMP_DATA_MID_DIR / "frames"

    for i in range(total):
        if (fd / f"frame_{i:06d}.jpg").exists():
            shutil.copy2(fd / f"frame_{i:06d}.jpg", ofram / f"frame_{i:06d}.jpg")
        lp = ld / f"frame_{i:06d}.json"
        olp = old / f"frame_{i:06d}.json"
        if lp.exists():
            with open(lp) as f:
                anns = json.load(f)
            fa = []
            for ann in anns:
                tid = ann.get('track_id', 0)
                if tid >= 999999:
                    idx = tid - 1000000
                    if 0 <= idx <= 7:
                        ann['category'] = cat_maps[idx] if idx < len(cat_maps) else 'Detect'
                        ann['category_id'] = idx
                    else:
                        ann['category'] = 'Detect'
                        ann['category_id'] = tid - 1000000
                    fa.append(ann)
            with open(olp, 'w') as f:
                json.dump(fa, f)

    # 写入导出后的coco
    export_annotations = []
    for ann in coco.get('annotations', []):
        tid = ann.get('track_id', 0)
        if tid >= 999999:
            idx = tid - 1000000
            if 0 <= idx <= 7:
                ann['category'] = cat_maps[idx] if idx < len(cat_maps) else 'Detect'
                ann['category_id'] = idx
            else:
                ann['category'] = 'Detect'
                ann['category_id'] = tid - 1000000
            export_annotations.append(ann)
    coco['annotations'] = export_annotations
    if 'categories' not in coco:
        coco['categories'] = []
    with open(out / 'annotations.json', 'w') as f:
        json.dump(coco, f)

    return jsonify({'msg': f'导出完成，{total}帧'})


# ==================== Trace ID 管理 ====================


@app.route('/api/save_mappings', methods=['POST'])
def save_mappings():
    data = request.json
    mappings = data.get('mappings', [])
    _apply_mappings(mappings)
    state['mappings'] = mappings
    return jsonify({'msg': '已保存'})


@app.route('/api/load_mappings')
def load_mappings():
    mf = TEMP_DATA_MID_DIR / "trace_id_changes.json"
    if mf.exists():
        with open(mf) as f:
            return jsonify({'mappings': json.load(f)})
    return jsonify({'mappings': []})


@app.route('/api/delete_track_id', methods=['POST'])
def delete_track_id():
    """删除指定track_id的标注：将标签中的track_id改为9999"""
    data = request.json
    track_id = int(data.get('track_id', 0))
    if track_id <= 0:
        return jsonify({'error': '无效track_id'}), 400

    ld = TEMP_DATA_MID_DIR / "labels"
    count = 0
    for lf in sorted(ld.glob("frame_*.json")):
        with open(lf) as f:
            fa = json.load(f)
        changed = False
        for a in fa:
            if a.get('track_id') == track_id:
                a['track_id'] = 9999
                changed = True
        if changed:
            with open(lf, 'w') as f:
                json.dump(fa, f)
            count += 1

    # 也更新annotations.json
    af = TEMP_DATA_MID_DIR / "annotations.json"
    if af.exists():
        with open(af) as f:
            coco = json.load(f)
        changed = False
        for a in coco.get('annotations', []):
            if a.get('track_id') == track_id:
                a['track_id'] = 9999
                changed = True
        if changed:
            with open(af, 'w') as f:
                json.dump(coco, f)

    return jsonify({'msg': f'已删除 track_id={track_id}，影响 {count} 帧'})


# ==================== 保存视频 + OBS上传 ====================


@app.route('/api/save_video', methods=['POST'])
def save_video():
    """从temp_data_post生成标注视频并上传OBS"""
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

                # 使用调色板颜色：所选颜色用于主标注，其他用palette
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

                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                cv2.putText(overlay, f"{cat} {conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        out.write(frame)

    out.release()

    # 上传到OBS
    upload_result = ""
    try:
        result = subprocess.run(
            ['curl', '--upload-file', str(output_path), 'http://obs.dimond.top/dst.mp4'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            upload_result = "上传成功"
        else:
            upload_result = f"上传失败: {result.stderr[:200]}"
    except Exception as e:
        upload_result = f"上传失败: {str(e)}"

    return jsonify({
        'msg': f'视频已保存: {output_path}',
        'path': str(output_path),
        'upload': upload_result
    })


# ==================== 围栏 (Fence) 管理 ====================


@app.route('/api/fence/state', methods=['GET'])
def get_fence_state():
    return jsonify({
        'active': state.get('fence_mode', False),
        'fences': state.get('fences', [])
    })


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


# ==================== 获取帧上的标注(用于点击检测) ====================


@app.route('/api/frame_annotations/<int:idx>')
def get_frame_annotations(idx):
    ld = TEMP_DATA_MID_DIR / "labels"
    lp = ld / f"frame_{idx:06d}.json"
    if lp.exists():
        with open(lp) as f:
            anns = json.load(f)
        return jsonify(anns)
    return jsonify([])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)
