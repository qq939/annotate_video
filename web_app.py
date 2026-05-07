#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web标注工具 - 与app.py功能同步"""

import os
import sys
import shutil
import json
import cv2
import numpy as np
from pathlib import Path
import base64

from flask import Flask, render_template, request, jsonify, Response

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

WARM_COLORS = [
    (180, 130, 255), (200, 100, 220), (255, 50, 200), (255, 0, 180),
    (220, 0, 150), (180, 0, 120), (139, 0, 100), (100, 0, 80)
]
COLD_COLORS = [
    (100, 150, 255), (100, 200, 255), (150, 200, 255), (100, 255, 200),
    (150, 100, 255), (100, 200, 200), (150, 150, 255), (100, 180, 255)
]


def get_color_for_track_id(track_id):
    if track_id >= 1000000:
        return WARM_COLORS[(track_id - 1000000) % len(WARM_COLORS)]
    return COLD_COLORS[track_id % len(COLD_COLORS)]


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
    return jsonify({'filename': fname, 'width': w, 'height': h})


@app.route('/api/run_annotate', methods=['POST'])
def run_annotate():
    data = request.json
    video_name = data.get('video_name', '')
    merge_iou = float(data.get('merge_iou', 0.5))
    iou = float(data.get('iou', 0.5))
    items_text = data.get('items', '')
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
            cap.release()

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

            if find_list:
                try:
                    from app import _patch_sam3_video_semantic
                    _patch_sam3_video_semantic()
                    from ultralytics.models.sam import SAM3VideoSemanticPredictor
                    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
                except Exception:
                    from ultralytics.models.sam import SAM3VideoPredictor
                    predictor = SAM3VideoPredictor(overrides=overrides)
            else:
                from ultralytics.models.sam import SAM3VideoPredictor
                predictor = SAM3VideoPredictor(overrides=overrides)

            predictor_args = {'source': src_video, 'stream': True}
            if find_list:
                predictor_args['text'] = find_list

            from annotate_video import merge_masks_in_frame, TrackManager
            track_manager = TrackManager(iou_threshold=iou)
            ann_id = [0]
            coco_data = {
                'info': {'fps': fps, 'width': width, 'height': height, 'fourcc': fourcc_str, 'FIND': find_list},
                'images': [], 'annotations': [], 'categories': []
            }

            results = predictor(**predictor_args)
            cap_cnt = cv2.VideoCapture(src_video)
            total = int(cap_cnt.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_cnt.release()

            frame_count = 0
            for r in results:
                orig_img = getattr(r, 'orig_img', None)
                if orig_img is None:
                    ct = cv2.VideoCapture(src_video)
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
                                    tid = tids[idx] if idx < len(tids) else ann_id[0]
                                    conf = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                                    ann = {
                                        'id': ann_id[0], 'track_id': tid, 'image_id': frame_count,
                                        'category_id': tid, 'bbox': b, 'area': float(area),
                                        'segmentation': [poly], 'iscrowd': 0, 'confidence': conf
                                    }
                                    coco_data['annotations'].append(ann)
                                    frame_anns.append(ann)
                                    ann_id[0] += 1

                with open(labels_dir / f"frame_{frame_count:06d}.json", 'w') as f_out:
                    json.dump(frame_anns, f_out)

                frame_count += 1
                pct = int(frame_count / max(total, 1) * 100)
                yield 'data: ' + json.dumps({
                    'type': 'progress', 'frame': frame_count, 'total': total, 'percent': pct
                }) + '\n\n'

            with open(temp_data / 'annotations.json', 'w') as f_out:
                json.dump(coco_data, f_out)
            yield 'data: ' + json.dumps({'type': 'done', 'frames': frame_count}) + '\n\n'
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield 'data: ' + json.dumps({'type': 'error', 'msg': str(e)}) + '\n\n'
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/show_viewer', methods=['POST'])
def show_viewer():
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

    frames = sorted((TEMP_DATA_MID_DIR / 'frames').glob('*.jpg'))
    return jsonify({'frames': len(frames), 'total': len(frames)})


def _apply_mappings(mappings):
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
            except:
                pass

    ld = TEMP_DATA_MID_DIR / "labels"
    af = TEMP_DATA_MID_DIR / "annotations.json"
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

    _, buf = cv2.imencode('.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify({'image': base64.b64encode(buf).decode(), 'frame': idx, 'annotations': frame_anns})


@app.route('/api/get_video_info')
def get_video_info():
    af = TEMP_DATA_MID_DIR / "annotations.json"
    if not af.exists():
        return jsonify({'total': 0})
    frames = sorted((TEMP_DATA_MID_DIR / 'frames').glob('*.jpg'))
    return jsonify({'total': len(frames)})


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
        ml = TEMP_DATA_MID_DIR / "labels"
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

        from ultralytics.models.sam import SAM3VideoPredictor
        predictor = SAM3VideoPredictor(overrides=overrides)

        sf = cv2.imread(str(mf / "frame_000000.jpg"))
        h, w = sf.shape[:2]

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
        all_anns = []

        if forward_en and prompt_idx + 1 < total:
            result = _process_clip(mf, predictor, mgr, merge_iou, prompt_idx + 1, total, True, bboxes, h, w)
            all_anns.extend(result)

        if backward_en and prompt_idx > 0:
            result = _process_clip(mf, predictor, mgr, merge_iou, 0, prompt_idx, False, bboxes, h, w)
            all_anns.extend(result)

        if ma.exists():
            with open(ma) as f:
                coco = json.load(f)
        else:
            coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}

        max_aid = max([a['id'] for a in coco.get('annotations', [])], default=0)
        max_tid = max([a['track_id'] for a in coco.get('annotations', [])], default=first_id - 1)
        for ann in all_anns:
            max_aid += 1
            max_tid += 1
            ann['id'] = max_aid
            ann['track_id'] = max_tid
            ann['category_id'] = max_tid
            coco['annotations'].append(ann)

        with open(ma, 'w') as f:
            json.dump(coco, f)

        return jsonify({'msg': f'双向标注完成，新增{len(all_anns)}条', 'FIRST_ID': first_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _process_clip(mf, predictor, mgr, merge_iou, start, end, fwd, bboxes, h, w):
    tp = Path("temp_clip")
    tp.mkdir(exist_ok=True)
    fc = end - start
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

    results = predictor(source=cp, stream=True, bboxes=bboxes, labels=[1] * len(bboxes))
    all_anns = []
    aid = mgr.next_track_id

    for r in results:
        orig = getattr(r, 'orig_img', None) or np.zeros((h, w, 3), dtype=np.uint8)
        if len(orig.shape) == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        elif orig.shape[2] == 4:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGRA2BGR)

        frame_anns = []
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
                            conf = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                            ann = {
                                'id': aid, 'track_id': tid, 'image_id': 0, 'bbox': b, 'area': float(area),
                                'segmentation': [poly], 'iscrowd': 0, 'confidence': conf
                            }
                            all_anns.append(ann)
                            frame_anns.append(ann)
                            aid += 1

        shutil.rmtree(tp, ignore_errors=True)

    return all_anns


@app.route('/api/export', methods=['POST'])
def export():
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

    with open(out / 'annotations.json', 'w') as f:
        json.dump(coco, f)

    return jsonify({'msg': f'导出完成，{total}帧'})


@app.route('/api/save_mappings', methods=['POST'])
def save_mappings():
    data = request.json
    mappings = data.get('mappings', [])
    _apply_mappings(mappings)
    return jsonify({'msg': '已保存'})


@app.route('/api/load_mappings')
def load_mappings():
    mf = TEMP_DATA_MID_DIR / "trace_id_changes.json"
    if mf.exists():
        with open(mf) as f:
            return jsonify({'mappings': json.load(f)})
    return jsonify({'mappings': []})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, debug=False)
