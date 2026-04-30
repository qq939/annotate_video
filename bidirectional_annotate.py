# global参数
SAM_MODEL_PATH = "sam3.pt"  # 第4行：SAM3模型路径

import cv2
import numpy as np
import json
import shutil
import torch
from pathlib import Path


def do_bidirectional_annotate(data_path, prompt_frame_idx, boxes, iou_threshold=0.5, merge_iou_threshold=0.5, find_list=None):
    data_path = Path(data_path)
    frames_dir = data_path / "frames"
    labels_dir = data_path / "labels"

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    total = len(frame_files)
    if total == 0:
        raise RuntimeError("没有帧数据")

    sample = cv2.imread(str(frame_files[0]))
    height, width = sample.shape[:2]

    if not boxes and not find_list:
        raise RuntimeError("请至少提供一个bbox或文本提示词")

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

    predictor_name = "SAM3VideoSemanticPredictor" if find_list else "SAM3VideoPredictor"
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

    from annotate_video import merge_masks_in_frame, TrackManager

    FIRST_ID = 50000

    def process_clip(start_frame, end_frame, forward=True, prompt_bboxes=None):
        direction = "向前" if forward else "向后"
        print(f"[{direction}] 处理帧 {start_frame} → {end_frame}")

        temp_dir = Path("temp_inject") / ("forward" if forward else "backward")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        clip_path = str(temp_dir / "clip.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, 30, (width, height))
        if forward:
            for i in range(start_frame, end_frame):
                frame = cv2.imread(str(frames_dir / f"frame_{i:06d}.jpg"))
                if frame is not None:
                    out.write(frame)
        else:
            for i in range(end_frame - 1, start_frame - 1, -1):
                frame = cv2.imread(str(frames_dir / f"frame_{i:06d}.jpg"))
                if frame is not None:
                    out.write(frame)
        out.release()

        results = predictor(source=clip_path, stream=True, bboxes=prompt_bboxes, labels=[1]*len(prompt_bboxes))
        manager = TrackManager(iou_threshold=iou_threshold)
        manager.next_track_id = FIRST_ID
        ann_id = FIRST_ID
        frame_idx = 0
        result_anns = []

        for r in results:
            if frame_idx >= (end_frame - start_frame):
                break
            orig_img = r.orig_img if hasattr(r, 'orig_img') and r.orig_img is not None else None
            if orig_img is None:
                cap_t = cv2.VideoCapture(clip_path)
                cap_t.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_t, orig_img = cap_t.read()
                cap_t.release()
                if not ret_t:
                    orig_img = np.zeros((height, width, 3), dtype=np.uint8)
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

                    cur_masks = []
                    cur_bboxes = []
                    for mask in masks_tensor:
                        mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else np.array(mask)
                        mask_binary = (mask_np > 0.5).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if len(cnt) >= 3:
                                poly = cnt.squeeze().flatten().tolist()
                                xs = poly[0::2]
                                ys = poly[1::2]
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
                                    conf = float(confs[idx]) if confs is not None and idx < len(confs) else float(m.max())
                                    if forward:
                                        img_id = frame_idx + start_frame
                                    else:
                                        img_id = end_frame - 1 - frame_idx
                                    ann = {
                                        'id': ann_id, 'track_id': tid, 'image_id': img_id,
                                        'category_id': tid, 'bbox': bb, 'area': float(area2),
                                        'segmentation': [poly2], 'iscrowd': 0, 'confidence': conf
                                    }
                                    result_anns.append(ann)
                                    frame_anns.append(ann)
                                    ann_id += 1

            if forward:
                orig_frame_idx = frame_idx + start_frame
            else:
                orig_frame_idx = end_frame - 1 - frame_idx

            if 0 <= orig_frame_idx < total and frame_anns:
                label_file = labels_dir / f"frame_{orig_frame_idx:06d}.json"
                existing = []
                if label_file.exists():
                    with open(label_file) as f:
                        existing = json.load(f)
                with open(label_file, 'w') as f:
                    json.dump(existing + frame_anns, f)

            frame_idx += 1

        shutil.rmtree(temp_dir, ignore_errors=True)
        return result_anns

    print(f"提示帧帧: {prompt_frame_idx}, 总帧数: {total}, bboxes: {boxes}")
    forward_anns = process_clip(prompt_frame_idx + 1, total, forward=True, prompt_bboxes=boxes) if prompt_frame_idx + 1 < total else []
    backward_anns = process_clip(0, prompt_frame_idx, forward=False, prompt_bboxes=boxes) if prompt_frame_idx > 0 else []

    all_new_anns = backward_anns + forward_anns
    print(f"双向标注完成: 向后={len(backward_anns)}, 向前={len(forward_anns)}, 合计={len(all_new_anns)}")

    ann_file = data_path / "annotations.json"
    if ann_file.exists():
        with open(ann_file) as f:
            coco = json.load(f)
    else:
        coco = {'info': {}, 'images': [], 'annotations': [], 'categories': []}

    max_ann_id = max([ann['id'] for ann in coco.get('annotations', [])], default=FIRST_ID - 1)
    max_track_id = max([ann['track_id'] for ann in coco.get('annotations', [])], default=FIRST_ID - 1)

    for ann in all_new_anns:
        new_ann = dict(ann)
        max_ann_id += 1
        new_ann['id'] = max_ann_id
        new_ann['track_id'] = max_track_id + 1
        new_ann['category_id'] = new_ann['track_id']
        max_track_id = new_ann['track_id']
        coco['annotations'].append(new_ann)

    with open(ann_file, 'w') as f:
        json.dump(coco, f)

    shutil.rmtree(Path("temp_inject"), ignore_errors=True)
    print(f"已写入 {len(all_new_anns)} 条新标注到 {ann_file}")
