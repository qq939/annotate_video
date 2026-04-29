#!/usr/bin/env python3
"""
前向标注 Demo
功能：从指定帧（提示帧）开始，用SAM3VideoPredictor向前跟踪标注所有帧
"""

import cv2
import json
import shutil
from pathlib import Path


VIDEO_PATH = "1src/demo_video.mp4"       # 视频文件路径
PROMPT_FRAME_IDX = 10                     # 提示帧索引（第11帧，从0开始）
PROMPT_BBOXES = [                          # 提示帧上的bbox列表 [x1, y1, x2, y2]
    [100, 100, 300, 400],
    [400, 200, 600, 500],
]
OUTPUT_DIR = Path("demo_output")
FIRST_ID = 50000                           # track_id起始值
IOU_THRESHOLD = 0.5                        # 跟踪IoU阈值
MERGE_IOU_THRESHOLD = 0.5                  # 分割合并IoU阈值


def calculate_mask_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def calculate_bbox_iou(b1, b2):
    x1_1, y1_1, w1, h1 = b1
    x1_2, y1_2, w2, h2 = b2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


class SimpleTrackManager:
    """简单的跟踪管理器"""
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tracked = {}  # track_id -> {mask, bbox, last_seen}
        self.next_id = FIRST_ID

    def update(self, masks, bboxes, frame_idx):
        if not masks:
            return []

        track_ids = []

        if not self.tracked:
            for i in range(len(masks)):
                tid = self.next_id
                self.next_id += 1
                self.tracked[tid] = {'mask': masks[i], 'bbox': bboxes[i], 'last_seen': frame_idx}
                track_ids.append(tid)
            return track_ids

        for i, (mask, bbox) in enumerate(zip(masks, bboxes)):
            best_iou, best_tid = 0, None
            for tid, obj in self.tracked.items():
                iou = calculate_mask_iou(mask, obj['mask']) if obj['mask'] is not None else calculate_bbox_iou(bbox, obj['bbox'])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou >= self.iou_threshold:
                track_ids.append(best_tid)
                self.tracked[best_tid] = {'mask': mask, 'bbox': bbox, 'last_seen': frame_idx}
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracked[tid] = {'mask': mask, 'bbox': bbox, 'last_seen': frame_idx}
                track_ids.append(tid)

        self.tracked = {k: v for k, v in self.tracked.items() if v['last_seen'] >= frame_idx - 30}
        return track_ids


def main():
    # ============================================================
    # 第1步：从视频切帧到临时目录
    # ============================================================
    print(f"\n{'='*60}")
    print("第1步：从视频切帧")
    print(f"{'='*60}")

    temp_clip_dir = OUTPUT_DIR / "clip_frames"
    if temp_clip_dir.exists():
        shutil.rmtree(temp_clip_dir)
    temp_clip_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {total_frames}帧, {width}x{height}, {fps:.1f}fps")

    # 从PROMPT_FRAME_IDX开始切帧到视频结尾
    clip_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, PROMPT_FRAME_IDX)
    frame_idx_in_clip = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = temp_clip_dir / f"frame_{frame_idx_in_clip:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        clip_frames.append(frame_path)
        frame_idx_in_clip += 1
        if frame_idx_in_clip % 50 == 0:
            print(f"  已切帧: {frame_idx_in_clip} 帧")
    cap.release()
    print(f"✓ 切帧完成: 共 {len(clip_frames)} 帧 (帧{PROMPT_FRAME_IDX} → 帧{total_frames-1})")

    # ============================================================
    # 第2步：生成临时视频片段
    # ============================================================
    print(f"\n{'='*60}")
    print("第2步：生成临时视频片段")
    print(f"{'='*60}")

    clip_path = str(temp_clip_dir / "clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
    for fp in clip_frames:
        frame = cv2.imread(str(fp))
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"✓ 临时视频片段生成: {clip_path}")

    # ============================================================
    # 第3步：SAM3VideoPredictor 前向跟踪
    # ============================================================
    print(f"\n{'='*60}")
    print("第3步：SAM3VideoPredictor 前向跟踪")
    print(f"{'='*60}")

    from ultralytics.models.sam import SAM3VideoPredictor

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model="sam3.pt",          # 模型路径
        device='mps',              # 或 'cuda' / 'cpu'
        half=False,
        save=False,
        verbose=False
    )
    predictor = SAM3VideoPredictor(overrides=overrides)

    # 关键：传 bboxes 参数给 SAM3VideoPredictor
    results = predictor(
        source=clip_path,
        stream=True,
        bboxes=PROMPT_BBOXES,
        labels=[1] * len(PROMPT_BBOXES)
    )

    manager = SimpleTrackManager(iou_threshold=IOU_THRESHOLD)
    manager.next_id = FIRST_ID
    ann_id = FIRST_ID

    all_annotations = []
    frame_count = 0

    print(f"\n开始处理 {len(clip_frames)} 帧...")
    for result in results:
        orig_img = result.orig_img if hasattr(result, 'orig_img') and result.orig_img is not None else None
        if orig_img is None:
            cap_t = cv2.VideoCapture(clip_path)
            cap_t.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret_t, orig_img = cap_t.read()
            cap_t.release()
            if not ret_t:
                orig_img = None

        if orig_img is None:
            orig_img = cv2.imread(str(clip_frames[frame_count]))

        orig_h, orig_w = orig_img.shape[:2]
        cv2.imwrite(str(temp_clip_dir / f"annotated_{frame_count:06d}.jpg"), orig_img)

        frame_anns = []
        print(f"  帧 {frame_count}: ", end="")

        if hasattr(result, 'masks') and result.masks is not None:
            masks_tensor = result.masks.data
            if masks_tensor is not None and len(masks_tensor) > 0:
                print(f"检测到 {len(masks_tensor)} 个mask", end="")

                cur_masks = []
                cur_bboxes = []
                for mask in masks_tensor:
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'numpy') else mask
                    mask_bin = (mask_np > 0.5).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if len(cnt) >= 3:
                            poly = cnt.squeeze().flatten().tolist()
                            xs, ys = poly[0::2], poly[1::2]
                            x1, x2 = min(xs), max(xs)
                            y1, y2 = min(ys), max(ys)
                            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            area = cv2.contourArea(cnt)
                            if area > 0:
                                cur_masks.append(mask_bin)
                                cur_bboxes.append(bbox)

                print(f" → 有效polygon: {len(cur_masks)}", end="")
                if cur_masks:
                    track_ids = manager.update(cur_masks, cur_bboxes, frame_count)
                    print(f" → track_ids: {track_ids}", end="")

                    for idx, (mask, bbox) in enumerate(zip(cur_masks, cur_bboxes)):
                        mask_bin2 = (mask > 0.5).astype(np.uint8)
                        contours2, _ = cv2.findContours(mask_bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt2 in contours2:
                            if len(cnt2) >= 3:
                                poly = cnt2.squeeze().flatten().tolist()
                                area = cv2.contourArea(cnt2)
                                tid = track_ids[idx] if idx < len(track_ids) else ann_id
                                ann = {
                                    'id': ann_id,
                                    'track_id': tid,
                                    'image_id': frame_count + PROMPT_FRAME_IDX,
                                    'category_id': tid,
                                    'bbox': bbox,
                                    'area': float(area),
                                    'segmentation': [poly],
                                    'iscrowd': 0,
                                    'confidence': 1.0,
                                    'category': 'Detect'
                                }
                                all_annotations.append(ann)
                                frame_anns.append(ann)
                                ann_id += 1

        print(f" | 帧标注数: {len(frame_anns)}")
        frame_count += 1

    print(f"\n✓ 前向跟踪完成: {frame_count} 帧, {len(all_annotations)} 个标注")

    # ============================================================
    # 第4步：保存结果到 labels/ 和 annotations.json
    # ============================================================
    print(f"\n{'='*60}")
    print("第4步：保存结果")
    print(f"{'='*60}")

    output_labels_dir = OUTPUT_DIR / "labels"
    if output_labels_dir.exists():
        shutil.rmtree(output_labels_dir)
    output_labels_dir.mkdir(parents=True)

    output_frames_dir = OUTPUT_DIR / "frames"
    output_frames_dir.mkdir(parents=True)

    for ann in all_annotations:
        img_id = ann['image_id']
        label_file = output_labels_dir / f"frame_{img_id:06d}.json"
        frame_file = output_frames_dir / f"frame_{img_id:06d}.jpg"
        existing = []
        if label_file.exists():
            with open(label_file) as f:
                existing = json.load(f)
        merged = existing + [ann]
        with open(label_file, 'w') as f:
            json.dump(merged, f)

        src_frame = Path("temp_data/frames") / f"frame_{img_id:06d}.jpg"
        if src_frame.exists():
            shutil.copy2(src_frame, frame_file)

    coco = {
        'info': {'video_path': VIDEO_PATH, 'fps': fps, 'width': width, 'height': height, 'prompt_frame': PROMPT_FRAME_IDX},
        'images': [{'id': i + PROMPT_FRAME_IDX, 'frame_idx': i + PROMPT_FRAME_IDX} for i in range(frame_count)],
        'annotations': all_annotations,
        'categories': [{'id': i, 'name': f'track_{FIRST_ID + i}'} for i in range(manager.next_id - FIRST_ID)]
    }
    with open(OUTPUT_DIR / "annotations.json", 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"✓ 保存完成: {OUTPUT_DIR}")
    print(f"  - labels/: {len(list(output_labels_dir.glob('*.json')))} 个label文件")
    print(f"  - annotations.json: {len(all_annotations)} 条标注")
    print(f"  - track_id范围: {FIRST_ID} ~ {manager.next_id - 1}")


if __name__ == "__main__":
    import numpy as np
    main()
