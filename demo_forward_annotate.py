#!/usr/bin/env python3
"""最精简版：前向标注 demo"""
import cv2, json, shutil
from pathlib import Path
import numpy as np

VIDEO = "1src/demo.mp4"
PROMPT_FRAME = 10
BBOXES = [[100, 100, 300, 400]]
FIRST_ID = 50000

def main():
    clip = Path("demo_clip")
    if clip.exists(): shutil.rmtree(clip)
    clip.mkdir(parents=True)

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 向前：提示帧到结尾，正序
    for i in range(PROMPT_FRAME, total):
        ret, f = cap.read()
        if ret: cv2.imwrite(str(clip / f"f_{i - PROMPT_FRAME:03d}.jpg"), f)

    # 向后：提示帧-1到0，倒序（关键！）
    for rev_i in range(PROMPT_FRAME - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, rev_i)
        ret, f = cap.read()
        if ret: cv2.imwrite(str(clip / f"b_{PROMPT_FRAME - 1 - rev_i:03d}.jpg"), f)
    cap.release()

    # 生成视频片段
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for name, n in [("fwd", total - PROMPT_FRAME), ("bwd", PROMPT_FRAME)]:
        out = cv2.VideoWriter(str(clip / f"{name}.mp4"), fourcc, fps, (w, h))
        for i in range(n):
            img = cv2.imread(str(clip / f"{'f' if name=='fwd' else 'b'}_{i:03d}.jpg"))
            if img is not None: out.write(img)
        out.release()

    from ultralytics.models.sam import SAM3VideoPredictor
    pred = SAM3VideoPredictor(overrides=dict(
        conf=0.25, task="segment", mode="predict",
        model="sam3.pt", device='mps', half=False, save=False, verbose=False))

    out_dir = Path("demo_output/labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(name, n, orig_start):
        results = pred(source=str(clip / f"{name}.mp4"), stream=True,
                       bboxes=BBOXES, labels=[1])
        aid = FIRST_ID
        for idx, r in enumerate(results):
            oi = orig_start + idx
            anns = []
            if hasattr(r, 'masks') and r.masks:
                for m in r.masks.data:
                    mn = m.cpu().numpy() if hasattr(m, 'numpy') else m
                    mb = (mn > 0.5).astype(np.uint8)
                    cs, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cs:
                        if len(c) >= 3:
                            poly = c.squeeze().flatten().tolist()
                            anns.append({'id': aid, 'track_id': aid, 'image_id': oi,
                                        'bbox': [0,0,10,10], 'area': float(cv2.contourArea(c)),
                                        'segmentation': [poly], 'iscrowd': 0, 'confidence': 1.0, 'category': 'Detect'})
                            aid += 1
            lf = out_dir / f"frame_{oi:06d}.json"
            ex = json.load(open(lf)) if lf.exists() else []
            json.dump(ex + anns, open(lf, 'w'))
            print(f"  {name} {idx}→frame_{oi:06d}: {len(anns)} anns")

    print("向前...")
    run("fwd", total - PROMPT_FRAME, PROMPT_FRAME)
    print("向后...")
    run("bwd", PROMPT_FRAME, 0)
    print("完成!")

if __name__ == "__main__":
    main()
