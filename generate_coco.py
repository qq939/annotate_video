#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端生成 coco 数据集（正式产物）"""
import sys
from pathlib import Path
from cocomaker import convert_to_coco_dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

root = Path(__file__).parent
test_dir = root / "cocomaker" / "fromModel" / "test"

videos = sorted(test_dir.glob("*.mp4"))
model_file = test_dir / "sop2model.rar"

# 输出到 fromModel/test 同级，命名 fromModel_coco
out_dir = test_dir.parent / "fromModel_coco"

result = convert_to_coco_dataset(
    video_paths=[str(v) for v in videos],
    model_archive=str(model_file),
    output_dir=str(out_dir),
    target_w=2012,
    target_h=1518,
    frame_skip=10,      # 每10帧取1帧（避免帧数过大）
    conf_threshold=0.25,
    iou_threshold=0.45,
)

print("\n[完成] 输出目录:", result)
print("帧数:", len(list((result / "frames").glob("frame_*.jpg"))))
print("标注数:", len(list((result / "labels").glob("frame_*.json"))))
