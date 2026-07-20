#!/usr/bin/env python3
import json
from pathlib import Path

temp_mid = Path("temp_data_mid")
labels_dir = temp_mid / "labels"
annotations_file = temp_mid / "annotations.json"

# 读取现有coco
if annotations_file.exists():
    with open(annotations_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
else:
    coco = {"info": {}, "images": [], "annotations": [], "categories": []}

print(f"现有: images={len(coco['images'])}, annotations={len(coco['annotations'])}")

# 收集所有label文件，并补全id
all_annotations = []
ann_id = 1
for label_file in sorted(labels_dir.glob("frame_*.json")):
    with open(label_file, "r", encoding="utf-8") as f:
        anns = json.load(f)
    for ann in anns:
        if "id" not in ann:
            ann["id"] = ann_id
            ann_id += 1
        all_annotations.append(ann)

print(f"从label文件收集到: {len(all_annotations)} 条标注")

# 更新coco
coco["annotations"] = all_annotations

# 保存
with open(annotations_file, "w", encoding="utf-8") as f:
    json.dump(coco, f, ensure_ascii=False)

print("已保存到 annotations.json")
