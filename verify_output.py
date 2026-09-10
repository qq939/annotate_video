# -*- coding: utf-8 -*-
"""验证转换输出"""
import json
from pathlib import Path
from PIL import Image

src_dir = Path(r"C:\Users\qq939\Downloads\image")
dst_dir = Path(r"C:\Users\qq939\Downloads\annotate_video\verify_output")
dst_dir.mkdir(parents=True, exist_ok=True)
labels_dir = dst_dir / "labels"
labels_dir.mkdir(exist_ok=True)

# 目标分辨率
target_w, target_h = 2012, 1568

# 获取文件列表
json_files = sorted([f for f in src_dir.glob("*.json") if f.name != "annotations.json"])
print(f"Found {len(json_files)} JSON files")

# 从图片获取真实尺寸
img_files = sorted([f for f in src_dir.glob("*.jpg")])
with Image.open(img_files[0]) as img:
    src_w, src_h = img.size
print(f"Real source size from image: {src_w}x{src_h}")

ratio_x = target_w / src_w
ratio_y = target_h / src_h
print(f"Ratio: x={ratio_x}, y={ratio_y}")

# 处理第一个文件
jf = json_files[0]
print(f"\nProcessing: {jf.name}")
with open(jf, encoding='utf-8') as f:
    d = json.load(f)

# 文件名解析帧号
name = jf.stem
digits = ''.join(c for c in name if c.isdigit())
frame_idx = int(digits) if digits else 0
print(f"Frame idx: {frame_idx}")

# 处理 shapes
anns = []
cat_map = {}
cats = []
def get_cat(label):
    if label not in cat_map:
        cid = len(cats) + 1
        cat_map[label] = cid
        cats.append({"id": cid, "name": label, "supercategory": ""})
    return cat_map[label]

for shape in d.get("shapes", []):
    label = shape.get("label", "unknown")
    points = shape.get("points", [])
    if len(points) >= 2:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x, y = min(xs), min(ys)
        w, h = max(xs) - x, max(ys) - y

        x_scaled = x * ratio_x
        y_scaled = y * ratio_y
        w_scaled = w * ratio_x
        h_scaled = h * ratio_y

        seg = []
        for p in points:
            seg.append(p[0] * ratio_x)
            seg.append(p[1] * ratio_y)

        anns.append({
            "id": len(anns) + 1,
            "category_id": get_cat(label),
            "track_id": 0,
            "trace_id_list": [0],
            "bbox": [x_scaled, y_scaled, w_scaled, h_scaled],
            "area": w_scaled * h_scaled,
            "segmentation": [seg],
            "iscrowd": 0
        })

# 写入
output_file = labels_dir / f"frame_{frame_idx:06d}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(anns, f, ensure_ascii=False)

print(f"\nCategories: {cats}")
print(f"\nAnn 1 bbox: {anns[0]['bbox']}")
print(f"Ann 1 category_id: {anns[0]['category_id']}")
