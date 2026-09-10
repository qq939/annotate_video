# -*- coding: utf-8 -*-
"""Debug: 验证源文件和目标文件的帧号对应关系"""
import json
from pathlib import Path

src_dir = Path(r"C:\Users\qq939\Downloads\image")
dst_dir = Path(r"C:\Users\qq939\Downloads\image_coco")

# 1. 列出源文件的前10个
src_files = sorted([f for f in src_dir.glob("*.json") if f.name != "annotations.json"])
print("=== 源文件前10个 ===")
for f in src_files[:10]:
    name = f.stem
    # 解析帧号
    if name.startswith("frame_"):
        part = name.split("_")[1]
        digits = ''.join(c for c in part if c.isdigit())
        frame_idx = int(digits) if digits else None
    else:
        digits = ''.join(c for c in name if c.isdigit())
        frame_idx = int(digits) if digits else None
    print(f"  {f.name} -> frame_idx={frame_idx}")

# 2. 列出目标文件的前10个
dst_files = sorted([f for f in dst_dir.glob("labels/frame_*.json")])
print("\n=== 目标文件前10个 ===")
for f in dst_files[:10]:
    print(f"  {f.name}")

# 3. 对比第一个文件的坐标
print("\n=== 第一个文件的坐标对比 ===")
src_first = src_files[0]
dst_first = dst_files[0] if dst_files else None

with open(src_first, encoding='utf-8') as f:
    src_data = json.load(f)

with open(dst_first, encoding='utf-8') as f:
    dst_data = json.load(f)

print(f"源文件: {src_first.name}")
print(f"目标文件: {dst_first.name}")
print(f"源 shapes: {len(src_data.get('shapes', []))}")
print(f"目标 anns: {len(dst_data)}")

if src_data.get('shapes'):
    shape = src_data['shapes'][0]
    points = shape['points']
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    src_x, src_y = min(xs), min(ys)
    src_w, src_h = max(xs) - src_x, max(ys) - src_y
    print(f"\n源 bbox: [{src_x:.2f}, {src_y:.2f}, {src_w:.2f}, {src_h:.2f}]")

if dst_data:
    ann = dst_data[0]
    print(f"目标 bbox: {ann['bbox']}")

print(f"\n源 category_id: src文件没有")
print(f"目标 category_id: {dst_data[0]['category_id'] if dst_data else 'N/A'}")
