# -*- coding: utf-8 -*-
"""验证bbox计算"""
import json
from pathlib import Path
from PIL import Image

src_dir = Path(r"C:\Users\qq939\Downloads\image")

# 目标分辨率（和源相同）
target_w, target_h = 2012, 1518

# 获取文件列表
json_files = sorted([f for f in src_dir.glob("*.json") if f.name != "annotations.json"])

# 从图片获取真实尺寸
img_files = sorted([f for f in src_dir.glob("*.jpg")])
with Image.open(img_files[0]) as img:
    src_w, src_h = img.size
print(f"源尺寸(从图片): {src_w}x{src_h}")
print(f"目标尺寸: {target_w}x{target_h}")

ratio_x = target_w / src_w
ratio_y = target_h / src_h
print(f"比例: {ratio_x}, {ratio_y}")

# 处理第一个文件
jf = json_files[0]
with open(jf, encoding='utf-8') as f:
    d = json.load(f)

shape = d['shapes'][0]
points = shape['points']
print(f"\n源points: {points}")

# 手动计算正确的bbox
xs = [p[0] for p in points]
ys = [p[1] for p in points]
x, y = min(xs), min(ys)
w, h = max(xs) - x, max(ys) - y
print(f"正确bbox (ratio=1): [{x}, {y}, {w}, {h}]")

# 缩放后的bbox
x_scaled = x * ratio_x
y_scaled = y * ratio_y
w_scaled = w * ratio_x
h_scaled = h * ratio_y
print(f"缩放后bbox: [{x_scaled}, {y_scaled}, {w_scaled}, {h_scaled}]")
