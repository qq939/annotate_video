# -*- coding: utf-8 -*-
"""检查源图片真实尺寸"""
from PIL import Image
from pathlib import Path

src_dir = Path(r"C:\Users\qq939\Downloads\image")

# 从图片获取尺寸
img_files = sorted([f for f in src_dir.glob("*.jpg")])[:3]
for img_file in img_files:
    with Image.open(img_file) as img:
        print(f"{img_file.name}: {img.size}")
