# -*- coding: utf-8 -*-
"""检查源图片真实尺寸"""
from PIL import Image
from pathlib import Path

src_dir = Path(r"C:\Users\qq939\Downloads\image")

# 检查多张图片的尺寸
img_files = sorted([f for f in src_dir.glob("*.jpg")])[:5]
print("图片尺寸检查:")
for img_file in img_files:
    with Image.open(img_file) as img:
        print(f"  {img_file.name}: {img.size}")
