# -*- coding: utf-8 -*-
"""检查JSON中的尺寸"""
import json
from pathlib import Path

src_dir = Path(r"C:\Users\qq939\Downloads\image")

# 检查第一个JSON
json_files = sorted([f for f in src_dir.glob("*.json") if f.name != "annotations.json"])
first = json_files[0]

with open(first, encoding='utf-8') as f:
    data = json.load(f)

print(f"JSON: {first.name}")
print(f"JSON中的尺寸: {data.get('imageWidth')}x{data.get('imageHeight')}")

# 检查第一个shape的points
shape = data['shapes'][0]
print(f"\n第一个shape:")
print(f"  label: {shape['label']}")
print(f"  points: {shape['points']}")
