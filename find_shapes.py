import json
from pathlib import Path

labelme_dir = Path("label_x_label_me")
if not labelme_dir.exists():
    print("label_x_label_me 不存在")
    exit()

# 找一个有shapes的帧
json_files = sorted(labelme_dir.glob("*.json"))

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
    
    shapes = data.get('shapes', [])
    if len(shapes) > 1:
        print(f"{json_file.name}: {len(shapes)} 个shapes")
        for i, shape in enumerate(shapes):
            points = shape.get('points', [])
            print(f"  {i}: label={shape.get('label')}, points={points}")
        break
else:
    print("没有找到有多个shapes的帧")
