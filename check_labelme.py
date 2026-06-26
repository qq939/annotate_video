import json
from pathlib import Path

labelme_dir = Path("label_x_label_me")
if not labelme_dir.exists():
    print("label_x_label_me 不存在")
    exit()

# 查找JSON文件
json_files = sorted(labelme_dir.glob("*.json"))
print(f"找到 {len(json_files)} 个JSON文件")

for json_file in json_files[:5]:  # 只检查前5个
    with open(json_file) as f:
        data = json.load(f)
    
    shapes = data.get('shapes', [])
    print(f"\n{json_file.name}: {len(shapes)} 个shapes")
    
    # 检查重复的label
    labels = [s.get('label') for s in shapes]
    for i, shape in enumerate(shapes):
        print(f"  {i}: label={shape.get('label')}, points={shape.get('points')}")
