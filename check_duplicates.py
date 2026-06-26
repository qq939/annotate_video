import json
from collections import Counter
from pathlib import Path

labels_dir = Path("temp_data_post/labels")

# 统计所有帧的标注
total_annotations = 0
frames_with_annotations = 0

for label_file in sorted(labels_dir.glob("frame_*.json"))[:10]:  # 只检查前10帧
    with open(label_file) as f:
        annotations = json.load(f)
    if annotations:
        print(f"\n{label_file.name}: {len(annotations)} 个标注")
        total_annotations += len(annotations)
        frames_with_annotations += 1
        
        track_ids = [a.get('track_id') for a in annotations]
        count = Counter(track_ids)
        dup = [(k, v) for k, v in count.items() if v > 1]
        if dup:
            print(f"  重复: {dup}")
        
        for i, ann in enumerate(annotations):
            print(f"  {i}: track_id={ann.get('track_id')}, category={ann.get('category')}, bbox={ann.get('bbox')[:2] if ann.get('bbox') else None}")

print(f"\n总计: {frames_with_annotations} 帧有标注, 共 {total_annotations} 个标注")
