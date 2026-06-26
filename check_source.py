import json
from pathlib import Path

# 检查final_data
final_dir = Path("final_data")
if final_dir.exists():
    labels_dir = final_dir / "labels"
    if labels_dir.exists():
        label_file = labels_dir / "frame_000204.json"
        if label_file.exists():
            with open(label_file) as f:
                annotations = json.load(f)
            print(f"final_data/frame_000204.json: {len(annotations)} 个标注")
            for i, ann in enumerate(annotations):
                print(f"  {i}: track_id={ann.get('track_id')}, bbox={ann.get('bbox')}")
        else:
            print("final_data/labels/frame_000204.json 不存在")
    else:
        print("final_data/labels 不存在")
else:
    print("final_data 不存在")

# 检查temp_data_post
print("\n---")
temp_dir = Path("temp_data_post")
labels_dir = temp_dir / "labels"
if labels_dir.exists():
    label_file = labels_dir / "frame_000204.json"
    if label_file.exists():
        with open(label_file) as f:
            annotations = json.load(f)
        print(f"temp_data_post/frame_000204.json: {len(annotations)} 个标注")
        for i, ann in enumerate(annotations):
            print(f"  {i}: track_id={ann.get('track_id')}, bbox={ann.get('bbox')}")
    else:
        print("temp_data_post/labels/frame_000204.json 不存在")
