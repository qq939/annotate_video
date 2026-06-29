# 查找需要修改的位置
import re

with open(r'c:\Users\qq939\Downloads\annotate_video\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到 effect 部分的起始
marker = "cv2.addWeighted(overlay, self.ctrl.alpha, result_frame, 1 - self.ctrl.alpha, 0, result_frame)"
idx = content.find(marker)
print(f"Found marker at index: {idx}")

# 显示周围的内容
if idx > 0:
    print("Context around marker:")
    print(content[idx-200:idx+200])
