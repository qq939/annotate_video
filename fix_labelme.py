#!/usr/bin/env python3
"""修复 label_x_label_me 目录下的 JSON 文件"""

import json
from pathlib import Path

def fix_labelme_shapes():
    labelme_dir = Path("label_x_label_me")
    if not labelme_dir.exists():
        print(f"目录不存在: {labelme_dir}")
        return
    
    json_files = list(labelme_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个 JSON 文件")
    
    fixed = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modified = False
            
            # 添加 imagePath（如果缺失）
            if 'imagePath' not in data:
                data['imagePath'] = json_file.stem + '.jpg'
                modified = True
            
            # 添加 shape_type 到每个 shape
            for shape in data.get('shapes', []):
                if 'shape_type' not in shape:
                    shape['shape_type'] = 'rectangle'
                    modified = True
            
            if modified:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                fixed += 1
                print(f"已修复: {json_file.name}")
        
        except Exception as e:
            print(f"错误: {json_file.name} - {e}")
    
    print(f"\n完成！修复了 {fixed} 个文件")

if __name__ == "__main__":
    fix_labelme_shapes()
