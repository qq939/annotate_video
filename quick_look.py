#!/usr/bin/env python3
"""快速导出模型到OBS"""

import json
import shutil
import subprocess
from pathlib import Path

# 类别名称
CLASS_NAMES = ['压紧', '检测接头', '产品', '产品1']

def export_model():
    # 模型路径
    pt_path = Path("runs/detect/yolo_runs/train-2/weights/best.pt")
    if not pt_path.exists():
        print(f"[错误] 模型文件不存在: {pt_path}")
        return
    
    print(f"[导出] 加载模型: {pt_path}")
    
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    
    # 导出ONNX
    print("[导出] 正在导出ONNX...")
    model.export(format="onnx")
    
    onnx_path = pt_path.parent / "best.onnx"
    if not onnx_path.exists():
        print("[错误] ONNX导出失败")
        return
    
    print(f"[导出] ONNX文件: {onnx_path}")
    
    # 创建model.json
    model_json = {
        "id": "model_001",
        "displayname": "物体检测",
        "description": "物体检测模型",
        "model_path": "best.onnx",
        "classes": CLASS_NAMES,
        "nc": len(CLASS_NAMES),
        "input_size": [640, 640]
    }
    
    # 保存model.json到weights文件夹
    model_json_path = pt_path.parent / "model.json"
    with open(model_json_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)
    print(f"[导出] model.json: {model_json_path}")
    
    # 整体拷贝到1dst
    upload_dir = Path("1dst/model_001_train")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    shutil.copytree(pt_path.parent, upload_dir)
    print(f"[导出] 拷贝到: {upload_dir}")
    
    # 压缩上传
    import zipfile
    zip_filename = "model_001_train.zip"
    zip_path = Path("1dst") / zip_filename
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in upload_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(upload_dir.parent))
    print(f"[导出] 压缩完成: {zip_path}")
    
    # 上传到OBS
    zip_url = f"http://obs.dimond.top/{zip_filename}"
    print(f"[上传] 正在上传...")
    result = subprocess.run(['curl', '--upload-file', str(zip_path), zip_url], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[成功] 上传完成: {zip_url}")
    else:
        print(f"[失败] 上传失败: {result.stderr}")

if __name__ == "__main__":
    export_model()
