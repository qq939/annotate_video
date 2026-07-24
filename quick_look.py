#!/usr/bin/env python3
"""快速导出模型到OBS"""

import json
import shutil
import subprocess
from pathlib import Path


def export_model():
    # 查找最新训练文件夹（train*）
    yolo_runs_dir = Path(r"C:\Users\jiali\Downloads\annotate_video\runs\detect\yolo_runs")
    train_dirs = list(yolo_runs_dir.glob("train*"))
    train_dirs = [d for d in train_dirs if d.is_dir() and (d / "weights").exists()]
    
    if not train_dirs:
        print("[错误] 没有找到训练文件夹")
        return
    
    # 按编号排序，取最大的
    def get_train_num(p):
        name = p.name
        if name == "train":
            return 0
        suffix = name.replace("train", "")
        nums = [int(x) for x in suffix.split("-") if x.isdigit()]
        return nums[-1] if nums else 0
    
    train_dirs.sort(key=get_train_num, reverse=True)
    train_dir = train_dirs[0]
    
    # 优先使用best.pt
    pt_path = train_dir / "weights" / "best.pt"
    if not pt_path.exists():
        pt_path = train_dir / "weights" / "last.pt"
    if not pt_path.exists():
        print(f"[错误] 模型文件不存在: {train_dir / 'weights'}")
        return
    
    print(f"[导出] 使用模型: {pt_path}")
    
    # 加载同目录的model.json
    model_json_path = train_dir / "weights" / "model.json"
    if model_json_path.exists():
        with open(model_json_path, encoding='utf-8') as f:
            model_info = json.load(f)
        print(f"[导出] 读取model.json: {model_json_path}")
        model_id = model_info.get('id', 'model_001')
        model_name = model_info.get('displayname', '物体检测')
        model_desc = model_info.get('description', '物体检测模型')
    else:
        print(f"[警告] model.json不存在，使用默认信息")
        model_id = "model_001"
        model_name = "物体检测"
        model_desc = "物体检测模型"
    
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
    
    # 整体拷贝到1dst
    upload_dir = Path(f"1dst/{model_id}_train")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    shutil.copytree(pt_path.parent, upload_dir)
    print(f"[导出] 拷贝到: {upload_dir}")
    
    # 压缩上传
    import zipfile
    zip_filename = f"{model_id}_train.zip"
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
