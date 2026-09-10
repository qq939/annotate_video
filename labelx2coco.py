#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labelx2coco.py - 将labelme格式转换为COCO格式

用法：运行脚本后选择labelme格式文件夹，输入目标分辨率的height和width，
      转换为COCO格式并复制到新文件夹（原名+coco）
"""
import json
import shutil
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox, QLineEdit
import sys
from PIL import Image


def convert_labelme_to_coco(src_dir, dst_dir, target_w, target_h):
    """将labelme JSON文件批量转换为COCO格式

    Args:
        src_dir: 源labelme格式目录
        dst_dir: 目标COCO格式目录
        target_w: 目标宽度
        target_h: 目标高度
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # 检查目标目录是否存在
    if dst_dir.exists():
        reply = QMessageBox.question(
            None, "目录已存在",
            f"目标目录 {dst_dir.name} 已存在。\n\n选择 \"是\" 跳过已处理的帧（从断点继续）\n选择 \"否\" 完全重建（删除后重新转换）\n选择 \"取消\" 退出",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if reply == QMessageBox.Cancel:
            print("[INFO] Cancelled")
            return False
        elif reply == QMessageBox.No:
            try:
                import shutil
                shutil.rmtree(dst_dir)
            except PermissionError:
                print(f"[ERROR] 无法删除目录 {dst_dir}，可能是权限问题。请手动删除后重试。")
                return False
        # 如果选择 Yes (QMessageBox.Yes)，则继续但不删除目录，从断点继续

    # 创建目标目录结构
    dst_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = dst_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    frames_dir = dst_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # 查找所有labelme JSON文件（排除annotations.json）
    json_files = [f for f in src_dir.glob("*.json") if f.name != "annotations.json"]
    if not json_files:
        print(f"[ERROR] No .json files found in {src_dir}")
        return False

    print(f"[INFO] Found {len(json_files)} JSON files")

    print(f"[INFO] Target size: {target_w}x{target_h}")
    print(f"[DEBUG] Processing first file: {sorted(json_files)[0].name}")

    # 构建 frame_XXXXXX.json 文件，同时处理图片
    frame_jsons = {}  # frame_idx -> list of ann
    processed_frames = 0
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

    for idx, jf in enumerate(sorted(json_files)):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)

            # 文件名解析帧号
            name = jf.stem
            frame_idx = None
            if name.startswith("frame_"):
                part = name.split("_")[1]
                digits = ''.join(c for c in part if c.isdigit())
                if digits:
                    frame_idx = int(digits)
            else:
                digits = ''.join(c for c in name if c.isdigit())
                if digits:
                    frame_idx = int(digits)

            if frame_idx is None:
                frame_idx = idx

            # 查找该帧对应图片，每帧独立读取真实尺寸（混合尺寸源不能用全局尺寸）
            src_img = None
            for ext in img_extensions:
                potential_img = src_dir / f"{name}{ext}"
                if potential_img.exists():
                    src_img = potential_img
                    break
            if src_img is None:
                for subdir in [src_dir / "images", src_dir / "img", src_dir / "pics"]:
                    for ext in img_extensions:
                        potential_img = subdir / f"{name}{ext}"
                        if potential_img.exists():
                            src_img = potential_img
                            break
                    if src_img:
                        break

            src_w, src_h = 0, 0
            if src_img:
                with Image.open(src_img) as img:
                    src_w, src_h = img.size
            if src_w <= 0 or src_h <= 0:
                src_w = int(d.get("imageWidth", 0))
                src_h = int(d.get("imageHeight", 0))
            if src_w <= 0 or src_h <= 0:
                src_w, src_h = target_w, target_h

            # 每帧按自身真实尺寸计算缩放比例
            ratio_x = target_w / src_w if src_w > 0 else 1.0
            ratio_y = target_h / src_h if src_h > 0 else 1.0

            anns = []
            ann_id = 1001  # 每个shape一个唯一的track_id，从1001开始
            for shape in d.get("shapes", []):
                label = shape.get("label", "unknown")
                points = shape.get("points", [])

                # 处理任意点数的 rectangle 或 polygon，统一用 min/max 计算 bbox
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y

                    # 缩放坐标
                    x_scaled = x * ratio_x
                    y_scaled = y * ratio_y
                    w_scaled = w * ratio_x
                    h_scaled = h * ratio_y

                    # 生成4个角点的segmentation [x1,y1,x2,y1,x2,y2,x1,y2]
                    seg = [
                        x_scaled, y_scaled,           # 左上
                        x_scaled + w_scaled, y_scaled,  # 右上
                        x_scaled + w_scaled, y_scaled + h_scaled,  # 右下
                        x_scaled, y_scaled + h_scaled  # 左下
                    ]

                    anns.append({
                        "bbox": [x_scaled, y_scaled, w_scaled, h_scaled],
                        "track_id": ann_id,  # 每个shape有唯一的track_id
                        "segmentation": [seg],  # 4个角点 [x1,y1,x2,y1,x2,y2,x1,y2]
                        "category": label,  # 类别名称
                        "confidence": 1.0,
                        "category_id": 0,  # 由用户在UI中分配
                        "trace_id_list": [ann_id]  # 必须有这个字段
                    })
                    ann_id += 1

            frame_jsons[frame_idx] = anns

            # 处理图片文件（resize 到目标尺寸）
            if src_img:
                try:
                    with Image.open(src_img) as img:
                        # 转换为 RGB（如果是 RGBA 或其他模式）
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
                        dst_img_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                        img_resized.save(dst_img_path, "JPEG", quality=95)
                        processed_frames += 1
                except Exception as e:
                    print(f"[WARN] Could not process image {src_img.name}: {e}")
            else:
                print(f"[WARN] Image not found: {name}")

        except Exception as e:
            print(f"[SKIP] {jf.name}: {e}")

    # 写入 frame_XXXXXX.json 文件
    for frame_idx, anns in sorted(frame_jsons.items()):
        with open(labels_dir / f"frame_{frame_idx:06d}.json", 'w', encoding='utf-8') as f:
            json.dump(anns, f, ensure_ascii=False)

    # 生成 annotations.json（categories为空，由用户在UI中分配）
    ann_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "labelme converted",
            "width": target_w,
            "height": target_h,
            "fps": 30,
            "fourcc": "mp4v"
        },
        "categories": [],  # 类别由用户在UI中分配
        "images": [
            {"id": i, "file_name": f"frame_{i:06d}.jpg", "width": target_w, "height": target_h}
            for i in sorted(frame_jsons.keys())
        ],
        "annotations": []
    }

    with open(dst_dir / "annotations.json", 'w', encoding='utf-8') as f:
        json.dump(ann_data, f, ensure_ascii=False)

    print(f"[DONE] Output: {dst_dir}")
    print(f"[DONE] Frames: {len(frame_jsons)}, Images: {processed_frames}")
    return True


def main():
    app = QApplication(sys.argv)

    # 选择源文件夹
    src_dir = QFileDialog.getExistingDirectory(None, "Select labelme folder", ".")
    if not src_dir:
        print("[INFO] Cancelled")
        return

    src_path = Path(src_dir)

    # 自动检测源图片分辨率并设为默认值
    default_w, default_h = 2012, 1518
    try:
        json_files = [f for f in src_path.glob("*.json") if f.name != "annotations.json"]
        if json_files:
            with open(json_files[0], encoding='utf-8') as f:
                d = json.load(f)
            if d.get("imageWidth") and d.get("imageHeight"):
                default_w = int(d["imageWidth"])
                default_h = int(d["imageHeight"])
    except Exception:
        pass

    # 输入目标分辨率（使用默认值填入文本框）
    height_str, ok = QInputDialog.getText(None, "Target Height", "Enter target image height:", QLineEdit.Normal, str(default_h))
    if not ok:
        print("[INFO] Cancelled")
        return
    try:
        target_h = int(height_str.strip())
    except ValueError:
        QMessageBox.critical(None, "Error", "Height must be an integer!")
        return

    width_str, ok = QInputDialog.getText(None, "Target Width", "Enter target image width:", QLineEdit.Normal, str(default_w))
    if not ok:
        print("[INFO] Cancelled")
        return
    try:
        target_w = int(width_str.strip())
    except ValueError:
        QMessageBox.critical(None, "Error", "Width must be an integer!")
        return

    if target_h <= 0 or target_w <= 0:
        QMessageBox.critical(None, "Error", "Width and height must be positive integers!")
        return

    # 构建目标文件夹名
    dst_dir = src_path.parent / (src_path.name + "_coco")
    if dst_dir.exists():
        reply = QMessageBox.question(
            None, "Confirm",
            f"Target folder exists: {dst_dir}\nOverwrite?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            print("[INFO] Cancelled")
            return
        shutil.rmtree(dst_dir)

    print(f"[INFO] Source: {src_path}")
    print(f"[INFO] Target: {dst_dir}")
    print(f"[INFO] Size: {target_w}x{target_h}")

    success = convert_labelme_to_coco(src_path, dst_dir, target_w, target_h)

    if success:
        QMessageBox.information(None, "Done", f"Conversion complete!\n\nOutput: {dst_dir}")
    else:
        QMessageBox.critical(None, "Error", "Conversion failed!")


if __name__ == "__main__":
    main()
