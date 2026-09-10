#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labelx2coco.py - 将labelme格式转换为COCO格式

用法：运行脚本后选择labelme格式文件夹，输入目标分辨率的height和width，
      转换为COCO格式并复制到新文件夹（原名+coco）
"""
import json
import shutil
import random
import string
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
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

    # 从第一个labelme JSON获取源图片尺寸
    first_json = json_files[0]
    with open(first_json, encoding='utf-8') as f:
        first_data = json.load(f)
    src_w = int(first_data.get("imageWidth", 0))
    src_h = int(first_data.get("imageHeight", 0))

    # 如果JSON中没有尺寸，从图片获取
    if src_w <= 0 or src_h <= 0:
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in img_extensions:
            img_files = list(src_dir.glob(f"*{ext}")) + list(src_dir.glob(f"*{ext.upper()}"))
            if img_files:
                try:
                    with Image.open(img_files[0]) as img:
                        src_w, src_h = img.size
                        print(f"[INFO] Got size from image: {src_w}x{src_h}")
                    break
                except Exception:
                    pass

    if src_w <= 0 or src_h <= 0:
        src_w, src_h = target_w, target_h
        print(f"[WARNING] Could not determine source size, using target size")

    print(f"[INFO] Source size: {src_w}x{src_h}")
    print(f"[INFO] Target size: {target_w}x{target_h}")

    # 计算缩放比例
    ratio_x = target_w / src_w if src_w > 0 else 1.0
    ratio_y = target_h / src_h if src_h > 0 else 1.0
    print(f"[INFO] Ratio: x={ratio_x:.4f}, y={ratio_y:.4f}")

    # 收集类别（保持首次出现的顺序）
    cat_map = {}  # label -> category_id
    categories = []  # 按首次出现顺序

    def get_or_create_category(label):
        if label not in cat_map:
            cat_id = len(categories) + 1
            cat_map[label] = cat_id
            categories.append({"id": cat_id, "name": label, "supercategory": ""})
        return cat_map[label]

    # 构建 frame_XXXXXX.json 文件，同时处理图片
    frame_jsons = {}  # frame_idx -> list of ann
    processed_frames = 0

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

            anns = []
            for shape in d.get("shapes", []):
                label = shape.get("label", "unknown")
                stype = shape.get("shape_type", "rectangle")
                points = shape.get("points", [])

                # 确保类别已注册（保持首次出现顺序）
                get_or_create_category(label)

                if stype == "rectangle" and len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)

                    # 缩放坐标
                    x_scaled = x * ratio_x
                    y_scaled = y * ratio_y
                    w_scaled = w * ratio_x
                    h_scaled = h * ratio_y

                    anns.append({
                        "id": len(anns) + 1,
                        "category_id": cat_map[label],
                        "track_id": 0,
                        "trace_id_list": [0],
                        "bbox": [x_scaled, y_scaled, w_scaled, h_scaled],
                        "area": w_scaled * h_scaled,
                        "segmentation": [[x_scaled, y_scaled, x_scaled + w_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled, x_scaled, y_scaled + h_scaled]],
                        "iscrowd": 0
                    })
                elif stype == "polygon" and len(points) >= 3:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y

                    # 缩放segmentation坐标
                    seg = []
                    for p in points:
                        seg.append(p[0] * ratio_x)
                        seg.append(p[1] * ratio_y)

                    anns.append({
                        "id": len(anns) + 1,
                        "category_id": cat_map[label],
                        "track_id": 0,
                        "trace_id_list": [0],
                        "bbox": [x * ratio_x, y * ratio_y, w * ratio_x, h * ratio_y],
                        "area": w * h * ratio_x * ratio_y,
                        "segmentation": [seg],
                        "iscrowd": 0
                    })

            frame_jsons[frame_idx] = anns

            # 处理图片文件
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
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

    # 生成 annotations.json
    short_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    ann_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": f"labelme converted {short_id}",
            "width": target_w,
            "height": target_h
        },
        "categories": categories,
        "images": [
            {"id": i, "file_name": f"frame_{i:06d}.jpg", "width": target_w, "height": target_h}
            for i in sorted(frame_jsons.keys())
        ],
        "annotations": []
    }

    with open(dst_dir / "annotations.json", 'w', encoding='utf-8') as f:
        json.dump(ann_data, f, ensure_ascii=False)

    print(f"[DONE] Output: {dst_dir}")
    print(f"[DONE] Frames: {len(frame_jsons)}, Images: {processed_frames}, Categories: {len(categories)}")
    print(f"[DONE] Category mapping:")
    for cat in categories:
        print(f"       id={cat['id']}: {cat['name']}")
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
        first_json = src_path / [f for f in src_path.glob("*.json") if f.name != "annotations.json"][0].name if any(f.name != "annotations.json" for f in src_path.glob("*.json")) else None
        json_files = [f for f in src_path.glob("*.json") if f.name != "annotations.json"]
        if json_files:
            with open(json_files[0], encoding='utf-8') as f:
                d = json.load(f)
            if d.get("imageWidth") and d.get("imageHeight"):
                default_w = int(d["imageWidth"])
                default_h = int(d["imageHeight"])
    except Exception:
        pass

    # 输入目标分辨率（使用默认值）
    height_str, ok = QInputDialog.getText(None, "Target Height", f"Enter target image height\n(Default: {default_h}):")
    if not ok:
        print("[INFO] Cancelled")
        return
    if not height_str.strip():
        target_h = default_h
    else:
        try:
            target_h = int(height_str.strip())
        except ValueError:
            QMessageBox.critical(None, "Error", "Height must be an integer!")
            return

    width_str, ok = QInputDialog.getText(None, "Target Width", f"Enter target image width\n(Default: {default_w}):")
    if not ok:
        print("[INFO] Cancelled")
        return
    if not width_str.strip():
        target_w = default_w
    else:
        try:
            target_w = int(width_str.strip())
        except ValueError:
            QMessageBox.critical(None, "Error", "Width must be an integer!")
            return

    try:
        target_h = int(height_str.strip())
        target_w = int(width_str.strip())
    except ValueError:
        QMessageBox.critical(None, "Error", "Width and height must be integers!")
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
