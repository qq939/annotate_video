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

    # 查找所有labelme JSON文件
    json_files = list(src_dir.glob("*.json"))
    if not json_files:
        print(f"[错误] 目录 {src_dir} 中没有找到 .json 文件")
        return False

    print(f"[INFO] 找到 {len(json_files)} 个 JSON 文件")

    # 从 labelme JSON 推断源图片尺寸（取第一个有 imageWidth/imageHeight 的文件）
    src_w, src_h = None, None
    for jf in json_files[:20]:
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
            if d.get("imageWidth") and d.get("imageHeight"):
                src_w = int(d["imageWidth"])
                src_h = int(d["imageHeight"])
                break
        except Exception:
            pass

    # 如果labelme JSON中没有尺寸，尝试从图片文件获取
    if src_w is None or src_h is None:
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img_files = list(src_dir.glob(f"*{ext}")) + list(src_dir.glob(f"*{ext.upper()}"))
            if img_files:
                try:
                    with Image.open(img_files[0]) as img:
                        src_w, src_h = img.size
                        print(f"[INFO] 从图片获取源尺寸: {src_w}x{src_h}")
                    break
                except Exception:
                    pass

    # 默认尺寸
    if src_w is None or src_h is None:
        src_w, src_h = target_w, target_h
        print(f"[WARNING] 无法确定源尺寸，使用目标尺寸: {src_w}x{src_h}")

    print(f"[INFO] 源图片尺寸: {src_w}x{src_h}")
    print(f"[INFO] 目标图片尺寸: {target_w}x{target_h}")

    # 计算缩放比例（用于转换标注坐标）
    ratio_x = target_w / src_w if src_w > 0 else 1.0
    ratio_y = target_h / src_h if src_h > 0 else 1.0
    print(f"[INFO] 缩放比例: x={ratio_x:.4f}, y={ratio_y:.4f}")

    # 收集所有类别
    all_labels = set()
    for jf in json_files:
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)
            for shape in d.get("shapes", []):
                all_labels.add(shape.get("label", "unknown"))
        except Exception:
            pass

    categories = [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(sorted(all_labels))]
    cat_map = {c["name"]: c["id"] for c in categories}
    print(f"[INFO] 类别数量: {len(categories)}")

    # 构建 frame_XXXXXX.json 文件，同时处理图片
    frame_jsons = {}  # frame_idx -> list of ann
    processed_frames = 0

    for idx, jf in enumerate(sorted(json_files)):
        try:
            with open(jf, encoding='utf-8') as f:
                d = json.load(f)

            # 文件名解析帧号：frame_000123.json → 123，B00214 → 214，其他用索引
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
                        "category_id": cat_map.get(label, 1),
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
                        "category_id": cat_map.get(label, 1),
                        "track_id": 0,
                        "trace_id_list": [0],
                        "bbox": [x * ratio_x, y * ratio_y, w * ratio_x, h * ratio_y],
                        "area": w * h * ratio_x * ratio_y,
                        "segmentation": [seg],
                        "iscrowd": 0
                    })

            frame_jsons[frame_idx] = anns

            # 处理图片文件：查找对应的图片
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            src_img = None
            for ext in img_extensions:
                potential_img = src_dir / f"{name}{ext}"
                if potential_img.exists():
                    src_img = potential_img
                    break

            # 如果没找到，尝试在子目录中查找
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
                # 读取并resize图片
                try:
                    with Image.open(src_img) as img:
                        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
                        dst_img_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                        img_resized.save(dst_img_path, "JPEG")
                        processed_frames += 1
                except Exception as e:
                    print(f"[警告] 无法处理图片 {src_img.name}: {e}")
            else:
                print(f"[警告] 找不到对应图片: {name}")

        except Exception as e:
            print(f"[跳过] {jf.name}: {e}")

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

    print(f"[完成] 输出到 {dst_dir}")
    print(f"[完成] 帧数: {len(frame_jsons)}, 处理图片: {processed_frames}, 类别数: {len(categories)}")
    return True


def main():
    app = QApplication(sys.argv)

    # 选择源文件夹
    src_dir = QFileDialog.getExistingDirectory(None, "选择labelme格式文件夹", ".")
    if not src_dir:
        print("[INFO] 用户取消")
        return

    src_path = Path(src_dir)

    # 输入目标分辨率
    height_str, ok = QInputDialog.getText(None, "输入目标高度", "请输入目标图片的高度（像素）:")
    if not ok or not height_str.strip():
        print("[INFO] 用户取消")
        return

    width_str, ok = QInputDialog.getText(None, "输入目标宽度", "请输入目标图片的宽度（像素）:")
    if not ok or not width_str.strip():
        print("[INFO] 用户取消")
        return

    try:
        target_h = int(height_str.strip())
        target_w = int(width_str.strip())
    except ValueError:
        QMessageBox.critical(None, "错误", "高度和宽度必须是整数！")
        return

    if target_h <= 0 or target_w <= 0:
        QMessageBox.critical(None, "错误", "高度和宽度必须是正整数！")
        return

    # 构建目标文件夹名：原名 + coco
    dst_dir = src_path.parent / (src_path.name + "_coco")
    if dst_dir.exists():
        reply = QMessageBox.question(
            None, "确认",
            f"目标文件夹已存在: {dst_dir}\n是否覆盖？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            print("[INFO] 用户取消")
            return
        shutil.rmtree(dst_dir)

    print(f"[INFO] 源目录: {src_path}")
    print(f"[INFO] 目标目录: {dst_dir}")
    print(f"[INFO] 目标分辨率: {target_w}x{target_h}")

    success = convert_labelme_to_coco(src_path, dst_dir, target_w, target_h)

    if success:
        QMessageBox.information(None, "完成", f"转换完成！\n\n输出目录: {dst_dir}")
    else:
        QMessageBox.critical(None, "错误", "转换失败！")


if __name__ == "__main__":
    main()
