#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cocomaker.py - 模型文件+视频文件 -> temp_data 格式 COCO 数据集

功能：
1. 从模型压缩包（.zip/.rar）提取 yolo_runs/.../annotations.json 和 labels/*.json
2. 从视频文件（可多选）提取帧，按 frame_skip 跳帧采样，统一 resize 到目标尺寸
3. 合并所有来源的帧，按字符串排序后递增编号（frame_000000.jpg 等）
4. 输出符合 app 的 temp_data 格式：frames/、labels/、annotations.json

用法（无 GUI，直接命令行调用）：
    from cocomaker import convert_to_coco_dataset
    result_dir = convert_to_coco_dataset(
        video_paths=["path/to/video1.mp4", "path/to/video2.mp4"],
        model_archive="path/to/model.zip",
        output_dir="path/to/output",
        target_w=2012,
        target_h=1518,
        frame_skip=1
    )
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# PIL 在 labelx2coco.py 中已使用，此处复用相同路径
sys.path.insert(0, str(Path(__file__).parent))
try:
    from PIL import Image
except ImportError:
    Image = None

# ----------------------------------------------------------------------
# 核心函数
# ----------------------------------------------------------------------


def _extract_archive(archive_path: Path, dest_dir: Path):
    """解压压缩包，支持 ZIP 和 RAR，优先尝试 rarfile，再尝试 7z 命令"""
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_path, "rb") as f:
        magic = f.read(8)
    is_zip = magic[:2] == b"PK"
    is_rar = magic[:7] == b"Rar!\x1a\x07"

    if is_zip:
        import zipfile

        def _try_zip(mod):
            with mod.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
            return True

        tried = []
        # 策略1：标准 zipfile
        try:
            if _try_zip(zipfile):
                print("[INFO] 标准 zipfile 解压成功")
                return True
        except Exception as e:
            tried.append(f"标准zipfile: {e}")

        # 策略2：zipfile_deflate64
        try:
            import zipfile_deflate64

            if _try_zip(zipfile_deflate64):
                print("[INFO] zipfile_deflate64 解压成功")
                return True
        except ImportError:
            tried.append("zipfile_deflate64 未安装")
        except Exception as e:
            tried.append(f"zipfile_deflate64: {e}")

        raise RuntimeError(f"ZIP 解压失败: {'; '.join(tried)}")

    elif is_rar:
        # 策略1：rarfile（需要 unrar.dll）
        try:
            import rarfile

            rarfile.UNRAR_TOOL = str(
                Path(__file__).parent / "UnRAR.exe"
            )  # 优先使用同目录的 UnRAR.exe
            with rarfile.RarFile(archive_path, "r") as rf:
                rf.extractall(dest_dir)
            print("[INFO] rarfile 解压成功")
            return True
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] rarfile 解压失败: {e}")

        # 策略2：7z 命令行
        sevenz = shutil.which("7z") or shutil.which("7za")
        if sevenz:
            result = subprocess.run(
                [sevenz, "x", str(archive_path), f"-o{dest_dir}", "-y"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("[INFO] 7z 解压成功")
                return True
            else:
                print(f"[WARN] 7z 解压失败: {result.stderr}")
        else:
            print("[WARN] 未找到 7z 命令，rarfile 也未安装")

        raise RuntimeError(
            f"RAR 解压失败（rarfile: 未安装或缺少unrar，7z: {'未找到' if not sevenz else '执行失败'}）"
        )

    else:
        raise ValueError(f"不支持的压缩格式，magic: {magic[:8].hex()}")


def _find_model_annotation(model_dir: Path):
    """在模型解压目录中查找 yolo_runs/.../annotations.json"""
    # SOP2MODEL 结构：yolo_runs/{train_id}/weights/annotations.json
    for p in model_dir.rglob("annotations.json"):
        return p
    raise FileNotFoundError(
        f"模型目录 {model_dir} 中未找到 annotations.json，"
        "请确认压缩包包含 yolo_runs 结构"
    )


def _extract_video_frames(
    video_paths: list, dest_dir: Path, target_w: int, target_h: int, frame_skip: int = 1
) -> dict:
    """
    从视频文件提取帧并 resize 到目标尺寸。

    Args:
        video_paths: 视频文件路径列表
        dest_dir: 输出帧图片的目标目录
        target_w, target_h: 目标分辨率
        frame_skip: 每隔 frame_skip 帧取一帧（1=全取）

    Returns:
        {frame_idx: {"file_name": "...", "width": w, "height": h}}
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    frame_meta = {}  # frame_idx -> image metadata
    global_idx = 0

    for vp in video_paths:
        vp = Path(vp)
        if not vp.exists():
            print(f"[WARN] 视频不存在，跳过: {vp}")
            continue

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[WARN] 无法打开视频: {vp}")
            continue

        vid_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if vid_frames % frame_skip == 0:
                # resize
                if Image:
                    pil_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(pil_frame)
                    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
                    rgb_frame = np.array(pil_img)
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                else:
                    bgr_frame = cv2.resize(frame, (target_w, target_h))

                out_name = f"frame_{global_idx:06d}.jpg"
                out_path = dest_dir / out_name
                cv2.imwrite(str(out_path), bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                frame_meta[global_idx] = {
                    "file_name": out_name,
                    "width": target_w,
                    "height": target_h,
                }
                global_idx += 1

            vid_frames += 1

        cap.release()
        print(f"[INFO] 视频 {vp.name}: 提取 {vid_frames} 帧（采样率 1/{frame_skip}） -> {vid_frames // frame_skip} 帧")

    return frame_meta


def _extract_model_labels(
    model_dir: Path, dest_dir: Path, target_w: int, target_h: int
) -> dict:
    """
    从模型目录提取 labels/*.json，坐标缩放到目标尺寸。

    SOP2MODEL 格式 label：
    - src: 2012x1518, 640x640 混合
    - dst: 统一 target_w x target_h

    Returns:
        {frame_idx: [annotation, ...]}
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    ann_meta = {}  # frame_idx -> list of annotations

    # 查找 yolo_runs/.../labels/ 下的所有 frame_*.json
    labels_dir = None
    for p in model_dir.rglob("labels"):
        if p.is_dir():
            files = list(p.glob("frame_*.json"))
            if files:
                labels_dir = p
                break

    if labels_dir is None:
        print("[WARN] 模型目录中未找到 labels/ 目录，跳过模型标注")
        return ann_meta

    # 读取模型的 annotations.json 获取元信息（源尺寸等）
    ann_json_path = None
    for p in model_dir.rglob("annotations.json"):
        ann_json_path = p
        break

    frame_annotations = {}  # 原 frame_idx -> list of ann

    for lf in sorted(labels_dir.glob("frame_*.json")):
        with open(lf, encoding="utf-8") as f:
            labels = json.load(f)

        frame_idx_str = lf.stem  # "frame_000000"
        digits = "".join(c for c in frame_idx_str if c.isdigit())
        if digits:
            frame_idx = int(digits)
        else:
            frame_idx = list(frame_annotations.keys())[-1] + 1 if frame_annotations else 0

        ann_list = []
        ann_id = 1001

        for label in labels:
            bbox = label.get("bbox", [])
            category = label.get("category", label.get("class", "unknown"))
            seg = label.get("segmentation", [[]])[0] if label.get("segmentation") else []
            track_id = label.get("track_id", label.get("id", ann_id))
            trace_id_list = label.get(
                "trace_id_list", label.get("trace_ids", [track_id])
            )
            confidence = label.get("confidence", 1.0)
            category_id = label.get("category_id", 0)

            # 获取源尺寸
            src_w = label.get("width", target_w)
            src_h = label.get("height", target_h)

            # 计算缩放比例
            ratio_x = target_w / src_w if src_w > 0 else 1.0
            ratio_y = target_h / src_h if src_h > 0 else 1.0

            # 缩放 bbox
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
                bx_s = bx * ratio_x
                by_s = by * ratio_y
                bw_s = bw * ratio_x
                bh_s = bh * ratio_y
            else:
                bx_s, by_s, bw_s, bh_s = 0, 0, 0, 0

            # 缩放 segmentation
            seg_scaled = [p * ratio_x if i % 2 == 0 else p * ratio_y
                          for i, p in enumerate(seg)]

            ann_list.append({
                "bbox": [bx_s, by_s, bw_s, bh_s],
                "track_id": track_id,
                "segmentation": [seg_scaled],
                "category": category,
                "confidence": confidence,
                "category_id": category_id,
                "trace_id_list": trace_id_list if isinstance(trace_id_list, list) else [trace_id_list],
            })
            ann_id += 1

        frame_annotations[frame_idx] = ann_list

    # 写入输出目录
    for frame_idx, ann_list in sorted(frame_annotations.items()):
        out_name = f"frame_{frame_idx:06d}.json"
        out_path = dest_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ann_list, f, ensure_ascii=False)
        ann_meta[frame_idx] = ann_list

    print(f"[INFO] 模型标注：写入 {len(ann_meta)} 个 label 文件")
    return ann_meta


def _merge_and_write(
    frame_src_dir: Path,
    label_src_dir: Path,
    output_dir: Path,
    target_w: int,
    target_h: int,
):
    """
    合并视频帧和模型标注，统一编号输出。

    帧号按视频帧在前、模型标注补齐的顺序递增。
    如果模型标注的帧号与视频重叠，模型标注覆盖视频帧。
    """
    output_dir = Path(output_dir)
    frames_out = output_dir / "frames"
    labels_out = output_dir / "labels"
    frames_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # 收集所有需要输出的帧号
    frame_idxs = sorted(frame_src_dir.glob("frame_*.jpg"))
    frame_map = {}  # old_frame_idx -> src_file
    for f in frame_idxs:
        digits = "".join(c for c in f.stem.split("_")[1] if c.isdigit())
        if digits:
            frame_map[int(digits)] = f

    label_idxs = sorted(label_src_dir.glob("frame_*.json"))
    label_map = {}  # old_frame_idx -> src_file
    for f in label_idxs:
        digits = "".join(c for c in f.stem.split("_")[1] if c.isdigit())
        if digits:
            label_map[int(digits)] = f

    all_frames = sorted(set(frame_map.keys()) | set(label_map.keys()))

    images = []
    for new_idx, old_idx in enumerate(all_frames):
        new_name = f"frame_{new_idx:06d}"

        # 复制帧文件
        if old_idx in frame_map:
            shutil.copy2(str(frame_map[old_idx]), str(frames_out / f"{new_name}.jpg"))

        # 重命名 label 文件
        if old_idx in label_map:
            shutil.copy2(str(label_map[old_idx]), str(labels_out / f"{new_name}.json"))

        images.append({
            "id": new_idx,
            "file_name": f"{new_name}.jpg",
            "width": target_w,
            "height": target_h,
        })

    # 写入 annotations.json
    ann_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "cocomaker generated",
            "width": target_w,
            "height": target_h,
            "fps": 30,
            "fourcc": "mp4v",
        },
        "categories": [],  # 类别由用户在 UI 中分配
        "images": images,
        "annotations": [],
    }

    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(ann_data, f, ensure_ascii=False)

    return output_dir


# ----------------------------------------------------------------------
# 公开 API
# ----------------------------------------------------------------------


def convert_to_coco_dataset(
    video_paths: list,
    model_archive: str,
    output_dir: str,
    target_w: int = 2012,
    target_h: int = 1518,
    frame_skip: int = 1,
) -> Path:
    """
    从视频文件和模型压缩包生成 temp_data 格式 COCO 数据集。

    Args:
        video_paths: 视频文件路径列表（支持 mp4/avi/mov 等 cv2 支持的格式）
        model_archive: 模型压缩包路径（.zip 或 .rar，
                       内含 yolo_runs/{id}/weights/annotations.json 和 labels/）
        output_dir: 输出目录
        target_w, target_h: 目标分辨率（默认 2012x1518）
        frame_skip: 跳帧采样间隔（默认 1=全取）

    Returns:
        输出目录 Path
    """
    output_dir = Path(output_dir)
    temp_root = Path(tempfile.mkdtemp(prefix="cocomaker_"))

    try:
        print(f"[INFO] 目标尺寸: {target_w}x{target_h}")
        print(f"[INFO] 视频数量: {len(video_paths)}")
        print(f"[INFO] 模型文件: {model_archive}")
        print(f"[INFO] 跳帧采样: 每 {frame_skip} 帧取 1 帧")

        # 1. 从视频提取帧（输出到 temp_root/frames/）
        frames_dir = temp_root / "frames"
        frame_meta = _extract_video_frames(
            video_paths, frames_dir, target_w, target_h, frame_skip
        )
        print(f"[INFO] 视频帧提取完成: {len(frame_meta)} 帧")

        # 2. 解压模型，获取标注
        model_extract_dir = temp_root / "model_extract"
        _extract_archive(Path(model_archive), model_extract_dir)

        # 3. 从模型提取 labels/*.json（输出到 temp_root/labels/）
        labels_dir = temp_root / "labels"
        label_meta = _extract_model_labels(
            model_extract_dir, labels_dir, target_w, target_h
        )

        # 4. 合并视频帧和模型标注
        result = _merge_and_write(
            frames_dir, labels_dir, output_dir, target_w, target_h
        )

        print(f"[DONE] 输出: {result}")
        print(f"[DONE] 总帧数: {len(list((result / 'frames').glob('frame_*.jpg')))}")
        return result

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


# ----------------------------------------------------------------------
# GUI 入口
# ----------------------------------------------------------------------

def main():
    """命令行入口，选择文件后调用 convert_to_coco_dataset"""
    try:
        from PyQt5.QtWidgets import (
            QApplication, QFileDialog, QInputDialog, QMessageBox, QWidget
        )
    except ImportError:
        print("[ERROR] 需要 PyQt5，请 pip install PyQt5")
        sys.exit(1)

    app = QApplication(sys.argv)

    # 选择输出目录
    output_dir = QFileDialog.getExistingDirectory(
        None, "选择输出目录（coco数据集存放位置）", "."
    )
    if not output_dir:
        print("[INFO] 取消")
        return

    # 选择视频文件（可多选）
    videos, _ = QFileDialog.getOpenFileNames(
        None, "选择视频文件（可多选）", ".",
        "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
    )
    if not videos:
        print("[WARN] 未选择视频")
        # 不退出，允许只有模型的情况

    # 选择模型压缩包
    model_file, _ = QFileDialog.getOpenFileName(
        None, "选择模型压缩包（.zip/.rar）", ".",
        "压缩包 (*.zip *.rar);;所有文件 (*.*)"
    )
    if not model_file:
        print("[ERROR] 未选择模型文件")
        return

    # 输入目标分辨率
    w, ok_w = QInputDialog.getInt(None, "目标宽度", "输入目标宽度（像素）", value=2012, min=1, max=10000)
    if not ok_w:
        return
    h, ok_h = QInputDialog.getInt(None, "目标高度", "输入目标高度（像素）", value=1518, min=1, max=10000)
    if not ok_h:
        return

    # 输入跳帧比例
    skip, ok_skip = QInputDialog.getInt(
        None, "跳帧采样", "每隔 N 帧取 1 帧（1=全取，2=取一半，...）", value=1, min=1, max=100
    )
    if not ok_skip:
        return

    result_dir = convert_to_coco_dataset(
        video_paths=videos,
        model_archive=model_file,
        output_dir=output_dir,
        target_w=w,
        target_h=h,
        frame_skip=skip,
    )

    QMessageBox.information(
        None, "完成",
        f"COOC 数据集已生成：\n{result_dir}\n\n"
        f"共 {len(list((result_dir / 'frames').glob('frame_*.jpg')))} 帧"
    )


if __name__ == "__main__":
    main()
