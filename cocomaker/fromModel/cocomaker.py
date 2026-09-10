#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cocomaker.py - 模型文件+视频文件 -> temp_data 格式 COCO 数据集

功能：
1. 从模型压缩包（.zip/.rar）提取 ONNX 模型和配置
2. 从视频文件（可多选）提取帧，按 frame_skip 跳帧采样，统一 resize 到目标尺寸
3. 用 ONNX 模型对每帧推理，生成 COCO 格式标注
4. 输出符合 app 的 temp_data 格式：frames/、labels/、annotations.json

用法（无 GUI，直接命令行调用）：
    from cocomaker import convert_to_coco_dataset
    result_dir = convert_to_coco_dataset(
        video_paths=["path/to/video1.mp4", "path/to/video2.mp4"],
        model_archive="path/to/model.rar",
        output_dir="path/to/output",
        target_w=2012,
        target_h=1518,
        frame_skip=5,
        conf_threshold=0.25,
        iou_threshold=0.45
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

try:
    from PIL import Image
except ImportError:
    Image = None

# ONNX runtime，延迟导入（可选）
onnx_available = False
try:
    import onnxruntime as ort
    onnx_available = True
except ImportError:
    ort = None

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
    # RAR: 魔数 "Rar!" + 0x1a 0x07 + archive_flag (0=RAR4, 1=RAR5)
    is_rar = magic[:4] == b"Rar!" and len(magic) >= 6 and magic[4:6] == b"\x1a\x07"

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
        if not sevenz:
            # 尝试常见安装路径
            for p in [r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]:
                if Path(p).exists():
                    sevenz = p
                    break
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
        raise ValueError("不支持的压缩格式（只支持 ZIP 和 RAR），magic: " + magic[:8].hex())


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


def _run_onnx_inference(
    model_archive: Path,
    frames_dir: Path,
    labels_dir: Path,
    target_w: int,
    target_h: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> dict:
    """
    从模型压缩包提取 ONNX，对视频帧运行推理，生成 COCO 格式标注。

    Args:
        model_archive: 模型压缩包路径
        frames_dir: 视频帧目录（frame_000000.jpg 等）
        labels_dir: 标注输出目录
        target_w, target_h: 目标分辨率
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值

    Returns:
        {frame_idx: [annotation, ...]}
    """
    if not onnx_available:
        raise RuntimeError("onnxruntime 未安装，无法进行推理。请 pip install onnxruntime")

    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解压 ONNX 和 model.json
    temp_model_dir = Path(tempfile.mkdtemp(prefix="cocomaker_model_"))
    try:
        _extract_archive(model_archive, temp_model_dir)

        # 查找 best.onnx
        onnx_path = None
        for p in temp_model_dir.rglob("best.onnx"):
            onnx_path = p
            break
        if onnx_path is None:
            raise FileNotFoundError("模型压缩包中未找到 best.onnx")

        # 查找 model.json
        model_json_path = None
        for p in temp_model_dir.rglob("model.json"):
            model_json_path = p
            break

        classes = ["unknown"]
        # classes_id: 如果 model.json 有此字段，则 class id 就是 trace id（L329 用到）
        classes_id = None  # type: list | None
        model_input_size = (640, 640)
        if model_json_path:
            # 复制原 model.json 到临时根目录，供最终输出使用
            shutil.copy2(str(model_json_path), str(labels_dir.parent / "model.json"))
            with open(model_json_path, encoding="utf-8") as f:
                mj = json.load(f)
            classes = mj.get("classes", classes)
            classes_id = mj.get("classes_id")  # 有则用，无则 None
            input_size = mj.get("input_size", model_input_size)
            if isinstance(input_size, list):
                model_input_size = (int(input_size[0]), int(input_size[1]))
            else:
                model_input_size = (int(input_size), int(input_size))

        print("[INFO] 加载 ONNX: {} | classes: {} | input: {}".format(
            onnx_path.name, len(classes), model_input_size))

        # 2. 加载 ONNX 模型
        sess = ort.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # 3. 遍历帧推理
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        ann_meta = {}  # frame_idx -> list of annotations

        for ff in frame_files:
            digits = "".join(c for c in ff.stem.split("_")[1] if c.isdigit())
            frame_idx = int(digits) if digits else 0

            # 读取图片
            img = cv2.imread(str(ff))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]

            # letterbox resize to model input
            resized, ratio, pad = _letterbox(img, model_input_size)

            # normalize [0,1]
            blob = resized.astype(np.float32) / 255.0
            # HWC -> NCHW
            blob = np.transpose(blob, (2, 0, 1))[None, ...]

            # inference
            outputs = sess.run([output_name], {input_name: blob})[0]  # (1, 11, 8400)

            # 后处理：YOLO 输出格式
            boxes = _yolo_postprocess(outputs, orig_w, orig_h, model_input_size,
                                      ratio, pad, conf_threshold, iou_threshold)

            ann_list = []
            for box in boxes:
                x, y, w, h = box["bbox"]
                # 转原生 float，避免 numpy float32 无法 JSON 序列化
                x, y, w, h = float(x), float(y), float(w), float(h)
                # segmentation: 4 corners
                seg = [x, y, x + w, y, x + w, y + h, x, y + h]
                class_id = int(box["class_id"])
                # 如果 model.json 的类里面有 classes_id，则 class id 就是 trace id；否则按 1000 的倍数分配
                if classes_id and class_id < len(classes_id):
                    trace_id = int(classes_id[class_id])
                else:
                    trace_id = 1000 * (class_id + 1)
                ann = {
                    "bbox": [x, y, w, h],
                    "track_id": trace_id,
                    "segmentation": [seg],
                    "category": classes[class_id],
                    "confidence": float(box["confidence"]),
                    "category_id": 0,  # app 约定：类别由 category 字段存储，category_id 恒为 0
                    "trace_id_list": [trace_id],
                }
                ann_list.append(ann)

            # 写入 label
            with open(labels_dir / f"frame_{frame_idx:06d}.json", "w", encoding="utf-8") as f:
                json.dump(ann_list, f, ensure_ascii=False)
            ann_meta[frame_idx] = ann_list

            if frame_idx % 30 == 0:
                print("[INFO] 推理进度: frame_{:06d} (total {})".format(
                    frame_idx, len(frame_files)))

        print("[INFO] ONNX 推理完成: {}/{} 帧有标注".format(
            sum(1 for v in ann_meta.values() if v), len(frame_files)))
        return ann_meta

    finally:
        shutil.rmtree(temp_model_dir, ignore_errors=True)


def _letterbox(img, target_size):
    """Letterbox resize，保持宽高比，填充灰色边，返回(resized, ratio, pad_top_left)

    Args:
        img: BGR 图片 (H, W, 3)
        target_size: (target_w, target_h)

    Returns:
        canvas: 填充后的图片 (target_h, target_w, 3)
        scale: 缩放比例
        pad_top_left: (pad_top, pad_left)
    """
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_top = (th - nh) // 2
    pad_left = (tw - nw) // 2
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized
    return canvas, scale, (pad_top, pad_left)


def _yolo_postprocess(output, orig_w, orig_h, input_size, ratio, pad, conf_thresh, iou_thresh):
    """YOLO (1, 11, 8400) -> [{bbox, class_id, confidence}]"""
    # output: (1, num_classes+4, num_anchors)
    pred = output[0]  # (11, 8400)
    # 前 num_classes 列是类别置信度，最后4列是 (cx, cy, w, h) 在 input_size 坐标
    num_classes = pred.shape[0] - 4
    boxes_all = []

    for i in range(pred.shape[1]):  # 8400 anchors
        cx, cy, bw, bh = pred[0, i], pred[1, i], pred[2, i], pred[3, i]
        scores = pred[4:, i]  # num_classes
        max_score = float(scores.max())
        if max_score < conf_thresh:
            continue
        class_id = int(scores.argmax())

        # 转回原图坐标（先去除 pad，再除以 scale）
        pad_top, pad_left = pad
        lx = (cx - bw / 2 - pad_left) / ratio
        ly = (cy - bh / 2 - pad_top) / ratio
        lw = bw / ratio
        lh = bh / ratio

        # clamp to image bounds
        lx = max(0, lx)
        ly = max(0, ly)
        lw = min(orig_w - lx, lw)
        lh = min(orig_h - ly, lh)

        if lw <= 0 or lh <= 0:
            continue

        boxes_all.append({
            "bbox": [lx, ly, lw, lh],
            "class_id": class_id,
            "confidence": max_score,
        })

    # NMS
    if not boxes_all:
        return []
    boxes_all.sort(key=lambda x: x["confidence"], reverse=True)
    keep = []
    for b in boxes_all:
        overlap = False
        for k in keep:
            iou = _box_iou(b["bbox"], k["bbox"])
            if iou > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(b)
    return keep


def _box_iou(a, b):
    """计算两个 [x,y,w,h] bbox 的 IoU"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-9)


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

    # 复制原 model.json 到输出目录
    model_json_src = frame_src_dir.parent / "model.json"
    if model_json_src.exists():
        shutil.copy2(str(model_json_src), str(output_dir / "model.json"))

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
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Path:
    """
    从视频文件和模型压缩包生成 temp_data 格式 COCO 数据集。

    Args:
        video_paths: 视频文件路径列表（支持 mp4/avi/mov 等 cv2 支持的格式）
        model_archive: 模型压缩包路径（.zip 或 .rar，内含 best.onnx 和 model.json）
        output_dir: 输出目录
        target_w, target_h: 目标分辨率（默认 2012x1518）
        frame_skip: 跳帧采样间隔（默认 1=全取）
        conf_threshold: ONNX 推理置信度阈值（默认 0.25）
        iou_threshold: NMS IoU 阈值（默认 0.45）

    Returns:
        输出目录 Path
    """
    output_dir = Path(output_dir)
    temp_root = Path(tempfile.mkdtemp(prefix="cocomaker_"))

    try:
        print("[INFO] 目标尺寸: {}x{}".format(target_w, target_h))
        print("[INFO] 视频数量: {}".format(len(video_paths)))
        print("[INFO] 模型文件: {}".format(model_archive))
        print("[INFO] 跳帧采样: 每 {} 帧取 1 帧".format(frame_skip))
        print("[INFO] 推理阈值: conf={}, iou={}".format(conf_threshold, iou_threshold))

        # 1. 从视频提取帧（输出到 temp_root/frames/）
        frames_dir = temp_root / "frames"
        frame_meta = _extract_video_frames(
            video_paths, frames_dir, target_w, target_h, frame_skip
        )
        print(f"[INFO] 视频帧提取完成: {len(frame_meta)} 帧")

        # 2. ONNX 推理生成标注（输出到 temp_root/labels/）
        labels_dir = temp_root / "labels"
        label_meta = _run_onnx_inference(
            Path(model_archive),
            frames_dir,
            labels_dir,
            target_w,
            target_h,
            conf_threshold,
            iou_threshold,
        )

        # 3. 合并视频帧和模型标注
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
