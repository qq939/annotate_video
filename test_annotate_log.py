#!/usr/bin/env python3
"""测试：执行标注(run_annotate) 增加日志 + 修复「同一张mask按轮廓被重复写成N个目标」

背景（实测证据）：temp_data/annotations.json 里 frame 340 有 101 个标注，其中 10 个
area 完全相同(114585) 但 bbox 小到 1x2 —— bbox 与 area/polygon 不可能同源。
根因：run_annotate 第一个循环 `for contour in contours: current_masks.append(mask_binary)`
把「一张 mask」按轮廓个数重复追加；第二个循环又对每个条目重新 findContours，
用 mask 的轮廓面积 + 条目里的 bbox 组标注 → 一张 mask 被写成 C*C 个标注。
勾选 IoU 时这些重复 mask 会被 merge_masks_in_frame 合并(IoU=1)掩盖掉，
取消勾选后不再合并 → 重复标注全部暴露，看起来就是「手套被轮胎 merge 了」。

覆盖：
  1. extract_mask_targets：一个 mask 行 = 一个目标（多轮廓只取并集 bbox），不重复追加
  2. mask_binary_to_annotations：bbox / area / polygon 全部取自同一个轮廓（自洽）
  3. 组合后不会出现 C*C 重复标注
  4. 源码回归：run_annotate 用了这两个函数；run_annotate 会写 IoU 开关日志
  5. annotate_log 真写到「执行标注.log」

超时：进程级看门狗
"""
import os
import sys
import threading
import time
from pathlib import Path

TIMEOUT_SEC = 90
APP_FILE = "app.py"
LOG_FILE = "执行标注.log"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_APP = None


def _watchdog(limit=TIMEOUT_SEC):
    time.sleep(limit)
    print("\n[TIMEOUT] 测试超时(%ss)" % limit, flush=True)
    os._exit(1)


def _app():
    global _APP
    if _APP is None:
        sys.path.insert(0, str(Path(APP_FILE).parent.resolve()))
        import app as app_mod
        _APP = app_mod
    return _APP


def _mask_with_blobs(boxes, h=100, w=100):
    """在 (h,w) 的画布上画出若干互不相连的方块，返回 uint8 mask"""
    import numpy as np
    m = np.zeros((h, w), np.uint8)
    for (x1, y1, x2, y2) in boxes:
        m[y1:y2, x1:x2] = 1
    return m


def _func_body(src, name):
    """取出 def name(...) 的函数体（支持方法/嵌套函数）"""
    for pat in ("def %s(self" % name, "def %s(" % name):
        start = src.find(pat)
        if start != -1:
            break
    else:
        raise AssertionError("未找到函数 %s" % name)

    def _indent(idx):
        return idx - (src.rfind("\n", 0, idx) + 1)

    base = _indent(start)
    pos = start + 10
    while True:
        nxt = src.find("\ndef ", pos)
        if nxt == -1:
            nxt = len(src)
            break
        if _indent(nxt + 1) <= base:
            break
        pos = nxt + 1
    return src[start:nxt]


# ---------------- 1. 一个 mask 行 = 一个目标 ----------------

def test_extract_mask_targets_one_target_per_row():
    """一张 mask（含3个分散方块）只产出 1 个目标，且记下轮廓数=3"""
    app = _app()
    mask = _mask_with_blobs([(10, 10, 20, 20), (40, 40, 60, 60), (70, 70, 95, 95)])
    targets = app.extract_mask_targets([mask])
    assert len(targets) == 1, "一个mask行只能算一个目标, 实际=%d" % len(targets)
    m, bbox, n_contours = targets[0]
    assert n_contours == 3, "应记录到3个轮廓, 实际=%d" % n_contours
    assert bbox == [10.0, 10.0, 84.0, 84.0], "bbox应为全部轮廓的并集, 实际=%s" % (bbox,)


def test_extract_mask_targets_skips_empty_rows():
    """空 mask 行不产出目标"""
    import numpy as np
    app = _app()
    empty = np.zeros((50, 50), np.uint8)
    assert app.extract_mask_targets([empty]) == [], "空mask不应产出目标"


# ---------------- 2. bbox / area / polygon 同源 ----------------

def test_annotations_bbox_area_polygon_same_contour():
    """每个标注的 bbox 必须由它自己的 polygon/area 得来（不能出现 bbox 1x2 而 area 上万）"""
    app = _app()
    mask = _mask_with_blobs([(10, 10, 30, 30), (60, 60, 80, 80)])
    anns = app.mask_binary_to_annotations(mask, track_id=1000, image_id=0,
                                          first_ann_id=0, confidence=0.9)
    assert len(anns) == 2, "两个方块应产出两条标注, 实际=%d" % len(anns)
    for a in anns:
        xs = a['segmentation'][0][0::2]
        ys = a['segmentation'][0][1::2]
        pw = max(xs) - min(xs)
        ph = max(ys) - min(ys)
        assert abs(a['bbox'][2] - pw) <= 1 and abs(a['bbox'][3] - ph) <= 1, \
            "bbox 必须贴合自己的 polygon: bbox=%s polygon宽高=(%s,%s)" % (a['bbox'], pw, ph)
        assert abs(a['area'] - pw * ph) <= pw * ph * 0.5, \
            "area 必须与该轮廓量级一致: area=%s polygon面积~%s" % (a['area'], pw * ph)
        assert a['track_id'] == 1000 and a['category_id'] == 1000


def test_annotations_skip_zero_area():
    """退化轮廓（面积0）不应产出标注"""
    import numpy as np
    app = _app()
    line = np.zeros((50, 50), np.uint8)
    line[10:13, 10:13] = 1  # 3x3 会退化；用 2 像素线更稳
    line[20, 20:30] = 1     # 1像素宽线 -> contourArea 为 0
    anns = app.mask_binary_to_annotations(line, track_id=1, image_id=0,
                                          first_ann_id=0, confidence=1.0)
    assert all(a['area'] > 0 for a in anns), "不应出现 area<=0 的标注"


# ---------------- 3. 不再出现 C*C 重复 ----------------

def test_no_c_squared_duplication():
    """一张含3个轮廓的mask：修复前会产出 3*3=9 条，修复后只能 3 条"""
    app = _app()
    mask = _mask_with_blobs([(10, 10, 20, 20), (40, 40, 60, 60), (70, 70, 95, 95)])
    targets = app.extract_mask_targets([mask])
    total = []
    for i, (m, _bbox, _n) in enumerate(targets):
        total += app.mask_binary_to_annotations(m, track_id=1000, image_id=0,
                                                first_ann_id=len(total), confidence=1.0)
    assert len(total) == 3, "3个轮廓应产出3条标注(不是9条), 实际=%d" % len(total)
    keys = {(tuple(a['bbox']), a['track_id']) for a in total}
    assert len(keys) == len(total), "不应出现重复标注"


# ---------------- 4. 源码回归 ----------------

def test_run_annotate_uses_helpers_and_logs_iou():
    """run_annotate 必须用新函数提目标，并把 IoU 开关状态写进日志"""
    src = Path(APP_FILE).read_text(encoding="utf-8")
    assert "def extract_mask_targets(" in src, "应有 extract_mask_targets"
    assert "def mask_binary_to_annotations(" in src, "应有 mask_binary_to_annotations"
    assert "def annotate_log(" in src, "应有 annotate_log"
    body = _func_body(src, "run_annotate")
    assert "extract_mask_targets(" in body, "run_annotate 应用 extract_mask_targets 提目标"
    assert "mask_binary_to_annotations(" in body, "run_annotate 应用 mask_binary_to_annotations 组标注"
    assert "annotate_log(" in body, "run_annotate 应写日志"
    assert "启用IoU=" in body or "iou_enabled" in body, "日志应包含IoU开关状态"
    # 旧的「按轮廓重复追加 mask」写法必须消失
    assert "current_masks.append(mask_binary)" not in body, "旧的重复追加写法应被移除"


def test_annotate_log_file():
    """annotate_log 同时输出到控制台和「执行标注.log」"""
    app = _app()
    log_path = Path(LOG_FILE)
    if log_path.exists():
        log_path.unlink()
    app.annotate_log("__test_marker_执行标注日志__")
    assert log_path.exists(), "应创建 %s" % LOG_FILE
    content = log_path.read_text(encoding="utf-8")
    assert "__test_marker_执行标注日志__" in content, "日志内容应写入文件"
    log_path.unlink()


def test_run_annotate_iou_param_order():
    """回归：_iou_params 返回 (帧内合并IoU, 跨帧跟踪IoU, 开关)，run_annotate 不能解包错序。

    实测事故：曾写成 `iou_val, merge_iou_val, iou_enabled = self._iou_params()`，
    导致「同帧合并」误用了前后IoU(0.02)、「跨帧跟踪」误用了帧IoU(0.5)。
    """
    src = Path(APP_FILE).read_text(encoding="utf-8")
    body = _func_body(src, "run_annotate")
    assert "merge_iou_val, iou_val, iou_enabled = self._iou_params()" in body, \
        "必须按(帧内合并,跨帧跟踪,开关)顺序解包，否则同帧合并会误用前后IoU"
    assert "iou_val, merge_iou_val, iou_enabled = self._iou_params()" not in body, \
        "禁止把 merge / track 两个IoU解包反"
    assert "merge_masks_in_frame(current_masks, current_bboxes, merge_iou_val)" in body, \
        "同帧合并必须用帧IoU(merge_iou_val)"
    assert "TrackManager(iou_threshold=iou_val, merge_enabled=iou_enabled)" in body, \
        "跨帧跟踪必须用前后IoU(iou_val)"


def test_frame_log_has_mask_row_detail():
    """日志要能看出「SAM3 原始返回的每一行 mask 有多大」，用于区分是模型给的还是代码合并的"""
    src = Path(APP_FILE).read_text(encoding="utf-8")
    body = _func_body(src, "run_annotate")
    assert "mask行明细" in body, "每帧日志应记录原始mask每行的像素数/轮廓数"


def test_sam3_merge_knobs_wired():
    """SAM3 视频语义预测器要把「可控制误检/合并」的阈值显式传进去（文件顶部 global 参数调）"""
    src = Path(APP_FILE).read_text(encoding="utf-8")
    for name in ("SAM3_SCORE_THRESHOLD_DETECTION", "SAM3_DET_NMS_THRESH", "SAM3_NEW_DET_THRESH"):
        assert name in src, "应有 global 参数 %s" % name
    body = _func_body(src, "run_annotate")
    assert "score_threshold_detection=SAM3_SCORE_THRESHOLD_DETECTION" in body, \
        "构造 SAM3VideoSemanticPredictor 时应传入检测置信度门限"
    assert "det_nms_thresh=SAM3_DET_NMS_THRESH" in body, "应传入检测NMS门限"
    assert "new_det_thresh=SAM3_NEW_DET_THRESH" in body, "应传入新目标门限"
    assert "检测门限" in body, "日志应记录检测门限"


def test_sam3_default_thresholds_unchanged():
    """默认值保持 ultralytics 的 0.5/0.0/0.0，不改变现有行为"""
    app = _app()
    assert app.SAM3_SCORE_THRESHOLD_DETECTION == 0.5
    assert app.SAM3_DET_NMS_THRESH == 0.0
    assert app.SAM3_NEW_DET_THRESH == 0.0


def test_syntax_check():
    import py_compile
    py_compile.compile(APP_FILE, doraise=True)


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    tests = [
        ("一mask行=一目标", test_extract_mask_targets_one_target_per_row),
        ("空mask跳过", test_extract_mask_targets_skips_empty_rows),
        ("bbox/area/polygon同源", test_annotations_bbox_area_polygon_same_contour),
        ("零面积轮廓跳过", test_annotations_skip_zero_area),
        ("无C*C重复标注", test_no_c_squared_duplication),
        ("run_annotate用新函数+记日志", test_run_annotate_uses_helpers_and_logs_iou),
        ("IoU参数解包顺序不能反", test_run_annotate_iou_param_order),
        ("日志含原始mask行明细", test_frame_log_has_mask_row_detail),
        ("SAM3合并/误检阈值已接上", test_sam3_merge_knobs_wired),
        ("SAM3阈值默认值不变", test_sam3_default_thresholds_unchanged),
        ("执行标注.log写入", test_annotate_log_file),
        ("语法检查", test_syntax_check),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("[PASS] %s" % name)
        except Exception as e:
            failed += 1
            print("[FAIL] %s - %s: %s" % (name, type(e).__name__, e))
    print("\n总计: %d 通过, %d 失败" % (passed, failed))
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
