#!/usr/bin/env python3
"""测试：新增「启用IoU」勾选框，可一键禁用帧内IoU与帧间IoU

背景：刀片/手套在轮胎内部时，TrackManager 只按 IoU>=0.02 复用 track_id，导致
同一帧多个 mask 共用轮胎的 tid（类别名全变成轮胎）。用户要求不改合并算法，
而是加一个勾选框：勾选=启用帧内IoU+帧间IoU（现状），不勾选=全部禁用。

覆盖：
  1. TrackManager(merge_enabled=False)：每个 mask 独立编号，不做跨帧合并
  2. TrackManager 默认行为不变（同一 mask 跨帧仍是同一 id）
  3. merge_masks_in_frame 阈值为0时不合并（禁用帧内IoU）
  4. _iou_params：未勾选返回 (0.0, 0.0, False)；勾选返回输入框的值+True
  5. 源码回归：勾选框已加到面板；三个流程都用 _iou_params；两处去重受开关控制

超时：进程级看门狗
"""
import os
import sys
import threading
import time
import types
from pathlib import Path

TIMEOUT_SEC = 90
APP_FILE = "app.py"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_APP = None
_QAPP = None   # 保持QApplication引用，避免被回收


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


def _av():
    """annotate_video 模块（TrackManager / merge_masks_in_frame 所在）"""
    _app()
    import annotate_video
    return annotate_video


def _mask(h=20, w=20, fill=1):
    import numpy as np
    return np.full((h, w), fill, np.uint8)


def _func_body(src, name):
    """取出 def name(...) 的函数体。
    既支持带 self 的方法，也支持无 self 的嵌套函数（如 do_point_seg_clip）。
    结束位置=下一个缩进不大于当前的 def 行，或文件末尾。"""
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


# ---------------- 禁用的语义 ----------------

def test_trackmanager_disable_merging():
    """禁用帧间IoU后：每帧每个mask独立编号，不跨帧合并"""
    av = _av()
    tm = av.TrackManager(iou_threshold=0.5, merge_enabled=False)
    m, b = [_mask()], [[0, 0, 20, 20]]
    ids_frame0 = tm.update(m, b, 0)
    ids_frame0b = tm.update(m, b, 0)     # 同一帧再来一个相同的mask
    ids_frame1 = tm.update(m, b, 1)      # 下一帧同样的mask
    assert len(set(ids_frame0 + ids_frame0b + ids_frame1)) == 3, \
        "禁用后每个mask都要独立编号, 实际=%s" % (ids_frame0, ids_frame0b, ids_frame1)


def test_trackmanager_keeps_id_when_enabled():
    """默认（启用帧间IoU）行为不变：同一mask跨帧仍是同一id"""
    av = _av()
    tm = av.TrackManager(iou_threshold=0.5)
    m, b = [_mask()], [[0, 0, 20, 20]]
    first = tm.update(m, b, 0)
    second = tm.update(m, b, 1)
    assert first == second, "启用时应保持同一track_id, 实际=%s vs %s" % (first, second)


def test_merge_masks_disabled_by_zero_threshold():
    """帧内IoU阈值为0时不做同帧合并"""
    av = _av()
    masks = [_mask(), _mask()]
    bboxes = [[0, 0, 20, 20], [0, 0, 20, 20]]
    out_masks, out_bboxes = av.merge_masks_in_frame(masks, bboxes, 0)
    assert len(out_masks) == 2, "阈值为0时不应合并, 实际=%d" % len(out_masks)
    merged_masks, _ = av.merge_masks_in_frame(masks, bboxes, 0.5)
    assert len(merged_masks) == 1, "阈值0.5时重叠mask应合并"


def test_iou_params_reflects_checkbox():
    """_iou_params：不勾选→禁用(1.0,1.0,False)；勾选→输入框的值+True

    注意禁用值是 1.0 而不是 0.0：代码里都是 `iou >= 阈值` 这种判断（TrackManager.update），
    给 0 会「永远成立」= 反而全合并；给 1 才永不成立 = 真禁用。
    """
    app = _app()
    def fake(checked):
        return types.SimpleNamespace(
            merge_iou_input=types.SimpleNamespace(text=lambda: "0.5"),
            iou_input=types.SimpleNamespace(text=lambda: "0.02"),
            iou_enable_check=types.SimpleNamespace(isChecked=lambda: checked))
    off = app.UnifiedPanel._iou_params(fake(False))
    on = app.UnifiedPanel._iou_params(fake(True))
    assert off == (1.0, 1.0, False), "不勾选应禁用帧内/帧间IoU(值=1), 实际=%s" % (off,)
    assert on == (0.5, 0.02, True), "勾选应使用输入框的值, 实际=%s" % (on,)


# ---------------- 源码回归 ----------------

def test_checkbox_added_to_panel():
    """面板要有「启用IoU」勾选框，默认勾选"""
    src = Path(APP_FILE).read_text(encoding="utf-8")
    assert "self.iou_enable_check = QCheckBox(" in src, "应新增启用IoU勾选框"
    assert "IoU" in src[src.index("self.iou_enable_check = QCheckBox("):][:80]
    body = _func_body(src, "create_annotate_section")
    assert "addWidget(self.iou_enable_check)" in body, "勾选框要加到标注栏"
    assert "setChecked(True)" in src[src.index("self.iou_enable_check = QCheckBox("):][:200], "默认应勾选"


def test_all_flows_use_iou_params():
    """执行标注/双向提示帧/点分割都要用 _iou_params 决定是否合并"""
    src = Path(APP_FILE).read_text(encoding="utf-8")
    assert "def _iou_params(self)" in src, "应有统一的IoU开关读取"
    body = _func_body(src, "run_annotate")
    assert "_iou_params()" in body, "执行标注应使用开关"
    assert "TrackManager(iou_threshold=iou_val, merge_enabled=iou_enabled)" in body, \
        "TrackManager 应接收开关"
    for fn in ("do_bidirectional_inject", "do_point_seg_clip"):
        b = _func_body(src, fn)
        assert "_iou_params()" in b, "%s 应使用IoU开关" % fn
        assert "merge_enabled=" in b, "%s 应把开关传给TrackManager" % fn


def test_dedup_guarded_by_switch():
    """两处IoU去重在未勾选时不生效"""
    src = Path(APP_FILE).read_text(encoding="utf-8")
    assert "if iou_enabled and iou > overlap_threshold:" in src, "汇总去重应受开关控制"
    assert "if iou_enabled and iou > iou_threshold:" in src, "导出labelme去重应受开关控制"


def test_real_panel_has_checkbox():
    """真面板实测：勾选框存在且默认勾选"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    global _QAPP
    _QAPP = QApplication.instance() or QApplication([])
    app = _app()
    panel = app.UnifiedPanel()
    assert hasattr(panel, "iou_enable_check"), "面板应有启用IoU勾选框"
    assert panel.iou_enable_check.isChecked(), "默认应勾选（保持现有行为）"
    panel.iou_enable_check.setChecked(False)
    assert app.UnifiedPanel._iou_params(panel)[2] is False, "取消勾选后应为禁用"


def test_syntax_check():
    import py_compile
    py_compile.compile(APP_FILE, doraise=True)


def main():
    threading.Thread(target=_watchdog, daemon=True).start()
    tests = [
        ("禁用后每mask独立编号", test_trackmanager_disable_merging),
        ("启用时保持track_id", test_trackmanager_keeps_id_when_enabled),
        ("帧内IoU为0不合并", test_merge_masks_disabled_by_zero_threshold),
        ("_iou_params语义", test_iou_params_reflects_checkbox),
        ("勾选框加到面板", test_checkbox_added_to_panel),
        ("三个流程用开关", test_all_flows_use_iou_params),
        ("去重受开关控制", test_dedup_guarded_by_switch),
        ("真面板勾选框", test_real_panel_has_checkbox),
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
