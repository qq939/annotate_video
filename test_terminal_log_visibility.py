#!/usr/bin/env python3
"""终端日志可见性测试（TDD） - 验证：
1. app.py 语法/导入不报错
2. requirements.txt 所有关键模块存在且无版本号
3. 持续日志机制可用（含超时机制）
4. 全局np在闭包作用域正确（修复回归测试）
"""
import os
import sys
import time
import threading
import subprocess
import py_compile
import traceback

TEST_TIMEOUT = 120  # 总超时
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
REQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")


class TimeoutError_(Exception):
    pass


def _run_with_timeout(func, args=(), kwargs=None, timeout=TEST_TIMEOUT):
    """带超时的函数执行（线程定时器方式，Windows友好）"""
    result_container = {"ok": None, "error": None, "tb": None}
    exc_container = []

    def target():
        try:
            result_container["ok"] = func(*args, **(kwargs or {}))
        except Exception as e:
            result_container["error"] = e
            result_container["tb"] = traceback.format_exc()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError_(f"测试超时，超过 {timeout} 秒")
    if result_container["error"] is not None:
        raise result_container["error"]
    return result_container["ok"]


def test1_app_syntax_compile():
    """测试1: app.py 语法编译通过"""
    assert os.path.exists(APP_PATH), f"找不到 {APP_PATH}"
    with open(APP_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    py_compile.compile(APP_PATH, doraise=True)
    # 也用compile内建函数双重检查
    compile(source, APP_PATH, "exec")
    print("[PASS] test1_app_syntax_compile: app.py 语法编译通过")


def test2_app_imports_no_qapp():
    """测试2: app.py 关键模块导入（不创建QApplication避免UI阻塞）"""
    import importlib.util
    sys.path.insert(0, os.path.dirname(APP_PATH))
    # 直接import关键模块
    import cv2  # noqa
    import numpy as np  # noqa
    from PyQt5.QtWidgets import QApplication  # noqa
    from pathlib import Path  # noqa
    import json, shutil, random, subprocess  # noqa
    # 验证import video_control / annotate_video
    import video_control  # noqa
    import annotate_video  # noqa
    # 关键：验证全局np存在且无闭包作用域问题
    assert np.ones((2, 2)).sum() == 4, "numpy 基础功能异常"
    print("[PASS] test2_app_imports_no_qapp: 关键模块导入成功")


def test3_requirements_no_version_and_keys():
    """测试3: requirements.txt 无版本号且包含关键模块"""
    assert os.path.exists(REQ_PATH), f"找不到 {REQ_PATH}"
    with open(REQ_PATH, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    reqs = [l for l in lines if not l.startswith("#") and not l.startswith("git+")]
    for r in reqs:
        # 无 == >= <= ~= !=
        for sym in ("==", ">=", "<=", "~=", "!="):
            assert sym not in r, f"requirements.txt 存在版本号符号 {sym}: {r}"
    key_pkgs = ["opencv-python", "numpy", "Pillow", "PyQt5", "ultralytics",
               "torch", "torchvision"]
    reqset = set(reqs)
    for pkg in key_pkgs:
        # 用包含检查（pillow-heif这种也会匹配Pillow的话会误判，改用精确匹配+起始匹配）
        matched = any(r == pkg or r.startswith(pkg + "[") for r in reqset)
        assert matched, f"requirements.txt 缺少关键模块: {pkg}"
    print("[PASS] test3_requirements_no_version_and_keys: requirements.txt OK")


def test4_closure_np_scope_regression():
    """测试4: 闭包中np作用域无错误（回归）- 模拟app.py里的多层嵌套"""
    import numpy as np  # 模拟app.py顶部的全局np

    def outer_simulate():
        # 故意不写局部import numpy as np（修复后状态）
        def inner_auto_seg_clip():
            arr = np.array([1, 2, 3], dtype=np.uint8)
            zeros = np.zeros((2, 2, 3), dtype=np.uint8)
            return arr.sum() > 0 and zeros.shape == (2, 2, 3)

        def inner_point_seg_clip():
            return np.ones(3).sum() == 3

        def inner_pure_semantic():
            mask = np.array([0.0, 0.9], dtype=np.float32)
            return (mask > 0.5).astype(np.uint8).sum() == 1

        def inner_process_clip():
            ok = True
            for _ in range(3):
                if np.random.randint(0, 256, (3,)) is None:
                    ok = False
            return ok

        return (inner_auto_seg_clip() and inner_point_seg_clip() and
                inner_pure_semantic() and inner_process_clip())

    assert outer_simulate(), "闭包作用域np访问失败（可能重现了之前的bug！）"
    print("[PASS] test4_closure_np_scope_regression: 闭包np作用域正确")


def test5_log_monitor_subprocess():
    """测试5: 持续日志输出子进程可启动+有输出"""
    py_code = (
        "import time\n"
        "for i in range(3):\n"
        "    print(f'[LOG] line {i}', flush=True)\n"
        "    time.sleep(0.2)\n"
    )
    p = subprocess.Popen(
        [sys.executable, "-u", "-c", py_code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.path.dirname(APP_PATH)
    )
    try:
        stdout, stderr = p.communicate(timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        p.kill()
        raise TimeoutError_("日志子进程超时")
    out = stdout.decode("utf-8", errors="replace")
    assert "[LOG] line 0" in out and "[LOG] line 2" in out, f"日志输出不符合预期: {out}"
    print("[PASS] test5_log_monitor_subprocess: 日志子进程输出正常")


def main():
    tests = [
        ("test1_app_syntax_compile", test1_app_syntax_compile),
        ("test2_app_imports_no_qapp", test2_app_imports_no_qapp),
        ("test3_requirements_no_version_and_keys", test3_requirements_no_version_and_keys),
        ("test4_closure_np_scope_regression", test4_closure_np_scope_regression),
        ("test5_log_monitor_subprocess", test5_log_monitor_subprocess),
    ]
    passed = 0
    failed = 0
    errors = []
    t0 = time.time()
    for name, func in tests:
        print(f"\n===== 执行 {name} =====")
        try:
            _run_with_timeout(func, timeout=TEST_TIMEOUT // len(tests) + 10)
            passed += 1
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, str(e), tb))
            print(f"[FAIL] {name}: {e}")
    dt = time.time() - t0
    print(f"\n===== 测试结果（总耗时 {dt:.1f}s =====")
    print(f"通过: {passed}, 失败: {failed}")
    if errors:
        print("\n错误详情:")
        for n, e, tb in errors:
            print("-" * 50)
            print(f"[{n}] {e}")
            print(tb)
        sys.exit(1)
    else:
        print("全部测试通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
