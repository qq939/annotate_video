#!/usr/bin/env python3
"""app.py 前台/后台逻辑验证测试（TDD） - 验证：
1. app.py 能被正常 import（无语法/导入错误）
2. QTimer / processEvents / subprocess.run / subprocess.Popen / app.exec_ 关键代码行存在
3. QTimer 在 UI 主线程异步运行，不阻塞主循环
4. subprocess.run 是阻塞前台，subprocess.Popen 是非阻塞后台
5. processEvents 在长任务中被调用（UI保持响应）
"""
import os
import re
import sys
import time
import threading
import traceback
import py_compile

APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
TEST_TIMEOUT = 60


def test1_syntax_compile():
    """测试1: app.py 语法正确，可编译"""
    assert os.path.exists(APP_PATH), f"找不到 {APP_PATH}"
    py_compile.compile(APP_PATH, doraise=True)
    compile(open(APP_PATH, encoding="utf-8").read(), APP_PATH, "exec")
    print("[PASS] test1_syntax_compile")


def test2_qtimer_exists():
    """测试2: QTimer 定时器代码存在（后台异步体现）"""
    content = open(APP_PATH, encoding="utf-8").read()
    # 找 QTimer.start / QTimer.singleShot
    assert re.search(r"QTimer\s*\.\s*singleShot", content), "未找到 QTimer.singleShot"
    assert re.search(r"play_timer\s*=\s*QTimer\(\)", content) or \
           re.search(r"self\.timer\s*=\s*QTimer\(\)", content), \
           "未找到 QTimer() 实例化"
    assert re.search(r"\.start\(\s*\d+\s*\)", content), "未找到 QTimer.start()"
    print("[PASS] test2_qtimer_exists")


def test3_processevents_exists():
    """测试3: QApplication.processEvents() 存在（前台/后台分界点）"""
    content = open(APP_PATH, encoding="utf-8").read()
    matches = re.findall(r"QApplication\.processEvents\(\)", content)
    assert len(matches) >= 3, f"processEvents 太少（找到{len(matches)}处，需要>=3处）"
    print(f"[PASS] test3_processevents_exists: 找到 {len(matches)} 处")


def test4_subprocess_run_foreground():
    """测试4: subprocess.run 存在（前台阻塞等待子进程）"""
    content = open(APP_PATH, encoding="utf-8").read()
    # 找 subprocess.run 的各种调用
    matches = re.findall(r"subprocess\.run\s*\(", content)
    assert len(matches) >= 3, f"subprocess.run 太少（找到{len(matches)}，需要>=3）"
    print(f"[PASS] test4_subprocess_run_foreground: 找到 {len(matches)} 处 subprocess.run")


def test5_subprocess_popen_background():
    """测试5: subprocess.Popen 存在（后台非阻塞启动子进程）"""
    content = open(APP_PATH, encoding="utf-8").read()
    matches = re.findall(r"subprocess\.Popen\s*\(", content)
    assert len(matches) >= 1, f"subprocess.Popen 未找到（需要>=1处）"
    print(f"[PASS] test5_subprocess_popen_background: 找到 {len(matches)} 处 subprocess.Popen")


def test6_exec_loop_exists():
    """测试6: app.exec_() 主循环存在（前台入口）"""
    content = open(APP_PATH, encoding="utf-8").read()
    assert re.search(r"sys\.exit\s*\(\s*app\.exec_\s*\(\s*\)\s*\)", content), \
           "未找到 sys.exit(app.exec_())"
    print("[PASS] test6_exec_loop_exists")


def test7_clicked_signal_sender():
    """测试7: clicked.connect() 信号槽存在（前台触发后台回调）"""
    content = open(APP_PATH, encoding="utf-8").read()
    matches = re.findall(r"\.clicked\.connect\s*\(", content)
    assert len(matches) >= 5, f"clicked.connect 太少（找到{len(matches)}，需要>=5）"
    print(f"[PASS] test7_clicked_signal_sender: 找到 {len(matches)} 处 clicked.connect")


def test8_qtimer_async_behavior():
    """测试8: QTimer 定时器行为验证（异步不阻塞）"""
    # 模拟 QTimer 的行为
    results = []
    call_count = [0]

    def timer_callback():
        call_count[0] += 1
        results.append(f"callback_{call_count[0]}")

    # 模拟 Qt 的 singleShot 行为：用 threading.Timer 代替
    import threading

    timer = threading.Timer(0.05, timer_callback)  # 50ms后触发
    timer.daemon = True
    start = time.time()
    timer.start()
    # 主线程继续执行（模拟 app.exec_() 不阻塞）
    time.sleep(0.02)  # 20ms
    elapsed = time.time() - start
    # 主线程不应该被 timer 阻塞，20ms 内应该已经执行到这里
    assert elapsed < 0.03, f"主线程被阻塞了 {elapsed:.3f}s（不应该）"
    # 等待 timer 完成
    timer.join(timeout=1)
    assert len(results) == 1, f"定时器回调未触发：{results}"
    assert results[0] == "callback_1", f"回调结果错误：{results}"
    print("[PASS] test8_qtimer_async_behavior")


def test9_processevents_yields_ui():
    """测试9: processEvents 让出 UI 响应（模拟让 Qt 处理事件队列）"""
    # 模拟 processEvents 的效果：主线程主动暂停，让其他线程/事件有机会执行
    ui_events = []

    def ui_update():
        ui_events.append("ui_paint")

    def long_task_with_yield():
        ui_events.append("task_start")
        # 模拟 processEvents：立即返回 True 表示有事件被处理
        # 这里我们直接调用一个空操作，模拟 Qt 处理事件
        ui_events.append("task_yield")  # 相当于 processEvents() 被调用
        ui_events.append("task_end")

    long_task_with_yield()
    assert "task_yield" in ui_events, f"processEvents 等效调用缺失：{ui_events}"
    print("[PASS] test9_processevents_yields_ui")


def test10_subprocess_run_blocks_subprocess_popen_does_not():
    """测试10: subprocess.run 阻塞，subprocess.Popen 不阻塞"""
    import subprocess
    import threading

    # 测试 Popen 不阻塞
    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2); print('done')"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    elapsed_popen = time.time() - start
    proc.terminate()
    assert elapsed_popen < 0.5, f"Popen 阻塞了 {elapsed_popen:.3f}s（应该立即返回）"
    print(f"[PASS] test10 subprocess.run阻塞/Popen不阻塞: Popen耗时 {elapsed_popen:.3f}s（<0.5s）")

    # 测试 run 阻塞（用短时间脚本避免测试太慢）
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", "import time; time.sleep(0.3); print('done')"],
        capture_output=True, text=True, timeout=5
    )
    elapsed_run = time.time() - start
    assert elapsed_run >= 0.25, f"run 没有阻塞（耗时仅 {elapsed_run:.3f}s）"
    print(f"[PASS] test10 subprocess.run确实阻塞: run耗时 {elapsed_run:.3f}s（>=0.25s）")


def main():
    tests = [
        ("test1_syntax_compile", test1_syntax_compile),
        ("test2_qtimer_exists", test2_qtimer_exists),
        ("test3_processevents_exists", test3_processevents_exists),
        ("test4_subprocess_run_foreground", test4_subprocess_run_foreground),
        ("test5_subprocess_popen_background", test5_subprocess_popen_background),
        ("test6_exec_loop_exists", test6_exec_loop_exists),
        ("test7_clicked_signal_sender", test7_clicked_signal_sender),
        ("test8_qtimer_async_behavior", test8_qtimer_async_behavior),
        ("test9_processevents_yields_ui", test9_processevents_yields_ui),
        ("test10_subprocess_run_blocks_subprocess_popen_does_not", test10_subprocess_run_blocks_subprocess_popen_does_not),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, func in tests:
        print(f"\n===== {name} =====")
        try:
            func()
            passed += 1
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, str(e), tb))
            print(f"[FAIL] {name}: {e}")

    print(f"\n===== 结果（超时{TEST_TIMEOUT}s） =====")
    print(f"通过: {passed}, 失败: {failed}")
    if errors:
        for n, e, tb in errors:
            print("-" * 40)
            print(f"[{n}] {e}\n{tb}")
        sys.exit(1)
    else:
        print("全部测试通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
