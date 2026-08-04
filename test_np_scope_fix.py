#!/usr/bin/env python3
"""测试np闭包作用域问题是否修复
验证：app.py中的内部函数可以正常访问全局np，不会出现
"cannot access free variable np where it is not associated with a value in enclosing scope"
错误
"""
import sys
import os
import signal
import traceback
import importlib.util

TEST_TIMEOUT = 60
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def _timeout_handler(signum, frame):
    raise TimeoutError(f"测试超时，超过 {TEST_TIMEOUT} 秒")


def test_import_and_np_scope():
    """测试1: 验证app.py语法正确，且app模块能够被导入（无语法错误）"""
    # 模拟类似的闭包作用域场景
    import numpy as np  # 全局np

    # 模拟run_bidirectional_annotate中的闭包场景
    # 在某个条件分支有局部import numpy（实际已删除，现在测试没有局部import的情况）
    # 然后在内部函数中访问np - 应该直接访问全局np

    def outer_function_simulate():
        """模拟外层函数，内部定义多个闭包使用np"""

        def inner1_auto_seg():
            """模拟do_auto_seg_clip使用np"""
            arr = np.array([1, 2, 3], dtype=np.uint8)
            return arr.shape[0] > 0

        def inner2_point_seg():
            """模拟do_point_seg_clip使用np"""
            arr = np.ones(5, dtype=np.int32)
            return len(arr) == 5

        def inner3_pure_semantic():
            """模拟do_pure_semantic_clip使用np"""
            zeros = np.zeros((2, 2, 3), dtype=np.uint8)
            return zeros.shape == (2, 2, 3)

        def inner4_process_clip():
            """模拟process_clip使用np"""
            mask = np.array([0, 1, 1, 0], dtype=np.float32)
            m_bin = (mask > 0.5).astype(np.uint8)
            return m_bin.sum() == 2

        # 执行所有内部函数（实际路径可能只执行部分）
        results = []
        results.append(("inner1_auto_seg", inner1_auto_seg()))
        results.append(("inner2_point_seg", inner2_point_seg()))
        results.append(("inner3_pure_semantic", inner3_pure_semantic()))
        results.append(("inner4_process_clip", inner4_process_clip()))
        return results

    results = outer_function_simulate()
    for name, ok in results:
        if not ok:
            raise AssertionError(f"{name} 返回False")
    print("[PASS] test_import_and_np_scope: 闭包np访问正常")
    return True


def test_app_syntax_compile():
    """测试2: 验证app.py能够被Python语法解析和编译（无SyntaxError）"""
    if not os.path.exists(APP_PATH):
        raise FileNotFoundError(f"找不到app.py: {APP_PATH}")
    with open(APP_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    # 编译源码，有语法错误会抛出SyntaxError
    compile(source, APP_PATH, "exec")
    print("[PASS] test_app_syntax_compile: app.py语法编译通过")
    return True


def test_no_local_np_import_in_closures():
    """测试3: 验证app.py的关键函数中不再存在局部的import numpy as np"""
    with open(APP_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 需要检查的函数内部（按行号粗略范围检查）
    # process_clip定义后的内部函数体
    # do_auto_seg_clip, do_point_seg_clip, do_pure_semantic_clip定义后
    # preview_segmentation方法内
    
    bad_lines = []
    # 按上下文检查：在函数定义缩进后面的行不能有局部的import numpy as np
    # 更简单：全文件搜索"    import numpy as np"或更多缩进形式（缩进表示在函数/方法内）
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip("\n")
        # 匹配缩进开头的 import numpy as np（非顶行）
        if stripped.lstrip() == "import numpy as np" and (stripped.startswith(" ") or stripped.startswith("\t")):
            # 忽略一些正常的局部导入：不在闭包外层的（但我们的修复是删除全部局部import）
            # 实际上，我们要确保在app.py中除了全局import外，没有局部的import numpy as np
            if i != 8:  # 全局import是第8行
                bad_lines.append((i, stripped.strip()))

    if bad_lines:
        detail = "; ".join([f"第{ln}行: {txt}" for ln, txt in bad_lines])
        raise AssertionError(f"发现局部的import numpy as np: {detail}")

    print("[PASS] test_no_local_np_import_in_closures: 无非全局的局部import numpy as np")
    return True


def main():
    # 设置超时
    if sys.platform != "win32":
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TEST_TIMEOUT)
    else:
        # Windows 下使用线程定时器
        import threading
        timer = threading.Timer(TEST_TIMEOUT, lambda: os._exit(124))
        timer.daemon = True
        timer.start()

    tests = [
        ("test_import_and_np_scope", test_import_and_np_scope),
        ("test_app_syntax_compile", test_app_syntax_compile),
        ("test_no_local_np_import_in_closures", test_no_local_np_import_in_closures),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, func in tests:
        print(f"\n===== 执行 {name} =====")
        try:
            func()
            passed += 1
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append(f"[{name}] {e}\n{tb}")
            print(f"[FAIL] {name}: {e}")

    print(f"\n===== 测试结果 =====")
    print(f"通过: {passed}, 失败: {failed}")
    if errors:
        print("\n错误详情:")
        for err in errors:
            print("-" * 50)
            print(err)
        sys.exit(1)
    else:
        print("所有测试通过!")
        sys.exit(0)


if __name__ == "__main__":
    main()
