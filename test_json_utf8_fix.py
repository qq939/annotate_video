#!/usr/bin/env python3
"""JSON UTF-8 修复验证测试（TDD） - 验证：
1. video_viewer.py 所有 open() 都有 encoding='utf-8'
2. app.py 的 json.dump patch 正确（ensure_ascii=False）
3. 语法编译通过
4. 含超时机制
"""
import os, sys, re, py_compile, io, json, traceback, subprocess, threading

BASE = os.path.dirname(os.path.abspath(__file__))
VV_PATH = os.path.join(BASE, "video_viewer.py")
APP_PATH = os.path.join(BASE, "app.py")
TEST_TIMEOUT = 60


def _run(func, timeout=TEST_TIMEOUT):
    r = {"ok": None, "err": None}
    def t():
        try:
            r["ok"] = func()
        except Exception as e:
            r["err"] = e
    th = threading.Thread(target=t, daemon=True)
    th.start()
    th.join(timeout=timeout)
    if th.is_alive():
        raise TimeoutError(f"> {timeout}s")
    if r["err"]:
        raise r["err"]


def test1_videoviewer_all_open_have_encoding():
    """测试1: video_viewer.py 所有 open() 都有 encoding='utf-8'"""
    with open(VV_PATH, encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    bad = []
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        # 找 open(...) 调用的行（可能跨行）
        if re.search(r'\bopen\s*\(', stripped):
            # 简单检查：这一行有 open( 但没有 encoding=
            # 排除已注释的行
            if stripped.lstrip().startswith("#"):
                continue
            if "encoding=" not in stripped and "encoding =" not in stripped:
                bad.append((i, stripped.strip()[:80]))
    assert not bad, f"以下行 open() 缺少 encoding='utf-8': {bad}"
    print("[PASS] test1_videoviewer_all_open_have_encoding")


def test2_app_json_dump_patch():
    """测试2: app.py 的 json.dump patch 让 ensure_ascii=False"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("app_module", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    obj = {"text": "中文测试"}
    fp = io.StringIO()
    mod.json.dump(obj, fp)
    result = fp.getvalue()
    assert "中文" in result and "\\u" not in result, \
        f"中文被转义了: {result}"
    print("[PASS] test2_app_json_dump_patch")


def test3_app_syntax():
    """测试3: app.py + video_viewer.py 语法正确"""
    py_compile.compile(APP_PATH, doraise=True)
    py_compile.compile(VV_PATH, doraise=True)
    print("[PASS] test3_app_syntax")


def test4_json_load_reads_chinese():
    """测试4: Python 3.11+ json.load 默认 UTF-8，能读中文 JSON"""
    chinese_json = '{"text": "黑色管子，红色塞子，带中文✓"}'
    fp = io.StringIO(chinese_json)
    result = json.load(fp)
    assert result["text"] == "黑色管子，红色塞子，带中文✓", f"结果: {result}"
    print("[PASS] test4_json_load_reads_chinese")


def main():
    tests = [
        ("test1_videoviewer_all_open_have_encoding", test1_videoviewer_all_open_have_encoding),
        ("test2_app_json_dump_patch", test2_app_json_dump_patch),
        ("test3_app_syntax", test3_app_syntax),
        ("test4_json_load_reads_chinese", test4_json_load_reads_chinese),
    ]
    passed = failed = 0
    errors = []
    for name, func in tests:
        print(f"\n===== {name} =====")
        try:
            _run(func, timeout=TEST_TIMEOUT // len(tests) + 10)
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e), traceback.format_exc()))
            print(f"[FAIL] {name}: {e}")
    print(f"\n===== 结果（总超时 {TEST_TIMEOUT}s） =====")
    print(f"通过: {passed}, 失败: {failed}")
    if errors:
        for n, e, tb in errors:
            print("-" * 40)
            print(f"[{n}] {e}\n{tb}")
        sys.exit(1)
    print("全部通过！")
    sys.exit(0)


if __name__ == "__main__":
    main()
