import subprocess

venv = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("安装PyQt5...")
r = subprocess.run(
    [venv, "-m", "pip", "install", "PyQt5", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=120, encoding='utf-8', errors='replace'
)
print(f"返回码: {r.returncode}")
if r.returncode == 0:
    print("安装成功!")
else:
    print(r.stderr[-500:] if r.stderr else "无错误")

print("\n验证PyQt5...")
r2 = subprocess.run([venv, "-c", "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"], capture_output=True, text=True, timeout=30)
print(r2.stdout, r2.stderr)