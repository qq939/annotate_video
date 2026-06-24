import subprocess

venv = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("1. 安装ultralytics...")
r = subprocess.run([venv, "-m", "pip", "install", "ultralytics==8.3.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"], capture_output=True, text=True, timeout=300)
print(f"  返回码: {r.returncode}")
print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
if r.returncode != 0:
    print(r.stderr[-300:])

print("\n2. 验证...")
r2 = subprocess.run([venv, "-c", "import ultralytics; print('ultralytics OK:', ultralytics.__version__)"], capture_output=True, text=True, timeout=30)
print(r2.stdout, r2.stderr)

print("\n3. 检查torch版本...")
r3 = subprocess.run([venv, "-c", "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)"], capture_output=True, text=True, timeout=30)
print(r3.stdout, r3.stderr)