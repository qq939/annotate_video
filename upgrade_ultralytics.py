import subprocess

venv = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("升级ultralytics到最新版本...")
r = subprocess.run(
    [venv, "-m", "pip", "install", "ultralytics>=8.4.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace'
)
print(f"返回码: {r.returncode}")
if r.returncode == 0:
    print("安装成功!")
else:
    print(r.stderr[-500:] if r.stderr else "无错误")

print("\n验证SAM3VideoPredictor...")
r2 = subprocess.run([venv, "-c", "from ultralytics.models.sam import SAM3VideoPredictor; print('SAM3VideoPredictor OK')"], capture_output=True, text=True, timeout=60)
print(r2.stdout, r2.stderr)