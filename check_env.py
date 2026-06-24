import subprocess
venv = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("检查环境状态...")

checks = [
    ("torch", "import torch; print('OK', torch.__version__, torch.version.cuda)"),
    ("torchvision", "import torchvision; print('OK', torchvision.__version__)"),
    ("cv2", "import cv2; print('OK', cv2.__version__)"),
    ("numpy", "import numpy; print('OK', numpy.__version__)"),
    ("ultralytics", "import ultralytics; print('OK', ultralytics.__version__)"),
]

for name, code in checks:
    result = subprocess.run([venv, "-c", code], capture_output=True, text=True, timeout=30)
    status = "✓" if result.returncode == 0 else "✗"
    print(f"{status} {name}: {result.stdout.strip() if result.returncode == 0 else result.stderr.strip()[:100]}")

print("\n如果torch不可用，安装：")
print(f'{venv} -m pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps')