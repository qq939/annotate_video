import subprocess

venv = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("验证所有包...")
packages = [
    "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)",
    "import torchvision; print('torchvision:', torchvision.__version__)",
    "import cv2; print('cv2:', cv2.__version__)",
    "import numpy; print('numpy:', numpy.__version__)",
    "import ultralytics; print('ultralytics:', ultralytics.__version__)",
]

for code in packages:
    r = subprocess.run([venv, "-c", code], capture_output=True, text=True, timeout=30, errors='replace')
    print(r.stdout.strip() if r.stdout else f"FAIL: {r.stderr.strip()[:100]}")

print("\n注意: torch是CPU版本，CUDA: None")
print("如果需要GPU加速，运行：")
print(f'{venv} -m pip install torch==2.5.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps')