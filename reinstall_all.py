import subprocess
import os

python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Step 1: Uninstall old packages...")
subprocess.run([python, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "ultralytics", "ultralytics-thop", "numpy", "opencv-python"], capture_output=True, text=True, timeout=60)

print("\nStep 2: Install PyTorch 2.6.0 with CUDA 12.6...")
r2 = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126"],
    capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
)
print("PyTorch return:", r2.returncode)
print("STDOUT:", r2.stdout[-1000:])

print("\nStep 3: Install requirements...")
r3 = subprocess.run(
    [python, "-m", "pip", "install", "-r", r"c:\Users\qq939\Downloads\annotate_video\requirements.txt", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/", "--no-deps"],
    capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
)
print("Requirements return:", r3.returncode)

print("\nStep 4: Install ultralytics without dependencies...")
r4 = subprocess.run(
    [python, "-m", "pip", "install", "ultralytics", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/", "--no-deps"],
    capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace'
)
print("ultralytics return:", r4.returncode)

print("\n\nStep 5: Verifying...")
r5 = subprocess.run([python, "-c", "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print(r5.stdout, r5.stderr)