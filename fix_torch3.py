import subprocess
import sys

python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Step 1: Installing PyTorch 2.6.0 CUDA first...")
r1 = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.6.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126"],
    capture_output=True, text=True, timeout=600
)
print("Step 1 return:", r1.returncode)
if r1.returncode != 0:
    print("STDERR:", r1.stderr[-1000:])

print("\nStep 2: Installing torchvision without deps...")
r2 = subprocess.run(
    [python, "-m", "pip", "install", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126", "--no-deps"],
    capture_output=True, text=True, timeout=300
)
print("Step 2 return:", r2.returncode)

print("\nStep 3: Uninstalling old ultralytics...")
r3 = subprocess.run(
    [python, "-m", "pip", "uninstall", "-y", "ultralytics", "ultralytics-thop"],
    capture_output=True, text=True, timeout=60
)
print("Step 3 return:", r3.returncode)

print("\nStep 4: Installing ultralytics with dependencies...")
r4 = subprocess.run(
    [python, "-m", "pip", "install", "ultralytics", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=300
)
print("Step 4 return:", r4.returncode)
print("STDOUT:", r4.stdout[-1000:])
print("STDERR:", r4.stderr[-1000:])

print("\n\nTesting imports...")
r5 = subprocess.run(
    [python, "-c", "import torch; import torchvision; import ultralytics; print('OK: torch', torch.__version__, 'torchvision', torchvision.__version__, 'ultralytics', ultralytics.__version__)"],
    capture_output=True, text=True, timeout=60
)
print(r5.stdout, r5.stderr)