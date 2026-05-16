import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Uninstalling all torch packages...")
subprocess.run([python, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "ultralytics", "ultralytics-thop"], capture_output=True, text=True, timeout=60)

print("\nInstalling PyTorch 2.5.0 from PyPI (CPU version first)...")
r1 = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.5.0", "torchvision==0.20.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=600
)
print("Step 1 return:", r1.returncode)

print("\nInstalling ultralytics...")
r2 = subprocess.run(
    [python, "-m", "pip", "install", "ultralytics==8.3.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=300
)
print("Step 2 return:", r2.returncode)
print("STDOUT:", r2.stdout[-1000:])
print("STDERR:", r2.stderr[-1000:])

print("\n\nTesting imports...")
r3 = subprocess.run(
    [python, "-c", "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('CUDA available:', torch.cuda.is_available())"],
    capture_output=True, text=True, timeout=60
)
print(r3.stdout, r3.stderr)