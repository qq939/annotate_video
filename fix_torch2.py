import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Uninstalling all torch packages...")
result1 = subprocess.run([python, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "ultralytics", "ultralytics-thop"], capture_output=True, text=True, timeout=60)
print(result1.stdout, result1.stderr)

print("\nReinstalling PyTorch CUDA packages...")
result2 = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126", "--no-deps"],
    capture_output=True,
    text=True,
    timeout=600
)
print(result2.stdout)
print(result2.stderr[-2000:] if len(result2.stderr) > 2000 else result2.stderr)
print("Return code:", result2.returncode)

print("\nReinstalling ultralytics...")
result3 = subprocess.run(
    [python, "-m", "pip", "install", "ultralytics", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True,
    text=True,
    timeout=300
)
print(result3.stdout[-1000:] if len(result3.stdout) > 1000 else result3.stdout)
print(result3.stderr[-1000:] if len(result3.stderr) > 1000 else result3.stderr)
print("Return code:", result3.returncode)

if result3.returncode == 0:
    print("\n\nVerifying...")
    result4 = subprocess.run([python, "-c", "import torch; import torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"], capture_output=True, text=True, timeout=60)
    print(result4.stdout, result4.stderr)