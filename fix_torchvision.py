import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Checking installed versions...")
r1 = subprocess.run([python, "-c", "import torch; import torchvision; import ultralytics; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('ultralytics:', ultralytics.__version__)"], capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
print(r1.stdout, r1.stderr)

print("\nTrying to fix torchvision...")
r2 = subprocess.run([python, "-m", "pip", "uninstall", "-y", "torchvision"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print("Uninstall return:", r2.returncode)

print("\nReinstalling torchvision...")
r3 = subprocess.run([python, "-m", "pip", "install", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126", "--force-reinstall", "--no-deps"], capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace')
print("Reinstall return:", r3.returncode)
print("STDOUT:", r3.stdout[-500:])
print("STDERR:", r3.stderr[-500:])

print("\nTesting imports...")
r4 = subprocess.run([python, "-c", "import torch; import torchvision; print('OK')"], capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
print(r4.stdout, r4.stderr)