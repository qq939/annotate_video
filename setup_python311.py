import subprocess
import os

old_venv = r"c:\Users\qq939\Downloads\annotate_video\.venv"
backup = r"c:\Users\qq939\Downloads\annotate_video\.venv_old"

print("Removing old virtual environment...")
if os.path.exists(old_venv):
    try:
        if os.path.exists(backup):
            import shutil
            shutil.rmtree(backup, ignore_errors=True)
        os.rename(old_venv, backup)
        import time
        time.sleep(1)
        import shutil
        shutil.rmtree(backup, ignore_errors=True)
    except Exception as e:
        print(f"Failed to remove: {e}")

print("\nFinding Python 3.11...")
python311 = None
paths_to_check = [
    r"C:\Python311\python.exe",
    r"C:\Program Files\Python311\python.exe",
    r"F:\ComfyUI2\py311\python.exe",
    r"C:\Users\qq939\AppData\Local\Programs\Python\Python311\python.exe"
]
for path in paths_to_check:
    if os.path.exists(path):
        python311 = path
        print(f"Found Python 3.11 at: {python311}")
        break

if not python311:
    result = subprocess.run(["where", "python3.11"], capture_output=True, text=True)
    if result.returncode == 0:
        python311 = result.stdout.strip().split('\n')[0]
        print(f"Found via where: {python311}")

if not python311:
    result = subprocess.run(["py", "-3.11", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Python 3.11 via py: {result.stdout}")

if not python311:
    print("ERROR: Python 3.11 not found!")
    exit(1)

print(f"\nCreating venv with Python 3.11 at {old_venv}...")
result = subprocess.run(
    [python311, "-m", "venv", old_venv],
    capture_output=True, text=True, timeout=120
)
print("Return:", result.returncode)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

if result.returncode == 0:
    venv_python = os.path.join(old_venv, "Scripts", "python.exe")
    print(f"\nPython installed at: {venv_python}")

    print("\nInstalling PyTorch CUDA 2.5.0...")
    r2 = subprocess.run(
        [venv_python, "-m", "pip", "install", "torch==2.5.0+cu124", "torchvision==0.20.0+cu124", "--index-url", "https://download.pytorch.org/whl/cu124"],
        capture_output=True, text=True, timeout=600
    )
    print("PyTorch return:", r2.returncode)
    print("STDOUT:", r2.stdout[-2000:])
    print("STDERR:", r2.stderr[-2000:])

    print("\nInstalling requirements...")
    r3 = subprocess.run(
        [venv_python, "-m", "pip", "install", "-r", r"c:\Users\qq939\Downloads\annotate_video\requirements.txt", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
        capture_output=True, text=True, timeout=600
    )
    print("Requirements return:", r3.returncode)

    print("\n\nTesting...")
    r4 = subprocess.run(
        [venv_python, "-c", "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('available:', torch.cuda.is_available())"],
        capture_output=True, text=True, timeout=60
    )
    print(r4.stdout, r4.stderr)