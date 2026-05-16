import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Uninstalling torch and torchvision...")
r1 = subprocess.run([python, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print("Return:", r1.returncode)

print("\nInstalling torch 2.5.0 cu124...")
r2 = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.5.0+cu124", "torchvision==0.20.0+cu124", "--index-url", "https://download.pytorch.org/whl/cu124", "--no-deps"],
    capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
)
print("Return:", r2.returncode)
print("STDOUT:", r2.stdout[-500:])
print("STDERR:", r2.stderr[-500:])

if r2.returncode == 0:
    print("\n\nTesting...")
    r3 = subprocess.run([python, "-c", "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
    print(r3.stdout, r3.stderr)

    print("\nTesting torchvision...")
    r4 = subprocess.run([python, "-c", "import torchvision; print('torchvision:', torchvision.__version__)"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
    print(r4.stdout, r4.stderr)