import subprocess
import os

python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Step 1: Installing typing-extensions...")
r1 = subprocess.run(
    [python, "-m", "pip", "install", "typing-extensions==4.12.2"],
    capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace'
)
print("Return:", r1.returncode)

print("\nStep 2: Installing PyTorch with --no-deps...")
r2 = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126", "--no-deps"],
    capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
)
print("PyTorch return:", r2.returncode)
print("STDOUT:", r2.stdout[-500:])
print("STDERR:", r2.stderr[-500:])

if r2.returncode == 0:
    print("\n\nVerifying...")
    r3 = subprocess.run([python, "-c", "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('available:', torch.cuda.is_available())"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
    print(r3.stdout, r3.stderr)