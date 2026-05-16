import subprocess
import os

python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Installing PyTorch 2.6.0 with dependencies...")
r = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126", "--upgrade"],
    capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
)
print("Return:", r.returncode)
print("STDOUT:", r.stdout[-1000:])
print("STDERR:", r.stderr[-1000:])

if r.returncode == 0:
    print("\n\nVerifying...")
    r2 = subprocess.run([python, "-c", "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('available:', torch.cuda.is_available())"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
    print(r2.stdout, r2.stderr)