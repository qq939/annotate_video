import subprocess

venv = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("=" * 50)
print("环境版本信息")
print("=" * 50)

r = subprocess.run([venv, "-c", "import sys, torch, torchvision; print(f'Python: {sys.version.split()[0]}'); print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print(f'CUDA: {torch.version.cuda}')"], capture_output=True, text=True, timeout=30)
print(r.stdout)
print(r.stderr if r.stderr else "")
print("=" * 50)