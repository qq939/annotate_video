import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"
result = subprocess.run([python, "-c", "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"], capture_output=True, text=True, timeout=120)
print(result.stdout)
print(result.stderr)