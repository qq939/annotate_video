import subprocess
import sys
import os

venv_python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"
venv_site_packages = r"c:\Users\qq939\Downloads\annotate_video\.venv\Lib\site-packages"

cmd = [
    r"F:\ComfyUI2\py311\python.exe",
    "-m", "pip", "install", "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu126",
    "--target", venv_site_packages
]

print("Installing PyTorch CUDA version to virtual environment...")
print(f"Command: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)