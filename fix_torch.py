import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Uninstalling old torchvision...")
result1 = subprocess.run([python, "-m", "pip", "uninstall", "-y", "torchvision"], capture_output=True, text=True, timeout=60)
print(result1.stdout, result1.stderr)

print("\nReinstalling compatible torch and torchvision...")
result2 = subprocess.run(
    [python, "-m", "pip", "install", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126"],
    capture_output=True,
    text=True,
    timeout=600
)
print(result2.stdout[-2000:] if len(result2.stdout) > 2000 else result2.stdout)
print(result2.stderr[-2000:] if len(result2.stderr) > 2000 else result2.stderr)
print("Return code:", result2.returncode)

if result2.returncode == 0:
    print("\n\nVerifying...")
    result3 = subprocess.run([python, "-c", "import torch; import torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"], capture_output=True, text=True, timeout=60)
    print(result3.stdout, result3.stderr)