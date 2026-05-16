import os
import subprocess

venv_path = r"c:\Users\qq939\Downloads\annotate_video\.venv"
pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
python_path = os.path.join(venv_path, "Scripts", "python.exe")

print("Upgrading pip first...")
upgrade_result = subprocess.run(
    [python_path, "-m", "pip", "install", "--upgrade", "pip"],
    capture_output=True,
    text=True,
    timeout=120
)
print("Upgrade:", upgrade_result.returncode)

print("\nInstalling PyTorch 2.6.0 (minimum available for CUDA 12.6)...")
install_result = subprocess.run(
    [python_path, "-m", "pip", "install", "torch==2.6.0+cu126", "torchvision==0.21.0+cu126", "--index-url", "https://download.pytorch.org/whl/cu126"],
    capture_output=True,
    text=True,
    timeout=600
)
print("Install STDOUT:", install_result.stdout)
print("Install STDERR:", install_result.stderr[-2000:] if len(install_result.stderr) > 2000 else install_result.stderr)
print("Return code:", install_result.returncode)

if install_result.returncode == 0:
    print("\n\nVerifying installation...")
    verify_result = subprocess.run(
        [python_path, "-c", "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"],
        capture_output=True,
        text=True,
        timeout=30
    )
    print("Verify:", verify_result.stdout, verify_result.stderr)