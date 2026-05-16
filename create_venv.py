import os
import subprocess
import shutil

old_venv = r"c:\Users\qq939\Downloads\annotate_video\.venv"
backup_venv = r"c:\Users\qq939\Downloads\annotate_video\.venv_old"

print(f"Checking if {old_venv} exists...")
if os.path.exists(old_venv):
    print(f"Directory exists, attempting to rename...")
    try:
        if os.path.exists(backup_venv):
            print(f"Removing old backup {backup_venv}...")
            shutil.rmtree(backup_venv, ignore_errors=True)
            import time
            time.sleep(1)

        print(f"Renaming {old_venv} to {backup_venv}...")
        os.rename(old_venv, backup_venv)
        print("Rename successful!")
    except Exception as e:
        print(f"Rename failed: {e}")
        print("Trying to create venv anyway...")

print(f"\nCreating new virtual environment at {old_venv}...")
result = subprocess.run(
    ["python", "-m", "venv", old_venv],
    capture_output=True,
    text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)

if result.returncode == 0:
    print("\nVirtual environment created successfully!")
    print("\nNow installing PyTorch CUDA version...")

    pip_path = os.path.join(old_venv, "Scripts", "pip.exe")
    if os.path.exists(pip_path):
        print(f"pip found at {pip_path}")
        install_result = subprocess.run(
            [pip_path, "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"],
            capture_output=True,
            text=True,
            timeout=600
        )
        print("Install STDOUT:", install_result.stdout)
        print("Install STDERR:", install_result.stderr)
    else:
        print("pip.exe not found, trying python -m pip...")
        python_path = os.path.join(old_venv, "Scripts", "python.exe")
        install_result = subprocess.run(
            [python_path, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"],
            capture_output=True,
            text=True,
            timeout=600
        )
        print("Install STDOUT:", install_result.stdout)
        print("Install STDERR:", install_result.stderr)