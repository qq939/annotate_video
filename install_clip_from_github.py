import subprocess
import os
import shutil

venv_python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"
clip_dir = r"c:\temp_clip"

print("Step 1: Ensure pip works...")
result = subprocess.run([venv_python, "-m", "ensurepip", "--upgrade"], capture_output=True, text=True, timeout=60)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

print("\nStep 2: Upgrade pip...")
result = subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], capture_output=True, text=True, timeout=120)
print(result.returncode)

print("\nStep 3: Clone ultralytics CLIP...")
if os.path.exists(clip_dir):
    shutil.rmtree(clip_dir, ignore_errors=True)
result = subprocess.run(["git", "clone", "https://github.com/ultralytics/CLIP.git", clip_dir], capture_output=True, text=True, timeout=120)
print("Clone:", result.returncode)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

print("\nStep 4: Install CLIP...")
result = subprocess.run([venv_python, "-m", "pip", "install", "-e", clip_dir, "--no-deps"], capture_output=True, text=True, timeout=120)
print("Install:", result.returncode)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

print("\nStep 5: Cleanup...")
if os.path.exists(clip_dir):
    shutil.rmtree(clip_dir, ignore_errors=True)

print("\nStep 6: Verify...")
result = subprocess.run([venv_python, "-c", "import clip; print('clip:', clip.__version__ if hasattr(clip, '__version__') else 'OK')"], capture_output=True, text=True, timeout=30)
print(result.stdout, result.stderr)

print("\nDone!")