import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Uninstalling ultralytics...")
subprocess.run([python, "-m", "pip", "uninstall", "-y", "ultralytics", "ultralytics-thop"], capture_output=True, text=True, timeout=60)

print("\nInstalling compatible ultralytics 8.3.0...")
r = subprocess.run(
    [python, "-m", "pip", "install", "ultralytics==8.3.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=300
)
print("Return:", r.returncode)
print("STDOUT:", r.stdout[-1000:])
print("STDERR:", r.stderr[-500:])

print("\nRunning test...")
import os
env = os.environ.copy()
result = subprocess.run(
    [python, r"c:\Users\qq939\Downloads\annotate_video\test_sam3_gpu.py"],
    capture_output=True, text=True, timeout=300, env=env
)
print(result.stdout)
print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)