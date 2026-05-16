import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Uninstalling ultralytics...")
r1 = subprocess.run([python, "-m", "pip", "uninstall", "-y", "ultralytics", "ultralytics-thop"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print("Return:", r1.returncode)

print("\nInstalling ultralytics 8.3.0...")
r2 = subprocess.run(
    [python, "-m", "pip", "install", "ultralytics==8.3.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace'
)
print("Return:", r2.returncode)
print("STDOUT:", r2.stdout[-500:])
print("STDERR:", r2.stderr[-500:])

print("\nRunning test...")
r3 = subprocess.run([python, r"c:\Users\qq939\Downloads\annotate_video\test_sam3_gpu.py"], capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace')
print("STDOUT:", r3.stdout)
print("STDERR:", r3.stderr[-3000:] if len(r3.stderr) > 3000 else r3.stderr)