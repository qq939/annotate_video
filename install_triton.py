import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Installing triton...")
r = subprocess.run([python, "-m", "pip", "install", "triton", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"], capture_output=True, text=True, timeout=300)
print("Return:", r.returncode)
print("STDOUT:", r.stdout[-1000:])
print("STDERR:", r.stderr[-1000:])

print("\nRunning test again...")
script = r"c:\Users\qq939\Downloads\annotate_video\test_sam3_gpu.py"
r2 = subprocess.run([python, script], capture_output=True, text=True, timeout=300)
print(r2.stdout)
print(r2.stderr[-3000:] if len(r2.stderr) > 3000 else r2.stderr)