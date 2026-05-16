import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"
script = r"c:\Users\qq939\Downloads\annotate_video\test_sam3_gpu.py"

print("Running test...")
r = subprocess.run([python, script], capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace')
print("STDOUT:", r.stdout)
print("STDERR:", r.stderr[-3000:] if len(r.stderr) > 3000 else r.stderr)