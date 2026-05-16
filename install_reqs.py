import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Installing requirements...")
result = subprocess.run(
    [python, "-m", "pip", "install", "-r", r"c:\Users\qq939\Downloads\annotate_video\requirements.txt", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True,
    text=True,
    timeout=600
)
print("STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
print("STDERR:", result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
print("Return code:", result.returncode)