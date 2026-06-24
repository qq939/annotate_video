import subprocess

venv_python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Installing CLIP from git...")
result = subprocess.run(
    [venv_python, "-m", "pip", "install", "git+https://github.com/ultralytics/CLIP.git", "--no-deps", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=180
)
print("Return:", result.returncode)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

print("\nVerifying...")
result2 = subprocess.run([venv_python, "-c", "import clip; print('clip OK:', clip.__file__)"], capture_output=True, text=True, timeout=30)
print(result2.stdout, result2.stderr)