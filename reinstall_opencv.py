import subprocess

venv_python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Installing opencv-python...")
result = subprocess.run(
    [venv_python, "-m", "pip", "install", "opencv-python", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=120
)
print("Return:", result.returncode)
print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

print("\nVerifying...")
result2 = subprocess.run([venv_python, "-c", "import cv2; print('cv2 OK:', cv2.__version__)"], capture_output=True, text=True, timeout=30)
print(result2.stdout, result2.stderr)