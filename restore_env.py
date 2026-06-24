import subprocess
import os

venv_python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

packages = [
    "numpy>=1.26.0",
    "opencv-python>=4.10.0",
    "scipy>=1.12.0",
    "PyQt5>=5.15.0",
    "pillow>=10.0.0",
    "pillow-heif>=0.15.0",
    "ultralytics>=8.3.0",
    "timm>=0.9.0",
    "matplotlib>=3.8.0",
    "flask>=3.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
    "safetensors>=0.4.0",
    "psutil>=5.9.0",
    "polars>=1.0.0",
    "httpx>=0.27.0",
    "requests>=2.32.0",
    "seaborn>=0.13.0",
    "pandas>=2.2.0",
]

print("=" * 60)
print("恢复虚拟环境")
print("=" * 60)

print("\nStep 1: 升级pip...")
result = subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"], capture_output=True, text=True, timeout=120)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

print("\nStep 2: 安装PyTorch CUDA版本...")
result = subprocess.run(
    [venv_python, "-m", "pip", "install", "torch==2.5.0+cu124", "torchvision==0.20.0+cu124", "--index-url", "https://download.pytorch.org/whl/cu124", "--no-deps"],
    capture_output=True, text=True, timeout=600
)
print("PyTorch:", result.returncode)
if result.returncode != 0:
    print(result.stderr[-500:])

print("\nStep 3: 安装基础包...")
for pkg in packages[:5]:
    result = subprocess.run([venv_python, "-m", "pip", "install", pkg, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"], capture_output=True, text=True, timeout=120)
    print(f"  {pkg}: {'OK' if result.returncode == 0 else 'FAIL'}")

print("\nStep 4: 安装ultralytics...")
result = subprocess.run([venv_python, "-m", "pip", "install", "ultralytics==8.3.0", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"], capture_output=True, text=True, timeout=300)
print("ultralytics:", result.returncode)

print("\nStep 5: 安装其他包...")
for pkg in packages[5:]:
    result = subprocess.run([venv_python, "-m", "pip", "install", pkg, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"], capture_output=True, text=True, timeout=120)
    print(f"  {pkg}: {'OK' if result.returncode == 0 else 'FAIL'}")

print("\nStep 6: 验证核心包...")
core_check = ["torch", "torchvision", "cv2", "numpy", "ultralytics"]
for pkg in core_check:
    check_name = pkg.replace("cv2", "cv2").replace("torch", "torch").replace("torchvision", "torchvision")
    result = subprocess.run([venv_python, "-c", f"import {pkg.replace('cv2', 'cv2')}; print('{pkg}: OK')"], capture_output=True, text=True, timeout=30)
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"  {pkg}: {status}")
    if result.returncode != 0:
        print(f"    Error: {result.stderr[:200]}")

print("\n" + "=" * 60)
print("恢复完成")
print("=" * 60)
