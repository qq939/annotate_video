import subprocess

venv = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("安装PyTorch CUDA 12.6版本...")
r = subprocess.run(
    [venv, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126", "--no-deps"],
    capture_output=True, text=True, timeout=600, encoding='utf-8', errors='replace'
)
print(f"返回码: {r.returncode}")
if r.returncode == 0:
    print("安装成功!")
else:
    print("安装失败:")
    print(r.stderr[-1000:] if r.stderr else "无错误信息")

print("\n验证...")
r2 = subprocess.run([venv, "-c", "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)"], capture_output=True, text=True, timeout=30)
print(r2.stdout, r2.stderr)