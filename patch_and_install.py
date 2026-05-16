import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Installing ultralytics 8.4.51...")
r1 = subprocess.run(
    [python, "-m", "pip", "install", "ultralytics==8.4.51", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple/"],
    capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace'
)
print("Return:", r1.returncode)
print("STDOUT:", r1.stdout[-500:])

print("\nPatching SAM3 to disable torch.compile...")
patch_script = '''
import sys
import os

# Find ultralytics SAM build file
import ultralytics.models.sam.build_sam3 as build_file
file_path = build_file.__file__
print(f"Patching {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace compile_mode=compile with compile_mode='default' or None
content = content.replace('compile_mode=compile', 'compile_mode=None')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied!")
'''

r2 = subprocess.run([python, "-c", patch_script], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print("Patch result:", r2.stdout, r2.stderr)

print("\nTesting imports...")
r3 = subprocess.run([python, "-c", "from ultralytics.models.sam import SAM3VideoPredictor; print('SAM3VideoPredictor OK')"], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print(r3.stdout, r3.stderr)