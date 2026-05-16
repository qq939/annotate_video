import subprocess
python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("Properly patching SAM3...")
patch_script = '''
import sys
import os

# Find ultralytics SAM build file
import ultralytics.models.sam.build_sam3 as build_file
file_path = build_file.__file__
print(f"Patching {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace compile_mode=compile with compile_mode='default'
content = content.replace('compile_mode=compile', "compile_mode='default'")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied!")
'''

r = subprocess.run([python, "-c", patch_script], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print(r.stdout, r.stderr)

print("\nRunning test...")
r2 = subprocess.run([python, r"c:\Users\qq939\Downloads\annotate_video\test_sam3_gpu.py"], capture_output=True, text=True, timeout=300, encoding='utf-8', errors='replace')
print("STDOUT:", r2.stdout)
print("STDERR:", r2.stderr[-3000:] if len(r2.stderr) > 3000 else r2.stderr)