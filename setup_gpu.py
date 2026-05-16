#!/usr/bin/env python3
"""Setup script to configure GPU environment and patch ultralytics"""
import subprocess
import os

python = r"c:\Users\qq939\Downloads\annotate_video\.venv\Scripts\python.exe"

print("=" * 60)
print("Setting up GPU environment")
print("=" * 60)

print("\nStep 1: Patching ultralytics for torch.compile compatibility...")
patch_script = '''
import sys

try:
    import ultralytics.models.sam.build_sam3 as build_file
    file_path = build_file.__file__
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'compile_mode=None_mode' in content:
        content = content.replace('compile_mode=None_mode', "compile_mode='default'")
        content = content.replace('compile_mode=compile', "compile_mode='default'")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("[PATCH] build_sam3.py patched")
    
    import ultralytics.models.sam.sam3.vitdet as vitdet_file
    vitdet_path = vitdet_file.__file__
    with open(vitdet_path, 'r', encoding='utf-8') as f:
        vit_content = f.read()
    if 'torch.compile(self.forward' in vit_content:
        vit_content = vit_content.replace(
            'self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)',
            '# self.forward = torch.compile(self.forward, mode=compile_mode, fullgraph=True)  # Disabled'
        )
        with open(vitdet_path, 'w', encoding='utf-8') as f:
            f.write(vit_content)
        print("[PATCH] vitdet.py patched")
    
    print("[OK] Ultralytics patched successfully!")
except Exception as e:
    print(f"[WARN] Patch warning: {e}")
'''

r = subprocess.run([python, "-c", patch_script], capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace')
print(r.stdout, r.stderr)

print("\nStep 2: Verifying GPU environment...")
r2 = subprocess.run(
    [python, "-c", "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"],
    capture_output=True, text=True, timeout=60, encoding='utf-8', errors='replace'
)
print(r2.stdout, r2.stderr)

print("\n" + "=" * 60)
print("Setup complete!")
print("=" * 60)