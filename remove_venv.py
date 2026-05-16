import shutil
import os
import time

venv_path = r"c:\Users\qq939\Downloads\annotate_video\.venv"

print(f"Attempting to remove {venv_path}...")

if os.path.exists(venv_path):
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            time.sleep(1)
            shutil.rmtree(venv_path, ignore_errors=False)
            print(f"Successfully removed {venv_path}")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

    if os.path.exists(venv_path):
        print("Failed to remove .venv directory")
        print("You may need to close all applications using files in this directory")
        print("and then manually delete it.")
    else:
        print("Creating new virtual environment...")

        import subprocess
        result = subprocess.run(
            ["python", "-m", "venv", venv_path],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(result.stderr)
else:
    print(f"{venv_path} does not exist, creating new one...")
    result = subprocess.run(
        ["python", "-m", "venv", venv_path],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)