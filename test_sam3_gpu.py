#!/usr/bin/env python3
"""Test SAM3 model GPU loading and inference using video source"""
import sys
import os
import glob
import cv2
import numpy as np

print("=" * 60)
print("SAM3 GPU Loading Test")
print("=" * 60)

import torch
print(f"\n[INFO] PyTorch: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA version: {torch.version.cuda}")

print("\n[STEP 1] Creating test video...")
temp_video = "temp_test_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_video, fourcc, 10.0, (640, 480))
for i in range(30):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)
    out.write(frame)
out.release()
print(f"[PASS] Created test video: {temp_video}")

video_path = temp_video

print("\n[STEP 2] Testing SAM3VideoPredictor import...")
try:
    from ultralytics.models.sam import SAM3VideoPredictor
    print("[PASS] SAM3VideoPredictor imported")
except Exception as e:
    print(f"[FAIL] SAM3VideoPredictor import failed: {e}")
    sys.exit(1)

print("\n[STEP 3] Loading SAM3 model...")
try:
    overrides = {
        'conf': 0.25,
        'task': 'segment',
        'mode': 'predict',
        'model': 'sam3.pt',
        'device': '0',
        'half': True,
        'save': False,
        'batch': 1
    }

    print(f"[DEBUG] Using overrides: device={overrides['device']}, half={overrides['half']}")

    predictor = SAM3VideoPredictor(overrides=overrides)
    print("[PASS] SAM3VideoPredictor initialized")

except FileNotFoundError as e:
    print(f"[WARN] sam3.pt not found, trying sam_b.pt...")
    try:
        overrides['model'] = 'sam_b.pt'
        predictor = SAM3VideoPredictor(overrides=overrides)
        print("[PASS] SAM3VideoPredictor(sam_b.pt) initialized")
    except FileNotFoundError:
        print("[FAIL] sam_b.pt also not found")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] SAM3VideoPredictor init failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[INFO] Model args.device: {predictor.args.device if hasattr(predictor.args, 'device') else 'N/A'}")

print("\n[STEP 4] Running GPU inference...")
print(f"[DEBUG] Pre-inference GPU memory: {torch.cuda.memory_allocated(0)/1024**3:.3f}GB")
print(f"[DEBUG] Pre-inference GPU cache: {torch.cuda.memory_reserved(0)/1024**3:.3f}GB")

try:
    print(f"[INFO] Using video source: {video_path}")

    results = predictor(source=video_path, stream=True, bboxes=[[100, 100, 500, 400]], labels=[1])

    frame_count = 0
    for r in results:
        frame_count += 1
        if frame_count == 1:
            print(f"\n[DEBUG] First frame result:")
            print(f"  - results type: {type(r)}")
            if r.masks is not None:
                print(f"  - masks.data type: {type(r.masks.data)}")
                if hasattr(r.masks.data, 'device'):
                    print(f"  - masks.data.device: {r.masks.data.device}")
            if hasattr(r, 'orig_img') and r.orig_img is not None:
                print(f"  - orig_img shape: {r.orig_img.shape}")

        if frame_count % 10 == 0:
            print(f"[INFO] Processed {frame_count} frames, GPU memory: {torch.cuda.memory_allocated(0)/1024**3:.3f}GB")

        if frame_count >= 50:
            print(f"[INFO] Processed {frame_count} frames, stopping test")
            break

    print(f"\n[DEBUG] Post-inference GPU memory: {torch.cuda.memory_allocated(0)/1024**3:.3f}GB")
    print(f"[DEBUG] Post-inference GPU cache: {torch.cuda.memory_reserved(0)/1024**3:.3f}GB")

    print(f"\n[PASS] GPU inference completed! Processed {frame_count} frames")

    if torch.cuda.memory_allocated(0) > 0.1:
        print("[PASS] GPU memory is being used, GPU acceleration confirmed!")
    else:
        print("[WARN] GPU memory usage is low, may not be using GPU correctly")

except Exception as e:
    print(f"[FAIL] GPU inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    if os.path.exists(temp_video):
        try:
            os.remove(temp_video)
            print(f"[INFO] Temp video deleted: {temp_video}")
        except:
            pass

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)