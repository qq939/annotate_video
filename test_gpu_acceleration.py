#!/usr/bin/env python3
"""测试GPU加速功能"""
import sys
import time
import torch
import cv2
import numpy as np

def test_torch_cuda():
    """测试PyTorch CUDA是否可用"""
    print("=" * 60)
    print("测试1: PyTorch CUDA检测")
    print("=" * 60)

    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        return True
    else:
        print("ERROR: CUDA不可用")
        return False

def test_ultralytics_gpu():
    """测试ultralytics是否支持GPU"""
    print("\n" + "=" * 60)
    print("测试2: Ultralytics GPU支持")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        print("ultralytics导入成功")

        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")

        if torch.cuda.is_available():
            model = YOLO('yolov8n.pt')
            results = model('https://ultralytics.com/images/bus.jpg', device=device, verbose=False)
            print("YOLO GPU推理成功!")
            return True
        else:
            print("WARNING: CUDA不可用，只能使用CPU")
            return False
    except Exception as e:
        print(f"ERROR: Ultralytics测试失败: {e}")
        return False

def test_sam3_gpu():
    """测试SAM3模型GPU加速"""
    print("\n" + "=" * 60)
    print("测试3: SAM3 GPU加速测试")
    print("=" * 60)

    try:
        from ultralytics.models.sam import SAM3VideoPredictor, SAM3VideoSemanticPredictor

        device = '0' if torch.cuda.is_available() else 'cpu'
        half = device == '0'

        print(f"设备: {device}, 半精度(half): {half}")

        if device == '0':
            print("正在测试SAM3VideoPredictor GPU推理...")

            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model="sam_b.pt",
                device=device,
                half=half,
                save=False,
                verbose=False,
                batch=1
            )

            try:
                predictor = SAM3VideoPredictor(overrides=overrides)
                print("SAM3VideoPredictor GPU初始化成功!")

                test_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(test_img, (100, 100), (300, 300), (255, 255, 255), -1)

                start_time = time.time()
                results = predictor(source=test_img, verbose=False)
                elapsed = time.time() - start_time

                print(f"SAM3 GPU推理成功! 耗时: {elapsed:.3f}秒")
                print(f"检测到 {len(results)} 个结果")
                return True
            except Exception as e:
                print(f"SAM3VideoPredictor测试失败: {e}")
                print("这可能是正常的，如果sam_b.pt不存在")
                return False
        else:
            print("CUDA不可用，跳过SAM3 GPU测试")
            return False
    except ImportError as e:
        print(f"ERROR: 无法导入ultralytics.models.sam: {e}")
        return False
    except Exception as e:
        print(f"ERROR: SAM3测试失败: {e}")
        return False

def test_cuda_memory():
    """测试CUDA内存分配"""
    print("\n" + "=" * 60)
    print("测试4: CUDA内存测试")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA不可用，跳过内存测试")
        return False

    try:
        print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"当前内存分配: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"最大内存分配: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

        x = torch.randn(1000, 1000).cuda()
        print(f"分配1000x1000张量后: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        del x
        torch.cuda.empty_cache()
        print(f"释放后: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

        return True
    except Exception as e:
        print(f"ERROR: CUDA内存测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("GPU加速功能测试")
    print("=" * 60)
    print()

    results = []

    results.append(("PyTorch CUDA", test_torch_cuda()))
    results.append(("Ultralytics GPU", test_ultralytics_gpu()))
    results.append(("SAM3 GPU", test_sam3_gpu()))
    results.append(("CUDA Memory", test_cuda_memory()))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:20s}: {status}")

    all_passed = all(result for _, result in results)
    print()
    if all_passed:
        print("[PASS] All tests passed! GPU acceleration ready")
        return 0
    else:
        print("[WARNING] Some tests failed, please check configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())