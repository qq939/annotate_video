#!/usr/bin/env python3
"""测试临时视频生成功能"""
import os
import sys
import cv2
import tempfile
import time
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from app import extract_video_segment


def test_extract_video_segment():
    """测试从视频中提取片段生成临时视频"""
    # 创建测试视频
    test_video_path = os.path.join(tempfile.gettempdir(), "test_video.mp4")
    
    # 生成一个简单的测试视频（30帧，640x480）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))
    
    for i in range(30):  # 30帧的测试视频
        # 创建彩色帧
        frame = np.zeros((480, 640, 3), dtype='uint8')
        frame[:, :, 0] = (i * 10) % 255  # B
        frame[:, :, 1] = (i * 5) % 255   # G
        frame[:, :, 2] = (255 - i * 5) % 255  # R
        out.write(frame)
    out.release()
    
    print(f"[TEST] 创建测试视频: {test_video_path}")
    cap = cv2.VideoCapture(test_video_path)
    print(f"[TEST] 测试视频帧数: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    cap.release()
    
    # 测试提取片段
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_segment.mp4")
    
    # 测试1: 从第0秒开始，取前10帧
    start_time = 0
    max_frames = 10
    result = extract_video_segment(test_video_path, temp_video_path, start_time, max_frames)
    
    assert os.path.exists(result), f"临时视频未生成: {result}"
    cap = cv2.VideoCapture(result)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    assert frame_count == 10, f"帧数不对: {frame_count}, 期望: 10"
    print(f"[PASS] 测试1通过: 从第0秒开始取前10帧")
    
    # 测试2: 从第0.3秒开始，取前5帧
    start_time = 0.3
    max_frames = 5
    result = extract_video_segment(test_video_path, temp_video_path, start_time, max_frames)
    
    assert os.path.exists(result), f"临时视频未生成: {result}"
    cap = cv2.VideoCapture(result)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    assert frame_count == 5, f"帧数不对: {frame_count}, 期望: 5"
    print(f"[PASS] 测试2通过: 从第0.3秒开始取前5帧")
    
    # 测试3: 从第0.5秒开始，取前15帧
    start_time = 0.5
    max_frames = 15
    result = extract_video_segment(test_video_path, temp_video_path, start_time, max_frames)
    
    assert os.path.exists(result), f"临时视频未生成: {result}"
    cap = cv2.VideoCapture(result)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    assert frame_count == 15, f"帧数不对: {frame_count}, 期望: 15"
    print(f"[PASS] 测试3通过: 从第0.5秒开始取前15帧")
    
    # 清理
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    print("[ALL TESTS PASSED]")


if __name__ == "__main__":
    timeout = 30
    start = time.time()
    try:
        test_extract_video_segment()
    except Exception as e:
        print(f"[FAILED] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"[TIMEOUT] 测试超时: {elapsed:.1f}s > {timeout}s")
            sys.exit(1)
