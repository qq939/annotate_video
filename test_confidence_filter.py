#!/usr/bin/env python3
"""后处理模块测试 - 置信度筛选渲染"""
import sys
import signal
import time
import tempfile
import json
import cv2
import numpy as np
from pathlib import Path

signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
signal.alarm(30)

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

@timeout(30)
def test_render_annotations():
    sys.path.insert(0, '/Users/jimjiang/Downloads/biaozhu')
    from image_app import _render_filtered_image, BOX_COLORS

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    annotations = [
        {
            'bbox': [10, 10, 100, 80],
            'confidence': 0.95,
            'category_id': 0,
            'segmentation': [[10, 10, 110, 10, 110, 90, 10, 90]],
            'color': BOX_COLORS[0]
        },
        {
            'bbox': [200, 200, 150, 120],
            'confidence': 0.30,
            'category_id': 1,
            'segmentation': [[200, 200, 350, 200, 350, 320, 200, 320]],
            'color': BOX_COLORS[1]
        },
        {
            'bbox': [400, 50, 80, 60],
            'confidence': 0.70,
            'category_id': 2,
            'segmentation': [[400, 50, 480, 50, 480, 110, 400, 110]],
            'color': BOX_COLORS[2]
        },
    ]
    find_list = ['cat', 'dog', 'bird']

    result = _render_filtered_image(img, annotations, find_list, 0.5)
    assert result is not None, "渲染结果为空"
    assert result.shape == img.shape, "渲染后图片尺寸不一致"
    h, w = result.shape[:2]
    assert h == 480 and w == 640, f"尺寸错误: {w}x{h}"
    print(f"[PASS] _render_filtered_image 尺寸正确: {w}x{h}")

    result_none = _render_filtered_image(img, [], find_list, 0.5)
    assert result_none is not None, "空annotations不应返回None"
    print("[PASS] 空annotations处理正确")

@timeout(30)
def test_confidence_filtering():
    sys.path.insert(0, '/Users/jimjiang/Downloads/biaozhu')
    from image_app import _filter_by_confidence, BOX_COLORS

    annotations = [
        {'confidence': 0.95, 'bbox': [0,0,10,10]},
        {'confidence': 0.50, 'bbox': [0,0,20,20]},
        {'confidence': 0.30, 'bbox': [0,0,30,30]},
        {'confidence': 0.70, 'bbox': [0,0,40,40]},
    ]

    filtered = _filter_by_confidence(annotations, 0.6)
    assert len(filtered) == 2, f"阈值0.6期望2个, 实际{len(filtered)}"
    confs = [a['confidence'] for a in filtered]
    assert all(c >= 0.6 for c in confs), f"包含低于阈值的: {confs}"
    print(f"[PASS] 阈值0.6筛选: {len(filtered)}个 (confs={confs})")

    filtered_all = _filter_by_confidence(annotations, 0.0)
    assert len(filtered_all) == 4, f"阈值0.0应返回全部4个, 实际{len(filtered_all)}"
    print("[PASS] 阈值0.0返回全部4个")

    filtered_none = _filter_by_confidence(annotations, 1.0)
    assert len(filtered_none) == 0, f"阈值1.0应返回0个, 实际{len(filtered_none)}"
    print("[PASS] 阈值1.0返回0个")

    filtered_empty = _filter_by_confidence([], 0.5)
    assert len(filtered_empty) == 0, "空列表应返回空"
    print("[PASS] 空列表处理正确")

@timeout(30)
def test_load_annotations_json():
    sys.path.insert(0, '/Users/jimjiang/Downloads/biaozhu')
    from image_app import _load_temp_annotations

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        labels_dir = tmppath / 'labels'
        labels_dir.mkdir()
        ann_json = tmppath / 'annotations.json'
        img_path = tmppath / 'frames' / 'frame_000000.jpg'
        img_path.parent.mkdir()

        cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))
        test_anns = [{'confidence': 0.5, 'bbox': [0,0,10,10]}]
        ann_json.write_text(json.dumps({'annotations': test_anns}))
        (labels_dir / 'frame_000000.json').write_text(json.dumps(test_anns))

        coco, labels, img = _load_temp_annotations(str(tmppath))
        assert coco is not None, "coco数据为空"
        assert labels is not None, "labels数据为空"
        assert img is not None, "图片为空"
        print(f"[PASS] _load_temp_annotations 返回正确类型 coco={type(coco)}, labels={type(labels)}, img={type(img)}")

if __name__ == '__main__':
    print('=== 后处理模块测试 ===')
    test_confidence_filtering()
    test_render_annotations()
    test_load_annotations_json()
    print('\n=== 全部测试通过 ===')
