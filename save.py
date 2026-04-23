#!/usr/bin/env python3
"""从temp_data_post读取数据，制作标注视频并上传到OBS"""

INPUT_DIR = "temp_data_post"
OUTPUT_PATH = "dst/dst.mp4"
DEFAULT_ALPHA = 0.5
DEFAULT_CATEGORY = "Detect"

import cv2
import numpy as np
import json
from pathlib import Path
import subprocess
import sys

def save_video(input_path=None, output_path=None, alpha=None, category_name=None):
    if input_path is None:
        input_path = INPUT_DIR
    if output_path is None:
        output_path = OUTPUT_PATH
    if alpha is None:
        alpha = DEFAULT_ALPHA
    if category_name is None:
        category_name = DEFAULT_CATEGORY
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    annotations_path = input_path / "annotations.json"
    if not annotations_path.exists():
        print(f"错误：找不到 {annotations_path}")
        return
    
    with open(annotations_path) as f:
        coco_data = json.load(f)
    
    video_info = coco_data.get('info', {})
    total_frames = len(coco_data.get('images', []))
    
    if total_frames == 0:
        print("错误：没有找到帧数据")
        return
    
    width = int(video_info.get('width', 1280))
    height = int(video_info.get('height', 720))
    fps = int(video_info.get('fps', 30))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    labels_dir = input_path / "labels"
    frames_dir = input_path / "frames"
    
    mask_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    print(f"正在生成视频: {output_path}")
    print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    for i in range(total_frames):
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        label_path = labels_dir / f"frame_{i:06d}.json"
        if label_path.exists():
            with open(label_path) as f:
                annotations = json.load(f)
            
            result_frame = frame.copy()
            overlay = frame.copy()
            
            for ann in annotations:
                polygon = ann.get('segmentation')
                bbox = ann.get('bbox')
                
                if not bbox:
                    continue
                
                color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
                category = ann.get('category', category_name)
                conf = ann.get('confidence', 1.0)
                
                if polygon:
                    pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
                
                x, y = int(bbox[0]), int(bbox[1])
                w, h = int(bbox[2]), int(bbox[3])
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                cv2.putText(overlay, f"{category} {conf:.2f}", (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
            frame = result_frame
        
        out.write(frame)
        
        if (i + 1) % 30 == 0:
            print(f"已处理 {i + 1}/{total_frames} 帧")
    
    out.release()
    print(f"视频已保存: {output_path}")
    
    print("正在上传到OBS...")
    try:
        result = subprocess.run(
            ['curl', '--upload-file', str(output_path), 'http://obs.dimond.top/dst.mp4'],
            capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        if result.returncode == 0:
            print("上传成功!")
        else:
            print(f"上传失败: {result.stderr}")
    except Exception as e:
        print(f"上传失败: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='从temp_data_post生成标注视频')
    parser.add_argument('--input', '-i', default=INPUT_DIR, help='输入目录')
    parser.add_argument('--output', '-o', default=OUTPUT_PATH, help='输出视频路径')
    parser.add_argument('--alpha', '-a', type=float, default=DEFAULT_ALPHA, help='透明度 (0.0-1.0)')
    parser.add_argument('--category', '-c', default=DEFAULT_CATEGORY, help='类别名称')
    
    args = parser.parse_args()
    save_video(args.input, args.output, args.alpha, args.category)

if __name__ == "__main__":
    main()
