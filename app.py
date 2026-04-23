#!/usr/bin/env python3
"""
Flask Web应用：视频标注工具
实现annotate_video和video_viewer的全部功能
画面不允许缩放，画框位置用硬编码
"""

from flask import Flask, render_template, request, jsonify, Response, send_file, session
import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'video_annotator_secret_key_2024'

TEMP_DATA_DIR = "temp_data"
DST_DIR = "dst"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_CONF_THRESHOLD = 0.5

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', 
                         video_width=VIDEO_WIDTH, 
                         video_height=VIDEO_HEIGHT)

@app.route('/api/init', methods=['POST'])
def init():
    """初始化项目"""
    data = request.json
    project_name = data.get('project_name', f'project_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    temp_path = Path(TEMP_DATA_DIR)
    temp_path.mkdir(exist_ok=True)
    
    session['project_name'] = project_name
    session['del_track_ids'] = []
    session['del_points'] = []
    session['conf_threshold'] = DEFAULT_CONF_THRESHOLD
    session['current_frame'] = 0
    
    return jsonify({
        'success': True,
        'project_name': project_name,
        'video_width': VIDEO_WIDTH,
        'video_height': VIDEO_HEIGHT
    })

@app.route('/api/frames/<int:frame_idx>')
def get_frame(frame_idx):
    """获取指定帧"""
    frames_dir = Path(TEMP_DATA_DIR) / "frames"
    frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
    
    if frame_path.exists():
        return send_file(str(frame_path), mimetype='image/jpeg')
    return Response(status=404)

@app.route('/api/frame_with_masks/<int:frame_idx>')
def get_frame_with_masks(frame_idx):
    """获取带mask渲染的帧"""
    frames_dir = Path(TEMP_DATA_DIR) / "frames"
    labels_dir = Path(TEMP_DATA_DIR) / "labels"
    
    frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
    label_path = labels_dir / f"frame_{frame_idx:06d}.json"
    
    if frame_path.exists():
        frame = cv2.imread(str(frame_path))
    else:
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    
    del_track_ids = session.get('del_track_ids', [])
    conf_threshold = session.get('conf_threshold', DEFAULT_CONF_THRESHOLD)
    
    if label_path.exists():
        with open(label_path) as f:
            annotations = json.load(f)
        
        mask_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for ann in annotations:
            if ann.get('confidence', 1.0) < conf_threshold:
                continue
            
            track_id = ann.get('track_id', ann['id'])
            if track_id in del_track_ids:
                continue
            
            polygon = ann['segmentation'][0]
            pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
            
            cv2.fillPoly(frame, [pts], color)
            cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
            
            bbox = ann['bbox']
            x, y = int(bbox[0]), int(bbox[1])
            text = f"ID:{track_id} {ann.get('confidence', 1.0):.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    del_points = session.get('del_points', [])
    for dp in del_points:
        if dp.get('frame_idx') == frame_idx:
            x, y = int(dp['x']), int(dp['y'])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, dp.get('shortcut', 'X'), (x + 8, y + 3), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    ret, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/frame_info/<int:frame_idx>')
def get_frame_info(frame_idx):
    """获取帧信息"""
    labels_dir = Path(TEMP_DATA_DIR) / "labels"
    label_path = labels_dir / f"frame_{frame_idx:06d}.json"
    
    conf_threshold = session.get('conf_threshold', DEFAULT_CONF_THRESHOLD)
    del_track_ids = session.get('del_track_ids', [])
    
    visible_count = 0
    total_count = 0
    
    if label_path.exists():
        with open(label_path) as f:
            annotations = json.load(f)
        
        total_count = len(annotations)
        visible_count = sum(1 for ann in annotations 
                          if ann.get('confidence', 1.0) >= conf_threshold
                          and ann.get('track_id', ann['id']) not in del_track_ids)
    
    return jsonify({
        'frame_idx': frame_idx,
        'visible_count': visible_count,
        'total_count': total_count,
        'deleted_count': len(del_track_ids)
    })

@app.route('/api/metadata')
def get_metadata():
    """获取元数据"""
    temp_path = Path(TEMP_DATA_DIR)
    annotations_path = temp_path / "annotations.json"
    
    if annotations_path.exists():
        with open(annotations_path) as f:
            coco_data = json.load(f)
        
        video_info = coco_data.get('info', {})
        total_frames = len(coco_data.get('images', []))
        
        return jsonify({
            'video_width': video_info.get('width', VIDEO_WIDTH),
            'video_height': video_info.get('height', VIDEO_HEIGHT),
            'fps': video_info.get('fps', 30),
            'total_frames': total_frames
        })
    
    return jsonify({
        'video_width': VIDEO_WIDTH,
        'video_height': VIDEO_HEIGHT,
        'fps': 30,
        'total_frames': 0
    })

@app.route('/api/click', methods=['POST'])
def handle_click():
    """处理点击"""
    data = request.json
    x = int(data['x'])
    y = int(data['y'])
    frame_idx = int(data['frame_idx'])
    
    labels_dir = Path(TEMP_DATA_DIR) / "labels"
    label_path = labels_dir / f"frame_{frame_idx:06d}.json"
    
    if not label_path.exists():
        return jsonify({'success': False, 'message': '未找到标注文件'})
    
    with open(label_path) as f:
        annotations = json.load(f)
    
    del_track_ids = session.get('del_track_ids', [])
    del_points = session.get('del_points', [])
    conf_threshold = session.get('conf_threshold', DEFAULT_CONF_THRESHOLD)
    
    for ann in annotations:
        if ann.get('confidence', 1.0) < conf_threshold:
            continue
        
        track_id = ann.get('track_id', ann['id'])
        if track_id in del_track_ids:
            continue
        
        polygon = ann['segmentation'][0]
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
        
        if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
            del_track_ids.append(track_id)
            session['del_track_ids'] = del_track_ids
            
            del_point = {
                'x': x,
                'y': y,
                'frame_idx': frame_idx,
                'track_id': track_id,
                'shortcut': f'P{len(del_points) + 1}'
            }
            del_points.append(del_point)
            session['del_points'] = del_points
            
            return jsonify({
                'success': True,
                'track_id': track_id,
                'message': f'已删除 track_id={track_id}'
            })
    
    return jsonify({'success': False, 'message': '未找到标注'})

@app.route('/api/clear', methods=['POST'])
def clear_all():
    """清空所有删除点"""
    session['del_track_ids'] = []
    session['del_points'] = []
    return jsonify({'success': True})

@app.route('/api/remove_point', methods=['POST'])
def remove_point():
    """移除指定的删除点"""
    data = request.json
    index = int(data['index'])
    
    del_track_ids = session.get('del_track_ids', [])
    del_points = session.get('del_points', [])
    
    if 0 <= index < len(del_points):
        removed = del_points.pop(index)
        if removed['track_id'] in del_track_ids:
            del_track_ids.remove(removed['track_id'])
        
        session['del_points'] = del_points
        session['del_track_ids'] = del_track_ids
        
        return jsonify({'success': True})
    
    return jsonify({'success': False, 'message': '索引无效'})

@app.route('/api/set_conf', methods=['POST'])
def set_conf():
    """设置置信度阈值"""
    data = request.json
    conf = float(data['threshold'])
    session['conf_threshold'] = conf
    return jsonify({'success': True, 'threshold': conf})

@app.route('/api/del_points')
def get_del_points():
    """获取所有删除点"""
    return jsonify(session.get('del_points', []))

@app.route('/api/export', methods=['POST'])
def export_video():
    """导出视频"""
    temp_path = Path(TEMP_DATA_DIR)
    dst_path = Path(DST_DIR)
    output_path = dst_path / "output_annotated.mp4"
    
    dst_path.mkdir(exist_ok=True)
    
    annotations_path = temp_path / "annotations.json"
    if annotations_path.exists():
        with open(annotations_path) as f:
            coco_data = json.load(f)
        total_frames = len(coco_data.get('images', []))
    else:
        total_frames = 0
    
    frames_dir = temp_path / "frames"
    labels_dir = temp_path / "labels"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 30, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    del_track_ids = session.get('del_track_ids', [])
    conf_threshold = session.get('conf_threshold', DEFAULT_CONF_THRESHOLD)
    
    mask_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    
    for i in range(total_frames):
        frame_path = frames_dir / f"frame_{i:06d}.jpg"
        label_path = labels_dir / f"frame_{i:06d}.json"
        
        if frame_path.exists():
            frame = cv2.imread(str(frame_path))
        else:
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        
        if label_path.exists():
            with open(label_path) as f:
                annotations = json.load(f)
            
            for ann in annotations:
                if ann.get('confidence', 1.0) < conf_threshold:
                    continue
                
                track_id = ann.get('track_id', ann['id'])
                if track_id in del_track_ids:
                    continue
                
                polygon = ann['segmentation'][0]
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                color = mask_colors[ann.get('category_id', 0) % len(mask_colors)]
                
                cv2.fillPoly(frame, [pts], color)
                cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
                
                bbox = ann['bbox']
                x, y = int(bbox[0]), int(bbox[1])
                text = f"ID:{track_id} {ann.get('confidence', 1.0):.2f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    
    return jsonify({
        'success': True,
        'output_path': str(output_path)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("视频标注工具 Web版")
    print("=" * 60)
    print(f"访问地址: http://localhost:8082")
    print("按 Ctrl+C 退出")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8082, debug=True)
