#!/usr/bin/env python3
"""共享视频控制逻辑 - 可被 app.py, video_viewer.py, control_panel.py 共用"""

import json
import cv2
import numpy as np

CONF_THRESHOLD_DEFAULT = 0.5
ALPHA_DEFAULT = 0.5
CATEGORY_DEFAULT = "Detect"
FENCE_COLORS = [(0, 255, 0), (255, 165, 0), (255, 0, 255)]
MASK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255)
]
TRACK_ID_9999_COLOR = (128, 0, 128)
GREEN_POINT_COLOR = (0, 255, 0)


class VideoController:

    def __init__(self):
        self.conf_threshold = CONF_THRESHOLD_DEFAULT
        self.alpha = ALPHA_DEFAULT
        self.fences = []
        self.track_id_points = []
        self.track_ids_to_9999 = set()
        self.category_name = CATEGORY_DEFAULT

    def get_track_id_points(self):
        return list(self.track_id_points)

    def filter_annotations(self, annotations):
        if not annotations:
            return []
        deleted_ids = self.track_ids_to_9999 | {9999}

        fence_pts_list = []
        for fence in self.fences:
            if len(fence['points']) >= 3:
                fence_pts_list.append(np.array(fence['points'], dtype=np.int32))

        filtered = []
        for ann in annotations:
            track_id = ann.get('track_id', ann.get('id', 0))
            conf = ann.get('confidence', 1.0)

            if conf < self.conf_threshold:
                continue
            if track_id in deleted_ids:
                continue

            if fence_pts_list:
                bbox = ann.get('bbox')
                if bbox:
                    cx = bbox[0] + bbox[2] / 2
                    cy = bbox[1] + bbox[3] / 2
                    inside_any = False
                    for fence_pts in fence_pts_list:
                        if cv2.pointPolygonTest(fence_pts, (cx, cy), False) >= 0:
                            inside_any = True
                            break
                    if not inside_any:
                        continue

            filtered.append(ann)

        return filtered

    def apply_threshold_to_masks(self, frame, annotations):
        result_frame = frame.copy()
        if not annotations:
            return result_frame

        overlay = frame.copy()

        for ann in annotations:
            polygon = ann.get('segmentation')
            bbox = ann.get('bbox')
            if not bbox:
                continue

            color = MASK_COLORS[ann.get('category_id', 0) % len(MASK_COLORS)]
            if ann.get('track_id', 0) == 9999:
                color = TRACK_ID_9999_COLOR
            category = ann.get('category', ann.get('category_id', 0))
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

        for i, fence in enumerate(self.fences):
            if len(fence['points']) >= 3:
                color = FENCE_COLORS[i % len(FENCE_COLORS)]
                pts = np.array(fence['points'], dtype=np.int32)
                pts_array = pts.reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts_array], True, color, 3)
                for pt in fence['points']:
                    cv2.circle(overlay, pt, 5, color, -1)

        cv2.addWeighted(overlay, self.alpha, result_frame, 1 - self.alpha, 0, result_frame)
        return result_frame

    def fence_mode_active(self):
        return any(f.get('mode', False) for f in self.fences)

    def add_fence_point(self, fence_idx, point):
        if fence_idx < len(self.fences):
            self.fences[fence_idx]['points'].append(point)

    def find_annotation_at(self, annotations, video_x, video_y):
        for ann in annotations:
            polygon = ann.get('segmentation')
            if not polygon:
                continue
            pts = np.array(polygon[0], dtype=np.int32).reshape(-1, 2)
            if cv2.pointPolygonTest(pts, (float(video_x), float(video_y)), False) >= 0:
                return ann
        return None

    def add_track_id_point(self, x, y, frame_idx, track_id):
        self.track_id_points.append({
            'x': x, 'y': y,
            'frame_idx': frame_idx,
            'track_id': track_id
        })

    def remove_track_id_point(self, index):
        if 0 <= index < len(self.track_id_points):
            self.track_id_points.pop(index)

    def clear_track_id_points(self):
        self.track_id_points = []

    def toggle_fence_mode(self, fence_idx):
        if fence_idx >= len(self.fences):
            self.fences.append({'points': [], 'mode': True})
        else:
            self.fences[fence_idx]['mode'] = not self.fences[fence_idx]['mode']

    def clear_fence(self, fence_idx):
        if fence_idx < len(self.fences):
            self.fences[fence_idx]['points'] = []
            self.fences[fence_idx]['mode'] = False

    def export_filtered_annotations(self, total_frames, labels_dir, category_name=None):
        output_annotations = []
        cat_name = category_name or self.category_name

        for i in range(total_frames):
            label_path = labels_dir / f"frame_{i:06d}.json"
            if label_path.exists():
                with open(label_path) as f:
                    annotations = json.load(f)
                filtered = self.filter_annotations(annotations)
                for ann in filtered:
                    ann_copy = ann.copy()
                    ann_copy['category'] = cat_name
                    output_annotations.append(ann_copy)

        return output_annotations
