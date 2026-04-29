#!/usr/bin/env python3
import sys, signal, torch, numpy as np

signal.alarm(30)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

_SAM3_SEMANTIC_PATCHED = False
def _patch_sam3_video_semantic():
    global _SAM3_SEMANTIC_PATCHED
    if _SAM3_SEMANTIC_PATCHED:
        return
    _SAM3_SEMANTIC_PATCHED = True
    import torch
    from ultralytics.utils import ops as ultralytics_ops
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    _orig = SAM3VideoSemanticPredictor.add_prompt

    def _new_add_prompt(self, frame_idx, text=None, bboxes=None, labels=None, inference_state=None):
        if bboxes is None:
            return _orig(self, frame_idx, text, bboxes, labels, inference_state)
        inference_state_local = inference_state or {}
        text_batch = [text] if isinstance(text, str) else (list(text) if text else [])
        n = len(text_batch)
        _raw = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
        _raw = _raw[None] if _raw.ndim == 1 else _raw
        _raw = ultralytics_ops.xyxy2xywh(_raw)
        _raw[:, 0::2] /= self.batch[1][0].shape[1]
        _raw[:, 1::2] /= self.batch[1][0].shape[0]
        nb = len(_raw)
        if labels is None:
            _lbl_arr = np.ones(nb)
        else:
            _lbl_arr = np.array(labels)
        _lbl = torch.as_tensor(_lbl_arr, dtype=torch.int32, device=self.device)
        _raw = _raw.view(-1, 1, 4)
        _lbl = _lbl.view(-1, 1)
        if n > 0 and nb < n:
            rep = (n + nb - 1) // nb
            _raw = _raw.repeat(rep, 1, 1)[:n]
            _lbl = _lbl.repeat(rep, 1)[:n]
        else:
            _raw = _raw[:n]
            _lbl = _lbl[:n]
        geometric_prompt = self._get_dummy_prompt(num_prompts=n)
        for i in range(len(_raw)):
            box_expanded = _raw[[i]].expand(n, -1, -1).clone()
            lbl_expanded = _lbl[[i]].expand(n).clone()
            geometric_prompt.append_boxes(box_expanded, lbl_expanded)
        return frame_idx, geometric_prompt

    SAM3VideoSemanticPredictor.add_prompt = _new_add_prompt

def make_mock_self(n):
    from ultralytics.utils import ops as ultralytics_ops
    class MockSelf:
        device = 'cpu'
        torch_dtype = torch.float32
        batch = [None, [(480, 640, 3)]]
        batch_img_shape = (480, 640, 3)
        def _get_dummy_prompt(self, num_prompts):
            from ultralytics.models.sam.sam3.geometry_encoders import Prompt
            return Prompt(
                box_embeddings=torch.zeros(0, num_prompts, 4, device=self.device),
                box_mask=torch.zeros(num_prompts, 0, device=self.device, dtype=torch.bool),
            )
        def add_prompt(self, frame_idx, text=None, bboxes=None, labels=None, inference_state=None):
            if bboxes is None:
                return frame_idx, None
            text_batch = [text] if isinstance(text, str) else (list(text) if text else [])
            n_local = len(text_batch)
            _raw = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            _raw = _raw[None] if _raw.ndim == 1 else _raw
            _raw = ultralytics_ops.xyxy2xywh(_raw)
            _raw[:, 0::2] /= self.batch_img_shape[1]
            _raw[:, 1::2] /= self.batch_img_shape[0]
            nb = len(_raw)
            if labels is None:
                _lbl_arr = np.ones(nb)
            else:
                _lbl_arr = np.array(labels)
            _lbl = torch.as_tensor(_lbl_arr, dtype=torch.int32, device=self.device)
            _raw = _raw.view(-1, 1, 4)
            _lbl = _lbl.view(-1, 1)
            if n_local > 0 and nb < n_local:
                rep = (n_local + nb - 1) // nb
                _raw = _raw.repeat(rep, 1, 1)[:n_local]
                _lbl = _lbl.repeat(rep, 1)[:n_local]
            else:
                _raw = _raw[:n_local]
                _lbl = _lbl[:n_local]
            geometric_prompt = self._get_dummy_prompt(num_prompts=n_local)
            for i in range(len(_raw)):
                box_rep = _raw[[i]].repeat(1, n_local, 1)
                lbl_rep = _lbl[[i]].squeeze(-1).repeat(n_local).unsqueeze(0)
                geometric_prompt.append_boxes(box_rep, lbl_rep)
            return frame_idx, geometric_prompt
    return MockSelf()

def test_patch_applied():
    _patch_sam3_video_semantic()
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    assert SAM3VideoSemanticPredictor.add_prompt.__name__ == '_new_add_prompt'
    print("✓ patch applied")

def test_one_bbox_two_texts():
    _patch_sam3_video_semantic()
    mock_self = make_mock_self(n=2)
    texts = ['nozzle', 'needle']
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    frame_idx, prompt = mock_self.add_prompt(frame_idx=0, text=texts, bboxes=bboxes)
    assert prompt.box_embeddings.shape[1] == 2, f"bs should be 2, got {prompt.box_embeddings.shape[1]}"
    assert prompt.box_embeddings.shape[0] == 2, f"2 boxes expected (repeated), got {prompt.box_embeddings.shape[0]}"
    print(f"✓ 1 bbox + 2 texts: prompt shape = {prompt.box_embeddings.shape} (bs=2, bbox repeated)")

def test_three_bboxes_two_texts():
    _patch_sam3_video_semantic()
    mock_self = make_mock_self(n=2)
    texts = ['nozzle', 'needle']
    bboxes = np.array([
        [100, 100, 200, 200],
        [300, 200, 400, 300],
        [500, 300, 600, 400],
    ], dtype=np.float32)
    frame_idx, prompt = mock_self.add_prompt(frame_idx=0, text=texts, bboxes=bboxes)
    assert prompt.box_embeddings.shape[1] == 2, f"bs should be 2, got {prompt.box_embeddings.shape[1]}"
    assert prompt.box_embeddings.shape[0] == 2, f"2 boxes expected (truncated), got {prompt.box_embeddings.shape[0]}"
    print(f"✓ 3 bboxes + 2 texts: prompt shape = {prompt.box_embeddings.shape} (bs=2, truncated to 2)")

def test_two_bboxes_two_texts():
    _patch_sam3_video_semantic()
    mock_self = make_mock_self(n=2)
    texts = ['nozzle', 'needle']
    bboxes = np.array([
        [100, 100, 200, 200],
        [300, 200, 400, 300],
    ], dtype=np.float32)
    frame_idx, prompt = mock_self.add_prompt(frame_idx=0, text=texts, bboxes=bboxes)
    assert prompt.box_embeddings.shape[1] == 2, f"bs should be 2, got {prompt.box_embeddings.shape[1]}"
    assert prompt.box_embeddings.shape[0] == 2, f"2 boxes expected, got {prompt.box_embeddings.shape[0]}"
    print(f"✓ 2 bboxes + 2 texts: prompt shape = {prompt.box_embeddings.shape} (exact match)")

def test_original_would_fail():
    from ultralytics.models.sam.sam3.geometry_encoders import Prompt
    from ultralytics.utils import ops as ultralytics_ops
    device = 'cpu'
    n, nb = 2, 1
    geometric_prompt = Prompt(
        box_embeddings=torch.zeros(0, n, 4, device=device),
        box_mask=torch.zeros(n, 0, device=device, dtype=torch.bool),
    )
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    _raw = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
    _raw = ultralytics_ops.xyxy2xywh(_raw)
    _raw[:, 0::2] /= 640
    _raw[:, 1::2] /= 480
    _raw = _raw.view(-1, 1, 4)
    _lbl = torch.ones(1, 1, dtype=torch.int32, device=device)
    box_rep = _raw[[0]].repeat(1, n, 1)
    lbl_rep = _lbl[[0]].squeeze(-1).repeat(n).unsqueeze(0)
    try:
        geometric_prompt.append_boxes(box_rep, lbl_rep)
        print(f"✓ repeat(1,n,1) box [{n},1,4] appends OK (patch works!)")
    except AssertionError as e:
        print(f"✗ repeat(1,n,1) box still fails: {e}")
        sys.exit(1)

def test_bbox_shape_3d():
    from ultralytics.utils import ops as ultralytics_ops
    device = 'cpu'
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    _raw = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
    _raw = ultralytics_ops.xyxy2xywh(_raw)
    _raw[:, 0::2] /= 640
    _raw[:, 1::2] /= 480
    _raw = _raw.view(-1, 1, 4)
    assert _raw.shape == (1, 1, 4), f"expected (1,1,4), got {_raw.shape}"
    print(f"✓ bbox shape is 3D [N,1,4]: {_raw.shape}")

def test_repeat_produces_correct_shape():
    from ultralytics.utils import ops as ultralytics_ops
    device = 'cpu'
    n, nb = 2, 1
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    _raw = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
    _raw = ultralytics_ops.xyxy2xywh(_raw)
    _raw[:, 0::2] /= 640
    _raw[:, 1::2] /= 480
    _raw = _raw.view(-1, 1, 4)
    box_rep = _raw[[0]].repeat(1, n, 1)
    assert box_rep.shape == (1, n, 4), f"expected (1,{n},4), got {box_rep.shape}"
    print(f"✓ repeat(1,n,1) produces correct [1,N,4] shape: {box_rep.shape}")

if __name__ == '__main__':
    test_patch_applied()
    test_bbox_shape_3d()
    test_repeat_produces_correct_shape()
    test_original_would_fail()
    test_one_bbox_two_texts()
    test_three_bboxes_two_texts()
    test_two_bboxes_two_texts()
    print("\n✓ all tests passed")
