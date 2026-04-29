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
        text_batch = [text] if isinstance(text, str) else (list(text) if text else [])
        n = len(text_batch)
        _raw = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
        _raw = _raw[None] if _raw.ndim == 1 else _raw
        _raw = ultralytics_ops.xyxy2xywh(_raw)
        _raw[:, 0::2] /= self.batch[1][0].shape[1]
        _raw[:, 1::2] /= self.batch[1][0].shape[0]
        nb = len(_raw)
        if labels is None:
            _lbl = torch.ones(nb, device=self.device, dtype=torch.int32)
        else:
            _lbl = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
        _raw = _raw.view(-1, 1, 4)
        _lbl = _lbl.view(-1, 1)
        if n > 0 and nb < n:
            rep = (n + nb - 1) // nb
            _raw = _raw.repeat(rep, 1, 1)[:n]
            _lbl = _lbl.repeat(rep, 1)[:n]
        return _orig(self, frame_idx, text, _raw, _lbl, inference_state)

    SAM3VideoSemanticPredictor.add_prompt = _new_add_prompt

def test_patch_applied():
    _patch_sam3_video_semantic()
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    assert SAM3VideoSemanticPredictor.add_prompt.__name__ == '_new_add_prompt', \
        f"patch failed: method name={SAM3VideoSemanticPredictor.add_prompt.__name__}"
    print("✓ patch applied to SAM3VideoSemanticPredictor.add_prompt")

def test_bbox_expansion_logic():
    import torch
    from ultralytics.utils import ops as ultralytics_ops
    device = 'cpu'
    text_batch = ['nozzle', 'needle']
    n = len(text_batch)
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    _raw = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
    _raw = _raw[None] if _raw.ndim == 1 else _raw
    _raw = ultralytics_ops.xyxy2xywh(_raw)
    _raw[:, 0::2] /= 640
    _raw[:, 1::2] /= 480
    nb = len(_raw)
    assert nb == 1, f"expected 1 bbox, got {nb}"
    _lbl = torch.ones(nb, device=device, dtype=torch.int32)
    _raw = _raw.view(-1, 1, 4)
    _lbl = _lbl.view(-1, 1)
    if n > 0 and nb < n:
        rep = (n + nb - 1) // nb
        _raw = _raw.repeat(rep, 1, 1)[:n]
        _lbl = _lbl.repeat(rep, 1)[:n]
    assert _raw.shape == (2, 1, 4), f"expected (2,1,4), got {_raw.shape}"
    print(f"✓ bbox expanded: 1 bbox -> shape {_raw.shape} to match 2 text prompts")

def test_no_expansion_when_equal():
    import torch
    from ultralytics.utils import ops as ultralytics_ops
    device = 'cpu'
    text_batch = ['nozzle']
    n = len(text_batch)
    bboxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    _raw = torch.as_tensor(bboxes, dtype=torch.float32, device=device)
    _raw = _raw[None] if _raw.ndim == 1 else _raw
    _raw = ultralytics_ops.xyxy2xywh(_raw)
    _raw[:, 0::2] /= 640
    _raw[:, 1::2] /= 480
    nb = len(_raw)
    _lbl = torch.ones(nb, device=device, dtype=torch.int32)
    _raw = _raw.view(-1, 1, 4)
    _lbl = _lbl.view(-1, 1)
    if n > 0 and nb < n:
        rep = (n + nb - 1) // nb
        _raw = _raw.repeat(rep, 1, 1)[:n]
        _lbl = _lbl.repeat(rep, 1)[:n]
    assert _raw.shape == (1, 1, 4), f"expected (1,1,4), got {_raw.shape}"
    print(f"✓ no expansion when n==nb: shape stays {_raw.shape}")

def test_original_append_boxes_fails_without_patch():
    from ultralytics.models.sam.sam3.geometry_encoders import Prompt
    from ultralytics.utils import ops as ultralytics_ops
    device = 'cpu'
    prompt = Prompt(
        box_embeddings=torch.zeros(0, 2, 4, device=device),
        box_mask=torch.zeros(2, 0, device=device, dtype=torch.bool),
    )
    bbox = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32, device=device)
    bbox = ultralytics_ops.xyxy2xywh(bbox)
    bbox[:, 0::2] /= 640
    bbox[:, 1::2] /= 480
    bbox = bbox.view(-1, 1, 4)
    label = torch.ones(1, 1, dtype=torch.int32, device=device)
    try:
        prompt.append_boxes(bbox, label)
        print("✗ should have failed but didn't")
        sys.exit(1)
    except AssertionError as e:
        print(f"✓ original bug confirmed: {e}")

if __name__ == '__main__':
    test_patch_applied()
    test_bbox_expansion_logic()
    test_no_expansion_when_equal()
    test_original_append_boxes_fails_without_patch()
    print("\n✓ all tests passed")
