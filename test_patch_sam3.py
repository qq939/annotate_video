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
        inference_state = inference_state or self.inference_state
        text_batch = [text] if isinstance(text, str) else (list(text) if text else [])
        n = len(text_batch)
        inference_state["text_prompt"] = text if text else None
        text_ids = torch.arange(n, device=self.device, dtype=torch.long)
        inference_state["text_ids"] = text_ids
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)
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
        if n > 1:
            _raw = _raw.repeat(1, n, 1)
            _lbl = _lbl.repeat(1, n)
        geometric_prompt = self._get_dummy_prompt(num_prompts=n)
        for i in range(len(_raw)):
            geometric_prompt.append_boxes(_raw[[i]], _lbl[[i]])
        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, out

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
            if n_local > 1:
                _raw = _raw.repeat(1, n_local, 1)
                _lbl = _lbl.repeat(1, n_local)
            geometric_prompt = self._get_dummy_prompt(num_prompts=n_local)
            for i in range(len(_raw)):
                geometric_prompt.append_boxes(_raw[[i]], _lbl[[i]])
            return frame_idx, geometric_prompt
    return MockSelf()

def test_patch_applied():
    _patch_sam3_video_semantic()
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
    assert SAM3VideoSemanticPredictor.add_prompt.__name__ == '_new_add_prompt'
    print("✓ patch applied")

def test_1text_1bbox():
    _patch_sam3_video_semantic()
    m = make_mock_self(n=1)
    _, p = m.add_prompt(0, text=['obj'], bboxes=np.array([[100,100,200,200]]))
    assert p.box_embeddings.shape == (1, 1, 4), f"expected (1,1,4), got {p.box_embeddings.shape}"
    print(f"✓ 1text 1bbox: {p.box_embeddings.shape}")

def test_2text_1bbox():
    _patch_sam3_video_semantic()
    m = make_mock_self(n=2)
    _, p = m.add_prompt(0, text=['a','b'], bboxes=np.array([[100,100,200,200]]))
    assert p.box_embeddings.shape == (1, 2, 4), f"expected (1,2,4), got {p.box_embeddings.shape}"
    print(f"✓ 2text 1bbox: {p.box_embeddings.shape}")

def test_2text_2bbox():
    _patch_sam3_video_semantic()
    m = make_mock_self(n=2)
    _, p = m.add_prompt(0, text=['a','b'],
        bboxes=np.array([[100,100,200,200],[300,300,400,400]]))
    assert p.box_embeddings.shape == (2, 2, 4), f"expected (2,2,4), got {p.box_embeddings.shape}"
    print(f"✓ 2text 2bbox: {p.box_embeddings.shape}")

def test_3text_2bbox():
    _patch_sam3_video_semantic()
    m = make_mock_self(n=3)
    _, p = m.add_prompt(0, text=['a','b','c'],
        bboxes=np.array([[100,100,200,200],[300,300,400,400]]))
    assert p.box_embeddings.shape == (2, 3, 4), f"expected (2,3,4), got {p.box_embeddings.shape}"
    print(f"✓ 3text 2bbox: {p.box_embeddings.shape}")

def test_original_would_fail():
    from ultralytics.models.sam.sam3.geometry_encoders import Prompt
    device = 'cpu'
    n, nb = 2, 1
    prompt = Prompt(
        box_embeddings=torch.zeros(0, n, 4, device=device),
        box_mask=torch.zeros(n, 0, device=device, dtype=torch.bool),
    )
    box = torch.zeros(nb, 1, 4)
    lbl = torch.ones(nb, 1, dtype=torch.int32)
    try:
        prompt.append_boxes(box, lbl)
        print(f"✗ should have failed")
        sys.exit(1)
    except AssertionError as e:
        print(f"✓ original bug confirmed: {e}")

if __name__ == '__main__':
    test_patch_applied()
    test_original_would_fail()
    test_1text_1bbox()
    test_2text_1bbox()
    test_2text_2bbox()
    test_3text_2bbox()
    print("\n✓ all tests passed")
