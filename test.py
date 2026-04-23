from ultralytics import settings
settings.update({
    "runs_dir": "/Users/jimjiang/Downloads/laliu/runs"  # 强制锁死你的目录
}) # 强制切换到当前脚本目录
# 强制锁死在你当前的 laliu 文件夹里
BASE_DIR = "/Users/jimjiang/Downloads/laliu"
from ultralytics.models.sam.predict import SAM3SemanticPredictor
import clip
print(clip.__file__)
# Initialize predictor with configuration
overrides = dict(conf=0.25, task="segment", mode="predict", model="sam3.pt", half=False)
overrides.update(project="image", save_txt=True)
topk = 1
predictor = SAM3SemanticPredictor(overrides=overrides)
_postprocess = predictor.postprocess

def postprocess(preds, img, orig_imgs, *, _k=topk, _f=_postprocess):
    res = _f(preds, img, orig_imgs)
    out = []
    for r in res:
        if r.boxes is not None and len(r.boxes):
            idx = r.boxes.conf.argsort(descending=True)[:_k]
            r = r[idx]
        out.append(r)
    return out

predictor.postprocess = postprocess
# Required for text enc
# Set image once for multiple queries
predictor.set_image ("/Users/jimjiang/Downloads/laliu/streaming/last.jpg")

# Query with multiple text prompts
results = predictor (text=["metal plate", "bottle"], save=True)
results[0].show()
