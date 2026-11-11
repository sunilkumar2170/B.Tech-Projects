"""Compare old Haar-box preprocessing vs new landmark-aligned preprocessing.

Usage (from backend/):  python tests/compare_preprocessing.py [--device cuda] [--clips 16] [videos...]
Video label is inferred from FaceForensics++ naming: '000.mp4' = REAL, '000_003.mp4' = FAKE.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from app.core.config import settings  # noqa: E402  (also puts xai on sys.path)
from ml.aligned_crop import AlignedFaceCropper  # noqa: E402
from ml.face_detector import FaceDetector  # noqa: E402
from ml.preprocessing import read_video_frames, compute_clip_indices  # noqa: E402


@torch.inference_mode()
def probs_for(model, tf, clips_uint8, device):
    out = []
    for c in clips_uint8:  # c: (T,3,H,W) uint8
        x = tf(c).unsqueeze(0).to(device)
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                r = model.evaluate(x)
        else:
            r = model.evaluate(x)
        out.append(float(r["logits"].float().softmax(-1)[0, 1]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clips", type=int, default=16)
    a = ap.parse_args()
    vids = a.videos or [str(settings.SAMPLES_DIR / "000.mp4"), str(settings.SAMPLES_DIR / "000_003.mp4")]

    from xai.engine import load_model
    t0 = time.time()
    model = load_model(str(settings.CHECKPOINT_WEIGHTS), a.device, attn_record=False)
    tf = model.transform
    print(f"model loaded in {time.time()-t0:.1f}s on {a.device}")

    mean_face = np.load(BACKEND / "misc" / "20words_mean_face.npy")
    haar = FaceDetector(mean_face_path=str(BACKEND / "misc" / "20words_mean_face.npy"))
    aligned = AlignedFaceCropper(mean_face, device=a.device)

    rows = []
    for v in vids:
        name = Path(v).stem
        label = 1 if "_" in name else 0
        frames, fps, _, _ = read_video_frames(Path(v))
        offsets, starts = compute_clip_indices(len(frames), fps, 10, 0.333, a.clips)
        idx = sorted({min(s + o, len(frames) - 1) for s in starts for o in offsets})

        t1 = time.time()
        crops_new = aligned.crop_frames(frames, idx)
        t_align = time.time() - t1
        to_t = lambda arr: torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()

        old_clips, new_clips = [], []
        for s in starts:
            fi = [min(s + o, len(frames) - 1) for o in offsets]
            old_clips.append(to_t([haar.crop_and_align_face(frames[f], crop_size=150)[0] for f in fi]))
            new_clips.append(to_t([crops_new[f][0] for f in fi]))
        p_old = probs_for(model, tf, old_clips, a.device)
        p_new = probs_for(model, tf, new_clips, a.device)
        found = np.mean([crops_new[f][2] for f in idx])
        rows.append((name, label, p_old.mean(), p_new.mean(), found, t_align))
        print(f"{name:>10} label={'FAKE' if label else 'REAL'}  old P(fake)={p_old.mean():.3f}  "
              f"new P(fake)={p_new.mean():.3f}  faces={found:.0%}  align={t_align:.1f}s")

    acc = lambda col: np.mean([(r[col] >= 0.5) == r[1] for r in rows])
    print(f"\naccuracy@0.5  old={acc(2):.2%}  new={acc(3):.2%}  (n={len(rows)})")
    if len({r[1] for r in rows}) == 2:
        from sklearn.metrics import roc_auc_score
        y = [r[1] for r in rows]
        print(f"AUC  old={roc_auc_score(y, [r[2] for r in rows]):.3f}  new={roc_auc_score(y, [r[3] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
