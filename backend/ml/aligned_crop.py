"""
Landmark-aligned face cropping that mirrors the upstream DFD-FCG preprocessing
(src/preprocess/fetch_landmark_bbox.py + src/preprocess/crop_main_face.py).

The released DFD-FCG checkpoint was trained on faces that were
  1. landmarked with 2D-FAN (68 points, S3FD detector),
  2. temporally smoothed over a small window,
  3. affine-warped onto the 20-words mean face at 256x256 using 8 stable points,
  4. centre-cropped to 150x150 around landmarks[15:68].
Its facial-component queries (lips / skin / eyes / nose) assume that geometry.
Feeding it loose Haar-cascade boxes instead (the previous behaviour) shifts the
face parts away from where the model expects them and destroys accuracy.
"""
from __future__ import annotations

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

STABLE_POINTS = (28, 33, 36, 39, 42, 45, 48, 54)
TARGET_SIZE = 256
CROP_SIZE = 150
START_IDX, STOP_IDX = 15, 68
MAX_RES = 800  # upstream fetch_landmark_bbox default


class AlignedFaceCropper:
    """Lazily builds a 2D-FAN landmark detector and produces DFD-FCG crops."""

    _lock = threading.Lock()

    def __init__(self, mean_face: np.ndarray, device: str = "cpu"):
        self.reference = np.asarray(mean_face, dtype=np.float64)
        self.device = device
        self._fa = None
        self.available: Optional[bool] = None
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ #
    def _detector(self):
        if self._fa is not None or self.available is False:
            return self._fa
        with self._lock:
            if self._fa is None and self.available is not False:
                try:
                    import face_alignment

                    lt = getattr(face_alignment.LandmarksType, "TWO_D", None)
                    if lt is None:  # face_alignment < 1.4
                        lt = face_alignment.LandmarksType._2D
                    self._fa = face_alignment.FaceAlignment(
                        lt, device=self.device, face_detector="sfd", flip_input=False
                    )
                    self.available = True
                except Exception as e:  # pragma: no cover - depends on env
                    self.available = False
                    self.error = repr(e)
                    print(f"[AlignedFaceCropper] 2D-FAN unavailable: {e!r}")
        return self._fa

    def is_available(self) -> bool:
        self._detector()
        return bool(self.available)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _scale_for(frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        return MAX_RES / max(h, w) if max(h, w) > MAX_RES else 1.0

    def _landmarks(
        self, frame_rgb: np.ndarray, bbox: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (68x2 landmarks, 2x2 bbox) of the largest face in full-res coords."""
        fa = self._detector()
        if fa is None:
            return None, None
        s = self._scale_for(frame_rgb)
        small = cv2.resize(frame_rgb, None, fx=s, fy=s) if s != 1.0 else frame_rgb
        det = None
        if bbox is not None:
            b = bbox.reshape(-1) * s
            det = [np.array([b[0], b[1], b[2], b[3], 1.0])]
        try:
            res = fa.get_landmarks_from_image(small, detected_faces=det, return_bboxes=True)
        except TypeError:  # older API without return_bboxes
            res = (fa.get_landmarks_from_image(small, detected_faces=det), None, det)
        if isinstance(res, tuple):
            lms, _, boxes = (list(res) + [None, None])[:3]
        else:
            lms, boxes = res, det
        if not lms:
            return None, None
        if boxes is None or len(boxes) != len(lms):
            boxes = [np.array([l[:, 0].min(), l[:, 1].min(), l[:, 0].max(), l[:, 1].max()]) for l in lms]
        sizes = [np.linalg.norm(np.asarray(b[:2]) - np.asarray(b[2:4])) for b in boxes]
        k = int(np.argmax(sizes))
        lm = np.asarray(lms[k], dtype=np.float64)[:, :2] / s
        bb = np.asarray(boxes[k][:4], dtype=np.float64).reshape(2, 2) / s
        return lm, bb

    # ------------------------------------------------------------------ #
    def _affine(self, frame, bboxes, landmarks):
        ref = np.vstack([self.reference[x] for x in STABLE_POINTS]).copy()
        ref[:, 0] *= TARGET_SIZE / 256
        ref[:, 1] *= TARGET_SIZE / 256
        tf = cv2.estimateAffinePartial2D(
            np.vstack([landmarks[x] for x in STABLE_POINTS]), ref, method=cv2.LMEDS
        )[0]
        if tf is None:
            return None, None, None
        out = cv2.warpAffine(
            frame, tf, dsize=(TARGET_SIZE, TARGET_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        t_lm = landmarks @ tf[:, :2].T + tf[:, 2]
        t_bb = bboxes @ tf[:, :2].T + tf[:, 2]
        return out, t_lm, t_bb

    @staticmethod
    def _crop(img, landmarks, size):
        cx, cy = np.mean(landmarks[START_IDX:STOP_IDX], axis=0)
        cy = min(max(cy, size + 1), img.shape[0] - size - 1)
        cx = min(max(cx, size + 1), img.shape[1] - size - 1)
        uy, lx = int(cy - size), int(cx - size)
        crop = np.copy(img[uy:uy + 2 * size, lx:lx + 2 * size])
        return crop, landmarks - [lx, uy]

    # ------------------------------------------------------------------ #
    def crop_frames(
        self,
        frames: Sequence[np.ndarray],
        indices: Sequence[int],
        smooth_offsets: Sequence[int] = (-3, 0, 3),
    ) -> Dict[int, Tuple[np.ndarray, Optional[np.ndarray], bool]]:
        """
        Produces aligned 150x150 RGB crops for every frame index in ``indices``.

        Landmarks are averaged over ``smooth_offsets`` neighbours (an approximation
        of upstream's +/-6 frame window that keeps CPU cost bounded) and then
        re-centred on the current frame's landmarks exactly as upstream does.

        Returns {frame_idx: (crop_rgb, crop_landmarks or None, face_found)}.
        """
        n = len(frames)
        need = sorted({min(max(i + o, 0), n - 1) for i in indices for o in smooth_offsets})
        cache: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        # Detect boxes with S3FD on the anchor frames only; reuse them for neighbours.
        anchor_box: Dict[int, Optional[np.ndarray]] = {}
        for i in sorted(set(indices)):
            lm, bb = self._landmarks(frames[i])
            cache[i] = (lm, bb)
            anchor_box[i] = bb
        for j in need:
            if j in cache:
                continue
            nearest = min(anchor_box, key=lambda a: abs(a - j))
            cache[j] = self._landmarks(frames[j], bbox=anchor_box[nearest])

        # Fill frames where the face was missed from the nearest successful frame.
        good = [k for k, v in cache.items() if v[0] is not None]
        out: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], bool]] = {}
        for i in indices:
            lm_i, bb_i = cache.get(i, (None, None))
            found = lm_i is not None
            if not found:
                if not good:
                    out[i] = (np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8), None, False)
                    continue
                k = min(good, key=lambda g: abs(g - i))
                lm_i, bb_i = cache[k]
            neigh = [cache[min(max(i + o, 0), n - 1)] for o in smooth_offsets]
            neigh = [v for v in neigh if v[0] is not None] or [(lm_i, bb_i)]
            s_lm = np.mean([v[0] for v in neigh], axis=0)
            s_lm += lm_i.mean(axis=0) - s_lm.mean(axis=0)
            s_bb = np.mean([v[1] for v in neigh], axis=0)
            s_bb += bb_i.mean(axis=0) - s_bb.mean(axis=0)
            warped, t_lm, _ = self._affine(frames[i], s_bb, s_lm)
            if warped is None:
                out[i] = (np.zeros((CROP_SIZE, CROP_SIZE, 3), np.uint8), None, False)
                continue
            crop, c_lm = self._crop(warped, t_lm, CROP_SIZE // 2)
            out[i] = (crop, c_lm, found)
        return out
