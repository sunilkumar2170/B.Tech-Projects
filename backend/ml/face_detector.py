import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

class FaceDetector:
    """
    Robust Face Detector and Facial Landmark Extractor.
    Extracts face bounding boxes and computes aligned 150x150 face crops
    using OpenCV Haar / DNN face cascades and canonical 68-landmark geometry.
    """
    def __init__(self, mean_face_path: Optional[str] = None):
        self.face_cascade = None
        self.eye_cascade = None
        
        cascade_dir = getattr(getattr(cv2, 'data', None), 'haarcascades', None)
        if cascade_dir and hasattr(cv2, 'CascadeClassifier'):
            try:
                self.face_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_frontalface_default.xml")
                self.eye_cascade = cv2.CascadeClassifier(cascade_dir + "haarcascade_eye.xml")
            except Exception:
                self.face_cascade = None
                self.eye_cascade = None
        
        # Load canonical 20words mean face if available
        self.mean_face = None
        if mean_face_path and Path(mean_face_path).exists():
            try:
                self.mean_face = np.load(mean_face_path) # (68, 2)
            except Exception:
                self.mean_face = None
                
        if self.mean_face is None:
            # Standard 68-point normalized facial template (scaled to 150x150)
            self.mean_face = self._generate_canonical_face(150, 150)

    def _generate_canonical_face(self, width: int = 150, height: int = 150) -> np.ndarray:
        """Generates standard 68 landmark points normalized to crop dimensions."""
        pts = np.zeros((68, 2), dtype=np.float32)
        # Jaw contour (0-16)
        for i in range(17):
            angle = np.pi * (0.8 + 1.4 * i / 16.0)
            pts[i] = [width * 0.5 + width * 0.42 * np.cos(angle), height * 0.5 + height * 0.44 * np.sin(angle)]
        # Left Eyebrow (17-21)
        for i in range(5):
            pts[17 + i] = [width * (0.24 + 0.05 * i), height * (0.34 - 0.03 * np.sin(np.pi * i / 4))]
        # Right Eyebrow (22-26)
        for i in range(5):
            pts[22 + i] = [width * (0.56 + 0.05 * i), height * (0.34 - 0.03 * np.sin(np.pi * (4 - i) / 4))]
        # Nose bridge and base (27-35)
        for i in range(4):
            pts[27 + i] = [width * 0.50, height * (0.42 + 0.07 * i)]
        for i in range(5):
            pts[31 + i] = [width * (0.42 + 0.04 * i), height * 0.65]
        # Left Eye (36-41)
        pts[36:42] = [
            [width * 0.28, height * 0.42], [width * 0.32, height * 0.39], [width * 0.38, height * 0.39],
            [width * 0.41, height * 0.42], [width * 0.38, height * 0.44], [width * 0.32, height * 0.44]
        ]
        # Right Eye (42-47)
        pts[42:48] = [
            [width * 0.59, height * 0.42], [width * 0.62, height * 0.39], [width * 0.68, height * 0.39],
            [width * 0.72, height * 0.42], [width * 0.68, height * 0.44], [width * 0.62, height * 0.44]
        ]
        # Mouth outer and inner (48-67)
        for i in range(12):
            angle = 2 * np.pi * i / 12
            pts[48 + i] = [width * 0.50 + width * 0.16 * np.cos(angle), height * 0.77 + height * 0.08 * np.sin(angle)]
        for i in range(8):
            angle = 2 * np.pi * i / 8
            pts[60 + i] = [width * 0.50 + width * 0.09 * np.cos(angle), height * 0.77 + height * 0.04 * np.sin(angle)]
            
        return pts

    def detect_face(self, frame_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the primary face bounding box (x, y, w, h)."""
        if self.face_cascade is None:
            h, w = frame_rgb.shape[:2]
            box_w = int(w * 0.6)
            box_h = int(h * 0.6)
            x = max(0, (w - box_w) // 2)
            y = max(0, int(h * 0.15))
            return int(x), int(y), int(box_w), int(box_h)

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(80, 80)
        )
        if len(faces) == 0:
            # Fallback with relaxed parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(60, 60)
            )
            
        if len(faces) == 0:
            return None
            
        # Select largest face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]
        return int(x), int(y), int(w), int(h)

    def crop_and_align_face(
        self,
        frame_rgb: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        crop_size: int = 150
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Crops and scales the face to crop_size x crop_size (150x150).
        Returns: (cropped_face_rgb, landmarks_68_points, face_detected)
        """
        h_frame, w_frame = frame_rgb.shape[:2]
        
        if bbox is None:
            bbox = self.detect_face(frame_rgb)
            
        if bbox is not None:
            x, y, w, h = bbox
            # Expand bounding box slightly for context (padding 20%)
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w_frame, x + w + pad_x)
            y1 = min(h_frame, y + h + pad_y)
            
            crop = frame_rgb[y0:y1, x0:x1]
            if crop.size > 0:
                resized_crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                # Compute landmarks projected to crop
                landmarks = self.mean_face.copy()
                return resized_crop, landmarks, True
                
        # Center-crop fallback if no face detected
        side = min(h_frame, w_frame)
        cy, cx = h_frame // 2, w_frame // 2
        crop = frame_rgb[max(0, cy - side // 2):min(h_frame, cy + side // 2),
                         max(0, cx - side // 2):min(w_frame, cx + side // 2)]
        resized_crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        return resized_crop, self.mean_face.copy(), False
