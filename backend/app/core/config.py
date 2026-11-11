import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import sys
from pathlib import Path
from pydantic import BaseModel, Field

from typing import Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BASE_DIR.parent
PARENT_DIR = WORKSPACE_DIR.parent

def _find_path(rel_path: str) -> Path:
    for candidate in [WORKSPACE_DIR / rel_path, PARENT_DIR / rel_path]:
        if candidate.exists():
            return candidate
    return WORKSPACE_DIR / rel_path

# Ensure backend and explainability repos are in sys.path globally
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

XAI_ROOT = _find_path("dfd-fcg-explainability")
if str(XAI_ROOT) not in sys.path:
    sys.path.insert(0, str(XAI_ROOT))

class Settings(BaseModel):
    PROJECT_NAME: str = "Video Deepfake Detection"
    TAGLINE: str = "DFD-FCG Explainable AI Forensic Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Base directories
    BASE_DIR: Path = BASE_DIR
    WORKSPACE_DIR: Path = WORKSPACE_DIR
    PARENT_DIR: Path = PARENT_DIR
    
    # Google Colab / Remote GPU Worker Support (Optional)
    COLAB_GPU_URL: Optional[str] = os.environ.get("COLAB_GPU_URL", None)
    
    # Neon Serverless PostgreSQL Database
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL", None)
    NEON_BRANCH: Optional[str] = os.environ.get("NEON_BRANCH", "production")
    
    # Storage settings
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    TEMPORARY_DIR: Path = STORAGE_DIR / "temporary"
    PROCESSED_DIR: Path = STORAGE_DIR / "processed"
    REPORTS_DIR: Path = STORAGE_DIR / "reports"
    HEATMAPS_DIR: Path = STORAGE_DIR / "heatmaps"
    SAMPLES_DIR: Path = STORAGE_DIR / "samples"
    RESEARCH_DATA_DIR: Path = STORAGE_DIR / "research_data"
    
    # Checkpoint settings
    CHECKPOINT_DIR: Path = _find_path("checkpoint")
    CHECKPOINT_WEIGHTS: Path = CHECKPOINT_DIR / "weights.ckpt"
    CHECKPOINT_SETTINGS: Path = CHECKPOINT_DIR / "setting.yaml"
    
    # Reference explainability repo
    XAI_ROOT: Path = XAI_ROOT
    XAI_DIR: Path = XAI_ROOT / "xai"
    
    # Thresholds (CVPR 2025 Binary Decision Boundary = 0.50)
    THRESHOLD_AUTHENTIC: float = 0.50      # P(FAKE) < 0.50 -> LIKELY AUTHENTIC
    THRESHOLD_MANIPULATED: float = 0.50    # P(FAKE) >= 0.50 -> LIKELY MANIPULATED
    
    # Upload limits
    MAX_UPLOAD_SIZE: int = 200 * 1024 * 1024 # 200 MB
    ALLOWED_EXTENSIONS: set = {".mp4", ".mov", ".avi", ".mkv"}
    
    # ML settings
    FACE_CROP_SIZE: int = 150
    GRID_SIZE: int = 16
    DEFAULT_MAX_CLIPS: int = 4 if not os.environ.get("CUDA_VISIBLE_DEVICES") else 30
    NUM_FRAMES_PER_CLIP: int = 10
    CLIP_STRIDE: float = 0.333
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    def ensure_directories(self):
        for directory in [
            self.STORAGE_DIR,
            self.UPLOADS_DIR,
            self.TEMPORARY_DIR,
            self.PROCESSED_DIR,
            self.REPORTS_DIR,
            self.HEATMAPS_DIR,
            self.SAMPLES_DIR,
            self.CHECKPOINT_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
