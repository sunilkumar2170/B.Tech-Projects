import os
from pathlib import Path
from pydantic import BaseModel, Field

from typing import Optional
from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BASE_DIR.parent

# Load environment files (.env in backend or project root .env.local)
load_dotenv(BASE_DIR / ".env")
load_dotenv(WORKSPACE_DIR / ".env.local")
load_dotenv(WORKSPACE_DIR / ".env")

class Settings(BaseModel):
    PROJECT_NAME: str = "Video Deepfake Detection"
    TAGLINE: str = "DFD-FCG Explainable AI Forensic Platform"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Base directories
    BASE_DIR: Path = BASE_DIR
    WORKSPACE_DIR: Path = WORKSPACE_DIR
    
    # Google Colab / Remote GPU Worker Support (Optional)
    COLAB_GPU_URL: Optional[str] = os.environ.get("COLAB_GPU_URL", None)
    
    # Neon Database settings (Lakebase Postgres)
    DATABASE_URL: Optional[str] = os.environ.get("DATABASE_URL", None)
    DATABASE_URL_UNPOOLED: Optional[str] = os.environ.get("DATABASE_URL_UNPOOLED", None)
    NEON_BRANCH: Optional[str] = os.environ.get("NEON_BRANCH", None)
    
    # Storage settings (NO database, local filesystem storage only)
    STORAGE_DIR: Path = BASE_DIR / "storage"
    UPLOADS_DIR: Path = STORAGE_DIR / "uploads"
    TEMPORARY_DIR: Path = STORAGE_DIR / "temporary"
    PROCESSED_DIR: Path = STORAGE_DIR / "processed"
    REPORTS_DIR: Path = STORAGE_DIR / "reports"
    HEATMAPS_DIR: Path = STORAGE_DIR / "heatmaps"
    SAMPLES_DIR: Path = STORAGE_DIR / "samples"
    RESEARCH_DATA_DIR: Path = STORAGE_DIR / "research_data"
    
    # Checkpoint settings
    CHECKPOINT_DIR: Path = WORKSPACE_DIR / "checkpoint"
    CHECKPOINT_WEIGHTS: Path = CHECKPOINT_DIR / "weights.ckpt"
    CHECKPOINT_SETTINGS: Path = CHECKPOINT_DIR / "setting.yaml"
    
    # Reference explainability repo
    XAI_DIR: Path = WORKSPACE_DIR / "dfd-fcg-explainability" / "xai"
    
    # Thresholds (Configurable)
    THRESHOLD_AUTHENTIC: float = 0.30      # P(FAKE) < 0.30 -> LIKELY AUTHENTIC
    THRESHOLD_MANIPULATED: float = 0.70    # P(FAKE) >= 0.70 -> LIKELY MANIPULATED
    # Between 0.30 and 0.70 -> INCONCLUSIVE
    
    # Upload limits
    MAX_UPLOAD_SIZE: int = 200 * 1024 * 1024 # 200 MB
    ALLOWED_EXTENSIONS: set = {".mp4", ".mov", ".avi", ".mkv"}
    
    # ML settings
    FACE_CROP_SIZE: int = 150
    GRID_SIZE: int = 16
    DEFAULT_MAX_CLIPS: int = 30
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
