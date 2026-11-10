import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from app.core.config import settings

class ModelLoader:
    _instance = None
    _model = None
    _status: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._init_loader()
        return cls._instance

    def _init_loader(self):
        self.device = "cuda" if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu"
        self.weights_path = settings.CHECKPOINT_WEIGHTS
        self.settings_path = settings.CHECKPOINT_SETTINGS
        
        self.is_checkpoint_present = self.weights_path.exists() and self.weights_path.stat().st_size > 1000000
        
        self._status = {
            "checkpoint_present": self.is_checkpoint_present,
            "weights_path": str(self.weights_path),
            "settings_path": str(self.settings_path),
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_loaded": False,
            "model_architecture": "ViT-L/14 (Frozen CLIP + FCG)",
            "clip_layers": 24,
            "message": "Ready" if self.is_checkpoint_present else (
                "Checkpoint weights not found at checkpoint/weights.ckpt. "
                "Download from official release: https://drive.google.com/file/d/1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk/view?usp=sharing"
            )
        }
        
        if self.is_checkpoint_present:
            self._load_checkpoint()

    def _load_checkpoint(self):
        try:
            # Set Windows workaround for Dynamo / Triton
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            
            # Ensure backend directory is in sys.path for `src` imports
            backend_dir = str(settings.STORAGE_DIR.parent)
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
                
            from src.model.clip.svl import FFGSynoVideoLearner
            
            print(f"[ModelLoader] Loading DFD-FCG checkpoint from {self.weights_path} to {self.device}...")
            try:
                self._model = FFGSynoVideoLearner.load_from_checkpoint(
                    str(self.weights_path),
                    map_location="cpu",
                    attn_record=True
                )
            except Exception:
                self._model = FFGSynoVideoLearner.load_from_checkpoint(
                    str(self.weights_path),
                    map_location="cpu",
                    attn_record=True,
                    strict=False
                )
            self._model.eval()
            self._model.to(self.device)
            self._status["model_loaded"] = True
            self._status["message"] = "Model loaded successfully into memory."
            print("[ModelLoader] Model loaded successfully.")
        except Exception as e:
            import traceback
            self._status["model_loaded"] = False
            self._status["error"] = repr(e)
            self._status["message"] = f"Failed to load checkpoint: {repr(e)}"
            print(f"[ModelLoader] Warning: could not load model checkpoint: {repr(e)}")
            traceback.print_exc()

    def get_model(self):
        return self._model

    def get_status(self) -> Dict[str, Any]:
        return self._status

model_loader = ModelLoader()
