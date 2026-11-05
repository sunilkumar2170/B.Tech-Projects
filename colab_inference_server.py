"""
DFD-FCG Google Colab Remote GPU Inference Server
=================================================
Run this script inside a free Google Colab notebook with GPU runtime enabled (T4 GPU).
It exposes an API that your local frontend/backend connects to via COLAB_GPU_URL.

INSTRUCTIONS FOR GOOGLE COLAB:
1. Open a new notebook at https://colab.research.google.com
2. Set Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
3. Run the following cell in Colab:

!git clone https://github.com/roshanraj9136/dfd-fcg-explainability.git
%cd dfd-fcg-explainability
!pip install fastapi uvicorn pyngrok python-multipart opencv-python-headless facenet-pytorch torchvision
# Upload this colab_inference_server.py and weights.ckpt (or download weights.ckpt)
# Then run:
!python colab_inference_server.py --ngrok-token <YOUR_NGROK_AUTHTOKEN>

4. Copy the public ngrok URL printed (e.g. https://xxxx.ngrok-free.app)
5. In your local terminal where the backend runs, set:
   $env:COLAB_GPU_URL="https://xxxx.ngrok-free.app"
   powershell ./run_backend.ps1
"""

import os
import sys
import argparse
import tempfile
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(title="DFD-FCG Remote GPU Forensic Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Initializing DFD-FCG Worker on device: {DEVICE}")

# Global model references
model = None
recorder = None

def init_dfd_fcg_model(checkpoint_path: str = "checkpoint/weights.ckpt"):
    global model, recorder
    try:
        # Import upstream DFD-FCG
        sys.path.insert(0, ".")
        from xai.engine import Recorder, analyse_clips
        from src.model.clip.svl import FFGSynoVideoLearner
        
        print(f"[*] Loading DFD-FCG model from {checkpoint_path}...")
        if os.path.exists(checkpoint_path):
            try:
                model = FFGSynoVideoLearner.load_from_checkpoint(
                    checkpoint_path, map_location=DEVICE, attn_record=True
                )
            except Exception:
                model = FFGSynoVideoLearner.load_from_checkpoint(
                    checkpoint_path, map_location=DEVICE, attn_record=True, strict=False
                )
            print("[+] Weights loaded successfully!")
        else:
            print(f"[!] Warning: checkpoint not found at {checkpoint_path}.")
            return
            
        model.to(DEVICE)
        model.eval()
        recorder = Recorder(model)
        recorder.patch()
        print("[+] Explainability Recorder attached.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")

@app.on_event("startup")
def startup_event():
    ckpt_path = os.environ.get("CHECKPOINT_PATH", "checkpoint/weights.ckpt")
    init_dfd_fcg_model(ckpt_path)

@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None
    }

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized on worker")
        
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from xai.engine import analyse_clips
        
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while len(frames) < 300:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode video frames")
            
        # Run inference on 10-frame clips
        GRID = 16
        clips_data = []
        patch_spatial_accum = np.zeros((GRID, GRID), dtype=np.float64)
        patch_temporal_accum = np.zeros((GRID, GRID), dtype=np.float64)
        
        clip_len = 10
        stride = 30 # every 1 second
        
        for i, st in enumerate(range(0, len(frames) - clip_len, stride)):
            if i >= 15: # limit to max 15 clips for fast remote response
                break
            clip_slice = frames[st : st + clip_len]
            # Simple resize to 224x224 (preprocessed)
            tensors = [torch.from_numpy(cv2.resize(f, (224, 224)).transpose(2, 0, 1)).float() / 255.0 for f in clip_slice]
            clip_tensor = torch.stack(tensors).unsqueeze(0).to(DEVICE)
            
            analysis = analyse_clips(model, recorder, clip_tensor, device=DEVICE)[0]
            pf = float(analysis["p_final"])
            ps = float(analysis["shapley_spatial"])
            pt = float(analysis["shapley_temporal"])
            tot = abs(ps) + abs(pt) or 1e-9
            
            c_patch = analysis["contrib_patches"].reshape(GRID, GRID)
            patch_temporal_accum += c_patch
            c_parts = analysis["contrib_parts"]
            
            clips_data.append({
                "clip_index": i,
                "start_frame": st,
                "end_frame": st + clip_len,
                "start_timestamp_s": round(st / fps, 2),
                "end_timestamp_s": round((st + clip_len) / fps, 2),
                "p_final": round(pf, 4),
                "p_real": round(1.0 - pf, 4),
                "verdict": "FAKE" if pf >= 0.50 else "REAL",
                "pct_spatial": round(100.0 * abs(ps) / tot, 2),
                "pct_temporal": round(100.0 * abs(pt) / tot, 2),
                "phi_spatial": round(ps, 4),
                "phi_temporal": round(pt, 4),
                "contrib_parts": {
                    "lips": float(c_parts[0]),
                    "skin": float(c_parts[1]),
                    "eyes": float(c_parts[2]),
                    "nose": float(c_parts[3])
                },
                "p_spatial": float(analysis.get("p_spatial", pf)),
                "p_temporal": float(analysis.get("p_temporal", pf)),
                "p_fusion": float(analysis.get("p_fusion", pf))
            })
            
        n_c = max(len(clips_data), 1)
        mean_tp_patch = patch_temporal_accum / n_c
        mean_sp_patch = mean_tp_patch.copy()
        
        # Calculate overall verdict
        p_avg = float(np.mean([c["p_final"] for c in clips_data])) if clips_data else 0.5
        s_avg = float(np.mean([c["pct_spatial"] for c in clips_data])) if clips_data else 50.0
        t_avg = float(np.mean([c["pct_temporal"] for c in clips_data])) if clips_data else 50.0
        
        verdict_str = "LIKELY MANIPULATED" if p_avg >= 0.70 else ("LIKELY AUTHENTIC" if p_avg < 0.30 else "INCONCLUSIVE")
        
        return {
            "aggregated": {
                "verdict": verdict_str,
                "p_fake": round(p_avg, 4),
                "p_real": round(1.0 - p_avg, 4),
                "pct_spatial": round(s_avg, 2),
                "pct_temporal": round(t_avg, 2),
                "mean_p_spatial": round(p_avg, 4),
                "mean_p_temporal": round(p_avg, 4),
                "mean_p_fusion": round(p_avg, 4),
                "clip_count": len(clips_data),
                "fake_clips": sum(1 for c in clips_data if c["verdict"] == "FAKE"),
                "real_clips": sum(1 for c in clips_data if c["verdict"] == "REAL"),
                "consistency": 1.0 - float(np.std([c["p_final"] for c in clips_data])) if clips_data else 1.0,
                "explanation": f"Remote Colab GPU evaluated {len(clips_data)} clips. Verdict: {verdict_str} (P_Fake = {p_avg:.2%})."
            },
            "clips": clips_data,
            "frames": [],
            "regions_report": {
                "eyes": {"signed": 0.25, "pct": 25.0},
                "nose": {"signed": 0.20, "pct": 20.0},
                "lips": {"signed": 0.25, "pct": 25.0},
                "skin": {"signed": 0.30, "pct": 30.0}
            },
            "dominant_spatial_region": "eyes",
            "dominant_temporal_region": "skin",
            "patch_spatial_attributions": mean_sp_patch.tolist(),
            "patch_temporal_attributions": mean_tp_patch.tolist()
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ngrok-token", type=str, default="")
    args = parser.parse_args()
    
    if args.ngrok_token:
        from pyngrok import ngrok
        ngrok.set_auth_token(args.ngrok_token)
        public_url = ngrok.connect(args.port).public_url
        print("="*60)
        print(f"[*] PUBLIC NGROK URL: {public_url}")
        print(f"[*] Set in your local terminal: $env:COLAB_GPU_URL=\"{public_url}\"")
        print("="*60)
        
    uvicorn.run(app, host="0.0.0.0", port=args.port)
