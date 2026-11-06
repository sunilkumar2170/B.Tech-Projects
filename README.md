# B.Tech-Projects — Explainable Video Deepfake Detection

> **"Detect Deepfakes. Understand the Evidence."**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-success?style=for-the-badge&logo=vercel)](https://frontend-psi-lime-98.vercel.app)
[![API & Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%7C%20React-blue?style=for-the-badge)](https://frontend-psi-lime-98.vercel.app)
[![Research Benchmark](https://img.shields.io/badge/CVPR%20'25-DFD--FCG-blueviolet?style=for-the-badge)](https://arxiv.org/abs/2404.05583)

🔗 **Live Production Web Application:** [https://frontend-psi-lime-98.vercel.app](https://frontend-psi-lime-98.vercel.app)  
🚀 **Production Alternate Mirror:** [https://frontend-dsq9oh8vw-roshan-rajs-projects-239fc114.vercel.app](https://frontend-dsq9oh8vw-roshan-rajs-projects-239fc114.vercel.app)

A complete, explainable AI-powered video deepfake detection and forensic analysis platform built on the state-of-the-art **DFD-FCG** (CVPR 2025) architecture.

This platform transforms deepfake detection from an opaque binary prediction into a transparent, evidence-driven forensic investigation.

---

## 🌟 Key Capabilities

Instead of merely outputting **REAL** or **FAKE**, the platform explains the model's decision using:

- **Spatial Attribution:** Decomposes facial appearance, boundary blending, and texture anomalies.
- **Temporal Attribution:** Tracks frame-to-frame motion continuity and micro-jitter across overlapping 3-second clips.
- **Exact Shapley Branch Decomposition:** Measures the diagnostic split between appearance and motion cues ($\phi_S + \phi_T = P_{final} - 0.5$) with zero residual error.
- **Facial Component Guidance (FCG):** Explains contributions across 4 anatomical components: **Eyes**, **Nose**, **Lips**, and **Skin**.
- **16×16 Patch Grid Heatmaps:** Visualizes exact patch-level attributions mapped directly to facial landmark coordinates.
- **Independent Video Quality Audit:** Measures Laplacian blur variance, H.264 compression artifacts, and exposure balance independently of the manipulation score.
- **Publication-Grade Forensic PDF Reports:** Generates structured, downloadable evidence reports.

---

## 🏛️ System Architecture

```
                          ┌───────────────────────────┐
                          │   REACT + VITE FRONTEND   │
                          │   (Tailwind, Recharts)    │
                          └─────────────┬─────────────┘
                                        │ REST API
                                        ▼
                          ┌───────────────────────────┐
                          │      FASTAPI BACKEND      │
                          │ (In-Memory Job Scheduler) │
                          └──────┬─────────────┬──────┘
                                 │             │
                    ┌────────────▼──┐       ┌──▼────────────────┐
                    │ Video Ingest  │       │ Video Quality     │
                    │ & Alignment   │       │ (Blur/Compression)│
                    └────────────┬──┘       └───────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 68-Point Face Tracker │
                     │   (150×150 Aligned)   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   DFD-FCG (CVPR '25)  │
                     │ Frozen CLIP ViT-L/14  │
                     └───────────┬───────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
       ┌───────────────────────┐   ┌───────────────────────┐
       │  SPATIAL BRANCH (S)   │   │  TEMPORAL BRANCH (T)  │
       │  4 Component Queries  │   │ 16×16 Patch Dynamics  │
       └───────────┬───────────┘   └───────────┬───────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ EXPLAINABILITY ENGINE │
                     │  Shapley Attribution  │
                     │ Zero-Residual Decomp  │
                     └───────────┬───────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
         ┌─────────────────────┐   ┌─────────────────────┐
         │ 16×16 Visual Map    │   │ Forensic PDF Report │
         │ & Component Shares  │   │  & JSON Export      │
         └─────────────────────┘   └─────────────────────┘
```

---

## 🔬 Scientific Foundations

1. **Upstream Model Architecture:**
   - [aiiu-lab/DFD-FCG](https://github.com/aiiu-lab/DFD-FCG)
   - *Yue-Hua Han, Tai-Ming Huang, Kai-Lung Hua, Jun-Cheng Chen*, *"Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model"*, **CVPR 2025** ([arXiv:2404.05583](https://arxiv.org/abs/2404.05583)).
2. **Exact Additive Explainability:**
   - DFD-FCG computes its final probability as an explicit mean of three head softmaxes:
     $$P_{final} = \frac{\text{softmax}(\text{logits}_s) + \text{softmax}(\text{logits}_t) + \text{softmax}(\text{logits}_a)}{3}$$
   - Each head consists of `LayerNorm → Linear`. Because LayerNorm means are additive across facial parts and patches, logit differences decompose with **no residual term**:
     $$d = \sum_k \text{contribution}(\text{part } k) + \text{constant}$$
   - Verified to float64 machine precision ($10^{-16}$).

---

## ⚖️ Configurable Decision Thresholds

| Prediction Probability | Forensic Verdict | Interpretation |
|---|---|---|
| $P(\text{FAKE}) < 0.30$ | **LIKELY AUTHENTIC** | Natural motion and texture cues exhibit consistent physiological patterns. |
| $0.30 \le P(\text{FAKE}) < 0.70$ | **INCONCLUSIVE** | Evidence is statistically insufficient to confirm manipulation or rule out synthesis. |
| $P(\text{FAKE}) \ge 0.70$ | **LIKELY MANIPULATED** | Strong spatial boundary anomalies and/or temporal motion incoherence detected. |

---

## 📁 Repository Structure

```
.
├── checkpoint/                   # Checkpoint storage directory
│   ├── setting.yaml             # DFD-FCG ViT-L/14 architecture config
│   ├── weights.ckpt             # Model weights (user-provided)
│   └── README.md                # Download & verification guide
│
├── backend/                      # Python FastAPI Forensic Service
│   ├── app/
│   │   ├── main.py              # Application entrypoint & static mounting
│   │   ├── api/                 # Endpoints: health, upload, analysis, reports
│   │   ├── core/                # Configuration & centralized exceptions
│   │   ├── schemas/             # Pydantic request/response models
│   │   └── services/            # Upload, Analysis worker, PDF report service
│   ├── ml/                      # Forensic ML Adapter Layer
│   │   ├── model_loader.py      # Singleton loader with CUDA/CPU fallback
│   │   ├── model_adapter.py     # Inference & explainability orchestrator
│   │   ├── face_detector.py     # 68-point facial landmark tracker
│   │   ├── preprocessing.py     # 10-frame clip generator & CLIP normalizer
│   │   ├── attribution.py       # Exact LayerNorm-Linear & Shapley algebra
│   │   ├── region_analysis.py   # 16×16 patch-to-region mapping & heatmaps
│   │   ├── clip_aggregation.py  # Video-level aggregation & dynamic explanation
│   │   └── quality_assessment.py# Laplacian blur, compression, exposure audit
│   ├── storage/                 # Local filesystem storage & Neon Serverless PostgreSQL persistence
│   │   ├── uploads/             # Received video files
│   │   ├── processed/           # Completed analysis JSON records
│   │   ├── reports/             # Generated PDF audit reports
│   │   ├── heatmaps/            # Generated 16×16 overlay images
│   │   └── samples/             # Bundled FaceForensics++ benchmark videos
│   ├── tests/                   # Automated API and pipeline tests
│   └── requirements.txt         # Python dependencies
│
├── frontend/                     # React + TypeScript + Vite SPA
│   ├── src/
│   │   ├── components/layout/   # Navbar & Footer
│   │   ├── pages/
│   │   │   ├── Home.tsx         # Landing page with interactive technology showcase
│   │   │   ├── Upload.tsx       # Video upload & benchmark sample selector
│   │   │   ├── Processing.tsx   # Real-time multi-stage pipeline monitor
│   │   │   └── Results.tsx      # Comprehensive forensic results dashboard
│   │   ├── services/api.ts      # Axios API client
│   │   └── types/index.ts       # Strict TypeScript interface contracts
│   └── package.json
│
├── dfd-fcg-explainability/       # Existing research implementation & benchmarks
├── run_fullstack.bat             # 1-Click Windows Batch Fullstack Launcher
├── run_fullstack.ps1             # 1-Click Windows PowerShell Fullstack Launcher
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started on Windows

### Prerequisites

- **Python:** 3.10 to 3.13
- **Node.js:** v18+ and `npm`
- **PyTorch:** 2.0+ (CUDA optional; CPU execution supported)

### 1. Download Model Checkpoint (Optional for Benchmark Demo)

The project includes preloaded research audit benchmarks for FaceForensics++ videos (`000.mp4` and `000_003.mp4`).

To run live neural inference on arbitrary new video uploads:
1. Download `weights.ckpt` from the official author link:
   - [Google Drive Download (1.2 GB)](https://drive.google.com/file/d/1ydD5rnaaF0i2zLE7NidLtAhjonHoVQOk/view?usp=sharing)
2. Move the downloaded file to:
   ```
   checkpoint/weights.ckpt
   ```

### 2. Launching the Application

You can start both backend and frontend servers with one click:

**Using Windows PowerShell:**
```powershell
.\run_fullstack.ps1
```

**Using Windows Command Prompt:**
```cmd
run_fullstack.bat
```

Alternatively, run them in separate terminals:

**Terminal 1 (Backend):**
```powershell
$env:TORCHDYNAMO_DISABLE="1"
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```powershell
cd frontend
npm run dev
```

Open your browser at:
- **Frontend Dashboard:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Interactive Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 🌐 Cloud Deployment (Vercel & Render)

### 1. Vercel (Frontend Live Web Application)
The frontend is deployed to Vercel production with SPA routing rewrites:
- **Live App URL:** [https://frontend-psi-lime-98.vercel.app](https://frontend-psi-lime-98.vercel.app)
- **Production Alternate:** [https://frontend-dsq9oh8vw-roshan-rajs-projects-239fc114.vercel.app](https://frontend-dsq9oh8vw-roshan-rajs-projects-239fc114.vercel.app)
- **High-Availability Edge Fallback:** Bundled with exact CVPR 2025 FaceForensics++ benchmark data (`000.mp4` authentic & `000_003.mp4` deepfake) and static heatmaps under `/public/storage/` for offline/edge resilience.

### 2. Render & Docker (Backend Web Service)
The backend is configured for 1-click cloud deployment:
- **Render Blueprint:** Configured via [`render.yaml`](./render.yaml). Simply connect this repo to Render and choose **New > Blueprint**.
- **Containerized Build:** Configured via [`backend/Dockerfile`](./backend/Dockerfile) with Python 3.12, OpenCV system libraries (`ffmpeg`, `libgl1`), and uvicorn.

### 3. Neon Serverless PostgreSQL Database
The backend is integrated with Neon Serverless PostgreSQL for persistent forensic audit records:
- **Connection Mode:** Serverless PgBouncer connection pooling (`DATABASE_URL`) with direct unpooled migration support (`DATABASE_URL_UNPOOLED`).
- **Provisioning:** Managed via [`neon.ts`](./neon.ts) and the Neon CLI (`neon deploy`).
- **Audit Schema:** Tracks `forensic_analysis_records` (SHA-256 integrity, DFD-FCG fake probability, classification, heatmap metadata) and `database_audit_logs`.
- **Status API:** Query live database telemetry and record counts via `GET /api/database/status` and `GET /api/database/records`.


---

## 🧪 Testing

Run backend API tests and the end-to-end forensic pipeline verification:

```powershell
# API unit and integration tests
python backend/tests/test_api.py

# End-to-end pipeline & PDF generation verification
python backend/tests/verify_pipeline.py
```

---

## 📜 Citations

```bibtex
@inproceedings{cvpr25_dfd_fcg,
  title={Towards More General Video-based Deepfake Detection through Facial
         Component Guided Adaptation for Foundation Model},
  author={Yue-Hua Han and Tai-Ming Huang and Kai-Lung Hua and Jun-Cheng Chen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
