from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any

from app.core.config import settings
from app.schemas.analysis import AnalysisStatus, AnalysisResult
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/analysis", tags=["Analysis"])

@router.post("/{video_id}")
def start_analysis(video_id: str, sample_key: Optional[str] = Query(None)):
    """Initiates asynchronous explainability deepfake analysis on a video."""
    # Check if video_id starts with sample_
    if video_id.startswith("sample_"):
        sample_key = video_id.replace("sample_", "")
        
    if sample_key:
        if sample_key not in ("000", "000_003"):
            raise HTTPException(status_code=404, detail=f"Benchmark sample '{sample_key}' not found.")
    else:
        # Validate that uploaded file actually exists
        matches = list(settings.UPLOADS_DIR.glob(f"{video_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Video with id '{video_id}' not found.")

    analysis_id = AnalysisService.create_analysis_job(video_id, sample_key=sample_key)
    return {
        "analysis_id": analysis_id,
        "video_id": video_id,
        "status": "queued",
        "message": "Analysis started in background."
    }

@router.get("/{analysis_id}/status", response_model=AnalysisStatus)
def get_analysis_status(analysis_id: str):
    """Retrieves real-time processing stage and progress for an active job."""
    return AnalysisService.get_analysis_status(analysis_id)

@router.get("/{analysis_id}/result")
def get_analysis_result(analysis_id: str):
    """Returns full forensic deepfake detection and explainability results."""
    return AnalysisService.get_analysis_result(analysis_id)

@router.get("/{analysis_id}/spatial")
def get_spatial_analysis(analysis_id: str):
    """Returns spatial branch attribution and FCG component breakdown."""
    res = AnalysisService.get_analysis_result(analysis_id)
    return {
        "analysis_id": analysis_id,
        "spatial_contribution_pct": res["branch_analysis"]["pct_spatial"],
        "phi_spatial": res["branch_analysis"]["phi_spatial"],
        "facial_components": res["facial_components"],
        "heatmap_spatial_url": res["facial_regions"].get("heatmap_spatial_url")
    }

@router.get("/{analysis_id}/temporal")
def get_temporal_analysis(analysis_id: str):
    """Returns temporal branch attribution and patch motion metrics."""
    res = AnalysisService.get_analysis_result(analysis_id)
    return {
        "analysis_id": analysis_id,
        "temporal_contribution_pct": res["branch_analysis"]["pct_temporal"],
        "phi_temporal": res["branch_analysis"]["phi_temporal"],
        "heatmap_temporal_url": res["facial_regions"].get("heatmap_temporal_url")
    }

@router.get("/{analysis_id}/components")
def get_facial_components(analysis_id: str):
    """Returns component-level attributions for eyes, nose, lips, and skin."""
    res = AnalysisService.get_analysis_result(analysis_id)
    return res["facial_components"]

@router.get("/{analysis_id}/regions")
def get_facial_regions(analysis_id: str):
    """Returns 16x16 patch-level and anatomical region attributions."""
    res = AnalysisService.get_analysis_result(analysis_id)
    return res["facial_regions"]

@router.get("/{analysis_id}/clips")
def get_clips_timeline(analysis_id: str):
    """Returns overlapping 3-second clips predictions and Shapley shares."""
    res = AnalysisService.get_analysis_result(analysis_id)
    return {
        "analysis_id": analysis_id,
        "total_clips": len(res["clips"]),
        "clips": res["clips"]
    }

@router.get("/{analysis_id}/report")
def get_summary_report(analysis_id: str):
    """Returns executive forensic summary, verdict, and quality assessment."""
    res = AnalysisService.get_analysis_result(analysis_id)
    return {
        "analysis_id": analysis_id,
        "filename": res["filename"],
        "verdict": res["verdict"],
        "explanation": res["final_explanation"],
        "quality_report": res["quality_report"],
        "download_pdf_url": res["report_pdf_url"]
    }
