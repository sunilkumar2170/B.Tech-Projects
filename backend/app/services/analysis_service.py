import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

from app.core.config import settings
from app.core.exceptions import VideoNotFoundError, AnalysisNotFoundError
from app.schemas.analysis import AnalysisStatus, ProcessingStage
from ml.model_adapter import model_adapter
from app.services.report_service import ReportService

# Global in-memory job tracker
_JOBS: Dict[str, Dict[str, Any]] = {}
# Concurrency lock/semaphore
_GPU_SEMAPHORE = threading.Semaphore(1)

PIPELINE_STAGES = [
    ("video_validation", "Video validation and integrity verification"),
    ("frame_extraction", "Video frame extraction & temporal sequencing"),
    ("face_detection", "Face detection & bounding box tracking"),
    ("quality_assessment", "Video sharpness, compression, & lighting assessment"),
    ("dfd_fcg_inference", "DFD-FCG spatial-temporal neural network inference"),
    ("region_attribution", "16x16 patch & facial component regional attribution"),
    ("result_aggregation", "Forensic evidence synthesis & PDF report generation"),
]

class AnalysisService:
    @staticmethod
    def create_analysis_job(video_id: str, sample_key: Optional[str] = None) -> str:
        analysis_id = video_id
        
        stages = [
            ProcessingStage(
                id=stage_id,
                name=stage_name,
                status="pending",
                progress=0,
                message="Waiting..."
            )
            for stage_id, stage_name in PIPELINE_STAGES
        ]
        
        _JOBS[analysis_id] = {
            "analysis_id": analysis_id,
            "video_id": video_id,
            "sample_key": sample_key,
            "status": "queued",
            "current_stage": "video_validation",
            "progress": 0,
            "stages": stages,
            "elapsed_seconds": 0.0,
            "start_time": time.time(),
            "error_message": None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_at": None,
            "result": None
        }
        
        # Start background analysis thread
        thread = threading.Thread(
            target=AnalysisService._run_analysis_worker,
            args=(analysis_id, video_id, sample_key),
            daemon=True
        )
        thread.start()
        
        return analysis_id

    @staticmethod
    def _run_analysis_worker(analysis_id: str, video_id: str, sample_key: Optional[str]):
        job = _JOBS.get(analysis_id)
        if not job:
            return
            
        with _GPU_SEMAPHORE:
            job["status"] = "processing"
            job["start_time"] = time.time()
            
            def progress_callback(stage_id: str, progress: int, message: str):
                job["current_stage"] = stage_id
                job["progress"] = progress
                job["elapsed_seconds"] = round(time.time() - job["start_time"], 2)
                for st in job["stages"]:
                    if st.id == stage_id:
                        st.status = "in_progress"
                        st.progress = progress
                        st.message = message
                    elif st.progress < progress:
                        st.status = "completed"
                        st.progress = 100
                        
            try:
                # Locate video file
                if sample_key:
                    video_path = settings.SAMPLES_DIR / f"{sample_key}.mp4"
                else:
                    # Check in uploads
                    matches = list(settings.UPLOADS_DIR.glob(f"{video_id}.*"))
                    if not matches:
                        raise VideoNotFoundError(video_id)
                    video_path = matches[0]

                # Run ML adapter analysis
                result_data = model_adapter.analyze_video(
                    video_path=video_path,
                    analysis_id=analysis_id,
                    sample_key=sample_key,
                    progress_callback=progress_callback
                )
                
                # Generate PDF report
                pdf_path = settings.REPORTS_DIR / f"{analysis_id}.pdf"
                ReportService.generate_pdf_report(result_data, pdf_path)
                
                # Persist forensic record to Neon PostgreSQL
                try:
                    from app.db.service import DatabaseService
                    DatabaseService.save_analysis(result_data)
                except Exception as db_err:
                    import logging
                    logging.getLogger("uvicorn").warning(f"[Neon] Notice: could not persist to database: {db_err}")
                
                # Mark all stages completed
                for st in job["stages"]:
                    st.status = "completed"
                    st.progress = 100
                job["status"] = "completed"
                job["progress"] = 100
                job["elapsed_seconds"] = round(time.time() - job["start_time"], 2)
                job["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                job["result"] = result_data
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                job["status"] = "failed"
                job["error_message"] = str(e)
                for st in job["stages"]:
                    if st.id == job["current_stage"]:
                        st.status = "failed"
                        st.message = str(e)
                        break

    @staticmethod
    def get_analysis_status(analysis_id: str) -> AnalysisStatus:
        job = _JOBS.get(analysis_id)
        if not job:
            # Check if JSON exists on disk from earlier run
            result_file = settings.PROCESSED_DIR / f"{analysis_id}.json"
            if result_file.exists():
                return AnalysisStatus(
                    analysis_id=analysis_id,
                    video_id=analysis_id,
                    status="completed",
                    current_stage="result_aggregation",
                    progress=100,
                    stages=[],
                    elapsed_seconds=0.0,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                )
            raise AnalysisNotFoundError(analysis_id)
            
        elapsed = round(time.time() - job["start_time"], 2) if job["status"] == "processing" else job["elapsed_seconds"]
        return AnalysisStatus(
            analysis_id=job["analysis_id"],
            video_id=job["video_id"],
            status=job["status"],
            current_stage=job["current_stage"],
            progress=job["progress"],
            stages=job["stages"],
            elapsed_seconds=elapsed,
            error_message=job["error_message"],
            created_at=job["created_at"],
            completed_at=job["completed_at"]
        )

    @staticmethod
    def get_analysis_result(analysis_id: str) -> Dict[str, Any]:
        job = _JOBS.get(analysis_id)
        if job and job.get("result"):
            return job["result"]
            
        result_file = settings.PROCESSED_DIR / f"{analysis_id}.json"
        if result_file.exists():
            import json
            with open(result_file, "r") as f:
                return json.load(f)

        # Retrieve persistent record from Neon PostgreSQL if local file was purged
        try:
            from app.db.session import SessionLocal
            from app.db.models import ForensicAnalysisRecord
            if SessionLocal:
                with SessionLocal() as session:
                    rec = session.query(ForensicAnalysisRecord).filter_by(id=analysis_id).first()
                    if rec and rec.raw_payload:
                        return rec.raw_payload
        except Exception:
            pass
                
        raise AnalysisNotFoundError(analysis_id)
