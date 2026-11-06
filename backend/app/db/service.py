import time
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from app.db.session import SessionLocal, get_database_url
from app.db.models import ForensicAnalysisRecord, DatabaseAuditLog

logger = logging.getLogger("uvicorn")

class DatabaseService:
    @staticmethod
    def is_connected() -> bool:
        return SessionLocal is not None

    @staticmethod
    def test_connection() -> Dict[str, Any]:
        """Pings Neon Lakebase Postgres and retrieves connection diagnostics."""
        if not SessionLocal:
            return {
                "connected": False,
                "provider": "Neon (Lakebase Postgres)",
                "error": "DATABASE_URL is not configured."
            }

        start = time.time()
        try:
            with SessionLocal() as session:
                result = session.execute(text("SELECT version();")).fetchone()
                db_name = session.execute(text("SELECT current_database();")).scalar()
                count = session.query(ForensicAnalysisRecord).count()
                elapsed_ms = round((time.time() - start) * 1000, 2)
                
                return {
                    "connected": True,
                    "provider": "Neon (Lakebase Postgres)",
                    "database": db_name,
                    "latency_ms": elapsed_ms,
                    "version": result[0] if result else "Unknown",
                    "total_saved_analyses": count,
                    "pooling": "PgBouncer Serverless Pooler Active"
                }
        except Exception as e:
            return {
                "connected": False,
                "provider": "Neon (Lakebase Postgres)",
                "error": str(e)
            }

    @staticmethod
    def save_analysis(result_data: Dict[str, Any]) -> bool:
        """Persists video analysis results into Neon PostgreSQL."""
        if not SessionLocal:
            return False

        try:
            analysis_id = result_data.get("analysis_id")
            verdict_info = result_data.get("verdict", {})
            branch_info = result_data.get("branch_analysis", {})
            comp_info = result_data.get("facial_components", {})

            with SessionLocal() as session:
                existing = session.query(ForensicAnalysisRecord).filter_by(id=analysis_id).first()
                if existing:
                    existing.verdict = verdict_info.get("verdict", "UNKNOWN")
                    existing.p_fake = float(verdict_info.get("p_fake", 0.0))
                    existing.p_real = float(verdict_info.get("p_real", 0.0))
                    existing.pct_spatial = float(branch_info.get("pct_spatial", 0.0))
                    existing.pct_temporal = float(branch_info.get("pct_temporal", 0.0))
                    existing.dominant_branch = branch_info.get("dominant_branch")
                    existing.dominant_component = comp_info.get("dominant_component")
                    existing.video_duration_s = verdict_info.get("video_duration_s")
                    existing.processing_time_s = verdict_info.get("processing_time_s")
                    existing.inference_mode = verdict_info.get("inference_mode")
                    existing.raw_payload = result_data
                else:
                    record = ForensicAnalysisRecord(
                        id=analysis_id,
                        video_id=result_data.get("video_id", analysis_id),
                        filename=result_data.get("filename", f"{analysis_id}.mp4"),
                        verdict=verdict_info.get("verdict", "UNKNOWN"),
                        p_fake=float(verdict_info.get("p_fake", 0.0)),
                        p_real=float(verdict_info.get("p_real", 0.0)),
                        pct_spatial=float(branch_info.get("pct_spatial", 0.0)),
                        pct_temporal=float(branch_info.get("pct_temporal", 0.0)),
                        dominant_branch=branch_info.get("dominant_branch"),
                        dominant_component=comp_info.get("dominant_component"),
                        video_duration_s=verdict_info.get("video_duration_s"),
                        processing_time_s=verdict_info.get("processing_time_s"),
                        inference_mode=verdict_info.get("inference_mode"),
                        raw_payload=result_data
                    )
                    session.add(record)

                # Add audit log entry
                audit = DatabaseAuditLog(
                    event_type="ANALYSIS_COMPLETED",
                    details=f"Analysis '{analysis_id}' saved with verdict '{verdict_info.get('verdict')}' (P_fake={verdict_info.get('p_fake')})"
                )
                session.add(audit)
                session.commit()
                logger.info(f"[Neon] Successfully saved forensic record '{analysis_id}' to Neon PostgreSQL.")
                return True
        except Exception as e:
            logger.error(f"[Neon] Failed to save record to Neon: {e}")
            return False

    @staticmethod
    def list_records(limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves recent analysis records from Neon PostgreSQL."""
        if not SessionLocal:
            return []

        try:
            with SessionLocal() as session:
                records = session.query(ForensicAnalysisRecord).order_by(ForensicAnalysisRecord.created_at.desc()).limit(limit).all()
                return [
                    {
                        "analysis_id": r.id,
                        "video_id": r.video_id,
                        "filename": r.filename,
                        "verdict": r.verdict,
                        "p_fake": r.p_fake,
                        "p_real": r.p_real,
                        "pct_spatial": r.pct_spatial,
                        "pct_temporal": r.pct_temporal,
                        "dominant_branch": r.dominant_branch,
                        "dominant_component": r.dominant_component,
                        "duration_s": r.video_duration_s,
                        "created_at": r.created_at.isoformat() if r.created_at else None
                    }
                    for r in records
                ]
        except Exception as e:
            logger.error(f"[Neon] Failed to query records: {e}")
            return []
