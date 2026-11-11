import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ForensicAnalysisRecord(Base):
    """Stores full explainability and forensic deepfake detection results in Neon PostgreSQL."""
    __tablename__ = "forensic_analyses"

    id = Column(String(128), primary_key=True, index=True)
    video_id = Column(String(128), index=True, nullable=False)
    filename = Column(String(256), nullable=False)
    verdict = Column(String(64), nullable=False)
    p_fake = Column(Float, nullable=False)
    p_real = Column(Float, nullable=False)
    pct_spatial = Column(Float, nullable=False)
    pct_temporal = Column(Float, nullable=False)
    dominant_branch = Column(String(64), nullable=True)
    dominant_component = Column(String(64), nullable=True)
    video_duration_s = Column(Float, nullable=True)
    processing_time_s = Column(Float, nullable=True)
    inference_mode = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    raw_payload = Column(JSON, nullable=True)

class DatabaseAuditLog(Base):
    """Audit log tracking system events, analyses, and health pings in Neon PostgreSQL."""
    __tablename__ = "database_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(64), nullable=False, index=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
