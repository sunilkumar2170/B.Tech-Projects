import os
import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

logger = logging.getLogger("uvicorn")

def get_database_url() -> Optional[str]:
    url = settings.DATABASE_URL or os.environ.get("DATABASE_URL")
    if not url:
        return None
    # SQLAlchemy requires postgresql:// instead of postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

db_url = get_database_url()

if db_url:
    # Use connection pooling suitable for Neon serverless pooler
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=300,
        connect_args={"connect_timeout": 5}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    engine = None
    SessionLocal = None

def init_db():
    """Initializes schema tables in Neon PostgreSQL on application startup with resilient error handling."""
    global engine, SessionLocal
    url = get_database_url()
    if not url:
        logger.info("[Neon] DATABASE_URL not set; running with local file-based storage.")
        return

    try:
        if not engine:
            engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=300,
                connect_args={"connect_timeout": 5}
            )
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

        from app.db.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("[Neon] Successfully initialized database tables in Neon PostgreSQL.")
    except Exception as e:
        logger.warning(f"[Neon] Database connection/initialization failed during startup: {e}. Falling back to local storage mode.")


def get_db() -> Generator[Optional[Session], None, None]:
    """Dependency for obtaining a database session."""
    if not SessionLocal:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
