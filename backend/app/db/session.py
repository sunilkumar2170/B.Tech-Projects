import os
import logging
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

logger = logging.getLogger("uvicorn")

import os
import ssl
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

def create_neon_engine(url: str):
    """Creates a resilient SQLAlchemy engine supporting both psycopg2/psycopg and pg8000."""
    try:
        eng = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_recycle=300,
            connect_args={"connect_timeout": 5}
        )
        # Test connection
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return eng
    except Exception as e1:
        # Fallback to pure-Python pg8000 (especially on Windows where native DLLs might be restricted)
        try:
            clean_url = url.split('?')[0].replace("postgresql://", "postgresql+pg8000://")
            eng = create_engine(
                clean_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=300,
                connect_args={"ssl_context": ssl.create_default_context(), "timeout": 5}
            )
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return eng
        except Exception as e2:
            logger.warning(f"[Neon] Database engine connection failed: {e1} | {e2}")
            return None

db_url = get_database_url()
engine = None
_session_maker = None

def get_engine():
    global engine
    if engine is None and db_url:
        engine = create_neon_engine(db_url)
    return engine

def get_session_local():
    global _session_maker
    if _session_maker is None:
        eng = get_engine()
        if eng:
            _session_maker = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return _session_maker

class _SessionLocalProxy:
    """Proxy allowing `SessionLocal()` usage while keeping engine initialization lazy."""
    def __call__(self, *args, **kwargs):
        factory = get_session_local()
        if factory is None:
            raise RuntimeError("Database engine not initialized or DATABASE_URL not available.")
        return factory(*args, **kwargs)

    def __bool__(self):
        return get_engine() is not None

SessionLocal = _SessionLocalProxy()

def init_db():
    """Initializes schema tables in Neon PostgreSQL on application startup with resilient error handling."""
    url = get_database_url()
    if not url:
        logger.info("[Neon] DATABASE_URL not set; running with local file-based storage.")
        return

    try:
        eng = get_engine()
        if eng:
            from app.db.models import Base
            Base.metadata.create_all(bind=eng)
            logger.info("[Neon] Successfully initialized database tables in Neon PostgreSQL.")
        else:
            logger.warning("[Neon] Could not establish Neon database connection; operating in local mode.")
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
