from fastapi import APIRouter
from app.db.service import DatabaseService

router = APIRouter(prefix="/database", tags=["Database"])

@router.get("/status")
def get_database_status():
    """Returns real-time connection status and metrics for Neon PostgreSQL."""
    return DatabaseService.test_connection()

@router.get("/records")
def get_database_records(limit: int = 20):
    """Lists persistent forensic analysis records stored in Neon PostgreSQL."""
    return {
        "status": "success",
        "records": DatabaseService.list_records(limit=limit)
    }
