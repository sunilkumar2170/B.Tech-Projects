import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Video Deepfake Detection"
    assert "checkpoint_status" in data
    assert "thresholds" in data

def test_samples_endpoint():
    response = client.get("/api/videos/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 2
    keys = [s["sample_key"] for s in data["samples"]]
    assert "000" in keys
    assert "000_003" in keys

def test_sample_selection_and_analysis_flow():
    # Select sample 000
    res_select = client.post("/api/videos/sample/000")
    assert res_select.status_code == 200
    video_data = res_select.json()["video"]
    video_id = video_data["video_id"]
    
    # Start analysis
    res_start = client.post(f"/api/analysis/{video_id}")
    assert res_start.status_code == 200
    analysis_id = res_start.json()["analysis_id"]
    
    # Poll status
    res_status = client.get(f"/api/analysis/{analysis_id}/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "status" in status_data

def test_invalid_video_id_returns_404():
    res = client.post("/api/analysis/nonexistent_video_12345")
    assert res.status_code == 404

if __name__ == "__main__":
    test_health_endpoint()
    test_samples_endpoint()
    test_sample_selection_and_analysis_flow()
    test_invalid_video_id_returns_404()
    print("All backend API tests passed!")
