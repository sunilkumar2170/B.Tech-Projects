import sys
import time
import requests
import numpy as np
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def wait_for_completion(analysis_id: str, max_wait: float = 30.0):
    start = time.time()
    while time.time() - start < max_wait:
        res = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}/status")
        assert res.status_code == 200, f"Status call failed: {res.text}"
        data = res.json()
        if data["status"] == "completed":
            return True
        elif data["status"] == "failed":
            raise RuntimeError(f"Analysis failed: {data.get('error_message')}")
        time.sleep(0.3)
    raise TimeoutError(f"Analysis {analysis_id} did not finish within {max_wait}s")

def run_analysis(sample_key: str):
    # Select sample video
    res = requests.post(f"{BASE_URL}/api/videos/sample/{sample_key}")
    assert res.status_code == 200, f"Select sample failed: {res.text}"
    video_id = res.json()["video"]["video_id"]
    
    # Start analysis
    res = requests.post(f"{BASE_URL}/api/analysis/{video_id}?sample_key={sample_key}")
    assert res.status_code == 200, f"Start analysis failed: {res.text}"
    analysis_id = res.json()["analysis_id"]
    
    wait_for_completion(analysis_id)
    
    # Fetch result
    res = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}/result")
    assert res.status_code == 200, f"Get result failed: {res.text}"
    return res.json()

def test_determinism_sample(sample_key: str, expected_p_fake: float, expected_spatial: float, expected_temporal: float):
    print(f"\n--- Testing determinism and precision for sample: {sample_key} ---")
    
    # Run 1
    t0 = time.time()
    res1 = run_analysis(sample_key)
    print(f"Run 1 completed in {time.time() - t0:.2f}s")
    
    # Run 2
    t1 = time.time()
    res2 = run_analysis(sample_key)
    print(f"Run 2 completed in {time.time() - t1:.2f}s")
    
    # Verify exact precision
    p_fake1 = res1["verdict"]["p_fake"]
    p_fake2 = res2["verdict"]["p_fake"]
    print(f"P(Fake): Run 1 = {p_fake1}, Run 2 = {p_fake2}, Expected = {expected_p_fake}")
    assert abs(p_fake1 - expected_p_fake) < 1e-4, f"p_fake {p_fake1} deviated from expected {expected_p_fake}"
    assert p_fake1 == p_fake2, f"Deviation across runs in p_fake: {p_fake1} != {p_fake2}"
    
    pct_sp1 = res1["branch_analysis"]["pct_spatial"]
    pct_sp2 = res2["branch_analysis"]["pct_spatial"]
    print(f"Pct Spatial: Run 1 = {pct_sp1}%, Run 2 = {pct_sp2}%, Expected = {expected_spatial}%")
    assert abs(pct_sp1 - expected_spatial) < 1e-2, f"pct_spatial {pct_sp1} deviated from expected {expected_spatial}"
    assert pct_sp1 == pct_sp2, f"Deviation across runs in pct_spatial: {pct_sp1} != {pct_sp2}"
    
    pct_tp1 = res1["branch_analysis"]["pct_temporal"]
    pct_tp2 = res2["branch_analysis"]["pct_temporal"]
    print(f"Pct Temporal: Run 1 = {pct_tp1}%, Run 2 = {pct_tp2}%, Expected = {expected_temporal}%")
    assert abs(pct_tp1 - expected_temporal) < 1e-2, f"pct_temporal {pct_tp1} deviated from expected {expected_temporal}"
    assert pct_tp1 == pct_tp2, f"Deviation across runs in pct_temporal: {pct_tp1} != {pct_tp2}"
    
    # Verify components determinism
    comp1 = res1["facial_components"]["components"]
    comp2 = res2["facial_components"]["components"]
    for k in comp1:
        assert comp1[k] == comp2[k], f"Facial component {k} deviated: {comp1[k]} != {comp2[k]}"
    print("Facial components are 100% identical across runs:", comp1)
    
    # Verify patch attributions determinism
    sp_grid1 = np.array(res1["facial_regions"]["patch_spatial_attributions"])
    sp_grid2 = np.array(res2["facial_regions"]["patch_spatial_attributions"])
    diff_sp = np.max(np.abs(sp_grid1 - sp_grid2))
    assert diff_sp == 0.0, f"Spatial patch attributions deviated by max {diff_sp}"
    
    tp_grid1 = np.array(res1["facial_regions"]["patch_temporal_attributions"])
    tp_grid2 = np.array(res2["facial_regions"]["patch_temporal_attributions"])
    diff_tp = np.max(np.abs(tp_grid1 - tp_grid2))
    assert diff_tp == 0.0, f"Temporal patch attributions deviated by max {diff_tp}"
    print(f"16x16 patch grids: Max delta between runs = 0.0 (Zero deviation)")
    
    # Verify clip count and values
    clips1 = res1["clips"]
    clips2 = res2["clips"]
    assert len(clips1) == len(clips2) == 60, f"Expected 60 clips, got {len(clips1)} and {len(clips2)}"
    for idx, (c1, c2) in enumerate(zip(clips1, clips2)):
        assert c1["p_final"] == c2["p_final"], f"Clip {idx} p_final deviated: {c1['p_final']} != {c2['p_final']}"
        assert c1["phi_spatial"] == c2["phi_spatial"], f"Clip {idx} phi_spatial deviated"
        assert c1["phi_temporal"] == c2["phi_temporal"], f"Clip {idx} phi_temporal deviated"
    print(f"All 60 clips verified: Identical bit-for-bit across consecutive runs.")

if __name__ == "__main__":
    print("Starting Determinism & Accuracy Verification...")
    
    # Sample 000 (Authentic FaceForensics++)
    test_determinism_sample(
        sample_key="000",
        expected_p_fake=0.4207,
        expected_spatial=29.09,
        expected_temporal=70.91
    )
    
    # Sample 000_003 (Manipulated FaceForensics++ FaceSwap)
    test_determinism_sample(
        sample_key="000_003",
        expected_p_fake=0.9317,
        expected_spatial=50.38,
        expected_temporal=49.62
    )
    
    print("\n========================================================")
    print("SUCCESS: ALL BACKEND RESULTS ARE ACCURATE AND DETERMINISTIC!")
    print("Zero deviation observed across consecutive executions.")
    print("========================================================")
