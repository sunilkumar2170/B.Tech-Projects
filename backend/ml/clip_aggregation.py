import numpy as np
from typing import List, Dict, Any, Tuple

def aggregate_clip_results(
    clip_results: List[Dict[str, Any]],
    threshold_authentic: float = 0.30,
    threshold_manipulated: float = 0.70
) -> Dict[str, Any]:
    """
    Aggregates per-clip predictions and attributions into video-level metrics.
    Preserves exact DFD-FCG ensemble formula:
        P_final = (P_spatial + P_temporal + P_fusion) / 3
    """
    if not clip_results:
        raise ValueError("No clip results to aggregate.")
        
    p_finals = [c.get("p_final_raw", c["p_final"]) for c in clip_results]
    p_spatials = [c.get("p_spatial_raw", c["p_spatial"]) for c in clip_results]
    p_temporals = [c.get("p_temporal_raw", c["p_temporal"]) for c in clip_results]
    p_fusions = [c.get("p_fusion_raw", c["p_fusion"]) for c in clip_results]
    
    phi_spatials = [c.get("phi_spatial_raw", c.get("shapley_spatial", c.get("phi_spatial", 0.0))) for c in clip_results]
    phi_temporals = [c.get("phi_temporal_raw", c.get("shapley_temporal", c.get("phi_temporal", 0.0))) for c in clip_results]
    
    mean_p_final = float(np.mean(p_finals))
    std_p_final = float(np.std(p_finals))
    mean_p_spatial = float(np.mean(p_spatials))
    mean_p_temporal = float(np.mean(p_temporals))
    mean_p_fusion = float(np.mean(p_fusions))
    
    mean_phi_s = float(np.mean(phi_spatials))
    mean_phi_t = float(np.mean(phi_temporals))
    
    # Compute branch share faithfully from per-clip percentages (matching CVPR 2025 benchmark)
    pct_spatials = [c["pct_spatial"] for c in clip_results if "pct_spatial" in c]
    pct_temporals = [c["pct_temporal"] for c in clip_results if "pct_temporal" in c]
    if pct_spatials and pct_temporals:
        pct_spatial = float(np.mean(pct_spatials))
        pct_temporal = float(np.mean(pct_temporals))
    else:
        tot_phi = abs(mean_phi_s) + abs(mean_phi_t)
        if tot_phi == 0:
            tot_phi = 1e-9
        pct_spatial = float(100.0 * abs(mean_phi_s) / tot_phi)
        pct_temporal = float(100.0 * abs(mean_phi_t) / tot_phi)
    
    # Final verdict based on configured thresholds
    p_fake = mean_p_final
    p_real = 1.0 - p_fake
    if p_fake >= threshold_manipulated:
        verdict = "LIKELY MANIPULATED"
    elif p_fake < threshold_authentic:
        verdict = "LIKELY AUTHENTIC"
    else:
        verdict = "INCONCLUSIVE"
        
    dominant_branch = "Spatial" if pct_spatial >= pct_temporal else "Temporal"
    
    # Aggregate facial components
    component_names = ["lips", "skin", "eyes", "nose"]
    part_contribs = {k: [] for k in component_names}
    for c in clip_results:
        parts = c.get("contrib_parts", {})
        for k in component_names:
            if isinstance(parts, dict):
                part_contribs[k].append(parts.get(k, 0.0))
            elif isinstance(parts, (list, np.ndarray)) and len(parts) == 4:
                idx = component_names.index(k)
                part_contribs[k].append(parts[idx])
                
    mean_parts = {k: float(np.mean(v)) if v else 0.0 for k, v in part_contribs.items()}
    
    # Find dominant facial component by magnitude
    sorted_parts = sorted(mean_parts.items(), key=lambda x: abs(x[1]), reverse=True)
    dominant_component = sorted_parts[0][0] if sorted_parts else "eyes"
    
    # Generate dynamic, evidence-driven explanation
    explanation = generate_forensic_explanation(
        verdict=verdict,
        p_fake=p_fake,
        pct_spatial=pct_spatial,
        pct_temporal=pct_temporal,
        mean_parts=mean_parts,
        dominant_component=dominant_component,
        n_clips=len(clip_results)
    )
    
    return {
        "verdict": verdict,
        "p_fake": round(p_fake, 4),
        "p_real": round(p_real, 4),
        "final_mean": round(mean_p_final, 4),
        "final_std": round(std_p_final, 4),
        "spatial_mean": round(mean_p_spatial, 4),
        "temporal_mean": round(mean_p_temporal, 4),
        "fusion_mean": round(mean_p_fusion, 4),
        "pct_spatial": round(pct_spatial, 2),
        "pct_temporal": round(pct_temporal, 2),
        "phi_spatial": round(mean_phi_s, 4),
        "phi_temporal": round(mean_phi_t, 4),
        "dominant_branch": dominant_branch,
        "mean_facial_parts": mean_parts,
        "dominant_component": dominant_component,
        "explanation": explanation
    }

def generate_forensic_explanation(
    verdict: str,
    p_fake: float,
    pct_spatial: float,
    pct_temporal: float,
    mean_parts: Dict[str, float],
    dominant_component: str,
    n_clips: int
) -> str:
    """
    Constructs a human-readable, evidence-based forensic explanation dynamically
    from the actual DFD-FCG model outputs.
    """
    if verdict == "LIKELY MANIPULATED":
        evidence_summary = []
        for k, v in mean_parts.items():
            if v > 0.15:
                evidence_summary.append(f"{k} (+{v:.2f})")
        ev_str = ", ".join(evidence_summary) if evidence_summary else f"{dominant_component} (+{mean_parts.get(dominant_component, 0):.2f})"
        
        branch_desc = (
            f"The spatial branch drove {pct_spatial:.1f}% of the decision, revealing prominent synthetic boundaries and textural anomalies "
            f"concentrated around the {ev_str}. The temporal branch accounted for {pct_temporal:.1f}%, indicating frame-to-frame inconsistencies."
            if pct_spatial >= 50.0 else
            f"The temporal branch contributed {pct_temporal:.1f}% of the evidence, highlighting unnatural motion dynamics across the 3-second clips. "
            f"Concurrently, the spatial branch ({pct_spatial:.1f}%) observed anomalies centered primarily on the {dominant_component}."
        )
        return (
            f"Forensic analysis classified this video as LIKELY MANIPULATED with P(FAKE) = {p_fake:.1%} across {n_clips} evaluated clips. "
            f"{branch_desc} The evidence consistently exceeds the configured manipulation threshold."
        )
    elif verdict == "LIKELY AUTHENTIC":
        temporal_strong = pct_temporal > 60.0
        return (
            f"Forensic analysis classified this video as LIKELY AUTHENTIC with P(FAKE) = {p_fake:.1%} across {n_clips} clips. "
            f"{'Temporal consistency across video frames was the primary authentic indicator (' + f'{pct_temporal:.1f}% share).' if temporal_strong else 'Both spatial textures and motion cues exhibit natural physiological patterns.'} "
            f"The facial components showed coherent natural landmarks with no significant synthesis artifacts detected on the {dominant_component}."
        )
    else:
        return (
            f"Forensic analysis returned an INCONCLUSIVE verdict with P(FAKE) = {p_fake:.1%}. "
            f"The model's probability falls within the uncertain range (0.30 ≤ P < 0.70). Neither branch provided sufficient statistical confidence "
            f"to confirm manipulation or verify authenticity. Potential factors include high compression, motion blur, or subtle blending."
        )
