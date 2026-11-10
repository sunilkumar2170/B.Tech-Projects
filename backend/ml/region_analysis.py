import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Fix (2026-11-06): aggregate_region_attributions now accepts an optional
# temporal_patch_attributions argument so that spatial and temporal regional
# attributions are genuinely independent (previously both columns were identical).
# Also added dominant_temporal_region computation in model_adapter live inference.

GRID = 16
CROP = 150
REGIONS = ["eyes", "nose", "lips", "skin", "background"]

# 68-point landmark indices (dlib / 2D-FAN convention)
LM = {
    "eyes": list(range(36, 48)) + list(range(17, 27)), # eyes + eyebrows
    "nose": list(range(27, 36)),
    "lips": list(range(48, 68)),
}

def create_patch_region_map(landmarks: np.ndarray, grid: int = GRID, crop: int = CROP, near: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assigns each of the grid*grid (256) patches to one of 5 regions:
    eyes, nose, lips, skin, background.
    """
    step = crop / grid
    centres = np.array([[(c + 0.5) * step, (r + 0.5) * step]
                        for r in range(grid) for c in range(grid)])
    
    # Facial convex hull for skin vs background
    try:
        hull = ConvexHull(landmarks)
        inside = MplPath(landmarks[hull.vertices])
    except Exception:
        inside = None

    labels = []
    for p in centres:
        best_name, best_dist = None, 1e9
        for name, idxs in LM.items():
            d = np.linalg.norm(landmarks[idxs] - p, axis=1).min()
            if d < best_dist:
                best_name, best_dist = name, d
        if best_dist <= near:
            labels.append(best_name)
        elif inside is not None and inside.contains_point(p):
            labels.append("skin")
        else:
            labels.append("background")
            
    return np.array(labels).reshape(grid, grid), centres

def aggregate_region_attributions(
    patch_attributions: np.ndarray,
    region_map: np.ndarray,
    temporal_patch_attributions: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregates per-patch attributions (16x16) into 5 regions.
    Computes signed sum, absolute share %, density (attribution / patch count).
    
    Args:
        patch_attributions: (GRID, GRID) spatial attribution values
        region_map: (GRID, GRID) string array of region labels
        temporal_patch_attributions: optional (GRID, GRID) temporal attributions.
            If provided, the temporal_signed / temporal_share_pct fields are
            computed independently from the temporal patch values.
    """
    region_signed = {}
    region_signed_t = {}
    patch_counts = {}
    
    for r in REGIONS:
        mask = (region_map == r)
        count = int(np.sum(mask))
        patch_counts[r] = count
        if count > 0:
            val_s = float(np.sum(patch_attributions[mask]))
            val_t = (float(np.sum(temporal_patch_attributions[mask]))
                     if temporal_patch_attributions is not None else val_s)
        else:
            val_s = 0.0
            val_t = 0.0
        region_signed[r] = val_s
        region_signed_t[r] = val_t
        
    # Normalized share % among face regions (eyes, nose, lips, skin)
    face_regions = ["eyes", "nose", "lips", "skin"]
    face_abs_sum_s = sum(abs(region_signed[r]) for r in face_regions)
    if face_abs_sum_s == 0:
        face_abs_sum_s = 1e-9

    face_abs_sum_t = sum(abs(region_signed_t[r]) for r in face_regions)
    if face_abs_sum_t == 0:
        face_abs_sum_t = 1e-9
        
    shares_s = {}
    shares_t = {}
    densities = {}
    for r in REGIONS:
        if r in face_regions:
            shares_s[r] = 100.0 * abs(region_signed[r]) / face_abs_sum_s
            shares_t[r] = 100.0 * abs(region_signed_t[r]) / face_abs_sum_t
        else:
            shares_s[r] = 0.0
            shares_t[r] = 0.0
        cnt = max(patch_counts[r], 1)
        densities[r] = region_signed[r] / cnt
        
    result = {}
    for r in REGIONS:
        val_signed = round(region_signed[r], 4)
        val_share = round(shares_s[r], 2)
        val_signed_t = round(region_signed_t[r], 4)
        val_share_t = round(shares_t[r], 2)
        val_density = round(densities[r], 5)
        result[r] = {
            "name": r,
            "patch_count": patch_counts[r],
            "signed": val_signed,
            "share_pct": val_share,
            "spatial_signed": val_signed,
            "spatial_share_pct": val_share,
            "temporal_signed": val_signed_t,
            "temporal_share_pct": val_share_t,
            "density": val_density
        }
    return result

def generate_heatmap_overlay(
    face_crop_rgb: np.ndarray,
    patch_values_16x16: np.ndarray,
    output_path: Path,
    title: str = "Attribution Heatmap",
    cmap_name: str = "coolwarm"
) -> str:
    """
    Renders an attribution overlay on top of the 150x150 face crop
    and saves it as a high-resolution PNG image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    ax.imshow(face_crop_rgb)
    
    # Normalize patch values around 0
    max_abs = max(float(np.abs(patch_values_16x16).max()), 1e-5)
    
    # Overlay heatmap
    im = ax.imshow(
        patch_values_16x16,
        cmap=cmap_name,
        alpha=0.55,
        extent=[0, face_crop_rgb.shape[1], face_crop_rgb.shape[0], 0],
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="bilinear"
    )
    
    # Grid overlay
    grid_size = patch_values_16x16.shape[0]
    step_x = face_crop_rgb.shape[1] / grid_size
    step_y = face_crop_rgb.shape[0] / grid_size
    for i in range(1, grid_size):
        ax.axvline(i * step_x, color="white", alpha=0.15, lw=0.5)
        ax.axhline(i * step_y, color="white", alpha=0.15, lw=0.5)
        
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10, pad=6)
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(output_path)
