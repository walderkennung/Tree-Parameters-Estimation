import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import laspy
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import matplotlib.pyplot as plt  # Added at the top to avoid errors

# ----------------------------
# Config (edit these)
# ----------------------------
INPUT_LAZ = r"C:\Users\saeed.k\Desktop\clipped.laz"
OUTPUT_GPKG = "step2_multi_tree_fitvaaaal.gpkg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_BBOX = True
BBOX = dict(
    xmin=325100, 
    xmax=325200,   
    ymin=5311600, 
    ymax=5311700    
)

MAX_POINTS = 120_000          
KEEP_GROUND = True            
GROUND_CLASS = 2              

K_TREES = 40               

EPOCHS = 400
LR = 0.03
INSIDE_SHARPNESS = 6.0        
EPS = 1e-12

LAMBDA_REG = 1e-4             
RADIUS_REG = 1e-4             
MIN_RXY = 0.6
MAX_RXY = 8.0
MIN_RZ = 1.5
MAX_RZ = 22.0

# ----------------------------
# Utilities
# ----------------------------
def read_las_xyz(input_path: str):
    print(f"Opening file: {input_path}")
    las = laspy.read(input_path)
    x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)
    cls = np.array(las.classification) if hasattr(las, "classification") else None
    return x, y, z, cls

def apply_bbox(x, y, z, cls, bbox):
    m = (x >= bbox["xmin"]) & (x <= bbox["xmax"]) & (y >= bbox["ymin"]) & (y <= bbox["ymax"])
    return x[m], y[m], z[m], (cls[m] if cls is not None else None)

def subsample_points(x, y, z, cls, max_points):
    n = len(x)
    if n <= max_points: return x, y, z, cls
    idx = np.random.choice(n, size=max_points, replace=False)
    return x[idx], y[idx], z[idx], (cls[idx] if cls is not None else None)

def init_tree_centers_from_canopy(points_xyz: np.ndarray, k: int):
    xyz = points_xyz
    z = xyz[:, 2]
    z_thr = np.quantile(z, 0.85)
    top = xyz[z >= z_thr]
    if len(top) < k: top = xyz
    xy = top[:, :2]
    centers = [xy[np.random.randint(len(xy))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((xy - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(xy[np.argmax(d2)])
    centers = np.stack(centers, axis=0)
    z0 = np.median(xyz[:, 2])
    return np.column_stack([centers, np.full(k, z0, dtype=np.float32)]).astype(np.float32)

# ----------------------------
# Step 2 Model
# ----------------------------
class MultiTreeSpheroid(nn.Module):
    def __init__(self, mu_init: np.ndarray, rxy_init=3.0, rz_init=8.0, lam_init=2.0):
        super().__init__()
        k = mu_init.shape[0]
        self.k = k
        self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))
        self.log_rxy = nn.Parameter(torch.log(torch.full((k,), float(rxy_init), dtype=torch.float32)))
        self.log_rz = nn.Parameter(torch.log(torch.full((k,), float(rz_init), dtype=torch.float32)))
        self.log_lam = nn.Parameter(torch.log(torch.full((k,), float(lam_init), dtype=torch.float32)))

    def forward(self, points: torch.Tensor):
        N = points.shape[0]
        K = self.k
        rxy = torch.clamp(torch.exp(self.log_rxy), min=MIN_RXY, max=MAX_RXY)
        rz = torch.clamp(torch.exp(self.log_rz), min=MIN_RZ, max=MAX_RZ)
        lam = torch.exp(self.log_lam)
        p = points[:, None, :]
        mu = self.mu[None, :, :]
        dx = (p[..., 0] - mu[..., 0]) / (rxy[None, :] + 1e-6)
        dy = (p[..., 1] - mu[..., 1]) / (rxy[None, :] + 1e-6)
        dz = (p[..., 2] - mu[..., 2]) / (rz[None, :] + 1e-6)
        dist_sq = dx * dx + dy * dy + dz * dz
        w = torch.sigmoid(INSIDE_SHARPNESS * (1.0 - dist_sq))
        dx_xy = (p[..., 0] - mu[..., 0]) / (rxy[None, :] + 1e-6)
        dy_xy = (p[..., 1] - mu[..., 1]) / (rxy[None, :] + 1e-6)
        u = dx_xy * dx_xy + dy_xy * dy_xy
        inside_xy = (u < 1.0).float()
        sqrt_term = torch.sqrt(torch.clamp(1.0 - u, min=0.0))
        z_top = mu[..., 2] + rz[None, :] * sqrt_term
        dz_above = torch.clamp(z_top - p[..., 2], min=0.0) * inside_xy
        optical_depth = torch.sum(lam[None, :] * dz_above, dim=1)
        Lambda_point = torch.sum(lam[None, :] * w, dim=1)
        likelihood = Lambda_point * torch.exp(-optical_depth)
        return likelihood + EPS, (rxy, rz, lam)

# ----------------------------
# Main
# ----------------------------
def main():
    np.random.seed(0)
    torch.manual_seed(0)
    print(f"Device: {DEVICE}")

    # 1) Load data
    x, y, z, cls = read_las_xyz(INPUT_LAZ)
    if USE_BBOX:
        x, y, z, cls = apply_bbox(x, y, z, cls, BBOX)
    x, y, z, cls = subsample_points(x, y, z, cls, MAX_POINTS)
    pts = np.column_stack([x, y, z]).astype(np.float32)

    # 2) Initialize
    mu0 = init_tree_centers_from_canopy(pts, K_TREES)
    rz0 = max(6.0, float(np.quantile(z, 0.95) - np.quantile(z, 0.10)) / 2.0)
    model = MultiTreeSpheroid(mu0, rxy_init=3.0, rz_init=rz0, lam_init=2.0).to(DEVICE)
    points_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE)

    # 3) Optimize
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print("Optimizing...")
    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()
        lik, (rxy, rz, lam) = model(points_t)
        nll = -torch.log(lik).mean()
        reg = LAMBDA_REG * torch.mean(model.log_lam ** 2) + RADIUS_REG * (
            torch.mean(model.log_rxy ** 2) + torch.mean(model.log_rz ** 2))
        (nll + reg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    # 4) Export results (FIXED CRS HERE)
    with torch.no_grad():
        mu = model.mu.detach().cpu().numpy()
        rxy = torch.clamp(torch.exp(model.log_rxy), min=MIN_RXY).detach().cpu().numpy()
        rz = torch.clamp(torch.exp(model.log_rz), min=MIN_RZ).detach().cpu().numpy()
        lam = torch.exp(model.log_lam).detach().cpu().numpy()

    df = pd.DataFrame({
        "tree_id": np.arange(K_TREES, dtype=int),
        "x": mu[:, 0], "y": mu[:, 1], "z": mu[:, 2],
        "radius_xy": rxy, "radius_z": rz, "lambda": lam,
        "z_top": mu[:, 2] + rz,
    })

    # Saving with the correct CRS for QGIS
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["x"], df["y"])], crs="EPSG:25832")
    gdf.to_file(OUTPUT_GPKG, driver="GPKG")
    print(f"\nSaved: {OUTPUT_GPKG}")

    # 5) PLOT FOR VERIFICATION
    pts_xy = pts[:, :2]
    idx_plot = np.random.choice(len(pts_xy), size=min(30000, len(pts_xy)), replace=False)
    plt.figure(figsize=(8,8))
    plt.scatter(pts_xy[idx_plot,0], pts_xy[idx_plot,1], s=1, alpha=0.2, color='grey', label='LiDAR')
    plt.scatter(df["x"].values, df["y"].values, s=60, marker="x", color='red', label='AI Centers')
    plt.axis("equal")
    plt.legend()
    plt.title("Visual Check: AI Trees vs LiDAR Points")
    plt.show()

if __name__ == "__main__":
    main()