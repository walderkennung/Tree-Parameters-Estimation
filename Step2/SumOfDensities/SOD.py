"""
STEP 2 (Multiple trees, known K): occlusion-aware, unsegmented LiDAR

Goal:
- Fit K tree crowns simultaneously to an unsegmented point cloud.
- Uses a physical first-hit likelihood:
    p(x,y,z) = Λ(x,y,z) * exp( - ∫_z^∞ Λ(x,y,z') dz' )
  where Λ = sum_i λ_i(x,y,z) and contributions add up if multiple crowns overlap.

Key idea:
- Each tree i is an axis-aligned spheroid (rxy_i, rz_i, center mu_i).
- Leaf density inside crown is λ_i (constant) times a soft "inside" weight.
- Occlusion is modeled via optical depth above the point along the vertical line at (x,y).

Inputs:
- A "full scene" LAZ/LAS (prefer: 03_w_overlap_wo_inconsistent_returns)
- No segmentation field needed.

Outputs:
- GeoPackage with K optimized trees: center (x,y,z), rxy, rz, lambda

Dependencies:
pip install torch laspy[lazrs] numpy pandas geopandas shapely tqdm
"""

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
import matplotlib.pyplot as plt
    

# ----------------------------
# Config (edit these)
# ----------------------------
INPUT_LAZ = r"D:\HTWK\3\Object und Gesterkenung Project\LIDAR-Project\LIDAR-Project\Traunstein_ForestGEO\las_data\preprocessed\02_clipped_to_inventory_area\2018\clipped.laz"  # put your Step-2 source file here (prefer from 03_...)
OUTPUT_GPKG = r"D:\HTWK\3\Object und Gesterkenung Project\Trees_project\S1_imp1\S888.gpkg"
INIT_GPKG = r"D:\HTWK\3\Object und Gesterkenung Project\Trees_project\Final\S1Physics_Based_LiDAR_Tree_Modeling_2000.gpkg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spatial subset (optional) to make Step 2 feasible:
USE_BBOX = True

#BBOX 
BBOX = dict(
    xmin=325100, 
    xmax=325200,   # A 100-meter wide slice
    ymin=5311600, 
    ymax=5311700    # A 100-meter deep slice
)

# Sampling for speed
MAX_POINTS = 120_000          # total points used in optimization
KEEP_GROUND = True            # keep ground points (recommended for Step 2 constraints)
GROUND_CLASS = 2              # ASPRS ground classification usually 2

# Model size (known number of trees for Step 2)
K_TREES = 60            # start small (10–40) for a tile

# Optimization
EPOCHS = 1500
LR = 0.03
INSIDE_SHARPNESS = 6.0        # sigmoid sharpness for soft inside
EPS = 1e-12

# Regularization (keep things sane, but not “hard penalties”)
LAMBDA_REG = 1e-4             # keeps lambda from exploding
RADIUS_REG = 1e-4             # small radius regularization
MIN_RXY = 0.6
MAX_RXY = 8.0
MIN_RZ = 1.5
MAX_RZ = 18.0


# ----------------------------
# Utilities
# ----------------------------
def read_las_xyz(input_path: str):
    print(f"Opening file: {input_path}")
    las = laspy.read(input_path)

    # Calculate boundaries
    x_min, x_max = las.header.min[0], las.header.max[0]
    y_min, y_max = las.header.min[1], las.header.max[1]
    z_min, z_max = las.header.min[2], las.header.max[2]

    print("-" * 30)
    print(f"FILE BOUNDS REPORT:")
    print(f"  X Range: {x_min:.2f} to {x_max:.2f}")
    print(f"  Y Range: {y_min:.2f} to {y_max:.2f}")
    print(f"  Z Range: {z_min:.2f} to {z_max:.2f}")
    print(f"  Total Points in File: {len(las.points):,}")
    print("-" * 30)

    x = np.array(las.x)
    y = np.array(las.y)
    z = np.array(las.z)

    cls = None
    if hasattr(las, "classification"):
        cls = np.array(las.classification)

    return x, y, z, cls


def apply_bbox(x, y, z, cls, bbox):
    m = (x >= bbox["xmin"]) & (x <= bbox["xmax"]) & (y >= bbox["ymin"]) & (y <= bbox["ymax"])
    x, y, z = x[m], y[m], z[m]
    cls = cls[m] if cls is not None else None
    return x, y, z, cls


def subsample_points(x, y, z, cls, max_points):
    n = len(x)
    if n <= max_points:
        return x, y, z, cls
    idx = np.random.choice(n, size=max_points, replace=False)
    x, y, z = x[idx], y[idx], z[idx]
    cls = cls[idx] if cls is not None else None
    return x, y, z, cls


def init_tree_centers_from_canopy(points_xyz: np.ndarray, k: int):
    """
    Simple initializer:
    - take high points (top canopy) and pick k centers by greedy farthest-point sampling in XY.
    This avoids needing segmentation/CHM for a prototype.
    """
    xyz = points_xyz
    z = xyz[:, 2]
    z_thr = np.quantile(z, 0.85)  # top canopy subset
    top = xyz[z >= z_thr]
    if len(top) < k:
        top = xyz

    xy = top[:, :2]
    # start from random point
    centers = [xy[np.random.randint(len(xy))]]
    # greedy farthest point
    for _ in range(1, k):
        d2 = np.min([np.sum((xy - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(xy[np.argmax(d2)])
    centers = np.stack(centers, axis=0)

    # initial z center around mid of local points (rough)
    z0 = np.median(xyz[:, 2])
    mu0 = np.column_stack([centers, np.full(k, z0, dtype=np.float32)])
    return mu0.astype(np.float32)


# ----------------------------
# Step 2 Model
# ----------------------------
class MultiTreeSpheroid(nn.Module):
    """
    K trees. Each tree i:
      mu_i = (mx,my,mz)
      rxy_i > 0
      rz_i  > 0
      lam_i > 0

    Λ(point) = sum_i lam_i * w_i(point)
    optical_depth(point) ≈ sum_i lam_i * L_i(x,y,z)   (vertical path length inside crown above z)

    likelihood(point) = Λ(point) * exp(-optical_depth(point))
    """

    def __init__(self, mu_init: np.ndarray, rxy_init=3.0, rz_init=8.0, lam_init=2.0):
        super().__init__()
        k = mu_init.shape[0]
        self.k = k

        self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))  # (K,3)

        self.log_rxy = nn.Parameter(torch.log(torch.full((k,), float(rxy_init), dtype=torch.float32)))
        self.log_rz = nn.Parameter(torch.log(torch.full((k,), float(rz_init), dtype=torch.float32)))
        self.log_lam = nn.Parameter(torch.log(torch.full((k,), float(lam_init), dtype=torch.float32)))

    def forward(self, points: torch.Tensor):
        """
        points: (N,3) in meters
        returns: likelihood per point (N,)
        """
        N = points.shape[0]
        K = self.k

        rxy = torch.clamp(torch.exp(self.log_rxy), min=MIN_RXY,max=MAX_RXY)  # (K,)
        rz = torch.clamp(torch.exp(self.log_rz), min=MIN_RZ,max=MAX_RZ)     # (K,)
        lam = torch.exp(self.log_lam)                            # (K,)

        # Broadcast shapes:
        # points -> (N,1,3), mu -> (1,K,3)
        p = points[:, None, :]           # (N,K,3)
        mu = self.mu[None, :, :]         # (N,K,3)

        dx = (p[..., 0] - mu[..., 0]) / (rxy[None, :] + 1e-6)
        dy = (p[..., 1] - mu[..., 1]) / (rxy[None, :] + 1e-6)
        dz = (p[..., 2] - mu[..., 2]) / (rz[None, :] + 1e-6)

        dist_sq = dx * dx + dy * dy + dz * dz  # (N,K)

        # Soft inside weight w_i(point)
        w = torch.sigmoid(INSIDE_SHARPNESS * (1.0 - dist_sq))  # (N,K)

        # ---- Optical depth approximation along vertical line at (x,y) ----
        # For axis-aligned spheroid, at fixed (x,y):
        # cross_section = 1 - (dx_xy^2 + dy_xy^2)
        dx_xy = (p[..., 0] - mu[..., 0]) / (rxy[None, :] + 1e-6)
        dy_xy = (p[..., 1] - mu[..., 1]) / (rxy[None, :] + 1e-6)
        u = dx_xy * dx_xy + dy_xy * dy_xy  # (N,K)

        # If u >= 1, vertical line does not pass through the crown => path length 0
        inside_xy = (u < 1.0).float()

        # z_top(x,y) = mz + rz * sqrt(1 - u)
        sqrt_term = torch.sqrt(torch.clamp(1.0 - u, min=0.0))  # (N,K)
        z_top = mu[..., 2] + rz[None, :] * sqrt_term           # (N,K)

        # vertical path above point inside crown: max(0, z_top - z)
        dz_above = torch.clamp(z_top - p[..., 2], min=0.0) * inside_xy  # (N,K)

        # optical depth sums contributions from all trees
        optical_depth = torch.sum(lam[None, :] * dz_above, dim=1)  # (N,)

        # Λ(point) sums local densities at the point
        Lambda_point = torch.sum(lam[None, :] * w, dim=1)  # (N,)

        likelihood = Lambda_point * torch.exp(-optical_depth)
        return likelihood + EPS, (rxy, rz, lam)


# ----------------------------
# Main: run Step 2
# ----------------------------
def main():
    np.random.seed(0)
    torch.manual_seed(0)

    print(f"Device: {DEVICE}")

    # 1) Load data
    print("Reading LAZ/LAS...")
    x, y, z, cls = read_las_xyz(INPUT_LAZ)

    if USE_BBOX:
        x, y, z, cls = apply_bbox(x, y, z, cls, BBOX)
        print(f"After BBOX: {len(x)} points")

    # Optionally keep ground (recommended)
    if cls is not None and not KEEP_GROUND:
        m = cls != GROUND_CLASS
        x, y, z = x[m], y[m], z[m]
        cls = cls[m]
        print(f"After removing ground: {len(x)} points")

    # Subsample
    x, y, z, cls = subsample_points(x, y, z, cls, MAX_POINTS)
    print(f"Using {len(x)} points for optimization")

    pts = np.column_stack([x, y, z]).astype(np.float32)

    # 2) Initialize K trees
    mu0 = init_tree_centers_from_canopy(pts, K_TREES)
    # crude initial radii
    rxy0 = 3.0
    rz0 = max(6.0, float(np.quantile(z, 0.95) - np.quantile(z, 0.10)) / 2.0)
    lam0 = 2.0

    model = MultiTreeSpheroid(mu0, rxy_init=rxy0, rz_init=rz0, lam_init=lam0).to(DEVICE)

    points_t = torch.tensor(pts, dtype=torch.float32, device=DEVICE)

    # 3) Optimize (maximize likelihood => minimize NLL)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Optimizing (Step 2)...")
    for epoch in tqdm(range(EPOCHS)):
        optimizer.zero_grad()

        lik, (rxy, rz, lam) = model(points_t)
        nll = -torch.log(lik).mean()

        # gentle regularization (not “extra penalties”, just stability)
        reg = LAMBDA_REG * torch.mean(model.log_lam ** 2) + RADIUS_REG * (
            torch.mean(model.log_rxy ** 2) + torch.mean(model.log_rz ** 2)
        )

        loss = nll + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                print(
                    f"Epoch {epoch:04d} | NLL={nll.item():.4f} | "
                    f"rxy(mean)={rxy.mean().item():.2f} rz(mean)={rz.mean().item():.2f} lam(mean)={lam.mean().item():.2f}"
                )

    # 4) Export results
    # 4) Export results
    with torch.no_grad():
        mu = model.mu.detach().cpu().numpy()
        rxy = torch.clamp(torch.exp(model.log_rxy), min=MIN_RXY).detach().cpu().numpy()
        rz = torch.clamp(torch.exp(model.log_rz), min=MIN_RZ).detach().cpu().numpy()
        lam = torch.exp(model.log_lam).detach().cpu().numpy()

    df = pd.DataFrame({
        "tree_id": np.arange(len(mu), dtype=int),
        "x": mu[:, 0], "y": mu[:, 1], "z": mu[:, 2],
        "radius_xy": rxy, "radius_z": rz, "lambda": lam,
        "z_top": mu[:, 2] + rz,
    })

    # --- COORDINATE HANDLING ---
    # 1. Define the local system (Meters). For Traunstein, this is usually 25832.
    local_crs = "EPSG:25832" 
    
    # 2. Create the GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["x"], df["y"])],
        crs=local_crs
    )

    # 3. OPTIONAL: Convert to EPSG:4326 (Degrees) 
    # Only do this if you specifically need Lat/Lon for a web map.
    # gdf = gdf.to_crs("EPSG:4326")

    # Save to file (This creates the file if it doesn't exist)
    gdf.to_file(OUTPUT_GPKG, driver="GPKG")
    print(f"\nSaved successfully to: {OUTPUT_GPKG}")

    

    # subsample points for plotting
    pts_xy = pts[:, :2]
    idx = np.random.choice(len(pts_xy), size=min(30000, len(pts_xy)), replace=False)

    plt.figure(figsize=(7,7))
    plt.scatter(pts_xy[idx,0], pts_xy[idx,1], s=1, alpha=0.2)
    plt.scatter(df["x"].values, df["y"].values, s=60, marker="x")
    plt.axis("equal")
    plt.title("Step 2: LiDAR XY (subsample) + estimated tree centers")
    plt.show()


if __name__ == "__main__":
    main()

