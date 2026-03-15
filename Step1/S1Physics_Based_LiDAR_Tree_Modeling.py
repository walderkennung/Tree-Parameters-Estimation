'''This is a physics-informed machine learning algorithm that automatically 
models individual tree crowns from 3D LiDAR point clouds. It fits a prolate spheroid
 to each tree while estimating foliage density using the Beer-Lambert
law of light attenuation.'''


#1. Imports & Setup
#Uses PyTorch for differentiable optimization and GPU acceleration.
import torch
import torch.nn as nn
import torch.optim as optim
import laspy
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Physics-based Tree Model (constant λ, soft crown membership)
class TreeModel(nn.Module):
    '''
          This defines the "parameters" the model will try to learn
          used Log values so the optimizer can work with any number,
          but the radius always stays positive when we calculate exp(log_r).
    '''
    def __init__(self, init_center, init_radii):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_center, dtype=torch.float32))  #The 3D center of the tree crown.
        #The horizontal and vertical radii-Dimensions of a 3D shape
        self.log_rxy = nn.Parameter(torch.log(torch.tensor(float(init_radii[0]), dtype=torch.float32)))
        self.log_rz = nn.Parameter(torch.log(torch.tensor(float(init_radii[2]), dtype=torch.float32)))
        # leaf density parameter
        self.log_lambda = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def radii(self):
        '''
         Adam optimizer tries adding or subtracting numbers to find the best fit.it might 
         subtract from the radius that the radius becomes a negative number
         A negative radius is physically impossible and will cause the math equations crash.
        '''
        rxy = torch.exp(self.log_rxy)
        rz = torch.exp(self.log_rz)
        return rxy, rz

    def inside_weight(self, points): 
        #does this point belong to this specific tree?
        """
        Soft membership in crown spheroid.
        points: (N,3)
        returns: (N,) in (0,1)
        """
        mu = self.mu
        rxy, rz = self.radii()
        #Normalization of the space
        dx = (points[:, 0] - mu[0]) / (rxy + 1e-6)
        dy = (points[:, 1] - mu[1]) / (rxy + 1e-6)
        dz = (points[:, 2] - mu[2]) / (rz + 1e-6)
        #Calculating the Squared Distance
        #If dist_sq < 1.0: The point is Inside the spheroid.
        #If dist_sq > 1.0: The point is Outside the spheroid.
        dist_sq = dx * dx + dy * dy + dz * dz
        # smooth boundary: ~1 inside, ~0 outside
        return torch.sigmoid(4.0 * (1.0 - dist_sq))
    
    #LiDAR Likelihood Model (Physics)
    def forward(self, points):
        
        inside = self.inside_weight(points)  #Membership Weight: Is the point inside the spheroid? (0 to 1)                      
        lam = torch.exp(self.log_lambda) + 1e-12  #λ-Leaf Density scalar > 0
        lam_at_point = lam * inside   #λ⋅w(x) density at a specific point in space.                          

        # approximate crown top
        _, rz = self.radii()
        z_top = self.mu[2] + rz  #The absolute highest point of the crown geometry.                                    # crown top height-scalar

        vertical_path = torch.clamp(z_top - points[:, 2], min=0.0)  #Depth: The distance from the top of the tree down to the point.  # (N,)

        #make survival term consistent with "inside"
        optical_depth = lam * inside * vertical_path  #Total leaf material the laser must pass through to reach the point.               # (N,)

        likelihood = lam_at_point * torch.exp(-optical_depth)  #The chance of the laser hitting that specific point.      # (N,)
        return likelihood + 1e-12



#Optimize one tree

def optimize_single_tree(points_np, n_epochs=300):
    """
    Returns dict with center, radius_xy, radius_z, lambda, loss, z_top.
    """
    points = torch.tensor(points_np, dtype=torch.float32, device=device) #convert to tensor
    if points.shape[0] < 15: #If tree has fewer than 15 points Skip it.
        return None

    # robust "visible canopy top"
    z_vals = points[:, 2]
    z_min = z_vals.min().item() #lowest point
    z_max = z_vals.max().item() #Highest point
    z_p95 = torch.quantile(z_vals, 0.95).item() #true top
    visible_height = max(z_max - z_min, 0.5) #Calculates how tall the cloud of points is

    # initialization: vertical radius,3D Center,horizontal raduis.

    #initial vertical radius
    radius_z_init = max(visible_height / 2.0, 1.0) #half the crown height
    center_z_init = z_min + radius_z_init #center is halfway between bottom and top
    # Extract Horizontal Coordinates from point cloud in two seperate lists
    x_vals, y_vals = points[:, 0], points[:, 1]


    #Full 3D Center Initialization
    '''
    Purpose: Creates initial 3D center μ = (μ_x, μ_y, μ_z)
    X-center: Mean of all X coordinates (μ_x)
    Y-center: Mean of all Y coordinates (μ_y)
    Z-center: Already calculated center_z_init
    Assumption: Crown center ≈ centroid of LiDAR points
    this will happen at the start only for every crown_id.
    '''
    center_init = [x_vals.mean().item(), y_vals.mean().item(), center_z_init]
    
    #Measures horizontal extent of points
    x_span = (x_vals.max() - x_vals.min()).item() #east-west spread
    y_span = (y_vals.max() - y_vals.min()).item() #north-south spread
    #Horizontal Radius Initialization
    radius_xy_init = max((x_span + y_span) / 4.0, 1.0) #Average of x_span and y_span divided by 2
    
    #Create the Model
    model = TreeModel(center_init, [radius_xy_init, radius_xy_init, radius_z_init]).to(device)

    #the algorithm that updates the parameters
    optimizer = optim.Adam(
        [
            {"params": model.mu, "lr": 0.02},
            {"params": [model.log_rxy, model.log_rz], "lr": 0.08},
            {"params": model.log_lambda, "lr": 0.01},
        ]
    )

    best_loss = float("inf") #Tracking Best Results
    best = None

    #Training Loop
    for _ in range(n_epochs):
        optimizer.zero_grad()

        #Forward Pass & Negative Log-Likelihood
        lik = model(points) #lik: Calls TreeModel.forward(),returns likelihood for each LiDAR point
        nll = -torch.log(lik).mean() #nll: Negative Log-Likelihood = main loss term

        #Force model to reach actual tree height
        rxy, rz = model.radii()
        z_top = model.mu[2] + rz
        #penalize only if model top is BELOW the robust top
        top_penalty = 5.0 * torch.relu(torch.tensor(z_p95, device=device) - z_top) ** 2

        #Regularize λ
        lam = torch.exp(model.log_lambda) #Prevent λ from exploding to unrealistic values
        reg_lambda = 1e-3 * (model.log_lambda ** 2)  # weaker + stable

        # --- Optional: prevent collapse to extremely tiny crowns ---
        # (keeps radii from going to ~0 in degenerate cases)
        min_rxy = 0.5
        min_rz = 0.8
        rxy_pen = 2.0 * torch.relu(torch.tensor(min_rxy, device=device) - rxy) ** 2
        rz_pen = 2.0 * torch.relu(torch.tensor(min_rz, device=device) - rz) ** 2
        


        #Backpropagation & Optimization
        total_loss = nll + top_penalty + reg_lambda + rxy_pen + rz_pen
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_val = float(total_loss.item())
        if loss_val < best_loss:
            best_loss = loss_val
            with torch.no_grad():
                rxy_v, rz_v = model.radii()
                best = {
                    "center": model.mu.detach().cpu().numpy(),
                    "radius_xy": float(rxy_v.detach().cpu().item()),
                    "radius_z": float(rz_v.detach().cpu().item()),
                    "lambda": float(torch.exp(model.log_lambda).detach().cpu().item()),
                    "loss": loss_val,
                    "z_top": float((model.mu[2] + rz_v).detach().cpu().item()),
                    "z_p95": float(z_p95),
                }

    return best



# Main 
if __name__ == "__main__":
    input_laz = "cd2th_0_25_ch2th_0_5.laz"
    output_gpkg = "refined_traunstein_mle_physics_fixedH_v2_5000.gpkg"

    print("Reading LAZ...")
    las = laspy.read(input_laz)

    # Get CRS from LAS if present
    out_crs = None
    try:
        out_crs = las.header.parse_crs()
    except Exception:
        out_crs = None

    # Choose segmentation id field
    if hasattr(las, "crown_id"):
        seg = las.crown_id
    elif hasattr(las, "user_data"):
        seg = las.user_data
    else:
        raise RuntimeError("No crown_id or user_data field found for segmentation IDs.")

    crown_ids = np.unique(seg)
    crown_ids = crown_ids[crown_ids > 0]

    # Subset for testing
    crown_ids = crown_ids[:2000]
    print(f"Processing {len(crown_ids)} crowns...")

    results = []
    '''
    Every time this loop runs, it identifies a unique group of points (one tree).
      It then calls optimize_single_tree,
      which creates a fresh, independent spheroid.
    '''
    for cid in tqdm(crown_ids, desc="Optimizing"):
        mask = seg == cid
        p_np = np.vstack([las.x[mask], las.y[mask], las.z[mask]]).T
        if p_np.shape[0] < 15:
            continue

        params = optimize_single_tree(p_np, n_epochs=300)
        if not params:
            continue

        # use model top as "LiDAR-visible height proxy"
        tree_height_proxy = params["z_top"]

        results.append(
            {
                "crown_id": int(cid),
                "x": float(params["center"][0]),
                "y": float(params["center"][1]),
                "z": float(params["center"][2]),
                "radius_x": float(params["radius_xy"]),
                "radius_y": float(params["radius_xy"]),
                "radius_z": float(params["radius_z"]),
                "lambda": float(params["lambda"]),
                "loss": float(params["loss"]),
                "z_top": float(params["z_top"]),
                "z_p95": float(params["z_p95"]),
                "tree_height_proxy": float(tree_height_proxy),
                "n_points": int(p_np.shape[0]),
                "z_max_points": float(p_np[:, 2].max()),
                "z_min_points": float(p_np[:, 2].min()),
            }
        )

    if not results:
        raise RuntimeError("No trees processed (results empty). Check segmentation IDs and point counts.")

    df = pd.DataFrame(results)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=out_crs)

    # If CRS missing in LAS, set it explicitly here (only if you KNOW it)
    if gdf.crs is None:
        # Example fallback; change if your LAS is different
        gdf = gdf.set_crs("EPSG:32633")

    gdf.to_file(output_gpkg, driver="GPKG")

    print("\n=== SUMMARY ===")
    print(f"Saved: {output_gpkg}")
    print(f"Trees written: {len(gdf)}")
    print(f"Mean height proxy (z_top): {df['tree_height_proxy'].mean():.2f} m")
    print(f"Median height proxy (z_top): {df['tree_height_proxy'].median():.2f} m")
    print(f"Mean radius_z: {df['radius_z'].mean():.2f} m")
    print(f"Mean lambda: {df['lambda'].mean():.2f}")
