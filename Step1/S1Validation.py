import geopandas as gpd
import pandas as pd
import numpy as np
import laspy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Reference LAZ
# ----------------------------
inv_path = r"D:\HTWK\3\Object und Gesterkenung Project\LIDAR-Project\LIDAR-Project\Traunstein_ForestGEO\las_data\preprocessed\99_inventory_area_normalized\2018\inventory_plot_normalized.las"
print("Reading Reference LAZ file...")
inv_las = laspy.read(inv_path)

inv_ids = np.unique(inv_las.user_data)
inv_ids = inv_ids[inv_ids > 0] 

inv_list = []
for cid in inv_ids:
    mask = (inv_las.user_data == cid)
    inv_list.append({
        "ref_crown_id": int(cid),
        "x_inv": np.mean(inv_las.x[mask]), 
        "y_inv": np.mean(inv_las.y[mask]),
        "height_inv": np.max(inv_las.z[mask]) 
    })

df_inv = pd.DataFrame(inv_list)
print(f"Extracted {len(df_inv)} reference trees.")

# ----------------------------
# 2. Load and Force Align MLE Results
# ----------------------------
mle_path = r"D:\HTWK\3\Object und Gesterkenung Project\Trees_project\refined_traunstein_mle_physics_fixedH_v2.gpkg"
gdf_mle = gpd.read_file(mle_path)

# Ensure coordinate columns exist
gdf_mle["x_mle"] = gdf_mle.geometry.x
gdf_mle["y_mle"] = gdf_mle.geometry.y

# --- THE STABLE NUDGE FIX ---
# Instead of mean-to-mean, let's look at the Bounding Box of the Inventory
buffer = 2.0
min_x, max_x = df_inv["x_inv"].min(), df_inv["x_inv"].max()
min_y, max_y = df_inv["y_inv"].min(), df_inv["y_inv"].max()

# Find MLE trees that fall inside the Inventory square
mle_in_plot = gdf_mle[
    (gdf_mle["x_mle"] >= min_x - buffer) & (gdf_mle["x_mle"] <= max_x + buffer) &
    (gdf_mle["y_mle"] >= min_y - buffer) & (gdf_mle["y_mle"] <= max_y + buffer)
]

if len(mle_in_plot) > 5:
    x_shift = df_inv["x_inv"].mean() - mle_in_plot["x_mle"].mean()
    y_shift = df_inv["y_inv"].mean() - mle_in_plot["y_mle"].mean()
    print(f"Applying Stable Nudge based on {len(mle_in_plot)} local trees.")
else:
    # If coordinates are drastically different, the filter fails. 
    # Use global mean as a last resort.
    x_shift = df_inv["x_inv"].mean() - gdf_mle["x_mle"].mean()
    y_shift = df_inv["y_inv"].mean() - gdf_mle["y_mle"].mean()
    print("WARNING: Using Global Nudge (Subsetting failed). Check coordinate systems!")

print(f"Nudge: X={x_shift:.2f}m, Y={y_shift:.2f}m")

gdf_mle["x_mle_aligned"] = gdf_mle["x_mle"] + x_shift
gdf_mle["y_mle_aligned"] = gdf_mle["y_mle"] + y_shift

# ----------------------------
# 3. Unique (Greedy) Spatial Matching
# ----------------------------
inv_xy = df_inv[["x_inv", "y_inv"]].values
mle_xy = gdf_mle[["x_mle_aligned", "y_mle_aligned"]].values

print("Calculating unique spatial matches...")
dist_mat = distance_matrix(inv_xy, mle_xy)

matched_indices_inv = []
matched_indices_mle = []
used_mle = set()
MAX_DIST = 5.0 

# Potential pairs: (distance, inv_idx, mle_idx)
potential_pairs = []
for i in range(len(inv_xy)):
    for j in range(len(mle_xy)):
        d = dist_mat[i, j]
        if d <= MAX_DIST:
            potential_pairs.append((d, i, j))

potential_pairs.sort() # Sort by closest distance first

for d, i, j in potential_pairs:
    if i not in matched_indices_inv and j not in used_mle:
        matched_indices_inv.append(i)
        matched_indices_mle.append(j)
        used_mle.add(j)

# ----------------------------
# 4. Evaluation & Detailed Output
# ----------------------------
if len(matched_indices_inv) > 0:
    df_inv_matched = df_inv.iloc[matched_indices_inv].copy().reset_index(drop=True)
    gdf_mle_matched = gdf_mle.iloc[matched_indices_mle].copy().reset_index(drop=True)
    
    # Create the evaluation dataframe
    df_eval = pd.concat([df_inv_matched, gdf_mle_matched], axis=1)
    
    # Calculate Height (Z_center + Radius_Z)
    df_eval["height_mle"] = df_eval["z"] + df_eval["radius_z"]
    height_err = df_eval["height_mle"] - df_eval["height_inv"]
    
    # Calculate horizontal distance of the match (how far the nudge was off)
    match_distances = np.sqrt(
        (df_eval["x_inv"] - df_eval["x_mle_aligned"])**2 + 
        (df_eval["y_inv"] - df_eval["y_mle_aligned"])**2
    )

    print("\n" + "="*40)
    print(f"UNIQUE MATCHED TREES: {len(df_eval)} / {len(df_inv)}")
    print(f"Height RMSE:           {np.sqrt(np.mean(height_err**2)):.2f} m")
    print(f"Height Bias:           {np.mean(height_err):.2f} m")
    print(f"Avg Match Offset:      {np.mean(match_distances):.2f} m")
    print("="*40)
    
    # --- YOUR REQUESTED TABLE ---
    print("\n--- FIRST 10 UNIQUE MATCHES ---")
    print(f"{'Ref_H':>7} | {'MLE_Z':>7} | {'Rad_Z':>7} | {'MLE_H':>7} | {'Dist':>5}")
    print("-" * 45)
    for i in range(min(10, len(df_eval))):
        row = df_eval.iloc[i]
        d = match_distances.iloc[i]
        print(f"{row['height_inv']:7.2f} | {row['z']:7.2f} | {row['radius_z']:7.2f} | {row['height_mle']:7.2f} | {d:5.1f}m")

    # Save results
    df_eval.to_csv("mle_validation_results_full.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(df_eval["height_inv"], df_eval["height_mle"], alpha=0.6, color='forestgreen', label='Matched Trees')
    plt.plot([20, 50], [20, 50], 'r--', label="1:1 Perfect Match")
    plt.xlabel("Inventory Reference Height (m)")
    plt.ylabel("Physics Model Predicted Height (m)")
    plt.title(f"Validation: {len(df_eval)} Trees Matched")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("\n[!] MATCHING FAILED: No trees found within 5m search radius.")
    print(f"Inventory Center: {df_inv['x_inv'].mean():.1f}, {df_inv['y_inv'].mean():.1f}")
    print(f"MLE Center:       {gdf_mle['x_mle'].mean():.1f}, {gdf_mle['y_mle'].mean():.1f}")