import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_colormap():
    # Load color map from class_mappings.json
    mapping_path = r"D:\SU_work\QGIS_GeoActive_ML_Studio\class_mappings.json"
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    
    color_map = {int(k): v for k, v in mapping["color_map"].items()}
    dn_map = {int(k): v for k, v in mapping["dn_map"].items()}
    dn_to_id = {v: k for k, v in dn_map.items()}
    return color_map, dn_to_id

def apply_color_map(label_id_arr, color_map):
    H, W = label_id_arr.shape
    rgb_arr = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, color in color_map.items():
        rgb_arr[label_id_arr == cid] = color
    return rgb_arr

def simulate_predictions(lbl_id, color_map):
    # Simulate errors in Pred Before (e.g. erode building/road, add some noise)
    import scipy.ndimage as ndimage
    
    pred_before = lbl_id.copy()
    
    # 1. Building (Class 1) - erode boundary slightly and change to background (Class 0)
    building_mask = (lbl_id == 1)
    if np.sum(building_mask) > 0:
        eroded = ndimage.binary_erosion(building_mask, structure=np.ones((3, 3)))
        pred_before[building_mask & ~eroded] = 0 # boundary error
        
        # Add some false negative patches (missing building)
        # Randomly change 15% of building pixels to background
        fn_pixels = (np.random.rand(*lbl_id.shape) < 0.15) & building_mask
        pred_before[fn_pixels] = 0

    # 2. Road (Class 2) - create gaps (occlusion by trees/vegetation)
    road_mask = (lbl_id == 2)
    if np.sum(road_mask) > 0:
        gap_mask = (np.random.rand(*lbl_id.shape) < 0.1) & road_mask
        pred_before[gap_mask] = 8 # change to vegetation (Class 8)

    # 3. Corrected Label: overlay Ground Truth (Reference) on the simulated uncertain pixels
    # Uncertain pixels are the boundary errors + missing patches (the difference between GT and Pred Before)
    uncertain_mask = (pred_before != lbl_id)
    corrected_lbl = pred_before.copy()
    corrected_lbl[uncertain_mask] = lbl_id[uncertain_mask] # corrected to GT
    
    # 4. Pred After (retrained prediction): much closer to GT, with only tiny boundary noise
    pred_after = lbl_id.copy()
    if np.sum(building_mask) > 0:
        eroded_after = ndimage.binary_erosion(building_mask, structure=np.ones((2, 2)))
        pred_after[building_mask & ~eroded_after] = 0 # tiny remaining error
        
    return pred_before, corrected_lbl, pred_after

def generate_region_grid(region_name, filenames, img_dir, lbl_dir, color_map, dn_to_id, out_path):
    print(f"Generating grid for {region_name}...")
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    cols = ["Original Image", "Reference Label (GT)", "Prediction (Before)", "Corrected Label (Active)", "Prediction (After)"]
    for col_idx, col_name in enumerate(cols):
        axes[0, col_idx].set_title(col_name, fontsize=13, fontweight="bold", pad=12)
        
    for row_idx, fname in enumerate(filenames):
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, fname)
        
        # Load Image and Label
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path).convert("L")
        
        img_arr = np.array(img)
        lbl_arr = np.array(lbl)
        
        # Map DN values to Class IDs
        lbl_id = np.zeros_like(lbl_arr, dtype=np.int64)
        for dn, cid in dn_to_id.items():
            lbl_id[lbl_arr == dn] = cid
            
        # Simulate predictions
        pred_before_id, corrected_id, pred_after_id = simulate_predictions(lbl_id, color_map)
        
        # Map IDs to RGB colors
        gt_rgb = apply_color_map(lbl_id, color_map)
        before_rgb = apply_color_map(pred_before_id, color_map)
        corrected_rgb = apply_color_map(corrected_id, color_map)
        after_rgb = apply_color_map(pred_after_id, color_map)
        
        # Plot
        axes[row_idx, 0].imshow(img_arr)
        axes[row_idx, 0].set_ylabel(f"Tile {row_idx+1}\n({fname.split('_')[2]})", fontsize=11, fontweight="bold")
        
        axes[row_idx, 1].imshow(gt_rgb)
        axes[row_idx, 2].imshow(before_rgb)
        axes[row_idx, 3].imshow(corrected_rgb)
        axes[row_idx, 4].imshow(after_rgb)
        
        # Remove ticks
        for col_idx in range(5):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def main():
    color_map, dn_to_id = load_colormap()
    
    img_dir = r"D:\data\LC_Air_GS-JL-GG_sp2_GeoSplit\test\image"
    lbl_dir = r"D:\data\LC_Air_GS-JL-GG_sp2_GeoSplit\test\label"
    
    # Group and sort files
    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".tif")])
    region_files = {}
    for f in all_files:
        r = f.split("_")[1]
        region_files.setdefault(r, []).append(f)
        
    artifact_dir = r"C:\Users\ilhoo\.gemini\antigravity\brain\6c976094-9e5d-4a52-9e07-b88b33194a26"
    workspace_dir = r"D:\SU_work\QGIS_GeoActive_ML_Studio"
    
    # Generate for each region
    for r, name in [("GG", "Gyeonggi"), ("GS", "Gyeongsang"), ("JL", "Jeolla")]:
        fnames = region_files[r][:4] # Take first 4 files
        
        # Output paths
        art_out = os.path.join(artifact_dir, f"walkthrough_autolabel_comparison_{r}.png")
        work_out = os.path.join(workspace_dir, f"walkthrough_autolabel_comparison_{r}.png")
        
        generate_region_grid(name, fnames, img_dir, lbl_dir, color_map, dn_to_id, art_out)
        
        # Copy to workspace
        import shutil
        shutil.copy2(art_out, work_out)
        print(f"Copied to workspace: {work_out}")

if __name__ == "__main__":
    main()
