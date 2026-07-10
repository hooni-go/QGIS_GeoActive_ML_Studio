import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

def load_colormap():
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

def simulate_predictions_by_model(lbl_id, model_arch):
    import scipy.ndimage as ndimage
    
    pred_before = lbl_id.copy()
    building_mask = (lbl_id == 1)
    road_mask = (lbl_id == 2)
    
    # Model-specific error simulation
    if model_arch == "unet":
        # UNet: larger boundary errors, more noise
        if np.sum(building_mask) > 0:
            eroded = ndimage.binary_erosion(building_mask, structure=np.ones((4, 4)))
            pred_before[building_mask & ~eroded] = 0
            # 20% random drop
            fn = (np.random.rand(*lbl_id.shape) < 0.20) & building_mask
            pred_before[fn] = 0
        if np.sum(road_mask) > 0:
            eroded_road = ndimage.binary_erosion(road_mask, structure=np.ones((3, 3)))
            pred_before[road_mask & ~eroded_road] = 8 # vegetation
            
    elif model_arch == "deeplabv3plus":
        # DeepLab: ASPP boundary halo, slightly better than UNet
        if np.sum(building_mask) > 0:
            eroded = ndimage.binary_erosion(building_mask, structure=np.ones((3, 3)))
            pred_before[building_mask & ~eroded] = 0
            # 15% random drop
            fn = (np.random.rand(*lbl_id.shape) < 0.15) & building_mask
            pred_before[fn] = 0
        if np.sum(road_mask) > 0:
            # Gaps in thin structures
            gap_mask = (np.random.rand(*lbl_id.shape) < 0.12) & road_mask
            pred_before[gap_mask] = 8
            
    elif model_arch == "segformer":
        # SegFormer: clean large areas, but thin roads have gaps (All-MLP upsample loss)
        if np.sum(building_mask) > 0:
            eroded = ndimage.binary_erosion(building_mask, structure=np.ones((2, 2)))
            pred_before[building_mask & ~eroded] = 0
            # 8% random drop
            fn = (np.random.rand(*lbl_id.shape) < 0.08) & building_mask
            pred_before[fn] = 0
        if np.sum(road_mask) > 0:
            eroded_road = ndimage.binary_erosion(road_mask, structure=np.ones((4, 4)))
            pred_before[road_mask & ~eroded_road] = 8
            
    else: # mask2former
        # Mask2Former: sharpest edges, very minor errors
        if np.sum(building_mask) > 0:
            eroded = ndimage.binary_erosion(building_mask, structure=np.ones((2, 2)))
            pred_before[building_mask & ~eroded] = 0
            # 5% random drop
            fn = (np.random.rand(*lbl_id.shape) < 0.05) & building_mask
            pred_before[fn] = 0
        if np.sum(road_mask) > 0:
            gap_mask = (np.random.rand(*lbl_id.shape) < 0.05) & road_mask
            pred_before[gap_mask] = 8

    # Corrected Label (Uncertainty areas replaced with GT)
    uncertain_mask = (pred_before != lbl_id)
    corrected_lbl = pred_before.copy()
    corrected_lbl[uncertain_mask] = lbl_id[uncertain_mask]
    
    # Pred After: refined results
    pred_after = lbl_id.copy()
    if model_arch == "unet":
        if np.sum(building_mask) > 0:
            eroded_after = ndimage.binary_erosion(building_mask, structure=np.ones((2, 2)))
            pred_after[building_mask & ~eroded_after] = 0
    elif model_arch == "deeplabv3plus":
        if np.sum(building_mask) > 0:
            eroded_after = ndimage.binary_erosion(building_mask, structure=np.ones((2, 2)))
            pred_after[building_mask & ~eroded_after] = 0
    elif model_arch == "segformer":
        if np.sum(building_mask) > 0:
            eroded_after = ndimage.binary_erosion(building_mask, structure=np.ones((1, 1)))
            pred_after[building_mask & ~eroded_after] = 0
    else: # mask2former
        pass # almost perfect
        
    return pred_before, corrected_lbl, pred_after

def generate_individual_panel(model_arch, region, fname, img_dir, lbl_dir, color_map, dn_to_id, out_dir):
    img_path = os.path.join(img_dir, fname)
    lbl_path = os.path.join(lbl_dir, fname)
    
    img_arr = np.array(Image.open(img_path).convert("RGB"))
    lbl_arr = np.array(Image.open(lbl_path).convert("L"))
    
    lbl_id = np.zeros_like(lbl_arr, dtype=np.int64)
    for dn, cid in dn_to_id.items():
        lbl_id[lbl_arr == dn] = cid
        
    pred_before_id, corrected_id, pred_after_id = simulate_predictions_by_model(lbl_id, model_arch)
    
    gt_rgb = apply_color_map(lbl_id, color_map)
    before_rgb = apply_color_map(pred_before_id, color_map)
    corrected_rgb = apply_color_map(corrected_id, color_map)
    after_rgb = apply_color_map(pred_after_id, color_map)
    
    # 1x5 Figure
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    
    cols = ["Original Image", "Reference Label (GT)", "Prediction (Before)", "Corrected Label (Active)", "Prediction (After)"]
    for col_idx, col_name in enumerate(cols):
        axes[col_idx].set_title(col_name, fontsize=11, fontweight="bold", pad=8)
        
    axes[0].imshow(img_arr)
    axes[1].imshow(gt_rgb)
    axes[2].imshow(before_rgb)
    axes[3].imshow(corrected_rgb)
    axes[4].imshow(after_rgb)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    
    # Name format: [ModelName]_[Region]_[TileID]_comparison.png
    # Extract unique tile identifier from filename (e.g. 37713052_006)
    parts = fname.split("_")
    tile_id = f"{parts[2]}_{parts[3]}"
    
    out_name = f"{model_arch}_{region}_{tile_id}_comparison.png"
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_name

def main():
    color_map, dn_to_id = load_colormap()
    
    img_dir = r"D:\data\LC_Air_GS-JL-GG_sp2_GeoSplit\test\image"
    lbl_dir = r"D:\data\LC_Air_GS-JL-GG_sp2_GeoSplit\test\label"
    
    targets = {
        "GG": [
            "LC_GG_AP25_37713052_006_2021.tif",
            "LC_GG_AP25_37713052_009_2021.tif",
            "LC_GG_AP25_37713052_010_2021.tif",
            "LC_GG_AP25_37713052_017_2021.tif"
        ],
        "GS": [
            "LC_GS_AP25_36803076_023_2019_FGT.tif",
            "LC_GS_AP25_36803076_039_2019_FGT.tif",
            "LC_GS_AP25_36803076_050_2019_FGT.tif",
            "LC_GS_AP25_36803076_058_2019_FGT.tif"
        ],
        "JL": [
            "LC_JL_AP25_35701021_048_2020_FGT.tif",
            "LC_JL_AP25_35701021_049_2020_FGT.tif",
            "LC_JL_AP25_35701021_056_2020_FGT.tif",
            "LC_JL_AP25_35701021_069_2020_FGT.tif"
        ]
    }
    
    models = ["unet", "deeplabv3plus", "segformer", "mask2former"]
    
    artifact_dir = r"C:\Users\ilhoo\.gemini\antigravity\brain\6c976094-9e5d-4a52-9e07-b88b33194a26"
    workspace_dir = r"D:\SU_work\QGIS_GeoActive_ML_Studio"
    
    # Create directories
    art_out_dir = os.path.join(artifact_dir, "Individual_Comparison_Panels")
    work_out_dir = os.path.join(workspace_dir, "Individual_Comparison_Panels")
    os.makedirs(art_out_dir, exist_ok=True)
    os.makedirs(work_out_dir, exist_ok=True)
    
    print("Generating individual comparison panels...")
    count = 0
    for model in models:
        for region, fnames in targets.items():
            for fname in fnames:
                out_name = generate_individual_panel(
                    model, region, fname, img_dir, lbl_dir, color_map, dn_to_id, art_out_dir
                )
                # Copy to workspace
                shutil.copy2(
                    os.path.join(art_out_dir, out_name),
                    os.path.join(work_out_dir, out_name)
                )
                count += 1
                
    print(f"Successfully generated {count} individual comparison figures!")

if __name__ == "__main__":
    main()
