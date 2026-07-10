import os
import argparse
import json
import numpy as np
from PIL import Image
import math

def calculate_weights():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.mapping_file, 'r') as f:
        mappings = json.load(f)
        dn_map = mappings.get("dn_map", {})
        
    norm_root = os.path.normpath(args.dataset_dir)
    if os.path.basename(norm_root) in ["train", "val", "test"]:
        root_dir = os.path.dirname(norm_root)
    else:
        root_dir = args.dataset_dir
        
    lbl_dir = os.path.join(root_dir, "train", "label")
    if not os.path.isdir(lbl_dir):
        fallback_lbl = os.path.join(root_dir, "label")
        if os.path.isdir(fallback_lbl):
            lbl_dir = fallback_lbl
        else:
            print(json.dumps({"error": f"Label directory not found. Checked: {lbl_dir} and {fallback_lbl}"}))
            return
    labels = sorted([f for f in os.listdir(lbl_dir) if f.endswith(('.tif', '.tiff', '.png'))])
    
    Image.MAX_IMAGE_PIXELS = None
    dn_to_id = {v: int(k) for k, v in dn_map.items()}
    class_counts = {int(k): 0 for k in dn_map.keys()}
    
    total_imgs = len(labels)
    for i, lbl_file in enumerate(labels):
        if i == 0 or (i + 1) % 100 == 0:
            print(f"Processing image {i+1}/{total_imgs} ...")
            
        if i % max(1, total_imgs // 100) == 0:
            print(f"PROGRESS:{int((i / total_imgs) * 100)}", flush=True)
            
        lbl_path = os.path.join(lbl_dir, lbl_file)
        lbl = Image.open(lbl_path).convert("L")
        lbl_arr = np.array(lbl, dtype=np.int32)
        
        # Count pixels for each DN
        unique_vals, counts = np.unique(lbl_arr, return_counts=True)
        for val, count in zip(unique_vals, counts):
            if val in dn_to_id:
                class_counts[dn_to_id[val]] += int(count)
                
    # Calculate Inverse Frequency Weights
    total_pixels = sum(class_counts.values())
    if total_pixels == 0:
        print(json.dumps({"error": "No pixels found in train set."}))
        return
        
    weights = {}
    for cid, count in class_counts.items():
        if cid == 0: continue # Skip background
        if count == 0:
            weights[cid] = 10.0 # Max weight if no pixels found
        else:
            freq = count / total_pixels
            # Heuristic: base weight 1.0. For rare classes (freq 0.01), W=2.0.
            # freq 0.001 -> W=3.0. freq 0.0001 -> W=4.0.
            w = -math.log10(freq)
            # Clip between 0.1 and 10.0, round to 1 decimal place
            w = max(0.1, min(10.0, round(w, 1)))
            weights[cid] = w
            
    print(json.dumps(weights))

if __name__ == "__main__":
    calculate_weights()
