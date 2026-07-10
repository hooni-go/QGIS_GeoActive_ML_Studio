import os
import glob
import numpy as np
import shutil
from PIL import Image

def analyze_uncertainty(unc_dir, threshold=0.15, ratio_threshold=0.05):
    """
    Scans NPZ files in the uncertainty_data directory and determines which files need retraining.
    """
    npz_files = glob.glob(os.path.join(unc_dir, "*.npz"))
    if not npz_files:
        return {
            "total_files": 0,
            "candidate_count": 0,
            "candidate_files": [],
            "mean_uncertainty": 0.0,
            "status": "no_data"
        }
        
    candidate_files = []
    total_uncertainty_sum = 0.0
    
    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            if "std" not in data or "filename" not in data:
                continue
            std_map = data["std"]
            filename = str(data["filename"])
            
            mean_std = float(np.mean(std_map))
            total_uncertainty_sum += mean_std
            
            # Pixels exceeding threshold
            uncertain_pixels = np.sum(std_map > threshold)
            ratio = float(uncertain_pixels) / std_map.size
            
            if ratio > ratio_threshold:
                candidate_files.append({
                    "filename": filename,
                    "npz_path": npz_path,
                    "mean_std": mean_std,
                    "ratio": ratio
                })
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            
    mean_uncertainty = total_uncertainty_sum / len(npz_files) if npz_files else 0.0
    
    return {
        "total_files": len(npz_files),
        "candidate_count": len(candidate_files),
        "candidate_files": candidate_files,
        "mean_uncertainty": mean_uncertainty,
        "status": "retraining_recommended" if len(candidate_files) > 0 else "model_stable"
    }

def auto_correct_labels(candidate_files, img_dir, dataset_dir, threshold=0.15, ref_dir=None):
    """
    Performs pixel-level label correction on candidate files using reference labels.
    Copies corrected labels and original images into the train split.
    """
    # Locate reference label directory
    if ref_dir and os.path.isdir(ref_dir):
        ref_label_dir = ref_dir
    else:
        parent_dir = os.path.dirname(os.path.normpath(img_dir))
        ref_label_dir = os.path.join(parent_dir, "label")
    
    if not os.path.isdir(ref_label_dir):
        print(f"Warning: Reference label directory not found at {ref_label_dir}. Skipping auto-correction.")
        return 0
        
    train_img_dir = os.path.join(dataset_dir, "train", "image")
    train_lbl_dir = os.path.join(dataset_dir, "train", "label")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    
    corrected_count = 0
    
    for candidate in candidate_files:
        filename = candidate["filename"]
        npz_path = candidate["npz_path"]
        
        # Paths
        src_img_path = os.path.join(img_dir, filename)
        src_ref_path = os.path.join(ref_label_dir, filename)
        
        if not os.path.exists(src_img_path) or not os.path.exists(src_ref_path):
            print(f"Warning: Image or reference label missing for {filename}. Skipping.")
            continue
            
        try:
            # Load NPZ
            data = np.load(npz_path)
            pred_id = data["pred_id"]
            std_map = data["std"]
            
            # Load reference label using PIL
            ref_img = Image.open(src_ref_path)
            ref_arr = np.array(ref_img)
            
            # Create corrected label (copy pred_id first)
            corrected_arr = pred_id.copy()
            
            # Logic: If std > threshold and reference label is valid, correct it
            correction_mask = (std_map > threshold)
            
            # Apply corrections
            corrected_arr[correction_mask] = ref_arr[correction_mask]
            
            # If reference label is 0 (ignore), set to 0
            corrected_arr[ref_arr == 0] = 0
            
            # Save corrected label to train/label
            dst_lbl_path = os.path.join(train_lbl_dir, filename)
            Image.fromarray(corrected_arr.astype(np.uint8)).save(dst_lbl_path)
            
            # Copy original image to train/image
            dst_img_path = os.path.join(train_img_dir, filename)
            shutil.copy2(src_img_path, dst_img_path)
            corrected_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return corrected_count
