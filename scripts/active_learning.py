import os
import glob
import numpy as np
import shutil
from PIL import Image

def analyze_uncertainty(unc_dir, threshold=0.15, ratio_threshold=0.05, uncertainty_method="bald", budget=0.20):
    """
    Scans the uncertainty data directory, calculates mean uncertainty of each tile using the
    selected uncertainty method (BALD, Entropy, STD), and selects the top tiles based on the budget.
    """
    if not os.path.exists(unc_dir):
        return {
            "total_files": 0,
            "candidate_count": 0,
            "candidate_files": [],
            "mean_uncertainty": 0.0,
            "status": "no_data"
        }
        
    npz_files = [os.path.join(unc_dir, f) for f in os.listdir(unc_dir) if f.endswith(".npz")]
    all_tiles = []
    total_uncertainty_sum = 0.0
    
    # Map friendly method names to npz keys
    method_key = uncertainty_method.lower()
    if method_key not in ["bald", "entropy", "std", "conf"]:
        method_key = "bald" # Default to BALD
        
    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            if method_key not in data or "filename" not in data:
                continue
            unc_map = data[method_key]
            std_map = data["std"]
            filename = str(data["filename"])
            
            mean_unc = float(np.mean(unc_map))
            total_uncertainty_sum += mean_unc
            
            # For ratio tracking (micro-level STD)
            uncertain_pixels = np.sum(std_map > threshold)
            ratio = float(uncertain_pixels) / std_map.size
            
            all_tiles.append({
                "filename": filename,
                "npz_path": npz_path,
                "mean_uncertainty": mean_unc,
                "ratio": ratio
            })
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            
    # Sort tiles by mean uncertainty in descending order (highest uncertainty first)
    all_tiles.sort(key=lambda x: x["mean_uncertainty"], reverse=True)
    
    # Select top files according to the active learning budget (e.g. 20%)
    num_to_select = max(1, int(len(all_tiles) * budget)) if all_tiles else 0
    candidate_files = all_tiles[:num_to_select]
    
    mean_uncertainty = total_uncertainty_sum / len(npz_files) if npz_files else 0.0
    
    return {
        "total_files": len(npz_files),
        "candidate_count": len(candidate_files),
        "candidate_files": candidate_files,
        "mean_uncertainty": mean_uncertainty,
        "status": "retraining_recommended" if len(candidate_files) > 0 else "model_stable"
    }

def auto_correct_labels(candidate_files, img_dir, dataset_dir, threshold=0.15, ref_dir=None, strategy="uncertainty"):
    """
    Performs pixel-level label correction on candidate files using reference ground-truth labels
    under a specific active learning refinement strategy for academic comparative evaluation.
    
    Parameters:
    -----------
    candidate_files : list
        List of dictionaries containing candidate tile metadata.
    img_dir : str
        Path to the target input image directory.
    dataset_dir : str
        Path to the top-level dataset directory to save the retraining split.
    threshold : float
        Uncertainty threshold (standard deviation) to determine uncertain pixels (proposed method).
    ref_dir : str, optional
        Path to the reference ground truth labels acting as the human annotator (oracle).
    strategy : str
        The active learning pixel selection strategy:
        - "uncertainty": Queries and corrects pixels where prediction uncertainty (std) > threshold.
        - "random": Queries and corrects the same number of pixels randomly to act as a baseline.
        - "full": Queries and corrects the entire tile (100% annotation, upper bound).
        
    Returns:
    --------
    int : The number of successfully processed and corrected tiles.
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
    strategy = strategy.lower()
    
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
            # Load NPZ containing model prediction and voxel-level uncertainty
            data = np.load(npz_path)
            pred_id = data["pred_id"]
            std_map = data["std"]
            
            # Load reference label using PIL (acting as human oracle)
            ref_img = Image.open(src_ref_path)
            ref_arr = np.array(ref_img)
            
            # Create corrected label array, starting from model prediction
            corrected_arr = pred_id.copy()
            
            # Formulate the mask based on active learning strategy
            k = int(np.sum(std_map > threshold))
            
            if strategy in ["uncertainty", "std"]:
                # Proposed: Select pixels where standard deviation exceeds threshold
                correction_mask = (std_map > threshold)
                
            elif strategy == "bald":
                # Pure BALD: Select top K pixels with highest BALD values
                bald_map = data["bald"]
                flat_indices = np.argpartition(bald_map.ravel(), -k)[-k:] if k > 0 else []
                bald_mask = np.zeros(bald_map.size, dtype=bool)
                if len(flat_indices) > 0:
                    bald_mask[flat_indices] = True
                correction_mask = bald_mask.reshape(bald_map.shape)
                
            elif strategy == "entropy":
                # Pure Entropy: Select top K pixels with highest Entropy values
                entropy_map = data["entropy"]
                flat_indices = np.argpartition(entropy_map.ravel(), -k)[-k:] if k > 0 else []
                entropy_mask = np.zeros(entropy_map.size, dtype=bool)
                if len(flat_indices) > 0:
                    entropy_mask[flat_indices] = True
                correction_mask = entropy_mask.reshape(entropy_map.shape)
                
            elif strategy == "random":
                # Baseline: Select the exact same number of pixels randomly (Same Annotation Budget)
                flat_indices = np.random.choice(std_map.size, k, replace=False) if k > 0 else []
                random_mask = np.zeros(std_map.size, dtype=bool)
                if len(flat_indices) > 0:
                    random_mask[flat_indices] = True
                correction_mask = random_mask.reshape(std_map.shape)
                
            elif strategy == "full":
                # Upper Bound: Select all pixels for 100% correction
                correction_mask = np.ones_like(std_map, dtype=bool)
                
            else:
                print(f"Warning: Unknown active learning strategy '{strategy}'. Defaulting to uncertainty.")
                correction_mask = (std_map > threshold)
            
            # Apply corrections: Overwrite selected pixels with reference ground truth
            corrected_arr[correction_mask] = ref_arr[correction_mask]
            
            # If reference label indicates invalid/background (0), enforce 0 mapping
            corrected_arr[ref_arr == 0] = 0
            
            # Save corrected label to train/label split
            dst_lbl_path = os.path.join(train_lbl_dir, filename)
            Image.fromarray(corrected_arr.astype(np.uint8)).save(dst_lbl_path)
            
            # Copy original image to train/image split
            dst_img_path = os.path.join(train_img_dir, filename)
            shutil.copy2(src_img_path, dst_img_path)
            corrected_count += 1
            
            print(f"[Active Learning] Corrected {filename} using '{strategy}' strategy. Labeled pixels: {np.sum(correction_mask)} / {std_map.size}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return corrected_count
