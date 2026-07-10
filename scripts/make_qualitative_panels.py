import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import random

def load_mapping(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dn_map = data.get('dn_map', {})
    dn_to_id = {int(v): int(k) for k, v in dn_map.items()}
    return dn_to_id, len(dn_to_id)

def main():
    test_dir = r"H:\22_database\LC_Air_GS-JL-GG_sp2\test"
    res_dir = os.path.join(test_dir, "Swin_Mask2Former_InferenceResults", "uncertainty_data")
    out_dir = os.path.join(test_dir, "Qualitative_Panels")
    mapping_path = r"D:\SU_work\QGIS_GeoActive_ML_Studio\class_mappings.json"
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load DN mapping
    dn_to_id, num_classes = load_mapping(mapping_path)
    
    npz_files = glob.glob(os.path.join(res_dir, "*.npz"))
    if not npz_files:
        print(f"No npz files found in {res_dir}")
        return
        
    # Select 10 random files
    random.seed(42)
    selected_npz = random.sample(npz_files, min(10, len(npz_files)))
    
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    for i, npz_path in enumerate(selected_npz):
        basename = os.path.basename(npz_path)
        base_no_ext = os.path.splitext(basename)[0]
        
        # Load Image
        img_path = os.path.join(test_dir, "image", base_no_ext + ".tif")
        if not os.path.exists(img_path):
            img_path = os.path.join(test_dir, "image", base_no_ext + ".png")
            
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            print(f"Image not found for {base_no_ext}")
            continue
            
        # Load npz
        d = np.load(npz_path)
        pred = d['pred_id']
        unc = d['std']  # Ensemble STD
        
        # Load Label
        lbl_path = os.path.join(test_dir, "label", base_no_ext + ".tif")
        if not os.path.exists(lbl_path):
            lbl_path = os.path.join(test_dir, "label", base_no_ext + ".png")
            
        if os.path.exists(lbl_path):
            lbl_raw = np.array(Image.open(lbl_path).convert("L"))
            gt = np.zeros_like(lbl_raw, dtype=np.int64)
            for dn, cid in dn_to_id.items():
                gt[lbl_raw == dn] = cid
        else:
            gt = np.zeros_like(pred)
            
        # Error Map (Pred != GT) excluding ignore label (0)
        valid = (gt != 0)
        error_map = np.zeros_like(pred, dtype=np.float32)
        error_map[valid & (pred != gt)] = 1.0
        
        # Plotting 1x5 Panels: Image | GT | Pred | Error | Uncertainty
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[2].set_title("Prediction (Mean)")
        axes[2].axis('off')
        
        axes[3].imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        axes[3].set_title("Error Map (Red=False)")
        axes[3].axis('off')
        
        im_unc = axes[4].imshow(unc, cmap='jet', vmin=0, vmax=unc.max())
        axes[4].set_title("Uncertainty (STD)")
        axes[4].axis('off')
        
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"panel_{i+1:02d}_{base_no_ext}.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
