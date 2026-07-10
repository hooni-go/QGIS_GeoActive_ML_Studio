import os
import torch
from torch.utils.data import Dataset
import numpy as np

try:
    import rasterio
    from PIL import Image
except ImportError:
    pass

class RemoteSensingDataset(Dataset):
    def __init__(self, root_dir, split="train", is_16bit=False, dn_map=None, transform=None):
        """
        root_dir: Dataset Root
        split: 'train', 'val', or 'test'
        """
        # If the user accidentally selected a split folder (e.g., .../train) as the root, step back one directory
        norm_root = os.path.normpath(root_dir)
        if os.path.basename(norm_root) in ["train", "val", "test"]:
            root_dir = os.path.dirname(norm_root)
            
        self.img_dir = os.path.join(root_dir, split, "image")
        self.lbl_dir = os.path.join(root_dir, split, "label")
        
        # If standard split folders don't exist, try looking directly in root_dir
        if not os.path.isdir(self.img_dir) or not os.path.isdir(self.lbl_dir):
            fallback_img = os.path.join(root_dir, "image")
            fallback_lbl = os.path.join(root_dir, "label")
            if os.path.isdir(fallback_img) and os.path.isdir(fallback_lbl):
                self.img_dir = fallback_img
                self.lbl_dir = fallback_lbl
            else:
                raise FileNotFoundError(f"Directory not found. Expected 'image' and 'label' folders in either '{os.path.join(root_dir, split)}' or '{root_dir}'.")
                
        self.is_16bit = is_16bit
        self.split = split
        self.transform = transform
        
        # dn_map: dict mapping Label ID -> DN in mask image. e.g. {0: 100, 1: 10, ...}
        # We need to invert this to map DN -> Label ID for training
        self.dn_to_id = {v: int(k) for k, v in dn_map.items()} if dn_map else {}
            
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
        self.labels = sorted([f for f in os.listdir(self.lbl_dir) if f.endswith(('.tif', '.tiff', '.png'))])
        
        if len(self.images) != len(self.labels):
            raise ValueError(f"Mismatch in number of images ({len(self.images)}) and labels ({len(self.labels)}) in {split}.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        lbl_path = os.path.join(self.lbl_dir, self.labels[idx])
        
        # Load Image
        if self.is_16bit:
            with rasterio.open(img_path) as src:
                # rasterio reads as (C, H, W)
                img = src.read()
                # Transpose to (H, W, C) for transforms
                img = np.transpose(img, (1, 2, 0)).astype(np.float32)
                # Normalize 16-bit to [0, 1] approximately (max val is 65535, but usually data is 12-14 bit)
                # A robust scaling might be needed, here we use simple min-max or divide by 65535.0
                img = img / 65535.0
        else:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0
            
        # Load Label
        # Mask is usually 1-band 8-bit even for 16-bit imagery
        if self.is_16bit and lbl_path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(lbl_path) as src:
                lbl = src.read(1).astype(np.int32)
        else:
            lbl = Image.open(lbl_path).convert("L")
            lbl = np.array(lbl, dtype=np.int32)
        
        # Map DN values to Class IDs (0, 1, 2...)
        mapped_lbl = np.zeros_like(lbl, dtype=np.int64)
        if self.dn_to_id:
            for dn_val, class_id in self.dn_to_id.items():
                mapped_lbl[lbl == dn_val] = class_id
        else:
            mapped_lbl = lbl # Fallback
            
        if self.transform:
            transformed = self.transform(image=img, mask=mapped_lbl)
            img = transformed['image']
            mapped_lbl = transformed['mask']
            
        # Convert to Tensor
        if isinstance(img, np.ndarray):
            # (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float()
            
        mapped_lbl = torch.from_numpy(mapped_lbl).long()
        
        # 1. Data Augmentation (Train only)
        import random
        if self.split == "train":
            if random.random() > 0.5:
                # Horizontal Flip
                img = torch.flip(img, dims=[2])
                mapped_lbl = torch.flip(mapped_lbl, dims=[1])
            if random.random() > 0.5:
                # Vertical Flip
                img = torch.flip(img, dims=[1])
                mapped_lbl = torch.flip(mapped_lbl, dims=[0])
                
            # Random Rotate 90, 180, 270 (위성영상은 방향성이 없으므로 매우 효과적)
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = torch.rot90(img, rot_k, [1, 2])
                mapped_lbl = torch.rot90(mapped_lbl, rot_k, [0, 1])
                
            # Random Brightness (빛의 노출 변화 모사, 0.8배 ~ 1.2배)
            # 3-band, 4-band 모두 호환되도록 단순 스케일링 적용
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                img = torch.clamp(img * brightness_factor, 0.0, 1.0)
                
        # 2. ImageNet Normalization (Required for pre-trained Swin-Large backbone)
        # Swin-Large was trained on ImageNet with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        if self.is_16bit:
            # 4-band: use ImageNet for RGB, and 0.5 for NIR
            mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225, 0.225]).view(4, 1, 1)
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
        img = (img - mean) / std
        
        # We return a dict compatible with transformers / custom models
        return {
            "pixel_values": img,
            "labels": mapped_lbl,
            "filename": self.images[idx]
        }
