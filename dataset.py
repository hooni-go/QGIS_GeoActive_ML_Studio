# -*- coding: utf-8 -*-
"""
PyTorch Dataset module for QGIS_GeoActive ML Studio.
Ensures deterministic ordering of files for academic reproducibility.
"""

import os
from PIL import Image
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoImageProcessor
except ImportError:
    pass # Will be handled by check_and_install_dependencies

class GeoActiveDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor=None, img_size=(512, 512), id2label=None):
        """
        Args:
            image_dir (str): Path to the image directory.
            label_dir (str): Path to the label directory (mask images).
            processor: Hugging Face AutoImageProcessor.
            img_size (tuple): Target image size.
            id2label (dict): Mapping from pixel value to class name.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.img_size = img_size
        self.id2label = id2label or {0: "Background", 1: "Graffiti"}
        
        # 파일 이름을 정렬하여(Sort) 항상 동일한 순서로 데이터가 로드되도록 보장 (재현성)
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        
        # 이미지와 라벨의 개수가 일치하는지 확인
        assert len(self.image_filenames) == len(self.label_filenames), "이미지와 라벨 파일의 개수가 일치하지 않습니다."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        label_name = self.label_filenames[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert("RGB")
        
        # 라벨은 일반적으로 1채널 (Grayscale)
        # 안티그래피티 라벨을 0(배경), 1(그래피티) 등 정수 클래스 인덱스로 변환해야 합니다.
        segmentation_map = Image.open(label_path).convert("L")
        
        # 리사이즈 (간단한 Nearest Neighbor 보간법 사용하여 클래스 값 유지)
        image = image.resize(self.img_size, Image.BILINEAR)
        segmentation_map = segmentation_map.resize(self.img_size, Image.NEAREST)
        
        # NumPy 배열 변환
        segmentation_map = np.array(segmentation_map, dtype=np.int32)
        
        # 이전처럼 (segmentation_map > 128) 로 강제 0,1 이진화하지 않고,
        # 사용자가 입력한 클래스 매핑(id2label)의 픽셀값을 그대로 따르도록 원본을 유지합니다.
        # (단, 라벨 이미지가 0, 1, 2 등의 클래스 인덱스로 구성된 경우에 한함)
        
        if self.processor is not None:
            # Mask2Former Processor 적용
            encoded_inputs = self.processor(image, segmentation_map, return_tensors="pt")
            # Return squeezed tensors to match Dataset expected format (batching is done by DataLoader)
            for k, v in encoded_inputs.items():
                encoded_inputs[k] = v.squeeze()
            return encoded_inputs
        else:
            return image, segmentation_map

def collate_fn(batch):
    """
    Mask2Former DataLoader를 위한 collate_fn
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }
