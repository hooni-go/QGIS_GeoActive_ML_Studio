import os
import sys
import argparse
import json

# Add current directory to sys.path to fix ModuleNotFoundError in Embedded Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import datetime
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import RemoteSensingDataset
from models.model_builder import build_model

def fast_hist(a, b, n, ignore_index=0):
    # GT가 ignore_index(미분류, class 0)인 픽셀은 평가에서 제외한다.
    k = (a >= 0) & (a < n) & (a != ignore_index)
    return torch.bincount(n * a[k].to(torch.int64) + b[k].to(torch.int64), minlength=n**2).reshape(n, n)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--is_16bit", type=str, default="False")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--use_focal_loss", action="store_true", help="Enable Auxiliary Focal Loss for hard classes")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=5.0)
    parser.add_argument("--class_weights", type=str, default="")
    parser.add_argument("--model_arch", type=str, default="mask2former")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()

    is_16bit = args.is_16bit.lower() == "true"
    
    # Use Native CUDA (cu128 natively supports sm_120)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.mapping_file, 'r') as f:
        mappings = json.load(f)
        dn_map = mappings.get("dn_map", {})
        
    num_classes = len(dn_map)
    
    # Checkpoints dir naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    if dataset_name.lower() in ["train", "val", "test", "images", "labels"]:
        dataset_name = os.path.basename(os.path.dirname(os.path.normpath(args.dataset_dir)))
        
    model_name = args.model_arch.upper()
    parent_ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", dataset_name)
    ckpt_dir = os.path.join(parent_ckpt_dir, f"{model_name}_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Load Class Weights
    class_weights_dict = {}
    if args.class_weights and os.path.isfile(args.class_weights):
        with open(args.class_weights, 'r') as f:
            class_weights_dict = json.load(f)

    # Save training configuration
    config_dict = {
        "timestamp": timestamp,
        "dataset_dir": args.dataset_dir,
        "is_16bit": is_16bit,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "resume_from": args.resume_from,
        "model": model_name,
        "backbone": "mask2former-swin-large",
        "use_focal_loss": args.use_focal_loss,
        "focal_gamma": args.focal_gamma if args.use_focal_loss else "N/A",
        "focal_alpha": args.focal_alpha if args.use_focal_loss else "N/A",
        "class_weights": class_weights_dict,
        "num_classes": num_classes
    }
    with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    print(f"Dataset: {args.dataset_dir}")
    print(f"Data Type: {'16-bit 4-Band' if is_16bit else '8-bit 3-Band'}")
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    # Datasets
    train_dataset = RemoteSensingDataset(args.dataset_dir, split="train", is_16bit=is_16bit, dn_map=dn_map)
    val_dataset = RemoteSensingDataset(args.dataset_dir, split="val", is_16bit=is_16bit, dn_map=dn_map)
    
    # num_workers=0 설정 (Windows OS의 고질적인 Shared Memory 누수 에러 1455 방지)
    drop_last = (len(train_dataset) > args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Create Class Weights Tensor (1, C, 1, 1) for broadcasting in loss calculation
    class_weight_tensor = torch.ones((num_classes,), device=device, dtype=torch.float32)
    for model_id_str, w_val in class_weights_dict.items():
        mid = int(model_id_str)
        if 0 <= mid < num_classes:
            class_weight_tensor[mid] = float(w_val)
    class_weight_tensor = class_weight_tensor.view(1, -1, 1, 1)

    # Model
    model = build_model(num_classes, is_16bit=is_16bit, model_arch=args.model_arch)
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint (Transfer Learning): {args.resume_from}")
        loaded_sd = torch.load(args.resume_from, map_location="cpu")
        
        # Clean and remap weights for compatibility
        new_state_dict = {}
        for k, v in loaded_sd.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        if "segmentation_head.1.weight" in new_state_dict and "segmentation_head.0.weight" not in new_state_dict:
            if "segmentation_head.0.weight" in model.state_dict():
                print("Remapping checkpoint keys from 'segmentation_head.1' to 'segmentation_head.0' for compatibility.")
                new_state_dict["segmentation_head.0.weight"] = new_state_dict.pop("segmentation_head.1.weight")
                new_state_dict["segmentation_head.0.bias"] = new_state_dict.pop("segmentation_head.1.bias")
                
        if "segmentation_head.0.weight" in new_state_dict and "segmentation_head.1.weight" not in new_state_dict:
            if "segmentation_head.1.weight" in model.state_dict():
                print("Remapping checkpoint keys from 'segmentation_head.0' to 'segmentation_head.1' for compatibility.")
                new_state_dict["segmentation_head.1.weight"] = new_state_dict.pop("segmentation_head.0.weight")
                new_state_dict["segmentation_head.1.bias"] = new_state_dict.pop("segmentation_head.1.bias")
                
        # SegFormer remapping
        if "decode_head.classifier.1.weight" in new_state_dict and "decode_head.classifier.weight" not in new_state_dict:
            if "decode_head.classifier.weight" in model.state_dict():
                print("Remapping checkpoint keys from 'decode_head.classifier.1' to 'decode_head.classifier' for compatibility.")
                new_state_dict["decode_head.classifier.weight"] = new_state_dict.pop("decode_head.classifier.1.weight")
                new_state_dict["decode_head.classifier.bias"] = new_state_dict.pop("decode_head.classifier.1.bias")
                
        if "decode_head.classifier.weight" in new_state_dict and "decode_head.classifier.1.weight" not in new_state_dict:
            if "decode_head.classifier.1.weight" in model.state_dict():
                print("Remapping checkpoint keys from 'decode_head.classifier' to 'decode_head.classifier.1' for compatibility.")
                new_state_dict["decode_head.classifier.1.weight"] = new_state_dict.pop("decode_head.classifier.weight")
                new_state_dict["decode_head.classifier.1.bias"] = new_state_dict.pop("decode_head.classifier.bias")
                
        model.load_state_dict(new_state_dict, strict=False)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Enable Gradient Checkpointing for VRAM optimization
    try:
        model.gradient_checkpointing_enable()
        print("Gradient Checkpointing enabled.")
    except Exception as e:
        print(f"Failed to enable Gradient Checkpointing: {e}")

    # Multi-GPU Support (Windows 환경 및 QGIS 임베디드 안정성을 위해 단일 GPU 사용)
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Detected {n_gpus} GPUs. Using 1 GPU (cuda:0) for stability in QGIS Windows environment.")
    # Single Learning Rate 1e-5 (Benchmarked from H: drive)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    # PolyLR (LambdaLR) Scheduler
    max_steps = args.epochs * len(train_loader)
    lr_lambda = lambda step: (1.0 - step / max_steps) ** 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # AMP Scaler (torch.amp 신 API로 통일)
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n--- Epoch {epoch}/{args.epochs} (LR: {current_lr:.6f}) ---")
        model.train()
        
        # 1-Batch Size 이슈 방지: Batch Size가 1이거나 DeepLabV3+ 등에서 BN 에러를 막기 위해 BN을 eval 모드로 고정
        if args.batch_size == 1:
            for m in model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            current_total_step = (epoch - 1) * len(train_loader) + step
            if current_total_step % max(1, max_steps // 100) == 0:
                print(f"PROGRESS:{int((current_total_step / max_steps) * 100)}", flush=True)
                
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Transformers Mask2Former expects lists of 3D binary masks and 1D class labels
            batch_mask_labels = []
            batch_class_labels = []
            for b in range(pixel_values.shape[0]):
                lbl = labels[b]
                unique_classes = torch.unique(lbl)
                
                b_masks = []
                b_classes = []
                for c in unique_classes:
                    # Class 0을 학습에 포함 (오탐 방지 목적)
                    b_masks.append(lbl == c)
                    b_classes.append(c)
                
                if len(b_masks) > 0:
                    batch_mask_labels.append(torch.stack(b_masks).to(device).float())
                    batch_class_labels.append(torch.tensor(b_classes, dtype=torch.long).to(device))
                else:
                    # 이미지 전체가 미분류인 경우의 폴백 (드묾)
                    batch_mask_labels.append(torch.zeros((1, lbl.shape[0], lbl.shape[1]), device=device).float())
                    batch_class_labels.append(torch.tensor([0], dtype=torch.long).to(device))
            
            optimizer.zero_grad()
            
            # AMP Autocast
            with torch.amp.autocast('cuda'):
                if args.model_arch == "mask2former":
                    outputs = model(pixel_values=pixel_values, mask_labels=batch_mask_labels, class_labels=batch_class_labels)
                    loss = outputs.loss
                    if loss is not None and loss.dim() > 0:
                        loss = loss.mean()
                        
                    # Compute auxiliary focal loss outside autocast context to prevent binary_cross_entropy error
                    if args.use_focal_loss and loss is not None:
                        with torch.autocast(device_type='cuda', enabled=False):
                            # 1. 픽셀맵 복원 (B, C, H, W) - float32로 강제 변환
                            cls_probs = outputs.class_queries_logits.float().softmax(dim=-1)[..., :-1]
                            m_probs = outputs.masks_queries_logits.float().sigmoid()
                            sem_seg = torch.einsum("bqc,bqhw->bchw", cls_probs, m_probs)
                            
                            if sem_seg.shape[-2:] != labels.shape[-2:]:
                                sem_seg = torch.nn.functional.interpolate(sem_seg, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                            
                            # 2. 전체 클래스 추출 (B, C, H, W)
                            sem_seg_fg = sem_seg.to(torch.float32)
                            sem_seg_fg = torch.clamp(sem_seg_fg, 1e-5, 1.0 - 1e-5)
                            
                            # 3. 라벨 인덱싱 및 안전장치 적용
                            labels_fg_safe = torch.clamp(labels, min=0)
                            
                            # 4. 정답 One-hot 벡터화
                            target_one_hot = torch.zeros_like(sem_seg_fg).scatter_(1, labels_fg_safe.unsqueeze(1), 1.0)
                            
                            # 5. BCE & Focal Loss 계산 (- (1-pt)^gamma * log(pt))
                            bce_loss = torch.nn.functional.binary_cross_entropy(sem_seg_fg, target_one_hot, reduction='none')
                            p_t = sem_seg_fg * target_one_hot + (1.0 - sem_seg_fg) * (1.0 - target_one_hot)
                            
                            focal_loss_map = bce_loss * ((1.0 - p_t) ** args.focal_gamma)
                            
                            # 5.5 클래스별 커스텀 가중치 스케일링 적용 (Broadcasting)
                            focal_loss_map = focal_loss_map * class_weight_tensor
                            
                            # 6. 전체 픽셀에 적용
                            aux_focal_loss = focal_loss_map.mean()
                            loss = loss + (aux_focal_loss * args.focal_alpha)
                
                else:
                    # SMP or SegFormer
                    if args.model_arch == "segformer":
                        outputs = model(pixel_values=pixel_values)
                        logits = outputs.logits
                    else:
                        logits = model(pixel_values)
                        
                    if logits.shape[-2:] != labels.shape[-2:]:
                        logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        
                    # Standard Cross Entropy Loss
                    # Include class 0
                    ce_loss_fn = torch.nn.CrossEntropyLoss()
                    loss = ce_loss_fn(logits, labels)
                    
                    if args.use_focal_loss:
                        probs = torch.softmax(logits.float(), dim=1)
                        # Gather probabilities of true classes
                        # labels are [0, 1, ..., C-1]
                        labels_safe = torch.clamp(labels, min=0)
                        true_probs = probs.gather(1, labels_safe.unsqueeze(1)).squeeze(1)
                        focal_term = ((1.0 - true_probs) ** args.focal_gamma)
                        
                        # Apply class weights manually
                        weights_for_pixels = torch.ones_like(labels_safe).float()
                        for mid in range(0, num_classes):
                            weights_for_pixels[labels == mid] = class_weight_tensor.squeeze()[mid]
                            
                        pixel_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
                        focal_pixel_loss = pixel_loss * focal_term * weights_for_pixels * args.focal_alpha
                        
                        loss = loss + focal_pixel_loss.mean()
            
            if loss is None:
                print("Warning: Loss is None!")
                continue
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update scheduler per step for PolyLR
            scheduler.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        # 혼동행렬은 int64로 누적 (float32 정밀도 손실 방지)
        hist = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                batch_mask_labels = []
                batch_class_labels = []
                for b in range(pixel_values.shape[0]):
                    lbl = labels[b]
                    unique_classes = torch.unique(lbl)
                    b_masks = [lbl == c for c in unique_classes if c != 0]
                    b_classes = [c for c in unique_classes if c != 0]
                    
                    if len(b_masks) > 0:
                        batch_mask_labels.append(torch.stack(b_masks).to(device).float())
                        batch_class_labels.append(torch.tensor(b_classes, dtype=torch.long).to(device))
                    else:
                        batch_mask_labels.append(torch.zeros((1, lbl.shape[0], lbl.shape[1]), device=device).float())
                        batch_class_labels.append(torch.tensor([0], dtype=torch.long).to(device))

                if args.model_arch == "mask2former":
                    outputs = model(pixel_values=pixel_values, mask_labels=batch_mask_labels, class_labels=batch_class_labels)
                    if outputs.loss is not None:
                        batch_loss = outputs.loss.mean() if outputs.loss.dim() > 0 else outputs.loss
                        val_loss += batch_loss.item()
                    
                    class_queries_logits = outputs.class_queries_logits
                    masks_queries_logits = outputs.masks_queries_logits
                    mask_cls_probs = class_queries_logits.softmax(dim=-1)[..., :-1]
                    mask_pred = masks_queries_logits.sigmoid()
                    semantic_segmentation = torch.einsum("bqc,bqhw->bchw", mask_cls_probs, mask_pred)
                    
                    if semantic_segmentation.shape[-2:] != labels.shape[-2:]:
                        semantic_segmentation = torch.nn.functional.interpolate(semantic_segmentation, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    
                    preds = semantic_segmentation.argmax(dim=1)
                else:
                    if args.model_arch == "segformer":
                        outputs = model(pixel_values=pixel_values)
                        logits = outputs.logits
                    else:
                        logits = model(pixel_values)
                        
                    if logits.shape[-2:] != labels.shape[-2:]:
                        logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        
                    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
                    batch_loss = ce_loss_fn(logits, labels)
                    val_loss += batch_loss.item()
                    
                    # Logits are (B, C, H, W)
                    preds = logits.argmax(dim=1)
                
                for b_idx in range(labels.shape[0]):
                    hist += fast_hist(labels[b_idx].flatten(), preds[b_idx].flatten(), num_classes)
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Metrics from Histogram (class 0 = 미분류는 제외)
        acc = torch.diag(hist).sum() / (hist.sum() + 1e-8)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        mean_iu = torch.nanmean(iu[1:])  # class 0(미분류) 제외하고 클래스 1~N-1 평균
        precision = torch.diag(hist) / (hist.sum(dim=0) + 1e-8)
        recall = torch.diag(hist) / (hist.sum(dim=1) + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mIoU: {mean_iu.item():.4f} | Acc: {acc.item():.4f}")
        
        # Save metrics to CSV
        metrics_csv_path = os.path.join(ckpt_dir, "metrics.csv")
        file_exists = os.path.isfile(metrics_csv_path)
        with open(metrics_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = ['Epoch', 'Train_Loss', 'Val_Loss', 'mIoU', 'Pixel_Acc']
                for c in range(num_classes):
                    headers.extend([f'Class_{c}_IoU', f'Class_{c}_Precision', f'Class_{c}_Recall', f'Class_{c}_F1'])
                writer.writerow(headers)
                
            row = [epoch, f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", f"{mean_iu.item():.4f}", f"{acc.item():.4f}"]
            for c in range(num_classes):
                row.extend([f"{iu[c].item():.4f}", f"{precision[c].item():.4f}", f"{recall[c].item():.4f}", f"{f1[c].item():.4f}"])
            writer.writerow(row)
        
        # Save checkoints
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # 1. Last (현재 Epoch 덮어쓰기)
        torch.save(model_to_save.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        
        # 2. Best (가장 성능이 좋았던 모델 유지)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model_to_save.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            print(f" -> Saved new best model! (Val Loss: {best_loss:.4f})")
            
        # Scheduler step is handled per-iteration inside the train loop

if __name__ == "__main__":
    train()
