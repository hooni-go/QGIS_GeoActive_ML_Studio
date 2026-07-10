import os
import sys
import argparse
import json
import datetime

# Add current directory to sys.path to fix ModuleNotFoundError in Embedded Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

try:
    import rasterio
    from PIL import Image
except ImportError as e:
    sys.exit(f"필수 패키지 누락: {e}. 'pip install rasterio pillow' 후 다시 실행하세요.")

from models.model_builder import build_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def enable_dropout(model):
    # DataParallel wrapper check
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model, 'final_dropout'):
        actual_model.final_dropout.train()

def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="단일 .pt 파일 또는 쉼표(,)로 구분된 다중 .pt 파일 (Deep Ensemble 지원)")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--is_16bit", type=str, default="False")
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="42", help="Comma separated seeds")
    parser.add_argument("--mc_dropout", action="store_true")
    parser.add_argument("--bg_threshold", type=float, default=0.5,
                        help="객체 신뢰도가 이 값 미만인 픽셀은 배경(class 0)으로 처리")
    parser.add_argument("--model_arch", type=str, default="mask2former")
    args = parser.parse_args()

    is_16bit = args.is_16bit.lower() == "true"
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    # 시드 앙상블은 mc_dropout이 켜져 있을 때만 의미가 있다 (eval+dropout off면 모든 시드가 동일 출력 -> std=0)
    if len(seeds) > 1 and not args.mc_dropout:
        print("[경고] 시드가 여러 개지만 --mc_dropout이 꺼져 있어 모든 시드 결과가 동일합니다 (불확실성 맵이 전부 0).")
    
    # Use Native CUDA (cu128 natively supports sm_120)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.mapping_file, 'r') as f:
        mappings = json.load(f)
        dn_map = mappings.get("dn_map", {})
        color_map = mappings.get("color_map", {})
        
    num_classes = len(dn_map)
    id_to_dn = {int(k): v for k, v in dn_map.items()}

    # Determine Output Directory (Above target image folder level)
    # e.g., if img_dir is /path/to/data/image, output should be /path/to/data/UNET_InferenceResults
    parent_dir = os.path.dirname(os.path.normpath(args.img_dir))
    dataset_name = os.path.basename(parent_dir)
    if dataset_name.lower() in ["train", "val", "test", "images", "labels"]:
        parent_dir = os.path.dirname(parent_dir)
        dataset_name = os.path.basename(parent_dir)
        
    model_name = args.model_arch.upper()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_root = os.path.join(parent_dir, "Results", dataset_name)
    out_dir = os.path.join(results_root, f"{model_name}_Inference_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    mean_dir = os.path.join(out_dir, "ensemble_mean")
    std_dir = os.path.join(out_dir, "ensemble_std")
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(std_dir, exist_ok=True)
    
    checkpoints = [ckpt.strip() for ckpt in args.checkpoint.split(',') if ckpt.strip()]
    if not checkpoints:
        sys.exit("에러: 유효한 체크포인트 경로가 제공되지 않았습니다.")

    # seed_dirs = {} (Removed to prevent saving hundreds of redundant seed images)
        
    print(f"Output Directory: {out_dir}")

    print(f"Loading {len(checkpoints)} checkpoints into memory for inference...")
    state_dicts = [torch.load(ckpt, map_location="cpu") for ckpt in checkpoints]

    # Load state dict first to determine backbone
    state_dict = state_dicts[0]
    
    # Infer backbone from patch embedding projection weights
    backbone = "swin-large" # default
    proj_weight_key = "model.pixel_level_module.encoder.embeddings.patch_embeddings.projection.weight"
    if proj_weight_key in state_dict:
        embed_dim = state_dict[proj_weight_key].shape[0]
        if embed_dim == 96:
            backbone = "swin-tiny"
            print("Detected Swin-Tiny backbone from checkpoint weights.")
        elif embed_dim == 128:
            backbone = "swin-base"
            print("Detected Swin-Base backbone from checkpoint weights.")
        elif embed_dim == 192:
            backbone = "swin-large"
            print("Detected Swin-Large backbone from checkpoint weights.")
            
    # Model
    model = build_model(num_classes, is_16bit=is_16bit, backbone=backbone, model_arch=args.model_arch)
    # Clean all loaded state dicts for compatibility (DataParallel module prefix, segmentation_head wrapping format)
    def clean_state_dict(sd, model_state_dict):
        new_sd = {}
        for k, v in sd.items():
            name = k[7:] if k.startswith('module.') else k
            new_sd[name] = v
            
        # 1. SMP Models (U-Net, DeepLabV3+)
        if "segmentation_head.1.weight" in new_sd and "segmentation_head.0.weight" not in new_sd:
            if "segmentation_head.0.weight" in model_state_dict:
                new_sd["segmentation_head.0.weight"] = new_sd.pop("segmentation_head.1.weight")
                new_sd["segmentation_head.0.bias"] = new_sd.pop("segmentation_head.1.bias")
                
        if "segmentation_head.0.weight" in new_sd and "segmentation_head.1.weight" not in new_sd:
            if "segmentation_head.1.weight" in model_state_dict:
                new_sd["segmentation_head.1.weight"] = new_sd.pop("segmentation_head.0.weight")
                new_sd["segmentation_head.1.bias"] = new_sd.pop("segmentation_head.0.bias")
                
        # 2. SegFormer Models
        if "decode_head.classifier.1.weight" in new_sd and "decode_head.classifier.weight" not in new_sd:
            if "decode_head.classifier.weight" in model_state_dict:
                new_sd["decode_head.classifier.weight"] = new_sd.pop("decode_head.classifier.1.weight")
                new_sd["decode_head.classifier.bias"] = new_sd.pop("decode_head.classifier.1.bias")
                
        if "decode_head.classifier.weight" in new_sd and "decode_head.classifier.1.weight" not in new_sd:
            if "decode_head.classifier.1.weight" in model_state_dict:
                new_sd["decode_head.classifier.1.weight"] = new_sd.pop("decode_head.classifier.weight")
                new_sd["decode_head.classifier.1.bias"] = new_sd.pop("decode_head.classifier.bias")
        return new_sd

    state_dicts = [clean_state_dict(sd, model.state_dict()) for sd in state_dicts]
    model.load_state_dict(state_dicts[0], strict=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 학습(dataset.py)과 동일한 ImageNet 정규화 파라미터
    if is_16bit:
        norm_mean = np.array([0.485, 0.456, 0.406, 0.5], dtype=np.float32)
        norm_std = np.array([0.229, 0.224, 0.225, 0.225], dtype=np.float32)
    else:
        norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    images = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
    
    total_images = len(images)
    with torch.no_grad():
        for i, img_name in enumerate(images):
            if i % max(1, total_images // 100) == 0:
                print(f"PROGRESS:{int((i / total_images) * 100)}", flush=True)
            img_path = os.path.join(args.img_dir, img_name)
            
            # Load (img: (H, W, C), [0,1] 범위 — 시각화용 원본으로 보존)
            profile = None
            if is_16bit:
                with rasterio.open(img_path) as src:
                    img = src.read()
                    profile = src.profile
                    img = np.transpose(img, (1, 2, 0)).astype(np.float32) / 65535.0
            else:
                img = Image.open(img_path).convert("RGB")
                img = np.array(img, dtype=np.float32) / 255.0
                
            # 모델 입력은 학습과 동일하게 ImageNet 정규화를 적용 (img 원본은 시각화용으로 그대로 유지)
            img_norm = (img - norm_mean) / norm_std
            img_tensor = np.transpose(img_norm, (2, 0, 1))
            img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)
            
            all_probs = []
            id_preds_per_seed = []
            
            # Determine safe batch size based on model architecture to prevent GPU OOM
            if args.model_arch == "mask2former":
                max_batch_size = 2 # Mask2Former is heavy
            elif args.model_arch == "segformer":
                max_batch_size = 4 # SegFormer is lighter
            else:
                max_batch_size = 5 # U-Net/DeepLabV3+ are standard CNNs
                
            if not args.mc_dropout:
                max_batch_size = 1
                
            for ckpt_idx, s_dict in enumerate(state_dicts):
                model.load_state_dict(s_dict)
                
                num_seeds = len(seeds)
                seed_idx = 0
                
                while seed_idx < num_seeds:
                    curr_batch_size = min(max_batch_size, num_seeds - seed_idx)
                    batch_seeds = seeds[seed_idx : seed_idx + curr_batch_size]
                    seed_idx += curr_batch_size
                    
                    # Set seed using the first seed of the batch
                    set_seed(batch_seeds[0])
                    
                    if args.mc_dropout:
                        enable_dropout(model)
                        
                    # Repeat input along batch dimension
                    img_tensor_batch = img_tensor.repeat(curr_batch_size, 1, 1, 1)
                    
                    # Infer
                    if args.model_arch in ["mask2former", "segformer"]:
                        outputs = model(pixel_values=img_tensor_batch)
                    else:
                        outputs = model(img_tensor_batch)
                        
                    # Extract logits and probabilities
                    if args.model_arch == "mask2former":
                        class_queries_logits = outputs.class_queries_logits
                        masks_queries_logits = outputs.masks_queries_logits
                        mask_cls_probs = class_queries_logits.softmax(dim=-1)[..., :-1]
                        mask_pred = masks_queries_logits.sigmoid()
                        
                        # Preserve batch dimension: bqc, bqhw -> bchw
                        semantic_segmentation = torch.einsum("bqc,bqhw->bchw", mask_cls_probs, mask_pred)
                        
                        if semantic_segmentation.shape[-2:] != img_tensor.shape[-2:]:
                            semantic_segmentation = torch.nn.functional.interpolate(
                                semantic_segmentation, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
                            )
                        probs_batch = semantic_segmentation.cpu().numpy()
                    else:
                        if args.model_arch == "segformer":
                            logits = outputs.logits
                        else:
                            logits = outputs
                            
                        if logits.shape[-2:] != img_tensor.shape[-2:]:
                            logits = torch.nn.functional.interpolate(
                                logits, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
                            )
                        probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
                        
                    # Process batch results
                    for b in range(curr_batch_size):
                        probs = probs_batch[b]
                        max_conf = np.max(probs, axis=0)
                        preds = np.argmax(probs, axis=0).astype(np.uint8)
                        preds = np.where(max_conf >= args.bg_threshold, preds, 0).astype(np.uint8)
                        
                        all_probs.append(probs)
                        id_preds_per_seed.append(preds.copy())
                    
            # Compute Mean and Std across seeds
            all_probs_np = np.stack(all_probs, axis=0) # (num_seeds, num_classes, H, W)
            mean_probs = np.mean(all_probs_np, axis=0) # (num_classes, H, W)
            std_probs = np.std(all_probs_np, axis=0)   # (num_classes, H, W)
            
            # Uncertainty map (Max std dev across classes)
            uncertainty_map = np.max(std_probs, axis=0) # (H, W)
            
            # Ensemble prediction (동일한 배경 임계값 규칙 적용)
            max_conf_ens = np.max(mean_probs, axis=0)
            raw_ensemble_preds = np.argmax(mean_probs, axis=0).astype(np.uint8)
            ensemble_preds = np.where(max_conf_ens >= args.bg_threshold, raw_ensemble_preds, 0).astype(np.uint8)
            ensemble_dn_preds = np.zeros_like(ensemble_preds, dtype=np.uint8)
            for class_id, dn_val in id_to_dn.items():
                ensemble_dn_preds[ensemble_preds == class_id] = dn_val
                
            # Save Ensemble Mean
            mean_out_path = os.path.join(mean_dir, img_name)
            if is_16bit and profile is not None:
                profile.update(dtype=rasterio.uint8, count=1)
                with rasterio.open(mean_out_path, 'w', **profile) as dst:
                    dst.write(ensemble_dn_preds, 1)
            else:
                out_img_mean = Image.fromarray(ensemble_dn_preds)
                out_img_mean.save(mean_out_path)
                
            # Save Uncertainty Map as Blue-to-Red Heatmap (Jet Colormap)
            # Normalize uncertainty_map to [0, 1]. Max possible std for probabilities is 0.5.
            # So multiply by 2.0 to stretch 0.0->0.5 into 0.0->1.0.
            norm_uncertainty = np.clip(uncertainty_map * 2.0, 0, 1.0)
            
            # cm.jet returns (H, W, 4) in range [0, 1]
            heatmap = cm.jet(norm_uncertainty)
            
            # Extract RGB, convert to uint8
            heatmap_rgb = (heatmap[:, :, :3] * 255.0).astype(np.uint8)
            
            std_out_path = os.path.join(std_dir, img_name.rsplit('.', 1)[0] + '_std.png')
            Image.fromarray(heatmap_rgb).save(std_out_path)

            # --- 원칙적 불확실성: 예측 엔트로피(전체) & 상호정보량 BALD(epistemic) ---
            # 전체 클래스(0..C-1)에 대한 픽셀별 카테고리 분포로 계산
            probs_all = np.clip(all_probs_np, 1e-8, None)                      # (T, C, H, W)
            probs_all = probs_all / probs_all.sum(axis=1, keepdims=True)
            mean_p = probs_all.mean(axis=0)                                   # (C, H, W)
            entropy_map = -np.sum(mean_p * np.log(mean_p + 1e-12), axis=0)     # 전체(total) 불확실성
            exp_entropy = -np.sum(probs_all * np.log(probs_all + 1e-12), axis=1).mean(axis=0)
            bald_map = np.clip(entropy_map - exp_entropy, 0, None)             # epistemic 불확실성
            conf_map = mean_p.max(axis=0)                                      # 예측 클래스 신뢰도 (calibration용)

            # 평가용 배열 저장 (.npz) — evaluate_uncertainty.py 입력
            unc_dir = os.path.join(out_dir, "uncertainty_data")
            os.makedirs(unc_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(unc_dir, img_name.rsplit('.', 1)[0] + ".npz"),
                filename=img_name,
                pred_raw=raw_ensemble_preds,
                pred_id=ensemble_preds.astype(np.uint8),
                pred_id_single=id_preds_per_seed[0].astype(np.uint8),
                pred_id_all_seeds=np.stack(id_preds_per_seed, axis=0).astype(np.uint8),
                entropy=entropy_map.astype(np.float32),
                bald=bald_map.astype(np.float32),
                std=uncertainty_map.astype(np.float32),
                conf=conf_map.astype(np.float32),
            )

            # 논문 Figure용 엔트로피/BALD 히트맵(이미지별 정규화 — 비교 시 고정 스케일 권장)
            for arr, suffix in [(entropy_map, "_entropy"), (bald_map, "_bald")]:
                if arr.max() < 1e-5:
                    a = np.zeros_like(arr)
                else:
                    a = arr / (arr.max() + 1e-8)
                hm = (cm.jet(np.clip(a, 0, 1))[:, :, :3] * 255.0).astype(np.uint8)
                Image.fromarray(hm).save(os.path.join(std_dir, img_name.rsplit('.', 1)[0] + suffix + ".png"))
            
            # --- Visualization ---
            vis_dir = os.path.join(out_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create colorized prediction
            pred_rgb = np.zeros((ensemble_preds.shape[0], ensemble_preds.shape[1], 3), dtype=np.uint8)
            for class_id_str, color in color_map.items():
                pred_rgb[ensemble_preds == int(class_id_str)] = color
            
            # Prepare original image for display (정규화 이전의 img 원본 사용)
            if is_16bit:
                orig_rgb_display = (img[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
            else:
                orig_rgb_display = (img * 255.0).clip(0, 255).astype(np.uint8)
                
            # Check for label
            label_dir = os.path.join(os.path.dirname(os.path.normpath(args.img_dir)), "label")
            if not os.path.isdir(label_dir):
                label_dir = os.path.join(parent_dir, "label")
                
            has_label = False
            label_rgb = None
            if os.path.exists(label_dir):
                label_path = os.path.join(label_dir, img_name)
                if os.path.exists(label_path):
                    has_label = True
                    if is_16bit:
                        with rasterio.open(label_path) as l_src:
                            lbl_img = l_src.read(1)
                    else:
                        lbl_img = np.array(Image.open(label_path))
                        
                    lbl_rgb = np.zeros((lbl_img.shape[0], lbl_img.shape[1], 3), dtype=np.uint8)
                    dn_to_id = {v: int(k) for k, v in id_to_dn.items()}
                    for dn_val, c_id in dn_to_id.items():
                        if str(c_id) in color_map:
                            lbl_rgb[lbl_img == dn_val] = color_map[str(c_id)]
                    label_rgb = lbl_rgb
            
            # Concatenate images horizontally
            images_to_concat = [Image.fromarray(orig_rgb_display)]
            if has_label:
                images_to_concat.append(Image.fromarray(label_rgb))
            images_to_concat.append(Image.fromarray(pred_rgb))
            
            total_width = sum(i.width for i in images_to_concat)
            max_height = max(i.height for i in images_to_concat)
            
            vis_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images_to_concat:
                vis_img.paste(im, (x_offset, 0))
                x_offset += im.width
                
            # Save visualization
            vis_out_path = os.path.join(vis_dir, img_name.rsplit('.', 1)[0] + '_compare.png')
            vis_img.save(vis_out_path)
            # ---------------------
            
            print(f"Processed ensemble for: {img_name}")

if __name__ == "__main__":
    infer()
