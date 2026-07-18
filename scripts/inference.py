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

def infer_seeds_adaptive(model, img_tensor, sub_seeds, args, device, id_to_dn):
    """
    Recursively processes inference seeds in parallel batches.
    If a GPU Out of Memory (OOM) error occurs, it empties CUDA cache,
    halves the batch size, and retries the left and right halves recursively.
    """
    sub_size = len(sub_seeds)
    
    try:
        # Repeat input tensor along the batch dimension
        img_batch = img_tensor.repeat(sub_size, 1, 1, 1)
        
        # Set seed for reproducibility (First seed of batch starts the generator)
        set_seed(sub_seeds[0])
        if args.mc_dropout:
            enable_dropout(model)
            
        with torch.no_grad():
            if args.model_arch in ["mask2former", "segformer"]:
                outputs = model(pixel_values=img_batch)
            else:
                outputs = model(img_batch)
                
        # Extract predictions for each element in this successful batch
        batch_results = []
        for b_idx in range(sub_size):
            if args.model_arch == "mask2former":
                class_queries_logits = outputs.class_queries_logits[b_idx:b_idx+1]
                masks_queries_logits = outputs.masks_queries_logits[b_idx:b_idx+1]
                
                mask_cls_probs = class_queries_logits.softmax(dim=-1)[..., :-1]
                mask_pred = masks_queries_logits.sigmoid()
                semantic_segmentation = torch.einsum("bqc,bqhw->bchw", mask_cls_probs, mask_pred)
                
                if semantic_segmentation.shape[-2:] != img_tensor.shape[-2:]:
                    semantic_segmentation = torch.nn.functional.interpolate(
                        semantic_segmentation, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
                    )
                probs = semantic_segmentation.squeeze(0).cpu().numpy()
            else:
                if args.model_arch == "segformer":
                    logits = outputs.logits[b_idx:b_idx+1]
                else:
                    logits = outputs[b_idx:b_idx+1]
                    
                if logits.shape[-2:] != img_tensor.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
                    )
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                
            # Class thresholding (10-class vs 11-class backward compatibility)
            max_conf = np.max(probs, axis=0)
            raw_preds = np.argmax(probs, axis=0).astype(np.uint8)
            if probs.shape[0] == len(id_to_dn) - 1:
                preds = np.where(max_conf >= args.bg_threshold, raw_preds + 1, 0).astype(np.uint8)
            else:
                preds = np.where(max_conf >= args.bg_threshold, raw_preds, 0).astype(np.uint8)
            
            # Map Class ID back to DN
            dn_preds = np.zeros_like(preds, dtype=np.uint8)
            for class_id, dn_val in id_to_dn.items():
                dn_preds[preds == class_id] = dn_val
                
            batch_results.append((probs, preds))
            
        return batch_results
        
    except RuntimeError as e:
        # Detect VRAM Out of Memory
        if "out of memory" in str(e).lower() and sub_size > 1:
            torch.cuda.empty_cache()
            mid = sub_size // 2
            print(f"[OOM 방어] VRAM 메모리 초과 감지. 슬라이딩 미니배치 분할 연산 실행: {sub_size}개 ➔ ({mid}개, {sub_size - mid}개)")
            res_left = infer_seeds_adaptive(model, img_tensor, sub_seeds[:mid], args, device, id_to_dn)
            res_right = infer_seeds_adaptive(model, img_tensor, sub_seeds[mid:], args, device, id_to_dn)
            return res_left + res_right
        else:
            # Re-raise error if it's already batch size 1 or not an OOM error
            raise e

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            test_val = torch.tensor([1.0], device=device) * 2.0
        except Exception as e:
            print(f"[WARNING] CUDA device found but test execution failed: {str(e)}")
            print("[WARNING] Falling back to CPU for stability due to GPU capability mismatch (e.g. Blackwell sm_120 vs old PyTorch).")
            device = torch.device("cpu")
    
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
            
    # Detect the number of classes from checkpoint weights (10-class vs 11-class backward compatibility)
    detected_classes = None
    try:
        def detect_num_classes(sd):
            cleaned_sd = {}
            for k, v in sd.items():
                name = k[7:] if k.startswith("module.") else k
                cleaned_sd[name] = v
            for k, v in cleaned_sd.items():
                if "class_predictor.weight" in k:
                    return v.shape[0] - 1
                elif "decode_head.classifier" in k and "weight" in k:
                    return v.shape[0]
                elif "segmentation_head" in k and "weight" in k:
                    return v.shape[0]
            return None
        detected_classes = detect_num_classes(state_dict)
    except Exception as e:
        print(f"[WARNING] Failed to detect class count from checkpoint: {e}")
        
    model_num_classes = detected_classes if detected_classes is not None else num_classes
    print(f"Building model with {model_num_classes} classes (Detected from checkpoint: {detected_classes})")
    
    # Model
    model = build_model(model_num_classes, is_16bit=is_16bit, backbone=backbone, model_arch=args.model_arch)
    # Clean all loaded state dicts for compatibility (DataParallel module prefix, segmentation_head wrapping format)
    def clean_state_dict(sd, model_state_dict):
        new_sd = {}
        for k, v in sd.items():
            name = k[7:] if k.startswith('module.') else k
            
            # 3. Mask2Former Swin backbone remapping for HF version compatibility
            if name.startswith("model.pixel_level_module.encoder.") and not name.startswith("model.pixel_level_module.encoder.swin."):
                if "hidden_states_norms" not in name:
                    suffix = name[len("model.pixel_level_module.encoder."):]
                    
                    # Map attention keys
                    suffix = suffix.replace(".attention.self.query.", ".attention.q_proj.")
                    suffix = suffix.replace(".attention.self.key.", ".attention.k_proj.")
                    suffix = suffix.replace(".attention.self.value.", ".attention.v_proj.")
                    suffix = suffix.replace(".attention.output.dense.", ".attention.o_proj.")
                    
                    # Map MLP keys
                    suffix = suffix.replace(".intermediate.dense.", ".mlp.fc1.")
                    suffix = suffix.replace(".output.dense.", ".mlp.fc2.")
                    
                    # Map relative position table
                    suffix = suffix.replace(".attention.self.relative_position_bias_table", ".attention.relative_position_bias.relative_position_bias_table")
                    
                    name = "model.pixel_level_module.encoder.swin." + suffix
            
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
    is_hf = args.model_arch.lower() in ["mask2former", "segformer"]
    try:
        model.load_state_dict(state_dicts[0], strict=not is_hf)
    except RuntimeError as e:
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e) or "size mismatch" in str(e):
            print("\n" + "="*80)
            print("[FATAL ERROR] 선택하신 '모델 아키텍처'와 불러온 '가중치(.pt)' 파일이 서로 일치하지 않습니다!")
            print(f"  - 현재 드롭다운 선택: {args.model_arch.upper()}")
            print(f"  - 지정한 가중치 파일: {args.checkpoint}")
            print("\n  - 조치 방법:")
            print("    1. 선택한 가중치 파일에 맞는 아키텍처를 드롭다운(U-Net, Mask2Former, SegFormer 등)에서 올바르게 골라주시거나,")
            print("    2. 현재 아키텍처에 호환되는 올바른 .pt 가중치 파일을 찾아 다시 지정해 주세요.")
            print("="*80 + "\n")
        raise e
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            test_val = torch.tensor([1.0], device=device) * 2.0
        except Exception as e:
            print(f"[WARNING] CUDA device found but test execution failed: {str(e)}")
            print("[WARNING] Falling back to CPU for stability due to GPU capability mismatch.")
            device = torch.device("cpu")
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
    current_loaded_ckpt = -1
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
            
            # --- 1. Deterministic Pass (MC Dropout Off, 1st Checkpoint) ---
            if current_loaded_ckpt != 0:
                model.load_state_dict(state_dicts[0], strict=not is_hf)
                current_loaded_ckpt = 0
            model.eval() # Turn off dropout
            with torch.no_grad():
                if args.model_arch in ["mask2former", "segformer"]:
                    det_outputs = model(pixel_values=img_tensor)
                else:
                    det_outputs = model(img_tensor)
            
            if args.model_arch == "mask2former":
                class_queries_logits = det_outputs.class_queries_logits[0:1]
                masks_queries_logits = det_outputs.masks_queries_logits[0:1]
                mask_cls_probs = class_queries_logits.softmax(dim=-1)[..., :-1]
                mask_pred = masks_queries_logits.sigmoid()
                semantic_segmentation = torch.einsum("bqc,bqhw->bchw", mask_cls_probs, mask_pred)
                if semantic_segmentation.shape[-2:] != img_tensor.shape[-2:]:
                    semantic_segmentation = torch.nn.functional.interpolate(
                        semantic_segmentation, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
                    )
                probs_det_np = semantic_segmentation.squeeze(0).cpu().numpy()
                probs_det_np = probs_det_np / (np.sum(probs_det_np, axis=0, keepdims=True) + 1e-12)
                logits_det_np = np.log(probs_det_np + 1e-12)
            else:
                if args.model_arch == "segformer":
                    logits_det_raw = det_outputs.logits[0:1]
                else:
                    logits_det_raw = det_outputs[0:1]
                if logits_det_raw.shape[-2:] != img_tensor.shape[-2:]:
                    logits_det_raw = torch.nn.functional.interpolate(
                        logits_det_raw, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False
                    )
                logits_det_np = logits_det_raw.squeeze(0).cpu().numpy()
                # Compute stable softmax for deterministic pass
                logits_det_shifted = logits_det_np - np.max(logits_det_np, axis=0, keepdims=True)
                exp_logits = np.exp(logits_det_shifted)
                probs_det_np = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                
            pred_deterministic = np.argmax(probs_det_np, axis=0).astype(np.uint8)
            entropy_deterministic = -np.sum(probs_det_np * np.log(probs_det_np + 1e-12), axis=0).astype(np.float32)

            # Parse sheet_id from filename (e.g. tile_GG_37604_001.png -> GG_37604)
            import re
            m = re.search(r"(GG|GS|JL)_(?:[A-Za-z0-9]+_)?(\d{5,8})", img_name)
            if m:
                sheet_id = f"{m.group(1)}_{m.group(2)}"
            else:
                sheet_id = "unknown"
            
            # --- 2. MC Dropout Ensemble Inference ---
            all_probs = []
            id_preds_per_seed = []
            
            for ckpt_idx, s_dict in enumerate(state_dicts):
                if current_loaded_ckpt != ckpt_idx:
                    model.load_state_dict(s_dict, strict=not is_hf)
                    current_loaded_ckpt = ckpt_idx
                
                # Run adaptive sliding batch inference across all requested seeds
                batch_results = infer_seeds_adaptive(model, img_tensor, seeds, args, device, id_to_dn)
                
                for probs, preds in batch_results:
                    all_probs.append(probs)
                    id_preds_per_seed.append(preds.copy())
                    
            # Compute normalized mean probability map (probs_mean) and original std across seeds
            all_probs_np = np.stack(all_probs, axis=0) # (num_seeds, num_classes, H, W)
            probs_all = np.clip(all_probs_np, 1e-8, None)
            probs_all = probs_all / probs_all.sum(axis=1, keepdims=True)
            mean_p = probs_all.mean(axis=0)   # Normalized mean probabilities
            
            std_probs = np.std(all_probs_np, axis=0)   # (num_classes, H, W)
            uncertainty_map = np.max(std_probs, axis=0) # (H, W)
            
            # Ensemble prediction derived directly from normalized mean probabilities (mean_p) to ensure 100% agreement
            max_conf_ens = np.max(mean_p, axis=0)
            raw_ensemble_preds = np.argmax(mean_p, axis=0).astype(np.uint8)
            if mean_p.shape[0] == len(id_to_dn) - 1:
                ensemble_preds = np.where(max_conf_ens >= args.bg_threshold, raw_ensemble_preds + 1, 0).astype(np.uint8)
            else:
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
            norm_uncertainty = np.clip(uncertainty_map * 2.0, 0, 1.0)
            heatmap = cm.jet(norm_uncertainty)
            heatmap_rgb = (heatmap[:, :, :3] * 255.0).astype(np.uint8)
            std_out_path = os.path.join(std_dir, img_name.rsplit('.', 1)[0] + '_std.png')
            Image.fromarray(heatmap_rgb).save(std_out_path)
 
            # --- 원칙적 불확실성: 예측 엔트로피(전체) & 상호정보량 BALD(epistemic) ---
            entropy_map = -np.sum(mean_p * np.log(mean_p + 1e-12), axis=0)
            exp_entropy = -np.sum(probs_all * np.log(probs_all + 1e-12), axis=1).mean(axis=0)
            bald_map = np.clip(entropy_map - exp_entropy, 0, None)
            conf_map = mean_p.max(axis=0)
 
            # 평가용 배열 저장 (.npz)
            unc_dir = os.path.join(out_dir, "uncertainty_data")
            os.makedirs(unc_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(unc_dir, img_name.rsplit('.', 1)[0] + ".npz"),
                filename=img_name,
                sheet_id=sheet_id,
                pred_raw=raw_ensemble_preds,
                pred_id=ensemble_preds.astype(np.uint8),
                pred_id_single=id_preds_per_seed[0].astype(np.uint8),
                pred_id_all_seeds=np.stack(id_preds_per_seed, axis=0).astype(np.uint8),
                entropy=entropy_map.astype(np.float32),
                bald=bald_map.astype(np.float32),
                std=uncertainty_map.astype(np.float32),
                conf=conf_map.astype(np.float32),
                probs_mean=mean_p.astype(np.float32),
                logits_det=logits_det_np.astype(np.float32),
                pred_deterministic=pred_deterministic,
                entropy_deterministic=entropy_deterministic,
            )
            
            # 논문 Figure용 엔트로피/BALD 히트맵
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
            
            pred_rgb = np.zeros((ensemble_preds.shape[0], ensemble_preds.shape[1], 3), dtype=np.uint8)
            for class_id_str, color in color_map.items():
                pred_rgb[ensemble_preds == int(class_id_str)] = color
            
            if is_16bit:
                orig_rgb_display = (img[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
            else:
                orig_rgb_display = (img * 255.0).clip(0, 255).astype(np.uint8)
                
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
            
            images_to_concat = [Image.fromarray(orig_rgb_display)]
            if has_label:
                images_to_concat.append(Image.fromarray(label_rgb))
            images_to_concat.append(Image.fromarray(pred_rgb))
            
            total_width = sum(im.width for im in images_to_concat)
            max_height = max(im.height for im in images_to_concat)
            
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
