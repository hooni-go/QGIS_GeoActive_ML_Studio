"""
benchmark_mc_dropout.py

MC Dropout 회수 N (1, 5, 10, 20, 50)에 따른 성능 수렴도 및 리소스 프로파일링 벤치마크 스크립트.
최적화를 위해 단 1회의 50-seed 추론만 가동한 후 누적 평균을 슬라이싱하여 메트릭을 고속 연산합니다.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Add parent and script paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
model_builder = None
try:
    from models.model_builder import build_model
except ImportError:
    # Try importing from scripts root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts.models.model_builder import build_model

from inference import set_seed, enable_dropout, infer_seeds_adaptive
from evaluate_uncertainty import load_label_as_id, find_label, fast_hist_np, miou_from_hist, expected_calibration_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--is_16bit", default="False")
    ap.add_argument("--mapping_file", type=str, required=True)
    ap.add_argument("--model_arch", type=str, default="mask2former")
    ap.add_argument("--results_dir", type=str, required=True)
    args = ap.parse_args()

    is_16bit = args.is_16bit.lower() == "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load mapping
    with open(args.mapping_file, 'r') as f:
        mappings = json.load(f)
    dn_map = mappings.get("dn_map", {})
    num_classes = len(dn_map)
    dn_to_id = {int(k): v for k, v in dn_map.items()}

    label_dir = os.path.join(os.path.dirname(os.path.normpath(args.img_dir)), "label")
    if not os.path.exists(label_dir):
        label_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(args.img_dir))), "label")

    # Build model
    checkpoints = [ckpt.strip() for ckpt in args.checkpoint.split(',') if ckpt.strip()]
    state_dicts = [torch.load(ckpt, map_location="cpu") for ckpt in checkpoints]
    
    # Simple compatibility clean
    def clean_state_dict(sd, model_state_dict):
        new_sd = {}
        for k, v in sd.items():
            name = k[7:] if k.startswith('module.') else k
            new_sd[name] = v
        return new_sd
        
    # Model configuration
    model = build_model(num_classes, is_16bit=is_16bit, backbone="swin-large", model_arch=args.model_arch)
    state_dicts = [clean_state_dict(sd, model.state_dict()) for sd in state_dicts]
    model.load_state_dict(state_dicts[0], strict=False)
    model.to(device)
    model.eval()

    # Image list
    images = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
    images = images[:10] # 벤치마킹을 위해 대표 10개 이미지에 대해서만 빠르게 산출
    
    # 50개 시드 정의
    seeds = list(range(1, 51))
    
    norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    n_values = [1, 5, 10, 20, 50]
    
    # 지표 누적 구조
    n_results = {n: {"hists": np.zeros((num_classes, num_classes), dtype=np.int64), "confs": [], "corrects": []} for n in n_values}
    
    # 리소스 누적
    runtimes = {n: [] for n in n_values}
    vram_peaks = {n: [] for n in n_values}

    print(f"[Benchmark] MC Dropout N={n_values} 벤치마크 구동 시작...")
    
    for idx, img_name in enumerate(images):
        img_path = os.path.join(args.img_dir, img_name)
        lbl_path = find_label(label_dir, img_name)
        if lbl_path is None:
            continue
            
        gt = load_label_as_id(lbl_path, dn_to_id, is_16bit)
        valid = gt != 0
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        img_norm = (img - norm_mean) / norm_std
        img_tensor = np.transpose(img_norm, (2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(device)

        # 50개 시드에 대한 1회 일괄 추론 및 리소스 측정
        for n in n_values:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            
            start_time = time.time()
            
            # N개의 시드만큼 adaptive batch inference 수행
            sub_seeds = seeds[:n]
            batch_results = infer_seeds_adaptive(model, img_tensor, sub_seeds, args, device, dn_map)
            
            elapsed = time.time() - start_time
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
            
            runtimes[n].append(elapsed)
            vram_peaks[n].append(peak_mem)
            
            # 결과 가공
            probs_list = [probs for probs, _ in batch_results]
            mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0) # (num_classes, H, W)
            
            pred = np.argmax(mean_probs, axis=0)
            conf = np.max(mean_probs, axis=0)
            
            # Confusion matrix 누적
            hist = fast_hist_np(gt, pred, num_classes, ignore=0)
            n_results[n]["hists"] += hist
            
            # Calibration 누적
            n_results[n]["confs"].append(conf[valid])
            n_results[n]["corrects"].append((pred == gt)[valid])
            
    # 최종 지표 산출
    csv_path = os.path.join(args.results_dir, "mc_dropout_benchmark.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("N,mIoU,ECE,Avg_Runtime_Sec,Peak_VRAM_MB\n")
        print("\n===== MC Dropout 벤치마크 결과 요약 =====")
        print(f" {'N':<5} {'mIoU':>8} {'ECE':>8} {'Runtime(s)':>12} {'VRAM(MB)':>10}")
        for n in n_values:
            hist = n_results[n]["hists"]
            miou, _ = miou_from_hist(hist, include_zero=False)
            
            confs_concat = np.concatenate(n_results[n]["confs"])
            corrects_concat = np.concatenate(n_results[n]["corrects"])
            ece, _, _, _, _ = expected_calibration_error(confs_concat, corrects_concat)
            
            avg_time = np.mean(runtimes[n])
            avg_vram = np.mean(vram_peaks[n])
            
            print(f" {n:<5} {miou:>8.4f} {ece:>8.4f} {avg_time:>12.2f} {avg_vram:>10.1f}")
            f.write(f"{n},{miou:.4f},{ece:.4f},{avg_time:.2f},{avg_vram:.1f}\n")
            
    print(f"\n벤치마크 결과 저장 완료: {csv_path}")

if __name__ == "__main__":
    main()
