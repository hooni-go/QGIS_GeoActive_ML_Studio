"""
experiment_threshold.py

inference.py가 남긴 .npz 파일(pred_raw, conf 포함)을 로드하여,
배경 임계값(bg_threshold)을 0.0부터 0.9까지 변화시킬 때
성능 지표(mIoU, Pixel Acc)와 불확실성 지표(AUROC, Coverage)가
어떻게 변하는지 추적하는 통제 실험 스크립트.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

try:
    from PIL import Image
except ImportError:
    pass

def fast_hist_np(gt, pred, n, ignore=0):
    k = (gt >= 0) & (gt < n) & (gt != ignore)
    return np.bincount(n * gt[k].astype(np.int64) + pred[k].astype(np.int64),
                       minlength=n * n).reshape(n, n)

def miou_from_hist(hist):
    diag = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        iu = diag / union
    return float(np.nanmean(iu[1:])), iu

def _average_ranks(sorted_vals):
    n = sorted_vals.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[i:j] = avg
        i = j
    return ranks

def auroc_error_detection(unc, err):
    err = err.astype(bool)
    n = err.size
    n_pos = int(err.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(unc, kind="mergesort")
    ranks_sorted = _average_ranks(unc[order])
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    auc = (ranks[err].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def load_label_as_id(label_path, dn_to_id, is_16bit):
    if is_16bit and label_path.lower().endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(label_path) as src:
            lbl = src.read(1)
    else:
        lbl = np.array(Image.open(label_path).convert("L"))
    out = np.zeros_like(lbl, dtype=np.int64)
    for dn_val, cid in dn_to_id.items():
        out[lbl == dn_val] = cid
    return out

def find_label(label_dir, filename):
    base = os.path.splitext(filename)[0]
    for ext in (os.path.splitext(filename)[1], ".png", ".tif", ".tiff"):
        p = os.path.join(label_dir, base + ext)
        if os.path.exists(p):
            return p
    p = os.path.join(label_dir, filename)
    return p if os.path.exists(p) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--mapping_file", required=True)
    ap.add_argument("--label_dir", default="")
    ap.add_argument("--is_16bit", default="False")
    ap.add_argument("--max_pixels_per_img", type=int, default=100000)
    args = ap.parse_args()

    is_16bit = str(args.is_16bit).lower() == "true"
    unc_dir = os.path.join(args.results_dir, "uncertainty_data")
    if not os.path.isdir(unc_dir):
        sys.exit(f"uncertainty_data 폴더가 없습니다: {unc_dir}")

    label_dir = args.label_dir or os.path.join(os.path.dirname(os.path.normpath(args.results_dir)), "label")
    with open(args.mapping_file, "r") as f:
        mapping = json.load(f)
    dn_to_id = {int(v): int(k) for k, v in mapping.get("dn_map", {}).items()}
    num_classes = len(dn_to_id)
    class_names = [f"C{i}" for i in range(num_classes)]
    cn = mapping.get("class_names")
    if isinstance(cn, dict):
        for k, v in cn.items():
            if 0 <= int(k) < num_classes:
                class_names[int(k)] = str(v)

    npz_files = sorted(f for f in os.listdir(unc_dir) if f.endswith(".npz"))
    if not npz_files:
        sys.exit("npz 파일이 없습니다.")

    thresholds = np.arange(0.0, 1.0, 0.1) # 0.0 ~ 0.9
    
    results = []
    print(f"총 {len(npz_files)}개의 npz 파일을 분석하여 임계값 민감도를 측정합니다...")
    
    # 캐싱용 리스트
    data_cache = []
    rng = np.random.default_rng(42)
    
    # 1. 파일 전체 로드
    for nf in npz_files:
        d = np.load(os.path.join(unc_dir, nf), allow_pickle=True)
        if "pred_raw" not in d.files:
            sys.exit("오류: pred_raw 가 없습니다. 수정된 inference.py로 추론을 다시 실행하세요.")
            
        filename = str(d["filename"])
        lbl_path = find_label(label_dir, filename)
        if not lbl_path:
            continue
            
        gt = load_label_as_id(lbl_path, dn_to_id, is_16bit)
        pred_raw = d["pred_raw"].astype(np.int64)
        conf = d["conf"].astype(np.float32)
        unc = d["std"].astype(np.float32) # 표준편차 불확실성 사용
        
        valid = gt != 0
        if valid.sum() == 0:
            continue
            
        gt_v = gt[valid]
        pred_raw_v = pred_raw[valid]
        conf_v = conf[valid]
        unc_v = unc[valid]
        
        if gt_v.size > args.max_pixels_per_img:
            sel = rng.choice(gt_v.size, args.max_pixels_per_img, replace=False)
            gt_v, pred_raw_v, conf_v, unc_v = gt_v[sel], pred_raw_v[sel], conf_v[sel], unc_v[sel]
            
        data_cache.append((gt_v, pred_raw_v, conf_v, unc_v))

    if not data_cache:
        sys.exit("유효한 라벨 데이터가 없습니다.")

    # 2. 임계값 Sweep
    for thresh in thresholds:
        hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        px_unc, px_err = [], []
        
        total_pixels = 0
        accepted_pixels = 0
        correct_accepted = 0
        
        for gt_v, pred_raw_v, conf_v, unc_v in data_cache:
            # Thresholding Logic (추론과 동일)
            pred_t = np.where(conf_v >= thresh, pred_raw_v, 0)
            
            # Confusion Matrix
            for g, p in zip(gt_v, pred_t):
                hist[g, p] += 1
                
            # AUROC Calculation (오류: gt != pred_t)
            err = (gt_v != pred_t).astype(np.uint8)
            px_unc.append(unc_v)
            px_err.append(err)
            
            # Coverage (선택된/수락된 픽셀 비율)
            # 예측이 0(배경)이 아닌 픽셀
            is_accepted = pred_t != 0
            total_pixels += gt_v.size
            acc_count = is_accepted.sum()
            accepted_pixels += acc_count
            if acc_count > 0:
                correct_accepted += (gt_v[is_accepted] == pred_t[is_accepted]).sum()
                
        # Calculate overall metrics for this threshold
        px_unc = np.concatenate(px_unc)
        px_err = np.concatenate(px_err)
        
        auroc = auroc_error_detection(px_unc, px_err)
        miou, iu_arr = miou_from_hist(hist)
        
        coverage = accepted_pixels / total_pixels if total_pixels > 0 else 0
        accepted_acc = correct_accepted / accepted_pixels if accepted_pixels > 0 else 0
        
        print(f"Threshold {thresh:.1f} | mIoU: {miou:.4f} | AUROC: {auroc:.4f} | Coverage: {coverage*100:.1f}%")
        
        res = {
            "Threshold": float(thresh),
            "mIoU": float(miou),
            "AUROC": float(auroc),
            "Coverage": float(coverage),
            "Accepted_Accuracy": float(accepted_acc)
        }
        for c in range(1, num_classes):
            res[f"{class_names[c]}_IoU"] = float(iu_arr[c])
            
        results.append(res)
        
    # 3. CSV 저장
    csv_path = os.path.join(args.results_dir, "threshold_ablation.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
        
    # 4. Plotting (Risk-Coverage Curve)
    thresh_arr = [r["Threshold"] for r in results]
    miou_arr = [r["mIoU"] for r in results]
    auroc_arr = [r["AUROC"] for r in results]
    cov_arr = [r["Coverage"] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(7, 5))

    color1 = 'tab:blue'
    ax1.set_xlabel('Background Threshold')
    ax1.set_ylabel('mIoU', color=color1)
    ax1.plot(thresh_arr, miou_arr, 'o-', color=color1, label="mIoU")
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('AUROC / Coverage', color=color2)
    ax2.plot(thresh_arr, auroc_arr, 's-', color=color2, label="AUROC (Error Detection)")
    ax2.plot(thresh_arr, cov_arr, '^-', color='tab:green', label="Coverage (Fraction Retained)")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("Ablation Study: Background Threshold vs Performance/Uncertainty")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    png_path = os.path.join(args.results_dir, "threshold_sensitivity_curve.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    print(f"\n실험 완료! 결과가 저장되었습니다.")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")

if __name__ == "__main__":
    main()
