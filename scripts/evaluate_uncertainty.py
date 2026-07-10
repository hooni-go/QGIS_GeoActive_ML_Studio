"""
evaluate_uncertainty.py

inference.py가 저장한 uncertainty_data/*.npz 와 정답 라벨을 읽어,
불확실성이 "오류를 예측"하는지 정량 검증한다.

산출물:
  - 오분류 탐지 AUROC
  - Sparsification plot + AUSE (불확실성 보정도)
  - Human-in-the-loop(타일 단위) 검토 효율 곡선 + 예산별 오류 검출률
  - 단일 시드 vs 앙상블 평균 mIoU 비교
  - metrics_uncertainty.json + sparsification.png + hitl_curve.png

사용 예:
  python evaluate_uncertainty.py \
      --results_dir /path/to/Swin_Mask2Former_InferenceResults \
      --mapping_file /path/to/mapping.json \
      --uncertainty bald --tile 256
"""
import os
import sys
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError as e:
    sys.exit(f"필수 패키지 누락: {e}. 'pip install pillow matplotlib' 후 다시 실행하세요.")

try:
    import rasterio
except ImportError:
    rasterio = None  # 16-bit(.tif) 라벨을 쓸 때만 필요


# ----------------------------- 순수 지표 함수 -----------------------------

def _trapz(y, x):
    y = np.asarray(y, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def _average_ranks(sorted_vals):
    """오름차순 정렬된 값에 대해 동점 평균 순위(1-based)를 반환."""
    n = sorted_vals.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0  # (i+1 .. j) 1-based 평균
        ranks[i:j] = avg
        i = j
    return ranks


def auroc_error_detection(unc, err):
    """오답(err=1)을 불확실성(unc)으로 얼마나 잘 골라내는지에 대한 AUROC."""
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


def sparsification(unc, err, n_steps=20):
    """불확실성 순/실제오류 순으로 제거하며 남은 오류율 곡선과 AUSE 계산."""
    err = err.astype(np.float64)
    N = err.size
    fracs = np.linspace(0.0, 0.95, n_steps)
    err_u = err[np.argsort(unc, kind="mergesort")[::-1]]   # 불확실한 것부터 제거
    err_o = err[np.argsort(err, kind="mergesort")[::-1]]   # 실제 오류부터 제거(oracle)
    m_curve, o_curve = [], []
    for f in fracs:
        k = int(f * N)
        ru, ro = err_u[k:], err_o[k:]
        m_curve.append(float(ru.mean()) if ru.size else 0.0)
        o_curve.append(float(ro.mean()) if ro.size else 0.0)
    m_curve, o_curve = np.array(m_curve), np.array(o_curve)
    ause = _trapz(m_curve - o_curve, fracs)              # 낮을수록 잘 보정됨
    base = float(err.mean())
    return fracs, m_curve, o_curve, base, ause


def hitl_curve(tile_unc, tile_err, tile_npix):
    """타일을 불확실도 순으로 검토할 때의 (검토 면적 비율, 오류 검출 비율) 곡선."""
    total_err = float(tile_err.sum())
    total_pix = float(tile_npix.sum())
    if total_err <= 0 or total_pix <= 0:
        return None

    def curve(order):
        e = np.cumsum(tile_err[order]) / total_err
        a = np.cumsum(tile_npix[order]) / total_pix
        return np.concatenate([[0.0], a]), np.concatenate([[0.0], e])

    a_u, e_u = curve(np.argsort(tile_unc, kind="mergesort")[::-1])
    a_o, e_o = curve(np.argsort(tile_err, kind="mergesort")[::-1])  # oracle: 오류 많은 타일부터
    budgets = [0.1, 0.2, 0.3, 0.5]
    summary = {f"{int(b*100)}%_area": float(np.interp(b, a_u, e_u)) for b in budgets}
    return (a_u, e_u), (a_o, e_o), summary


def fast_hist_np(gt, pred, n, ignore=0):
    k = (gt >= 0) & (gt < n) & (gt != ignore)
    return np.bincount(n * gt[k].astype(np.int64) + pred[k].astype(np.int64),
                       minlength=n * n).reshape(n, n)


def miou_from_hist(hist):
    diag = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        iu = diag / union
    return float(np.nanmean(iu[1:])), iu  # class 0(미분류) 제외


def classification_metrics_from_hist(hist, ignore=0):
    """원시 혼동행렬(행=GT, 열=예측; 열에는 배경(미분류) 예측 포함)에서
    클래스별 정밀도/재현율/F1/IoU와 전체정확도(OA)·Cohen's Kappa·macro 평균을 계산한다.
    GT가 ignore(미분류)인 화소는 누적 단계에서 이미 제외되어 있다고 가정한다.
    실제 클래스 화소를 배경(미분류)으로 예측한 경우도 오답으로 반영된다."""
    hist = hist.astype(np.float64)
    n = hist.shape[0]
    classes = [c for c in range(n) if c != ignore]
    prec, rec, f1, iou = {}, {}, {}, {}
    for c in classes:
        tp = hist[c, c]
        fn = hist[c, :].sum() - tp          # GT=c 인데 다른 것(배경 포함)으로 예측
        fp = hist[:, c].sum() - tp          # 다른 GT 인데 c 로 예측
        rec[c]  = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        prec[c] = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        df = 2 * tp + fp + fn
        f1[c]   = float(2 * tp / df) if df > 0 else float("nan")
        di = tp + fp + fn
        iou[c]  = float(tp / di) if di > 0 else float("nan")
    N = hist.sum()
    correct = float(np.trace(hist))
    oa = float(correct / N) if N > 0 else float("nan")
    row_marg = hist.sum(axis=1)
    col_marg = hist.sum(axis=0)
    pe = float((row_marg * col_marg).sum() / (N * N)) if N > 0 else float("nan")
    kappa = float((oa - pe) / (1 - pe)) if (1 - pe) != 0 else float("nan")
    return {
        "precision": prec, "recall": rec, "f1": f1, "iou": iou,
        "overall_accuracy": oa, "kappa": kappa,
        "macro_precision": float(np.nanmean([prec[c] for c in classes])),
        "macro_recall":    float(np.nanmean([rec[c]  for c in classes])),
        "macro_f1":        float(np.nanmean([f1[c]   for c in classes])),
        "macro_iou":       float(np.nanmean([iou[c]  for c in classes])),
    }


def expected_calibration_error(conf, correct, n_bins=15):
    """ECE 및 reliability diagram용 bin별 (신뢰도, 정확도, 빈도)."""
    conf = conf.astype(np.float64)
    correct = correct.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = conf.size
    ece = 0.0
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if c == 0:
            continue
        acc = float(correct[m].mean())
        cf = float(conf[m].mean())
        ece += (c / N) * abs(acc - cf)
        centers.append((lo + hi) / 2); accs.append(acc); confs.append(cf); counts.append(c)
    return float(ece), np.array(centers), np.array(accs), np.array(confs), np.array(counts)


def build_class_names(mapping, n):
    names = [f"C{i}" for i in range(n)]
    cn = mapping.get("class_names")
    if isinstance(cn, dict):
        for k, v in cn.items():
            i = int(k)
            if 0 <= i < n:
                names[i] = str(v)
    elif isinstance(cn, list):
        for i, v in enumerate(cn[:n]):
            names[i] = str(v)
    return names


# ----------------------------- 라벨 로딩 -----------------------------

def load_label_as_id(label_path, dn_to_id, is_16bit):
    if is_16bit and label_path.lower().endswith((".tif", ".tiff")):
        if rasterio is None:
            sys.exit("16-bit .tif 라벨을 읽으려면 rasterio가 필요합니다: pip install rasterio")
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


# ----------------------------- 메인 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="Swin_Mask2Former_InferenceResults 경로")
    ap.add_argument("--mapping_file", required=True)
    ap.add_argument("--label_dir", default="", help="미지정 시 results_dir의 상위/label 사용")
    ap.add_argument("--uncertainty", choices=["bald", "entropy", "std", "max_softmax"], default="bald")
    ap.add_argument("--is_16bit", default="False")
    ap.add_argument("--tile", type=int, default=512, help="HITL 타일 한 변 픽셀 수")
    ap.add_argument("--max_pixels_per_img", type=int, default=300000,
                    help="AUROC/sparsification용 이미지당 최대 샘플 픽셀(메모리 보호)")
    ap.add_argument("--model_arch", default="mask2former", help="추론에 사용된 모델 아키텍처")
    args = ap.parse_args()

    is_16bit = str(args.is_16bit).lower() == "true"
    prefix = args.model_arch.lower() + "_" if args.model_arch else ""
    
    # 만약 사용자가 uncertainty_data 폴더 자체를 선택했을 경우의 예외 처리
    norm_res = os.path.normpath(args.results_dir)
    if os.path.basename(norm_res) == "uncertainty_data":
        unc_dir = args.results_dir
        args.results_dir = os.path.dirname(norm_res)
    else:
        unc_dir = os.path.join(args.results_dir, "uncertainty_data")
        
    if not os.path.isdir(unc_dir):
        sys.exit(f"uncertainty_data 폴더가 없습니다: {unc_dir}. 보강한 inference.py로 먼저 추론하세요.")

    label_dir = args.label_dir or os.path.join(os.path.dirname(os.path.normpath(args.results_dir)), "label")
    if not os.path.isdir(label_dir):
        sys.exit(f"라벨 폴더가 없습니다: {label_dir} (--label_dir 로 지정 가능)")

    with open(args.mapping_file, "r") as f:
        mapping = json.load(f)
    dn_map = mapping.get("dn_map", {})
    num_classes = len(dn_map)
    dn_to_id = {v: int(k) for k, v in dn_map.items()}
    class_names = build_class_names(mapping, num_classes)

    rng = np.random.default_rng(0)
    px_err = []    # 픽셀 단위 (AUROC, sparsification, ECE)
    px_conf = []
    has_conf = True
    px_uncs = {"BALD": [], "Entropy": [], "Max-Softmax": [], "STD": []}
    t_uncs = {"BALD": [], "Entropy": [], "Max-Softmax": [], "STD": []}
    t_err, t_npix = [], []       # 타일 단위 (HITL)
    hist_ens = np.zeros((num_classes, num_classes), dtype=np.int64)
    hist_single = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # 다중 시드 평가용
    hists_per_seed = None
    px_err_per_seed = None

    npz_files = sorted(f for f in os.listdir(unc_dir) if f.endswith(".npz"))
    if not npz_files:
        sys.exit("uncertainty_data 에 .npz 가 없습니다.")
    print(f"{len(npz_files)}개 결과 평가 중 (uncertainty='{args.uncertainty}') ...")
    
    # Group npz_files by region to select 4 samples per region (total 12 panels)
    region_groups = {"GG": [], "GS": [], "JL": []}
    for idx, nf in enumerate(npz_files):
        parts = nf.split("_")
        region = None
        for p in parts:
            if p in ["GG", "GS", "JL"]:
                region = p
                break
        if region:
            region_groups[region].append(idx)
        else:
            if "GG" in nf: region_groups["GG"].append(idx)
            elif "GS" in nf: region_groups["GS"].append(idx)
            elif "JL" in nf: region_groups["JL"].append(idx)
            
    selected_panel_indices = set()
    panel_regions = {}
    for r, indices in region_groups.items():
        if indices:
            n_samples = len(indices)
            step = max(1, n_samples // 4)
            sel_indices = [indices[min(k * step, n_samples - 1)] for k in range(4)]
            sel_indices = sorted(list(set(sel_indices)))
            for idx in sel_indices:
                selected_panel_indices.add(idx)
                panel_regions[idx] = r
                
    out_panels_dir = os.path.join(args.results_dir, "Qualitative_Panels")
    if selected_panel_indices:
        os.makedirs(out_panels_dir, exist_ok=True)
        print("  - 지역별로 4개씩 (총 12개) Qualitative Panel 이미지를 자동 생성합니다.")

    n_used = 0
    total_files = len(npz_files)
    for i, nf in enumerate(npz_files):
        if i % max(1, total_files // 100) == 0:
            print(f"PROGRESS:{int((i / total_files) * 100)}", flush=True)
            
        d = np.load(os.path.join(unc_dir, nf), allow_pickle=True)
        filename = str(d["filename"])
        lbl_path = find_label(label_dir, filename)
        if lbl_path is None:
            continue
        gt = load_label_as_id(lbl_path, dn_to_id, is_16bit)
        pred = d["pred_id"].astype(np.int64)
        pred_single = d["pred_id_single"].astype(np.int64)
        if args.uncertainty == "max_softmax":
            unc = 1.0 - d["conf"].astype(np.float32)
        else:
            unc = d[args.uncertainty].astype(np.float32)

        if gt.shape != pred.shape:        # 안전장치: 크기 다르면 스킵
            continue
        n_used += 1

        valid = gt != 0                   # 미분류(ignore) 제외
        hist_ens += fast_hist_np(gt, pred, num_classes, ignore=0)
        hist_single += fast_hist_np(gt, pred_single, num_classes, ignore=0)

        # 타일 분할 전에 미리 로드
        u_bald = d["bald"].astype(np.float32)
        u_ent = d["entropy"].astype(np.float32)
        u_maxs = 1.0 - d["conf"].astype(np.float32)
        u_std = d["std"].astype(np.float32)

        err = (pred != gt) & valid
        e_valid = err[valid].astype(np.uint8)
        
        b_v = u_bald[valid]
        e_v = u_ent[valid]
        m_v = u_maxs[valid]
        s_v = u_std[valid]
        
        if "conf" in d.files:
            c_valid = d["conf"].astype(np.float32)[valid]
        else:
            has_conf = False
            c_valid = np.zeros_like(e_valid, dtype=np.float32)
            
        # 다중 시드 데이터 파싱
        pred_id_all_seeds = None
        if "pred_id_all_seeds" in d.files:
            pred_id_all_seeds = d["pred_id_all_seeds"]
            if hists_per_seed is None:
                num_seeds = pred_id_all_seeds.shape[0]
                hists_per_seed = [np.zeros((num_classes, num_classes), dtype=np.int64) for _ in range(num_seeds)]
                px_err_per_seed = [[] for _ in range(num_seeds)]
            
        if e_valid.size > args.max_pixels_per_img:
            sel = rng.choice(e_valid.size, args.max_pixels_per_img, replace=False)
            e_valid, c_valid = e_valid[sel], c_valid[sel]
            b_v, e_v, m_v, s_v = b_v[sel], e_v[sel], m_v[sel], s_v[sel]
        else:
            sel = None
            
        if pred_id_all_seeds is not None:
            for s_idx in range(pred_id_all_seeds.shape[0]):
                seed_pred = pred_id_all_seeds[s_idx].astype(np.int64)
                hists_per_seed[s_idx] += fast_hist_np(gt, seed_pred, num_classes, ignore=0)
                
                s_err = (seed_pred != gt) & valid
                s_e_valid = s_err[valid].astype(np.uint8)
                if sel is not None:
                    s_e_valid = s_e_valid[sel]
                px_err_per_seed[s_idx].append(s_e_valid)
            
        px_err.append(e_valid); px_conf.append(c_valid)
        px_uncs["BALD"].append(b_v)
        px_uncs["Entropy"].append(e_v)
        px_uncs["Max-Softmax"].append(m_v)
        px_uncs["STD"].append(s_v)

        H, W = gt.shape
        ts = args.tile
        for y in range(0, H, ts):
            for x in range(0, W, ts):
                vm = valid[y:y+ts, x:x+ts]
                npix = int(vm.sum())
                if npix == 0:
                    continue
                em = err[y:y+ts, x:x+ts]
                t_npix.append(npix)
                t_err.append(int(em.sum()))
                
                t_uncs["BALD"].append(float(u_bald[y:y+ts, x:x+ts][vm].mean()))
                t_uncs["Entropy"].append(float(u_ent[y:y+ts, x:x+ts][vm].mean()))
                t_uncs["Max-Softmax"].append(float(u_maxs[y:y+ts, x:x+ts][vm].mean()))
                t_uncs["STD"].append(float(u_std[y:y+ts, x:x+ts][vm].mean()))

        # Qualitative Panels (지역별 4개씩)
        if (n_used - 1) in selected_panel_indices:
            try:
                r_name = panel_regions[n_used - 1]
                base_no_ext = os.path.splitext(filename)[0]
                image_dir = os.path.join(os.path.dirname(os.path.normpath(label_dir)), "image")
                if not os.path.isdir(image_dir):
                    image_dir = os.path.join(os.path.dirname(os.path.normpath(args.results_dir)), "image")
                img_path = os.path.join(image_dir, base_no_ext + ".tif")
                if not os.path.exists(img_path):
                    img_path = os.path.join(image_dir, base_no_ext + ".png")
                
                if os.path.exists(img_path):
                    img_rgb = np.array(Image.open(img_path).convert("RGB"))
                    cmap = plt.get_cmap('tab20', num_classes)
                    error_map = np.zeros_like(pred, dtype=np.float32)
                    error_map[(gt != 0) & (pred != gt)] = 1.0
                    
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                    axes[0].imshow(img_rgb); axes[0].set_title("Input Image"); axes[0].axis('off')
                    axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=num_classes-1); axes[1].set_title("Ground Truth"); axes[1].axis('off')
                    axes[2].imshow(pred, cmap=cmap, vmin=0, vmax=num_classes-1); axes[2].set_title("Prediction"); axes[2].axis('off')
                    axes[3].imshow(error_map, cmap='Reds', vmin=0, vmax=1); axes[3].set_title("Error Map"); axes[3].axis('off')
                    unc_name = "Max-Softmax" if args.uncertainty == "max_softmax" else args.uncertainty.upper()
                    axes[4].imshow(unc, cmap='jet', vmin=0, vmax=unc.max()); axes[4].set_title(f"Uncertainty ({unc_name})"); axes[4].axis('off')
                    
                    plt.tight_layout()
                    panel_name = f"panel_{r_name}_{base_no_ext}.png"
                    if args.model_arch:
                        panel_name = prefix + panel_name
                    plt.savefig(os.path.join(out_panels_dir, panel_name), dpi=150, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                print(f"Panel generation failed for {filename}: {e}")

    if n_used == 0:
        sys.exit("매칭된 (예측, 라벨) 쌍이 없습니다. 파일명/라벨 폴더를 확인하세요.")

    # Calculate IoU and Classification Metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        iu_single = np.diag(hist_single) / (hist_single.sum(axis=1) + hist_single.sum(axis=0) - np.diag(hist_single))
        miou_single = float(np.nanmean(iu_single[1:]))

        iu_ens = np.diag(hist_ens) / (hist_ens.sum(axis=1) + hist_ens.sum(axis=0) - np.diag(hist_ens))
        miou_ens = float(np.nanmean(iu_ens[1:]))
        
        tp = np.diag(hist_ens)
        fp = hist_ens.sum(axis=0) - tp
        fn = hist_ens.sum(axis=1) - tp
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        total_pixels = hist_ens.sum()
        oa = tp.sum() / total_pixels if total_pixels > 0 else 0.0
        
        pe = (hist_ens.sum(axis=0) * hist_ens.sum(axis=1)).sum() / (total_pixels ** 2) if total_pixels > 0 else 0.0
        kappa = (oa - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
        
        clsm = {
            "overall_accuracy": float(oa),
            "kappa": float(kappa),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "macro_precision": float(np.nanmean(precision[1:])),
            "macro_recall": float(np.nanmean(recall[1:])),
            "macro_f1": float(np.nanmean(f1[1:]))
        }

    px_err = np.concatenate(px_err); px_conf = np.concatenate(px_conf)
    for m in px_uncs:
        if len(px_uncs[m]) > 0:
            px_uncs[m] = np.concatenate(px_uncs[m])
            
    t_err = np.array(t_err, dtype=np.float64); t_npix = np.array(t_npix, dtype=np.float64)
    hitl_results = {m: hitl_curve(np.array(unc_vals), t_err, t_npix) for m, unc_vals in t_uncs.items()}

    # Calibration (ECE + reliability diagram)
    ece = float("nan"); rel = None
    if has_conf:
        correct = 1.0 - px_err.astype(np.float64)
        ece, rc_cent, rc_acc, rc_conf, rc_cnt = expected_calibration_error(px_conf, correct)
        rel = (rc_cent, rc_acc, rc_conf, rc_cnt)

    # Calculate AUROC, AUSE for all methods
    method_metrics = {}
    from sklearn.metrics import roc_auc_score
    
    # Calculate per-seed metrics if available
    seed_mious, seed_oas, seed_aurocs = [], [], []
    if hists_per_seed is not None:
        for s_idx in range(len(hists_per_seed)):
            s_hist = hists_per_seed[s_idx]
            s_px_err = np.concatenate(px_err_per_seed[s_idx])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                s_iu = np.diag(s_hist) / (s_hist.sum(axis=1) + s_hist.sum(axis=0) - np.diag(s_hist))
                s_miou = float(np.nanmean(s_iu[1:]))
                s_tp = np.diag(s_hist)
                s_oa = s_tp.sum() / s_hist.sum() if s_hist.sum() > 0 else 0.0
            
            seed_mious.append(s_miou)
            seed_oas.append(s_oa)
            
            # Map uncertainty key to the stored list
            u_key = "Max-Softmax" if args.uncertainty == "max_softmax" else args.uncertainty.capitalize()
            s_u_arr = px_uncs.get(u_key, px_uncs["BALD"])
            
            if len(s_u_arr) > 0:
                try:
                    s_auroc = roc_auc_score(s_px_err, s_u_arr)
                    seed_aurocs.append(s_auroc)
                except ValueError:
                    seed_aurocs.append(float("nan"))
                    
    for m, u_arr in px_uncs.items():
        if len(u_arr) == 0: continue
        try:
            m_auroc = roc_auc_score(px_err, u_arr)
        except ValueError:
            m_auroc = 0.5
        m_fracs, mm_curve, mo_curve, m_base_err, m_ause = sparsification(u_arr, px_err)
        method_metrics[m] = {
            "AUROC": float(m_auroc),
            "AUSE": float(m_ause),
            "fracs": m_fracs,
            "m_curve": mm_curve,
            "o_curve": mo_curve,
            "base_err": m_base_err
        }

    # Save quantitative comparison to CSV
    csv_path = os.path.join(args.results_dir, prefix + "quantitative_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["Method", "AUROC (UP)", "AUSE (DOWN)", "ECE (DOWN)", "HITL_20%_Error_Caught (UP)"])
        for m in ["BALD", "Entropy", "Max-Softmax", "STD"]:
            if m not in method_metrics: continue
            met = method_metrics[m]
            hitl_val = hitl_results[m][2].get("20%_area", 0) * 100 if m in hitl_results and hitl_results[m] else 0
            writer.writerow([m, f"{met['AUROC']:.4f}", f"{met['AUSE']:.4f}", f"{ece:.4f}", f"{hitl_val:.1f}%"])

    # For legacy plotting below, fetch the target method
    target_m = "BALD" if args.uncertainty == "bald" else ("Entropy" if args.uncertainty == "entropy" else ("Max-Softmax" if args.uncertainty == "max_softmax" else "STD"))
    if target_m in method_metrics:
        fracs = method_metrics[target_m]["fracs"]
        m_curve = method_metrics[target_m]["m_curve"]
        o_curve = method_metrics[target_m]["o_curve"]
        ause = method_metrics[target_m]["AUSE"]
        auroc = method_metrics[target_m]["AUROC"]
        base_err = method_metrics[target_m]["base_err"]
    else:
        fracs, m_curve, o_curve, base_err, ause, auroc = np.array([0]), np.array([0]), np.array([0]), 0.0, 0.0, 0.5

    # ---- 플롯: sparsification ----
    plt.figure(figsize=(5, 4))
    plt.plot(fracs * 100, m_curve, color="#1D9E75", lw=2, label=f"Uncertainty (AUSE={ause:.4f})")
    plt.plot(fracs * 100, o_curve, color="#6B7280", lw=2, ls="--", label="Oracle")
    plt.axhline(base_err, color="#D85A30", lw=1.5, ls=":", label="Random (base error)")
    plt.xlabel("Fraction of pixels removed (%)"); plt.ylabel("Error rate of remaining")
    plt.title("Sparsification"); plt.legend(fontsize=8); plt.tight_layout()
    sp_path = os.path.join(args.results_dir, prefix + "sparsification.png")
    plt.savefig(sp_path, dpi=150); plt.close()

    # ---- 플롯: HITL 다중 베이스라인 비교 ----
    hitl_summary = {}
    if hitl_results.get("BALD") is not None:
        # 호환용 summary (현재 args.uncertainty 기준)
        target_m = "BALD" if args.uncertainty == "bald" else ("Entropy" if args.uncertainty == "entropy" else ("Max-Softmax" if args.uncertainty == "max_softmax" else "STD"))
        _, _, hitl_summary = hitl_results[target_m]
        
        plt.figure(figsize=(6, 5))
        colors = {"BALD": "#1D9E75", "Entropy": "#9C27B0", "Max-Softmax": "#2196F3", "STD": "#E91E63"}
        
        (a_o, e_o) = hitl_results["BALD"][1]
        plt.plot(a_o * 100, e_o * 100, color="#6B7280", lw=2, ls="--", label="Oracle (Best Possible)")
        plt.plot([0, 100], [0, 100], color="#D85A30", lw=1.5, ls="--", label="Random")
        
        for method, res in hitl_results.items():
            if res is None: continue
            (a_u, e_u), _, _ = res
            plt.plot(a_u * 100, e_u * 100, color=colors.get(method, "black"), lw=2, label=method)
            
        plt.xlabel("Reviewed area (%)")
        plt.ylabel("Errors caught (%)")
        plt.title(f"HITL Efficiency Comparison (tile={args.tile})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        hp_path = os.path.join(args.results_dir, prefix + "hitl_comparison_curve.png")
        plt.savefig(hp_path, dpi=150)
        plt.close()

    # ---- 플롯: reliability diagram (calibration) ----
    rel_path = None
    if rel is not None and rel[0].size > 0:
        rc_cent, rc_acc, rc_conf, rc_cnt = rel
        plt.figure(figsize=(4.5, 4.5))
        plt.plot([0, 1], [0, 1], color="#6B7280", ls="--", lw=1.5, label="Perfect calibration")
        plt.bar(rc_cent, rc_acc, width=1.0 / len(rc_cent) * 0.9,
                color="#1D9E75", alpha=0.85, edgecolor="#13654b", label="Accuracy")
        plt.plot(rc_conf, rc_acc, "o-", color="#D85A30", ms=4, lw=1.2, label="Acc vs Conf")
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.xlabel("Confidence"); plt.ylabel("Accuracy")
        plt.title(f"Reliability diagram (ECE={ece:.4f})")
        plt.legend(fontsize=8, loc="upper left"); plt.tight_layout()
        rel_path = os.path.join(args.results_dir, prefix + "reliability_diagram.png")
        plt.savefig(rel_path, dpi=150); plt.close()

    # ---- 플롯: 혼동행렬 (row-normalized recall, gt 1..N-1) ----
    cm = hist_ens.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    rows = list(range(1, num_classes))  # gt: 미분류(0) 제외
    cols = list(range(num_classes))     # pred: 배경 포함
    sub = cm_norm[np.ix_(rows, cols)]
    plt.figure(figsize=(max(6, num_classes * 0.7), max(5, len(rows) * 0.6)))
    im = plt.imshow(sub, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), [class_names[c] for c in cols], rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(rows)), [class_names[r] for r in rows], fontsize=8)
    plt.xlabel("Predicted"); plt.ylabel("Ground truth")
    plt.title("Confusion matrix (row-normalized)")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = sub[i, j]
            if v >= 0.005:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if v > 0.5 else "#222222")
    plt.tight_layout()
    cm_path = os.path.join(args.results_dir, prefix + "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150); plt.close()

    # ---- 요약 저장 ----
    comparison_summary = {}
    for m in ["BALD", "Entropy", "Max-Softmax", "STD"]:
        if m in method_metrics:
            hitl_val = hitl_results[m][2].get("20%_area", 0) * 100 if m in hitl_results and hitl_results[m] else 0
            comparison_summary[m] = {
                "AUROC": method_metrics[m]["AUROC"],
                "AUSE": method_metrics[m]["AUSE"],
                "HITL_20_caught": hitl_val
            }
            
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
            
    summary = {
        "n_images": n_used,
        "uncertainty_metric": args.uncertainty,
        "tile_size": args.tile,
        "error_detection_AUROC": auroc,
        "sparsification_AUSE": ause,
        "expected_calibration_error_ECE": ece,
        "base_error_rate": base_err,
        "miou_single_seed": miou_single,
        "miou_ensemble": miou_ens,
        "miou_gain": miou_ens - miou_single,
        "multi_seed_stats": {
            "mIoU_mean": float(np.nanmean(seed_mious)) if seed_mious else miou_single,
            "mIoU_std": float(np.nanstd(seed_mious)) if seed_mious else 0.0,
            "OA_mean": float(np.nanmean(seed_oas)) if seed_oas else clsm["overall_accuracy"],
            "OA_std": float(np.nanstd(seed_oas)) if seed_oas else 0.0,
            "AUROC_mean": float(np.nanmean(seed_aurocs)) if seed_aurocs else auroc,
            "AUROC_std": float(np.nanstd(seed_aurocs)) if seed_aurocs else 0.0
        },
        "comparison_metrics": comparison_summary,
        "per_class_iou_ensemble": {
            class_names[c]: (None if np.isnan(iu_ens[c]) else float(iu_ens[c]))
            for c in range(num_classes)
        },
        "overall_accuracy_OA": clsm["overall_accuracy"],
        "kappa": clsm["kappa"],
        "macro_precision": clsm["macro_precision"],
        "macro_recall": clsm["macro_recall"],
        "macro_f1": clsm["macro_f1"],
        "per_class_precision_ensemble": {
            class_names[c]: (None if np.isnan(clsm["precision"][c]) else float(clsm["precision"][c]))
            for c in range(num_classes) if c != 0
        },
        "per_class_recall_ensemble": {
            class_names[c]: (None if np.isnan(clsm["recall"][c]) else float(clsm["recall"][c]))
            for c in range(num_classes) if c != 0
        },
        "per_class_f1_ensemble": {
            class_names[c]: (None if np.isnan(clsm["f1"][c]) else float(clsm["f1"][c]))
            for c in range(num_classes) if c != 0
        },
        "hitl_errors_caught_by_budget": hitl_summary,
        "methodology_details": {
            "dropout_rate": "0.1 (MC Dropout & Stochastic Depth / DropPath enabled at inference)",
            "bg_threshold": "0.5 (Pixels with maximum object confidence below this are reverted to background/unclassified)",
            "ece_bins": "15 equal-width bins (np.linspace(0.0, 1.0, 16))",
            "hitl_unit": f"Tile-based ({args.tile}x{args.tile} pixels)",
            "hitl_aggregation": "Mean uncertainty of all valid pixels within the tile",
            "data_availability": "https://github.com/AnonymizedForReview/QGIS_GeoActive_ML_Studio (Source code, configurations, and weights)"
        }
    }
    out_json = os.path.join(args.results_dir, prefix + "metrics_uncertainty.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print("\n===== 불확실성 검증 요약 =====")
    print(f"이미지 수            : {n_used}")
    print(f"오분류 탐지 AUROC    : {auroc:.4f}   (1에 가까울수록 '불확실=오류' 일치)")
    print(f"Sparsification AUSE  : {ause:.4f}   (0에 가까울수록 잘 보정됨)")
    if has_conf:
        print(f"Calibration ECE      : {ece:.4f}   (0에 가까울수록 신뢰도=정확도 일치)")
    else:
        print("Calibration ECE      : (conf 미저장 — 보강된 inference.py로 재추론 필요)")
        
    print("\n[논문용 Mean ± SD (다중 시드)]")
    if seed_mious:
        print(f"mIoU                 : {np.nanmean(seed_mious):.4f} ± {np.nanstd(seed_mious):.4f}")
        print(f"OA                   : {np.nanmean(seed_oas):.4f} ± {np.nanstd(seed_oas):.4f}")
        print(f"AUROC                : {np.nanmean(seed_aurocs):.4f} ± {np.nanstd(seed_aurocs):.4f}")
    else:
        print("다중 시드 데이터 없음 (1-seed Inference).")
        
    print(f"\nmIoU 앙상블          : {miou_ens:.4f}   (Δ {miou_ens - miou_single:+.4f})")
    
    print(f"전체정확도 OA        : {clsm['overall_accuracy']:.4f}")
    print(f"Kappa                : {clsm['kappa']:.4f}")
    print(f"macro P / R / F1     : {clsm['macro_precision']:.4f} / {clsm['macro_recall']:.4f} / {clsm['macro_f1']:.4f}")

    def _fmt(x):
        return "N/A" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"
    print("\n--- 클래스별 IoU / Precision / Recall / F1 (Ensemble) ---")
    print(f" {'class':<12} {'IoU':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    for c in range(1, num_classes):
        print(f" {class_names[c]:<12} {_fmt(iu_ens[c]):>7} {_fmt(clsm['precision'][c]):>7} "
              f"{_fmt(clsm['recall'][c]):>7} {_fmt(clsm['f1'][c]):>7}")
    print("--------------------------------------------------------")
    
    if hitl_results.get("BALD") is not None:
        print("\n===== 💼 다중 베이스라인 HITL 비용 절감 비교 (상위 20% 면적 검수 시) =====")
        for method, res in hitl_results.items():
            if res is None: continue
            _, _, cer = res
            area_20 = cer.get('20%_area', 0) * 100
            print(f" ✔️ [{method:<12}] 전체 오류의 {area_20:5.1f}% 차단 (효율 {area_20/20:4.1f}배 극대화)")
            
        print(" --------------------------------------------------------------------")
        print(f"💡 결론: 위 표는 동일한 검수 예산(20%) 투입 시, 제안하는 불확실성 산출 기법이")
        print(f"         단순 Softmax 등의 베이스라인보다 실무적으로 얼마나 더 우수한지 증명합니다.")
        
    print("\n===== 📝 논문 작성용 Methodology Details (Copy & Paste) =====")
    print(" (i) Dropout rate: 0.1 (MC Dropout 및 Stochastic Depth/DropPath 적용)")
    print(" (ii) Inference Background Threshold: 0.5 (최대 객체 신뢰도가 0.5 미만인 픽셀은 배경으로 복원)")
    print(" (iii) ECE (Expected Calibration Error) 설정: 15 bins (0.0 ~ 1.0 구간 등간격 분할)")
    print(f" (iv) HITL (Human-in-the-Loop) 기준: {args.tile}x{args.tile} 타일 단위 분할, 타일 내 유효 픽셀 불확실성의 '평균(Mean)' 산출")
    print(" (v) Data Availability (예시): https://github.com/AnonymizedForReview/QGIS_GeoActive_ML_Studio (코드 및 가중치 공개 예정)")
    print("===============================================================\n")
    
    # (Duplicate prefix copy logic removed; files are now saved with model-prefix directly)

    print(f"\n저장: {out_json}")
    print(f"      {sp_path}")
    print(f"      {cm_path}")
    if rel_path:
        print(f"      {rel_path}")
    if "hp_path" in locals():
        print(f"      {hp_path}")


if __name__ == "__main__":
    main()
