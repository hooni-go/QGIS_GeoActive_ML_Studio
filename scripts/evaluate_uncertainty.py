"""
evaluate_uncertainty.py

JSTARS Major Revision 대응 최종 통합 평가 엔진.
인프라 엔진(inference.py)이 저장한 .npz 파일들을 기반으로 다음을 수행한다:
  1. 검증세트(validation)를 이용한 Temperature Scaling (L-BFGS-B 최적화) 피팅 및 테스트세트 적용
  2. All-pixel (주 분석) 및 Foreground-only (부 분석) 병렬 연산
  3. 3대 조건별 불확실성 평가 (조건 1: Pure Argmax, 조건 2: Selective Classification, 조건 3: Operating Rule)
     - 조건 2의 거부(Abstain) 화소는 Confusion Matrix/mIoU 연산에서 마스킹 제외
     - 조건 3의 배경 강등 화소는 ECE 계산 시 신뢰도를 배경 확률 p0(x)로 보정 적용
  4. 52개 지도도엽 군집(sheet_id) 기반 1,000회 Bootstrap Resampling (95% CI 및 p-value 산출)
  5. 타일 집계 비교 (그리드 크기: 256/512/1024/도엽 x 통계치: Mean/Median/90%/95%/Top-10% Mean)
     - oracle_pixel (AUSE용) 및 oracle_tile (HITL용) 명칭 구분 사용
     - 잔여 오류율(Residual Map Error) 곡선 및 50%/80% 오류 포착 비율 산출
  6. 공간 독립성 통계 (도엽간 최근접 거리 분포 및 권역별 GG/GS/JL 개별 성능 분리)
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.optimize as opt

try:
    import rasterio
except ImportError:
    rasterio = None

def derive_val_label_dir(test_label_dir, val_results_dir):
    """
    테스트 라벨 폴더 경로와 검증 결과 폴더 경로로부터
    검증셋 라벨 폴더를 추정한다.
    """
    # (1) 경로 문자열에서 test → val 치환
    norm = os.path.normpath(test_label_dir)
    for a, b in (("test", "val"), ("Test", "Val"), ("TEST", "VAL"),
                 ("testing", "validation"), ("Testing", "Validation")):
        parts = norm.split(os.sep)
        replaced = [b if p == a else p for p in parts]
        cand = os.sep.join(replaced)
        if cand != norm and os.path.isdir(cand):
            return cand

    # (2) 검증 결과 폴더에서 상위로 올라가며 label 폴더 탐색
    if val_results_dir:
        p = os.path.normpath(val_results_dir)
        for _ in range(5):
            p = os.path.dirname(p)
            if not p or p == os.path.dirname(p):
                break
            for name in ("label", "labels", "gt", "mask"):
                cand = os.path.join(p, name)
                if os.path.isdir(cand):
                    return cand
            # ROOT/val/label 형태도 확인
            cand = os.path.join(p, "val", "label")
            if os.path.isdir(cand):
                return cand
    return None

# ----------------------------- 순수 지표 함수 -----------------------------

def _trapz(y, x):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))

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

def fast_auroc(y_true, y_score, max_thresholds=1000):
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score)
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tps = np.cumsum(y_true)
    fps = np.arange(1, len(y_true) + 1) - tps
    
    distinct_value_indices = np.where(np.diff(y_score[desc_score_indices]))[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_true) - 1]
    
    if len(threshold_idxs) > max_thresholds:
        threshold_idxs = threshold_idxs[np.linspace(0, len(threshold_idxs) - 1, max_thresholds, dtype=int)]
    
    tps_t = tps[threshold_idxs]
    fps_t = fps[threshold_idxs]
    
    tps_t = np.r_[0, tps_t]
    fps_t = np.r_[0, fps_t]
    
    auc = np.sum((fps_t[1:] - fps_t[:-1]) * (tps_t[1:] + tps_t[:-1])) / (2.0 * n_pos * n_neg)
    return float(auc)

def fast_auprc(y_true, y_score, max_thresholds=1000):
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score)
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    
    n_pos = np.sum(y_true)
    if n_pos == 0 or len(y_true) == 0:
        return 0.0
    
    tps = np.cumsum(y_true)
    fps = np.arange(1, len(y_true) + 1) - tps
    
    distinct_value_indices = np.where(np.diff(y_score[desc_score_indices]))[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_true) - 1]
    
    if len(threshold_idxs) > max_thresholds:
        threshold_idxs = threshold_idxs[np.linspace(0, len(threshold_idxs) - 1, max_thresholds, dtype=int)]
    
    tps_t = tps[threshold_idxs]
    fps_t = fps[threshold_idxs]
    
    precisions = tps_t / (tps_t + fps_t)
    recalls = tps_t / n_pos
    
    recalls = np.r_[0.0, recalls]
    precisions = np.r_[1.0, precisions]
    
    return float(np.sum(np.diff(recalls) * precisions[1:]))

def sparsification(unc, err, n_steps=20):
    err = err.astype(np.float64)
    N = err.size
    fracs = np.linspace(0.0, 0.95, n_steps)
    err_u = err[np.argsort(unc)[::-1]]
    err_o = err[np.argsort(err)[::-1]] # oracle_pixel
    m_curve, o_curve = [], []
    for f in fracs:
        k = int(f * N)
        ru, ro = err_u[k:], err_o[k:]
        m_curve.append(float(ru.mean()) if ru.size else 0.0)
        o_curve.append(float(ro.mean()) if ro.size else 0.0)
    m_curve, o_curve = np.array(m_curve), np.array(o_curve)
    ause = _trapz(m_curve - o_curve, fracs)
    base = float(err.mean())
    return fracs, m_curve, o_curve, base, ause

def hitl_curve(tile_unc, tile_err, tile_npix):
    total_err = float(tile_err.sum())
    total_pix = float(tile_npix.sum())
    if total_err <= 0 or total_pix <= 0:
        return None

    def curve(order):
        e = np.cumsum(tile_err[order]) / total_err
        a = np.cumsum(tile_npix[order]) / total_pix
        return np.concatenate([[0.0], a]), np.concatenate([[0.0], e])

    a_u, e_u = curve(np.argsort(tile_unc, kind="mergesort")[::-1])
    a_o, e_o = curve(np.argsort(tile_err, kind="mergesort")[::-1]) # oracle_tile
    budgets = [0.1, 0.2, 0.3, 0.5]
    summary = {f"{int(b*100)}%_area": float(np.interp(b, a_u, e_u)) for b in budgets}
    return (a_u, e_u), (a_o, e_o), summary

def fast_hist_np(gt, pred, n, ignore=None):
    if ignore is not None:
        k = (gt >= 0) & (gt < n) & (gt != ignore)
    else:
        k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(np.int64) + pred[k].astype(np.int64),
                       minlength=n * n).reshape(n, n)

def miou_from_hist(hist, include_zero=False):
    diag = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        iu = diag / union
    if include_zero:
        return float(np.nanmean(iu)), iu
    else:
        return float(np.nanmean(iu[1:])), iu

def classification_metrics_from_hist(hist, ignore=0):
    hist = hist.astype(np.float64)
    n = hist.shape[0]
    classes = [c for c in range(n) if c != ignore]
    prec, rec, f1, iou = {}, {}, {}, {}
    for c in classes:
        tp = hist[c, c]
        fn = hist[c, :].sum() - tp
        fp = hist[:, c].sum() - tp
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
    conf = np.clip(conf, 0.0, 1.0)
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
        centers.append((lo + hi) / 2)
        accs.append(acc)
        confs.append(cf)
        counts.append(c)
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

def sheet_number_to_lonlat(sheet_num, n50_per_deg=4, n5_per_50k=10, row_from_top=True):
    s = str(sheet_num)
    if len(s) != 8 or not s.isdigit():
        return None
    lat_deg = int(s[:2])
    lon_deg = 120 + int(s[2])
    idx50 = int(s[3:5])
    idx5 = int(s[5:8])
    if not (1 <= idx50 <= n50_per_deg ** 2):
        return None
    r50 = (idx50 - 1) // n50_per_deg
    c50 = (idx50 - 1) % n50_per_deg
    step50 = 1.0 / n50_per_deg
    lat_top50 = lat_deg + 1.0 - r50 * step50 if row_from_top else lat_deg + (r50 + 1) * step50
    lon_left50 = lon_deg + c50 * step50
    if not (1 <= idx5 <= n5_per_50k ** 2):
        return None
    r5 = (idx5 - 1) // n5_per_50k
    c5 = (idx5 - 1) % n5_per_50k
    step5 = step50 / n5_per_50k
    lat_top5 = lat_top50 - r5 * step5 if row_from_top else lat_top50 + r5 * step5
    lon_left5 = lon_left50 + c5 * step5
    lat_c = lat_top5 - step5 / 2 if row_from_top else lat_top5 + step5 / 2
    lon_c = lon_left5 + step5 / 2
    return (lon_c, lat_c)

def lonlat_to_meters(lonlat_list):
    arr = np.asarray(lonlat_list, dtype=float)
    lon, lat = arr[:, 0], arr[:, 1]
    lat0 = lat.mean()
    x = lon * 111320.0 * np.cos(np.radians(lat0))
    y = lat * 110540.0
    return np.column_stack([x, y])

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

def run_spatial_independence(results_dir, label_dir, prefix=""):
    # 3. 공간 독립성 (GeoTIFF 및 도엽번호 변환 물리 거리 기반 분석)
    from collections import defaultdict
    from scipy.spatial import cKDTree
    
    dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(results_dir)))
    image_dirs = {}
    for subset in ("train", "val", "test"):
        lbl_dir = os.path.join(dataset_root, subset, "label")
        if os.path.isdir(lbl_dir):
            image_dirs[subset] = lbl_dir
        else:
            image_dirs[subset] = os.path.join(dataset_root, subset, "image")
    
    coords = {}
    crs = None
    
    def sheet_id_from_filename(fn):
        parts = os.path.splitext(fn)[0].split("_")
        region = next((p for p in parts if p in ("GG", "GS", "JL")), None)
        num = next((p for p in parts if p.isdigit() and len(p) >= 5), None)
        return f"{region}_{num}" if (region and num) else None
        
    if rasterio is not None:
        try:
            print("[공간 분석] 방법 A (GeoTIFF) 시작...")
            coords_raw = {k: defaultdict(list) for k in image_dirs}
            crs_seen = None
            
            # 첫 번째 파일의 geotransform을 미리 검사하여 NotGeoreferenced 상태인지 확인
            test_dir = image_dirs["test"]
            if os.path.isdir(test_dir):
                tifs = [f for f in os.listdir(test_dir) if f.lower().endswith((".tif", ".tiff"))]
                if tifs:
                    with rasterio.open(os.path.join(test_dir, tifs[0])) as src:
                        if src.transform == rasterio.transform.IDENTITY:
                            raise ValueError("Dataset has no geotransform (not georeferenced)")
                            
            for subset, d_dir in image_dirs.items():
                if not os.path.isdir(d_dir):
                    continue
                for f in os.listdir(d_dir):
                    if not f.lower().endswith((".tif", ".tiff")):
                        continue
                    sid = sheet_id_from_filename(f)
                    if sid is None:
                        continue
                    with rasterio.open(os.path.join(d_dir, f)) as src:
                        b = src.bounds
                        cx, cy = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
                        if crs_seen is None:
                            crs_seen = src.crs
                    coords_raw[subset][sid].append((cx, cy))
            for subset, sheets in coords_raw.items():
                coords[subset] = {sid: tuple(np.mean(v, axis=0)) for sid, v in sheets.items()}
            crs = crs_seen
            print(f"[공간 분석] GeoTIFF 실제 좌표 획득 완료 (m). 좌표계: {crs}")
        except Exception as e:
            print(f"[공간 분석] 방법 A 실패 또는 지오레퍼런스 없음: {e}. 방법 B로 전환합니다.")
            coords = {}
            
    if not coords:
        print("[공간 분석] 방법 B (도엽번호 위경도 환산) 시작...")
        for subset, d_dir in image_dirs.items():
            if not os.path.isdir(d_dir):
                continue
            sheets = {}
            for fn in os.listdir(d_dir):
                sid = sheet_id_from_filename(fn)
                if not sid:
                    continue
                num = sid.split("_")[-1]
                ll = sheet_number_to_lonlat(num)
                if ll:
                    sheets[sid] = ll
            if sheets:
                ids = list(sheets.keys())
                xy = lonlat_to_meters(list(sheets.values()))
                coords[subset] = {i: tuple(p) for i, p in zip(ids, xy)}
                
    nn_mean = 0.0
    nn_std = 0.0
    report = None
    test_sheet_distances = {}
    if "train" in coords and "test" in coords:
        tr_coords = coords["train"]
        te_coords = coords["test"]
        va_coords = coords.get("val", {})
        
        tr_xy = np.array(list(tr_coords.values()))
        te_xy = np.array(list(te_coords.values()))
        
        tr_ids = list(tr_coords.keys())
        te_ids = list(te_coords.keys())
        
        tree_tr = cKDTree(tr_xy)
        d_te2tr, idx = tree_tr.query(te_xy, k=1)
        d_km = d_te2tr / 1000.0
        
        test_sheet_distances = {sid: float(dist) for sid, dist in zip(te_ids, d_km)}
        i_min = int(np.argmin(d_km))
        
        d_te2te = np.array([0.0])
        if len(te_xy) > 1:
            tree_te = cKDTree(te_xy)
            dd, _ = tree_te.query(te_xy, k=2)
            d_te2te = dd[:, 1] / 1000.0
            
        nn_mean = np.mean(d_te2te)
        nn_std = np.std(d_te2te)
        
        report = {
            "n_sheets": {"train": len(tr_coords), "val": len(va_coords), "test": len(te_coords)},
            "test_to_nearest_train_km": {
                "mean": float(d_km.mean()), "std": float(d_km.std()),
                "min": float(d_km.min()), "max": float(d_km.max()),
                "median": float(np.median(d_km)),
                "p05": float(np.percentile(d_km, 5)),
                "p95": float(np.percentile(d_km, 95)),
            },
            "test_to_nearest_test_km": {
                "mean": float(nn_mean), "std": float(nn_std),
                "min": float(d_te2te.min()), "max": float(d_te2te.max()),
            },
            "test_sheets_adjacent_to_train": int((d_km < 2.5).sum()),
            "test_sheets_within_5km_of_train": int((d_km < 5.0).sum()),
        }
        
        print("\n=== 공간 독립성 분석 ===")
        print(f"도엽 수: train {len(tr_coords)} / val {len(va_coords)} / test {len(te_coords)}")
        r_t2tr = report["test_to_nearest_train_km"]
        print(f"시험→최근접 학습 도엽 거리: {r_t2tr['mean']:.2f} ± {r_t2tr['std']:.2f} km "
              f"(최소 {r_t2tr['min']:.2f}, 중앙 {r_t2tr['median']:.2f}, 최대 {r_t2tr['max']:.2f})")
        print(f"학습 도엽과 인접(<2.5km)한 시험 도엽: {report['test_sheets_adjacent_to_train']}개")
        print(f"학습 도엽 5km 이내 시험 도엽:        {report['test_sheets_within_5km_of_train']}개")
        print(f"[공간 분석] 최소 거리 쌍: 시험 {te_ids[i_min]} ↔ 학습 {tr_ids[idx[i_min]]} = {d_km[i_min]:.2f} km")
        print(f"[공간 분석] 시험 도엽 간 최근접 물리적 거리 분포: {nn_mean:.2f} ± {nn_std:.2f} km")
        
        out_json = os.path.join(results_dir, prefix + "spatial_independence.json")
        with open(out_json, "w", encoding="utf-8") as f_json:
            json.dump(report, f_json, ensure_ascii=False, indent=2)
        print(f"공간 독립성 분석 결과 저장 완료: {out_json}")
        
    return report, crs, test_sheet_distances

# ----------------------------- 온도 피팅 -----------------------------

def fit_temperature(val_unc_dir, label_dir, dn_to_id, is_16bit):
    """검증세트의 logits_det로 NLL을 최소화하는 온도 T를 피팅한다."""
    print("[TS] 검증세트 Logit 수집 및 온도 피팅 시작...")
    val_files = sorted([f for f in os.listdir(val_unc_dir) if f.endswith(".npz")])
    if not val_files:
        print("[TS] [경고] 검증세트 npz 없음. T=1.0 적용.")
        return 1.0

    rng = np.random.default_rng(42)
    rng.shuffle(val_files)
    val_files = val_files[:100]

    all_logits, all_labels = [], []
    n_total, n_no_label, n_no_logit = 0, 0, 0
    for f in val_files:
        n_total += 1
        d = np.load(os.path.join(val_unc_dir, f), allow_pickle=True)
        if "logits_det" not in d.files:
            n_no_logit += 1
            continue
        lbl_path = find_label(label_dir, str(d["filename"]))
        if lbl_path is None:
            n_no_label += 1
            continue
        gt = load_label_as_id(lbl_path, dn_to_id, is_16bit)   # 0=배경, 1~10=전경
        logits_det = d["logits_det"]                          # (C, H, W)

        # ★ 수정 1: 전체 화소 사용 (배경 포함). 라벨 0~10 그대로 → 11채널 인덱싱 안전.
        #    (calibration은 최종 지도 전체에 대한 것이므로 전체 화소가 옳다)
        l_v = logits_det.transpose(1, 2, 0).reshape(-1, logits_det.shape[0])  # (P, C)
        g_v = gt.reshape(-1)                                                   # (P,)
        all_logits.append(l_v)
        all_labels.append(g_v)

    if not all_logits:
        print(f"[TS] [실패] 유효 샘플 0개 "
              f"(전체 {n_total} / 라벨 못 찾음 {n_no_label} / logits_det 없음 {n_no_logit})")
        print(f"       라벨 폴더: {label_dir}")
        return 1.0

    logits_concat = np.concatenate(all_logits, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)

    # 최대 500k 픽셀 샘플링
    max_samples = 500000
    if logits_concat.shape[0] > max_samples:
        idx = rng.choice(logits_concat.shape[0], max_samples, replace=False)
        logits_sample = logits_concat[idx].astype(np.float64)
        labels_sample = labels_concat[idx].astype(np.int64)
    else:
        logits_sample = logits_concat.astype(np.float64)
        labels_sample = labels_concat.astype(np.int64)

    # 방어: 라벨 범위가 채널 수와 맞는지 확인
    C = logits_sample.shape[1]
    if labels_sample.max() >= C or labels_sample.min() < 0:
        print(f"[TS] [경고] 라벨 범위 {labels_sample.min()}~{labels_sample.max()} vs 채널 {C}. 클리핑.")
        labels_sample = np.clip(labels_sample, 0, C - 1)

    def nll_for_T(T):
        if T <= 0:
            return 1e9
        z = logits_sample / T
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        p = np.clip(p, 1e-12, 1.0)
        pc = p[np.arange(len(labels_sample)), labels_sample]
        return -np.log(pc).mean()

    # ★ 수정 2: 스칼라 1D 탐색(황금분할). L-BFGS-B의 유한차분 실패를 회피.
    res = opt.minimize_scalar(nll_for_T, bounds=(0.5, 5.0), method="bounded",
                              options={"xatol": 1e-4})
    T_opt = float(res.x)

    # 안전장치: T가 경계(1.0 근처)에 붙었는지, NLL이 실제로 개선됐는지 로그
    nll_1 = nll_for_T(1.0); nll_opt = nll_for_T(T_opt)
    print(f"[TS] 피팅 완료: T = {T_opt:.4f}  (NLL {nll_1:.4f} → {nll_opt:.4f})")
    if abs(T_opt - 1.0) < 1e-3:
        print("[TS] [주의] T≈1.0. 검증셋 logit이 이미 잘 보정됐거나, 데이터 확인 필요.")
    return T_opt

def generate_qualitative_panels(results_dir, label_dir, mapping_file, num_classes, dn_to_id, prefix=""):
    print("[Vis] 정성적 분석 패널(Qualitative Panels) 자동 생성 시작...")
    import glob
    import random
    import matplotlib.pyplot as plt
    
    test_dir = os.path.dirname(os.path.normpath(label_dir))
    image_dir = os.path.join(test_dir, "image")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(test_dir, "images")
        
    res_dir = os.path.join(results_dir, "uncertainty_data")
    out_dir = os.path.join(results_dir, "Qualitative_Panels")
    os.makedirs(out_dir, exist_ok=True)
    
    npz_files = glob.glob(os.path.join(res_dir, "*.npz"))
    if not npz_files:
        print("[Vis] [경고] npz 파일이 존재하지 않아 정성적 패널 생성을 건너뜁니다.")
        return
        
    random.seed(42)
    selected_npz = random.sample(npz_files, min(10, len(npz_files)))
    try:
        cmap = plt.colormaps.get_cmap('tab20')
    except AttributeError:
        cmap = plt.cm.get_cmap('tab20')
    
    for i, npz_path in enumerate(selected_npz):
        basename = os.path.basename(npz_path)
        base_no_ext = os.path.splitext(basename)[0]
        
        # 1. 이미지 찾기
        img_path = None
        for ext in [".tif", ".tiff", ".png", ".jpg"]:
            p = os.path.join(image_dir, base_no_ext + ext)
            if os.path.exists(p):
                img_path = p
                break
        if not img_path:
            continue
            
        try:
            if img_path.lower().endswith((".tif", ".tiff")) and rasterio is not None:
                with rasterio.open(img_path) as src:
                    rgb = src.read([1, 2, 3])
                    rgb_arr = np.transpose(rgb, (1, 2, 0))
                    if rgb_arr.max() > 255:
                        rgb_arr = (rgb_arr / 65535.0 * 255.0).astype(np.uint8)
                    else:
                        rgb_arr = rgb_arr.astype(np.uint8)
                    img = Image.fromarray(rgb_arr)
            else:
                img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Vis] 이미지 로드 실패: {img_path}, {e}")
            continue
            
        # 2. npz 로드
        try:
            d = np.load(npz_path)
            pred = d['pred_id']
            unc = d['std']
        except Exception as e:
            print(f"[Vis] npz 로드 실패: {npz_path}, {e}")
            continue
            
        # 3. 라벨 로드
        lbl_path = None
        for ext in [".tif", ".tiff", ".png", ".jpg"]:
            p = os.path.join(label_dir, base_no_ext + ext)
            if os.path.exists(p):
                lbl_path = p
                break
        if lbl_path:
            try:
                lbl_raw = load_label_as_id(lbl_path, dn_to_id, is_16bit=False)
                gt = lbl_raw
            except Exception as e:
                gt = np.zeros_like(pred)
        else:
            gt = np.zeros_like(pred)
            
        valid = (gt != 0)
        error_map = np.zeros_like(pred, dtype=np.float32)
        error_map[valid & (pred != gt)] = 1.0
        
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
        
        im_unc = axes[4].imshow(unc, cmap='jet', vmin=0, vmax=unc.max() if unc.max() > 0 else 1.0)
        axes[4].set_title("Uncertainty (STD)")
        axes[4].axis('off')
        
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"panel_{i+1:02d}_{base_no_ext}.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"[Vis] 정성적 분석 패널 생성 완료: {out_dir}")

# ----------------------------- 메인 스크립트 -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, help="테스트셋 InferenceResults 경로")
    ap.add_argument("--val_results_dir", default="", help="검증셋 InferenceResults 경로 (온도 스케일링용)")
    ap.add_argument("--mapping_file", required=True)
    ap.add_argument("--label_dir", default="")
    ap.add_argument("--uncertainty", choices=["bald", "entropy", "std", "max_softmax"], default="bald")
    ap.add_argument("--is_16bit", default="False")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--model_arch", default="mask2former")
    ap.add_argument("--bootstrap_runs", type=int, default=1000)
    ap.add_argument("--spatial_only", action="store_true", help="공간 독립성 분석만 수행하고 종료")
    args = ap.parse_args()

    is_16bit = str(args.is_16bit).lower() == "true"
    prefix = args.model_arch.lower() + "_" if args.model_arch else ""

    # 공간 변수 초기화
    report = None
    crs = None
    test_sheet_distances = {}
    spatial_sensitivity = None

    # 라벨 폴더 파악 (공간 분석과 메인 분석 공통으로 사용)
    label_dir = args.label_dir or os.path.join(os.path.dirname(os.path.normpath(args.results_dir)), "label")
    if not os.path.isdir(label_dir):
        # Fallback to parent sibling
        label_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(args.results_dir))), "label")
        if not os.path.isdir(label_dir):
            sys.exit(f"라벨 폴더가 존재하지 않습니다: {label_dir}")

    # 공간 독립성 분석만 수행하는 모드인 경우 여기서 실행하고 조기 종료
    if args.spatial_only:
        run_spatial_independence(args.results_dir, label_dir, prefix)
        sys.exit(0)

    test_unc_dir = os.path.join(args.results_dir, "uncertainty_data")
    if not os.path.isdir(test_unc_dir):
        sys.exit(f"테스트셋 uncertainty_data 폴더가 없습니다: {test_unc_dir}")

    with open(args.mapping_file, "r") as f:
        mapping = json.load(f)
    dn_map = mapping.get("dn_map", {})
    num_classes = len(dn_map)
    dn_to_id = {v: int(k) for k, v in dn_map.items()}
    class_names = build_class_names(mapping, num_classes)

    # 1. Temperature Scaling 피팅
    T = 1.0
    val_results_dir = args.val_results_dir
    if not val_results_dir or not os.path.exists(val_results_dir):
        try:
            results_dir_norm = os.path.normpath(args.results_dir)
            model_part = os.path.basename(results_dir_norm).split("_Inference_", 1)[0]
            parent_results = os.path.dirname(os.path.dirname(results_dir_norm))
            
            # 검증셋의 이미지 파일 개수를 동적으로 측정 (e.g. 1233개)
            dataset_root = os.path.dirname(os.path.dirname(os.path.dirname(results_dir_norm)))
            val_img_dir = os.path.join(dataset_root, "val", "image")
            n_val_images = 0
            if os.path.exists(val_img_dir):
                n_val_images = len([f for f in os.listdir(val_img_dir) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))])
            
            candidates = []
            for root, dirs, files in os.walk(parent_results):
                for d in dirs:
                    path = os.path.join(root, d)
                    if os.path.normpath(path) == results_dir_norm:
                        continue
                    if d.startswith(f"{model_part}_Inference_"):
                        unc_dir = os.path.join(path, "uncertainty_data")
                        if os.path.exists(unc_dir):
                            npzs = [f for f in os.listdir(unc_dir) if f.endswith(".npz")]
                            # 경로명에 'val'이 있거나, npz 파일 개수가 검증셋 이미지 개수와 대략 일치하는 경우(오차 범위 +-10개) 검증 폴더로 판정
                            is_val_path = "val" in root.replace("\\", "/").split("/")
                            is_val_count = (n_val_images > 0 and abs(len(npzs) - n_val_images) <= 10)
                            
                            if is_val_path or is_val_count:
                                candidates.append((path, os.path.getmtime(path)))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                val_results_dir = candidates[0][0]
                print(f"[TS] 최신 매칭 검증 폴더가 자동 탐색되어 연동합니다: {val_results_dir}")
        except Exception:
            pass

    if val_results_dir and os.path.exists(val_results_dir):
        val_unc_dir = os.path.join(val_results_dir, "uncertainty_data")
        if os.path.isdir(val_unc_dir):
            # ★ 검증셋 전용 라벨 폴더를 찾는다 (테스트 라벨 폴더를 쓰면 매칭 실패)
            val_label_dir = derive_val_label_dir(label_dir, val_results_dir)
            if val_label_dir is None:
                print(f"[TS] [경고] 검증셋 라벨 폴더를 찾지 못했습니다. "
                      f"(테스트 라벨: {label_dir}) T=1.0으로 진행합니다.")
            else:
                print(f"[TS] 검증셋 라벨 폴더: {val_label_dir}")
                T = fit_temperature(val_unc_dir, val_label_dir, dn_to_id, is_16bit)

    # 2. 테스트셋 npz 파일들 로드
    npz_files = sorted([f for f in os.listdir(test_unc_dir) if f.endswith(".npz")])
    if not npz_files:
        sys.exit("테스트셋 uncertainty_data에 .npz가 존재하지 않습니다.")

    print(f"테스트셋 {len(npz_files)}개 파일 로딩 중...")
    
    loaded_tiles = []
    unique_sheets = set()
    region_counts = {"GG": 0, "GS": 0, "JL": 0}
    
    reliability_gt_list = []
    reliability_logits_list = []
    
    rng_load = np.random.default_rng(42)
    
    for f_idx, f in enumerate(npz_files):
        if (f_idx + 1) % 200 == 0:
            print(f"  - Loaded {f_idx + 1} / {len(npz_files)} files...", flush=True)
            
        d = np.load(os.path.join(test_unc_dir, f), allow_pickle=True)
        lbl_path = find_label(label_dir, str(d["filename"]))
        if lbl_path is None:
            continue
        gt = load_label_as_id(lbl_path, dn_to_id, is_16bit)
        
        # sheet_id 추출
        sheet_id = str(d.get("sheet_id", "unknown"))
        parts_sid = sheet_id.split("_")
        has_digit_code = len(parts_sid) >= 2 and parts_sid[1].isdigit()
        
        if sheet_id in ["unknown", "None", None, ""] or not has_digit_code:
            import re
            m = re.search(r"(GG|GS|JL)_(?:[A-Za-z0-9]+_)?(\d{5,8})", f)
            if m:
                sheet_id = f"{m.group(1)}_{m.group(2)}"
        
        region = sheet_id.split("_")[0] if "_" in sheet_id else "GG"
        if region in region_counts:
            region_counts[region] += 1
            
        unique_sheets.add(sheet_id)
        
        # 2D arrays
        t_gt = gt
        t_probs_mean = d["probs_mean"]
        t_logits_det = d["logits_det"]
        t_pred_det = d["pred_deterministic"]
        t_entropy_det = d["entropy_deterministic"]
        t_entropy = d["entropy"]
        t_bald = d["bald"]
        t_std = d["std"]
        t_conf = d["conf"]
        
        # 앙상블 예측 argmax 미리 연산
        pred_ensemble = np.argmax(t_probs_mean, axis=0).astype(np.uint8)
        t_pred_id = d["pred_id"]
        
        # Reliability Diagram용 foreground 픽셀들 샘플링
        gt_flat_rd = t_gt.flatten()
        fg_mask_rd = gt_flat_rd != 0
        if fg_mask_rd.any():
            fg_indices = np.where(fg_mask_rd)[0]
            n_rd_samples = min(len(fg_indices), 500)
            sel_rd = rng_load.choice(fg_indices, n_rd_samples, replace=False)
            reliability_gt_list.append(gt_flat_rd[sel_rd])
            reliability_logits_list.append(t_logits_det.reshape(num_classes, -1).T[sel_rd])
            
        # Pre-sampled 1D arrays (for fast bootstrap evaluation)
        gt_flat = gt.flatten()
        probs_mean_flat = t_probs_mean.reshape(num_classes, -1).T
        logits_det_flat = t_logits_det.reshape(num_classes, -1).T
        pred_det_flat = t_pred_det.flatten()
        entropy_det_flat = t_entropy_det.flatten()
        pred_id_flat = t_pred_id.flatten()
        bald_flat = t_bald.flatten()
        entropy_flat = t_entropy.flatten()
        std_flat = t_std.flatten()
        conf_flat = t_conf.flatten()
        
        n_pix = gt_flat.size
        sample_size = 300
        if n_pix > sample_size:
            sel = rng_load.choice(n_pix, sample_size, replace=False)
            gt_sampled = gt_flat[sel]
            probs_mean_sampled = probs_mean_flat[sel]
            logits_det_sampled = logits_det_flat[sel]
            pred_det_sampled = pred_det_flat[sel]
            entropy_det_sampled = entropy_det_flat[sel]
            pred_id_sampled = pred_id_flat[sel]
            bald_sampled = bald_flat[sel]
            entropy_sampled = entropy_flat[sel]
            std_sampled = std_flat[sel]
            conf_sampled = conf_flat[sel]
        else:
            gt_sampled = gt_flat
            probs_mean_sampled = probs_mean_flat
            logits_det_sampled = logits_det_flat
            pred_det_sampled = pred_det_flat
            entropy_det_sampled = entropy_det_flat
            pred_id_sampled = pred_id_flat
            bald_sampled = bald_flat
            entropy_sampled = entropy_flat
            std_sampled = std_flat
            conf_sampled = conf_flat
            
        loaded_tiles.append({
            "filename": str(d["filename"]),
            "sheet_id": sheet_id,
            
            # 2D arrays
            "gt": t_gt,
            "pred_ensemble": pred_ensemble,
            "pred_id": t_pred_id,
            "entropy_deterministic": t_entropy_det,
            "entropy": t_entropy,
            "bald": t_bald,
            "std": t_std,
            "conf": t_conf,
            
            # 1D pre-sampled arrays
            "gt_sampled": gt_sampled,
            "probs_mean_sampled": probs_mean_sampled,
            "logits_det_sampled": logits_det_sampled,
            "pred_deterministic_sampled": pred_det_sampled,
            "entropy_deterministic_sampled": entropy_det_sampled,
            "pred_id_sampled": pred_id_sampled,
            "bald_sampled": bald_sampled,
            "entropy_sampled": entropy_sampled,
            "std_sampled": std_sampled,
            "conf_sampled": conf_sampled,
            
            "region": region
        })

    print(f"로드 완료: 총 {len(loaded_tiles)}개 타일, 고유 도엽 수: {len(unique_sheets)}개")

    report, crs, test_sheet_distances = run_spatial_independence(args.results_dir, label_dir, prefix)

    # 4. 평가 함수 정의 (All-pixel vs FG-only, 3 Conditions)
    def run_eval_pipeline(tiles_subset):
        # 결과 누적 배열 준비
        results = {
            "all_pixel": {"c1": {}, "c2": {}, "c3": {}},
            "fg_only": {"c1": {}, "c2": {}, "c3": {}}
        }
        
        # 이미 300픽셀로 pre-sampling 되었으므로 리스트 내포로 병합만 수행 (극도로 빠름)
        all_gt = [t["gt_sampled"] for t in tiles_subset]
        all_mean_p = [t["probs_mean_sampled"] for t in tiles_subset]
        all_logits_det = [t["logits_det_sampled"] for t in tiles_subset]
        all_entropy_det = [t["entropy_deterministic_sampled"] for t in tiles_subset]
        all_pred_det = [t["pred_deterministic_sampled"] for t in tiles_subset]
        all_pred_id = [t["pred_id_sampled"] for t in tiles_subset]
        all_bald = [t["bald_sampled"] for t in tiles_subset]
        all_entropy = [t["entropy_sampled"] for t in tiles_subset]
        all_std = [t["std_sampled"] for t in tiles_subset]
        
        gt = np.concatenate(all_gt, axis=0)
        probs_mean = np.concatenate(all_mean_p, axis=0)
        logits_det = np.concatenate(all_logits_det, axis=0)
        entropy_det = np.concatenate(all_entropy_det, axis=0)
        pred_det = np.concatenate(all_pred_det, axis=0)
        pred_id = np.concatenate(all_pred_id, axis=0)
        bald = np.concatenate(all_bald, axis=0)
        entropy = np.concatenate(all_entropy, axis=0)
        std = np.concatenate(all_std, axis=0)
        
        # 유효 마스크 정의
        masks = {
            "all_pixel": (gt >= 0),
            "fg_only": (gt != 0)
        }
        
        for mode, mask in masks.items():
            gt_m = gt[mask]
            probs_mean_m = probs_mean[mask]
            logits_det_m = logits_det[mask]
            pred_det_m = pred_det[mask]
            entropy_det_m = entropy_det[mask]
            pred_id_m = pred_id[mask]
            bald_m = bald[mask]
            entropy_m = entropy[mask]
            std_m = std[mask]
            
            if gt_m.size == 0:
                continue
                
            # --- Condition 1 (임계값 없음, 순수 argmax) ---
            pred_c1 = np.argmax(probs_mean_m, axis=1)
            conf_c1 = np.max(probs_mean_m, axis=1)
            hist_c1 = fast_hist_np(gt_m, pred_c1, num_classes, ignore=(0 if mode == "fg_only" else None))
            miou_c1, iu_c1 = miou_from_hist(hist_c1, include_zero=(mode == "all_pixel"))
            
            clsm_ignore = 0 if mode == "fg_only" else 99999
            clsm_c1 = classification_metrics_from_hist(hist_c1, ignore=clsm_ignore)
            
            # 실제 저장된 pred_id 평가 (진단용)
            hist_pred_id = fast_hist_np(gt_m, pred_id_m, num_classes, ignore=(0 if mode == "fg_only" else None))
            miou_pred_id, _ = miou_from_hist(hist_pred_id, include_zero=(mode == "all_pixel"))
            if mode == "all_pixel":
                print(f"[TS] [진단 - {mode}] argmax(probs_mean) mIoU: {miou_c1:.4f} | 실제 저장된 pred_id mIoU: {miou_pred_id:.4f}")
            
            correct_c1 = (pred_c1 == gt_m)
            ece_c1, _, _, _, _ = expected_calibration_error(conf_c1, correct_c1)
            
            err_c1 = ~correct_c1
            try:
                auroc_c1 = fast_auroc(err_c1, entropy_m)
                auprc_c1 = fast_auprc(err_c1, entropy_m)
            except:
                auroc_c1, auprc_c1 = 0.5, 0.0
                
            # Sparsification AUSE (oracle_pixel vs entropy)
            _, _, _, _, ause_c1 = sparsification(entropy_m, err_c1)
            
            results[mode]["c1"] = {
                "mIoU": miou_c1, "ECE": ece_c1, "AUROC": auroc_c1, "AUPRC": auprc_c1, "AUSE": ause_c1,
                "error_base_rate": err_c1.mean(),
                "OA": clsm_c1["overall_accuracy"],
                "Kappa": clsm_c1["kappa"],
                "macro_precision": clsm_c1["macro_precision"],
                "macro_recall": clsm_c1["macro_recall"],
                "macro_f1": clsm_c1["macro_f1"],
                "class_precision": clsm_c1["precision"],
                "class_recall": clsm_c1["recall"],
                "class_f1": clsm_c1["f1"],
                "class_iou": clsm_c1["iou"]
            }
            
            # --- Condition 2 (선택적 분류, Abstain 필터링) ---
            # threshold t = 0.5
            t = 0.5
            max_p_c2 = np.max(probs_mean_m, axis=1)
            pred_c2_raw = np.argmax(probs_mean_m, axis=1)
            
            # accepted 화소 마스킹
            accepted_mask = max_p_c2 >= t
            coverage = float(accepted_mask.sum()) / gt_m.size
            
            if accepted_mask.any():
                pred_c2_acc = pred_c2_raw[accepted_mask]
                gt_c2_acc = gt_m[accepted_mask]
                # Abstain(-1)을 마스킹하여 순수 accepted 화소 혼동행렬만 산출
                hist_c2 = fast_hist_np(gt_c2_acc, pred_c2_acc, num_classes, ignore=(0 if mode == "fg_only" else None))
                miou_c2, _ = miou_from_hist(hist_c2, include_zero=(mode == "all_pixel"))
                sel_acc = float((pred_c2_acc == gt_c2_acc).mean())
            else:
                miou_c2 = 0.0
                sel_acc = 1.0
                
            results[mode]["c2"] = {
                "coverage": coverage, "selective_accuracy": sel_acc, "mIoU": miou_c2
            }
            
            # --- Condition 3 (배경 환원 Operating Rule) ---
            # threshold t = 0.5 미만은 Class 0(배경)으로 강등
            max_p_c3 = np.max(probs_mean_m, axis=1)
            pred_c3_raw = np.argmax(probs_mean_m, axis=1)
            pred_c3 = np.where(max_p_c3 >= t, pred_c3_raw, 0)
            
            # ECE 신뢰도 교정: 강등 화소는 해당 픽셀의 p0 확률로 교체!
            conf_c3 = np.where(max_p_c3 >= t, max_p_c3, probs_mean_m[:, 0])
            correct_c3 = (pred_c3 == gt_m)
            ece_c3, _, _, _, _ = expected_calibration_error(conf_c3, correct_c3)
            
            hist_c3 = fast_hist_np(gt_m, pred_c3, num_classes, ignore=(0 if mode == "fg_only" else None))
            miou_c3, _ = miou_from_hist(hist_c3, include_zero=(mode == "all_pixel"))
            
            err_c3 = ~correct_c3
            try:
                auroc_c3 = fast_auroc(err_c3, entropy_m)
                auprc_c3 = fast_auprc(err_c3, entropy_m)
            except:
                auroc_c3, auprc_c3 = 0.5, 0.0
                
            _, _, _, _, ause_c3 = sparsification(entropy_m, err_c3)
            
            # Omission & Commission error
            if mode == "all_pixel":
                omission = float(((gt_m > 0) & (pred_c3 == 0)).sum()) / max(1, (gt_m > 0).sum())
                commission = float(((gt_m == 0) & (pred_c3 > 0)).sum()) / max(1, (gt_m == 0).sum())
            else:
                omission, commission = 0.0, 0.0
                
            results[mode]["c3"] = {
                "mIoU": miou_c3, "ECE": ece_c3, "AUROC": auroc_c3, "AUPRC": auprc_c3, "AUSE": ause_c3,
                "omission_error": omission, "commission_error": commission
            }
            
            # --- Temperature Scaled Deterministic Pass (Condition 1 적용) ---
            # 1) Raw (T=1.0) Deterministic metrics
            logits_det_raw_m = logits_det_m
            logits_det_raw_shifted = logits_det_raw_m - np.max(logits_det_raw_m, axis=1, keepdims=True)
            exp_logits_raw = np.exp(logits_det_raw_shifted)
            probs_det_raw = exp_logits_raw / np.sum(exp_logits_raw, axis=1, keepdims=True)
            
            pred_det_raw = np.argmax(probs_det_raw, axis=1)
            conf_det_raw = np.max(probs_det_raw, axis=1)
            correct_det_raw = (pred_det_raw == gt_m)
            ece_det_raw, _, _, _, _ = expected_calibration_error(conf_det_raw, correct_det_raw)

            # 2) Scaled (T fitted) Deterministic metrics
            logits_det_scaled_m = logits_det_m / T
            logits_det_shifted = logits_det_scaled_m - np.max(logits_det_scaled_m, axis=1, keepdims=True)
            exp_logits = np.exp(logits_det_shifted)
            probs_det_scaled = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            pred_det_scaled = np.argmax(probs_det_scaled, axis=1)
            conf_det_scaled = np.max(probs_det_scaled, axis=1)
            
            correct_det_scaled = (pred_det_scaled == gt_m)
            ece_det_scaled, _, _, _, _ = expected_calibration_error(conf_det_scaled, correct_det_scaled)
            
            # Brier Score & NLL (Clamped to avoid underflow)
            probs_det_clamped = np.clip(probs_det_scaled, 1e-7, 1.0)
            nll_det = -np.log(probs_det_clamped[np.arange(len(gt_m)), gt_m]).mean()
            
            # Brier Score
            one_hot_gt = np.zeros_like(probs_det_scaled)
            one_hot_gt[np.arange(len(gt_m)), gt_m] = 1.0
            brier_det = np.mean(np.sum((probs_det_scaled - one_hot_gt)**2, axis=1))
            
            results[mode]["det_scaled"] = {
                "ECE_raw": ece_det_raw,
                "ECE_scaled": ece_det_scaled,
                "NLL": float(nll_det),
                "Brier": float(brier_det)
            }
            
            # --- MC Dropout vs Deterministic Error Detection Comparison ---
            conf_det_m = conf_det_scaled
            err_det_m = ~correct_det_scaled
            unc_det_max_softmax = 1.0 - conf_det_m
            unc_det_entropy = entropy_det_m
            
            conf_mc_m = np.max(probs_mean_m, axis=1)
            correct_mc_m = (np.argmax(probs_mean_m, axis=1) == gt_m)
            err_mc_m = ~correct_mc_m
            unc_mc_max_softmax = 1.0 - conf_mc_m
            
            unc_mc_entropy = entropy_m
            unc_mc_bald = bald_m
            unc_mc_std = std_m
            
            def get_auc_ause(unc_arr, err_arr):
                try:
                    auc = roc_auc_score(err_arr, unc_arr)
                except:
                    auc = 0.5
                try:
                    _, _, _, _, ause_val = sparsification(unc_arr, err_arr)
                except:
                    ause_val = 0.0
                return auc, ause_val
                
            auc_det_ms, ause_det_ms = get_auc_ause(unc_det_max_softmax, err_det_m)
            auc_det_ent, ause_det_ent = get_auc_ause(unc_det_entropy, err_det_m)
            
            auc_mc_ms, ause_mc_ms = get_auc_ause(unc_mc_max_softmax, err_mc_m)
            auc_mc_ent, ause_mc_ent = get_auc_ause(unc_mc_entropy, err_mc_m)
            auc_mc_bald, ause_mc_bald = get_auc_ause(unc_mc_bald, err_mc_m)
            auc_mc_std, ause_mc_std = get_auc_ause(unc_mc_std, err_mc_m)
            
            results[mode]["mc_dropout_vs_deterministic"] = {
                "deterministic_max_softmax": {"auroc": float(auc_det_ms), "ause": float(ause_det_ms)},
                "deterministic_entropy": {"auroc": float(auc_det_ent), "ause": float(ause_det_ent)},
                "mc_max_softmax": {"auroc": float(auc_mc_ms), "ause": float(ause_mc_ms)},
                "mc_entropy": {"auroc": float(auc_mc_ent), "ause": float(ause_mc_ent)},
                "mc_bald": {"auroc": float(auc_mc_bald), "ause": float(ause_mc_bald)},
                "mc_std": {"auroc": float(auc_mc_std), "ause": float(ause_mc_std)}
            }
            
        return results

    # 5. 52개 지도도엽 군집 단위 Bootstrap (1000회 리샘플링)
    print(f"[Bootstrap] {args.bootstrap_runs}회 군집 부트스트랩 기동 중...")
    
    # Pre-concatenate tiles by sheet_id to accelerate bootstrap concatenation (from 1,689 arrays to 52 arrays)
    print("[Bootstrap] 도엽 단위 사전 병합(Pre-concatenation) 시작...")
    from collections import defaultdict
    sheet_grouped = defaultdict(list)
    for t in loaded_tiles:
        sheet_grouped[t["sheet_id"]].append(t)
        
    loaded_sheets = []
    for s_id, tiles in sheet_grouped.items():
        loaded_sheets.append({
            "sheet_id": s_id,
            "gt_sampled": np.concatenate([t["gt_sampled"] for t in tiles], axis=0),
            "probs_mean_sampled": np.concatenate([t["probs_mean_sampled"] for t in tiles], axis=0),
            "logits_det_sampled": np.concatenate([t["logits_det_sampled"] for t in tiles], axis=0),
            "pred_deterministic_sampled": np.concatenate([t["pred_deterministic_sampled"] for t in tiles], axis=0),
            "entropy_deterministic_sampled": np.concatenate([t["entropy_deterministic_sampled"] for t in tiles], axis=0),
            "pred_id_sampled": np.concatenate([t["pred_id_sampled"] for t in tiles], axis=0),
            "bald_sampled": np.concatenate([t["bald_sampled"] for t in tiles], axis=0),
            "entropy_sampled": np.concatenate([t["entropy_sampled"] for t in tiles], axis=0),
            "std_sampled": np.concatenate([t["std_sampled"] for t in tiles], axis=0),
            "conf_sampled": np.concatenate([t["conf_sampled"] for t in tiles], axis=0),
        })
    sheet_dict = {sh["sheet_id"]: sh for sh in loaded_sheets}
    print(f"[Bootstrap] 사전 병합 완료! 고유 도엽 수: {len(loaded_sheets)}개")
    
    sheet_list = list(unique_sheets)
    n_sheets = len(sheet_list)
    rng_bs = np.random.default_rng(42)
    
    bootstrap_results = []
    
    for bs_idx in range(args.bootstrap_runs):
        if (bs_idx + 1) % max(1, args.bootstrap_runs // 10) == 0:
            print(f"  - Resampling Progress: {int(((bs_idx + 1) / args.bootstrap_runs) * 100)}%")
        sampled_sheets = rng_bs.choice(sheet_list, size=n_sheets, replace=True)
        sampled_tiles = [sheet_dict[s] for s in sampled_sheets if s in sheet_dict]
                    
        if len(sampled_tiles) > 0:
            bs_met = run_eval_pipeline(sampled_tiles)
            bootstrap_results.append(bs_met)
            
    # Bootstrap 신뢰구간(CI) 분석
    def get_ci_bounds(vals):
        vals_sorted = np.sort(vals)
        low = np.percentile(vals_sorted, 2.5)
        high = np.percentile(vals_sorted, 97.5)
        return float(np.mean(vals_sorted)), float(low), float(high)

    # Helper to run grid analysis for HITL inspection efficiency
    def run_grid_analysis(tiles, uncertainty_key):
        grid_results = {}
        for grid_size in [256, 512, 1024]:
            t_err_grid = []
            t_npix_grid = []
            stat_uncs = {
                "mean": [], "median": [], "p90": [], "p95": [], "top10_mean": []
            }
            for t in tiles:
                gt = t["gt"]
                pred = t["pred_ensemble"]
                err = (pred != gt) & (gt != 0)
                unc_bald = t[uncertainty_key]
                
                H, W = gt.shape
                for y in range(0, H, grid_size):
                    for x in range(0, W, grid_size):
                        vm = (gt[y:y+grid_size, x:x+grid_size] != 0)
                        npix = int(vm.sum())
                        if npix == 0:
                            continue
                        em = err[y:y+grid_size, x:x+grid_size]
                        tile_unc_pixels = unc_bald[y:y+grid_size, x:x+grid_size][vm]
                        
                        t_npix_grid.append(npix)
                        t_err_grid.append(int(em.sum()))
                        
                        stat_uncs["mean"].append(float(tile_unc_pixels.mean()))
                        stat_uncs["median"].append(float(np.median(tile_unc_pixels)))
                        stat_uncs["p90"].append(float(np.percentile(tile_unc_pixels, 90)))
                        stat_uncs["p95"].append(float(np.percentile(tile_unc_pixels, 95)))
                        
                        n_top = max(1, int(len(tile_unc_pixels) * 0.1))
                        top10_val = float(np.sort(tile_unc_pixels)[-n_top:].mean())
                        stat_uncs["top10_mean"].append(top10_val)
                        
            if not t_err_grid:
                continue
                
            t_err_grid = np.array(t_err_grid, dtype=np.float64)
            t_npix_grid = np.array(t_npix_grid, dtype=np.float64)
            
            grid_results[f"grid_{grid_size}"] = {}
            for stat, unc_list in stat_uncs.items():
                unc_arr = np.array(unc_list)
                res = hitl_curve(unc_arr, t_err_grid, t_npix_grid)
                if res:
                    (a_u, e_u), (a_o, e_o), summ = res
                    idx50 = np.searchsorted(e_u, 0.5)
                    idx80 = np.searchsorted(e_u, 0.8)
                    area_50 = float(a_u[idx50]) if idx50 < len(a_u) else 1.0
                    area_80 = float(a_u[idx80]) if idx80 < len(a_u) else 1.0
                    
                    auc_proposed = _trapz(e_u, a_u)
                    auc_oracle = _trapz(e_o, a_o)
                    norm_efficiency = auc_proposed / auc_oracle if auc_oracle > 0 else 0.0
                    
                    grid_results[f"grid_{grid_size}"][stat] = {
                        "50%_error_area": area_50,
                        "80%_error_area": area_80,
                        "normalized_efficiency": norm_efficiency,
                        "summ_20": summ.get("20%_area", 0)
                    }
        return grid_results

    # 6. 전체 테스트셋 원본 성능 산출
    final_metrics = run_eval_pipeline(loaded_tiles)

    # 6.5. 공간 독립성 민감도 분석 (Sensitivity Analysis - 5km 이내 인접 도엽 제외 테스트)
    if test_sheet_distances:
        near_sheets = {sid for sid, dist in test_sheet_distances.items() if dist < 5.0}
        if near_sheets:
            tiles_far = [t for t in loaded_tiles if t["sheet_id"] not in near_sheets]
            print(f"\n[공간 민감도 분석] 학습 도엽과 5km 이내 인접한 시험 도엽 {len(near_sheets)}개 감지: {near_sheets}")
            print(f"[공간 민감도 분석] 해당 도엽들을 제외한 엄격한 공간 독립성 검수 세트 (타일 수: {len(tiles_far)} / {len(loaded_tiles)}개)로 검증 수행 중...")
            if tiles_far:
                far_metrics = run_eval_pipeline(tiles_far)
                grid_comparisons_far = run_grid_analysis(tiles_far, args.uncertainty)
                spatial_sensitivity = {
                    "excluded_sheets": list(near_sheets),
                    "n_images_filtered": len(tiles_far),
                    "all_pixel": {
                        "c1": {
                            "mIoU": far_metrics["all_pixel"]["c1"]["mIoU"],
                            "ECE": far_metrics["all_pixel"]["c1"]["ECE"],
                            "AUROC": far_metrics["all_pixel"]["c1"]["AUROC"],
                            "AUPRC": far_metrics["all_pixel"]["c1"]["AUPRC"],
                            "AUSE": far_metrics["all_pixel"]["c1"]["AUSE"]
                        }
                    },
                    "fg_only": {
                        "c1": {
                            "mIoU": far_metrics["fg_only"]["c1"]["mIoU"],
                            "ECE": far_metrics["fg_only"]["c1"]["ECE"]
                        }
                    },
                    "grid_comparisons": grid_comparisons_far
                }
                print(f"[공간 민감도 분석 결과 (Far Set - 5km 제외)]")
                print(f"  - All-pixel (C1) mIoU  : {far_metrics['all_pixel']['c1']['mIoU']:.4f} (전체셋: {final_metrics['all_pixel']['c1']['mIoU']:.4f})")
                print(f"  - All-pixel (C1) ECE   : {far_metrics['all_pixel']['c1']['ECE']:.4f} (전체셋: {final_metrics['all_pixel']['c1']['ECE']:.4f})")
                print(f"  - All-pixel (C1) AUROC : {far_metrics['all_pixel']['c1']['AUROC']:.4f} (전체셋: {final_metrics['all_pixel']['c1']['AUROC']:.4f})")
            else:
                print("[공간 민감도 분석] 제외 후 남은 타일이 없어 분석을 건너뜁니다.")
        else:
            print("\n[공간 민감도 분석] 5km 이내에 인접한 시험 도엽이 없어 분석이 불필요합니다.")

    bootstrap_summary = {}
    for mode in ["all_pixel", "fg_only"]:
        bootstrap_summary[mode] = {}
        for cond in ["c1", "c3"]:
            bootstrap_summary[mode][cond] = {}
            for metric in ["mIoU", "ECE", "AUROC", "AUPRC", "AUSE"]:
                m_vals = [r[mode][cond][metric] for r in bootstrap_results if r]
                mean_v, low_v, high_v = get_ci_bounds(m_vals)
                bootstrap_summary[mode][cond][metric] = {
                    "mean": mean_v, "ci_low": low_v, "ci_high": high_v
                }
                
    # 7. 타일 집계 및 운영 효율성 지표 고도화 (그리드 크기: 256/512/1024/도엽 비교)
    print("[HITL] 타일/그리드 집계 비교 분석 중...")
    grid_comparisons = run_grid_analysis(loaded_tiles, args.uncertainty)

    # 8. 권역별 성능 및 불확실도 지표 분리 저장 (GG, GS, JL)
    regional_metrics = {}
    for region in ["GG", "GS", "JL"]:
        r_tiles = [t for t in loaded_tiles if t["region"] == region]
        if len(r_tiles) > 0:
            regional_metrics[region] = run_eval_pipeline(r_tiles)

    # 9. 임계값 sweep CSV 저장 (sensitivity_metrics.csv)
    sweep_csv_path = os.path.join(args.results_dir, "sensitivity_metrics.csv")
    with open(sweep_csv_path, "w", encoding="utf-8") as f:
        f.write("Threshold,Coverage,Selective_Accuracy,Condition3_mIoU,Condition3_ECE\n")
        gt_all = np.concatenate([t["gt_sampled"] for t in loaded_tiles])
        probs_all = np.concatenate([t["probs_mean_sampled"] for t in loaded_tiles])
        
        if gt_all.size > 500000:
            sel = np.random.default_rng(42).choice(gt_all.size, 500000, replace=False)
            gt_all = gt_all[sel]
            probs_all = probs_all[sel]
            
        for th in np.linspace(0.3, 0.8, 6):
            max_p = np.max(probs_all, axis=1)
            pred_raw = np.argmax(probs_all, axis=1)
            
            accepted = max_p >= th
            cov = float(accepted.sum()) / gt_all.size
            sel_acc = float((pred_raw[accepted] == gt_all[accepted]).mean()) if accepted.any() else 1.0
            
            pred_c3 = np.where(accepted, pred_raw, 0)
            hist_c3 = fast_hist_np(gt_all, pred_c3, num_classes)
            miou_c3, _ = miou_from_hist(hist_c3, include_zero=True)
            
            conf_c3 = np.where(accepted, max_p, probs_all[:, 0])
            correct_c3 = (pred_c3 == gt_all)
            ece_c3, _, _, _, _ = expected_calibration_error(conf_c3, correct_c3)
            
            f.write(f"{th:.1f},{cov:.4f},{sel_acc:.4f},{miou_c3:.4f},{ece_c3:.4f}\n")

    # 10. 시각화 그래프 자동 생성 및 저장

    # (A) Risk-Coverage 및 Accuracy-Coverage 곡선 저장
    covs, risks, accs = [], [], []
    for th in np.linspace(0.0, 0.99, 100):
        max_p = np.max(probs_all, axis=1)
        pred_raw = np.argmax(probs_all, axis=1)
        accepted = max_p >= th
        cov = float(accepted.sum()) / gt_all.size
        acc = float((pred_raw[accepted] == gt_all[accepted]).mean()) if accepted.any() else 1.0
        covs.append(cov)
        risks.append(1.0 - acc)
        accs.append(acc)
        
    plt.figure(figsize=(5, 4))
    plt.plot(covs, risks, color="#E91E63", lw=2, label="Selective Risk (Error)")
    plt.xlabel("Coverage"); plt.ylabel("Selective Risk")
    plt.title("Risk-Coverage Curve")
    plt.grid(True, ls=":")
    plt.savefig(os.path.join(args.results_dir, "risk_coverage_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(covs, accs, color="#4CAF50", lw=2, label="Selective Accuracy")
    plt.xlabel("Coverage"); plt.ylabel("Selective Accuracy")
    plt.title("Accuracy-Coverage Curve")
    plt.grid(True, ls=":")
    plt.savefig(os.path.join(args.results_dir, "accuracy_coverage_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # (B) Residual Map Error Curve 저장 (512 그리드 기준)
    plt.figure(figsize=(5, 4))
    errs_grid = []
    npix_grid = []
    uncs_grid = []
    for t in loaded_tiles:
        gt = t["gt"]
        pred = t["pred_ensemble"]
        err = (pred != gt) & (gt != 0)
        unc_bald = t[args.uncertainty]
        H, W = gt.shape
        for y in range(0, H, 512):
            for x in range(0, W, 512):
                vm = (gt[y:y+512, x:x+512] != 0)
                npix = int(vm.sum())
                if npix == 0:
                    continue
                errs_grid.append(err[y:y+512, x:x+512].sum())
                npix_grid.append(npix)
                uncs_grid.append(unc_bald[y:y+512, x:x+512][vm].mean())
                
    errs_grid = np.array(errs_grid)
    npix_grid = np.array(npix_grid)
    uncs_grid = np.array(uncs_grid)
    
    order = np.argsort(uncs_grid)[::-1]
    cum_errors_fixed = np.cumsum(errs_grid[order])
    total_errors = float(errs_grid.sum())
    
    residual_ratio = 1.0 - (cum_errors_fixed / total_errors)
    audit_area_ratio = np.cumsum(npix_grid[order]) / npix_grid.sum()
    
    audit_area_ratio = np.concatenate([[0.0], audit_area_ratio])
    residual_ratio = np.concatenate([[1.0], residual_ratio])
    
    plt.plot(audit_area_ratio * 100, residual_ratio * 100, color="#FF9800", lw=2, label="Proposed Audit")
    plt.xlabel("Audited Area (%)")
    plt.ylabel("Residual Map Error (%)")
    plt.title("Residual Map Error Curve")
    plt.grid(True, ls=":")
    plt.savefig(os.path.join(args.results_dir, "residual_error_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # (C) Reliability Diagram (scaled) 및 표본수 마킹
    plt.figure(figsize=(4.5, 4.5))
    gt_fg_all = np.concatenate(reliability_gt_list, axis=0) if reliability_gt_list else np.array([])
    logits_det_fg_all = np.concatenate(reliability_logits_list, axis=0) if reliability_logits_list else np.array([])
    
    if len(gt_fg_all) > 500000:
        sel = np.random.default_rng(42).choice(len(gt_fg_all), 500000, replace=False)
        logits_det_fg = logits_det_fg_all[sel]
        gt_fg = gt_fg_all[sel]
    else:
        logits_det_fg = logits_det_fg_all
        gt_fg = gt_fg_all
        
    logits_det_scaled = logits_det_fg / T
    logits_max = np.max(logits_det_scaled, axis=1, keepdims=True)
    probs_det_scaled = np.exp(logits_det_scaled - logits_max) / np.sum(np.exp(logits_det_scaled - logits_max), axis=1, keepdims=True)
    
    pred_det_scaled = np.argmax(probs_det_scaled, axis=1)
    conf_det_scaled = np.max(probs_det_scaled, axis=1)
    correct_det_scaled = (pred_det_scaled == gt_fg)
    
    ece_det, rc_cent, rc_acc, rc_conf, rc_cnt = expected_calibration_error(conf_det_scaled, correct_det_scaled)
    
    plt.plot([0, 1], [0, 1], color="#6B7280", ls="--", lw=1.5, label="Perfect calibration")
    plt.bar(rc_cent, rc_acc, width=1.0 / len(rc_cent) * 0.9,
            color="#1D9E75", alpha=0.85, edgecolor="#13654b", label="Accuracy")
    plt.plot(rc_conf, rc_acc, "o-", color="#D85A30", ms=4, lw=1.2, label="Acc vs Conf")
    
    for cent, acc, cnt in zip(rc_cent, rc_acc, rc_cnt):
        if cnt > 0:
            plt.text(cent, acc + 0.02, f"{cnt}", ha="center", va="bottom", fontsize=6, rotation=45)
            
    plt.xlim(0, 1); plt.ylim(0, 1.1)
    plt.xlabel("Confidence (Scaled)"); plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram (T={T:.2f}, ECE={ece_det:.4f})")
    plt.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "reliability_diagram_scaled.png"), dpi=150)
    # (D) Confusion Matrix (전체 테스트셋 기준)
    try:
        gt_cm = np.concatenate([t["gt_sampled"] for t in loaded_tiles])
        probs_cm = np.concatenate([t["probs_mean_sampled"] for t in loaded_tiles])
        pred_cm = np.argmax(probs_cm, axis=1)
        
        hist_cm = fast_hist_np(gt_cm, pred_cm, num_classes)
        
        plt.figure(figsize=(6, 5))
        hist_norm = hist_cm.astype('float') / (hist_cm.sum(axis=1, keepdims=True) + 1e-12)
        plt.imshow(hist_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Normalized Confusion Matrix (Recall)", fontsize=12, fontweight='bold')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=8)
        plt.yticks(tick_marks, class_names, fontsize=8)
        
        for i in range(num_classes):
            for j in range(num_classes):
                v = hist_norm[i, j]
                if hist_cm[i, j] > 0:
                    plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if v > 0.5 else "#222222")
        plt.tight_layout()
        cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[Vis] Confusion Matrix saved to: {cm_path}")
    except Exception as e:
        print(f"[Vis] [경고] Confusion Matrix 생성 실패: {e}")

    # (E) 정성적 분석 패널(Qualitative Panels) 자동 생성
    try:
        generate_qualitative_panels(args.results_dir, label_dir, args.mapping_file, num_classes, dn_to_id, prefix=prefix)
    except Exception as e:
        print(f"[Vis] [경고] 정성적 분석 패널 생성 실패: {e}")

    # 11. 최종 metrics_uncertainty.json 작성 및 저장
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    summary_json = {
        "n_images": len(loaded_tiles),
        "temperature_parameter_T": T,
        "spatial_independence": report,
        "spatial_sensitivity_analysis_5km": spatial_sensitivity,
        "nearest_neighbor_sheet_distance": {
            "test_to_nearest_train_km": report["test_to_nearest_train_km"] if report else None,
            "test_to_nearest_test_km": report["test_to_nearest_test_km"] if report else None,
            "coordinate_source": "geotiff" if crs else "nominal_sheet_grid",
        },
        # Backward compatibility top-level keys
        "overall_accuracy_OA": final_metrics["all_pixel"]["c1"]["OA"],
        "kappa": final_metrics["all_pixel"]["c1"]["Kappa"],
        "macro_precision": final_metrics["all_pixel"]["c1"]["macro_precision"],
        "macro_recall": final_metrics["all_pixel"]["c1"]["macro_recall"],
        "macro_f1": final_metrics["all_pixel"]["c1"]["macro_f1"],
        "per_class_iou_ensemble": {
            class_names[c]: (None if np.isnan(final_metrics["all_pixel"]["c1"]["class_iou"].get(c, np.nan)) else float(final_metrics["all_pixel"]["c1"]["class_iou"][c]))
            for c in range(num_classes)
        },
        "per_class_precision_ensemble": {
            class_names[c]: (None if np.isnan(final_metrics["all_pixel"]["c1"]["class_precision"].get(c, np.nan)) else float(final_metrics["all_pixel"]["c1"]["class_precision"][c]))
            for c in range(num_classes)
        },
        "per_class_recall_ensemble": {
            class_names[c]: (None if np.isnan(final_metrics["all_pixel"]["c1"]["class_recall"].get(c, np.nan)) else float(final_metrics["all_pixel"]["c1"]["class_recall"][c]))
            for c in range(num_classes)
        },
        "per_class_f1_ensemble": {
            class_names[c]: (None if np.isnan(final_metrics["all_pixel"]["c1"]["class_f1"].get(c, np.nan)) else float(final_metrics["all_pixel"]["c1"]["class_f1"][c]))
            for c in range(num_classes)
        },
        "miou_single_seed": final_metrics["all_pixel"]["c1"]["mIoU"],
        "miou_ensemble": final_metrics["all_pixel"]["c1"]["mIoU"],
        "error_detection_AUROC": final_metrics["all_pixel"]["c1"]["AUROC"],
        "sparsification_AUSE": final_metrics["all_pixel"]["c1"]["AUSE"],
        "expected_calibration_error_ECE": final_metrics["all_pixel"]["c1"]["ECE"],
        "all_pixel_primary": {
            "c1": {
                "mIoU": final_metrics["all_pixel"]["c1"]["mIoU"],
                "ECE": final_metrics["all_pixel"]["c1"]["ECE"],
                "AUROC": final_metrics["all_pixel"]["c1"]["AUROC"],
                "AUPRC": final_metrics["all_pixel"]["c1"]["AUPRC"],
                "AUSE": final_metrics["all_pixel"]["c1"]["AUSE"],
                "error_base_rate": final_metrics["all_pixel"]["c1"]["error_base_rate"],
                "OA": final_metrics["all_pixel"]["c1"]["OA"],
                "Kappa": final_metrics["all_pixel"]["c1"]["Kappa"],
                "macro_precision": final_metrics["all_pixel"]["c1"]["macro_precision"],
                "macro_recall": final_metrics["all_pixel"]["c1"]["macro_recall"],
                "macro_f1": final_metrics["all_pixel"]["c1"]["macro_f1"],
                "bootstrap_95ci": bootstrap_summary["all_pixel"]["c1"]
            },
            "c2": {
                "coverage": final_metrics["all_pixel"]["c2"]["coverage"],
                "selective_accuracy": final_metrics["all_pixel"]["c2"]["selective_accuracy"],
                "mIoU": final_metrics["all_pixel"]["c2"]["mIoU"]
            },
            "c3": {
                "mIoU": final_metrics["all_pixel"]["c3"]["mIoU"],
                "ECE": final_metrics["all_pixel"]["c3"]["ECE"],
                "AUROC": final_metrics["all_pixel"]["c3"]["AUROC"],
                "AUPRC": final_metrics["all_pixel"]["c3"]["AUPRC"],
                "AUSE": final_metrics["all_pixel"]["c3"]["AUSE"],
                "omission_error": final_metrics["all_pixel"]["c3"]["omission_error"],
                "commission_error": final_metrics["all_pixel"]["c3"]["commission_error"],
                "bootstrap_95ci": bootstrap_summary["all_pixel"]["c3"]
            },
            "deterministic_scaled": final_metrics["all_pixel"]["det_scaled"],
            "mc_dropout_vs_deterministic": final_metrics["all_pixel"]["mc_dropout_vs_deterministic"]
        },
        "fg_only_supplemental": {
            "c1": {
                "mIoU": final_metrics["fg_only"]["c1"]["mIoU"],
                "ECE": final_metrics["fg_only"]["c1"]["ECE"],
                "AUROC": final_metrics["fg_only"]["c1"]["AUROC"],
                "AUPRC": final_metrics["fg_only"]["c1"]["AUPRC"],
                "AUSE": final_metrics["fg_only"]["c1"]["AUSE"],
                "error_base_rate": final_metrics["fg_only"]["c1"]["error_base_rate"],
                "OA": final_metrics["fg_only"]["c1"]["OA"],
                "Kappa": final_metrics["fg_only"]["c1"]["Kappa"],
                "macro_precision": final_metrics["fg_only"]["c1"]["macro_precision"],
                "macro_recall": final_metrics["fg_only"]["c1"]["macro_recall"],
                "macro_f1": final_metrics["fg_only"]["c1"]["macro_f1"],
                "bootstrap_95ci": bootstrap_summary["fg_only"]["c1"]
            },
            "c2": {
                "coverage": final_metrics["fg_only"]["c2"]["coverage"],
                "selective_accuracy": final_metrics["fg_only"]["c2"]["selective_accuracy"],
                "mIoU": final_metrics["fg_only"]["c2"]["mIoU"]
            },
            "c3": {
                "mIoU": final_metrics["fg_only"]["c3"]["mIoU"],
                "ECE": final_metrics["fg_only"]["c3"]["ECE"],
                "AUROC": final_metrics["fg_only"]["c3"]["AUROC"],
                "AUPRC": final_metrics["fg_only"]["c3"]["AUPRC"],
                "AUSE": final_metrics["fg_only"]["c3"]["AUSE"],
                "bootstrap_95ci": bootstrap_summary["fg_only"]["c3"]
            },
            "deterministic_scaled": final_metrics["fg_only"]["det_scaled"],
            "mc_dropout_vs_deterministic": final_metrics["fg_only"]["mc_dropout_vs_deterministic"]
        },
        "grid_comparisons": grid_comparisons,
        "regional_metrics": regional_metrics
    }

    out_json_path = os.path.join(args.results_dir, prefix + "metrics_uncertainty.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print("\n===== JSTARS 불확실성 통합 검증 요약 =====")
    print(f"  - 최적 보정 온도 T : {T:.4f}")
    print(f"  - 결정론적 ECE (Raw)   : {final_metrics['all_pixel']['det_scaled']['ECE_raw']:.4f}")
    print(f"  - 결정론적 ECE (Scaled): {final_metrics['all_pixel']['det_scaled']['ECE_scaled']:.4f}")
    print(f"  - MC Dropout ECE (C1)  : {final_metrics['all_pixel']['c1']['ECE']:.4f}")
    print(f"  - All-pixel (C1) mIoU  : {final_metrics['all_pixel']['c1']['mIoU']:.4f}")
    print(f"  - All-pixel (C1) OA    : {final_metrics['all_pixel']['c1']['OA']:.4f}")
    print(f"  - All-pixel (C1) Kappa : {final_metrics['all_pixel']['c1']['Kappa']:.4f}")
    print(f"  - All-pixel (C1) macro-F1: {final_metrics['all_pixel']['c1']['macro_f1']:.4f}")
    print(f"  - All-pixel (C1) AUROC : {final_metrics['all_pixel']['c1']['AUROC']:.4f}")
    print(f"  - All-pixel (C1) AUPRC : {final_metrics['all_pixel']['c1']['AUPRC']:.4f}")
    print(f"  - FG-only   (C1) mIoU  : {final_metrics['fg_only']['c1']['mIoU']:.4f}")
    print(f"  - FG-only   (C1) ECE   : {final_metrics['fg_only']['c1']['ECE']:.4f}")
    print(f"\n성공적으로 결과가 저장되었습니다: {out_json_path}")

if __name__ == "__main__":
    main()
