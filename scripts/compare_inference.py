import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

def clean_model_name(name):
    name_upper = name.upper()
    if "MASK2FORMER" in name_upper:
        return "Mask2Former"
    elif "SEGFORMER" in name_upper:
        return "SegFormer"
    elif "DEEPLABV3PLUS" in name_upper:
        return "DeepLabV3+"
    elif "UNET" in name_upper:
        return "U-Net"
    return name.split("/")[-1].split("\\")[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_files", type=str, required=True)
    parser.add_argument("--names", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    files = [f.strip() for f in args.json_files.split(",") if f.strip()]
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(files) < 2:
        print("Need at least 2 files to compare.")
        return

    data = []
    for idx, f in enumerate(files):
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                d = json.load(jf)
                raw_name = names[idx] if idx < len(names) else "Model"
                model_name = clean_model_name(raw_name)
                
                # Try JSTARS new format first
                if "all_pixel_primary" in d:
                    c1 = d["all_pixel_primary"].get("c1", {})
                    miou = c1.get("mIoU", 0.0) * 100
                    oa = c1.get("OA", 0.0) * 100
                    kappa = c1.get("Kappa", 0.0)
                    auroc = c1.get("AUROC", 0.0)
                    ece = c1.get("ECE", 0.0)
                    ause = c1.get("AUSE", 0.0)
                    grid = d.get("grid_comparisons", {})
                    ne = grid.get("grid_512", {}).get("mean", {}).get("normalized_efficiency", 0.0) * 100
                    hitl = grid.get("grid_512", {}).get("mean", {}).get("summ_20", 0.0) * 100
                else:
                    # Fallback to old format
                    miou = d.get("miou_ensemble", d.get("miou_single_seed", 0.0)) * 100
                    oa = d.get("overall_accuracy_OA", 0.0) * 100
                    kappa = d.get("kappa", 0.0)
                    ece = d.get("ece", 0.0)
                    comp = d.get("comparison_metrics", {})
                    if comp:
                        best_m = "BALD" if "BALD" in comp else list(comp.keys())[0]
                        auroc = comp[best_m].get("AUROC", 0.0)
                        ause = comp[best_m].get("AUSE", 0.0)
                        ece = comp[best_m].get("ECE", ece)
                        hitl = comp[best_m].get("HITL_20_caught", 0.0)
                        ne = comp[best_m].get("Normalized_Audit_Efficiency", 0.0) * 100
                    else:
                        auroc = d.get("error_detection_AUROC", 0.0)
                        ause = d.get("sparsification_AUSE", 0.0)
                        h_info = d.get("hitl_errors_caught_by_budget", {})
                        hitl = h_info.get("20%_area", 0.0) * 100
                        ne = d.get("normalized_audit_efficiency", 0.0) * 100

                data.append({
                    "name": model_name,
                    "mIoU": miou,
                    "OA": oa,
                    "Kappa": kappa,
                    "AUROC": auroc,
                    "AUSE": ause,
                    "ECE": ece,
                    "HITL": hitl,
                    "NE": ne
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not data: return

    # Plot 1: Basic (model_comparison_basic.png & model_comparison.png)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    names = [d["name"] for d in data]
    y = np.arange(len(names))
    height = 0.5
    
    # Premium flat colors for academic publication
    color_miou = "#1ABC9C"
    color_auroc = "#E67E22"
    color_hitl = "#3498DB"

    # 1. Segmentation Performance (mIoU %) (Horizontal) - Higher is Better
    mious = [d["mIoU"] for d in data]
    axes[0].barh(y, mious, height, color=color_miou, edgecolor='none')
    axes[0].set_title("Segmentation Performance (mIoU %)", fontweight='bold', fontsize=12)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[0].invert_yaxis()  # Top-down order matching list
    for i, v in enumerate(mious):
        axes[0].text(v + 1.0, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold', fontsize=10)
    axes[0].set_xlim(0, max(mious) * 1.18)

    # 2. Uncertainty AUROC (Higher is Better) (Horizontal) - Higher is Better
    aurocs = [d["AUROC"] for d in data]
    axes[1].barh(y, aurocs, height, color=color_auroc, edgecolor='none')
    axes[1].set_title("Uncertainty AUROC (Higher is Better)", fontweight='bold', fontsize=12)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()
    for i, v in enumerate(aurocs):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va='center', ha='left', fontweight='bold', fontsize=10)
    max_auroc = max(aurocs) if max(aurocs) > 0 else 1.0
    axes[1].set_xlim(0, max_auroc * 1.18)

    # 3. HITL 20% Cost Efficiency (%) (Horizontal) - Higher is Better
    hitls = [d["HITL"] for d in data]
    axes[2].barh(y, hitls, height, color=color_hitl, edgecolor='none')
    axes[2].set_title("HITL 20% Cost Efficiency (%)", fontweight='bold', fontsize=12)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[2].invert_yaxis()
    for i, v in enumerate(hitls):
        axes[2].text(v + 1.0, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold', fontsize=10)
    max_hitl = max(hitls) if max(hitls) > 0 else 100.0
    axes[2].set_xlim(0, max_hitl * 1.18)

    # Style improvements (Spines removal for modern look)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.grid(False)

    plt.tight_layout()
    out_png_basic = os.path.join(args.out_dir, "model_comparison_basic.png")
    out_png_default = os.path.join(args.out_dir, "model_comparison.png")
    plt.savefig(out_png_basic, dpi=150, bbox_inches='tight')
    plt.savefig(out_png_default, dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Scientific (model_comparison_scientific.png)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    color_ece = "#E74C3C"  # Coral Red
    color_ne = "#9B59B6"   # Purple

    # 1. Segmentation Performance (mIoU %)
    axes[0].barh(y, mious, height, color=color_miou, edgecolor='none')
    axes[0].set_title("Segmentation Performance (mIoU %)", fontweight='bold', fontsize=12)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[0].invert_yaxis()
    for i, v in enumerate(mious):
        axes[0].text(v + 1.0, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold', fontsize=10)
    axes[0].set_xlim(0, max(mious) * 1.18)

    # 2. Expected Calibration Error (ECE %) - Lower is Better
    eces = [d["ECE"] * 100 for d in data]  # Convert decimal to percent
    axes[1].barh(y, eces, height, color=color_ece, edgecolor='none')
    axes[1].set_title("Expected Calibration Error (ECE %)", fontweight='bold', fontsize=12)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()
    for i, v in enumerate(eces):
        axes[1].text(v + 0.1, i, f"{v:.2f}%", va='center', ha='left', fontweight='bold', fontsize=10)
    max_ece = max(eces) if max(eces) > 0 else 10.0
    axes[1].set_xlim(0, max_ece * 1.18)

    # 3. Normalized Audit Efficiency (NE %) - Higher is Better
    nes = [d["NE"] for d in data]
    axes[2].barh(y, nes, height, color=color_ne, edgecolor='none')
    axes[2].set_title("Normalized Audit Efficiency (NE %)", fontweight='bold', fontsize=12)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[2].invert_yaxis()
    for i, v in enumerate(nes):
        axes[2].text(v + 1.0, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold', fontsize=10)
    max_ne = max(nes) if max(nes) > 0 else 100.0
    axes[2].set_xlim(0, max_ne * 1.18)

    # Style improvements
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.grid(False)

    plt.tight_layout()
    out_png_sci = os.path.join(args.out_dir, "model_comparison_scientific.png")
    plt.savefig(out_png_sci, dpi=150, bbox_inches='tight')
    plt.close()
    
    # HTML Report
    winner_miou = max(data, key=lambda x: x["mIoU"])["name"]
    winner_hitl = max(data, key=lambda x: x["HITL"])["name"]
    
    tr_html = ""
    for d in data:
        w_miou = "background-color:#E8F8F5;font-weight:bold;" if d["name"] == winner_miou else ""
        w_hitl = "background-color:#FEF9E7;font-weight:bold;" if d["name"] == winner_hitl else ""
        
        tr_html += f"<tr><td>{d['name']}</td><td style='{w_miou}'>{d['mIoU']:.2f}%</td>"
        tr_html += f"<td>{d['OA']:.2f}%</td><td>{d['Kappa']:.4f}</td>"
        tr_html += f"<td>{d['AUROC']:.4f}</td><td>{d['AUSE']:.4f}</td>"
        tr_html += f"<td>{d['ECE']:.4f}</td><td style='{w_hitl}'>{d['HITL']:.1f}%</td></tr>"

    html = f"""
    <h2 style='color: #2E86C1; font-family: sans-serif;'>🏆 Multi-Model Comparison Report</h2>
    <p style='font-family: sans-serif;'>Compared <b>{len(data)}</b> models based on Segmentation Performance and Uncertainty Estimation.</p>
    
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-family: sans-serif;">
        <tr style="background-color: #f2f2f2;">
            <th>Model Name</th><th>mIoU (↑)</th><th>OA (↑)</th><th>Kappa (↑)</th><th>AUROC (↑)</th><th>AUSE (↓)</th><th>ECE (↓)</th><th>HITL @20% (↑)</th>
        </tr>
        {tr_html}
    </table>
    
    <div style='margin-top:20px; padding:10px; background-color:#F8F9F9; border-left: 4px solid #3498DB; font-family: sans-serif;'>
        <b>💡 Insights:</b><br>
        - <b>Highest Performance (mIoU):</b> <span style='color:green;'>{winner_miou}</span><br>
        - <b>Best Uncertainty Estimation (HITL):</b> <span style='color:#D35400;'>{winner_hitl}</span>
    </div>

    <h3 style='color: #34495E; margin-top:30px; font-family: sans-serif;'>📊 1. Basic Comparison Plot (mIoU / AUROC / HITL)</h3>
    <img src="model_comparison_basic.png" style="width: 100%; max-width: 900px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 20px;">

    <h3 style='color: #34495E; margin-top:10px; font-family: sans-serif;'>📊 2. Scientific Comparison Plot (mIoU / ECE / NE)</h3>
    <img src="model_comparison_scientific.png" style="width: 100%; max-width: 900px; border: 1px solid #ddd; border-radius: 4px;">
    """
    out_html = os.path.join(args.out_dir, "comparison_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
        
    # Save as CSV for Excel
    import csv
    out_csv = os.path.join(args.out_dir, "comparison_report.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Name", "mIoU (%)", "OA (%)", "Kappa", "AUROC", "AUSE", "ECE", "HITL @20% (%)"])
        for d in data:
            writer.writerow([d['name'], f"{d['mIoU']:.2f}", f"{d['OA']:.2f}", f"{d['Kappa']:.4f}", f"{d['AUROC']:.4f}", f"{d['AUSE']:.4f}", f"{d['ECE']:.4f}", f"{d['HITL']:.1f}"])
        
    print(f"Comparison graph saved to: {out_png_basic} & {out_png_sci}")
    print(f"Comparison report saved to: {out_html}")
    print(f"Comparison CSV saved to: {out_csv}")

if __name__ == "__main__":
    main()
