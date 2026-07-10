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
                
                miou = d.get("miou_ensemble", d.get("miou_single_seed", 0)) * 100
                
                # Check for specific metrics, default to BALD if available, else first available
                comp = d.get("comparison_metrics", {})
                auroc, ause, hitl = 0, 0, 0
                if comp:
                    best_m = "BALD" if "BALD" in comp else list(comp.keys())[0]
                    auroc = comp[best_m].get("AUROC", 0)
                    ause = comp[best_m].get("AUSE", 0)
                    hitl = comp[best_m].get("HITL_20_caught", 0)
                else:
                    auroc = d.get("error_detection_AUROC", 0)
                    ause = d.get("sparsification_AUSE", 0)
                    h_info = d.get("hitl_errors_caught_by_budget", {})
                    hitl = h_info.get("20%_area", 0) * 100
                    
                oa = d.get("overall_accuracy_OA", 0) * 100
                kappa = d.get("kappa", 0)

                data.append({
                    "name": model_name,
                    "mIoU": miou,
                    "OA": oa,
                    "Kappa": kappa,
                    "AUROC": auroc,
                    "AUSE": ause,
                    "HITL": hitl
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not data: return

    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    names = [d["name"] for d in data]
    y = np.arange(len(names))
    height = 0.5
    
    # Premium flat colors for academic publication
    color_miou = "#1ABC9C"
    color_auroc = "#E67E22"
    color_hitl = "#3498DB"

    # mIoU (Horizontal)
    mious = [d["mIoU"] for d in data]
    axes[0].barh(y, mious, height, color=color_miou, edgecolor='none')
    axes[0].set_title("Segmentation Performance (mIoU %)", fontweight='bold', fontsize=12)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[0].invert_yaxis()  # Top-down order matching list
    for i, v in enumerate(mious):
        axes[0].text(v + 1.0, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold', fontsize=10)
    axes[0].set_xlim(0, max(mious) * 1.18)

    # AUROC (Horizontal)
    aurocs = [d["AUROC"] for d in data]
    axes[1].barh(y, aurocs, height, color=color_auroc, edgecolor='none')
    axes[1].set_title("Uncertainty AUROC (Higher is Better)", fontweight='bold', fontsize=12)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()
    for i, v in enumerate(aurocs):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va='center', ha='left', fontweight='bold', fontsize=10)
    axes[1].set_xlim(0, max(aurocs) * 1.18)

    # HITL (Horizontal)
    hitls = [d["HITL"] for d in data]
    axes[2].barh(y, hitls, height, color=color_hitl, edgecolor='none')
    axes[2].set_title("HITL 20% Cost Efficiency (%)", fontweight='bold', fontsize=12)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names, fontsize=11, fontweight='bold')
    axes[2].invert_yaxis()
    for i, v in enumerate(hitls):
        axes[2].text(v + 1.0, i, f"{v:.1f}%", va='center', ha='left', fontweight='bold', fontsize=10)
    axes[2].set_xlim(0, max(hitls) * 1.18)

    # Style improvements (Spines removal for modern look)
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.grid(False)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "model_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
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
        tr_html += f"<td>{d['AUROC']:.4f}</td><td>{d['AUSE']:.4f}</td><td style='{w_hitl}'>{d['HITL']:.1f}%</td></tr>"

    html = f"""
    <h2 style='color: #2E86C1;'>🏆 Multi-Model Comparison Report</h2>
    <p>Compared <b>{len(data)}</b> models based on Segmentation Performance and Uncertainty Estimation.</p>
    
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center;">
        <tr style="background-color: #f2f2f2;">
            <th>Model Name</th><th>mIoU (↑)</th><th>OA (↑)</th><th>Kappa (↑)</th><th>AUROC (↑)</th><th>AUSE (↓)</th><th>HITL @20% (↑)</th>
        </tr>
        {tr_html}
    </table>
    
    <div style='margin-top:20px; padding:10px; background-color:#F8F9F9; border-left: 4px solid #3498DB;'>
        <b>💡 Insights:</b><br>
        - <b>Highest Performance (mIoU):</b> <span style='color:green;'>{winner_miou}</span><br>
        - <b>Best Uncertainty Estimation (HITL):</b> <span style='color:#D35400;'>{winner_hitl}</span>
    </div>
    """
    out_html = os.path.join(args.out_dir, "comparison_report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
        
    # Save as CSV for Excel
    import csv
    out_csv = os.path.join(args.out_dir, "comparison_report.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Name", "mIoU (%)", "OA (%)", "Kappa", "AUROC", "AUSE", "HITL @20% (%)"])
        for d in data:
            writer.writerow([d['name'], f"{d['mIoU']:.2f}", f"{d['OA']:.2f}", f"{d['Kappa']:.4f}", f"{d['AUROC']:.4f}", f"{d['AUSE']:.4f}", f"{d['HITL']:.1f}"])
        
    print(f"Comparison graph saved to: {out_png}")
    print(f"Comparison report saved to: {out_html}")
    print(f"Comparison CSV saved to: {out_csv}")

if __name__ == "__main__":
    main()
