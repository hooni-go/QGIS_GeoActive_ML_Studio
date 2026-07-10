import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    parser.add_argument("--csv_files", type=str, required=True)
    parser.add_argument("--names", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    files = [f.strip() for f in args.csv_files.split(",") if f.strip()]
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if len(files) < 2:
        print("Need at least 2 files to compare.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors

    for idx, file in enumerate(files):
        try:
            df = pd.read_csv(file)
            df.columns = [c.lower() for c in df.columns]
            raw_name = names[idx] if idx < len(names) else os.path.basename(os.path.dirname(file))
            model_name = clean_model_name(raw_name)
            if 'epoch' not in df.columns: continue
            
            epochs = df['epoch']
            color = colors[idx % len(colors)]
            
            if 'val_loss' in df.columns:
                ax1.plot(epochs, df['val_loss'], label=model_name, color=color, linewidth=2, marker='o', markersize=4)
            if 'miou' in df.columns:
                ax2.plot(epochs, df['miou'], label=model_name, color=color, linewidth=2, marker='o', markersize=4)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    ax1.set_title("Validation Loss Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=10)

    ax2.set_title("Validation mIoU Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "training_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison graph saved to: {out_path}")

if __name__ == "__main__":
    main()
