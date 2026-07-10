import os
import sys
import argparse
import csv
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Training Metrics")
    parser.add_argument("--csv", type=str, required=True, help="Path to metrics.csv")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: File not found: {args.csv}")
        sys.exit(1)

    epochs = []
    train_loss = []
    val_loss = []
    miou = []

    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row['Epoch']))
                train_loss.append(float(row['Train_Loss']))
                val_loss.append(float(row['Val_Loss']))
                miou.append(float(row['mIoU']))
            except (ValueError, KeyError):
                continue

    if not epochs:
        print("No valid data found in CSV.")
        sys.exit(1)

    # Use a modern, beautiful style
    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title(f"Training Metrics - {os.path.basename(os.path.dirname(args.csv))}")

    # Loss Plot
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:blue', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', color='tab:orange', linewidth=2)
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # mIoU Plot
    ax2.plot(epochs, miou, label='mIoU', color='tab:green', linewidth=2)
    ax2.set_title('Mean IoU (mIoU) over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Save the plot and open it (avoids backend issues)
    out_img = os.path.join(os.path.dirname(args.csv), "metrics_plot.png")
    plt.savefig(out_img, dpi=150)
    print(f"Saved plot to {out_img}")
    
    # Open image using Windows default viewer
    if os.name == 'nt':
        os.startfile(out_img)

if __name__ == "__main__":
    main()
