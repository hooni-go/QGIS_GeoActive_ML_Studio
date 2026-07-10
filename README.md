# QGIS_GeoActive ML Studio Plugin

This QGIS plugin enables the training, inference, and uncertainty evaluation of state-of-the-art Deep Learning models directly within QGIS. It is specifically designed for Remote Sensing imagery (supporting both 8-bit RGB and 16-bit 4-band TIFF) and integrates multiple architectures with advanced uncertainty quantification for quality control.

## 🌟 Supported Architectures
1. **Mask2Former (Swin-Large Backbone)**: Mask-classification based advanced architecture.
2. **SegFormer (MiT-B2 Backbone)**: Light and powerful Transformer-based architecture.
3. **DeepLabV3+ (ResNet-50 Backbone)**: ASPP-based multi-scale context aggregation model.
4. **U-Net (ResNet-50 Backbone)**: Classic encoder-decoder baseline with skip connections.

## 🛠️ Environment Requirements

To ensure the plugin runs smoothly, please configure a Python environment with the following dependencies.

### 1. CUDA & PyTorch Version
* **NVIDIA GPU:** Minimum 8GB+ VRAM recommended (VRAM requirements are optimized via sliding mini-batch inference).
* **CUDA Toolkit:** CUDA 11.8 or CUDA 12.1+ recommended.
* **PyTorch:** PyTorch **2.0.0 or higher** is required (for `torch.amp.autocast` compatibility).

### 2. Key Python Dependencies
All dependencies are listed in `requirements.txt`.
* `transformers >= 4.30.0` (For Mask2Former and SegFormer HuggingFace models)
* `segmentation-models-pytorch` (For U-Net and DeepLabV3+ models)
* `rasterio` (For 16-bit GeoTIFF imagery loading)
* `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`

## 🚀 Installation & Setup in QGIS

1. **Plugin Installation:** Copy this entire `QGIS_GeoActive_ML_Studio` folder into your QGIS plugins directory.
   * Windows: `C:\Users\%USERNAME%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\QGIS_GeoActive_ML_Studio`
2. **Python Environment Binding:** 
   * In the QGIS Plugin UI, select **"Custom Portable Python"** under **"1. Python Environment"**.
   * Point the path to the `python.exe` of the virtual environment where dependencies are installed.
3. **Offline Models (Optional):**
   * Put offline model weights (e.g., HuggingFace pretrained folders) into the `offline_setup/models/` directory for offline environments.

## 🧠 Key Features

* **Auto-Calculate Optimal Class Weights:** Scans dataset masks to compute Inverse Class Frequency weights to mitigate extreme class imbalance.
* **Auxiliary Focal Loss:** Custom Focal Loss integrated into model training to focus on hard-to-predict boundary pixels.
* **Seed-based MC Dropout Uncertainty:** Quantifies epistemic uncertainty at pixel level via stochastic forward passes (MC Dropout, N=20).
* **Uncertainty Reliability Verification:** Computes calibration error (ECE), sparsification error (AUSE), and error detection rates (AUROC).
* **Human-in-the-Loop (HITL) Quality Control:** Identifies high-uncertainty areas (top 20%) to optimize manual inspection budgets and maximize error correction efficiency.
* **Interactive Dashboard:** Beautiful HTML-rendered reports, confusion matrices, and reliability charts directly loaded in the QGIS interface.
