# Anti-Graffiti Mask2Former QGIS Plugin

This QGIS plugin enables the training and inference of a state-of-the-art **Mask2Former (Swin-Large Backbone)** model directly within QGIS. It is specifically designed for Remote Sensing imagery (supports both 8-bit RGB and 16-bit 4-band TIFF) and includes advanced features such as Focal Loss for class imbalance, automated class weight calculation, and uncertainty evaluation.

## 🛠️ Environment Requirements (Crucial)

To ensure the plugin runs smoothly without CUDA memory errors or PyTorch autocast crashes, please configure your standalone Python environment with the following specific versions.

### 1. CUDA & PyTorch Version
This plugin uses Automatic Mixed Precision (AMP) and requires a modern PyTorch build with CUDA support.
* **NVIDIA GPU:** Minimum 16GB VRAM recommended for Swin-Large.
* **CUDA Toolkit:** CUDA 11.8 or CUDA 12.1+ recommended.
* **PyTorch:** PyTorch **2.0.0 or higher** is strictly required (for `torch.amp.autocast` compatibility).

**Installation Command Example (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Python Packages (`requirements.txt`)
All dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
* `transformers >= 4.30.0` (Required for Mask2Former HuggingFace integration)
* `rasterio` (Required for 16-bit GeoTIFF data loading)
* `Pillow` (Image processing)
* `numpy`, `pandas`, `matplotlib`, `scipy`

## 🚀 Installation & Setup in QGIS

1. **Plugin Installation:** Copy this entire `AntiGraffitiMask2Former` folder into your QGIS plugins directory.
   * Windows: `C:\Users\%USERNAME%\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\AntiGraffitiMask2Former`
2. **Python Environment Binding:** 
   * In the QGIS Plugin UI, look for **"1. Python Environment"**.
   * Select `Custom Portable Python`.
   * Point the path to the `python.exe` of the virtual environment where you installed the PyTorch and requirements mentioned above.
3. **Offline Model (Optional):**
   * If you are in an offline network, place the downloaded HuggingFace `mask2former-swin-large` model files into the `offline_setup/models/mask2former-swin-large` directory.

## 🧠 Key Features
* **Auto-Calculate Optimal Weights:** Automatically scans your dataset masks and calculates Inverse Class Frequency to assign optimal loss weights for imbalanced classes (e.g., Rare Greenhouses vs Common Forests).
* **Auxiliary Focal Loss:** Custom Focal Loss map integrated into the HuggingFace Mask2Former outputs to force the model to focus on hard-to-predict pixels.
* **Transfer Learning:** Select previous `.pt` checkpoints to resume training. The plugin automatically syncs past hyperparameters and class weights.
* **Uncertainty Evaluation:** Multi-modal uncertainty evaluation (Entropy, Variance, Drop-Out) and CSV generation for academic papers.
