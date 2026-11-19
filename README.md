# Robust Image Restoration Learning via Noisy Supervision in the Fourier Domain (IR-NSF)

**Course**: Computer Vision  
**Project**: Executable Code Submission  
**Base Reference Paper**:  
Haosen Liu, Jiahao Liu, Shan Tan, and Edmund Y. Lam,  
*"Image Restoration Learning via Noisy Supervision in the Fourier Domain"*,  
**IEEE Transactions on Image Processing (IEEE TIP)**, Vol. 35, pp. 6432-6445, 2026.  
DOI: [10.1109/TIP.2026.3700926](https://doi.org/10.1109/TIP.2026.3700926)  

---

## 1. Project Overview

This repository provides an executable PyTorch implementation of an extended Fourier-domain noisy supervision framework (IR-NSF) for unsupervised image restoration. While the base framework performs restoration under Gaussian and periodic stripe noise, this codebase introduces robust modifications tailored to handle severe impulse (salt-and-pepper) and mixed sensor degradations without requiring clean ground-truth images for supervision.

### Key Implementation Features
* **Dual-Domain Loss Formulation**: Rank-deviation spatial impulse detection coupled with supervisory target inpainting prior to 2D-DFT, preventing high-frequency noise smearing during Fourier supervision.
* **Feature Attention Mechanisms**: Residual blocks equipped with Squeeze-and-Excitation (SE) channel attention and midpoint subsampled non-local spatial attention for global context.
* **Stable Training Dynamics**: Smooth GELU activations, identity tail convolution initialization ($0.001$ scale), and conditional evaluation-only clamping to eliminate dead gradients and training collapses.
* **End-to-End Evaluation Tools**: Modular scripts for single-image inference, benchmark dataset evaluation, visual comparison rendering, and multi-stage compound restoration.

---


### Module Descriptions
1. **Hybrid Spatial-Impulse Detection & Masked Fourier Loss** (`loss.py`):
   Detects impulse outliers via local rank deviation $|y - \text{median}(y)| > \tau$, inpaints supervisory targets prior to the 2D-DFT, and applies a combined spatial-frequency loss with masked outlier regions.
2. **Squeeze-and-Excitation (SE) Channel Attention** (`model.py`):
   Integrated into residual blocks ($r=16$ reduction ratio with GELU bottleneck) to adaptively recalibrate channel-wise feature responses.
3. **Smooth GELU Activations** (`model.py`):
   Maintains continuous gradient flow across negative regimes to prevent dead neurons under high-magnitude noise gradients.
4. **Midpoint Subsampled Non-Local Spatial Attention** (`model.py`):
   Captures long-range spatial correlations across distant regions with bounded computational overhead.
5. **Anti-Collapse Invariants** (`model.py`):
   Tail convolution weights are scaled down by $0.001$ with zeroed biases for an identity-preserving initial pass, while value clamping is restricted strictly to evaluation mode.

---

## 3. Environment Setup

### 3.1 Clone & Environment Creation

```bash
# Navigate to project directory
cd cv_project_submission

# Create a virtual environment
python -m venv .venv

# Activate environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Quick Start: Running Inference

The primary inference script is `inference.py`. It automatically loads the pre-trained weights from `checkpoints/`, restores the degraded image, and saves results to `outputs/`:

```bash
# 1. Quick test on sample bird image (20% impulse noise):
python inference.py --input sample_images/bird.png --noise_type impulse --noise_ratio 0.20

# 2. Test on sample butterfly image (20% impulse noise):
python inference.py --input sample_images/butterfly.png --noise_type impulse --noise_ratio 0.20

# 3. Test on Kodak #3 under mixed noise (Gaussian + Poisson + Stripe + Impulse):
python inference.py --input sample_images/kodim03.png --noise_type mixed

# 4. Restore an existing noisy image without injecting synthetic degradation:
python inference.py --input path/to/your_noisy_image.png --noise_type none

# 5. Batch inference over an entire directory:
python inference.py --input sample_images/ --output_dir outputs/

# 6. Run on CPU (if CUDA GPU is unavailable):
python inference.py --input sample_images/bird.png --gpu -1
```

**Outputs generated in `outputs/`**:
* `<image_name>_restored.png`: Restored output image.
* `<image_name>_noisy.png`: Corrupted input image (when synthetic noise is applied).
* `<image_name>_comparison.png`: Side-by-side comparison (`[Clean Reference | Corrupted Input | Restored Output]`).

---

## 5. Evaluation & Analysis Scripts

```bash
# Generate visual comparisons with zoomed crops:
python run_visual_eval.py

# Evaluate restoration on mixed degradations:
python run_mixed_noise_eval.py

# Run benchmark evaluation across Kodak24 dataset (20% impulse noise):
python evaluate.py --task dn --checkpoint checkpoints/checkpoints_improved/best_model.pt --datasets Kodak24 --noise_type impulse --impulse_ratio 0.20 --n_feats 32 --n_resblocks 9 --gpu 0

# Execute the multi-stage compound restoration pipeline (Denoising -> Deblurring -> Super-Resolution):
python pipeline_restore.py --scale 2 --out_dir results/composite_pipeline

# Run frequency spectral analysis:
python impulse_noise_spectral_analysis.py --image sample_images/kodim03.png --noise_ratio 0.20
```

---

## 6. Training

To train the improved model from scratch using noisy supervision:

```bash
python train.py \
    --train_dir data/DIV2K/patches_480 \
    --val_dir data/DIV2K/val_patches_480 \
    --noise_type impulse \
    --impulse_ratio 0.20 \
    --loss_type robust_impulse \
    --n_feats 32 \
    --n_resblocks 9 \
    --epochs 30 \
    --lr 1e-4 \
    --out_dir checkpoints/checkpoints_custom
```

Specialized training scripts are also available for downstream tasks:
* `train_db.py`: MIMO-UNet deblurring training loop.
* `train_sr.py`: EDSR super-resolution training loop.
* `train_usr.py`: Periodic stripe removal training loop.

---

## 7. Pre-trained Checkpoints

Pre-trained weights are located in `checkpoints/`:
* `checkpoints/checkpoints_improved/best_model.pt`: Proposed robust denoising model (32 features, 9 residual blocks).
* `checkpoints/checkpoints_baseline/best_model.pt`: Baseline IR-NSF denoising model (64 features, 16 residual blocks).
* `checkpoints/checkpoints_db/db_best_model.pt`: Deblurring MIMO-UNet checkpoint.
* `checkpoints/checkpoints_sr/sr_best_model.pt`: Super-resolution EDSR checkpoint.

---

## 8. System Requirements

* **Python**: 3.9+
* **PyTorch**: >= 2.0.0
* **CUDA** (optional, recommended): 11.8+ or 12.x
* **Operating System**: Windows 10/11 or Linux

---

## 9. References

1. Haosen Liu, Jiahao Liu, Shan Tan, and Edmund Y. Lam, *"Image Restoration Learning via Noisy Supervision in the Fourier Domain"*, IEEE Transactions on Image Processing (IEEE TIP), Vol. 35, pp. 6432-6445, 2026.
2. Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, *"Enhanced Deep Residual Networks for Single Image Super-Resolution"*, CVPRW 2017 (EDSR).
3. Sung-Jea Cho, Seo-Won Ji, Jun-Pyo Hong, Seung-Won Jung, and Sung-Jea Ko, *"Rethinking Coarse-to-Fine Approach in Single Image Deblurring"*, ICCV 2021 (MIMO-UNet).
4. Jie Hu, Li Shen, and Gang Sun, *"Squeeze-and-Excitation Networks"*, CVPR 2018.
5. Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He, *"Non-local Neural Networks"*, CVPR 2018.
6. Dan Hendrycks and Kevin Gimpel, *"Gaussian Error Linear Units (GELUs)"*, arXiv:1606.08415, 2016.
