# Robust Image Restoration Learning via Noisy Supervision in the Fourier Domain (IR-NSF)

**Course**: Computer Vision  
**Project**: Executable Implementation & Final Technical Report  
**Base Reference Paper**:  
Haosen Liu, Jiahao Liu, Shan Tan, and Edmund Y. Lam,  
*"Image Restoration Learning via Noisy Supervision in the Fourier Domain"*,  
**IEEE Transactions on Image Processing (IEEE TIP)**, Vol. 35, pp. 6432-6445, 2026.  
DOI: [10.1109/TIP.2026.3700926](https://doi.org/10.1109/TIP.2026.3700926)  

---

## 1. Quick Start: How to Run Inference

This submission contains fully pre-trained checkpoints, evaluation scripts, and sample images ready for immediate out-of-the-box execution.

### 1.1 Environment Setup

```bash
# 1. Clone / Navigate to submission directory
cd cv_project_submission

# 2. Create and activate a Python virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### 1.2 Running Single-Image Inference (`inference.py`)

The primary executable script is `inference.py`. It automatically detects pre-trained weights in `checkpoints/`, performs restoration, reports PSNR/SSIM, and saves restored images and side-by-side comparison plots to `outputs/`:

```bash
# 1. Quick test on sample bird image (20% impulse noise):
python inference.py --input sample_images/bird.png --noise_type impulse --noise_ratio 0.20

# 2. Test on sample butterfly image (20% impulse noise):
python inference.py --input sample_images/butterfly.png --noise_type impulse --noise_ratio 0.20

# 3. Test on Kodak #3 under mixed noise (Gaussian + Poisson + Stripe + Impulse):
python inference.py --input sample_images/kodim03.png --noise_type mixed

# 4. Denoise an existing noisy image without injecting synthetic noise:
python inference.py --input path/to/your_noisy_image.png --noise_type none

# 5. Batch inference over an entire folder of images:
python inference.py --input sample_images/ --output_dir outputs/

# 6. Run on CPU (if no CUDA GPU is available):
python inference.py --input sample_images/bird.png --gpu -1
```

**Outputs generated in `outputs/`**:
* `<image_name>_restored.png`: High-fidelity restored output image.
* `<image_name>_noisy.png`: Corrupted input image (if synthetic degradation was evaluated).
* `<image_name>_comparison.png`: 3-panel side-by-side visual comparison (`[Clean Reference | Corrupted Input | Proposed Restoration]`).

### 1.3 Reproducing Full Benchmark Evaluations and Plots

```bash
# Generate publication 4-panel visual comparisons and zoomed detail crops:
python run_visual_eval.py

# Evaluate on combined mixed degradations:
python run_mixed_noise_eval.py

# Evaluate the improved model across full Kodak24 dataset (20% impulse noise):
python evaluate.py --task dn --checkpoint checkpoints/checkpoints_improved/best_model.pt --datasets Kodak24 --noise_type impulse --impulse_ratio 0.20 --n_feats 32 --n_resblocks 9 --gpu 0

# Run the chained multi-stage compound restoration pipeline (DN -> DB -> SR):
python pipeline_restore.py --scale 2 --out_dir results/composite_pipeline

# Reproduce frequency spectral analysis & mathematical theorem proofs:
python impulse_noise_spectral_analysis.py --image sample_images/kodim03.png --noise_ratio 0.20
```

---

## 2. Executive Summary & Key Highlights

This project replicates, mathematically analyzes, and significantly improves the unsupervised Fourier-domain noisy supervision framework (IR-NSF) for image denoising. While theoretically sound for Gaussian and stripe noise, our investigations proved that the paper's three foundational theorems break down under non-Gaussian degradations (e.g. 20% impulse noise and mixed sensor noise).

### Key Technical Achievements:
* **Restoration Quality**: Up to **+4.77 dB PSNR** and **+0.1431 SSIM** improvement over the baseline on standard benchmarks.
* **85.3% Parameter Reduction**: Streamlined from **1,219,841 parameters** (14.7 MB) down to **179,908 parameters** (2.26 MB), speeding up inference by **4.7x**.
* **Elimination of Dead-Gradient Collapses**: Tail weight initialization scaling ($0.001$) and conditional clamping permanently solved the "all-black output" collapse.
* **Theoretical Invalidation & Recovery**: Suppressed the broadband Dirac delta spectral noise floor by **~20 dB** (from $-12.2\text{ dB}$ down to $-31.9\text{ dB}$), restoring the paper's zero-bias and sparsity conditions.

---

## 3. Theoretical Problem Formulation: Breakdown of the 3 Base Theorems

The base IR-NSF paper relies on three mathematical pillars that collapse when images are corrupted by impulse noise:

| Theorem | Base Paper Assumption | Breakdown Under Impulse Noise | Empirical Proof |
| :--- | :--- | :--- | :--- |
| **Theorem 1: Asymptotic Gaussianity** | 2D-DFT noise coefficients converge to independent zero-mean complex Gaussians $\mathcal{N}(0, \sigma^2/2UV)$. | Spatial impulses act as Dirac delta functions $\delta[u - u_0, v - v_0]$, projecting constant-magnitude complex exponentials across all frequencies and coupling all frequency bins. | Empirical Excess Kurtosis reached **921.50** (Gaussian is 0.00), strictly rejecting normality ($p < 10^{-300}$). |
| **Theorem 2: Spectral Sparsity** | Structured noise energy is concentrated in a tiny subset $\Omega \ll UV$, preserving uncorrupted high-SNR frequency bins. | Spatial delta spikes have flat Fourier spectra, dispersing noise energy across 100% of frequency bins. | Generates a flat broadband noise floor at **-12.2 dB**, completely smearing high-frequency textures. |
| **Theorem 3: Zero-Bias Minimum** | Expected Fourier loss achieves its unique global minimum at clean signal $z$ under signal-independent noise ($\operatorname{Cov}(z, n) = 0$). | Salt-and-pepper noise replaces clean pixels with 0 or 1, making residual noise $n = y - z$ inherently signal-dependent. | Significant non-zero correlation: $\operatorname{Cov}(z, n) = \mathbf{-0.0031}$ ($p < 10^{-300}$), biasing network convergence toward distorted states. |

---

## 4. Final Proposed Architecture and Loss Improvements

The final winning design comprises six core innovations:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 INPUT: Noisy Image y                    │
                  └────────────────────────────┬────────────────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
    [Inference / Forward Pass]                                      [Supervisory Loss Calculation]
               │                                                               │
  Conv 3x3 (1 -> 32 channels)                                         Fast 3x3 Median Filter
               │                                                               │
     8x Residual Blocks                                                Spatial Impulse Mask M
  [Conv -> GELU -> Conv -> SE -> Skip]                                         │
               │                                            ┌──────────────────┴──────────────────┐
  Block 8: Subsampled Non-Local Attention                   ▼                                     ▼
  [Global Context Spatial Pooling]                 Inpainted Target (y_clean)           Spatial Valid Mask (1 - M)
               │                                            │                                     │
     1x Residual Block                               2D Discrete Fourier                 Spatial Huber Loss
  [Conv -> GELU -> Conv -> SE -> Skip]                      │                                     │
               │                                    Fourier Huber Loss                            │
  Conv 3x3 (Tail init scaled by 0.001)                      │                                     │
               │                                            └──────────────────┬──────────────────┘
       Identity Skip (+ x)                                                     ▼
               │                                                    Combined Loss Gradients
               ▼
   Restored Denoised Output
```

1. **Hybrid Spatial-Impulse Detection & Masked Fourier Loss** (`loss.py`):
   Detects impulse spikes via rank deviation $|y - \text{median}(y)| > \tau$, inpaints supervisory targets prior to 2D-DFT, and applies dual-domain loss. Corrupted pixels are masked out from spatial backpropagation, suppressing the spectral noise floor by ~20 dB.
2. **Squeeze-and-Excitation (SE) Channel Attention** (`model.py`):
   Integrated into every residual block ($r=16$ with GELU bottleneck), enabling adaptive feature recalibration to suppress noise-corrupted channels.
3. **Smooth GELU Activation Gating** (`model.py`):
   Replaces rigid ReLU activations, providing continuous non-zero gradient flow for negative values and preventing neuron death under large impulse gradients.
4. **Midpoint Subsampled Non-Local Spatial Attention** (`model.py`):
   Positioned at block 8 of the 9-block backbone. Captures global contextual relationships across distant pixels with constant memory footprint (<400 MB).
5. **Anti-Collapse Invariants (Identity Tail Initialization & Conditional Clamping)** (`model.py`):
   Tail convolution weights are initialized scaled by $0.001$ with zeroed biases, ensuring $f_\theta(x) \approx x$ at initialization. Output clamping to $[0, 1]$ is restricted strictly to evaluation mode (`if not self.training`), preventing dead-gradient traps.
6. **Lightweight Model Topology** (`model.py`):
   Optimized to 9 residual blocks and 32 channels (179,908 parameters) versus 16 blocks and 64 channels (1,219,841 parameters). Constrains network capacity to learn true image priors rather than memorizing high-frequency supervisory noise.

---

## 5. Explored Alternatives and Ablation Analysis (Features Tried But Not Superior)

During research and development, several alternative architectures, loss formulations, and pipelines were implemented and evaluated. The following options were tested and rejected in favor of the final proposed configuration:

### 5.1 Deep and Overparameterized ResNet Backbone (16 Blocks, 64 Channels)
* **Hypothesis**: Retaining the base paper's 16-block, 64-channel ResNet topology (1.22M parameters) would maximize restoration capacity.
* **Finding**: In noisy-supervised training without clean ground truth, excessive network capacity causes the model to overfit to and memorize high-frequency noise spikes present in the supervisory targets. Furthermore, deep unguided residual paths suffered from vanishing gradients, frequently causing the output to collapse into an all-black image ($0.0$).
* **Conclusion**: Reducing channel width to 32 and depth to 9 blocks, augmented with channel and spatial attention, delivered +3.21 dB higher mean PSNR and completely eliminated training collapse.

### 5.2 Cascaded Multi-Stage Restoration Pipeline (Denoising -> Deblurring -> Super-Resolution)
* **Hypothesis**: Chaining three distinct networks sequentially (ResNet for Denoising, MIMO-UNet for Deblurring, and EDSR for Super-Resolution) would handle severe compound degradations.
* **Finding**:
  1. Compounding artifact propagation: In a sequential multi-stage pipeline, residual reconstruction errors and slight edge smudges from Stage 1 (Denoising) are treated as real structural edges by Stage 2 (MIMO-UNet), causing ringing artifacts during deconvolution.
  2. Diminishing returns: Stage 1 delivered the overwhelming majority of restoration gain (+7.40 dB to +11.05 dB PSNR). Downstream deblurring contributed minimal additional improvement (+0.16 dB on Bird, +0.06 dB on Kodak03) while tripling compute latency and memory consumption.
* **Conclusion**: The dedicated single-stage improved denoiser is superior in efficiency, stability, and edge sharpness. While the multi-stage pipeline code is preserved (`pipeline_restore.py`) for compound degradation experiments, the standalone improved denoiser remains the primary recommended solution.

### 5.3 Standard ReLU and LeakyReLU Activation Functions
* **Hypothesis**: Standard ReLU non-linearities with He/Kaiming initialization are standard in residual restoration architectures.
* **Finding**: Under high-amplitude impulse noise (where pixel deviations jump between $0.0$ and $1.0$), large negative gradient backpropagation pushes activations into the inactive flat region of the ReLU curve ($x < 0$). This resulted in the "dying ReLU" trap, where large subsets of feature channels permanently ceased updating. LeakyReLU alleviated dying neurons but introduced gradient variance that hindered loss convergence.
* **Conclusion**: GELU non-linearities provided the optimal curvature, maintaining smooth gradient flow through negative regimes and yielding faster, more stable convergence.

### 5.4 Global Spatial Median Filtering Alone
* **Hypothesis**: Applying a spatial median filter directly to the corrupted image prior to evaluation could serve as a baseline restoration method without neural network training.
* **Finding**: While uniform spatial median filtering removes impulse spikes, it indiscriminately destroys delicate high-frequency textures, edge boundaries, and fine alphanumeric text (e.g. embroidered text on caps in Kodak #3 was completely erased).
* **Conclusion**: Spatial median filtering is best utilized strictly as an internal detection and frequency-inpainting mask within the supervisory loss function (`RobustImpulseIRNSFLoss`), enabling the deep neural network to reconstruct sharp, high-frequency details.

### 5.5 Unmasked Frequency-Domain Loss Formulations
* **Hypothesis**: Relying solely on the paper's original Fourier Huber loss without spatial impulse masking.
* **Finding**: Because spatial delta spikes inject a flat broadband spectrum across all frequencies, unmasked Fourier loss forces the network to minimize error against a flat -12.2 dB noise floor. This resulted in heavy edge ghosting, severe blur, and an objective loss plateau at 0.03188.
* **Conclusion**: Spatial masking of impulse outliers prior to DFT computation is mathematically required to restore the paper's zero-bias and sparsity conditions.

---

## 6. Quantitative Benchmark Results

Evaluated across standard benchmarks with identical seeds on NVIDIA GeForce RTX 4050 GPU:

### 6.1 Denoising Under 20% Impulse Noise ($p = 0.20$)

| Benchmark Dataset | Image Count | Corrupted Input | Baseline IR-NSF (Paper) | Proposed Improved Model | Performance Gain ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Set5** | 5 | 12.01 dB / 0.2144 | 20.88 dB / 0.6090 | **25.65 dB / 0.7521** | **+4.77 dB / +0.1431** |
| **Set14** | 14 | 12.09 dB / 0.1783 | 20.82 dB / 0.5574 | **24.18 dB / 0.6638** | **+3.36 dB / +0.1064** |
| **Kodak24** | 24 | 12.22 dB / 0.1651 | 22.11 dB / 0.5735 | **25.07 dB / 0.6540** | **+2.96 dB / +0.0805** |
| **DIV2K (Crops)** | 3 | 11.98 dB / 0.1592 | 21.02 dB / 0.5023 | **22.78 dB / 0.5199** | **+1.76 dB / +0.0176** |
| **Overall Mean** | **46** | **12.08 dB / 0.1793** | **21.21 dB / 0.5606** | **24.42 dB / 0.6475** | **+3.21 dB / +0.0869** |

### 6.2 Per-Image Quantitative Performance

| Test Image | Resolution | Corrupted Input | Baseline IR-NSF | Proposed Improved Model | Absolute Gain ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Set5: Bird** | $288 \times 288$ | 11.90 dB / 0.1130 | 22.88 dB / 0.6456 | **26.99 dB / 0.7891** | **+4.12 dB / +0.1435** |
| **Set5: Butterfly** | $256 \times 256$ | 12.08 dB / 0.2659 | 18.13 dB / 0.6225 | **21.14 dB / 0.7395** | **+3.00 dB / +0.1170** |
| **Kodak: #3 (kodim03)** | $768 \times 512$ | 12.46 dB / 0.0688 | 24.73 dB / 0.6500 | **27.67 dB / 0.6863** | **+2.94 dB / +0.0363** |
| **Kodak: #5 (kodim05)** | $768 \times 512$ | 12.00 dB / 0.1975 | 20.13 dB / 0.5269 | **22.28 dB / 0.6479** | **+2.16 dB / +0.1210** |
| **Kodak: #1 (kodim01)** | $768 \times 512$ | 12.53 dB / 0.1912 | 21.12 dB / 0.4657 | **22.57 dB / 0.5577** | **+1.45 dB / +0.0921** |

---

## 7. Directory Structure

```
cv_project_submission/
├── README.md                          # Comprehensive documentation & execution guide
├── requirements.txt                   # Complete dependencies (torch, torchvision, scipy, etc.)
├── .gitignore                         # Git ignore configuration
├── inference.py                       # Executable CLI for single/batch image inference
├── evaluate.py                        # Benchmark suite for Kodak24, Set5, Set14, Urban100
├── model.py                           # Network architectures (Improved ResNet, EDSR, MIMO-UNet)
├── loss.py                            # Supervisory losses (RobustImpulseIRNSFLoss, IRNSFLoss)
├── dataset.py                         # Synthetic degradation models (impulse, Gaussian, Poisson, stripe)
├── run_visual_eval.py                 # Generates 4-panel comparisons and zoomed detail crops
├── run_mixed_noise_eval.py            # Generates visual comparisons under mixed degradations
├── pipeline_restore.py                # Chained multi-stage compound restoration pipeline
├── impulse_noise_spectral_analysis.py # Theoretical proofs and frequency spectrum analysis
├── train.py                           # Training engine for the improved denoising model
├── train_db.py                        # Training engine for deblurring
├── train_sr.py                        # Training engine for super-resolution
├── train_usr.py                       # Training engine for stripe removal
├── checkpoints/                       # Pre-trained weights ready for immediate evaluation
│   ├── checkpoints_improved/
│   │   ├── best_model.pt              # Proposed 30-epoch improved model (2.26 MB)
│   │   └── training_log.csv           # Loss optimization history
│   ├── checkpoints_baseline/
│   │   └── best_model.pt              # Baseline reference checkpoint (14.7 MB)
│   ├── checkpoints_db/
│   │   └── db_best_model.pt           # Deblurring MIMO-UNet checkpoint (7.4 MB)
│   └── checkpoints_sr/
│       └── sr_best_model.pt           # Super-resolution EDSR checkpoint (2.45 MB)
├── sample_images/                     # Benchmark inputs for out-of-the-box testing
│   ├── bird.png                       # Set5 Toucan
│   ├── butterfly.png                  # Set5 Butterfly
│   ├── kodim01.png                    # Kodak #1 Cottage
│   ├── kodim03.png                    # Kodak #3 Landscape
│   └── kodim05.png                    # Kodak #5 Motorcycle
├── results/                           # Curated visual evaluations and metric summaries
│   ├── impulse_noise/                 # 4-panel comparison plots and zoom crops (20% impulse)
│   ├── mixed_noise/                   # 4-panel comparison plots under mixed noise
│   ├── composite_pipeline/            # 5-panel progressive restoration outputs
│   ├── spectral_analysis/             # Frequency smearing plots and theorem violation proofs
│   └── div2k_comparison/              # High-resolution benchmark crops
└── docs/                              # In-depth architectural & theoretical documentation
    ├── IR_NSF_ARCHITECTURE_UPDATED.drawio.xml
    ├── IMPROVEMENTS_GUIDE.md
    ├── MODEL_COMPARISON_REPORT.md
    ├── COMPARISON_METRICS.md
    └── BASELINE_METRICS.md
```

---

## 8. Hardware Environment and Reproducibility

* **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
* **CUDA Version**: 12.4
* **PyTorch Version**: 2.6.0+cu124
* **Operating System**: Windows 11 / Linux compatible
* **Random Seed**: Fixed to `42` across all degradation and evaluation scripts for strict reproducibility.

---

## 9. References

1. Haosen Liu, Jiahao Liu, Shan Tan, and Edmund Y. Lam, *"Image Restoration Learning via Noisy Supervision in the Fourier Domain"*, IEEE Transactions on Image Processing (IEEE TIP), Vol. 35, pp. 6432-6445, 2026.
2. Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, *"Enhanced Deep Residual Networks for Single Image Super-Resolution"*, CVPRW 2017 (EDSR).
3. Sung-Jea Cho, Seo-Won Ji, Jun-Pyo Hong, Seung-Won Jung, and Sung-Jea Ko, *"Rethinking Coarse-to-Fine Approach in Single Image Deblurring"*, ICCV 2021 (MIMO-UNet).
4. Jie Hu, Li Shen, and Gang Sun, *"Squeeze-and-Excitation Networks"*, CVPR 2018.
5. Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He, *"Non-local Neural Networks"*, CVPR 2018.
6. Dan Hendrycks and Kevin Gimpel, *"Gaussian Error Linear Units (GELUs)"*, arXiv:1606.08415, 2016.
