# Base Paper Benchmark & Baseline Metrics

> **Reference Paper**:  
> Haosen Liu, Jiahao Liu, Shan Tan, and Edmund Y. Lam,  
> *"Image Restoration Learning via Noisy Supervision in Fourier Domain"*,  
> **IEEE Transactions on Image Processing (IEEE TIP)**, Vol. 35, pp. 6432–6445, 2026.  
> DOI: [10.1109/TIP.2026.3700926](https://doi.org/10.1109/TIP.2026.3700926)

---

## 1. Metric Definitions & Evaluation Conventions (Paper §IV-A)

All experiments evaluate restoration performance using **Peak Signal-to-Noise Ratio (PSNR)** in dB and **Structural Similarity Index (SSIM)** with an $11 \times 11$ Gaussian sliding window ($\sigma = 1.5$):

* **Denoising (DN)**: Calculated across **all color channels** (converted to single-channel grayscale in synthetic benchmarks).
* **Super-Resolution (SR)**: Evaluated strictly on the **Y channel (luminance)** of the $\text{YCbCr}$ color space, with border pixels cropped corresponding to the scaling factor $s \in \{2, 3, 4\}$.
* **Deblurring (DB)**: Calculated across **all RGB channels**.
* **Supervision Settings**:
  * **N2C (Noise2Clean)**: Fully supervised upper bound trained with pristine clean targets $z$.
  * **IR-CSF**: Fourier-loss model trained with clean targets $z$.
  * **N2N (Noise2Noise)**: Baseline spatial weakly supervised model trained with paired noisy targets $y$.
  * **IR-NSF**: Proposed weakly supervised model trained with noisy targets $y$ using Fourier Huber/L1 loss.

---

## 2. Table I: Denoising (DN) — ResNet Backbone
* **Task**: Synthetic Denoising on 1-channel Grayscale.
* **Degradation**: Mixture of Gaussian ($\sigma = 5$), Poisson noise, and periodic column/stripe noise.
* **Backbone**: Modified EDSR / ResNet (16 residual blocks, 64 feature channels, global residual skip connection).

| Target | Method | Set5 (PSNR / SSIM) | Set14 (PSNR / SSIM) | Kodak24 (PSNR / SSIM) | McMaster (PSNR / SSIM) | Urban100 (PSNR / SSIM) | Manga109 (PSNR / SSIM) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean | **N2C (Supervised Bound)** | 35.07 / 0.9315 | 33.77 / 0.9147 | 34.45 / 0.9176 | 35.92 / 0.9398 | 34.39 / 0.9477 | 36.16 / 0.9508 |
| Clean | **IR-CSF** | 35.07 / 0.9317 | 33.77 / 0.9149 | 34.45 / 0.9176 | 35.92 / 0.9399 | 34.41 / 0.9478 | 36.16 / 0.9508 |
| Noisy | **N2N (Spatial Baseline)** | **34.94 / 0.9295** | **33.62 / 0.9125** | **34.32 / 0.9155** | **35.77 / 0.9380** | **34.04 / 0.9450** | **35.97 / 0.9492** |
| Noisy | **IR-NSF (Paper Proposed)** | **35.06 / 0.9312** | **33.75 / 0.9146** | **34.43 / 0.9175** | **35.90 / 0.9396** | **34.34 / 0.9474** | **36.14 / 0.9505** |

---

## 3. Table II: Super-Resolution (SR $\times 4$) — EDSR Backbone
* **Task**: $4\times$ Single-Image Super-Resolution (RGB input $\to$ Y-channel metric).
* **Degradation**: Bicubic downsampling $\times 4$ + target corrupted by Gaussian noise ($\sigma = 10$).
* **Backbone**: EDSR (16 residual blocks, 64 features, pixel shuffle upsampler).

| Target | Method | Set5 (PSNR / SSIM) | Set14 (PSNR / SSIM) | Kodak24 (PSNR / SSIM) | McMaster (PSNR / SSIM) | Urban100 (PSNR / SSIM) | Manga109 (PSNR / SSIM) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean | **N2C (Supervised Bound)** | 37.95 / 0.9602 | 33.54 / 0.9175 | 34.01 / 0.9229 | 38.32 / 0.9636 | 32.05 / 0.9277 | 38.37 / 0.9763 |
| Clean | **IR-CSF** | 38.07 / 0.9607 | 33.68 / 0.9181 | 34.08 / 0.9237 | 38.45 / 0.9642 | 32.20 / 0.9286 | 38.75 / 0.9772 |
| Noisy | **N2N (Spatial Baseline)** | **37.94 / 0.9603** | **33.54 / 0.9173** | **33.99 / 0.9227** | **38.29 / 0.9635** | **32.04 / 0.9277** | **38.27 / 0.9762** |
| Noisy | **IR-NSF (Paper Proposed)** | **38.04 / 0.9606** | **33.65 / 0.9183** | **34.06 / 0.9234** | **38.42 / 0.9642** | **32.20 / 0.9288** | **38.55 / 0.9769** |

---

## 4. Table III: Motion Deblurring (DB) — MIMO-UNet Backbone
* **Task**: Image Deblurring on RGB channels.
* **Degradation**: 8 Levin motion blur kernels + input Gaussian noise ($\sigma = 2$) + target Gaussian noise ($\sigma = 10$).
* **Backbone**: Multi-input Multi-output MIMO-UNet.

| Target | Method | Set5 (PSNR / SSIM) | Set14 (PSNR / SSIM) | Kodak24 (PSNR / SSIM) | McMaster (PSNR / SSIM) | Urban100 (PSNR / SSIM) | Manga109 (PSNR / SSIM) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean | **N2C (Supervised Bound)** | 32.75 / 0.9162 | 31.56 / 0.8954 | 33.09 / 0.9186 | 34.56 / 0.9325 | 30.75 / 0.9218 | 34.61 / 0.9431 |
| Clean | **IR-CSF** | 32.95 / 0.9179 | 32.05 / 0.9004 | 33.64 / 0.9224 | 34.74 / 0.9344 | 31.30 / 0.9274 | 34.98 / 0.9455 |
| Noisy | **N2N (Spatial Baseline)** | **32.79 / 0.9157** | **31.75 / 0.8967** | **33.12 / 0.9189** | **34.49 / 0.9322** | **30.84 / 0.9221** | **34.65 / 0.9430** |
| Noisy | **IR-NSF (Paper Proposed)** | **32.96 / 0.9179** | **32.05 / 0.9000** | **33.59 / 0.9223** | **34.73 / 0.9344** | **31.31 / 0.9274** | **34.88 / 0.9454** |

---

## 5. Table IV: Real-World Low-Light Denoising (SID Sony Dataset)
* **Dataset**: See-in-the-Dark (SID) Sony RAW ($4240 \times 2832$ resolution).
* **Degradation**: Real sensor read noise, thermal noise, and electronic column striping.

| Method | SID Validation (PSNR / SSIM) | SID Test Set (PSNR / SSIM) |
| :--- | :---: | :---: |
| **N2N (Baseline)** | 36.85 dB / 0.8466 | 40.10 dB / 0.9124 |
| **IR-NSF (Paper Proposed)** | **37.07 dB / 0.8476** | **40.32 dB / 0.9140** |

---

## 6. Table V: Real-World Motion Deblurring (ReLoBlur Dataset)
* **Dataset**: ReLoBlur synchronized camera pair ($2152 \times 1436$ resolution).
* **Metrics**: Global PSNR/SSIM and weighted $\text{PSNR}_w / \text{SSIM}_w$ calculated over locally blurry moving objects.

| Method | Global PSNR (dB) | Global SSIM | Weighted $\text{PSNR}_w$ (dB) | Weighted $\text{SSIM}_w$ |
| :--- | :---: | :---: | :---: | :---: |
| **N2N (Baseline)** | 34.19 dB | 0.9186 | 27.18 dB | 0.8505 |
| **IR-NSF (Paper Proposed)** | **34.43 dB** | **0.9250** | **27.39 dB** | **0.8630** |

---

## 7. Table VI: Unsupervised Stripe Removal (USR)
* **Task**: Destriping with/without Gaussian noise post-processed by BM3D.

| Noise Condition | Method | Set5 (PSNR / SSIM) | Kodak24 (PSNR / SSIM) | McMaster (PSNR / SSIM) |
| :--- | :--- | :---: | :---: | :---: |
| **Row Stripe Only** | VSNR [20] | 38.52 / **0.9915** | 36.16 / 0.9845 | 38.08 / 0.9841 |
| | SEID [44] | 35.90 / 0.9474 | 35.69 / 0.9818 | 35.05 / 0.9565 |
| | Self-BSR [58] | 36.04 / 0.9774 | 35.41 / 0.9858 | 36.27 / 0.9737 |
| | **USR (Paper Proposed)** | **39.99** / 0.9905 | **38.51 / 0.9953** | **39.84 / 0.9888** |
| **Row Stripe + Gaussian** | VSNR [20] | 34.40 / 0.9375 | 33.30 / 0.9481 | 34.81 / 0.9418 |
| | SEID [44] | 34.81 / 0.9134 | 34.92 / 0.9533 | 36.22 / 0.9499 |
| | Self-BSR [58] | 34.75 / 0.9424 | 33.58 / 0.9459 | 34.41 / 0.9408 |
| | **USR (Paper Proposed)** | **35.89 / 0.9470** | **35.08 / 0.9533** | **36.23 / 0.9524** |

---

## 8. Theoretical Breakdown Under Impulse Noise (Our Extension)

The base paper assumes that noise is either spatially independent (Gaussian/Poisson) or sparse in the Fourier domain (stripes). Under **20% salt-and-pepper impulse noise**, all three foundational theorems break down:

| Foundational Theorem | Theoretical Metric | Base Paper Value Under Impulse Noise | Ground Truth Target Value | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Theorem 1 (Gaussianity)** | **Excess Kurtosis ($\gamma_2$)** | **921.50** (Extreme heavy tails) | **0.00** (Gaussian distribution) | **Violated** |
| **Theorem 1 (Independence)** | **Normality $p$-value** | $p < 10^{-300}$ (Normality rejected) | $p > 0.05$ | **Violated** |
| **Theorem 2 (Frequency Sparsity)** | **High-Frequency Floor** | **$-12.2\text{ dB}$** (Broadband haze) | Steep decay ($\sim -40\text{ dB}$, natural $1/f^\alpha$) | **Violated** |
| **Theorem 2 (Sparsity)** | **Top 1% Fourier Energy** | **$8.9\%$** (Smeared across 100% of bins) | Concentrated in subset ($\Omega$) | **Violated** |
| **Theorem 3 (Signal Independence)** | **Signal Covariance $\operatorname{Cov}(z, n)$** | **$-0.0031$ ($r \approx -0.1$)** | **$0.00$** (Zero covariance) | **Violated** |
| **Theorem 3 (Zero Bias)** | **Optimal Solution $\hat{z}^*$** | $\hat{z}^* \neq z$ (Systematic bias & blur) | $\hat{z}^* = z$ (Unbiased global minimum) | **Violated** |
