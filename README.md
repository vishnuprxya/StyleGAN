# 🎨 StyleGAN-style Generator on CIFAR-10 (TensorFlow / Keras)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**Author:** Eshan M (E0324009) | Batch-1 AIDA | Deep Learning
**Project:** TensorFlow/Keras implementation of the core ideas of **StyleGAN (Karras et al., 2019)** trained on CIFAR-10 (32 × 32 RGB, 20,000-image subset).

---

## 📌 Architectural Overview

```
Latent Z (128) → 8-Layer Mapping MLP → Intermediate Style Vector W (128)

Learned Constant (4x4x128) → StyledConv + AdaIN(W) ×2 (4x4)
                           → Upsample → StyledConv + AdaIN(W) ×2 (8x8)
                           → Upsample → StyledConv + AdaIN(W) ×2 (16x16)
                           → Upsample → StyledConv + AdaIN(W) ×2 (32x32) → 1x1 Conv + Tanh → RGB Image
```

## 🔑 Key Innovations Implemented

1. **Mapping Network (f: Z → W):** 8-layer MLP (LeakyReLU 0.2, He-normal init) with input pixel-norm maps `z ∈ R^128` into the intermediate style space `w ∈ R^128`.
2. **Adaptive Instance Normalization (AdaIN):** a learned affine layer turns `w` into per-channel scale and bias, applied to the normalized feature map: `AdaIN(x, w) = norm(x) · (1 + scale(w)) + bias(w)`.
3. **Learned Constant Input:** synthesis starts from a learned `1×4×4×128` tensor instead of a sampled noise vector.
4. **Stochastic Noise Injection:** per-pixel Gaussian noise multiplied by a learned per-layer strength is added after each convolution.
5. **Style Mixing:** two style vectors (`w_A`, `w_B`) are used at different layers (cutoffs 2, 4, 6) to show which layers control coarse vs fine appearance.
6. **Non-saturating logistic GAN loss** with Adam (lr 2e-4, β1 = 0, β2 = 0.99).
7. **Latent Interpolation:** linear interpolation between two latent vectors (9 steps) shows smooth transitions.

## 📁 Repository Structure

```
Style-GAN/
├── stylegan_cifar10.ipynb   # Complete notebook: data, model, training, results
├── outputs/                 # Figures saved by the notebook
│   ├── real_samples.png
│   ├── training_losses.png
│   ├── generated_samples.png
│   ├── style_mixing.png
│   └── latent_interpolation.png
├── requirements.txt         # Dependencies
├── .gitignore
└── README.md                # Project documentation
```

## 📊 Output Visualizations

### 1. Generated samples after training
![Generated samples](outputs/generated_samples.png)

### 2. Style Mixing
Source A supplies early (coarse) styles, Source B supplies later (fine) styles.

![Style mixing](outputs/style_mixing.png)

### 3. Latent Interpolation (z₁ → z₂, 9 steps)
![Latent interpolation](outputs/latent_interpolation.png)

### 4. Training losses
![Training losses](outputs/training_losses.png)

## 📈 Training Loss Log (10 Epochs)

> Copy these values from your notebook's training output.

| Epoch | Discriminator Loss (L_D) | Generator Loss (L_G) |
|-------|--------------------------|----------------------|
| 1/10  | _fill_                   | _fill_               |
| 2/10  | _fill_                   | _fill_               |
| ...   | ...                      | ...                  |
| 10/10 | _fill_                   | _fill_               |

## ⚡ Quick Start & Usage

```bash
git clone https://github.com/m-eshan/Style-GAN.git
cd Style-GAN
pip install -r requirements.txt
jupyter notebook stylegan_cifar10.ipynb
```

Run all cells. Figures are written to `outputs/` and model weights to `checkpoints/` (git-ignored).

## 📜 Citation & Acknowledgments

- Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks.* IEEE/CVF CVPR.
- TensorFlow / Keras teams for the CIFAR-10 dataset loader and neural network primitives.
