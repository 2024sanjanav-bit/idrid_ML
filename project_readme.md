# Generalization of Deep Convolutional Architectures for Retinal Fundus Grading

An empirical study evaluating the generalization performance of **VGG-16** and **ResNet-18** on diabetic retinopathy grading using the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**. This project compares training from scratch against transfer learning under severe class imbalance.

---

## Overview

Medical imaging datasets often suffer from limited sample sizes and skewed class distributions, making model generalization challenging. This repository investigates how architectural depth, skip connections, and initialization strategies impact classification performance and overfitting on retinal fundus images.

### Core Experimental Dimensions
- **Architectures Evaluated:** VGG-16 vs. ResNet-18
- **Training Regimes:** Random Initialization (From-Scratch) vs. ImageNet Pretrained Weights (Transfer Learning)
- **Class Imbalance Mitigation:** Inverse-frequency weighted Cross-Entropy Loss
- **Regularization & Optimization:** Domain-appropriate data augmentation and **AdamW** optimization

---

## Dataset

- **Dataset:** Indian Diabetic Retinopathy Image Dataset (IDRiD)
- **Task:** Multi-class grading of diabetic retinopathy severity (Grades 0–4) from retinal fundus photographs
- **Preprocessing:** Image resizing, normalization, and training-time data augmentation (random flips, rotations, and color/contrast adjustments) to improve out-of-sample robustness

---

## Methodology

| Component | Configuration |
| :--- | :--- |
| **Models** | `VGG-16`, `ResNet-18` |
| **Initialization** | From-Scratch vs. Pretrained (`ImageNet`) |
| **Loss Function** | Inverse-Frequency Weighted Cross-Entropy Loss |
| **Optimizer** | `AdamW` (decoupled weight decay) |
| **Evaluation Metrics** | Accuracy, Macro F1-Score, Per-Class Precision/Recall, Quadratic Weighted Kappa, Confusion Matrix |

---

## Repository Structure

```text
ml-generalisation-idrid/
├── data/                   # Dataset directory and train/val/test split manifests
├── models/                 # VGG-16 and ResNet-18 model definitions and weight modifiers
├── src/
│   ├── dataset.py          # PyTorch Dataset, preprocessing, and augmentation pipelines
│   ├── train.py            # Training loop with weighted loss and AdamW optimizer
│   └── evaluate.py         # Validation/test evaluation, metrics, and confusion matrix plots
├── notebooks/              # Exploratory data analysis and visual result comparisons
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone and Install Dependencies

```bash
git clone <your-repository-url>
cd ml-generalisation-idrid

pip install -r requirements.txt
```

### 2. Train Models

```bash
# Example: Train ResNet-18 with Transfer Learning
python src/train.py --model resnet18 --pretrained --epochs 50 --batch_size 16 --lr 1e-4

# Example: Train VGG-16 from Scratch
python src/train.py --model vgg16 --epochs 50 --batch_size 16 --lr 1e-4
```

### 3. Evaluate on the Test Set

```bash
python src/evaluate.py --model resnet18 --weights checkpoints/resnet18_pretrained_best.pth
```