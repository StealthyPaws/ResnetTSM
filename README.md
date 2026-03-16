# ResnetTSM Demo
# ResNet-TSM: Video Action Recognition with Temporal Shift Module

A PyTorch implementation of **Temporal Shift Module (TSM)** integrated into a ResNet-18 backbone for efficient video action recognition on the UCF-101 dataset.

---

## Overview

This project implements the [TSM paper](https://arxiv.org/abs/1811.08383) — a zero-parameter, zero-FLOPs technique that enables 2D CNNs to capture temporal information across video frames. By shifting a portion of channels along the temporal dimension, TSM allows a standard ResNet to reason about motion without the cost of 3D convolutions.

**Key results on UCF-101 (10-class subset):**

| Split | Accuracy |
|-------|----------|
| Train | 92.44%   |
| Val   | 91.40%   |
| Test  | 86.60%   |

---

## How TSM Works

The core idea is simple: before each residual block's first convolution, channels are **shifted in time**:

- **Forward shift** — 1/8 of channels are shifted from frame `t+1` → `t` (look ahead)
- **Backward shift** — 1/8 of channels are shifted from frame `t-1` → `t` (look back)
- **No shift** — remaining 6/8 channels are unchanged

This lets each frame "see" information from neighboring frames at zero additional cost.

```
Input [B, T, C, H, W]
        │
        ▼
  ┌─────────────┐
  │  TSM Layer  │  ← shift C/8 forward, C/8 backward
  └─────────────┘
        │
        ▼
  ResNet Conv Block
```

---

## Architecture

Two model variants are implemented:

### `ResNetTSM` (simple)
- ResNet-18 backbone with `fc` replaced by `Identity`
- A single `TSM` module applied on top-level features `[B, T, 512, 1, 1]`
- Temporal average pooling → linear classifier

### `TSMResNet` (full, used for training)
- TSM inserted **inside every residual block** (before `conv1`) across all 4 stages
- Processes input as `[B*T, C, H, W]` (standard 2D CNN throughput)
- Temporal average pooling over `T` frames → linear classifier

```python
model = TSMResNet(num_classes=10, n_segment=8, pretrained=True)
# Input:  [B, T, C, H, W] = [8, 8, 3, 112, 112]
# Output: [B, num_classes] = [8, 10]
```

---

## Dataset

**UCF-101** — a benchmark dataset of 101 human action categories from YouTube videos.

- Source: [Kaggle – UCF101 Action Recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition)
- This notebook uses a **10-class subset** for training
- Split: `train / val / test` directory structure

| Split | Videos |
|-------|--------|
| Train | 1,125  |
| Val   | 186    |
| Test  | 194    |

---

## Installation

```bash
git clone https://github.com/your-username/resnet-tsm.git
cd resnet-tsm

pip install torch torchvision opencv-python numpy matplotlib tqdm pandas
```

> Requires Python 3.10+ and a CUDA-capable GPU (recommended).

---

## Usage

### 1. Download the dataset

```bash
# Using Kaggle CLI
kaggle datasets download matthewjansen/ucf101-action-recognition
```

Update `DATA_ROOT` in the notebook to point to your dataset path.

### 2. Run the notebook

```bash
jupyter notebook ResnetTSM.ipynb
```

The notebook is organized into sequential sections:

| Section | Description |
|---|---|
| TSM | Core `TSM` module definition and visualization |
| ResNetTSM | Model architecture definitions |
| Dataset Preparation | `UCF101Dataset` class, transforms, data loaders |
| Training | Train/eval loop, optimizer, scheduler |
| Results | Training curves, accuracy plots |
| Inference | Per-video prediction with top-5 confidence + video output |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-18 (ImageNet pretrained) |
| Frames per clip (`T`) | 8 |
| Input resolution | 112 × 112 |
| Batch size | 8 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | StepLR (step=5, γ=0.5) |
| Loss | CrossEntropyLoss |
| TSM mode | Bidirectional |
| TSM fold divisor | 8 (shifts C/8 each direction) |

---

## Data Augmentation

**Training:**
- Resize → 128×171
- Random crop → 112×112
- Random horizontal flip
- ImageNet normalization

**Validation / Test:**
- Resize → 128×171
- Center crop → 112×112
- ImageNet normalization

---

## Inference

The inference pipeline loads a trained checkpoint, samples 8 frames uniformly from each video, and outputs:

- Predicted class vs. ground truth
- Top-5 class probabilities (bar chart)
- Annotated MP4 video with prediction overlay

```python
# Example inference output
  Video: v_BabyCrawling_g09_c03.avi
   True label:      BabyCrawling
   Predicted:       BabyCrawling  
   Confidence:      94.3%
```

## References

- Lin, J., Gan, C., & Han, S. (2019). [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08767). ICCV 2019.
- [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php) — University of Central Florida
- [torchvision ResNet](https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18)

---
