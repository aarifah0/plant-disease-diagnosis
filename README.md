# Plant Disease Detection

A web application that classifies photographs of plant leaves into one of **38 disease or healthy categories** using a fine-tuned ResNet50 convolutional neural network trained on the PlantVillage dataset. Upload a single leaf image or process an entire batch — the app returns confidence scores, colour-coded severity indicators, and actionable treatment and prevention information for every prediction.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Model Architecture](#model-architecture)
  - [ResNet50 and Residual Learning](#resnet50-and-residual-learning)
  - [Bottleneck Blocks](#bottleneck-blocks)
  - [Network Topology](#network-topology)
  - [Why ResNet50?](#why-resnet50)
- [Transfer Learning Strategy](#transfer-learning-strategy)
- [Loss Function](#loss-function)
- [Optimiser](#optimiser)
- [Data Augmentation and Preprocessing](#data-augmentation-and-preprocessing)
- [Inference Pipeline](#inference-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Dataset](#dataset)
- [Supported Plant Species and Diseases](#supported-plant-species-and-diseases)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Web App](#web-app)
  - [CLI](#cli)
  - [Training Your Own Model](#training-your-own-model)
- [Evaluation](#evaluation)
- [Reproducing the Dataset](#reproducing-the-dataset)

---

## Demo

**Single Image tab** — upload one leaf photo, click Analyze, and get:

- Top-3 predicted classes with a confidence bar chart
- Colour-coded confidence badge: 🟢 High (> 80 %) · 🟡 Medium (60–80 %) · 🔴 Low (< 60 %)
- Expandable panel with description, symptoms, treatment, and prevention tips

**Batch Processing tab** — upload multiple images, process them as a queue, and download a CSV report with filenames, predictions, confidence scores, and error status for any invalid files.

---

## Features

- **Transfer learning** on ResNet50 pretrained on ImageNet — strong visual prior with minimal training time
- **38-class classification** across 14 plant species (disease variants + healthy controls)
- **Confidence-level thresholding** with human-readable severity labels
- **Disease knowledge base** — description, symptoms, treatment, and prevention for every class
- **Batch processing** with real-time progress bar and one-click CSV export
- **Input validation** — enforces file type (JPG/PNG/JPEG) and 10 MB size limit before any inference work
- **Device-aware inference** — automatically selects CUDA → Apple MPS → CPU
- **Flexible training** — freeze backbone (CPU-friendly) or full fine-tune (best accuracy)

---

## Model Architecture

### ResNet50 and Residual Learning

The backbone is **ResNet50** — a 50-layer deep residual network introduced by He et al. (2016, *Deep Residual Learning for Image Recognition*). The central insight is the *residual block*, which reformulates the layer's learning target. Instead of fitting a direct mapping $\mathcal{H}(\mathbf{x})$, the network learns the *residual* $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$, and the true output is recovered via a shortcut connection:

$$\mathbf{y} = \mathcal{F}(\mathbf{x},\, \{W_i\}) + \mathbf{x}$$

This reformulation has two critical consequences:

**1. Solving the vanishing gradient problem.** During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to an early-layer activation $\mathbf{x}$ becomes:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$$

The additive constant $1$ guarantees a direct gradient path regardless of what $\mathcal{F}$ does, preventing the multiplicative decay that collapses gradients in very deep plain networks.

**2. Ease of optimisation.** If the identity is the optimal transformation for a given block, the network only needs to push $\mathcal{F}(\mathbf{x}) \to \mathbf{0}$, which is trivially achieved by driving weights toward zero. Learning the identity directly through a stack of nonlinear layers would be substantially harder.

### Bottleneck Blocks

ResNet50 uses **bottleneck blocks** rather than the plain two-layer blocks of ResNet18/34. Each bottleneck applies three convolutions in sequence:

$$\text{Conv}_{1\times1}(C \to C/4) \;\longrightarrow\; \text{BN} + \text{ReLU} \;\longrightarrow\; \text{Conv}_{3\times3}(C/4 \to C/4) \;\longrightarrow\; \text{BN} + \text{ReLU} \;\longrightarrow\; \text{Conv}_{1\times1}(C/4 \to C)$$

The flanking $1\times1$ convolutions *compress* the channel dimension before and *restore* it after the expensive $3\times3$ convolution. The costly spatial convolution therefore operates on a feature map with $4\times$ fewer channels, reducing FLOPs by roughly 75 % for that operation while preserving representational capacity.

### Network Topology

| Stage | Operation | Blocks | Output shape |
|---|---|---|---|
| Stem | $7\times7$ Conv, stride 2 + BN + ReLU | 1 | $112\times112\times64$ |
| Pool | $3\times3$ Max-pool, stride 2 | — | $56\times56\times64$ |
| Layer 1 | Bottleneck ($C = 256$) | 3 | $56\times56\times256$ |
| Layer 2 | Bottleneck ($C = 512$) | 4 | $28\times28\times512$ |
| Layer 3 | Bottleneck ($C = 1024$) | 6 | $14\times14\times1024$ |
| Layer 4 | Bottleneck ($C = 2048$) | 3 | $7\times7\times2048$ |
| Head | Global average-pool | — | $2048$ |
| **Custom FC** | $\text{Linear}(2048 \to 38)$ | — | $38$ |

The original ImageNet classification head ($\text{Linear}(2048 \to 1000)$) is replaced by $\text{Linear}(2048 \to 38)$, matching the 38 PlantVillage classes. All other weights are preserved from ImageNet pretraining.

Total parameters: ~25.6 M (only ~77 K in the custom FC layer).

### Why ResNet50?

| Alternative | Reason not chosen |
|---|---|
| ResNet18 / 34 | Plain blocks have less capacity; fine-grained disease texture discrimination benefits from the deeper bottleneck hierarchy |
| ResNet101 / 152 | Diminishing accuracy returns on this dataset size; 2–3× inference latency with no practical gain on CPU |
| VGG16 | No residual connections → severe vanishing gradients; 138 M parameters vs 25 M; substantially slower at inference |
| EfficientNet-B0 | Superior accuracy-per-FLOP at the cost of compound-scaling hyperparameters and a less interpretable architecture |
| Vision Transformer (ViT) | Lacks CNN inductive biases (translation equivariance, locality); requires far larger datasets or aggressive augmentation to match CNN accuracy on this scale |
| MobileNetV3 | Optimised for mobile inference; lower accuracy ceiling due to depthwise separable convolutions sacrificing capacity |

ResNet50 hits the practical sweet spot: **25 M parameters**, well-understood training dynamics, strong ImageNet pretraining, runnable on CPU, and widely documented behaviour on similar fine-grained recognition tasks.

---

## Transfer Learning Strategy

Training a 25 M-parameter network from scratch on 87 k images would overfit severely and require GPU-days of compute. Instead the model exploits **transfer learning**: the convolutional backbone is initialised with weights trained on 1.28 M ImageNet images, encoding a rich hierarchy of visual features — from edges and colour blobs in early layers to complex textures and part-level structures in deeper ones. These features transfer well to plant disease recognition because the discriminating signal (leaf texture, colour patterns, lesion morphology) is precisely the kind of mid-to-high-level visual structure encoded by a well-trained CNN backbone.

Two training regimes are supported:

### Backbone-Frozen (default)

All convolutional parameters are frozen (gradients disabled):

```python
for name, param in model.named_parameters():
    param.requires_grad = name.startswith("fc.")
```

The optimisation problem reduces to fitting a linear classifier on top of fixed 2048-dimensional feature vectors extracted by the frozen backbone. This is fast (only ~77 K trainable parameters), converges in a few epochs, and runs on CPU — at the cost of not adapting the backbone to the domain shift between ImageNet and controlled-background leaf photography.

### Full Fine-Tuning (`--full-finetune`)

All layers train simultaneously. The learning rate is reduced (recommended $\eta = 10^{-4}$ vs the default $10^{-3}$) to prevent large gradient steps from destroying the pretrained representations — a phenomenon known as *catastrophic forgetting*. This mode reaches higher accuracy and is the recommended approach when a GPU is available.

---

## Loss Function

Training uses **categorical cross-entropy**, the standard loss for multi-class classification:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{p}_{i,\, y_i}$$

where $N$ is the batch size, $y_i \in \{0, \ldots, 37\}$ is the true class index for sample $i$, and $\hat{p}_{i,c}$ is the predicted probability for class $c$. Probabilities are obtained by applying **softmax** to the raw logit vector $\mathbf{z}_i \in \mathbb{R}^{38}$ output by the FC layer:

$$\hat{p}_{i,c} = \frac{\exp(z_{i,c})}{\displaystyle\sum_{j=0}^{37} \exp(z_{i,j})}$$

Softmax enforces $\hat{p}_{i,c} \geq 0$ and $\sum_c \hat{p}_{i,c} = 1$, producing a valid probability distribution over all 38 classes.

Cross-entropy is the natural choice here for two reasons:

1. It is the negative log-likelihood under a categorical distribution, so minimising it is equivalent to maximum likelihood estimation.
2. Minimising $\mathcal{L}$ is equivalent to minimising the KL divergence $D_{\mathrm{KL}}(p_{\text{true}} \,\|\, \hat{p})$ between the predicted distribution and the one-hot ground truth — directly penalising probability mass placed on incorrect classes.

---

## Optimiser

All training uses **Adam** (Kingma & Ba, 2015 — *Adam: A Method for Stochastic Optimization*). Adam is an adaptive-learning-rate method that maintains exponential moving averages of both the gradient and its squared magnitude:

**First moment** (mean):
$$m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t$$

**Second moment** (uncentred variance):
$$v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2$$

Both are initialised at zero, introducing bias toward zero in early steps. Bias-corrected estimates compensate:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The parameter update is:

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\, \hat{m}_t$$

with defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

The effective per-parameter step size $\alpha / (\sqrt{\hat{v}_t} + \epsilon)$ is small for parameters whose gradient is consistently large (well-conditioned directions are already converging) and larger for parameters with noisy or small gradients (exploring underfit directions). This makes Adam significantly more robust to learning rate tuning than vanilla SGD — particularly valuable when fine-tuning a pretrained network where different layer groups may have very different gradient scales.

---

## Data Augmentation and Preprocessing

### Training Augmentation

Two stochastic transforms are applied per-batch during training:

| Transform | Parameters | Rationale |
|---|---|---|
| Random horizontal flip | $p = 0.5$ | Disease patterns are spatially symmetric; free 2× effective dataset enlargement |
| Random rotation | $\theta \sim \mathcal{U}(-15°,\, +15°)$ | Accounts for variable leaf orientation at image capture time |

No colour jitter or aggressive cropping is used — PlantVillage images are controlled-background photographs with consistent lighting, so heavy appearance augmentation would introduce distribution mismatch rather than reduce it.

### ImageNet Normalisation

All images (train and inference) are normalised channel-wise using the ImageNet population statistics:

$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \quad c \in \{\mathrm{R,\, G,\, B}\}$$

$$\boldsymbol{\mu} = [0.485,\; 0.456,\; 0.406], \qquad \boldsymbol{\sigma} = [0.229,\; 0.224,\; 0.225]$$

These statistics are *not* recomputed on PlantVillage. The pretrained backbone was trained on ImageNet-normalised inputs; applying the same normalisation ensures every convolutional filter operates in its trained input domain. Recomputing statistics on PlantVillage would shift the input distribution seen by the frozen backbone, degrading the quality of its extracted features.

---

## Inference Pipeline

Each prediction follows a deterministic five-step pipeline:

1. **Validation** — file extension and byte-size are checked before any image decoding or model work, rejecting unsupported formats or oversized files early.
2. **Loading** — the file is decoded into an RGB PIL image, converting any RGBA, palette, or greyscale input to a consistent three-channel format.
3. **Preprocessing** — the image is resized to $224 \times 224$ and normalised with ImageNet statistics (see above). A batch dimension is prepended: $\mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{1 \times 3 \times 224 \times 224}$.
4. **Forward pass** — the model runs in `eval()` mode under `torch.no_grad()`, disabling dropout and batch-norm running-stat updates and suppressing gradient computation. Softmax converts logits to probabilities; `torch.topk` extracts the top-$k$ classes.
5. **Knowledge lookup** — each predicted class name is matched against `disease_db.json` to attach human-readable description, symptoms, treatment, and prevention text.

---

## Evaluation Metrics

Model performance is measured on the held-out validation split (≈17,572 images). Four metrics are reported, each computed per-class and then averaged across all 38 classes.

**Precision** — of all instances predicted as class $c$, what fraction truly belongs to it:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

**Recall** — of all true instances of class $c$, what fraction was correctly identified:

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

**F1-score** — harmonic mean of precision and recall, penalising imbalance between the two:

$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

**Macro averaging** weights all classes equally regardless of how many samples they contain:

$$\text{Macro-}F1 = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

**Weighted averaging** weights by class support $n_c$, giving more influence to larger classes:

$$\text{Weighted-}F1 = \frac{\sum_{c=1}^{C} n_c \cdot F1_c}{\sum_{c=1}^{C} n_c}$$

Because PlantVillage classes are approximately balanced, macro and weighted F1 are expected to be close. A large gap would indicate the model performs disproportionately well or poorly on high-frequency classes.

The **confusion matrix** is row-normalised to recall: cell $(i, j)$ shows the fraction of true-class-$i$ samples predicted as class $j$. Off-diagonal hotspots reveal which disease pairs the model most frequently confuses — useful for directing targeted data collection or targeted augmentation.

---

## Dataset

**New Plant Diseases Dataset** (PlantVillage) — available on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

| Split | Images (approx.) |
|---|---|
| Train | ~70,295 |
| Validation | ~17,572 |
| **Total** | **~87,867** |

Images are controlled-background leaf photographs taken under laboratory conditions. The dataset covers 38 classes across 14 plant species, with classes approximately balanced across disease variants and healthy specimens. This controlled setting means the model is best suited to similarly lit, isolated-leaf photographs — performance on field photographs with complex backgrounds may be lower.

---

## Supported Plant Species and Diseases

38 classes across 14 plant species. `Healthy` denotes a disease-free specimen.

| Plant | Conditions |
|---|---|
| Apple | Apple scab · Black rot · Cedar apple rust · Healthy |
| Blueberry | Healthy |
| Cherry (incl. sour) | Powdery mildew · Healthy |
| Corn (maize) | Cercospora leaf spot / Gray leaf spot · Common rust · Northern leaf blight · Healthy |
| Grape | Black rot · Esca (Black Measles) · Leaf blight (Isariopsis) · Healthy |
| Orange | Huanglongbing (Citrus greening) |
| Peach | Bacterial spot · Healthy |
| Bell Pepper | Bacterial spot · Healthy |
| Potato | Early blight · Late blight · Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch · Healthy |
| Tomato | Bacterial spot · Early blight · Late blight · Leaf mold · Septoria leaf spot · Spider mites · Target spot · Yellow leaf curl virus · Mosaic virus · Healthy |

---

## Project Structure

```
plant-disease-detection/
├── app/
│   └── streamlit_app.py       # Streamlit web UI (single image + batch tabs)
├── src/
│   ├── model.py               # ResNet50 definition and weight loading
│   ├── predict.py             # Predictor class + CLI entry point
│   ├── preprocess.py          # Image validation, loading, and preprocessing
│   ├── train.py               # Training loop with CLI arguments
│   ├── evaluate.py            # Validation-set evaluation: metrics + confusion matrix
│   ├── download_data.py       # Kaggle download + Windows-safe extraction
│   └── utils.py               # Shared helpers: paths, device, label formatting
├── data/
│   ├── class_names.json       # 38 class names in model output order
│   └── disease_db.json        # Per-class description, symptoms, treatment, prevention
├── models/
│   ├── resnet50_plantdisease.pth   # Trained weights (~90 MB state_dict)
│   └── README.md              # Expected checkpoint format
├── results/                   # Evaluation outputs (generated by src/evaluate.py)
│   ├── summary.json
│   ├── classification_report.csv
│   └── confusion_matrix.png
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Model definition, training, inference |
| `torchvision` | ≥ 0.15.0 | ResNet50 weights, `ImageFolder`, transforms |
| `streamlit` | ≥ 1.30.0 | Web UI |
| `Pillow` | ≥ 10.0.0 | Image loading and decoding |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `pandas` | ≥ 2.0.0 | Batch results table and CSV export |
| `scikit-learn` | ≥ 1.3.0 | Classification report and confusion matrix |
| `matplotlib` | ≥ 3.7.0 | Confusion matrix heatmap |
| `kaggle` | ≥ 1.6.0 | Dataset download (optional) |

The trained weights at `models/resnet50_plantdisease.pth` are included. To retrain from scratch, see [Training Your Own Model](#training-your-own-model).

---

## Usage

### Web App

```bash
streamlit run app/streamlit_app.py
```

Opens in your browser at `http://localhost:8501`.

**Single Image tab:**
1. Upload a leaf image (JPG, PNG, JPEG — max 10 MB).
2. Click **Analyze**.
3. View the top-3 predictions, confidence bar chart, and disease info panel.

**Batch Processing tab:**
1. Upload multiple leaf images at once.
2. Click **Analyze All** — a progress bar tracks each file.
3. A results table appears with filename, prediction, confidence %, and status.
4. Click **Download results as CSV** to export.

### CLI

Run inference on a single image without the web UI:

```bash
python -m src.predict --image path/to/leaf.jpg
```

| Flag | Default | Description |
|---|---|---|
| `--image` | required | Path to the input image |
| `--weights` | `models/resnet50_plantdisease.pth` | Path to model weights |
| `--top-k` | `3` | Number of top predictions to display |

Example output:

```
Prediction: Tomato - Late blight (94.72%)
Description: Late blight is caused by Phytophthora infestans...
Treatment: Apply copper-based fungicides...

Inference time: 83 ms
Top predictions:
  Tomato - Late blight: 94.72%
  Tomato - Early blight: 3.11%
  Tomato - Leaf Mold: 1.47%
```

### Training Your Own Model

**Step 1 — Download the dataset** (requires a [Kaggle API token](https://www.kaggle.com/settings)):

```bash
python -m src.download_data
```

This downloads and extracts the dataset to `data/raw/` using Windows long-path prefixes (`\\?\`) to avoid the 260-character MAX_PATH limit.

Or supply an already-downloaded zip:

```bash
python -m src.download_data --archive path/to/archive.zip --dest data/raw
```

**Step 2 — Train:**

```bash
# Backbone frozen (fast, CPU-friendly)
python -m src.train --data-dir data/raw --epochs 10 --batch-size 32

# Full fine-tune (recommended with GPU)
python -m src.train --data-dir data/raw --epochs 10 --batch-size 32 --full-finetune --lr 1e-4
```

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | required | Directory containing `train/` and `valid/` subfolders |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--num-workers` | `2` | DataLoader worker processes |
| `--full-finetune` | off | Unfreeze all layers (default: FC only) |
| `--output` | `models/resnet50_plantdisease.pth` | Where to save the best checkpoint |
| `--max-train-per-class` | — | Cap training images per class (quick experiments) |
| `--max-valid-per-class` | — | Cap validation images per class |
| `--max-train-batches` | — | Cap batches per epoch (debugging) |
| `--max-valid-batches` | — | Cap validation batches (debugging) |

The trainer saves the best checkpoint by validation accuracy automatically:

```
Epoch 3/10 | train_loss=0.1832 train_acc=0.9421 | val_loss=0.1204 val_acc=0.9638 | 142.3s
  Saved new best model to models/resnet50_plantdisease.pth (val_acc=0.9638)
```

---

## Evaluation

Run the evaluation script against the validation split to generate a classification report, per-class metrics, and a confusion matrix:

```bash
python -m src.evaluate --data-dir data/raw
```

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | required | Directory containing the `valid/` subfolder |
| `--weights` | `models/resnet50_plantdisease.pth` | Path to model weights |
| `--batch-size` | `64` | Inference batch size |
| `--num-workers` | `2` | DataLoader worker processes |
| `--max-valid-per-class` | — | Cap images per class for a fast approximate run |
| `--out-dir` | `results/` | Where to write output files |

**Outputs** written to `results/`:

| File | Description |
|---|---|
| `summary.json` | Overall accuracy, macro/weighted F1, precision, recall |
| `classification_report.csv` | Per-class precision, recall, F1-score, and support |
| `confusion_matrix.png` | Row-normalised heatmap (recall per class) |

**Results on the full validation set (17,572 images, backbone-frozen training):**

```
==================================================
EVALUATION RESULTS
==================================================
  Overall accuracy : 89.82%
  Macro F1         : 89.51%
  Macro precision  : 90.21%
  Macro recall     : 89.82%
  Weighted F1      : 89.53%
  Samples evaluated: 17,572
  Inference time   : 5458.4s total (CPU)
==================================================
```

The confusion matrix highlights which disease pairs are most commonly confused — useful for directing targeted data collection or additional augmentation.

---

## Reproducing the Dataset

The Kaggle zip contains ~88 k images duplicated under two top-level folder names (upper-case and lower-case variant). The extraction in `src/download_data.py` strips the duplicate tree, keeps the canonical upper-case hierarchy, and prefixes all paths with `\\?\` to bypass the Windows 260-character MAX_PATH limit.

1. Get a Kaggle API token: [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token.
2. Place `kaggle.json` at `~/.kaggle/kaggle.json`, or export `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables.
3. Run:

```bash
python -m src.download_data
```
