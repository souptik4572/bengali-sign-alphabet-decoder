# Bengali Sign Language Alphabet Decoder

A PyTorch MobileNetV2 image classifier for recognising 38 Bengali sign language alphabet characters from hand gesture photographs. Designed for real-time inference and future integration with a MediaPipe webcam pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Class Labels](#class-labels)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Inference Example](#inference-example)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

Bengali sign language (BSL) is used by millions of people with hearing and speech impairments across Bangladesh and West Bengal. This project trains a lightweight convolutional neural network to classify static hand gesture images into one of 38 Bengali alphabet characters (11 vowels + 27 consonants).

Key design goals:
- **High accuracy** on unseen test images (~93%)
- **Small model footprint** (< 9 MB) suitable for edge deployment
- **Fast CPU inference** (~40 ms/image) for real-time applications
- **ONNX export** for cross-platform deployment

---

## Dataset

Source: [Bengali Sign Language Dataset](https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset) by muntakimrafi on Kaggle.

| Property | Value |
|---|---|
| Total images | 12,581 |
| Classes | 38 |
| Images per class (train) | ~291 |
| Test images per class | 40 (pre-split) |
| Image format | JPEG, 224×224 RGB |

**Directory layout:**

```
bengali-sign-language-dataset/
├── RESIZED_DATASET/          # training + validation images
│   ├── 0/                    # অ (class index 0)
│   ├── 1/                    # আ (class index 1)
│   └── ...                   # 38 folders total
└── RESIZED_TESTING_DATA/     # held-out test images
    ├── 0/
    └── ...
```

Folders are named numerically (0–37). Because `torchvision.ImageFolder` sorts directories alphabetically, the label mapping is applied manually via `class_labels.json`.

**Train / Val / Test split:** 8,848 / 1,106 / 1,520 images (stratified 80/10/10)

---

## Class Labels

38 classes covering the full Bengali alphabet:

| Index | Character | Type |
|---|---|---|
| 0 | অ | Vowel |
| 1 | আ | Vowel |
| 2 | ই | Vowel |
| 3 | ঈ | Vowel |
| 4 | উ | Vowel |
| 5 | ঊ | Vowel |
| 6 | ঋ | Vowel |
| 7 | এ | Vowel |
| 8 | ঐ | Vowel |
| 9 | ও | Vowel |
| 10 | ঔ | Vowel |
| 11 | ক | Consonant |
| 12 | খ | Consonant |
| 13 | গ | Consonant |
| 14 | ঘ | Consonant |
| 15 | ঙ | Consonant |
| 16 | চ | Consonant |
| 17 | ছ | Consonant |
| 18 | জ | Consonant |
| 19 | ঝ | Consonant |
| 20 | ঞ | Consonant |
| 21 | ট | Consonant |
| 22 | ঠ | Consonant |
| 23 | ড | Consonant |
| 24 | ঢ | Consonant |
| 25 | ণ | Consonant |
| 26 | ত | Consonant |
| 27 | থ | Consonant |
| 28 | দ | Consonant |
| 29 | ধ | Consonant |
| 30 | ন | Consonant |
| 31 | প | Consonant |
| 32 | ফ | Consonant |
| 33 | ব | Consonant |
| 34 | ভ | Consonant |
| 35 | ম | Consonant |
| 36 | য | Consonant |
| 37 | র | Consonant |

---

## Model Architecture

**Base model:** MobileNetV2 (pretrained on ImageNet)

**Custom classification head:**
```
nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 38)
)
```

**Loss:** CrossEntropyLoss
**Optimiser:** Adam

MobileNetV2 was chosen for its excellent accuracy-to-size ratio and suitability for real-time inference on resource-constrained hardware.

**Data augmentation (training only):**
- RandomRotation ±15°
- RandomAffine (translation + scale)
- ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2)
- RandomHorizontalFlip
- RandomPerspective
- Normalise with ImageNet mean/std

These augmentations are specifically chosen to reduce the domain gap between the controlled dataset photographs and future live webcam input.

---

## Training Strategy

Training proceeds in two phases:

### Phase 1 — Head-only (frozen backbone)
| Setting | Value |
|---|---|
| Epochs | 10 |
| Learning rate | 1e-3 |
| Frozen layers | All backbone layers |
| Best val accuracy | 66.55% |

A frozen backbone allows the new classification head to quickly converge without disturbing the pretrained ImageNet features.

### Phase 2 — Fine-tuning
| Setting | Value |
|---|---|
| Epochs | up to 20 (early stopping) |
| Learning rate | 1e-4 |
| Unfrozen layers | Last 5 blocks of MobileNetV2 feature extractor (blocks 14–18) |
| LR scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early stopping | patience=5 on validation loss |
| Best val accuracy | **93.67%** |

---

## Results

| Metric | Value |
|---|---|
| Best validation accuracy | 93.7% |
| **Test accuracy** | **93.4%** |
| Top-3 accuracy | 99.1% |
| Weighted F1 score | 0.934 |
| CPU inference time | ~40.6 ms/image |
| Model size (.pth) | 8.9 MB |
| Model size (.onnx) | 8.6 MB |

**Perfect accuracy (100%)** was achieved on 16 classes.

**Hardest classes:**
| Character | Test Accuracy |
|---|---|
| ঈ | 55.0% |
| উ | 67.5% |

The confusion between visually similar hand shapes (e.g., ঈ vs ই, উ vs ঊ) accounts for most errors. This is expected given the subtle visual differences between these letter pairs.

---

## Output Files

All artefacts are saved to `output/`:

| File | Description |
|---|---|
| `bengali_sign_model.pth` | Full PyTorch model state dict (8.9 MB) |
| `bengali_sign_model.onnx` | ONNX export for cross-platform inference (8.6 MB) |
| `class_labels.json` | Index → Bengali character mapping |
| `confusion_matrix.png` | Per-class confusion heatmap |
| `sample_predictions.png` | Grid of 16 test predictions (wrong = red) |
| `training_curves.png` | Loss and accuracy curves across both training phases |
| `class_distribution.png` | Class distribution bar chart |

---

## Project Structure

```
bengali-sign-alphabet-decoder/
├── bengali_sign_classifier.ipynb   # Main notebook — all 7 steps
├── output/
│   ├── bengali_sign_model.pth
│   ├── bengali_sign_model.onnx
│   ├── class_labels.json
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── training_curves.png
│   └── class_distribution.png
├── bengali-sign-language-dataset/
│   ├── RESIZED_DATASET/            # 38 class folders, ~291 images each
│   └── RESIZED_TESTING_DATA/       # 38 class folders, 40 images each
├── .venv/                          # Virtual environment
└── README.md
```

The notebook is structured as 7 self-contained steps:
1. Dataset exploration
2. Data pipeline with augmentation
3. Model architecture setup
4. Two-phase training
5. Evaluation (confusion matrix, per-class accuracy, top-5 confused pairs)
6. Model export (.pth, .onnx, class_labels.json)
7. Training report summary

---

## Setup & Usage

### Requirements

- Python 3.13+
- PyTorch 2.10+
- Apple Silicon (MPS) or CUDA GPU recommended; CPU training works but is slow

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd bengali-sign-alphabet-decoder

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install jupyter notebook ipykernel
pip install scikit-learn matplotlib seaborn tqdm pillow

# Register the Jupyter kernel
python -m ipykernel install --user --name=bengali-sign-venv --display-name "Bengali Sign (.venv)"
```

> **macOS SSL fix** — if you see SSL certificate errors when downloading pretrained weights, add the following to the top of the notebook:
> ```python
> import ssl, certifi, os
> ssl._create_default_https_context = ssl.create_default_context
> os.environ['SSL_CERT_FILE'] = certifi.where()
> ```

### Running the notebook

```bash
source .venv/bin/activate
jupyter notebook bengali_sign_classifier.ipynb
# Select kernel: "Bengali Sign (.venv)"
```

Run all cells from top to bottom. Training completes in two phases. The full run takes approximately 30–60 minutes on Apple Silicon MPS.

---

## Inference Example

Load the trained model and run inference on a single image:

```python
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from PIL import Image
import json

# Load class labels
with open('output/class_labels.json') as f:
    class_labels = json.load(f)

# Rebuild model
model = mobilenet_v2(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(1280, 38)
)
model.load_state_dict(torch.load('output/bengali_sign_model.pth', map_location='cpu'))
model.eval()

# Preprocessing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inference
img = Image.open('path/to/hand_sign.jpg').convert('RGB')
x = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

with torch.no_grad():
    logits = model(x)
    pred_idx = logits.argmax(dim=1).item()

print(f"Predicted character: {class_labels[str(pred_idx)]}")
```

### ONNX inference

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('output/bengali_sign_model.onnx')
input_name = sess.get_inputs()[0].name

# x is a numpy array of shape (1, 3, 224, 224), dtype=float32
outputs = sess.run(None, {input_name: x.numpy()})
pred_idx = np.argmax(outputs[0])
print(f"Predicted character: {class_labels[str(pred_idx)]}")
```

---

## Known Limitations

- The model is trained on controlled studio photographs. Accuracy may drop on webcam input with different lighting, backgrounds, or hand orientations.
- ঈ and উ are the weakest classes (55% and 67.5% test accuracy respectively) due to their visual similarity to ই and ঊ.
- `NUM_WORKERS=0` is required on macOS for DataLoader stability (multiprocessing fork safety).
- ONNX export requires `dynamo=False` and a fresh CPU model copy (not an MPS-device model).

---

## License

See [LICENSE](LICENSE) for details.

---

*Dataset source: [Bengali Sign Language Dataset](https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset) by muntakimrafi on Kaggle.*
