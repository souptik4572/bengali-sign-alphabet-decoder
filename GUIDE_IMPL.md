# Task: Build a Bengali Sign Language Alphabet Classification Model

## Context
I have the **Bengali Sign Language Dataset** from Kaggle (by muntakimrafi):
https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset

The dataset contains static images of hand signs representing Bengali alphabet characters. Based on published research about this dataset, it likely has the following structure:
- **36 classes** representing Bengali alphabet characters (vowels + consonants)
- Images organized in **folders named by the Bengali character/class label**
- Images are hand gesture photographs captured via smartphones
- Approximately **50 images per class** (though this may vary — verify from the actual data)

The dataset zip file is located at: `./bengali-sign-language-dataset.zip` (adjust this path if your downloaded file is named differently or located elsewhere).

## Objective
Build a **PyTorch image classification model** using **MobileNetV2 with transfer learning** that can classify Bengali sign language hand gestures. The model must be optimized for **real-time inference** (it will later be used in a live webcam pipeline with MediaPipe).

## Step-by-Step Instructions

### Step 1: Dataset Exploration
- Unzip and explore the dataset structure (folder names, image counts, formats, resolutions)
- Print a summary: number of classes, images per class, image dimensions (sample a few), any class imbalance
- Display the class-to-label mapping (folder name → Bengali character)
- Flag any issues: corrupted images, inconsistent sizes, near-empty classes

### Step 2: Data Pipeline
- Use `torchvision.datasets.ImageFolder` with appropriate transforms
- **Split**: 80% train / 10% validation / 10% test (use stratified splitting to preserve class balance)
- **Training transforms** (aggressive augmentation to reduce domain gap with future webcam input):
  - Resize to 224×224
  - RandomRotation(±15°)
  - RandomAffine (slight translation + scale)
  - ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2)
  - RandomHorizontalFlip (to handle left/right hand variation)
  - RandomPerspective (slight, to simulate angle changes)
  - Normalize with ImageNet mean/std
- **Validation/Test transforms**: Resize 224×224 + Normalize only
- Use DataLoader with batch_size=32, num_workers=4, pin_memory=True

### Step 3: Model Architecture
- Load `torchvision.models.mobilenet_v2(pretrained=True)`
- Freeze all backbone layers initially
- Replace the classifier head:
  ```
  nn.Sequential(
      nn.Dropout(0.2),
      nn.Linear(1280, num_classes)
  )
  ```
- Use CrossEntropyLoss and Adam optimizer

### Step 4: Training Strategy (Two-Phase)

**Phase 1 — Head-only training (frozen backbone):**
- Train only the classifier head for 10 epochs
- Learning rate: 1e-3
- This quickly learns a good classification head

**Phase 2 — Fine-tuning (unfreeze deeper layers):**
- Unfreeze the last 4-5 blocks of MobileNetV2's feature extractor
- Train for up to 20 more epochs
- Learning rate: 1e-4 (lower, to not destroy pretrained weights)
- Use ReduceLROnPlateau scheduler (patience=3, factor=0.5)
- Implement early stopping (patience=5 based on validation loss)

**During both phases, track and print per-epoch:**
- Training loss and accuracy
- Validation loss and accuracy
- Current learning rate

### Step 5: Evaluation
After training completes:
- Evaluate on the held-out **test set**
- Print overall accuracy, top-3 accuracy, and weighted F1 score
- Generate and save a **confusion matrix** as a heatmap image (`confusion_matrix.png`) with Bengali class labels
- Print the **top-5 most confused class pairs** (where the model makes the most errors)
- Print **per-class accuracy** sorted from worst to best (to identify problematic signs)
- Save a few **sample predictions** grid image (`sample_predictions.png`) — show 16 test images with predicted vs true labels, highlighting wrong predictions in red

### Step 6: Model Export
- Save the trained model as `bengali_sign_model.pth` (full state dict)
- Also export to **ONNX format** (`bengali_sign_model.onnx`) with input shape (1, 3, 224, 224) for optimized inference
- Save the **class index → label mapping** as `class_labels.json`
- Print the model size (MB) and estimated inference time per image on CPU

### Step 7: Summary Report
At the end, print a clean summary:
```
=== Bengali Sign Language Model — Training Report ===
Dataset: X classes, Y total images
Train/Val/Test split: A / B / C images
Model: MobileNetV2 (transfer learning)
Best validation accuracy: XX.X%
Test accuracy: XX.X%
Test top-3 accuracy: XX.X%
Test F1 score (weighted): X.XXX
Model size: X.X MB
Estimated CPU inference time: X.X ms/image
Saved files: bengali_sign_model.pth, bengali_sign_model.onnx, class_labels.json, confusion_matrix.png, sample_predictions.png
```

## Important Notes
- If the dataset structure is different from expected (e.g., nested folders, CSV labels, train/test pre-split), **adapt accordingly** — explore first, then build the pipeline to match
- If GPU is available, use it; otherwise train on CPU (it should still be feasible given the small dataset size)
- Use `tqdm` for progress bars during training
- Set random seeds (42) for reproducibility across numpy, torch, and random
- All output files should be saved in a dedicated `./output/` directory
- If any class has fewer than 5 images, print a warning and consider excluding it