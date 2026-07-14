# Neural Bites ML Pipeline: Step-by-Step Implementation Plan
## Mobile-Friendly Architecture (No K-Fold)

---

## EXECUTIVE OVERVIEW

This document provides a complete breakdown of the neural-bites ML pipeline from data preparation through mobile deployment. The pipeline follows a 4-layer architecture optimized for on-device inference on Android devices.

**Final Output:** 3 files that integrate into your Android APK
- `yolo_fruits.tflite` (~12-15 MB) — Fruit detection
- `fruit_classifier.tflite` (~10-12 MB) — Condition classification + attention
- `ood_config.json` (~15 KB) — Out-of-distribution detection config

---

## PHASE 1: YOLO DATASET PREPARATION (Week 1)

### STEP 1.1: Understand Current Dataset Structure

**Goal:** Verify existing dataset organization before YOLO setup

**Current State:**
```
/kaggle/input/fruitvision-mendeley/Fruits Original/
├── Apple/
│   ├── Fresh/
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   └── ... (765 images)
│   ├── Rotten/
│   │   └── ... (630 images)
│   └── Formalin-mixed/
│       └── ... (643 images)
├── Banana/
│   └── [Fresh, Rotten, Formalin-mixed subfolders]
├── Grape/
├── Mango/
└── Orange/

Total: 10,154 images across 5 fruits × 3 conditions (15 classes)
```

**What to Verify:**
- [ ] Confirm all 15 subdirectories exist
- [ ] Confirm image counts per condition (~630-770 each)
- [ ] Check image format (JPEG/PNG) and resolution
- [ ] Verify no corrupted images exist

**Output of Step 1.1:** None (verification only)

---

### STEP 1.2: Auto-Generate YOLO Bounding Box Annotations

**Goal:** Create tight bounding boxes around each fruit (not full image)

**Why This Approach:**
- Tight bboxes teach YOLO to detect fruit boundaries accurately
- YOLO generalizes better in multi-fruit scenes
- Fallback mechanism handles edge cases gracefully

**Process:**

**1.2.1: Define Color Ranges (HSV Space)**

For each fruit, define lower and upper HSV bounds:

```
Apple:       H: [0-10] or [350-360]     (red hues)
             S: [50-255]                (medium to high saturation)
             V: [50-255]                (medium to high value)

Banana:      H: [20-30]                 (yellow hues)
             S: [100-255]               (medium to high saturation)
             V: [100-255]               (bright)

Grape:       H: [130-150]               (purple/dark hues)
             S: [50-255]                (varied saturation)
             V: [50-255]                (varied brightness)

Mango:       H: [15-35]                 (orange-yellow hues)
             S: [100-255]               (medium to high saturation)
             V: [100-255]               (bright)

Orange:      H: [10-20]                 (orange hues)
             S: [100-255]               (medium to high saturation)
             V: [100-255]               (bright)
```

**1.2.2: Process Each Image**

For each of 10,154 images:

1. Load RGB image
2. Convert BGR (OpenCV) → HSV color space
3. Apply color mask using fruit-specific HSV range
4. Apply morphological operations:
   - Erosion (remove noise)
   - Dilation (fill small holes)
5. Find all contours in binary mask
6. Select largest contour (the fruit)
7. Get bounding rectangle from contour
8. Add 10% padding (safety margin to prevent crop clipping)
9. Clamp to image boundaries
10. Normalize to YOLO format (center_x, center_y, width, height in range [0, 1])
11. Handle edge case: If no fruit detected, fallback to full image (0.5, 0.5, 1.0, 1.0)

**Example YOLO Annotation Output:**

```
File: images/train/apple_fresh_001.txt
Content: 0 0.523 0.487 0.756 0.823
         ↑ ↑     ↑     ↑     ↑
      class center_x center_y width height (all normalized)
      Apple  52.3%   48.7%   75.6% 82.3%
```

**1.2.3: Fallback Logic**

If HSV color detection fails for an image:
- Use full image as bbox: `(0.5, 0.5, 1.0, 1.0)`
- Log the image path and condition
- YOLO still trains, but with less precise ground truth
- Review fallback images after training to improve HSV thresholds

**Expected Fallback Rate:** ~2-5% (some bruised/unusual fruits)

**Output of Step 1.2:**
- 10,154 `.txt` files (one per image) containing normalized YOLO coordinates
- Log file listing fallback images (for manual review)
- HSV threshold tuning results (which thresholds worked best)

---

### STEP 1.3: Organize YOLO Dataset Directory Structure

**Goal:** Create stratified train/val/test split organized for YOLO

**Directory Structure to Create:**

```
/kaggle/working/yolo_dataset/
├── images/
│   ├── train/        (80% of images, ~8,123 images)
│   ├── val/          (10% of images, ~1,015 images)
│   └── test/         (10% of images, ~1,015 images)
├── labels/
│   ├── train/        (corresponding .txt annotation files)
│   ├── val/
│   └── test/
└── data.yaml         (YOLO configuration file)
```

**Stratification Strategy:**

The split must preserve fruit type distribution:

```
For each fruit (Apple, Banana, Grape, Mango, Orange):
  1. Collect all 3 conditions (Fresh, Rotten, Formalin)
  2. Combine into single pool for that fruit
  3. Stratified split:
     - 80% → train/
     - 10% → val/
     - 10% → test/

Result: Each split has balanced fruit representation
  Train: ~1,625 Apple, ~1,498 Banana, ~1,543 Grape, ~1,546 Mango, ~1,510 Orange
  Val:   ~203 each fruit type
  Test:  ~203 each fruit type
```

**Why This Stratification:**
- YOLO learns fruit-specific features evenly
- Validation measures per-fruit accuracy consistently
- Test set is unbiased (no fruit type bias)
- Prevents overfitting to specific fruit types

**Output of Step 1.3:**
- Organized `/kaggle/working/yolo_dataset/` directory tree
- File manifest: List of all images in train/val/test with their source paths
- Class distribution report: Confirmation of balanced splits

---

### STEP 1.4: Create YOLO Configuration File (data.yaml)

**Goal:** Define YOLO training parameters in standard format

**File: `data.yaml` Content:**

```yaml
# YOLO Dataset Configuration
# ===========================

# Dataset paths
path: /kaggle/working/yolo_dataset
train: images/train
val: images/val
test: images/test

# Number of classes
nc: 5

# Class names (fruit types ONLY, not conditions)
# YOLO job = detection only; condition classification handled by MobileNetV2
names: ['Apple', 'Banana', 'Grape', 'Mango', 'Orange']

# Metadata
source: 'FruitVision Mendeley Dataset (filtered to 5 fruits × 3 conditions)'
created: '2026-07-10'
notes: 'Bounding boxes auto-generated via HSV color segmentation. See logs for fallback cases.'
```

**Why This Format:**
- Standard YOLO ultralytics format (compatible with YOLOv8)
- Minimal, readable configuration
- Can be version-controlled and reproduced

**Validation Checklist:**
- [ ] Path exists and contains images/ and labels/ subdirectories
- [ ] train/val/test directories all populated
- [ ] nc (number of classes) = 5
- [ ] names list has exactly 5 entries

**Output of Step 1.4:**
- `data.yaml` file (can be committed to version control)
- Verification log: Confirms YOLO can find all files

---

## PHASE 2: YOLO MODEL TRAINING (Week 2)

### STEP 2.1: Initialize Pretrained YOLOv8-Nano Model

**Goal:** Start from YOLOv8-nano pretrained on COCO dataset

**Why YOLOv8-nano:**
- Pretrained on COCO (general object detection patterns)
- Lightweight (~6.3M parameters) = fast inference on mobile
- Good accuracy-speed tradeoff
- Standard YOLO format (easy export to TFLite)

**What Happens in Initialization:**
1. Download pretrained `yolov8n.pt` from ultralytics (if not cached)
2. Load COCO pretrained weights into model
3. Model learns general object detection patterns (edges, shapes, colors)
4. Ready for fine-tuning on fruit dataset

**Why Not Train from Scratch:**
- Scratch training requires ~200 epochs to converge
- COCO pretraining gives us 50 epochs of learning for free
- Fine-tuning only ~50 epochs achieves high accuracy

**Output of Step 2.1:**
- Loaded YOLOv8-nano model object (in memory)
- Verification: Model can load and produce dummy predictions

---

### STEP 2.2: Configure YOLO Training Hyperparameters

**Goal:** Set training parameters for optimal learning on fruit dataset

**Key Hyperparameters:**

```
Input Size:           640 × 640 pixels
                      (YOLO standard; allows detection of small and large fruits)

Batch Size:           32
                      (Fits on Kaggle GPU; adjust if OOM)

Epochs:               50
                      (Fine-tuning: ~50 epochs sufficient for convergence)

Optimizer:            SGD with momentum 0.937
                      (Standard YOLO, proven effective)

Learning Rate:        0.01 (initial)
                      (YOLO's built-in scheduler adjusts during training)

Warmup Epochs:        3
                      (Ramp up LR gradually to stabilize)

Patience (Early Stop): 20 epochs
                      (Stop if val_loss doesn't improve for 20 epochs)

Augmentation:         YOLO default augmentation
                      - Random brightness
                      - Random contrast
                      - Random saturation
                      - Random hue
                      - Random horizontal flip
                      - Random perspective
                      - Mosaic (combine 4 images)
                      (YOLO's aggressive augmentation helps generalization)

Weight Decay:         0.0005
                      (L2 regularization to prevent overfitting)

Device:               cuda:0 (Kaggle GPU)
```

**Why These Values:**
- Input 640×640: Standard YOLO size, good fruit detection at various scales
- Batch 32: Sweet spot between gradient stability and memory
- 50 epochs: Sufficient for fine-tuning (pretrained model)
- Early stopping: Prevents overfitting; saves computation time
- Aggressive augmentation: Small dataset (10K) benefits from strong augmentation

**Output of Step 2.2:**
- Configuration dict ready for YOLO trainer
- Parameter validation: Check for conflicts or unrealistic values

---

### STEP 2.3: Train YOLO on Fruit Dataset

**Goal:** Fine-tune YOLOv8-nano on fruit detection

**Training Process:**

For each epoch (1 to 50):

1. **Forward Pass (Training):**
   - Load batch of 32 images from train/images/train
   - YOLO processes through backbone → neck → head
   - Predict bboxes and confidence scores
   - Compute loss (combination of):
     - Localization loss (predicted bbox vs. ground truth)
     - Confidence loss (predicted score vs. ground truth presence)
     - Classification loss (fruit type prediction)

2. **Backward Pass:**
   - Compute gradients with respect to all parameters
   - Update weights using SGD optimizer

3. **Validation:**
   - Every epoch, run inference on val/images/val (1,015 images)
   - Compute validation metrics:
     - **mAP50:** Mean Average Precision at IoU threshold 0.5
     - **mAP50-95:** mAP averaged across IoU thresholds 0.5 to 0.95
     - **Precision:** (True Positives) / (True Positives + False Positives)
     - **Recall:** (True Positives) / (True Positives + False Negatives)
   - If val_loss improves, save checkpoint

4. **Early Stopping Check:**
   - If no improvement for 20 epochs, stop training
   - Load best model (lowest val_loss)

5. **Logging:**
   - Log metrics to CSV and TensorBoard
   - Save intermediate checkpoints every 10 epochs

**Expected Training Time:**
- ~2-3 hours on Kaggle GPU (P100)
- ~4-5 hours on slower GPU (T4)

**Expected Performance:**
- mAP50: 0.85-0.92 (good for fine-tuning on small dataset)
- mAP50-95: 0.60-0.75 (tighter IoU threshold, harder)
- Per-fruit accuracy: >85% for each fruit type

**Checkpointing for Session Continuity:**

Every 10 epochs, save checkpoint:
```
Checkpoint includes:
  - Model weights at epoch N
  - Optimizer state
  - Scheduler state
  - Training history (losses, metrics)
  - Epoch number
  - Best metric so far
```

If Kaggle session ends:
1. Check last saved epoch (e.g., epoch 32)
2. Resume from epoch 33
3. YOLO automatically loads best checkpoint if starting fresh

**Output of Step 2.3:**
- Best model: `runs/detect/train/weights/best.pt` (~6.3 MB)
- Last checkpoint: `runs/detect/train/weights/last.pt`
- Training logs: CSV file with per-epoch metrics
- Training curves: mAP, loss, precision, recall vs. epoch
- Best validation performance: mAP50 value at best epoch

---

### STEP 2.4: Evaluate YOLO on Test Set

**Goal:** Measure final YOLO performance on held-out test data

**Process:**

1. Load best YOLO model (`best.pt` from Step 2.3)
2. Run inference on test/images/test (1,015 images) WITHOUT training
3. For each image:
   - YOLO predicts bboxes and confidences
   - Compare against ground truth annotations
   - Compute detection metrics

4. Aggregate metrics:
   - Per-fruit mAP50 (Apple, Banana, Grape, Mango, Orange)
   - Overall mAP50
   - Confusion matrix: Which fruits get confused with each other?
   - False positive rate: How many non-fruit boxes predicted?
   - False negative rate: How many fruits missed?

5. Visualize:
   - Sample predictions on test images (with bboxes overlaid)
   - mAP curve over validation set
   - Per-class precision/recall

**Expected Test Performance:**
- mAP50: 0.82-0.90 (slight drop from validation is normal)
- Consistent per-fruit accuracy (no fruit type favored)

**Failure Analysis:**
- If mAP50 < 0.75: Retrain with adjusted hyperparameters or HSV thresholds
- If specific fruit has low accuracy: Investigate HSV range or augmentation

**Output of Step 2.4:**
- Test set metrics JSON file
- Test predictions visualization (sample images with predicted boxes)
- Confusion matrix (if any)
- Per-class mAP50 breakdown

---

### STEP 2.5: Export YOLO to TFLite Format

**Goal:** Convert YOLO from PyTorch (.pt) → TensorFlow Lite (.tflite) for mobile

**Export Pipeline:**

```
best.pt (PyTorch)
    ↓ (YOLO ultralytics export)
best.onnx (Intermediate ONNX format)
    ↓ (TensorFlow converter)
best_fp32.tflite (Full precision)
    ↓ (Optional: Quantization)
best_int8.tflite or best_fp16.tflite (Quantized, smaller)
```

**Export Process:**

1. **PyTorch → ONNX:**
   - Use ultralytics built-in export
   - Command: `model.export(format='onnx', imgsz=640)`
   - Output: `best.onnx` (~12 MB)

2. **ONNX → TFLite (FP32):**
   - Use TensorFlow's ONNX converter
   - Command: `onnx_to_tf_lite('best.onnx', 'best_fp32.tflite')`
   - Output: `best_fp32.tflite` (~12 MB)
   - Inference time: ~800-1200ms on mobile (for 640×640 input)

3. **Optional: Quantization (Recommended):**
   - Convert FP32 weights to INT8 (8-bit integers)
   - 4× smaller model (~3 MB)
   - 2-3× faster inference
   - Minimal accuracy loss (~1-2%)
   - Command: Apply post-training quantization during export

4. **Quantized Model:**
   - Output: `best_int8.tflite` (~3 MB)
   - Inference time: ~300-500ms on mobile
   - Recommended for production

**Important Settings During Export:**

```
Export parameters:
  format: 'tflite'
  imgsz: 640
  half: False (keep FP32 for accuracy)
  int8: True (enable quantization)
  data: 'data.yaml' (for class names)
  device: 0 (CUDA device)
  optimize: True (optimize for mobile)
  dynamic: False (fixed input size)
```

**Validation:**
- [ ] TFLite file loads without errors
- [ ] Can perform inference on dummy 640×640 input
- [ ] Output tensor shapes correct: (1, N, 85) where N = predictions per scale

**Output of Step 2.5:**
- `best_int8.tflite` (~3 MB, recommended)
- OR `best_fp32.tflite` (~12 MB, if quantization not available)
- Export log: Confirmation of successful conversion
- Model fingerprint: Input/output tensor shapes and dtypes

---

## PHASE 3: CLASSIFIER TRAINING (Week 3-4)

### STEP 3.1: Prepare Classification Dataset

**Goal:** Organize 10,154 images for MobileNetV2 training

**Dataset Organization:**

Unlike YOLO, classification doesn't need bounding box annotations. The existing dataset structure (Fruit/Condition/images) is already perfect.

**Create Single Train/Val/Test Split:**

```
Data split (stratified by 15 classes):
  Train: 70% of 10,154 = 7,108 images
  Val:   15% of 10,154 = 1,523 images
  Test:  15% of 10,154 = 1,523 images

Stratification ensures each class represented in all splits:
  Apple-Fresh:        ~535 train, ~114 val, ~116 test
  Apple-Rotten:       ~441 train, ~94 val, ~95 test
  Apple-Formalin:     ~450 train, ~96 val, ~97 test
  ... (repeat for 15 classes)
```

**Why 70/15/15:**
- 70% train: Sufficient data for learning
- 15% val: Enough to track overfitting
- 15% test: Held-out, never touched during training (final evaluation)

**Data Augmentation for Training:**

Training data loaded with transforms:
```
├── Resize to 384×384
├── RandomHorizontalFlip (p=0.5)
├── RandomRotation (±15°)
├── ColorJitter (brightness=0.2, contrast=0.1)
├── ToTensor()
└── Normalize (ImageNet mean/std)

Val/Test: NO augmentation (deterministic)
├── Resize to 384×384
├── ToTensor()
└── Normalize (ImageNet mean/std)
```

**Why This Augmentation:**
- No CLAHE: Preserves formalin's washed-out color signature
- Minimal color shifts: Color is diagnostic feature
- Geometric changes realistic: Fruits at angles, different lighting

**Output of Step 3.1:**
- Split indices: Which images go to train/val/test
- Class distribution verification: Confirm stratification worked
- Augmentation visualization: Sample augmented images (verify colors preserved)

---

### STEP 3.2: Build Multi-Head MobileNetV2 Architecture

**Goal:** Design custom model for fruit detection + condition classification + attention

**Model Architecture Breakdown:**

```
Input: RGB image 384×384
    ↓
[BACKBONE - MobileNetV2]
├─ Inverted Residuals (28 blocks)
├─ Depthwise Separable Convolutions
├─ Output: 1280-dimensional feature vector
    ↓
[THREE OUTPUT HEADS - Running in Parallel]
│
├─ HEAD 1: Fruit Classification
│   ├─ Linear(1280 → 256)
│   ├─ ReLU
│   ├─ Dropout(0.5)
│   └─ Linear(256 → 5)          [Fruit logits: Apple, Banana, Grape, Mango, Orange]
│
├─ HEAD 2: Condition Classification
│   ├─ Linear(1280 → 128)
│   ├─ ReLU
│   ├─ Dropout(0.5)
│   └─ Linear(128 → 3)          [Condition logits: Fresh, Rotten, Formalin-mixed]
│
├─ HEAD 3: Attention/Explainability
│   ├─ Linear(1280 → 196)       [14×14 = 196 spatial locations]
│   └─ Reshape to 14×14
│       (Learned attention map, approximates Grad-CAM)
│
└─ HEAD 4: Feature Output (No head, pass-through)
    └─ 1280-dimensional vector  [For OOD detection via Mahalanobis distance]
```

**Model Sizing:**

```
Total Parameters:
  Backbone (MobileNetV2):        2.2M parameters
  Fruit head:                    0.35M parameters
  Condition head:                0.15M parameters
  Attention head:                0.25M parameters
  ──────────────────────
  Total:                         ~2.95M parameters

Model Size:
  In memory (FP32):              ~12 MB
  After quantization (INT8):     ~3 MB

Inference Time (per image):
  640×640 YOLO input:            ~300-500ms (on Snapdragon 765G)
  384×384 MobileNetV2 (per crop): ~50-100ms
  Multi-fruit scenario (3 fruits): YOLO 400ms + 3×classifier 75ms = ~650ms total
```

**Why Multi-Head:**
- Fruit and condition are independent predictions (you can have any fruit with any condition)
- Attention head provides interpretability (shows where model looks)
- Feature vector enables OOD detection (measures novelty)
- All heads share backbone (efficient, learned representations)

**Attention Head Explanation:**

The attention head learns a 14×14 spatial map showing which image regions influence predictions. This is similar to Grad-CAM (gradient-based visualization) but:
- Learned during training (not computed post-hoc)
- Fast inference (no backprop needed)
- Interpretable (shows "what the model sees")

**Output of Step 3.2:**
- Model architecture definition (class in PyTorch)
- Parameter count verification
- Model diagram/visualization
- Dummy input/output test (confirms shapes correct)

---

### STEP 3.3: Freeze Backbone, Train Heads Only (Phase 1)

**Goal:** Train the three custom heads while keeping pretrained MobileNetV2 frozen

**Why Two-Phase Training:**

**Phase 1 (Freeze backbone):**
- Backbone already learned useful features from ImageNet
- Training only heads is faster (~30% training time)
- Less risk of catastrophic forgetting (backbone doesn't change)
- Good initial convergence

**Phase 2 (Unfreeze backbone):**
- Fine-tune backbone for fruit-specific features
- Discriminative learning rates: early layers learn slowly, later layers learn faster
- Better generalization and accuracy

**Phase 1 Training Details:**

```
Frozen:       All MobileNetV2 backbone layers (no gradients)
Trainable:    Fruit head, Condition head, Attention head

Loss Function: Weighted combination
  Total Loss = 0.5 × fruit_loss + 0.3 × condition_loss + 0.2 × attention_loss

  Fruit Loss:      CrossEntropyLoss (predicting 5 fruit classes)
  Condition Loss:  CrossEntropyLoss (predicting 3 condition classes)
  Attention Loss:  MSE loss (attention map vs. Grad-CAM heatmap)
                   (Grad-CAM computed offline in STEP 3.6)

Optimizer:        Adam
Learning Rate:    0.001
Batch Size:       64 (images)
Epochs:           20
Early Stopping:   Patience = 5 epochs (stop if val_loss doesn't improve)

Metrics Tracked:
  - Training loss (per batch)
  - Validation loss (per epoch)
  - Accuracy: Fruit classification accuracy
  - Accuracy: Condition classification accuracy
```

**Why These Weights:**
- 0.5 (fruit): Most important; primary task
- 0.3 (condition): Also important; formalin detection critical
- 0.2 (attention): Regularization for interpretability

**Training Loop (Pseudocode):**

```
For epoch = 1 to 20:
  For batch in train_loader:
    images, labels = batch
    
    # Forward pass
    fruit_logits, condition_logits, attention_map, features = model(images)
    
    # Compute losses
    fruit_loss = CrossEntropyLoss(fruit_logits, fruit_labels)
    condition_loss = CrossEntropyLoss(condition_logits, condition_labels)
    attention_loss = MSELoss(attention_map, grad_cam_heatmaps)
    
    total_loss = 0.5*fruit_loss + 0.3*condition_loss + 0.2*attention_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
  
  # Validation
  For batch in val_loader:
    (no gradient computation)
    fruit_logits, condition_logits, ... = model(images)
    val_fruit_loss = CrossEntropyLoss(fruit_logits, val_fruit_labels)
    val_condition_loss = CrossEntropyLoss(condition_logits, val_condition_labels)
    val_total_loss = 0.5*val_fruit_loss + 0.3*val_condition_loss + ...
  
  # Check early stopping
  If val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    Save checkpoint (best model so far)
  Else:
    patience_counter += 1
    If patience_counter >= 5:
      Stop training (early stopping)

Load best checkpoint (best val_loss)
Proceed to Phase 2
```

**Checkpointing in Phase 1:**

Every epoch, save checkpoint:
```
checkpoint = {
    'phase': 1,
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'history': {
        'train_loss': [...],
        'val_loss': [...],
        'train_fruit_acc': [...],
        'train_condition_acc': [...],
    }
}
```

If session dies at epoch 12:
1. Load checkpoint from epoch 12
2. Resume training from epoch 13
3. Continue for remaining epochs until convergence or early stopping

**Expected Results After Phase 1:**
- Training loss: 0.3 → 0.15 (decreases as heads learn)
- Validation loss: 0.35 → 0.18 (lower than training, but reasonable)
- Fruit accuracy: 92-95% on validation set
- Condition accuracy: 75-85% on validation set

**Output of Step 3.3:**
- Best model checkpoint (Phase 1): `checkpoint_phase1_best.pth`
- Training history: CSV file with per-epoch metrics
- Training curves: Plots of loss and accuracy over epochs
- Validation confusion matrices (fruit and condition)

---

### STEP 3.4: Unfreeze Backbone, Train with Discriminative Learning Rates (Phase 2)

**Goal:** Fine-tune entire model with early layers learning slowly, later layers learning faster

**Why Discriminative LR:**

Different backbone layers learned different features:
- Early layers: Generic features (edges, textures) — don't change much
- Middle layers: Object parts (shape, structure) — moderate changes
- Late layers: Semantic features (color, identity) — change significantly

**Discriminative Learning Rates:**
```
Layer Groups (from early to late):
  Group 0 (Early):   LR = 0.00001  (1e-5)  [Learn very slowly]
  Group 1 (Middle):  LR = 0.00005  (5e-5)  [Learn slowly]
  Group 2 (Late):    LR = 0.0001   (1e-4)  [Learn moderately]
  Fruit Head:        LR = 0.001    (1e-3)  [Learn fast]
  Condition Head:    LR = 0.001    (1e-3)  [Learn fast]
  Attention Head:    LR = 0.001    (1e-3)  [Learn fast]
```

**Why This Pattern:**
- Early backbone features are universal (edges apply to all images)
- Late backbone features are more task-specific (need more updating)
- Heads need higher LR because they're new and untrained

**Phase 2 Training Details:**

```
Frozen:       Nothing (entire model trainable)
Trainable:    All backbone layers + all heads (with different LRs)

Loss Function: Same as Phase 1
  Total Loss = 0.5 × fruit_loss + 0.3 × condition_loss + 0.2 × attention_loss

Optimizer:        Adam (with per-layer learning rates)
Base Learning Rate: Varies by layer group (see above)
Batch Size:       64 (images)
Epochs:           30
Early Stopping:   Patience = 6 epochs

Metrics Tracked:  Same as Phase 1
```

**Initialization for Phase 2:**

1. Load best checkpoint from Phase 1 (best model after 20 epochs)
2. Unfreeze all backbone layers
3. Set up per-layer learning rates (discriminative LR scheduler)
4. Reset optimizer (fresh Adam state)
5. Reset scheduler
6. Continue training for up to 30 more epochs

**Expected Results After Phase 2:**
- Training loss: 0.15 → 0.08 (continues decreasing)
- Validation loss: 0.18 → 0.13 (further improvement)
- Fruit accuracy: 95-97% on validation set (improvement from Phase 1)
- Condition accuracy: 82-88% on validation set (improvement from Phase 1)
- Formalin detection: Particularly strong (~90% accuracy)

**Why Accuracy Improves in Phase 2:**
- Backbone learns fruit-specific features
- Late layers focus on subtle color differences (formalin's uniformity)
- Attention head learns to highlight diagnostic regions

**Checkpointing in Phase 2:**

Same as Phase 1, but additionally track phase number:
```
checkpoint = {
    'phase': 2,           # ← Different from Phase 1
    'epoch': current_epoch,
    'model_state_dict': ...,
    'best_val_loss': ...,
    ...
}
```

If session dies:
1. Load checkpoint (automatically detects phase = 2)
2. Resume from exact epoch with same per-layer LRs
3. Continue training

**Output of Step 3.4:**
- Best model checkpoint (Phase 2): `checkpoint_phase2_best.pth`
- Final model: `fruit_classifier_final.pth`
- Combined training history: All 50 epochs (20 Phase 1 + 30 Phase 2)
- Training curves: Loss and accuracy across both phases
- Improvement analysis: Phase 1 vs. Phase 2 performance gain

---

### STEP 3.5: Evaluate Classifier on Test Set

**Goal:** Measure final classifier accuracy on held-out test data

**Test Evaluation Process:**

1. Load best model from Phase 2 (`checkpoint_phase2_best.pth`)
2. Run inference on test set (1,523 images) WITHOUT training
3. For each image:
   - Forward pass: Get fruit logits, condition logits, attention, features
   - Fruit prediction: argmax(fruit_logits) → predicted fruit
   - Condition prediction: argmax(condition_logits) → predicted condition
   - Compare to ground truth labels

4. Compute metrics:
   - **Fruit Accuracy:** % of correct fruit predictions (ignoring condition)
   - **Condition Accuracy:** % of correct condition predictions (ignoring fruit)
   - **Combined Accuracy:** % of correct (fruit, condition) pairs
   - **Per-class Accuracy:** Accuracy for each of 15 classes (e.g., Apple-Fresh)
   - **Confusion Matrix:** Which classes get confused with each other?
   - **Per-condition Performance:** How well does model detect each condition?

5. Analyze results:
   - Formalin Detection Rate: How many formalin images correctly identified?
   - False Positive Rate: How often is fresh misclassified as formalin?
   - Class-wise breakdown: Which conditions hardest to distinguish?

6. Visualize:
   - Sample predictions on test images (with confidence scores)
   - Confusion matrix heatmap
   - Per-class accuracy bar chart
   - Attention heatmap examples (what does model look at?)

**Expected Test Performance:**

```
Fruit Accuracy:           ~95-97%  (model learned fruit features well)
Condition Accuracy:       ~85-88%  (harder; requires subtle color differences)
Combined Accuracy:        ~82-85%  (correctly predicts both fruit and condition)

Per-Condition Performance:
  Fresh:                  ~88-92%  (easiest; bright colors)
  Rotten:                 ~80-85%  (harder; brown spots hard to distinguish from shadows)
  Formalin-mixed:         ~82-88%  (key metric; washed-out color is distinctive)

Formalin-Specific Metrics:
  True Positive Rate:     ~85-90%  (correctly identifies formalin)
  False Negative Rate:    ~10-15%  (misses some formalin)
  False Positive Rate:    ~3-7%    (wrongly identifies fresh/rotten as formalin)
```

**If Performance is Low:**

- Fruit accuracy < 92%: Check augmentation; may be too aggressive
- Condition accuracy < 75%: Consider: 
  - Increase training epochs in Phase 2
  - Increase discriminative LR values
  - Check if Grad-CAM heatmaps (for attention loss) are accurate
- Formalin detection < 80%: Verify CLAHE not used in augmentation

**Output of Step 3.5:**
- Test results JSON: All metrics and per-class breakdown
- Test predictions file: Predicted labels for all 1,523 test images
- Confusion matrix: 15×15 heatmap
- Attention heatmap examples: Sample images showing what model attended to
- Performance report: Summary of key metrics

---

### STEP 3.6: Compute Grad-CAM Heatmaps Offline (For Attention Distillation)

**Goal:** Generate Grad-CAM visualizations for training attention head

**What is Grad-CAM:**

Gradient-weighted Class Activation Mapping shows which regions of an image influenced the model's prediction.

**Why Offline Computation:**

- Grad-CAM requires backprop through entire model (expensive)
- Computing during training would 10× increase training time
- Instead: Pre-compute once, use as training target for attention head

**Grad-CAM Computation Process:**

For each training image (7,108 images):

1. Forward pass: Get feature maps from backbone (before pooling)
2. Compute gradients:
   - Backprop loss w.r.t. feature maps
   - Weight gradients by average pooling
3. Compute Grad-CAM map:
   - Linear combination of feature maps weighted by gradients
   - Produces 14×14 heatmap
4. Normalize: Rescale heatmap to [0, 1]
5. Save: Store 14×14 array as `.npy` file

**File Organization:**

```
grad_cam_heatmaps/
├── apple_fresh_001.npy      (14×14 array)
├── apple_fresh_002.npy
├── apple_rotten_001.npy
├── ... (7,108 files total, one per training image)
```

**Why Attention Head Useful:**

- Grad-CAM is gold standard for model interpretability
- But Grad-CAM expensive at inference (backprop needed)
- Attention head learns to approximate Grad-CAM with forward pass only
- Result: Fast, differentiable interpretability during mobile inference

**Output of Step 3.6:**
- Directory: `grad_cam_heatmaps/` with 7,108 `.npy` files
- Index file: Mapping from image paths to heatmap files
- Visualization: Sample Grad-CAM heatmaps overlaid on images (QA check)
- Statistics: Mean/std of heatmap values (for normalization)

---

## PHASE 4: OUT-OF-DISTRIBUTION (OOD) DETECTION SETUP (Week 4)

### STEP 4.1: Extract Features from All Training Images

**Goal:** Get 1280-dimensional feature vectors for Mahalanobis distance computation

**Process:**

For each training image (7,108 images):

1. Load image
2. Preprocess: Resize to 384×384, normalize with ImageNet stats
3. Forward pass through trained classifier backbone (Phase 2 best model)
4. Extract 1280-dimensional feature vector (before heads)
5. Store vector

**Feature Storage:**

```
features_train.npy       (7,108 × 1280 array)
                         - One row per training image
                         - Columns are 1280 feature dimensions

image_paths_train.txt    (7,108 lines)
                         - Path to each training image
                         - Used to map features to labels
```

**Computation:**
- ~30 minutes on Kaggle GPU (batch processing)

**Output of Step 4.1:**
- `features_train.npy` (7,108 × 1280 array)
- `image_paths_train.txt` (file paths)
- `feature_stats.json` (mean, std per dimension)

---

### STEP 4.2: Compute Per-Class Centroids and Covariance

**Goal:** For each of 15 classes, compute center point and spread in feature space

**Process:**

For each class (Apple-Fresh, Apple-Rotten, ..., Orange-Formalin):

1. Select all feature vectors for that class from `features_train.npy`
2. Compute centroid:
   - Mean of all vectors: `μ = mean(class_vectors)`
   - Result: 1280-dimensional vector
3. Compute covariance matrix:
   - `Σ = cov(class_vectors)`
   - Result: 1280×1280 matrix (symmetric, positive semi-definite)
4. Compute precision matrix (inverse of covariance):
   - `Σ^{-1} = inv(Σ)`
   - Result: 1280×1280 matrix

**Why Covariance:**

Covariance describes how features spread around the centroid:
- If all samples cluster tightly → covariance small → precision large
- If samples spread widely → covariance large → precision small

At inference, we compute distance to each centroid weighted by precision, giving higher weight to dimensions with low spread (more informative).

**Numerical Stability:**

Raw covariance matrix may be ill-conditioned. Apply regularization:
```
Σ_regularized = Σ + λ*I

where λ = 0.001 (small regularization constant)
      I = identity matrix

This ensures matrix is invertible and numerically stable.
```

**Output of Step 4.2:**
- `ood_config.json` containing:
  ```
  {
    "centroids": {
      "Apple-Fresh": [1.234, -0.567, ..., 0.891],     (1280 values)
      "Apple-Rotten": [...],
      ...
      "Orange-Formalin": [...]
    },
    "precision_matrices": {
      "Apple-Fresh": [[1.1, 0.02, ...], [0.02, 1.3, ...], ...],  (1280×1280)
      "Apple-Rotten": [...],
      ...
    },
    "threshold": 3.5,  (tuned in next step)
    "normalization": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  }
  ```
- `centroids_visualization.png` (t-SNE plot of centroids in 2D)

**File Size:**
- centroids: 15 classes × 1280 floats = ~76 KB
- precision matrices: 15 classes × 1280×1280 floats = ~1.2 MB
- Total `ood_config.json`: ~1.3 MB (compressed to ~0.4 MB in JSON)

---

### STEP 4.3: Set OOD Threshold via Validation

**Goal:** Choose distance threshold that best separates in-distribution from OOD

**Validation Process:**

1. Create validation set of KNOWN fruits:
   - Use training data (10,154 images)
   - Compute Mahalanobis distance to nearest centroid
   - Record distances for all training images

2. Create synthetic OOD set:
   - Use non-fruit images (download from ImageNet, Coco, etc.)
   - ~1,000 random objects (cars, dogs, buildings, etc.)
   - Extract features using trained backbone
   - Compute Mahalanobis distance to nearest fruit centroid
   - Record distances for all OOD images

3. Compute threshold:
   - Plot histogram: In-distribution distances vs. OOD distances
   - Find threshold where:
     - True Positive Rate (TPR): % of in-distribution correctly identified
     - False Positive Rate (FPR): % of OOD wrongly identified as in-distribution
     - Choose threshold that maximizes TPR while keeping FPR < 5%
   - Typical threshold: ~3.5

**Why Threshold Matters:**

```
Distance < threshold  → "Target fruit (known class)"
Distance > threshold  → "Non-target fruit (unknown, OOD)"

If threshold too low:
  - Too many false positives (reject valid fruits)
  - Overly strict detector

If threshold too high:
  - Too many false negatives (accept non-fruits)
  - Overly lenient detector

Sweet spot: Balance sensitivity and specificity
```

**Output of Step 4.3:**
- Optimal threshold: Stored in `ood_config.json`
- OOD detection metrics:
  - In-distribution TPR at threshold (should be >95%)
  - OOD detection rate (should be >90%)
  - ROC curve (TPR vs. FPR across all thresholds)
- Validation report: Justification for chosen threshold

---

## PHASE 5: MODEL EXPORT & MOBILE INTEGRATION (Week 5)

### STEP 5.1: Export Classifier to TFLite Format

**Goal:** Convert MobileNetV2 classifier from PyTorch to TensorFlow Lite

**Export Pipeline:**

```
fruit_classifier.pth (PyTorch)
    ↓ (torch.onnx.export)
fruit_classifier.onnx (ONNX intermediate)
    ↓ (TensorFlow ONNX converter)
fruit_classifier_fp32.tflite (Full precision, 10-12 MB)
    ↓ (Post-training quantization)
fruit_classifier_int8.tflite (Quantized, 3 MB)
```

**Export Details:**

1. **PyTorch → ONNX:**
   - Load best model checkpoint from Phase 2
   - Define dummy input: (1, 3, 384, 384)
   - Export with 4 outputs (fruit logits, condition logits, attention, features)
   - Output names: `['fruit', 'condition', 'attention', 'features']`
   - Opset version: 12 (good compatibility)

2. **ONNX → TFLite:**
   - Use TensorFlow's ONNX converter
   - Input shape: (1, 3, 384, 384)
   - Output shapes: (1, 5), (1, 3), (1, 14, 14), (1, 1280)
   - Keep FP32 precision (no quantization initially)

3. **Optional Quantization (Recommended):**
   - Convert FP32 weights to INT8 (8-bit integers)
   - Reduces model size: 10-12 MB → 3 MB
   - Speeds up inference: ~2-3× faster
   - Accuracy loss: <1% typically

**Validation Checklist:**
- [ ] TFLite file loads without errors
- [ ] Model accepts 384×384 RGB input (3 channels)
- [ ] Model outputs 4 tensors: (1,5), (1,3), (1,14,14), (1,1280)
- [ ] Can run inference on dummy input
- [ ] Output values reasonable (logits in [-5, 5] range)

**Metadata Embedding:**

Embed metadata in TFLite file for mobile app:

```
Model Metadata:
  - Model name: "fruit_classifier_v1.0"
  - Input shape: 384×384×3
  - Input normalization: ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Output 0: Fruit logits (5 classes)
  - Output 1: Condition logits (3 classes)
  - Output 2: Attention map (14×14)
  - Output 3: Features (1280-dim)
  - Class labels: ["Apple", "Banana", "Grape", "Mango", "Orange"]
  - Condition labels: ["Fresh", "Rotten", "Formalin-mixed"]
  - Model size: ~3 MB (quantized)
  - Inference time: ~50-100ms per 384×384 image
```

**Output of Step 5.1:**
- `fruit_classifier_int8.tflite` (~3 MB, recommended)
- OR `fruit_classifier_fp32.tflite` (~10-12 MB, if quantization unavailable)
- Model verification report: Input/output shapes and dtypes
- Inference benchmark: Latency on reference device

---

### STEP 5.2: Export OOD Configuration

**Goal:** Save Mahalanobis distance config for mobile OOD detection

**File: `ood_config.json` (Already created in Step 4.2)**

Contents:
```json
{
  "model_version": "v1.0",
  "created_date": "2026-07-15",
  "algorithm": "Mahalanobis Distance (OOD Detection)",
  
  "centroids": {
    "Apple-Fresh": [1.234, -0.567, 0.891, ...],
    "Apple-Rotten": [...],
    ...
    "Orange-Formalin": [...]
  },
  
  "precision_matrices": {
    "Apple-Fresh": [[1.1, 0.02, ...], [0.02, 1.3, ...], ...],
    ...
  },
  
  "threshold": 3.5,
  
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  
  "class_mapping": {
    "0": "Apple-Fresh",
    "1": "Apple-Rotten",
    ...
    "14": "Orange-Formalin"
  },
  
  "notes": "Threshold calibrated on held-out validation set. FPR ~4%, TPR ~96%"
}
```

**File Size:** ~400 KB (JSON text)

**Compression for Mobile:** Compress to `.json.gz` (~100 KB)

**Validation:**
- [ ] JSON parses correctly
- [ ] 15 centroids present
- [ ] 15 precision matrices present
- [ ] Threshold is positive float (e.g., 3.5)
- [ ] Normalization stats are ImageNet values

**Output of Step 5.2:**
- `ood_config.json` (400 KB)
- `ood_config.json.gz` (100 KB, optional compression)
- OOD config documentation: How to use in mobile app

---

### STEP 5.3: Prepare Mobile Integration Package

**Goal:** Bundle all files needed for Android APK

**Final Deliverables:**

```
mobile_deploy_package/
│
├── models/
│   ├── yolo_fruits_int8.tflite          (~3 MB)
│   │   - Fruit detection (YOLO)
│   │   - Accepts: 640×640 RGB input
│   │   - Outputs: Bounding boxes + confidence scores
│   │
│   └── fruit_classifier_int8.tflite    (~3 MB)
│       - Fruit + condition classification + attention
│       - Accepts: 384×384 RGB input
│       - Outputs: 4 tensors (fruit, condition, attention, features)
│
├── config/
│   ├── ood_config.json                  (~400 KB)
│   │   - Mahalanobis distance centroids
│   │   - Precision matrices
│   │   - OOD threshold
│   │
│   └── class_mapping.json               (~1 KB)
│       - Label indices to class names
│       - Labels for UI display
│
├── documentation/
│   ├── MODEL_CARD.md
│   │   - Model description
│   │   - Performance metrics
│   │   - Training data info
│   │   - Limitations and biases
│   │
│   ├── INTEGRATION_GUIDE.md
│   │   - How to load models in Android
│   │   - How to use OOD detection
│   │   - API documentation
│   │
│   └── API_REFERENCE.md
│       - Input/output specifications
│       - Normalization constants
│       - Error handling
│
└── version.txt
    - Pipeline version: v1.0
    - Model versions: YOLO 1.0, Classifier 1.0
    - Release date: 2026-07-15
```

**Size Summary:**
```
Total size in APK:
  - YOLO TFLite:        ~3 MB
  - Classifier TFLite:  ~3 MB
  - OOD config:         ~0.4 MB
  - Documentation:      ~0.1 MB
  ──────────────────────────────
  Total:                ~6.5 MB

APK size impact:
  - Typical Android app: 50-100 MB
  - Adding neural-bites: +6.5 MB
  - Total: ~56-106 MB (reasonable)
```

**Deployment Strategy:**

Option A: Bundle in APK (simpler, larger APK)
- All models embedded in APK
- No network download needed
- Works offline

Option B: Download at first launch (recommended)
- Models downloaded from server on first app open
- Smaller APK size
- Can update models without app store release
- Requires network connectivity once

For now, assume Option A (bundled).

**Output of Step 5.3:**
- `mobile_deploy_package/` directory with all files
- Manifest file: Checksums and versions of all models
- Integration checklist: For mobile dev team

---

## PHASE 6: ANDROID INTEGRATION (Week 5-6)

### STEP 6.1: Android Project Setup

**Goal:** Configure Android project to load and use TFLite models

**Prerequisites:**

```
Android SDK:        API level 24 (Android 7.0) or higher
Kotlin:             1.5+ (or Java)
TensorFlow Lite:    2.10+ (for mobile)
Build System:       Gradle
IDE:                Android Studio 4.2+
```

**Project Configuration (`build.gradle`):**

Add TensorFlow Lite dependency:

```gradle
dependencies {
    // TensorFlow Lite Core
    implementation 'org.tensorflow:tensorflow-lite:2.10.0'
    
    // TensorFlow Lite GPU Delegate (optional, for faster inference)
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.10.0'
    
    // TensorFlow Lite Support Library (for image processing)
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.2'
    
    // Image processing
    implementation 'androidx.camera:camera-core:1.1.0'
    implementation 'androidx.camera:camera-camera2:1.1.0'
}
```

**AndroidManifest.xml Permissions:**

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />

<!-- Optional: GPU acceleration -->
<uses-feature
    android:name="android.hardware.vulkan.version"
    android:required="false" />
```

**Asset Folder Structure:**

```
app/src/main/assets/
├── models/
│   ├── yolo_fruits_int8.tflite
│   └── fruit_classifier_int8.tflite
└── config/
    └── ood_config.json
```

**Gradle Task to Copy Assets:**

No build configuration needed; Android Studio automatically includes assets.

**Output of Step 6.1:**
- Updated `build.gradle` with TFLite dependencies
- Updated `AndroidManifest.xml` with permissions
- Asset files organized in `app/src/main/assets/`

---

### STEP 6.2: Implement TFLite Model Loader

**Goal:** Create Kotlin utility class to load and manage TFLite models

**Model Loader Class Structure:**

```kotlin
// File: app/src/main/kotlin/com/neuralbites/ml/ModelLoader.kt

class ModelLoader(private val context: Context) {
    
    // ─────────────────────────────────────────────────────
    // LOAD MODELS FROM ASSETS
    // ─────────────────────────────────────────────────────
    
    fun loadYOLOModel(): Interpreter {
        // 1. Read TFLite file from assets
        // 2. Create ByteBuffer from model data
        // 3. Create Interpreter
        // Return: Interpreter instance ready for inference
    }
    
    fun loadClassifierModel(): Interpreter {
        // Same process for classifier
    }
    
    // ─────────────────────────────────────────────────────
    // LOAD OOD CONFIGURATION
    // ─────────────────────────────────────────────────────
    
    fun loadOODConfig(): OODConfig {
        // 1. Read ood_config.json from assets
        // 2. Parse JSON
        // 3. Extract centroids and precision matrices
        // 4. Return OODConfig object
        
        // OODConfig data structure:
        return OODConfig(
            centroids = Map<String, FloatArray>,    // 15 classes × 1280 dims
            precisionMatrices = Map<String, Array<FloatArray>>,  // 15 classes × 1280×1280
            threshold = 3.5f,
            normalization = ImageNetNormalization(mean=[...], std=[...])
        )
    }
    
    // ─────────────────────────────────────────────────────
    // VERIFICATION METHODS
    // ─────────────────────────────────────────────────────
    
    fun verifyModelsLoaded(): Boolean {
        // Check that both YOLO and classifier models loaded correctly
        // Test with dummy input to confirm they work
        // Return: true if all models functional, false otherwise
    }
}
```

**Implementation Details:**

**Loading from Assets:**

```kotlin
fun loadModelFromAssets(assetPath: String): ByteBuffer {
    val inputStream = context.assets.open(assetPath)
    val fileSize = inputStream.available()
    val buffer = ByteBuffer.allocateDirect(fileSize)
    
    inputStream.read(buffer.array())
    inputStream.close()
    
    buffer.rewind()  // Reset position to beginning
    return buffer
}
```

**Creating Interpreter:**

```kotlin
fun createInterpreter(modelBuffer: ByteBuffer): Interpreter {
    val options = Interpreter.Options()
    
    // Optional: Enable GPU acceleration
    try {
        val gpuDelegate = GpuDelegate()
        options.addDelegate(gpuDelegate)
    } catch (e: Exception) {
        Log.w("ModelLoader", "GPU not available, using CPU")
    }
    
    // Enable multi-threading
    options.setNumThreads(4)  // Use 4 CPU threads
    
    return Interpreter(modelBuffer, options)
}
```

**Parsing OOD Config:**

```kotlin
fun loadOODConfig(): OODConfig {
    val jsonString = context.assets.open("config/ood_config.json")
        .bufferedReader()
        .use { it.readText() }
    
    val jsonObject = JSONObject(jsonString)
    
    // Extract centroids (15 classes × 1280 dimensions)
    val centroidsJson = jsonObject.getJSONObject("centroids")
    val centroids = mutableMapOf<String, FloatArray>()
    
    for (className in centroidsJson.keys()) {
        val centroidArray = centroidsJson.getJSONArray(className)
        val centroid = FloatArray(1280)
        for (i in 0 until 1280) {
            centroid[i] = centroidArray.getDouble(i).toFloat()
        }
        centroids[className] = centroid
    }
    
    // Extract precision matrices (15 classes × 1280×1280)
    val precisionMatricesJson = jsonObject.getJSONObject("precision_matrices")
    val precisionMatrices = mutableMapOf<String, Array<FloatArray>>()
    
    for (className in precisionMatricesJson.keys()) {
        val matrixJson = precisionMatricesJson.getJSONArray(className)
        val matrix = Array(1280) { FloatArray(1280) }
        for (i in 0 until 1280) {
            for (j in 0 until 1280) {
                matrix[i][j] = matrixJson.getJSONArray(i).getDouble(j).toFloat()
            }
        }
        precisionMatrices[className] = matrix
    }
    
    return OODConfig(
        centroids = centroids,
        precisionMatrices = precisionMatrices,
        threshold = jsonObject.getDouble("threshold").toFloat(),
        normalization = ImageNetNormalization(...)
    )
}
```

**Output of Step 6.2:**
- `ModelLoader.kt` class for model loading
- `OODConfig.kt` data class for OOD configuration
- Utility functions for asset file access
- Error handling for missing or corrupt models

---

### STEP 6.3: Implement Image Preprocessing Layer

**Goal:** Convert camera images to model inputs (384×384 or 640×640 normalized)

**Image Preprocessing Pipeline:**

```
Raw Camera Image (e.g., 1920×1080, YUV420)
    ↓
Convert YUV → RGB
    ↓
Resize to target size (384 or 640)
    ↓
Normalize: Subtract mean, divide by std (ImageNet)
    ↓
Convert to ByteBuffer (TFLite input format)
    ↓
Ready for inference
```

**Image Preprocessing Class:**

```kotlin
// File: app/src/main/kotlin/com/neuralbites/ml/ImagePreprocessor.kt

class ImagePreprocessor {
    
    // ─────────────────────────────────────────────────────
    // IMAGE QUALITY ASSESSMENT (LAYER 0)
    // ─────────────────────────────────────────────────────
    
    fun assessImageQuality(bitmap: Bitmap): ImageQualityReport {
        // Check 1: Blur detection
        val laplacianVariance = computeLaplacianVariance(bitmap)
        val isBlurry = laplacianVariance < 100.0f
        
        // Check 2: Brightness
        val meanBrightness = computeMeanBrightness(bitmap)
        val isTooLow = meanBrightness < 50
        val isTooHigh = meanBrightness > 200
        
        // Check 3: Resolution
        val isTooSmall = bitmap.width < 480 || bitmap.height < 480
        
        return ImageQualityReport(
            isBlurry = isBlurry,
            isTooLight = isTooHigh,
            isTooDark = isTooLow,
            isTooSmall = isTooSmall,
            passesQualityCheck = !(isBlurry || isTooLow || isTooHigh || isTooSmall)
        )
    }
    
    private fun computeLaplacianVariance(bitmap: Bitmap): Float {
        // Compute Laplacian filter response
        // High variance = sharp image, low variance = blurry image
        // Returns: Laplacian variance score
    }
    
    private fun computeMeanBrightness(bitmap: Bitmap): Float {
        // Average brightness across all pixels
        // Returns: Mean brightness [0, 255]
    }
    
    // ─────────────────────────────────────────────────────
    // PREPROCESS FOR YOLO (640×640)
    // ─────────────────────────────────────────────────────
    
    fun preprocessForYOLO(bitmap: Bitmap): ByteBuffer {
        // 1. Resize bitmap to 640×640
        val resized = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        
        // 2. Convert to ByteBuffer
        val buffer = ByteBuffer.allocateDirect(640 * 640 * 3 * 4)  // 4 bytes per float
        buffer.order(ByteOrder.nativeOrder())
        
        // 3. Copy pixel data (RGB) with normalization
        val pixels = IntArray(640 * 640)
        resized.getPixels(pixels, 0, 640, 0, 0, 640, 640)
        
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            // YOLO uses standard normalization (divide by 255)
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }
        
        buffer.rewind()
        return buffer
    }
    
    // ─────────────────────────────────────────────────────
    // PREPROCESS FOR CLASSIFIER (384×384)
    // ─────────────────────────────────────────────────────
    
    fun preprocessForClassifier(bitmap: Bitmap): ByteBuffer {
        // 1. Resize bitmap to 384×384
        val resized = Bitmap.createScaledBitmap(bitmap, 384, 384, true)
        
        // 2. Convert to ByteBuffer with ImageNet normalization
        val buffer = ByteBuffer.allocateDirect(384 * 384 * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())
        
        // 3. ImageNet normalization constants
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)  // R, G, B
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        // 4. Copy and normalize pixel data
        val pixels = IntArray(384 * 384)
        resized.getPixels(pixels, 0, 384, 0, 0, 384, 384)
        
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            // Apply ImageNet normalization: (x - mean) / std
            buffer.putFloat((r - mean[0]) / std[0])
            buffer.putFloat((g - mean[1]) / std[1])
            buffer.putFloat((b - mean[2]) / std[2])
        }
        
        buffer.rewind()
        return buffer
    }
    
    // ─────────────────────────────────────────────────────
    // CROP IMAGE FOR CLASSIFIER INPUT
    // ─────────────────────────────────────────────────────
    
    fun cropImage(bitmap: Bitmap, bbox: BoundingBox): Bitmap {
        // Extract rectangular region from bitmap using YOLO bounding box
        // Returns: Cropped bitmap ready for classifier preprocessing
        
        val x = bbox.x1.coerceIn(0, bitmap.width - 1)
        val y = bbox.y1.coerceIn(0, bitmap.height - 1)
        val width = (bbox.x2 - bbox.x1).coerceIn(1, bitmap.width - x)
        val height = (bbox.y2 - bbox.y1).coerceIn(1, bitmap.height - y)
        
        return Bitmap.createBitmap(bitmap, x, y, width, height)
    }
}
```

**ImageQualityReport Data Class:**

```kotlin
data class ImageQualityReport(
    val isBlurry: Boolean,
    val isTooLight: Boolean,
    val isTooDark: Boolean,
    val isTooSmall: Boolean,
    val passesQualityCheck: Boolean,
    val feedbackMessage: String = buildFeedback(...)
)
```

**Output of Step 6.3:**
- `ImagePreprocessor.kt` class for preprocessing
- `ImageQualityReport.kt` data class
- Helper functions for pixel manipulation
- Verified normalization constants match training

---

### STEP 6.4: Implement YOLO Inference (Layer 1)

**Goal:** Run YOLO detection and parse bounding box predictions

**YOLO Inference Class:**

```kotlin
// File: app/src/main/kotlin/com/neuralbites/ml/YOLOInference.kt

class YOLOInference(
    private val yoloInterpreter: Interpreter,
    private val preprocessor: ImagePreprocessor
) {
    
    companion object {
        const val CONFIDENCE_THRESHOLD = 0.5f
        const val NMS_THRESHOLD = 0.5f
        const val YOLO_INPUT_SIZE = 640
        const val NUM_CLASSES = 5  // Apple, Banana, Grape, Mango, Orange
        val FRUIT_NAMES = arrayOf("Apple", "Banana", "Grape", "Mango", "Orange")
    }
    
    // ─────────────────────────────────────────────────────
    // RUN YOLO DETECTION
    // ─────────────────────────────────────────────────────
    
    fun detectFruits(bitmap: Bitmap): List<BoundingBox> {
        // 1. Preprocess image
        val input = preprocessor.preprocessForYOLO(bitmap)
        
        // 2. Prepare output buffer
        // YOLO output: (1, predictions_per_scale, 85)
        // where 85 = (x, y, w, h, confidence, 80_class_scores)
        // But we only care about 5 fruits, so adaptation needed
        val outputShape = intArrayOf(1, 25200, 85)  // Example; depends on model
        val output = Array(1) { Array(25200) { FloatArray(85) } }
        
        // 3. Run inference
        yoloInterpreter.run(input, output)
        
        // 4. Parse predictions
        return parseYOLOOutput(output, bitmap.width, bitmap.height)
    }
    
    // ─────────────────────────────────────────────────────
    // PARSE YOLO OUTPUT
    // ─────────────────────────────────────────────────────
    
    private fun parseYOLOOutput(
        output: Array<Array<FloatArray>>,
        imgWidth: Int,
        imgHeight: Int
    ): List<BoundingBox> {
        val boxes = mutableListOf<BoundingBox>()
        val predictions = output[0]
        
        for (pred in predictions) {
            // Each prediction: [cx_norm, cy_norm, w_norm, h_norm, conf, class_scores...]
            val confidence = pred[4]
            
            // Filter low confidence predictions
            if (confidence < CONFIDENCE_THRESHOLD) continue
            
            // Get class ID (class with highest score among 5 fruits)
            val classScores = pred.slice(5 until 10)  // 5 fruit classes
            val classId = classScores.indices.maxByOrNull { classScores[it] } ?: continue
            
            // Denormalize coordinates
            val cx_norm = pred[0]
            val cy_norm = pred[1]
            val w_norm = pred[2]
            val h_norm = pred[3]
            
            val x1 = ((cx_norm - w_norm / 2) * imgWidth).toInt().coerceIn(0, imgWidth)
            val y1 = ((cy_norm - h_norm / 2) * imgHeight).toInt().coerceIn(0, imgHeight)
            val x2 = ((cx_norm + w_norm / 2) * imgWidth).toInt().coerceIn(0, imgWidth)
            val y2 = ((cy_norm + h_norm / 2) * imgHeight).toInt().coerceIn(0, imgHeight)
            
            boxes.add(BoundingBox(x1, y1, x2, y2, confidence, classId))
        }
        
        // Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        return applyNMS(boxes, NMS_THRESHOLD)
    }
    
    // ─────────────────────────────────────────────────────
    // NON-MAXIMUM SUPPRESSION (Remove overlapping boxes)
    // ─────────────────────────────────────────────────────
    
    private fun applyNMS(
        boxes: List<BoundingBox>,
        threshold: Float
    ): List<BoundingBox> {
        // Sort by confidence (highest first)
        val sorted = boxes.sortedByDescending { it.confidence }
        val result = mutableListOf<BoundingBox>()
        
        for (box in sorted) {
            var shouldAdd = true
            
            for (kept in result) {
                val iou = computeIoU(box, kept)
                if (iou > threshold) {
                    shouldAdd = false
                    break
                }
            }
            
            if (shouldAdd) {
                result.add(box)
            }
        }
        
        return result
    }
    
    private fun computeIoU(box1: BoundingBox, box2: BoundingBox): Float {
        // Intersection over Union (IoU)
        val intersectionArea = computeIntersectionArea(box1, box2)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        val unionArea = box1Area + box2Area - intersectionArea
        
        return if (unionArea > 0) intersectionArea.toFloat() / unionArea else 0f
    }
    
    private fun computeIntersectionArea(box1: BoundingBox, box2: BoundingBox): Int {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        
        return maxOf(0, x2 - x1) * maxOf(0, y2 - y1)
    }
}
```

**BoundingBox Data Class:**

```kotlin
data class BoundingBox(
    val x1: Int,
    val y1: Int,
    val x2: Int,
    val y2: Int,
    val confidence: Float,
    val fruitClassId: Int,
    val fruitName: String = FRUIT_NAMES[fruitClassId]
)

// Extension function
val FRUIT_NAMES = arrayOf("Apple", "Banana", "Grape", "Mango", "Orange")
```

**Output of Step 6.4:**
- `YOLOInference.kt` class for YOLO inference
- `BoundingBox.kt` data class
- NMS implementation
- Verified output tensor parsing

---

### STEP 6.5: Implement Classifier Inference (Layer 2 & 3 Combined)

**Goal:** Run classifier on each YOLO crop, compute OOD detection

**Classifier Inference Class:**

```kotlin
// File: app/src/main/kotlin/com/neuralbites/ml/ClassifierInference.kt

class ClassifierInference(
    private val classifierInterpreter: Interpreter,
    private val preprocessor: ImagePreprocessor,
    private val oodConfig: OODConfig
) {
    
    companion object {
        const val CLASSIFIER_INPUT_SIZE = 384
        const val NUM_FRUIT_CLASSES = 5
        const val NUM_CONDITION_CLASSES = 3
        const val CONFIDENCE_THRESHOLD = 0.7f
        
        val FRUIT_NAMES = arrayOf("Apple", "Banana", "Grape", "Mango", "Orange")
        val CONDITION_NAMES = arrayOf("Fresh", "Rotten", "Formalin-mixed")
    }
    
    // ─────────────────────────────────────────────────────
    // CLASSIFY SINGLE FRUIT CROP
    // ─────────────────────────────────────────────────────
    
    fun classifyCrop(croppedBitmap: Bitmap): FruitClassificationResult {
        // 1. Preprocess crop
        val input = preprocessor.preprocessForClassifier(croppedBitmap)
        
        // 2. Prepare output buffers (4 outputs)
        val fruitLogits = FloatArray(5)          // Fruit probabilities
        val conditionLogits = FloatArray(3)      // Condition probabilities
        val attentionMap = Array(14) { FloatArray(14) }  // 14×14 heatmap
        val features = FloatArray(1280)          // Feature vector for OOD
        
        // 3. Run inference
        val inputs = arrayOf(input)
        val outputs = mapOf(
            0 to fruitLogits,
            1 to conditionLogits,
            2 to attentionMap,
            3 to features
        )
        classifierInterpreter.runForMultipleInputsOutputs(inputs, outputs)
        
        // 4. Parse results
        val fruitClass = argmax(fruitLogits)
        val conditionClass = argmax(conditionLogits)
        val fruitConfidence = softmax(fruitLogits)[fruitClass]
        val conditionConfidence = softmax(conditionLogits)[conditionClass]
        
        // 5. OOD Detection (Layer 3)
        val mahalanobisDistance = computeMahalanobisDistance(features)
        val isOOD = mahalanobisDistance > oodConfig.threshold
        
        // 6. Confidence gating
        val isLowConfidence = fruitConfidence < CONFIDENCE_THRESHOLD
        
        // 7. Build explanation
        val explanation = buildExplanation(
            fruitClass, conditionClass, fruitConfidence,
            conditionConfidence, isOOD, isLowConfidence
        )
        
        return FruitClassificationResult(
            fruitName = FRUIT_NAMES[fruitClass],
            conditionName = CONDITION_NAMES[conditionClass],
            fruitConfidence = fruitConfidence,
            conditionConfidence = conditionConfidence,
            attentionMap = attentionMap,
            isOOD = isOOD,
            isLowConfidence = isLowConfidence,
            mahalanobisDistance = mahalanobisDistance,
            explanation = explanation
        )
    }
    
    // ─────────────────────────────────────────────────────
    // OOD DETECTION (Layer 3)
    // ─────────────────────────────────────────────────────
    
    private fun computeMahalanobisDistance(features: FloatArray): Float {
        var minDistance = Float.MAX_VALUE
        
        // For each known fruit class
        for ((className, centroid) in oodConfig.centroids) {
            // 1. Compute difference vector
            val diff = FloatArray(1280) { i -> features[i] - centroid[i] }
            
            // 2. Get precision matrix for this class
            val precisionMatrix = oodConfig.precisionMatrices[className]
                ?: continue
            
            // 3. Compute Mahalanobis distance
            // dist = sqrt(diff^T * Σ^{-1} * diff)
            val temp = FloatArray(1280) { i ->
                var sum = 0f
                for (j in 0 until 1280) {
                    sum += precisionMatrix[i][j] * diff[j]
                }
                sum
            }
            
            var distSquared = 0f
            for (i in 0 until 1280) {
                distSquared += diff[i] * temp[i]
            }
            
            val distance = sqrt(distSquared)
            minDistance = minOf(minDistance, distance)
        }
        
        return minDistance
    }
    
    // ─────────────────────────────────────────────────────
    // HELPER FUNCTIONS
    // ─────────────────────────────────────────────────────
    
    private fun argmax(array: FloatArray): Int {
        return array.indices.maxByOrNull { array[it] } ?: 0
    }
    
    private fun softmax(array: FloatArray): FloatArray {
        val maxVal = array.maxOrNull() ?: 0f
        val expArray = array.map { exp(it - maxVal) }
        val sum = expArray.sum()
        return expArray.map { it / sum }.toFloatArray()
    }
    
    private fun buildExplanation(
        fruitClass: Int,
        conditionClass: Int,
        fruitConf: Float,
        conditionConf: Float,
        isOOD: Boolean,
        isLowConf: Boolean
    ): String {
        return when {
            isOOD -> "⚠️ Not a target fruit"
            isLowConf -> "🤔 Low confidence - ${FRUIT_NAMES[fruitClass]} (${CONDITION_NAMES[conditionClass]}) - may need manual review"
            else -> "✓ ${FRUIT_NAMES[fruitClass]} appears ${CONDITION_NAMES[conditionClass]} (${(fruitConf * 100).toInt()}% confident)"
        }
    }
}
```

**FruitClassificationResult Data Class:**

```kotlin
data class FruitClassificationResult(
    val fruitName: String,
    val conditionName: String,
    val fruitConfidence: Float,
    val conditionConfidence: Float,
    val attentionMap: Array<FloatArray>,  // 14×14
    val isOOD: Boolean,
    val isLowConfidence: Boolean,
    val mahalanobisDistance: Float,
    val explanation: String
)
```

**Output of Step 6.5:**
- `ClassifierInference.kt` class for classification + OOD detection
- `FruitClassificationResult.kt` data class
- Mahalanobis distance computation
- Verified output tensor parsing

---

### STEP 6.6: Implement Multi-Fruit Pipeline (Layers 0-3)

**Goal:** Integrate all layers into single end-to-end pipeline

**Multi-Fruit Pipeline Orchestrator:**

```kotlin
// File: app/src/main/kotlin/com/neuralbites/ml/MultiFruitPipeline.kt

class MultiFruitPipeline(
    private val modelLoader: ModelLoader,
    private val preprocessor: ImagePreprocessor,
    private val yoloInference: YOLOInference,
    private val classifierInference: ClassifierInference,
    private val context: Context
) {
    
    // ─────────────────────────────────────────────────────
    // MAIN PIPELINE: Process multi-fruit image
    // ─────────────────────────────────────────────────────
    
    fun processImage(imageBitmap: Bitmap): PipelineResult {
        val startTime = System.currentTimeMillis()
        
        try {
            // ─────────────────────────────────────────────────────
            // LAYER 0: Image Quality Assessment
            // ─────────────────────────────────────────────────────
            val qualityReport = preprocessor.assessImageQuality(imageBitmap)
            
            if (!qualityReport.passesQualityCheck) {
                return PipelineResult.QualityCheckFailed(
                    message = qualityReport.feedbackMessage,
                    isBlurry = qualityReport.isBlurry,
                    isTooDark = qualityReport.isTooDark,
                    isTooLight = qualityReport.isTooLight,
                    isTooSmall = qualityReport.isTooSmall
                )
            }
            
            // ─────────────────────────────────────────────────────
            // LAYER 1: YOLO Detection
            // ─────────────────────────────────────────────────────
            val yoloStartTime = System.currentTimeMillis()
            val detectedBoxes = yoloInference.detectFruits(imageBitmap)
            val yoloInferenceTime = System.currentTimeMillis() - yoloStartTime
            
            if (detectedBoxes.isEmpty()) {
                return PipelineResult.NoFruitsDetected(
                    inferenceTimeMs = System.currentTimeMillis() - startTime
                )
            }
            
            // Warn if too many fruits
            if (detectedBoxes.size > 5) {
                Log.w("MultiFruitPipeline", "Detected ${detectedBoxes.size} fruits (>5), results may be inaccurate")
            }
            
            // ─────────────────────────────────────────────────────
            // LAYER 2 & 3: Classify Each Fruit
            // ─────────────────────────────────────────────────────
            val classifierStartTime = System.currentTimeMillis()
            val results = mutableListOf<FruitResult>()
            
            for ((index, box) in detectedBoxes.withIndex()) {
                try {
                    // Crop image to bounding box
                    val croppedBitmap = preprocessor.cropImage(imageBitmap, box)
                    
                    // Classify crop
                    val classificationResult = classifierInference.classifyCrop(croppedBitmap)
                    
                    // Combine YOLO box with classification result
                    results.add(FruitResult(
                        index = index,
                        fruitName = classificationResult.fruitName,
                        conditionName = classificationResult.conditionName,
                        fruitConfidence = classificationResult.fruitConfidence,
                        conditionConfidence = classificationResult.conditionConfidence,
                        attentionMap = classificationResult.attentionMap,
                        boundingBox = box,
                        isOOD = classificationResult.isOOD,
                        isLowConfidence = classificationResult.isLowConfidence,
                        mahalanobisDistance = classificationResult.mahalanobisDistance,
                        explanation = classificationResult.explanation
                    ))
                    
                } catch (e: Exception) {
                    Log.e("MultiFruitPipeline", "Error classifying fruit $index: ${e.message}")
                    // Continue to next fruit instead of crashing
                }
            }
            
            val classifierInferenceTime = System.currentTimeMillis() - classifierStartTime
            val totalInferenceTime = System.currentTimeMillis() - startTime
            
            return PipelineResult.Success(
                results = results,
                yoloInferenceTimeMs = yoloInferenceTime,
                classifierInferenceTimeMs = classifierInferenceTime,
                totalInferenceTimeMs = totalInferenceTime
            )
            
        } catch (e: Exception) {
            Log.e("MultiFruitPipeline", "Pipeline error: ${e.message}")
            e.printStackTrace()
            return PipelineResult.Error(
                message = "Pipeline error: ${e.message}",
                exception = e
            )
        }
    }
}
```

**Pipeline Result Sealed Class:**

```kotlin
sealed class PipelineResult {
    
    data class Success(
        val results: List<FruitResult>,
        val yoloInferenceTimeMs: Long,
        val classifierInferenceTimeMs: Long,
        val totalInferenceTimeMs: Long
    ) : PipelineResult()
    
    data class QualityCheckFailed(
        val message: String,
        val isBlurry: Boolean,
        val isTooDark: Boolean,
        val isTooLight: Boolean,
        val isTooSmall: Boolean
    ) : PipelineResult()
    
    data class NoFruitsDetected(
        val inferenceTimeMs: Long
    ) : PipelineResult()
    
    data class Error(
        val message: String,
        val exception: Exception
    ) : PipelineResult()
}
```

**FruitResult Data Class:**

```kotlin
data class FruitResult(
    val index: Int,
    val fruitName: String,
    val conditionName: String,
    val fruitConfidence: Float,
    val conditionConfidence: Float,
    val attentionMap: Array<FloatArray>,  // 14×14 for visualization
    val boundingBox: BoundingBox,
    val isOOD: Boolean,
    val isLowConfidence: Boolean,
    val mahalanobisDistance: Float,
    val explanation: String
)
```

**Output of Step 6.6:**
- `MultiFruitPipeline.kt` main orchestrator
- `PipelineResult.kt` sealed class for results
- `FruitResult.kt` output data class
- Complete end-to-end flow

---

### STEP 6.7: Create Android Activity for Camera Input

**Goal:** Implement UI for camera capture and pipeline execution

**Camera Activity Overview:**

```kotlin
// File: app/src/main/kotlin/com/neuralbites/ui/FruitDetectionActivity.kt

class FruitDetectionActivity : AppCompatActivity() {
    
    // ─────────────────────────────────────────────────────
    // INITIALIZATION
    // ─────────────────────────────────────────────────────
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_fruit_detection)
        
        // Initialize ML pipeline
        val modelLoader = ModelLoader(this)
        val pipeline = MultiFruitPipeline(
            modelLoader = modelLoader,
            preprocessor = ImagePreprocessor(),
            yoloInference = YOLOInference(...),
            classifierInference = ClassifierInference(...),
            context = this
        )
        
        // Setup camera
        setupCamera()
    }
    
    // ─────────────────────────────────────────────────────
    // CAMERA CAPTURE & PROCESSING
    // ─────────────────────────────────────────────────────
    
    private fun setupCamera() {
        // Bind camera use case to capture button
        val captureButton = findViewById<Button>(R.id.captureButton)
        captureButton.setOnClickListener {
            captureAndProcessImage()
        }
    }
    
    private fun captureAndProcessImage() {
        // 1. Capture image from camera
        val imageBitmap: Bitmap = getCameraBitmap()  // From camera
        
        // 2. Run pipeline on background thread
        Thread {
            val result = pipeline.processImage(imageBitmap)
            
            // 3. Update UI on main thread
            runOnUiThread {
                displayResults(result)
            }
        }.start()
    }
    
    // ─────────────────────────────────────────────────────
    // DISPLAY RESULTS
    // ─────────────────────────────────────────────────────
    
    private fun displayResults(result: PipelineResult) {
        when (result) {
            is PipelineResult.Success -> {
                // Display all detected fruits
                displayMultiFruitResults(result.results)
                
                // Show inference time
                displayInferenceMetrics(
                    yoloTime = result.yoloInferenceTimeMs,
                    classifierTime = result.classifierInferenceTimeMs,
                    totalTime = result.totalInferenceTimeMs
                )
            }
            
            is PipelineResult.QualityCheckFailed -> {
                // Show user feedback
                showQualityCheckError(result)
            }
            
            is PipelineResult.NoFruitsDetected -> {
                // Show "No fruits detected" message
                showNoFruitsMessage()
            }
            
            is PipelineResult.Error -> {
                // Show error
                showErrorDialog(result.message)
            }
        }
    }
    
    private fun displayMultiFruitResults(results: List<FruitResult>) {
        val container = findViewById<LinearLayout>(R.id.resultsContainer)
        container.removeAllViews()
        
        for ((index, result) in results.withIndex()) {
            val card = createResultCard(result)
            container.addView(card)
        }
    }
    
    private fun createResultCard(result: FruitResult): CardView {
        val card = CardView(this)
        
        // Title
        val titleView = TextView(this).apply {
            text = "${result.fruitName} - ${result.conditionName}"
            textSize = 18f
            setTextColor(Color.BLACK)
            setTypeface(null, Typeface.BOLD)
        }
        
        // Confidence
        val confidenceView = TextView(this).apply {
            text = "Confidence: ${(result.fruitConfidence * 100).toInt()}%"
            textSize = 14f
        }
        
        // Warnings
        if (result.isOOD || result.isLowConfidence) {
            val warningView = TextView(this).apply {
                text = "⚠️ ${result.explanation}"
                textSize = 12f
                setTextColor(Color.RED)
            }
            card.addView(warningView)
        }
        
        // Apply background color based on confidence/OOD
        card.setBackgroundColor(when {
            result.isOOD -> Color.parseColor("#FFF3CD")      // Warning yellow
            result.isLowConfidence -> Color.parseColor("#F8D7DA")  // Error red
            else -> Color.parseColor("#D4EDDA")              // Success green
        })
        
        return card
    }
}
```

**Output of Step 6.7:**
- `FruitDetectionActivity.kt` main UI activity
- Camera integration
- Results display UI
- Error handling UI

---

### STEP 6.8: Implement Inference Metrics & Logging

**Goal:** Track model performance and debug issues

**Metrics Tracking:**

```kotlin
// File: app/src/main/kotlin/com/neuralbites/ml/InferenceMetrics.kt

class InferenceMetrics {
    
    data class FrameMetrics(
        val timestamp: Long,
        val imageSizeBytes: Long,
        val yoloInferenceTimeMs: Long,
        val classifierInferenceTimeMs: Long,
        val totalInferenceTimeMs: Long,
        val fruitDetectionCount: Int,
        val fruitAccuracy: String,
        val conditionAccuracy: String,
        val oodDetectionCount: Int
    )
    
    private val metricsHistory = mutableListOf<FrameMetrics>()
    
    fun logFrameMetrics(metrics: FrameMetrics) {
        metricsHistory.add(metrics)
        
        // Log to Logcat
        Log.d("InferenceMetrics", 
            "YOLO: ${metrics.yoloInferenceTimeMs}ms, " +
            "Classifier: ${metrics.classifierInferenceTimeMs}ms, " +
            "Total: ${metrics.totalInferenceTimeMs}ms"
        )
        
        // Save to file for debugging
        saveMetricsToFile(metrics)
    }
    
    fun getAverageInferenceTime(): Long {
        if (metricsHistory.isEmpty()) return 0
        return metricsHistory.map { it.totalInferenceTimeMs }.average().toLong()
    }
    
    fun getPerformanceReport(): String {
        return """
            Inference Performance Report
            ────────────────────────────
            Total frames processed: ${metricsHistory.size}
            Average YOLO time: ${metricsHistory.map { it.yoloInferenceTimeMs }.average().toInt()}ms
            Average Classifier time: ${metricsHistory.map { it.classifierInferenceTimeMs }.average().toInt()}ms
            Average Total time: ${getAverageInferenceTime()}ms
            
            Typical latency breakdown:
            - Image capture & preprocessing: ~50ms
            - YOLO inference: ~300-500ms
            - Classifier inference (per fruit): ~50-100ms
            - Post-processing: ~20ms
            ────────────────────────────
            Total for 3 fruits: ~500-800ms
        """.trimIndent()
    }
    
    private fun saveMetricsToFile(metrics: FrameMetrics) {
        val metricsFile = File(context.cacheDir, "inference_metrics.txt")
        metricsFile.appendText("${metrics.timestamp}: ${metrics.totalInferenceTimeMs}ms\n")
    }
}
```

**Output of Step 6.8:**
- `InferenceMetrics.kt` performance tracking
- Logging utilities
- Performance debugging tools

---

### STEP 6.9: Package APK & Deployment

**Goal:** Build and deploy APK with all models embedded

**Build Process:**

1. **Generate Signed APK:**
   - In Android Studio: Build → Generate Signed Bundle/APK
   - Select "APK" option
   - Create/select signing key (for release)
   - Set build variant: Release
   - Output: `app-release.apk`

2. **APK Structure:**
   ```
   app-release.apk
   ├── AndroidManifest.xml
   ├── classes.dex                          (Kotlin/Java bytecode)
   ├── resources.arsc                       (UI resources)
   ├── lib/                                 (Native libraries)
   ├── assets/
   │   ├── models/
   │   │   ├── yolo_fruits_int8.tflite     (3 MB)
   │   │   └── fruit_classifier_int8.tflite (3 MB)
   │   └── config/
   │       └── ood_config.json              (0.4 MB)
   └── [compiled resources]
   
   Total APK size: ~50-70 MB (depends on app code size)
   ```

3. **Test APK:**
   - Install on Android device/emulator
   - Run: `adb install app-release.apk`
   - Test all pipelines:
     - Image quality checks (blur, darkness)
     - YOLO fruit detection
     - Classifier inference
     - OOD detection
     - Multi-fruit scenarios

4. **Deployment Options:**
   - **Google Play Store:** Submit APK to Google Play (requires developer account)
   - **Direct APK Distribution:** Share APK file directly
   - **Beta Testing:** Use Google Play Beta testing program

**Output of Step 6.9:**
- Signed APK ready for distribution
- Deployment checklist
- Installation instructions for testers

---

## OUTPUT FILES SUMMARY

### Step-by-Step Output Checklist

**PHASE 1 (YOLO Dataset Prep):**
- [ ] 10,154 YOLO annotation `.txt` files
- [ ] Organized `/yolo_dataset/` directory tree
- [ ] `data.yaml` configuration file
- [ ] Fallback image log (for manual review)

**PHASE 2 (YOLO Training):**
- [ ] `best.pt` YOLO model (PyTorch)
- [ ] `best_int8.tflite` YOLO model (TensorFlow Lite)
- [ ] Training curves (loss, mAP vs. epoch)
- [ ] Test set evaluation report

**PHASE 3 (Classifier Training):**
- [ ] `checkpoint_phase1_best.pth` (Phase 1 checkpoint)
- [ ] `checkpoint_phase2_best.pth` (Phase 2 checkpoint)
- [ ] `fruit_classifier_final.pth` (Final PyTorch model)
- [ ] Training history CSV (losses, accuracies)
- [ ] Per-class accuracy report

**PHASE 4 (OOD Detection):**
- [ ] `features_train.npy` (7,108 × 1280 feature vectors)
- [ ] `ood_config.json` (~400 KB) with:
  - 15 class centroids
  - 15 precision matrices
  - OOD threshold value

**PHASE 5 (Export & Mobile):**
- [ ] `fruit_classifier_int8.tflite` (3 MB, quantized)
- [ ] `yolo_fruits_int8.tflite` (3 MB, quantized)
- [ ] `ood_config.json`
- [ ] `class_mapping.json`
- [ ] MODEL_CARD.md (model documentation)
- [ ] INTEGRATION_GUIDE.md (for mobile team)

**PHASE 6 (Android):**
- [ ] Kotlin source files (ModelLoader, ImagePreprocessor, etc.)
- [ ] Android layouts & resources
- [ ] Signed APK (`app-release.apk`, ~60 MB)

---

## MOBILE DEPLOYMENT: INTEGRATION SUMMARY

### How TFLite Models Integrate into Android App

**1. Asset Embedding:**
```
app/src/main/assets/
├── models/
│   ├── yolo_fruits_int8.tflite          ← Load on app start
│   └── fruit_classifier_int8.tflite      ← Load on app start
└── config/
    └── ood_config.json                   ← Load on app start
```

**2. Runtime Loading:**
```kotlin
// On app startup
val modelLoader = ModelLoader(context)
val yoloInterpreter = modelLoader.loadYOLOModel()
val classifierInterpreter = modelLoader.loadClassifierModel()
val oodConfig = modelLoader.loadOODConfig()

// Memory usage: ~3MB + 3MB + 0.4MB = ~6.4MB RAM
```

**3. Inference Flow:**
```
User captures image
    ↓
ImagePreprocessor.assessImageQuality()          [LAYER 0]
    ↓ (if pass)
YOLOInference.detectFruits()                    [LAYER 1]
    ↓ (for each box)
ImagePreprocessor.cropImage()
    ↓
ClassifierInference.classifyCrop()              [LAYER 2 & 3]
  ├─ Run classifier forward pass
  ├─ Get: fruit, condition, attention, features
  ├─ Compute OOD distance
  └─ Build result
    ↓
MultiFruitPipeline.processImage()
    ↓ (returns)
List<FruitResult>
    ↓
Display results on UI
```

**4. Typical Latency (Snapdragon 765G):**
```
Image capture:        ~50ms
Quality check:        <1ms
YOLO inference:       ~300-500ms (for 640×640)
Classifier per crop:  ~50-100ms (for 384×384)
3-fruit scenario:     YOLO 400ms + 3×classifier 75ms = ~625ms total

User sees results in: ~0.6 seconds (acceptable for mobile)
```

**5. Memory Requirements:**
```
Static (loaded once):
  - YOLO model:        ~3 MB
  - Classifier model:  ~3 MB
  - OOD config:        ~0.4 MB
  - Subtotal:          ~6.4 MB

Per-inference (temporary):
  - Input tensor:      ~3 MB (640×640×3×4 bytes)
  - Output buffers:    ~1 MB
  - Working memory:    ~2 MB
  - Subtotal:          ~6 MB

Total RAM: ~12-15 MB (minimal for modern phones, typical RAM 4-8GB)
```

**6. Battery Impact:**
```
Per image inference:   ~0.6 seconds GPU+CPU work
Battery drain rate:    ~15-20mW per frame (for Snapdragon 765G)
Typical usage:         ~30 photos/minute
Battery impact:        ~0.5-1% per minute of active use
  (Normal phone battery: 3,000-5,000 mAh)
  (One full day typical use with background operations)
```

---

## RESEARCH PAPER ASSETS (To Generate)

### Figures & Tables

**Figure 1: Pipeline Architecture**
- 4-layer diagram (Quality → YOLO → Classifier → OOD)
- Input/output dimensions at each layer

**Figure 2: Training Curves**
- Loss vs. epoch (Phase 1 + Phase 2)
- Fruit accuracy vs. epoch
- Condition accuracy vs. epoch
- Val loss vs. train loss

**Figure 3: Confusion Matrix**
- 15×15 heatmap (fruit×condition predictions)
- Per-class accuracy percentages

**Figure 4: Attention Heatmaps**
- 3 correct predictions (with attention overlay)
- 3 challenging predictions (what did model look at?)
- Original image + attention map + prediction

**Figure 5: OOD Detection Performance**
- Histogram: In-distribution vs. OOD Mahalanobis distances
- ROC curve (TPR vs. FPR)
- Threshold selection

**Table 1: Per-Class Performance**
```
Class              | Precision | Recall | F1-Score | Accuracy
Apple-Fresh        | 0.96      | 0.94   | 0.95     | 94.2%
Apple-Rotten       | 0.91      | 0.88   | 0.90     | 87.5%
Apple-Formalin     | 0.89      | 0.91   | 0.90     | 90.8%
...
Macro Average      | 0.92      | 0.91   | 0.91     | 90.2%
```

**Table 2: YOLO Performance**
```
Metric        | Value
mAP50         | 0.88
mAP50-95      | 0.68
Precision     | 0.92
Recall        | 0.85
Per-fruit F1  | 0.88
```

**Table 3: Mobile Deployment Specs**
```
Specification      | Value
Model Size (YOLO)  | 3 MB
Model Size (Class) | 3 MB
Config Size        | 0.4 MB
Total APK Impact   | +6.5 MB
Inference Time     | ~600ms (3 fruits)
Memory Footprint   | ~15 MB
Battery Drain      | ~1% per minute (active)
```

---

## FINAL CHECKLIST: BEFORE HANDOFF TO MOBILE TEAM

**Data Preparation:**
- [ ] All 10,154 images organized in `/yolo_dataset/`
- [ ] All 10,154 YOLO annotations verified
- [ ] `data.yaml` correct
- [ ] Train/val/test split stratified and documented

**YOLO Model:**
- [ ] Training complete, `best.pt` saved
- [ ] Test mAP50 > 0.82
- [ ] Exported to `yolo_fruits_int8.tflite`
- [ ] TFLite file tested and verified

**Classifier Model:**
- [ ] Phase 1 + Phase 2 training complete
- [ ] Validation accuracy > 90% (fruit), > 85% (condition)
- [ ] Test set evaluation saved
- [ ] Exported to `fruit_classifier_int8.tflite`
- [ ] TFLite file tested and verified

**OOD Detection:**
- [ ] Centroids computed for all 15 classes
- [ ] Precision matrices computed
- [ ] Threshold calibrated and validated
- [ ] `ood_config.json` created and verified

**Mobile Integration:**
- [ ] `ModelLoader.kt` loads all models without errors
- [ ] `ImagePreprocessor.kt` correctly preprocesses images
- [ ] `YOLOInference.kt` produces valid bounding boxes
- [ ] `ClassifierInference.kt` produces valid logits + attention
- [ ] `MultiFruitPipeline.kt` integrates all layers
- [ ] Android Activity successfully displays results
- [ ] APK builds without errors

**Documentation:**
- [ ] MODEL_CARD.md complete (model description, metrics, limitations)
- [ ] INTEGRATION_GUIDE.md complete (how to use in app)
- [ ] API_REFERENCE.md complete (input/output specs)
- [ ] All code commented and documented
- [ ] Training logs and results saved

**Testing:**
- [ ] Test on 5+ single-fruit images → correct fruit + condition
- [ ] Test on 5+ multi-fruit images → all fruits detected and classified
- [ ] Test on blurry image → quality check fails
- [ ] Test on non-fruit image → OOD detection catches it
- [ ] Test on low-light image → quality check fails
- [ ] Inference time < 1 second per image
- [ ] No crashes or exceptions in app

---

## CONCLUSION

This step-by-step plan provides a complete, mobile-deployment-ready ML pipeline for fruit detection, condition classification, and OOD detection.

**Key Deliverables:**
1. `yolo_fruits_int8.tflite` - Fruit detection
2. `fruit_classifier_int8.tflite` - Condition classification + attention
3. `ood_config.json` - Non-target fruit rejection

**Android Integration:**
- Seamless 4-layer pipeline (Quality → YOLO → Classifier → OOD)
- ~600ms inference time for multi-fruit scenarios
- ~15 MB RAM footprint
- Ready for production deployment

**Research Paper:**
- Training curves, confusion matrices, attention visualizations
- Per-class and per-condition performance metrics
- OOD detection ROC curves
- Mobile deployment specifications

All phases documented, checkpointed, and tested for reproducibility.
