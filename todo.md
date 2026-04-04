# 🧠 DA6401 Assignment 2 — Complete TODO Guide

## 📌 Project Setup

* [ ] Clone official repo
* [ ] Setup virtual environment
* [ ] Install dependencies:

  * [ ] torch
  * [ ] numpy
  * [ ] matplotlib
  * [ ] scikit-learn
  * [ ] wandb
  * [ ] albumentations
* [ ] Setup folder structure:

  * [ ] `models/`
  * [ ] `datasets/`
  * [ ] `training/`
  * [ ] `utils/`
  * [ ] `losses/`
  * [ ] `notebooks/`
* [ ] Initialize W&B project

---

# 🧩 Task 1 — VGG11 Classification

## 🔧 Model Implementation

* [ ] Implement VGG11 from scratch

  * [ ] Conv Blocks (correct channels)
  * [ ] MaxPooling layers
  * [ ] Fully connected classifier
* [ ] Add BatchNorm layers
* [ ] Implement **Custom Dropout**

  * [ ] Inherit `torch.nn.Module`
  * [ ] Apply binary mask
  * [ ] Implement inverted scaling
  * [ ] Handle `self.training` flag correctly

## 🧪 Training

* [ ] Dataset preprocessing

  * [ ] Resize images
  * [ ] Normalize
  * [ ] Train/Val split (NO leakage)
* [ ] Train classification model
* [ ] Log:

  * [ ] Loss
  * [ ] Accuracy
  * [ ] Macro F1 Score

## 📝 Analysis

* [ ] Justify:

  * [ ] Dropout placement
  * [ ] BatchNorm placement

---

# 📦 Task 2 — Object Localization

## 🔧 Encoder Setup

* [ ] Extract convolutional backbone from VGG11
* [ ] Decide:

  * [ ] Freeze weights OR
  * [ ] Fine-tune weights
* [ ] Document reasoning

## 📐 Regression Head

* [ ] Design FC/Conv head
* [ ] Output:

  * [ ] `[x_center, y_center, width, height]`
* [ ] Apply proper activation (e.g., sigmoid if normalized)

## 📉 Custom IoU Loss

* [ ] Implement IoU Loss from scratch
* [ ] Ensure:

  * [ ] Numerical stability
  * [ ] Differentiability

## 🧪 Training

* [ ] Train localization model
* [ ] Log:

  * [ ] IoU
  * [ ] mAP

---

# 🧠 Task 3 — U-Net Segmentation

## 🔧 Architecture

* [ ] Use VGG11 as encoder
* [ ] Implement decoder:

  * [ ] Transposed Convolutions (NO interpolation)
  * [ ] Skip connections (concatenation)
* [ ] Ensure symmetry with encoder

## 📉 Loss Function

* [ ] Choose and implement:

  * [ ] Dice Loss / BCE / Combo
* [ ] Justify choice

## 🧪 Training

* [ ] Train segmentation model
* [ ] Log:

  * [ ] Dice Score
  * [ ] Pixel Accuracy

---

# 🔗 Task 4 — Multi-Task Pipeline

## 🔧 Unified Model

* [ ] Single backbone (shared encoder)
* [ ] Add 3 heads:

  * [ ] Classification head
  * [ ] Bounding box head
  * [ ] Segmentation decoder

## 🔁 Forward Pass

* [ ] Implement:

  ```python
  def forward(self, x):
      return class_logits, bbox, segmentation_mask
  ```

## ⚖️ Multi-task Loss

* [ ] Combine losses:

  * [ ] Classification Loss
  * [ ] IoU Loss
  * [ ] Segmentation Loss
* [ ] Tune loss weights

## 🧪 Evaluation

* [ ] Classification → Macro F1
* [ ] Detection → mAP
* [ ] Segmentation → Dice Score

---

# 🤖 Automated Checks Preparation

* [ ] Verify VGG11 layer dimensions
* [ ] Test Custom Dropout:

  * [ ] Different p values
  * [ ] Train vs Eval behavior
* [ ] Test IoU Loss on sample boxes
* [ ] Ensure forward pass works end-to-end

---

# 📊 W&B Report

## 1️⃣ Dropout vs BatchNorm

* [ ] Train with/without BatchNorm
* [ ] Plot activation distributions

## 2️⃣ Internal Dynamics

* [ ] Train:

  * [ ] No dropout
  * [ ] p=0.2
  * [ ] p=0.5
* [ ] Compare loss curves

## 3️⃣ Transfer Learning Study

* [ ] Run experiments:

  * [ ] Frozen backbone
  * [ ] Partial fine-tuning
  * [ ] Full fine-tuning
* [ ] Compare:

  * [ ] Dice Score
  * [ ] Training time
  * [ ] Stability

## 4️⃣ Feature Map Visualization

* [ ] Extract:

  * [ ] First conv layer outputs
  * [ ] Last conv layer outputs
* [ ] Visualize patterns

## 5️⃣ Detection Analysis

* [ ] Log 10 test images
* [ ] Overlay:

  * [ ] GT (Green)
  * [ ] Prediction (Red)
* [ ] Compute IoU
* [ ] Identify failure case

## 6️⃣ Segmentation Metrics

* [ ] Compare:

  * [ ] Pixel Accuracy
  * [ ] Dice Score
* [ ] Explain imbalance effect

## 7️⃣ Final Pipeline Demo

* [ ] Test on 3 external images
* [ ] Evaluate:

  * [ ] Generalization
  * [ ] Failure modes

## 8️⃣ Meta Analysis

* [ ] Plot all metrics
* [ ] Reflect on:

  * [ ] Architecture choices
  * [ ] Backbone strategy
  * [ ] Loss effectiveness
  * [ ] Task interference

---

# 🚀 Final Submission Checklist

* [ ] Code runs end-to-end
* [ ] No data leakage
* [ ] W&B report is PUBLIC
* [ ] Metrics logged correctly
* [ ] Clean, readable code
* [ ] Comments added where necessary

---

# 💡 Bonus (Optional but Helpful)

* [ ] Add gradient clipping
* [ ] Try learning rate schedulers
* [ ] Add augmentation (albumentations)
* [ ] Experiment with loss balancing

---

# 🏁 Status Tracker

* [ ] Task 1 ✅
* [ ] Task 2 ✅
* [ ] Task 3 ✅
* [ ] Task 4 ✅
* [ ] W&B Report ✅
* [ ] Final Submission ✅

---
