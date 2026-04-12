wandb report :- https://wandb.ai/da25m021-iitm-indi/dl_assignment_2/reports/DL-assignment-2--VmlldzoxNjQ5OTQxNw?accessToken=r6ty642ojnjho4lukc2przmi33a3kd7qk90eyl1dunl4h508hvf4gsgu7a5flae0


Name :- Neer Patel

Roll :- DA25M021


#  DA6401 Assignment 2 

## 📌 Project Overview

This project focuses on building a **complete visual perception pipeline** using deep learning. Instead of solving isolated tasks, the goal is to design a **unified system** that can understand images at multiple levels simultaneously.

The pipeline performs three core computer vision tasks in a **single forward pass**:

* Image classification (pet breed recognition)
* Object localization (bounding box prediction)
* Semantic segmentation (pixel-wise labeling)

---

## 🎯 Problem Statement

Modern vision systems require the ability to:

* Identify *what* is in an image
* Locate *where* it is
* Understand *which pixels belong to it*

This assignment addresses the challenge of combining these capabilities into a **multi-task learning framework**. The key problem is to design a shared architecture that can efficiently learn and generalize across all three tasks without performance degradation due to task interference.

---

## 📊 Dataset

The project uses the **Oxford-IIIT Pet Dataset**, which provides rich annotations for multiple vision tasks:

* **37 pet breed classes** (for classification)
* **Bounding box annotations** (for localization)
* **Pixel-level trimaps** (for segmentation)

This dataset enables the development of a **multi-task pipeline** since all three types of labels are available for each image.

---

## 🏗️ Architecture

The system is built as a **unified deep learning model** with a shared backbone and multiple task-specific heads.

### 1. Backbone — VGG11 (Custom Implementation)

* Implemented from scratch using PyTorch
* Consists of stacked convolutional layers with max-pooling
* Enhanced with:

  * Batch Normalization
  * Custom Dropout layer (manually implemented)

This backbone acts as a **feature extractor**, learning hierarchical representations from images.

---

### 2. Classification Head

* Fully connected layers attached to the backbone
* Outputs probabilities over **37 pet breeds**
* Optimized using classification loss (e.g., Cross-Entropy)

---

### 3. Localization Head

* Regression head predicting:

  * `[x_center, y_center, width, height]`
* Built on top of shared features
* Uses a **custom IoU-based loss function**

This head enables the model to **localize the object** within the image.

---

### 4. Segmentation Head (U-Net Style Decoder)

* Symmetric decoder connected to the encoder (VGG11)
* Uses:

  * Transposed convolutions for upsampling
  * Skip connections for feature fusion

Produces a **pixel-wise segmentation mask** of the object.

---

### 5. Multi-Task Learning Framework

All three heads are trained jointly using a **combined loss function**:

* Classification loss
* Localization (IoU) loss
* Segmentation loss (e.g., Dice/BCE)

The model shares representations across tasks, improving efficiency and enabling **end-to-end learning**.

---

## 📈 Evaluation Metrics

The system is evaluated using task-specific metrics:

* **Classification:** Macro F1 Score
* **Localization:** Mean Average Precision (mAP)
* **Segmentation:** Dice Score

These metrics ensure a comprehensive evaluation of the pipeline’s performance.

---

## 🚀 Key Contributions

* End-to-end **multi-task visual perception system**
* Custom implementation of:

  * VGG11 architecture
  * Dropout layer
  * IoU loss function
* Integration of classification, detection, and segmentation into a **single unified model**

---

## 🏁 Conclusion

This project demonstrates how multiple computer vision tasks can be combined into a **cohesive pipeline**, highlighting the strengths and challenges of multi-task learning in deep neural networks. It provides practical experience in designing scalable architectures that mimic real-world perception systems.
