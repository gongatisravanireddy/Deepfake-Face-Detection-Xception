# Deepfake Face Detection using Xception

## Project Overview

This project is an end-to-end deepfake face detection system that classifies an input face image as **REAL** or **FAKE**.

The project uses an **ImageNet-pretrained Xception model** with transfer learning, image preprocessing and augmentation, model evaluation, and a Flask web application for interactive image prediction.

The project provides a complete pipeline starting from dataset acquisition and preprocessing to model training, evaluation, single-image prediction, and web-based deployment.

---

## Problem Statement

The increasing use of generative AI and deepfake technology makes it difficult to distinguish manipulated images from authentic images.

Deepfake images can create challenges in areas such as digital media, identity verification, content authenticity, and online trust.

This project aims to develop an automated deepfake detection system that can analyze a face image and classify it as **REAL** or **FAKE**.

---

## Objectives

- Build a deepfake image classification system.
- Use transfer learning with the Xception architecture.
- Preprocess and augment training images.
- Train a binary REAL/FAKE image classifier.
- Evaluate the trained model using multiple classification metrics.
- Generate visual evaluation results such as confusion matrix and ROC curve.
- Provide single-image prediction with a confidence score.
- Develop a Flask web application for interactive testing.

---

## Key Features

- Automatic Kaggle dataset download using the Kaggle API
- Automatic detection of dataset folder structure
- Train, validation, and test data pipelines
- Image resizing and pixel normalization
- Training image augmentation
- Xception transfer learning
- ImageNet-pretrained Xception backbone
- Binary REAL/FAKE classification
- Early stopping
- Model checkpointing
- Learning-rate reduction using `ReduceLROnPlateau`
- Accuracy evaluation
- Precision evaluation
- Recall evaluation
- F1 Score evaluation
- ROC-AUC analysis
- Classification report
- Confusion matrix
- Training and validation accuracy curves
- Training and validation loss curves
- Single-image prediction
- Confidence score generation
- Flask web application
- Image upload and validation
- Uploaded image display
- Error handling

---

## System Architecture

```text
Kaggle Dataset
       ↓
Dataset Download
       ↓
Data Preprocessing
       ↓
Image Augmentation
       ↓
Xception Transfer Learning
       ↓
Model Training
       ↓
Best Model Checkpoint
       ↓
 ┌───────────────┐
 ↓               ↓
Evaluation     Prediction
 ↓               ↓
Metrics        REAL / FAKE
 ↓               ↓
Plots          Confidence
 └───────┬───────┘
         ↓
   Flask Web App



