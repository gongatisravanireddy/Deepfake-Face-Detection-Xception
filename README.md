\# Deepfake Face Detection using Xception



\## Project Overview



This project is an end-to-end deepfake face detection system that classifies an input face image as \*\*REAL\*\* or \*\*FAKE\*\*.



The project uses an \*\*ImageNet-pretrained Xception model\*\* with transfer learning, image preprocessing and augmentation, model evaluation, and a Flask web application for interactive image prediction.



\## Problem Statement



The increasing use of generative AI and deepfake technology makes it difficult to distinguish manipulated images from authentic images.



This project aims to develop an automated deepfake detection system that can analyze a face image and classify it as REAL or FAKE.



\## Objectives



\- Build a deepfake image classification model.

\- Use transfer learning with the Xception architecture.

\- Preprocess and augment training images.

\- Evaluate the model using multiple classification metrics.

\- Provide a Flask web application for single-image prediction.

\- Display the prediction along with a confidence score.



\## Key Features



\- Automatic Kaggle dataset download using the Kaggle API

\- Automatic detection of dataset folder structure

\- Image preprocessing and augmentation

\- Xception transfer learning

\- Binary REAL/FAKE classification

\- Early stopping and model checkpointing

\- Learning-rate reduction on plateau

\- Accuracy, Precision, Recall and F1 evaluation

\- ROC-AUC analysis

\- Confusion matrix

\- Training and validation curves

\- Flask web application

\- Single-image prediction with confidence score



\## System Architecture



```text

Kaggle Dataset

&#x20;     ↓

Dataset Download

&#x20;     ↓

Data Preprocessing

&#x20;     ↓

Image Augmentation

&#x20;     ↓

Xception Transfer Learning

&#x20;     ↓

Model Training

&#x20;     ↓

Best Model Checkpoint

&#x20;     ↓

&#x20;┌───────────────┐

&#x20;↓               ↓

Evaluation      Prediction

&#x20;↓               ↓

Metrics          REAL / FAKE

&#x20;↓               ↓

Plots            Confidence

&#x20;     \\          /

&#x20;      ↓        ↓

&#x20;      Flask Web App

