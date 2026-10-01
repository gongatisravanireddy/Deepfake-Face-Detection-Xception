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



Methodology
The project follows these major steps:
1. Download the deepfake image dataset from Kaggle using the Kaggle API.
2. Automatically identify the training, validation, and test directories.
3. Load the images using Keras ImageDataGenerator.
4. Resize images to 299 × 299 pixels.
5. Normalize pixel values between 0 and 1.
6. Apply data augmentation to training images.
7. Use an ImageNet-pretrained Xception model as the feature extractor.
8. Freeze the Xception backbone initially.
9. Add a custom classification head for binary classification.
10. Train the classification model using the Adam optimizer.
11. Use callbacks such as Early Stopping, Model Checkpointing, and ReduceLROnPlateau.
12. Save the best model checkpoint and training history.
13. Evaluate the trained model on the test dataset.
14. Calculate Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
15. Generate confusion matrix and ROC curve plots.
16. Use the trained model to predict REAL or FAKE for individual images.
17. Provide the prediction functionality through a Flask web application.
Dataset
The project uses a deepfake image dataset downloaded from Kaggle using the Kaggle API.
The dataset is not included in this repository because image datasets can contain a large number of files and require significant storage space.
The expected dataset structure is:
dataset/
├── train/
│   ├── real/
│   └── fake/
│
├── valid/
│   ├── real/
│   └── fake/
│
└── test/
    ├── real/
    └── fake/

The project automatically detects the dataset structure and also supports alternative validation folder names such as validation.
Dataset Classes
The system performs binary classification between:
- REAL — authentic face images
- FAKE — manipulated or deepfake face images
Data Preprocessing
Images are processed using Keras ImageDataGenerator.
Image Size
299 × 299

The images are resized to 299 × 299 because this is the input size used by the Xception architecture.
Batch Size
32

A batch size of 32 images is used during training, validation, and testing.
Pixel Normalization
Pixel values are normalized using:
1 / 255

This converts pixel values from the range:
0 - 255

to approximately:
0 - 1

Training Data Augmentation
The training pipeline applies the following augmentation techniques:
- Rotation
- Zoom
- Width shifting
- Height shifting
- Horizontal flipping
- Brightness adjustment
These transformations help the model learn from different variations of the training images.
Validation and Test Data
Validation and test images use normalization only.
No image augmentation is applied to validation and test data.
Xception Model
The project uses an ImageNet-pretrained Xception architecture for deepfake image classification.
Xception is used as the main feature extraction backbone.
The current model architecture is:
Input (299 × 299 × 3)
        ↓
Xception Backbone
(ImageNet pretrained and frozen)
        ↓
Global Average Pooling
        ↓
Dense (256, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (1, Sigmoid)
        ↓
REAL / FAKE

Model Architecture Explanation
Xception Backbone
The Xception model is loaded with ImageNet pretrained weights.
The backbone is initially frozen so that the pretrained feature representations can be used without updating the complete network during the initial training process.
Global Average Pooling
GlobalAveragePooling2D converts the feature maps generated by Xception into a compact feature representation.
Dense Layer
A dense layer with 256 neurons and ReLU activation is added for learning task-specific features.
Dropout
A dropout rate of 0.5 is used to reduce overfitting.
Output Layer
A single neuron with sigmoid activation is used for binary classification.
The output represents the probability used to classify the image as REAL or FAKE.
Model Configuration
The model is compiled using:
- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Binary Cross-Entropy
- Metric: Accuracy
- Maximum Epochs: 10
Binary Cross-Entropy
Binary cross-entropy is used because this is a binary classification problem with two classes:
REAL
FAKE

Training
The training pipeline is implemented in:
src/train.py

The training process includes:
- GPU availability checking
- Dataset loading
- Image generator creation
- Xception model creation
- Model compilation
- Model training
- Training history saving
- Best model checkpoint saving
- Class mapping saving
Early Stopping
EarlyStopping monitors validation loss and stops training when the validation performance stops improving.
The best weights are restored after training.
Model Checkpointing
ModelCheckpoint saves the best model based on validation accuracy.
The best model is saved as:
models/best_xception_deepfake.keras

Learning Rate Reduction
ReduceLROnPlateau reduces the learning rate when validation loss stops improving.
This helps the model continue learning when the optimization process reaches a plateau.
Training History
Training history is saved as:
models/training_history.json

Class Mapping
The mapping between class names and class indices is saved as:
models/class_indices.json

These generated files are created after running the training process.
Model Evaluation
The trained model is evaluated on the test dataset using multiple classification metrics.
The evaluation is implemented in:
src/evaluate.py

Evaluation Metrics
The following metrics are calculated:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Classification Report
- Confusion Matrix
Accuracy
Accuracy measures the overall percentage of correctly classified images.
Accuracy = Correct Predictions / Total Predictions

Precision
Precision measures how many images predicted as a particular class were actually from that class.
Recall
Recall measures how many actual images of a class were correctly identified.
F1 Score
F1 Score combines Precision and Recall into a single metric.
ROC-AUC
ROC-AUC measures the model's ability to distinguish between the REAL and FAKE classes across different classification thresholds.
Evaluation Outputs
The evaluation process generates the following plots:
results/
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
└── roc_curve.png

Accuracy Curve
The accuracy curve compares training accuracy and validation accuracy across epochs.
Loss Curve
The loss curve compares training loss and validation loss across epochs.
Confusion Matrix
The confusion matrix shows the number of correctly and incorrectly classified REAL and FAKE images.
ROC Curve
The ROC curve shows the relationship between the True Positive Rate and False Positive Rate.
Evaluation Results
Actual evaluation values should be added after running the trained model on the test dataset.
No evaluation values are hard-coded or manually created in this repository.
This ensures that reported performance metrics come from the actual trained model and test dataset.
Prediction
The prediction functionality is implemented in:
src/predict.py

The module accepts a single image and returns a prediction with a confidence score.
Example output:
Prediction: REAL
Confidence: XX%

or:
Prediction: FAKE
Confidence: XX%

Prediction Process
The input image goes through the following steps:
1. Load the image.
2. Resize the image to 299 × 299.
3. Convert the image into an array.
4. Normalize pixel values using 1 / 255.
5. Pass the image to the trained Xception model.
6. Generate the prediction probability.
7. Determine the predicted class.
8. Calculate the confidence score.
9. Return the prediction and confidence.
The model and class mapping are loaded when required.
Flask Web Application
The project includes a Flask web application for interactive deepfake detection.
The Flask application is implemented in:
app.py

The application allows users to upload a face image and receive a REAL/FAKE prediction.
Web Application Features
- Image upload
- File type validation
- PNG support
- JPG support
- JPEG support
- BMP support
- 10 MB upload limit
- REAL/FAKE prediction
- Confidence score
- Uploaded image display
- Error handling
Supported Image Formats
PNG
JPG
JPEG
BMP

Upload Limit
The Flask application allows image uploads up to:
10 MB

Flask URL
After running the application, open:
http://127.0.0.1:5000

Project Workflow
Kaggle Deepfake Dataset
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
Model Evaluation
        ↓
REAL / FAKE Prediction
        ↓
Flask Web Application

Technologies Used
Programming Language
- Python
Deep Learning
- TensorFlow
- Keras
- Xception
- ImageNet Transfer Learning
Data Processing
- NumPy
- Keras ImageDataGenerator
- Pillow
- OpenCV
Machine Learning and Evaluation
- Scikit-learn
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
Visualization
- Matplotlib
Web Development
- Flask
- HTML
- CSS
Dataset
- Kaggle API
Project Structure
Deepfake_Detection/
│
├── dataset/
│   └── .gitkeep
│
├── models/
│
├── results/
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── download_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

Important Files
File	Purpose
app.py	Flask web application
src/download_dataset.py	Downloads and prepares the Kaggle dataset
src/data_preprocessing.py	Creates train, validation, and test data generators
src/train.py	Builds and trains the Xception model
src/evaluate.py	Evaluates the trained model and generates plots
src/predict.py	Performs single-image REAL/FAKE prediction
templates/index.html	Flask web interface
static/style.css	Web application styling
requirements.txt	Python dependencies
.gitignore	Prevents large/private files from being uploaded


Installation
1. Create a Virtual Environment
python -m venv venv

2. Activate the Virtual Environment on Windows
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

Requirements
The project uses the following major Python libraries:
tensorflow>=2.15
numpy
matplotlib
scikit-learn
flask
pillow
kaggle
opencv-python

The complete dependency list is available in:
requirements.txt

Kaggle Dataset Setup
The dataset is downloaded using the Kaggle API.
Step 1: Create a Kaggle Account
Create or use a Kaggle account to access the required dataset.
Step 2: Create Kaggle API Credentials
Generate the Kaggle API credentials from the Kaggle account settings.
Step 3: Configure Credentials
Configure the Kaggle credentials on the local system.
Important: Kaggle credentials must not be uploaded to GitHub.
The .gitignore file prevents sensitive Kaggle credential files from being tracked.
Step 4: Configure Dataset Identifier
Open:
src/download_dataset.py

and configure the required Kaggle dataset identifier.
The dataset is automatically downloaded and extracted into the project dataset directory.
How to Run the Project
Step 1: Download the Dataset
Run:
python src/download_dataset.py

This downloads and extracts the dataset into the dataset/ directory.
Step 2: Train the Model
Run:
python src/train.py

The training process builds the Xception model and trains it using the prepared dataset.
The best model is saved in:
models/best_xception_deepfake.keras

Training history is saved in:
models/training_history.json

Step 3: Evaluate the Model
Run:
python src/evaluate.py

The evaluation process calculates the classification metrics and generates the evaluation plots.
The generated results are stored in:
results/

Step 4: Run the Flask Application
Run:
python app.py

Then open:
http://127.0.0.1:5000

Upload a face image and the application will display:
REAL / FAKE
Confidence Score

Security and File Handling
The project uses a .gitignore file to prevent sensitive and unnecessary files from being uploaded.
The following types of files are excluded:
- Kaggle credentials
- Dataset images
- Trained model files
- Uploaded images
- Virtual environment files
- Python cache files
- Temporary files
Large datasets and trained model files are kept locally instead of being stored directly in the GitHub repository.
Future Scope
The project can be extended in several ways:
- Fine-tune deeper Xception layers after the initial training stage.
- Extend the system from image-level detection to video-level deepfake detection.
- Add frame-based video analysis.
- Analyze multiple video frames for improved detection.
- Add Grad-CAM visualizations for model interpretability.
- Provide visual explanations for model predictions.
- Explore additional deepfake detection architectures.
- Improve model performance using larger and more diverse datasets.
- Deploy the Flask application to a cloud platform.
- Develop an API for integrating deepfake detection into other applications.
Conclusion
This project demonstrates an end-to-end deepfake face detection pipeline using Xception transfer learning.
The system covers the complete workflow from dataset acquisition and image preprocessing to model training, evaluation, prediction, and web-based deployment.
The project includes:
- Kaggle dataset acquisition
- Image preprocessing
- Image augmentation
- Xception transfer learning
- Binary REAL/FAKE classification
- Model training
- Early stopping
- Model checkpointing
- Learning-rate scheduling
- Model evaluation
- Classification metrics
- Confusion matrix
- ROC-AUC analysis
- Single-image prediction
- Confidence score generation
- Flask web application
The final system provides a practical interface for analyzing a face image and predicting whether it is REAL or FAKE, together with a confidence score.
This project demonstrates the practical application of Deep Learning, Transfer Learning, Computer Vision, and Web Development in the area of deepfake detection.
