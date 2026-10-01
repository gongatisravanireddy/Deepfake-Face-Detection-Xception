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
Evaluation      Prediction
   ↓               ↓
Metrics        REAL / FAKE
   ↓               ↓
Plots          Confidence
   ↓               ↓
   └───────┬───────┘
           ↓
     Flask Web App


## Methodology

The project follows these major steps:

1. Download the dataset from Kaggle using the Kaggle API.
2. Automatically identify the training, validation, and test directories.
3. Resize images to 299 × 299 pixels.
4. Normalize pixel values between 0 and 1.
5. Apply data augmentation to training images.
6. Use an ImageNet-pretrained Xception model as the feature extractor.
7. Add a custom classification head for binary classification.
8. Train the model using Adam optimization.
9. Save the best model checkpoint.
10. Evaluate the trained model on the test dataset.
11. Use the trained model to predict REAL or FAKE for individual images.
12. Provide predictions through a Flask web application.

## Dataset

The project uses a deepfake image dataset downloaded from **Kaggle** using the Kaggle API.

The dataset is not included in this repository because datasets containing large numbers of images can be very large.

The expected dataset structure is:

```text
dataset/
├── train/
│   ├── real/
│   └── fake/
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/


Data Preprocessing
Images are processed using Keras ImageDataGenerator.
Image Size
299 × 299

Batch Size
32

Pixel Normalization
Pixel values are normalized using:
1 / 255

Training Data Augmentation
The training pipeline applies:
- Rotation
- Zoom
- Width shifting
- Height shifting
- Horizontal flipping
- Brightness adjustment
Validation and test images use normalization only and are not augmented.
Xception Model
The project uses an ImageNet-pretrained Xception architecture.
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

The Xception backbone is initially frozen and a custom classification head is trained for binary classification.
Model Configuration
- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: Binary Cross-Entropy
- Metric: Accuracy
- Maximum Epochs: 10
Training
The training pipeline uses:
- Early Stopping
- Model Checkpointing
- ReduceLROnPlateau
The best model is saved as:
models/best_xception_deepfake.keras

Training history is saved as:
models/training_history.json

Class mappings are saved as:
models/class_indices.json

These files are generated after running the training process.
Model Evaluation
The trained model is evaluated on the test dataset using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Classification Report
- Confusion Matrix
The evaluation process also generates:
results/
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
└── roc_curve.png

These files are generated after running the evaluation script.
Evaluation Results
Actual evaluation values should be added after running the trained model on the test dataset.
No evaluation values are hard-coded or manually created.
Prediction
The prediction module accepts a single image and returns:
Prediction: REAL / FAKE
Confidence: XX%

The image is:
1. Loaded
2. Resized to 299 × 299
3. Normalized
4. Passed to the trained Xception model
5. Classified as REAL or FAKE
The confidence score is calculated from the model prediction probability.
Flask Web Application
The project includes a Flask web application that allows users to upload an image and receive a prediction.
Web Application Features
- Image upload
- File type validation
- PNG, JPG, JPEG and BMP support
- 10 MB upload limit
- REAL/FAKE prediction
- Confidence score
- Uploaded image display
- Error handling
The Flask application runs on:
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
Model Evaluation
        ↓
REAL / FAKE Prediction
        ↓
Flask Web Application

Technologies Used
- Python
- TensorFlow
- Keras
- Xception
- NumPy
- Matplotlib
- Scikit-learn
- Flask
- Pillow
- OpenCV
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

The models/ and results/ folders contain files generated during training and evaluation.
Installation
1. Create a virtual environment
python -m venv venv

2. Activate the virtual environment on Windows
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

Kaggle Dataset Setup
The dataset is downloaded using the Kaggle API.
Create a Kaggle API token from your Kaggle account and configure the required credentials.
Then configure the dataset identifier in:
src/download_dataset.py

The dataset is automatically downloaded and extracted into the project dataset directory.
How to Train the Model
First download and prepare the dataset:
python src/download_dataset.py

Then train the Xception model:
python src/train.py

The best model and training history will be saved in the models/ directory.
How to Evaluate the Model
Run:
python src/evaluate.py

The evaluation metrics and plots will be generated in the results/ directory.
How to Run the Flask Application
Run:
python app.py

Then open:
http://127.0.0.1:5000

Upload a face image to receive a REAL/FAKE prediction and confidence score.
Future Scope
- Fine-tune deeper Xception layers after initial training.
- Extend the system from image detection to video-level deepfake detection.
- Add frame-based video analysis.
- Add Grad-CAM visualizations for model interpretability.
- Deploy the Flask application to a cloud platform.
- Explore additional deepfake detection architectures.
Conclusion
This project demonstrates an end-to-end deepfake face detection pipeline using Xception transfer learning.
It covers dataset acquisition, image preprocessing, augmentation, model training, evaluation, single-image prediction, and Flask-based deployment.
The system provides a practical interface for detecting whether an input face image is REAL or FAKE and displaying the model's confidence score.
