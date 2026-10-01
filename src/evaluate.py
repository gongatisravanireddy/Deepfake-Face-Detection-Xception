"""
evaluate.py
-----------
Loads the trained model and evaluates it on the test set. Computes
accuracy, precision, recall, F1, classification report, confusion matrix,
and ROC-AUC. Saves all plots to results/. All numbers come from actually
running the model on the actual test data - nothing here is fabricated.
"""

from pathlib import Path
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
)
from tensorflow.keras.models import load_model

sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import build_generators

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = MODELS_DIR / "best_xception_deepfake.keras"
HISTORY_PATH = MODELS_DIR / "training_history.json"


def load_trained_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Model file not found at {BEST_MODEL_PATH}. "
            "Run src/train.py first to train and save a model."
        )
    return load_model(BEST_MODEL_PATH)


def plot_training_curves():
    """Plots accuracy and loss curves from the saved training history."""
    if not HISTORY_PATH.exists():
        print(f"[WARNING] {HISTORY_PATH} not found. Skipping training curve plots.")
        return

    with open(HISTORY_PATH) as f:
        history = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png")
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()

    print(f"[Saved] {RESULTS_DIR / 'accuracy_curve.png'}")
    print(f"[Saved] {RESULTS_DIR / 'loss_curve.png'}")


def evaluate():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading trained model...")
    model = load_trained_model()

    print("[2/4] Loading test data...")
    _, _, test_gen, class_indices = build_generators()

    print("[3/4] Running predictions on test set...")
    y_true = test_gen.classes
    y_prob = model.predict(test_gen).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Determine which index corresponds to which label for reporting
    idx_to_label = {v: k for k, v in class_indices.items()}
    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, target_names=target_names)
    cm = confusion_matrix(y_true, y_pred)

    print("\n[Test Results]")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    roc_auc = None
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"  ROC-AUC       : {roc_auc:.4f}")
    except ValueError as e:
        print(f"[WARNING] Could not compute ROC-AUC: {e}")

    print("[4/4] Saving plots...")
    plot_training_curves()

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names)
    plt.yticks(tick_marks, target_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                      ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.close()
    print(f"[Saved] {RESULTS_DIR / 'confusion_matrix.png'}")

    # ROC curve plot
    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "roc_curve.png")
        plt.close()
        print(f"[Saved] {RESULTS_DIR / 'roc_curve.png'}")

    results = {
        "test_accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "class_mapping": class_indices,
    }
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {MODELS_DIR / 'evaluation_results.json'}")

    return results


if __name__ == "__main__":
    evaluate()
