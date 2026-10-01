"""
predict.py
----------
Loads the trained model once and exposes predict_image(image_path) for
running a REAL/FAKE prediction (with confidence) on a single image.
"""

from pathlib import Path
import sys
import json

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import IMG_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_xception_deepfake.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

_model = None
_idx_to_label = None


def _load_resources():
    """Lazily loads the model and class mapping (only once per process)."""
    global _model, _idx_to_label

    if _model is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"[ERROR] Model file not found at {BEST_MODEL_PATH}. "
                "Train the model first by running src/train.py."
            )
        _model = load_model(BEST_MODEL_PATH)

    if _idx_to_label is None:
        if not CLASS_INDICES_PATH.exists():
            raise FileNotFoundError(
                f"[ERROR] Class mapping file not found at {CLASS_INDICES_PATH}. "
                "Train the model first by running src/train.py."
            )
        with open(CLASS_INDICES_PATH) as f:
            class_indices = json.load(f)  # e.g. {"fake": 0, "real": 1}
        _idx_to_label = {v: k for k, v in class_indices.items()}

    return _model, _idx_to_label


def predict_image(image_path: str) -> dict:
    """
    Runs a REAL/FAKE prediction on a single image file.

    Returns a dict:
        {
            "prediction": "REAL" or "FAKE",
            "confidence": float (0-100),
        }
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"[ERROR] Image file not found: {image_path}")

    model, idx_to_label = _load_resources()

    try:
        img = keras_image.load_img(image_path, target_size=IMG_SIZE)
    except Exception as e:
        raise ValueError(f"[ERROR] Could not read image '{image_path}': {e}")

    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = float(model.predict(img_array, verbose=0)[0][0])

    # idx_to_label maps {0: 'fake', 1: 'real'} (or whatever the actual
    # training-time mapping was) - never assume the mapping blindly.
    predicted_idx = 1 if prob >= 0.5 else 0
    label = idx_to_label[predicted_idx].upper()
    confidence = prob * 100 if predicted_idx == 1 else (1 - prob) * 100

    return {"prediction": label, "confidence": round(confidence, 2)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    result = predict_image(sys.argv[1])
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
