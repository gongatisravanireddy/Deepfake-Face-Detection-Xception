"""
train.py
--------
Builds the Xception-based binary classifier, trains it with the configured
callbacks, and stores the training history + best model checkpoint.
"""

from pathlib import Path
import sys
import json

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.append(str(Path(__file__).resolve().parent))
from data_preprocessing import build_generators, IMG_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = MODELS_DIR / "best_xception_deepfake.keras"
HISTORY_PATH = MODELS_DIR / "training_history.json"

EPOCHS = 10
LEARNING_RATE = 0.0001


def check_gpu():
    """Prints GPU availability. Training proceeds on CPU if none found."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    else:
        print("[WARNING] No GPU detected. Training will run on CPU and may "
              "be significantly slower.")
    return len(gpus) > 0


def build_model():
    """
    Builds the Xception-based classifier:
    Input(299,299,3) -> Xception (frozen) -> GAP -> Dense(256, relu)
    -> Dropout(0.5) -> Dense(1, sigmoid)
    """
    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False  # freeze base model initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def get_callbacks():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


def train():
    print("[1/4] Checking GPU availability...")
    check_gpu()

    print("[2/4] Loading dataset...")
    train_gen, valid_gen, test_gen, class_indices = build_generators()

    print("[3/4] Building Xception model...")
    model, base_model = build_model()
    model.summary()

    print("[4/4] Training...")
    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"\n[Saved] Training history -> {HISTORY_PATH}")
    print(f"[Saved] Best model -> {BEST_MODEL_PATH}")

    # Also persist class_indices for use in evaluate.py / predict.py
    with open(MODELS_DIR / "class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=2)

    return model, history, class_indices


if __name__ == "__main__":
    train()
