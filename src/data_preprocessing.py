"""
data_preprocessing.py
----------------------
Builds Keras ImageDataGenerator pipelines for train/valid/test splits,
locating the dataset root automatically (does not assume a fixed folder
name), and prints dataset statistics + class mapping.
"""

from pathlib import Path
import sys

from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(str(Path(__file__).resolve().parent))
from download_dataset import find_dataset_root, SPLIT_ALIASES, DATASET_DIR

IMG_SIZE = (299, 299)
BATCH_SIZE = 32


def _resolve_split_folder(dataset_root: Path, split: str) -> Path:
    """Finds the actual folder name for a split, honoring aliases."""
    aliases = SPLIT_ALIASES.get(split, [split])
    for entry in dataset_root.iterdir():
        if entry.is_dir() and entry.name.lower() in aliases:
            return entry
    raise FileNotFoundError(
        f"Could not find a '{split}' folder (aliases: {aliases}) under {dataset_root}"
    )


def build_generators(dataset_root: Path = None):
    """
    Creates train/valid/test ImageDataGenerators.

    Returns:
        train_gen, valid_gen, test_gen, class_indices (dict)
    """
    if dataset_root is None:
        dataset_root = find_dataset_root()
    if dataset_root is None:
        raise RuntimeError(
            f"Dataset root not found under {DATASET_DIR}. "
            "Run src/download_dataset.py first."
        )
    dataset_root = Path(dataset_root)

    train_dir = _resolve_split_folder(dataset_root, "train")
    valid_dir = _resolve_split_folder(dataset_root, "valid")
    test_dir = _resolve_split_folder(dataset_root, "test")

    # Training data: normalize + augment
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
    )

    # Validation/test data: normalize only, no augmentation
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    valid_gen = eval_datagen.flow_from_directory(
        valid_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    class_indices = train_gen.class_indices  # e.g. {'fake': 0, 'real': 1}

    print("\n[Data Summary]")
    print(f"  Training images   : {train_gen.samples}")
    print(f"  Validation images : {valid_gen.samples}")
    print(f"  Test images       : {test_gen.samples}")
    print(f"  Class mapping     : {class_indices}")

    import numpy as np
    for name, gen in [("Train", train_gen), ("Validation", valid_gen), ("Test", test_gen)]:
        counts = {cls: 0 for cls in class_indices}
        labels = gen.classes
        for cls_name, cls_idx in class_indices.items():
            counts[cls_name] = int(np.sum(labels == cls_idx))
        breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {name:10s} breakdown: {breakdown}")

    return train_gen, valid_gen, test_gen, class_indices


if __name__ == "__main__":
    build_generators()
