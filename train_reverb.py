import argparse
import os
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_audio
from cnnModels import create_cnn_model

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset_reverb")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model_reverb.keras")
CLASS_FILE = os.path.join(os.path.dirname(__file__), "effect_classes_reverb.txt")


def parse_label_from_filename(filename: str) -> str:
    # Example: '01b_trumpet_Big-Cave.wav' -> 'Big-Cave'
    name = os.path.splitext(filename)[0]
    if "_" not in name:
        return name
    return name.split("_")[-1]


def build_labels(data_dir: str):
    labels = {}
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".wav"):
            continue
        label = parse_label_from_filename(fname)
        labels[fname] = label
    return labels


def load_dataset(data_dir: str, labels: dict, class_names: list):
    X, y = [], []
    for fname, label in labels.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        try:
            mel = preprocess_audio(path)
            X.append(mel)
            label_vec = np.zeros(len(class_names), dtype=np.float32)
            label_index = class_names.index(label)
            label_vec[label_index] = 1.0
            y.append(label_vec)
            print(f"Loaded: {fname} -> {label}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    if not X:
        raise RuntimeError("Tidak ada data yang berhasil dimuat dari dataset_reverb.")

    return np.array(X), np.array(y)


def run_training(epochs: int, batch_size: int, learning_rate: float):
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Folder dataset tidak ditemukan: {DATA_DIR}")

    raw_labels = build_labels(DATA_DIR)
    class_names = sorted(set(raw_labels.values()))

    print("Dataset directory:", DATA_DIR)
    print("Jumlah file:", len(raw_labels))
    print("Kelas reverb:", class_names)

    X, y = load_dataset(DATA_DIR, raw_labels, class_names)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    label_indices = [int(np.argmax(row)) for row in y]
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=label_indices
        )
    except ValueError as e:
        print("Warning: stratified split gagal karena kelas jarang. Menggunakan split acak tanpa stratifikasi.")
        print("Detail:", e)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

    model = create_cnn_model(input_shape=(128, 128, 1), num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    model.save(MODEL_PATH)
    print("Model saved ke:", MODEL_PATH)

    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")
    print("Class file saved ke:", CLASS_FILE)
    print("Selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model pada folder dataset_reverb")
    parser.add_argument("--epochs", type=int, default=30, help="Jumlah epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Ukuran batch")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    run_training(args.epochs, args.batch_size, args.learning_rate)
