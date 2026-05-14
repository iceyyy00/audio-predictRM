import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, multilabel_confusion_matrix

from preprocessing import preprocess_audio
from cnnModels import create_cnn_model

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.keras")

EFFECT_CLASSES_FILE = os.path.join(BASE_DIR, "effect_classes.txt")

def load_effect_classes():
    if os.path.exists(EFFECT_CLASSES_FILE):
        with open(EFFECT_CLASSES_FILE, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
            if classes:
                return classes

    return [
        "distortion",
        "reverb",
        "compression",
        "noise",
        "other"
    ]

EFFECT_CLASSES = load_effect_classes()

# Update EFFECT_CLASSES with all effects from dataset folders
dataset_dirs = [d for d in os.listdir(BASE_DIR) if d.startswith("dataset_") and os.path.isdir(os.path.join(BASE_DIR, d))]
for d in dataset_dirs:
    effect = d.replace("dataset_", "").lower()
    if effect not in EFFECT_CLASSES:
        EFFECT_CLASSES.append(effect)

# Add 'dry' if dry_audio exists
if os.path.exists(os.path.join(BASE_DIR, "dry_audio")):
    if "dry" not in EFFECT_CLASSES:
        EFFECT_CLASSES.append("dry")

def build_label_map_from_folders():
    print("Building labels from local dataset folders...")
    label_map = {}

    # Load from dataset_ folders
    dataset_dirs = [d for d in os.listdir(BASE_DIR) if d.startswith("dataset_") and os.path.isdir(os.path.join(BASE_DIR, d))]
    for d in dataset_dirs:
        effect = d.replace("dataset_", "").lower()
        path_dir = os.path.join(BASE_DIR, d)
        for file in os.listdir(path_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):  # assuming audio files
                full_path = os.path.join(path_dir, file)
                label_map[file] = (full_path, {effect})

    # Load from dry_audio as 'dry'
    dry_dir = os.path.join(BASE_DIR, "dry_audio")
    if os.path.exists(dry_dir):
        for file in os.listdir(dry_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                full_path = os.path.join(dry_dir, file)
                label_map[file] = (full_path, {"dry"})

    return label_map


def load_dataset(label_map):
    print("Loading dataset from local folders")
    X, y = [], []

    for audio_name, (path, effects) in label_map.items():
        try:
            mel = preprocess_audio(path)
            X.append(mel)

            label_vec = np.zeros(len(EFFECT_CLASSES), dtype=np.float32)
            for effect in effects:
                if effect in EFFECT_CLASSES:
                    label_vec[EFFECT_CLASSES.index(effect)] = 1.0
                else:
                    label_vec[EFFECT_CLASSES.index("other")] = 1.0

            y.append(label_vec)
            print(f"Loaded: {audio_name} => {sorted(list(effects))}")

        except Exception as e:
            print(f"Error loading audio {audio_name}:", e)

    X = np.array(X)
    y = np.array(y)

    return X, y


def run_training(epochs, batch_size, learning_rate):
    print("Training model for audio effect detection from local datasets.")
    print("Effect classes:", EFFECT_CLASSES)

    label_map = build_label_map_from_folders()
    if not label_map:
        raise RuntimeError("Dataset kosong, tidak ada file audio ditemukan di folder dataset.")

    X, y = load_dataset(label_map)

    if len(X) == 0 or len(y) == 0:
        raise RuntimeError("Dataset kosong, pastikan ada file audio di folder dataset.")

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_cnn_model(input_shape=(128,128,1), num_classes=len(EFFECT_CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("Training...")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    # Evaluasi model
    print("\nEvaluating model on validation set...")
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_val, y_pred_binary, target_names=EFFECT_CLASSES, zero_division=0))

    # Confusion Matrix per class
    print("\nConfusion Matrices per class:")
    mcm = multilabel_confusion_matrix(y_val, y_pred_binary)
    for i, matrix in enumerate(mcm):
        print(f"\n{EFFECT_CLASSES[i]}:")
        print(f"[[TN, FP],\n [FN, TP]] = {matrix}")

    # F1 Score, Precision, Recall per class dan average
    f1 = f1_score(y_val, y_pred_binary, average='macro', zero_division=0)
    precision = precision_score(y_val, y_pred_binary, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred_binary, average='macro', zero_division=0)

    print(f"\nMacro Average - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    f1_micro = f1_score(y_val, y_pred_binary, average='micro', zero_division=0)
    precision_micro = precision_score(y_val, y_pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(y_val, y_pred_binary, average='micro', zero_division=0)

    print(f"Micro Average - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1 Score: {f1_micro:.4f}")

    model.save(MODEL_PATH)
    print("Model saved at", MODEL_PATH)

    with open(EFFECT_CLASSES_FILE, "w", encoding="utf-8") as f:
        for c in EFFECT_CLASSES:
            f.write(c + "\n")

    print("Selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model untuk pendeteksian efek audio dari dataset lokal")
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    run_training(args.epochs, args.batch_size, args.learning_rate)

