import argparse
import os
import re
import sys
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_audio
from cnnModels import create_cnn_model

# Paths
BASE_DIR = r"c:\Users\ASUS\Documents\ProjectRM"
DATA_ROOT = os.path.join(BASE_DIR, "dataset", "ODAQ")
TRAIN_DIR = os.path.join(DATA_ROOT, "ODAQ_training")
LISTENING_DIR = os.path.join(DATA_ROOT, "ODAQ_listening_test")
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

KEYWORD_TO_EFFECT = {
    "distort": "distortion",
    "crunch": "distortion",
    "hiss": "distortion",
    "reverb": "reverb",
    "echo": "reverb",
    "compression": "compression",
    "compress": "compression",
    "pump": "compression",
    "pumping": "compression",
    "noise": "noise"
}


def extract_effects_from_comment(comment: str):
    label_set = set()
    text = comment.lower() if comment else ""

    for kw, effect in KEYWORD_TO_EFFECT.items():
        if kw in text:
            label_set.add(effect)

    if not label_set and text.strip():
        label_set.add("other")

    return label_set


def load_labels_from_xml(xml_dir):
    labels = {}

    for fname in os.listdir(xml_dir):
        if not fname.lower().endswith(".xml"):
            continue

        path = os.path.join(xml_dir, fname)
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            for test_file in root.iter("testFile"):
                audio_name = test_file.attrib.get("fileName")
                comment = test_file.attrib.get("comment", "")
                if not audio_name or audio_name.lower() == "reference.wav":
                    continue

                effect_set = extract_effects_from_comment(comment)

                if not effect_set:
                    continue

                labels.setdefault(audio_name, set()).update(effect_set)

        except Exception as e:
            print(f"Warning: gagal parsing XML {path}:", e)

    return labels


def construct_label_map(source):
    print("Membangun label dari metadata ODAQ untuk konteks listening pendidikan...")
    return load_labels_from_xml(LISTENING_DIR)


def find_audio_path(audio_name, source=None):
    for root, _, files in os.walk(TRAIN_DIR):
        if audio_name in files:
            return os.path.join(root, audio_name)

    return None


def load_dataset(label_map, source):
    print("Memuat dataset")
    X, y = [], []

    for audio_name, effects in label_map.items():
        path = find_audio_path(audio_name)
        if not path:
            print(f"Skip tidak ada file: {audio_name}")
            continue

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
            print(f"Error load audio {audio_name}:", e)

    X = np.array(X)
    y = np.array(y)

    return X, y


def run_training(source, epochs, batch_size, learning_rate):
    print("Konteks penelitian: pendeteksian efek audio pada rekaman listening pendidikan.")
    print("Kelas efek yang digunakan:", EFFECT_CLASSES)

    label_map = construct_label_map(source)
    if not label_map:
        raise RuntimeError("Dataset kosong, tidak ada label ditemukan untuk sumber yang dipilih.")

    X, y = load_dataset(label_map, source)

    if len(X) == 0 or len(y) == 0:
        raise RuntimeError("Dataset kosong, pastikan ODAQ dataset sudah terdownload dan label dapat dibaca.")

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

    model.save(MODEL_PATH)
    print("Model saved at", MODEL_PATH)

    with open(os.path.join(BASE_DIR, "effect_classes.txt"), "w", encoding="utf-8") as f:
        for c in EFFECT_CLASSES:
            f.write(c + "\n")

    print("Selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model ODAQ untuk pendeteksian efek audio dalam konteks listening pendidikan")
    parser.add_argument("--epochs", type=int, default=100, help="Jumlah epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    run_training("odaq", args.epochs, args.batch_size, args.learning_rate)

