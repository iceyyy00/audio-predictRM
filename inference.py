from preprocessing import preprocess_audio
import tensorflow as tf
import numpy as np
import sys
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.keras')
EFFECT_CLASSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'effect_classes.txt')

print("Context: pendeteksian efek audio dari dataset lokal.")
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Debug check: pastikan output layer sigmoid untuk klasifikasi multi-label.
output_layer = model.layers[-1]
output_shape = getattr(model, 'output_shape', None) or getattr(output_layer, 'output_shape', None)
print(f"Model output shape: {output_shape}, activation: {getattr(output_layer, 'activation', None)}")

if os.path.exists(EFFECT_CLASSES_FILE):
    with open(EFFECT_CLASSES_FILE, 'r', encoding='utf-8') as f:
        EFFECT_CLASSES = [line.strip() for line in f if line.strip()]
else:
    EFFECT_CLASSES = [
        'distortion', 'reverb', 'compression', 'noise', 'other'
    ]


def predict_audio(file_path, threshold=0.5):
    try:
        print(f"\nProcessing: {file_path}")

        mel = preprocess_audio(file_path)
        mel_batch = np.expand_dims(mel, axis=0)

        prediction = model.predict(mel_batch, verbose=0)[0]

        # normalisasi output agar menjadi probabilitas 0..1 jika model linear
        if prediction.ndim > 0 and np.any(prediction > 1.0):
            prediction = 1.0 / (1.0 + np.exp(-prediction))  # sigmoid fallback

        if prediction.ndim == 0 or len(prediction) == 1:
            pred_val = float(prediction)
            pred_prob = 1.0 / (1.0 + np.exp(-pred_val)) if output_layer.activation != tf.keras.activations.sigmoid else pred_val
            pred_prob = float(np.clip(pred_prob, 0.0, 1.0))
            print(f"✓ Predicted scalar: {pred_val} (mapped {pred_prob})")
            return {"score": pred_prob}

        prediction = np.clip(prediction, 0.0, 1.0)
        effects = [EFFECT_CLASSES[i] for i, v in enumerate(prediction) if v >= threshold]
        confs = {EFFECT_CLASSES[i]: float(v) for i, v in enumerate(prediction)}

        if not effects:
            effects = ['none_detected']

        print("✓ Detected effects:", effects)
        print("✓ Confidence:", confs)

        return {"effects": effects, "confidence": confs}

    except Exception as e:
        print("Error:", e)
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_audio(sys.argv[1])
    else:
        # Scan test_audio folder
        audio_files = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(base_dir, "test_audio")
        
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append(os.path.join(test_dir, f))
        else:
            print("Folder test_audio tidak ditemukan.")
            pass

        # Test on all files in test_audio
        for f in audio_files:
            predict_audio(f)

