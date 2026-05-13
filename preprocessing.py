import librosa
import numpy as np

def extract_features(file_path, n_mels=128):
    # Load audio
    signal, sr = librosa.load(file_path, sr=22050)

    # Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=n_mels
    )

    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram


def resize_spectrogram(mel_spec, max_len=128):
    if mel_spec.shape[1] < max_len:
        pad_width = max_len - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec = mel_spec[:, :max_len]

    return mel_spec


def preprocess_audio(file_path):
    mel = extract_features(file_path)
    mel = resize_spectrogram(mel)

    # Normalisasi aman
    denom = mel.max() - mel.min()
    if denom != 0:
        mel = (mel - mel.min()) / denom

    # Tambahkan channel dimension
    mel = np.expand_dims(mel, axis=-1)

    return mel
