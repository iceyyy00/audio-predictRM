import tensorflow as tf
from tensorflow.keras import layers, models


def create_cnn_model(input_shape=(128,128,1), num_classes=1):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    # Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))

    # Output (multi-label classification or regression)
    if num_classes == 1:
        model.add(layers.Dense(1, activation='linear'))
    else:
        model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model
