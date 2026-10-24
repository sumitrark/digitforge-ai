"""Generator architecture (identical to the original notebook, cell 13) and a tolerant loader."""
import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

LATENT_DIM = 100
DEFAULT_MODEL_PATHS = ("artifacts/dcgan_generator.h5", "artifacts/generator.keras")


def build_generator() -> tf.keras.Sequential:
    m = tf.keras.Sequential()
    m.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(LATENT_DIM,)))
    m.add(layers.BatchNormalization())
    m.add(layers.LeakyReLU())
    m.add(layers.Reshape((7, 7, 256)))
    m.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    m.add(layers.BatchNormalization())
    m.add(layers.LeakyReLU())
    m.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    m.add(layers.BatchNormalization())
    m.add(layers.LeakyReLU())
    m.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding="same",
                                 use_bias=False, activation="tanh"))
    return m

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same"
        ),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(
            128, (5, 5), strides=(2, 2), padding="same"
        ),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1),
    ])
    return model


def find_model_path(paths=DEFAULT_MODEL_PATHS):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_generator(path=None):
    """Load the trained Generator. Raises FileNotFoundError / RuntimeError with clear messages."""
    path = path or find_model_path()
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "Trained Generator not found. Put dcgan_generator.h5 (or generator.keras) in artifacts/.")
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as first_error:  # version mismatch / weights-only file
        try:
            model = build_generator()
            model.load_weights(path)
            return model
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load '{path}' with TensorFlow {tf.__version__}. "
                f"Full-model load: {first_error}. Weights-only load: {second_error}") from second_error


def load_discriminator(
    path="artifacts/dcgan_discriminator_weights.npz"
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Discriminator weights not found: {path}"
        )

    model = build_discriminator()

    with np.load(path) as data:
        weights = [
            data[f"arr_{i}"]
            for i in range(len(data.files))
        ]

    try:
        model.set_weights(weights)
    except Exception as error:
        raise RuntimeError(
            f"Discriminator weights do not match the "
            f"local architecture: {error}"
        ) from error

    return model