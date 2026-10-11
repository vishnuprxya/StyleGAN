"""Discriminator with Minibatch StdDev."""
from tensorflow import keras
from tensorflow.keras import layers
from .blocks import MinibatchStdDev


def build_discriminator():
    inp = keras.Input((32, 32, 3))
    x = layers.Conv2D(64, 1)(inp)
    x = layers.LeakyReLU(0.2)(x)
    for f in (64, 128, 128):           # 32 -> 16 -> 8 -> 4
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.AveragePooling2D()(x)
    x = MinibatchStdDev()(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1)(x)           # logit
    return keras.Model(inp, out, name="discriminator")
