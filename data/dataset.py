"""Fashion-MNIST pipeline: 28x28 gray -> 32x32 RGB in [-1, 1]."""
import tensorflow as tf
from tensorflow import keras


def load_fashion_mnist(batch_size=128, seed=42):
    (x, _), _ = keras.datasets.fashion_mnist.load_data()
    x = x.astype("float32")[..., None] / 127.5 - 1.0
    x = tf.image.resize(x, (32, 32))
    x = tf.repeat(x, 3, axis=-1)
    ds = (tf.data.Dataset.from_tensor_slices(x)
          .shuffle(10000, seed=seed)
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))
    return ds
