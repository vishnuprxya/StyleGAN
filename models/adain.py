"""Adaptive Instance Normalization."""
import tensorflow as tf
from tensorflow.keras import layers


class AdaIN(layers.Layer):
    """AdaIN(x, w) = y_s(w) * (x - mu(x)) / sigma(x) + y_b(w)"""

    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, input_shape):
        channels = input_shape[0][-1]
        self.channels = channels
        self.style = layers.Dense(channels * 2)
        super().build(input_shape)

    def call(self, inputs):
        x, w = inputs
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_norm = (x - mean) * tf.math.rsqrt(var + self.eps)
        style = self.style(w)[:, None, None, :]
        y_s, y_b = style[..., : self.channels], style[..., self.channels:]
        return (1.0 + y_s) * x_norm + y_b
