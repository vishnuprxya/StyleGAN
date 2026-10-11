"""4-layer MLP mapping network: z -> w."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MappingNetwork(keras.Model):
    def __init__(self, latent_dim=128, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.denses = [layers.Dense(latent_dim) for _ in range(num_layers)]
        self.acts = [layers.LeakyReLU(0.2) for _ in range(num_layers)]

    def call(self, z):
        x = z * tf.math.rsqrt(tf.reduce_mean(tf.square(z), axis=1, keepdims=True) + 1e-8)
        for dense, act in zip(self.denses, self.acts):
            x = act(dense(x))
        return x
