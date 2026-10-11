"""StyleGAN synthesis network with learned constant input."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .mapping import MappingNetwork
from .blocks import StyleConvBlock


class StyleGenerator(keras.Model):
    def __init__(self, latent_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.mapping = MappingNetwork(latent_dim)
        self.const = self.add_weight(name="const", shape=(1, 4, 4, 128),
                                     initializer="ones", trainable=True)
        # 4x4 -> 8x8 -> 16x16 -> 32x32
        self.blocks = [
            StyleConvBlock(128, upsample=False),
            StyleConvBlock(128),
            StyleConvBlock(64),
            StyleConvBlock(64),
        ]
        self.to_rgb = layers.Conv2D(3, 1, activation="tanh")

    def synthesize(self, w, w2=None, crossover=None):
        """Style mixing: blocks < crossover use w, blocks >= crossover use w2."""
        x = tf.tile(self.const, [tf.shape(w)[0], 1, 1, 1])
        for i, block in enumerate(self.blocks):
            wi = w2 if (w2 is not None and crossover is not None and i >= crossover) else w
            x = block(x, wi)
        return self.to_rgb(x)

    def call(self, z, training=False):
        return self.synthesize(self.mapping(z))
