"""StyleConvBlock, NoiseInjection and MinibatchStdDev."""
import tensorflow as tf
from tensorflow.keras import layers
from .adain import AdaIN


class NoiseInjection(layers.Layer):
    """Per-pixel Gaussian noise scaled by a learned per-channel factor."""

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="noise_scale", shape=(1, 1, 1, input_shape[-1]),
            initializer="zeros", trainable=True)

    def call(self, x):
        s = tf.shape(x)
        noise = tf.random.normal([s[0], s[1], s[2], 1])
        return x + self.scale * noise


class StyleConvBlock(layers.Layer):
    """[Upsample] -> (Conv -> Noise -> LeakyReLU -> AdaIN) x 2"""

    def __init__(self, filters, upsample=True, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.UpSampling2D(interpolation="bilinear") if upsample else None
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.noise1, self.noise2 = NoiseInjection(), NoiseInjection()
        self.adain1, self.adain2 = AdaIN(), AdaIN()
        self.act = layers.LeakyReLU(0.2)

    def call(self, x, w):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.adain1([self.act(self.noise1(self.conv1(x))), w])
        x = self.adain2([self.act(self.noise2(self.conv2(x))), w])
        return x


class MinibatchStdDev(layers.Layer):
    """Appends the batch-wide std statistic as an extra feature map."""

    def call(self, x):
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        std = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axis=0) + 1e-8)
        m = tf.reduce_mean(std)
        s = tf.shape(x)
        fmap = tf.fill(tf.stack([s[0], s[1], s[2], 1]), m)
        return tf.concat([x, fmap], axis=-1)
