"""
Stage 2: Statistics-Conditioned CsiNet
=======================================
Extends CsiNet with FiLM conditioning from channel statistics.
Architecture matches trained checkpoints (cond_dim=24, 2 refine blocks, 8 filters).
"""

import tensorflow as tf
from tensorflow.keras import layers


class FiLMLayer(layers.Layer):
    """Feature-wise Linear Modulation (single Dense layer).
    Identity-initialized at output to preserve transferred weights.
    """

    def __init__(self, cond_dim, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.gamma_net = tf.keras.Sequential([
            layers.Dense(feature_dim,
                         kernel_initializer="zeros",
                         bias_initializer="ones"),
        ], name="gamma_net")
        self.beta_net = tf.keras.Sequential([
            layers.Dense(feature_dim,
                         kernel_initializer="zeros",
                         bias_initializer="zeros"),
        ], name="beta_net")

    def call(self, x, cond):
        gamma = self.gamma_net(cond)
        beta = self.beta_net(cond)
        gamma = tf.reshape(gamma, [-1, 1, 1, tf.shape(gamma)[-1]])
        beta = tf.reshape(beta, [-1, 1, 1, tf.shape(beta)[-1]])
        return gamma * x + beta


class ConditionedRefineBlock(layers.Layer):
    """RefineNet block with FiLM conditioning."""

    def __init__(self, filters, cond_dim, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.cond_dim = cond_dim
        self.conv1 = layers.Conv2D(filters, 3, padding="same", name="conv1")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.film1 = FiLMLayer(cond_dim, filters, name="film1")
        self.conv2 = None
        self.bn2 = layers.BatchNormalization(name="bn2")

    def build(self, input_shape):
        in_ch = input_shape[-1]
        self.conv2 = layers.Conv2D(in_ch, 3, padding="same", name="conv2")
        self.film2 = FiLMLayer(self.cond_dim, in_ch, name="film2")
        super().build(input_shape)

    def call(self, x, cond, training=False):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.film1(out, cond)
        out = tf.nn.leaky_relu(out, 0.3)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.film2(out, cond)
        return tf.nn.leaky_relu(out + residual, 0.3)


class ConditionedEncoder(tf.keras.Model):
    """CsiNet Encoder with FiLM conditioning."""

    def __init__(self, Nt, Nc_prime, M, cond_dim, **kwargs):
        super().__init__(**kwargs)
        self.Nt = Nt
        self.Nc_prime = Nc_prime

        self.conv1 = layers.Conv2D(16, 3, padding="same", name="conv1")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.film1 = FiLMLayer(cond_dim, 16, name="film1")
        self.conv2 = layers.Conv2D(8, 3, padding="same", name="conv2")
        self.bn2 = layers.BatchNormalization(name="bn2")
        self.film2 = FiLMLayer(cond_dim, 8, name="film2")
        self.conv3 = layers.Conv2D(4, 3, padding="same", name="conv3")
        self.bn3 = layers.BatchNormalization(name="bn3")
        self.film3 = FiLMLayer(cond_dim, 4, name="film3")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(M, name="dense")

    def call(self, x, cond, training=False):
        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.bn1(self.conv1(x), training=training)
        x = tf.nn.leaky_relu(self.film1(x, cond), 0.3)
        x = self.bn2(self.conv2(x), training=training)
        x = tf.nn.leaky_relu(self.film2(x, cond), 0.3)
        x = self.bn3(self.conv3(x), training=training)
        x = tf.nn.leaky_relu(self.film3(x, cond), 0.3)
        return self.dense(self.flatten(x))


class ConditionedDecoder(tf.keras.Model):
    """CsiNet Decoder with FiLM conditioning, 2 refine blocks, 8 filters."""

    def __init__(self, Nt, Nc_prime, M, cond_dim, num_refine=2, **kwargs):
        super().__init__(**kwargs)
        self.Nt = Nt
        self.Nc_prime = Nc_prime

        self.dense = layers.Dense(2 * Nt * Nc_prime, name="dense")
        self.refine_blocks = [
            ConditionedRefineBlock(8, cond_dim)
            for _ in range(num_refine)
        ]
        self.conv_out = layers.Conv2D(2, 3, padding="same", name="conv_out")

    def call(self, z, cond, training=False):
        x = self.dense(z)
        x = tf.reshape(x, [-1, self.Nt, self.Nc_prime, 2])
        for blk in self.refine_blocks:
            x = blk(x, cond, training=training)
        x = self.conv_out(x)
        return tf.transpose(x, [0, 3, 1, 2])


class ConditionedCsiNet(tf.keras.Model):
    """Full conditioned CsiNet autoencoder (Stage 2)."""

    def __init__(self, Nt=4, Nc_prime=32, compression_ratio=1/4,
                 cond_dim=24, **kwargs):
        super().__init__(**kwargs)
        self.M = max(1, int(2 * Nt * Nc_prime * compression_ratio))
        self.encoder = ConditionedEncoder(Nt, Nc_prime, self.M, cond_dim,
                                          name="encoder")
        self.decoder = ConditionedDecoder(Nt, Nc_prime, self.M, cond_dim,
                                          name="decoder")

    def call(self, inputs, training=False):
        x, cond = inputs
        z = self.encoder(x, cond, training=training)
        x_hat = self.decoder(z, cond, training=training)
        return x_hat
