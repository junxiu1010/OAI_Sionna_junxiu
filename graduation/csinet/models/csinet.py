"""
CsiNet Baseline — TF/Keras 구현
================================
Wen et al., "Deep Learning for Massive MIMO CSI Feedback" (2018)

Encoder (UE): Conv2D layers -> Flatten -> Dense(M)
Decoder (gNB): Dense -> Reshape -> RefineNet blocks -> output
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RefineNetBlock(layers.Layer):
    """Residual refinement block used in CsiNet decoder."""

    def __init__(self, filters=8, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = None  # built lazily
        self.bn2 = layers.BatchNormalization()
        self.proj = None  # residual projection if channel mismatch

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.conv2 = layers.Conv2D(in_channels, 3, padding="same")
        if in_channels != self.filters:
            self.proj = layers.Conv2D(in_channels, 1, padding="same")
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = tf.nn.leaky_relu(out, alpha=0.3)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        return tf.nn.leaky_relu(out + residual, alpha=0.3)


class CsiNetEncoder(keras.Model):
    """CsiNet Encoder (runs at UE side).

    Input:  (batch, 2, Nt, Nc') — real/imag channels of angular-delay H
    Output: (batch, M) — compressed codeword
    """

    def __init__(self, Nt, Nc_prime, M, **kwargs):
        super().__init__(**kwargs)
        self.Nt = Nt
        self.Nc_prime = Nc_prime
        self.M = M

        self.conv1 = layers.Conv2D(16, 3, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(8, 3, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(4, 3, padding="same")
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(M)

    def call(self, x, training=False):
        # x: (batch, 2, Nt, Nc') -> transpose to (batch, Nt, Nc', 2) for Conv2D
        x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training), 0.3)
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training), 0.3)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training), 0.3)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class CsiNetDecoder(keras.Model):
    """CsiNet Decoder (runs at gNB side).

    Input:  (batch, M) — compressed codeword
    Output: (batch, 2, Nt, Nc') — reconstructed angular-delay H
    """

    def __init__(self, Nt, Nc_prime, M, num_refine=2, **kwargs):
        super().__init__(**kwargs)
        self.Nt = Nt
        self.Nc_prime = Nc_prime
        self.M = M

        self.dense = layers.Dense(2 * Nt * Nc_prime)
        self.refine_blocks = [RefineNetBlock(filters=8, name=f"refine_{i}")
                              for i in range(num_refine)]
        self.conv_out = layers.Conv2D(2, 3, padding="same")

    def call(self, z, training=False):
        # z: (batch, M)
        x = self.dense(z)
        x = tf.reshape(x, [-1, self.Nt, self.Nc_prime, 2])
        for blk in self.refine_blocks:
            x = blk(x, training=training)
        x = self.conv_out(x)
        # transpose back to (batch, 2, Nt, Nc')
        x = tf.transpose(x, [0, 3, 1, 2])
        return x


class CsiNetAutoencoder(keras.Model):
    """Full CsiNet autoencoder: Encoder + Decoder.

    compression_ratio = M / (2 * Nt * Nc')
    """

    def __init__(self, Nt=4, Nc_prime=32, compression_ratio=1/4, **kwargs):
        super().__init__(**kwargs)
        self.Nt = Nt
        self.Nc_prime = Nc_prime
        self.M = max(1, int(2 * Nt * Nc_prime * compression_ratio))
        self.compression_ratio = compression_ratio

        self.encoder = CsiNetEncoder(Nt, Nc_prime, self.M, name="encoder")
        self.decoder = CsiNetDecoder(Nt, Nc_prime, self.M, name="decoder")

    def call(self, x, training=False):
        z = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        return x_hat

    def get_config(self):
        return {"Nt": self.Nt, "Nc_prime": self.Nc_prime,
                "compression_ratio": self.compression_ratio}


def nmse_loss(y_true, y_pred):
    """Normalized Mean Squared Error in dB."""
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])
    power = tf.reduce_mean(tf.square(y_true), axis=[1, 2, 3])
    nmse = mse / tf.maximum(power, 1e-10)
    return tf.reduce_mean(nmse)


def cosine_similarity(y_true, y_pred):
    """Average cosine similarity."""
    y_t = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_p = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    dot = tf.reduce_sum(y_t * y_p, axis=1)
    norm_t = tf.norm(y_t, axis=1)
    norm_p = tf.norm(y_p, axis=1)
    cos_sim = dot / tf.maximum(norm_t * norm_p, 1e-10)
    return tf.reduce_mean(cos_sim)
