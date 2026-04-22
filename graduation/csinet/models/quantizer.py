"""
STE-based Quantizer for CsiNet codeword
========================================
Straight-Through Estimator: quantize in forward, pass gradient through in backward.
"""

import tensorflow as tf
from tensorflow.keras import layers


class STEQuantizer(layers.Layer):
    """Uniform scalar quantizer with STE for backprop.

    Quantizes each element of the input to B bits within [0, 1] range.
    """

    def __init__(self, bits=4, **kwargs):
        super().__init__(**kwargs)
        self.bits = bits
        self.n_levels = 2 ** bits

    def call(self, x, training=False):
        # Clamp to [0, 1]
        x_clamp = tf.clip_by_value(tf.sigmoid(x), 0.0, 1.0)

        if training:
            # STE: quantize but pass gradient through
            x_q = tf.round(x_clamp * (self.n_levels - 1)) / (self.n_levels - 1)
            return x_clamp + tf.stop_gradient(x_q - x_clamp)
        else:
            return tf.round(x_clamp * (self.n_levels - 1)) / (self.n_levels - 1)

    def get_config(self):
        config = super().get_config()
        config["bits"] = self.bits
        return config


class QuantizedCsiNetAutoencoder(tf.keras.Model):
    """CsiNet with quantization-aware training."""

    def __init__(self, csinet_ae, bits=4, **kwargs):
        super().__init__(**kwargs)
        self.csinet = csinet_ae
        self.quantizer = STEQuantizer(bits=bits)
        self.total_bits = csinet_ae.M * bits

    def call(self, x, training=False):
        z = self.csinet.encoder(x, training=training)
        z_q = self.quantizer(z, training=training)
        x_hat = self.csinet.decoder(z_q, training=training)
        return x_hat
