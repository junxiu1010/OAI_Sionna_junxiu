"""
Stage 1: Channel Statistics Autoencoder
========================================
Compresses covariance matrix R_H and/or PDP for slow-rate reporting.
Architecture matches trained checkpoints (cov_latent=16, pdp_latent=8).
"""

import tensorflow as tf
from tensorflow.keras import layers


class CovarianceEncoder(tf.keras.Model):
    def __init__(self, cov_dim, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(128, activation="relu", name="fc1")
        self.fc2 = layers.Dense(64, activation="relu", name="fc2")
        self.fc_out = layers.Dense(latent_dim, name="fc_out")

    def call(self, r_h_vec, training=False):
        x = self.fc1(r_h_vec)
        x = self.fc2(x)
        return self.fc_out(x)


class CovarianceDecoder(tf.keras.Model):
    def __init__(self, cov_dim, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(64, activation="relu", name="fc1")
        self.fc2 = layers.Dense(128, activation="relu", name="fc2")
        self.fc_out = layers.Dense(cov_dim, name="fc_out")

    def call(self, z, training=False):
        x = self.fc1(z)
        x = self.fc2(x)
        return self.fc_out(x)


class PDPEncoder(tf.keras.Model):
    def __init__(self, pdp_dim, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(64, activation="relu", name="fc1")
        self.fc_out = layers.Dense(latent_dim, name="fc_out")

    def call(self, p, training=False):
        x = self.fc1(p)
        return self.fc_out(x)


class PDPDecoder(tf.keras.Model):
    def __init__(self, pdp_dim, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(64, activation="relu", name="fc1")
        self.fc_out = layers.Dense(pdp_dim, activation="softplus", name="fc_out")

    def call(self, z, training=False):
        x = self.fc1(z)
        return self.fc_out(x)


class StatisticsAutoencoder(tf.keras.Model):
    """Combined covariance + PDP autoencoder (Stage 1)."""

    def __init__(self, cov_dim, pdp_dim, cov_latent=16, pdp_latent=8, **kwargs):
        super().__init__(**kwargs)
        self.cov_enc = CovarianceEncoder(cov_dim, cov_latent, name="cov_enc")
        self.cov_dec = CovarianceDecoder(cov_dim, cov_latent, name="cov_dec")
        self.pdp_enc = PDPEncoder(pdp_dim, pdp_latent, name="pdp_encoder")
        self.pdp_dec = PDPDecoder(pdp_dim, pdp_latent, name="pdp_decoder")
        self.cov_latent = cov_latent
        self.pdp_latent = pdp_latent

    def call(self, inputs, training=False):
        r_h_vec, pdp = inputs
        z_cov = self.cov_enc(r_h_vec, training=training)
        r_hat = self.cov_dec(z_cov, training=training)
        z_pdp = self.pdp_enc(pdp, training=training)
        pdp_hat = self.pdp_dec(z_pdp, training=training)
        return r_hat, pdp_hat, z_cov, z_pdp

    def get_condition_vector(self, r_h_vec, pdp, training=False):
        """Get concatenated conditioning vector for Stage 2."""
        z_cov = self.cov_enc(r_h_vec, training=training)
        z_pdp = self.pdp_enc(pdp, training=training)
        return tf.concat([z_cov, z_pdp], axis=-1)


def vectorize_covariance(R_H):
    """Extract upper-triangular elements, split real/imag.
    R_H: (batch, D, D) complex -> (batch, D*(D+1)) float
    """
    D = R_H.shape[-1]
    indices = []
    for i in range(D):
        for j in range(i, D):
            indices.append((i, j))

    parts = []
    for (i, j) in indices:
        elem = R_H[:, i, j]
        parts.append(tf.math.real(elem))
        parts.append(tf.math.imag(elem))
    return tf.stack(parts, axis=-1)
