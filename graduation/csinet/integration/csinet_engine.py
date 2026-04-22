"""
Phase 4: CsiNet Inference Engine
==================================
Intercepts H from the Sionna proxy, transforms to angular-delay domain,
runs trained CsiNet encoder/decoder, and returns reconstructed H_hat.
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import h5py

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.csinet import CsiNetAutoencoder
from models.stat_autoencoder import StatisticsAutoencoder, vectorize_covariance
from models.conditioned_csinet import ConditionedCsiNet
from integration.differential_cond import DifferentialConditioner

NT = 4
NR = 2
NC_PRIME = 32
COV_LATENT = 16
PDP_LATENT = 8
COND_DIM = COV_LATENT + PDP_LATENT  # 24
COV_DIM = NT * NR  # 8
COV_VEC_DIM = COV_DIM * (COV_DIM + 1)  # 72
PDP_DIM = COV_VEC_DIM  # 72 (matches training data)


def _load_h5_weights(filepath):
    """Read all weight arrays from h5 file into a flat dict keyed by path."""
    weights = {}
    with h5py.File(filepath, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)
        f.visititems(_visit)
    return weights


def _normalize_ckpt_key(key):
    """Normalize checkpoint key to a canonical form for matching.
    e.g. 'encoder/conv1/vars/0' -> 'encoder/conv1/0'
    """
    return key.replace("/vars/", "/")


def _normalize_model_path(path, root_name):
    """Normalize model variable path to canonical form.
    e.g. 'conditioned_csi_net/encoder/conv1/kernel' -> 'encoder/conv1/kernel'
    BN vars: gamma->0, beta->1, moving_mean->2, moving_variance->3
    Dense/Conv: kernel->0, bias->1
    """
    if path.startswith(root_name + "/"):
        path = path[len(root_name) + 1:]

    var_map = {
        "/kernel": "/0", "/bias": "/1",
        "/gamma": "/0", "/beta": "/1",
        "/moving_mean": "/2", "/moving_variance": "/3",
    }
    for old, new in var_map.items():
        if path.endswith(old):
            path = path[:-len(old)] + new
            break

    # Only strip auto-generated dense_N (with number) inside Sequential layers
    # Keep intentional 'dense' layer names (e.g. encoder/dense/0)
    import re
    path = re.sub(r'/dense_\d+/(\d+)$', r'/\1', path)

    return path


def load_weights_by_structure(model, filepath):
    """Load weights matching by normalized structural paths.
    Handles Keras 3 path format differences between save and load.
    """
    ckpt_weights = _load_h5_weights(filepath)

    ckpt_normalized = {}
    for k, v in ckpt_weights.items():
        nk = _normalize_ckpt_key(k)
        ckpt_normalized[nk] = v
        # Strip 'layers/' anywhere in path for matching
        nk_stripped = nk.replace("/layers/", "/")
        if nk_stripped.startswith("layers/"):
            nk_stripped = nk_stripped[len("layers/"):]
        if nk_stripped != nk:
            ckpt_normalized[nk_stripped] = v

    root_name = model.name
    all_vars = model.trainable_variables + model.non_trainable_variables
    loaded = 0
    missed = []

    import re

    for var in all_vars:
        path = getattr(var, 'path', var.name)
        norm_path = _normalize_model_path(path, root_name)

        # Try direct match, then fallback stripping plain 'dense/' from Sequential
        candidates = [norm_path]
        alt = re.sub(r'/dense/(\d+)$', r'/\1', norm_path)
        if alt != norm_path:
            candidates.append(alt)

        matched = False
        for candidate in candidates:
            if candidate in ckpt_normalized:
                arr = ckpt_normalized[candidate]
                if arr.shape == tuple(var.shape):
                    var.assign(tf.constant(arr))
                    loaded += 1
                    matched = True
                    break
                else:
                    missed.append((norm_path, var.shape, arr.shape))
                    matched = True
                    break
        if not matched:
            missed.append((norm_path, var.shape, None))

    if missed:
        print(f"[CsiNet Engine] WARNING: {len(missed)} vars not loaded:")
        for nm, vs, cs in missed[:5]:
            print(f"  {nm} model={vs} ckpt={cs}")
    print(f"[CsiNet Engine] Loaded {loaded}/{len(all_vars)} variables from {filepath}")
    return loaded


class CsiNetInferenceEngine:
    """Runs CsiNet encoder/decoder for real-time CSI compression."""

    def __init__(self, mode="baseline", compression_ratio=1/4,
                 checkpoint_dir="/workspace/csinet_checkpoints",
                 scenario="UMi_NLOS",
                 diff_enabled=False, diff_threshold=0.01,
                 diff_max_stale=100):
        self.mode = mode
        self.gamma = compression_ratio
        self.scenario = scenario
        self.ckpt_dir = checkpoint_dir
        self._codeword_dim = int(NT * NC_PRIME * compression_ratio)

        self.diff_conditioner = None
        if diff_enabled and mode == "conditioned":
            self.diff_conditioner = DifferentialConditioner(
                cond_dim=COND_DIM, threshold=diff_threshold,
                max_stale_slots=diff_max_stale)

        if mode == "baseline":
            self._load_baseline()
        elif mode == "conditioned":
            self._load_conditioned()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _load_baseline(self):
        self.model = CsiNetAutoencoder(NT, NC_PRIME, self.gamma)
        self.model.build(input_shape=(None, 2, NT, NC_PRIME))
        tag = f"{self.scenario}_gamma{self.gamma:.4f}"
        ckpt = os.path.join(self.ckpt_dir, f"csinet_{tag}_best.weights.h5")
        self.model.load_weights(ckpt)
        print(f"[CsiNet Engine] Loaded baseline: {ckpt}")

    def _load_conditioned(self):
        tf.keras.backend.clear_session()

        self.stat_ae = StatisticsAutoencoder(
            COV_VEC_DIM, PDP_DIM, COV_LATENT, PDP_LATENT)
        self.stat_ae([tf.zeros([1, COV_VEC_DIM]), tf.zeros([1, PDP_DIM])])
        stat_ckpt = os.path.join(self.ckpt_dir,
                                 f"stat_ae_{self.scenario}.weights.h5")
        load_weights_by_structure(self.stat_ae, stat_ckpt)

        self.model = ConditionedCsiNet(NT, NC_PRIME, self.gamma, COND_DIM)
        dummy_x = tf.zeros([1, 2, NT, NC_PRIME])
        dummy_c = tf.zeros([1, COND_DIM])
        self.model([dummy_x, dummy_c])

        tag = f"{self.scenario}_gamma{self.gamma:.4f}"
        ckpt = os.path.join(self.ckpt_dir,
                            f"cond_csinet_{tag}_best.weights.h5")
        load_weights_by_structure(self.model, ckpt)

    def _preprocess(self, H_freq):
        """H_freq: (n_rx, n_tx, Nsc) complex -> (1, 2, Nt, Nc') float32"""
        h = np.mean(H_freq, axis=0)  # (Nt, Nsc)

        Nt, Nsc = h.shape
        F_Nt = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)
        h_angular = F_Nt.conj().T @ h
        h_delay = np.fft.ifft(h_angular, axis=-1)
        h_trunc = h_delay[:, :NC_PRIME]

        power = np.mean(np.abs(h_trunc) ** 2)
        h_norm = h_trunc / np.sqrt(max(power, 1e-10))

        x = np.stack([h_norm.real, h_norm.imag], axis=0)  # (2, Nt, Nc')
        return x.astype(np.float32)[np.newaxis], power

    def _postprocess(self, x_hat, power):
        """(1, 2, Nt, Nc') float32 -> (Nt, Nsc) complex"""
        x_hat = x_hat[0]  # (2, Nt, Nc')
        h_ad = x_hat[0] + 1j * x_hat[1]  # (Nt, Nc')
        h_ad *= np.sqrt(power)

        Nt = h_ad.shape[0]
        h_freq_angular = np.fft.fft(h_ad, n=72, axis=-1)
        F_Nt = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)
        h_freq = F_Nt @ h_freq_angular

        return h_freq

    def encode_decode(self, H_freq, R_H=None, pdp=None):
        """Full encode-decode pipeline.
        H_freq: (n_rx, n_tx, Nsc) complex
        Returns: H_hat (Nt, Nsc) complex, codeword (M,)
        """
        x, power = self._preprocess(H_freq)
        x_tf = tf.constant(x)

        if self.mode == "baseline":
            z = self.model.encoder(x_tf, training=False)
            x_hat = self.model.decoder(z, training=False).numpy()
        else:
            if R_H is not None and pdp is not None:
                r_vec = vectorize_covariance(
                    tf.constant(R_H[np.newaxis].astype(np.complex64)))
                pdp_tf = tf.constant(pdp[np.newaxis].astype(np.float32))
                cond = self.stat_ae.get_condition_vector(r_vec, pdp_tf)
            else:
                cond = tf.zeros([1, COND_DIM])

            z = self.model.encoder(x_tf, cond, training=False)
            x_hat = self.model.decoder(z, cond, training=False).numpy()

        codeword = z.numpy().flatten()
        H_hat = self._postprocess(x_hat, power)
        return H_hat, codeword

    def _compute_cond_vector(self, R_H, pdp):
        """Compute the conditioning vector from R_H and PDP via stat_ae."""
        if R_H is not None and pdp is not None:
            r_vec = vectorize_covariance(
                tf.constant(R_H[np.newaxis].astype(np.complex64)))
            pdp_tf = tf.constant(pdp[np.newaxis].astype(np.float32))
            cond = self.stat_ae.get_condition_vector(r_vec, pdp_tf)
            return cond.numpy().flatten()
        return np.zeros(COND_DIM, dtype=np.float32)

    def encode_decode_differential(self, H_freq, R_H=None, pdp=None,
                                   cell_idx=0, ue_idx=0):
        """Delta-encoded conditioning pipeline.

        The encoder always sees the full conditioning vector (UE-side, no
        compression needed for local computation).  The decoder (gNB-side)
        uses the accumulated vector, which equals the full vector when an
        update is sent, or a cached version when the delta is skipped.

        Returns: (H_hat, codeword, diff_info_dict)
        """
        if self.mode != "conditioned" or self.diff_conditioner is None:
            H_hat, codeword = self.encode_decode(H_freq, R_H, pdp)
            return H_hat, codeword, {
                "was_updated": True, "delta_norm": 0.0,
                "overhead_bits": COND_DIM * 32,
                "codeword_bits": self._codeword_dim * 32,
                "total_bits": (COND_DIM + self._codeword_dim) * 32,
                "stale_count": 0, "relative_change": 0.0,
            }

        x, power = self._preprocess(H_freq)
        x_tf = tf.constant(x)

        c_full = self._compute_cond_vector(R_H, pdp)

        key = (cell_idx, ue_idx)
        info = self.diff_conditioner.update(
            key, c_full, codeword_dim=self._codeword_dim)

        cond_enc = tf.constant(c_full[np.newaxis], dtype=tf.float32)
        z = self.model.encoder(x_tf, cond_enc, training=False)

        cond_dec = tf.constant(info.c_used[np.newaxis], dtype=tf.float32)
        x_hat = self.model.decoder(z, cond_dec, training=False).numpy()

        codeword = z.numpy().flatten()
        H_hat = self._postprocess(x_hat, power)

        diff_info = {
            "was_updated": info.was_updated,
            "delta_norm": info.delta_norm,
            "relative_change": info.relative_change,
            "overhead_bits": info.overhead_bits,
            "codeword_bits": info.codeword_bits,
            "total_bits": info.total_bits,
            "stale_count": info.stale_count,
        }
        return H_hat, codeword, diff_info

    def get_diff_summary(self):
        """Return accumulated differential encoding statistics."""
        if self.diff_conditioner is not None:
            return self.diff_conditioner.get_summary()
        return {}
