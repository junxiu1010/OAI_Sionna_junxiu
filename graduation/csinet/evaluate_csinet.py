#!/usr/bin/env python3
"""
Phase 2: CsiNet 오프라인 평가 + Type 2 Codebook 대비 비교
==========================================================
docker exec -e CUDA_VISIBLE_DEVICES=0 -e TF_CPP_MIN_LOG_LEVEL=3 sionna-proxy \
    python3 /workspace/evaluate_csinet.py
"""

import os, sys, json
import numpy as np
import h5py
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from models.csinet import CsiNetAutoencoder, nmse_loss, cosine_similarity

DATA_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/csinet_datasets")
CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
OUT_DIR = os.environ.get("CSINET_OUT_DIR", "/workspace/csinet_results")
os.makedirs(OUT_DIR, exist_ok=True)

NT = 4
NC_PRIME = 32
SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_LOS", "UMa_NLOS"]
GAMMAS = [1/4, 1/8, 1/16, 1/32]
BATCH_SIZE = 256


def type2_codebook_nmse(X_test, L=2, phase_bits=3):
    """Approximate Type 2 codebook reconstruction error.
    Simulates L-beam linear combination with phase_bits PSK quantization.
    """
    N, _, Nt, Nc = X_test.shape
    H_complex = X_test[:, 0, :, :] + 1j * X_test[:, 1, :, :]

    # DFT codebook for Nt antennas
    codebook = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)

    nmses = []
    for i in range(N):
        h = H_complex[i]  # (Nt, Nc)
        # Find L strongest beams
        beam_powers = np.abs(codebook.conj().T @ h) ** 2  # (Nt, Nc)
        beam_energy = beam_powers.sum(axis=1)
        top_L = np.argsort(-beam_energy)[:L]

        # Project onto selected beams
        selected = codebook[:, top_L]  # (Nt, L)
        coeffs = selected.conj().T @ h  # (L, Nc)

        # Quantize phases
        n_levels = 2 ** phase_bits
        phases = np.angle(coeffs)
        q_phases = np.round(phases / (2 * np.pi / n_levels)) * (2 * np.pi / n_levels)
        q_coeffs = np.abs(coeffs) * np.exp(1j * q_phases)

        # Reconstruct
        h_hat = selected @ q_coeffs
        mse = np.mean(np.abs(h - h_hat) ** 2)
        power = np.mean(np.abs(h) ** 2)
        nmses.append(mse / max(power, 1e-10))

    return float(np.mean(nmses))


def evaluate():
    all_results = []

    for scenario in SCENARIOS:
        path = os.path.join(DATA_DIR, f"preprocessed_{scenario}.h5")
        with h5py.File(path, "r") as f:
            X_test = f["X_test"][:].astype(np.float32)

        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}, test samples: {X_test.shape[0]}")
        print(f"{'='*50}")

        # Type 2 baseline
        type2_nmse = type2_codebook_nmse(X_test, L=2, phase_bits=3)
        print(f"  Type 2 (L=2, 8PSK): NMSE = {10*np.log10(type2_nmse+1e-10):.2f} dB")

        for gamma in GAMMAS:
            tag = f"{scenario}_gamma{gamma:.4f}"
            ckpt = os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5")
            if not os.path.exists(ckpt):
                print(f"  SKIP gamma={gamma:.4f}: no checkpoint")
                continue

            model = CsiNetAutoencoder(Nt=NT, Nc_prime=NC_PRIME,
                                       compression_ratio=gamma)
            _ = model(tf.zeros([1, 2, NT, NC_PRIME]))
            model.load_weights(ckpt)

            test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)
            nmses, coss = [], []
            for x_b in test_ds:
                x_hat = model(x_b, training=False)
                nmses.append(nmse_loss(x_b, x_hat).numpy())
                coss.append(cosine_similarity(x_b, x_hat).numpy())

            test_nmse = float(np.mean(nmses))
            test_cos = float(np.mean(coss))
            M = model.M
            total_bits_csinet = M * 4  # assuming 4-bit quantization
            # Type 2 bits: L=2 beams, 3-bit phase, Nt=4 ports
            type2_bits = 2 * 4 * 3 + 4  # ~28 bits (approx)

            nmse_dB = 10 * np.log10(test_nmse + 1e-10)
            gain_dB = 10 * np.log10(type2_nmse + 1e-10) - nmse_dB

            print(f"  CsiNet gamma={gamma:.4f} (M={M}): "
                  f"NMSE={nmse_dB:.2f} dB, cos={test_cos:.4f}, "
                  f"gain_vs_Type2={gain_dB:+.2f} dB")

            all_results.append({
                "scenario": scenario,
                "gamma": gamma,
                "M": M,
                "nmse_dB": nmse_dB,
                "cos_sim": test_cos,
                "type2_nmse_dB": 10 * np.log10(type2_nmse + 1e-10),
                "gain_vs_type2_dB": gain_dB,
            })

    with open(os.path.join(OUT_DIR, "csinet_evaluation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/csinet_evaluation.json")


if __name__ == "__main__":
    evaluate()
