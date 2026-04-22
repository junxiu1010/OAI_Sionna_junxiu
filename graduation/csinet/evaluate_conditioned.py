#!/usr/bin/env python3
"""
Phase 3: 통계-조건부 CsiNet 오프라인 평가
==========================================
CsiNet baseline vs Conditioned CsiNet vs Type 2 비교
Ablation: 공분산만 / PDP만 / 둘 다

docker exec -e CUDA_VISIBLE_DEVICES=0 -e TF_CPP_MIN_LOG_LEVEL=3 sionna-proxy \
    python3 /workspace/evaluate_conditioned.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

sys.path.insert(0, os.path.dirname(__file__))
from models.csinet import CsiNetAutoencoder, nmse_loss, cosine_similarity
from models.stat_autoencoder import StatisticsAutoencoder, vectorize_covariance
from models.conditioned_csinet import ConditionedCsiNet

DATA_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/csinet_datasets")
CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
OUT_DIR = os.environ.get("CSINET_OUT_DIR", "/workspace/csinet_results")
os.makedirs(OUT_DIR, exist_ok=True)

NT = 4
NR = 2
NC_PRIME = 32
COV_DIM = NT * NR
COV_VEC_DIM = COV_DIM * (COV_DIM + 1)
COV_LATENT = 32
PDP_LATENT = 16
COND_DIM = COV_LATENT + PDP_LATENT  # 48
SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_LOS", "UMa_NLOS"]
GAMMAS = [1/4, 1/8, 1/16, 1/32]
BATCH_SIZE = 256


def evaluate():
    all_results = []

    for scenario in SCENARIOS:
        path = os.path.join(DATA_DIR, f"preprocessed_{scenario}.h5")
        with h5py.File(path, "r") as f:
            X_test = f["X_test"][:].astype(np.float32)
            R_H = f["R_H"][:].astype(np.complex64)
            PDP = f["PDP"][:].astype(np.float32)
            loc_test = f["loc_idx_test"][:]

        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")

        # Load Stage 1
        pdp_dim = PDP.shape[-1]
        stat_ae = StatisticsAutoencoder(COV_VEC_DIM, pdp_dim, COV_LATENT, PDP_LATENT)
        stat_ckpt = os.path.join(CKPT_DIR, f"stat_ae_{scenario}.weights.h5")
        if not os.path.exists(stat_ckpt):
            print(f"  SKIP: no Stage 1 checkpoint")
            continue

        # Build stat_ae by calling it once
        dummy_r = tf.zeros([1, COV_VEC_DIM])
        dummy_p = tf.zeros([1, pdp_dim])
        stat_ae([dummy_r, dummy_p])
        stat_ae.load_weights(stat_ckpt)

        # Prepare conditioning for test set
        R_test = R_H[loc_test]
        PDP_test = PDP[loc_test]
        R_vec_test = vectorize_covariance(tf.constant(R_test)).numpy().astype(np.float32)
        cond_test = stat_ae.get_condition_vector(
            tf.constant(R_vec_test), tf.constant(PDP_test)).numpy()

        for gamma in GAMMAS:
            # --- CsiNet baseline ---
            csinet_tag = f"{scenario}_gamma{gamma:.4f}"
            csinet_ckpt = os.path.join(CKPT_DIR, f"csinet_{csinet_tag}_best.weights.h5")
            baseline_nmse_dB = None
            if os.path.exists(csinet_ckpt):
                model_bl = CsiNetAutoencoder(NT, NC_PRIME, gamma)
                _ = model_bl(tf.zeros([1, 2, NT, NC_PRIME]))
                model_bl.load_weights(csinet_ckpt)
                ds = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)
                bl_nmses = [nmse_loss(x, model_bl(x, training=False)).numpy() for x in ds]
                baseline_nmse_dB = 10 * np.log10(np.mean(bl_nmses) + 1e-10)

            # --- Conditioned CsiNet ---
            cond_tag = f"{scenario}_gamma{gamma:.4f}"
            cond_ckpt = os.path.join(CKPT_DIR, f"cond_csinet_{cond_tag}_best.weights.h5")
            if not os.path.exists(cond_ckpt):
                print(f"  gamma={gamma:.4f}: no conditioned checkpoint")
                continue

            model_cond = ConditionedCsiNet(NT, NC_PRIME, gamma, COND_DIM)
            # Build
            dummy_x = tf.zeros([1, 2, NT, NC_PRIME])
            dummy_c = tf.zeros([1, COND_DIM])
            model_cond([dummy_x, dummy_c])
            model_cond.load_weights(cond_ckpt)

            ds = tf.data.Dataset.from_tensor_slices((X_test, cond_test)).batch(BATCH_SIZE)
            cond_nmses, cond_coss = [], []
            for x_b, c_b in ds:
                x_hat = model_cond([x_b, c_b], training=False)
                cond_nmses.append(nmse_loss(x_b, x_hat).numpy())
                cond_coss.append(cosine_similarity(x_b, x_hat).numpy())

            cond_nmse_dB = 10 * np.log10(np.mean(cond_nmses) + 1e-10)
            cond_cos = np.mean(cond_coss)

            gain = (baseline_nmse_dB - cond_nmse_dB) if baseline_nmse_dB else 0.0

            print(f"  gamma={gamma:.4f}: "
                  f"Baseline={baseline_nmse_dB:.2f} dB, "
                  f"Conditioned={cond_nmse_dB:.2f} dB, "
                  f"gain={gain:+.2f} dB, cos={cond_cos:.4f}")

            all_results.append({
                "scenario": scenario,
                "gamma": float(gamma),
                "baseline_nmse_dB": float(baseline_nmse_dB) if baseline_nmse_dB else None,
                "conditioned_nmse_dB": float(cond_nmse_dB),
                "conditioned_cos": float(cond_cos),
                "gain_dB": float(gain),
            })

    with open(os.path.join(OUT_DIR, "conditioned_evaluation.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/conditioned_evaluation.json")


if __name__ == "__main__":
    evaluate()
