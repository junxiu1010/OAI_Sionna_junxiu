#!/usr/bin/env python3
"""
Phase 2: CsiNet Baseline 학습 (optimized with tf.function)
============================================================
docker exec -e CUDA_VISIBLE_DEVICES=0 -e TF_CPP_MIN_LOG_LEVEL=3 sionna-proxy \
    python3 /workspace/train_csinet.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from tensorflow import keras

sys.path.insert(0, os.path.dirname(__file__))
from models.csinet import CsiNetAutoencoder, nmse_loss, cosine_similarity

DATA_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/csinet_datasets")
CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

NT = 4
NC_PRIME = 32
SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32]
EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3


def load_data(scenario):
    path = os.path.join(DATA_DIR, f"preprocessed_{scenario}.h5")
    with h5py.File(path, "r") as f:
        X_train = f["X_train"][:].astype(np.float32)
        X_val = f["X_val"][:].astype(np.float32)
        X_test = f["X_test"][:].astype(np.float32)
    return X_train, X_val, X_test


def train_one(scenario, gamma):
    print(f"\n{'='*60}", flush=True)
    print(f"Training CsiNet: {scenario}, gamma={gamma:.4f}", flush=True)
    M = max(1, int(2 * NT * NC_PRIME * gamma))
    print(f"  M={M}, total_params_input={2*NT*NC_PRIME}", flush=True)
    print(f"{'='*60}", flush=True)

    X_train, X_val, X_test = load_data(scenario)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)

    model = CsiNetAutoencoder(Nt=NT, Nc_prime=NC_PRIME, compression_ratio=gamma)
    # Warm-up call to build
    _ = model(tf.zeros([1, 2, NT, NC_PRIME]))

    total_steps = EPOCHS * (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR, decay_steps=total_steps)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    train_ds = tf.data.Dataset.from_tensor_slices(X_train)\
        .shuffle(50000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(X_val)\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(x_batch):
        with tf.GradientTape() as tape:
            x_hat = model(x_batch, training=True)
            loss = nmse_loss(x_batch, x_hat)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def eval_step(x_batch):
        x_hat = model(x_batch, training=False)
        return nmse_loss(x_batch, x_hat), cosine_similarity(x_batch, x_hat)

    best_val_nmse = float("inf")
    tag = f"{scenario}_gamma{gamma:.4f}"

    t0 = time.time()
    for epoch in range(EPOCHS):
        epoch_losses = []
        for x_batch in train_ds:
            loss = train_step(x_batch)
            epoch_losses.append(float(loss.numpy()))

        train_nmse = np.mean(epoch_losses)

        val_nmses, val_coss = [], []
        for x_batch in val_ds:
            vn, vc = eval_step(x_batch)
            val_nmses.append(float(vn.numpy()))
            val_coss.append(float(vc.numpy()))
        val_nmse = np.mean(val_nmses)
        val_cos = np.mean(val_coss)

        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            model.save_weights(os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5"))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"train_NMSE={10*np.log10(train_nmse+1e-10):.2f} dB, "
                  f"val_NMSE={10*np.log10(val_nmse+1e-10):.2f} dB, "
                  f"val_cos={val_cos:.4f}, elapsed={elapsed:.0f}s", flush=True)

    # Test evaluation
    model.load_weights(os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5"))
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)
    test_nmses, test_coss = [], []
    for x_batch in test_ds:
        tn, tc = eval_step(x_batch)
        test_nmses.append(float(tn.numpy()))
        test_coss.append(float(tc.numpy()))

    test_nmse = np.mean(test_nmses)
    test_cos = np.mean(test_coss)
    total_time = time.time() - t0
    print(f"\n  TEST: NMSE={10*np.log10(test_nmse+1e-10):.2f} dB, "
          f"cos_sim={test_cos:.4f}, time={total_time:.0f}s", flush=True)

    result = {
        "scenario": scenario, "gamma": gamma, "M": M,
        "test_nmse_dB": float(10 * np.log10(test_nmse + 1e-10)),
        "test_cos_sim": float(test_cos),
        "best_val_nmse_dB": float(10 * np.log10(best_val_nmse + 1e-10)),
    }
    with open(os.path.join(CKPT_DIR, f"csinet_{tag}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    all_results = []
    for scenario in SCENARIOS:
        for gamma in COMPRESSION_RATIOS:
            r = train_one(scenario, gamma)
            all_results.append(r)

    with open(os.path.join(CKPT_DIR, "csinet_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=== Summary ===", flush=True)
    for r in all_results:
        print(f"  {r['scenario']:10s} gamma={r['gamma']:.4f} M={r['M']:3d} | "
              f"NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}", flush=True)
