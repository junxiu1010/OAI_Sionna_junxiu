#!/usr/bin/env python3
"""Worker: 단일 Baseline CsiNet 모델 학습 (subprocess에서 호출)."""
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

CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

NT = 4
NC_PRIME = 32
EPOCHS = 300
BATCH_SIZE = 200
LR = 1e-3
POWER_FLOOR = 1e-30


def angular_delay_transform(H, nc_prime=NC_PRIME):
    Nt = H.shape[-2]
    F_Nt = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)
    H_angular = np.einsum("...at,ba->...bt", H, F_Nt.conj())
    H_delay = np.fft.ifft(H_angular, axis=-1)
    return H_delay[..., :nc_prime]


def preprocess_no_filter(raw_path):
    with h5py.File(raw_path, "r") as f:
        H_train_raw = f["H_train"][:]
        H_val_raw = f["H_val"][:]
        H_test_raw = f["H_test"][:]

    results = {}
    for name, H_raw in [("train", H_train_raw), ("val", H_val_raw), ("test", H_test_raw)]:
        H_avg = np.mean(H_raw, axis=1)
        H_ad = angular_delay_transform(H_avg)
        power = np.mean(np.abs(H_ad)**2, axis=(-2, -1), keepdims=True)
        power = np.maximum(power, POWER_FLOOR)
        H_norm = H_ad / np.sqrt(power)
        X = np.stack([H_norm.real, H_norm.imag], axis=1).astype(np.float32)
        x_power = np.mean(X**2, axis=(1, 2, 3))
        print(f"  {name}: N={len(X)}, power: min={x_power.min():.4f}, mean={x_power.mean():.4f}", flush=True)
        results[f"X_{name}"] = X
    return results["X_train"], results["X_val"], results["X_test"]


def save_fulldata_if_needed(scenario, X_train, X_val, X_test):
    save_path = f"/workspace/graduation/csinet/datasets/preprocessed_{scenario}_fulldata.h5"
    if not os.path.exists(save_path):
        with h5py.File(save_path, "w") as f:
            f.create_dataset("X_train", data=X_train, compression="gzip")
            f.create_dataset("X_val", data=X_val, compression="gzip")
            f.create_dataset("X_test", data=X_test, compression="gzip")
        print(f"  Saved: {save_path}", flush=True)


if __name__ == "__main__":
    scenario = sys.argv[1]
    gamma = float(sys.argv[2])
    M = max(1, int(2 * NT * NC_PRIME * gamma))
    tag = f"{scenario}_gamma{gamma:.4f}"

    print(f"{'='*60}")
    print(f"  Training: {scenario}, gamma={gamma:.4f}, M={M}")
    print(f"{'='*60}", flush=True)

    raw_path = f"/workspace/graduation/csinet/datasets/dataset_{scenario}.h5"
    X_train, X_val, X_test = preprocess_no_filter(raw_path)
    save_fulldata_if_needed(scenario, X_train, X_val, X_test)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)

    model = CsiNetAutoencoder(Nt=NT, Nc_prime=NC_PRIME, compression_ratio=gamma)
    _ = model(tf.zeros([1, 2, NT, NC_PRIME]))

    total_steps = EPOCHS * ((len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE)
    lr_schedule = keras.optimizers.schedules.CosineDecay(LR, total_steps)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    train_ds = tf.data.Dataset.from_tensor_slices(X_train)\
        .shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(X_val)\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices(X_test)\
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

    best_val = float("inf")
    ckpt_path = os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5")
    t0 = time.time()

    for epoch in range(EPOCHS):
        losses = []
        for xb in train_ds:
            losses.append(float(train_step(xb).numpy()))
        tr = np.mean(losses)

        vns, vcs = [], []
        for xb in val_ds:
            vn, vc = eval_step(xb)
            vns.append(float(vn.numpy()))
            vcs.append(float(vc.numpy()))
        vl = np.mean(vns)

        if vl < best_val:
            best_val = vl
            model.save_weights(ckpt_path)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  {epoch+1:3d}/{EPOCHS}: tr={10*np.log10(tr+1e-10):.2f}, "
                  f"val={10*np.log10(vl+1e-10):.2f} dB, cos={np.mean(vcs):.4f}, "
                  f"t={elapsed:.0f}s", flush=True)

    model.load_weights(ckpt_path)
    tns, tcs = [], []
    for xb in test_ds:
        tn, tc = eval_step(xb)
        tns.append(float(tn.numpy()))
        tcs.append(float(tc.numpy()))

    test_nmse = np.mean(tns)
    test_cos = np.mean(tcs)
    print(f"\n  TEST: NMSE={10*np.log10(test_nmse+1e-10):.2f} dB, cos={test_cos:.4f}", flush=True)

    result = {
        "scenario": scenario, "gamma": gamma, "M": M,
        "test_nmse_dB": round(float(10 * np.log10(test_nmse + 1e-10)), 2),
        "test_cos_sim": round(float(test_cos), 4),
    }
    result_file = os.path.join(CKPT_DIR, f"baseline_fulldata_{tag}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
