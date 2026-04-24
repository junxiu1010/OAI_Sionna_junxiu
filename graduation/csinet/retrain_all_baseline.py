#!/usr/bin/env python3
"""
전체 시나리오 Baseline CsiNet 재학습 — 데이터 품질 필터링 적용
===============================================================
UMi_LOS, UMa_NLOS를 power-필터링 후 재전처리 & 학습.
(UMi_NLOS는 이미 완료됨)
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

DATA_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/graduation/csinet/datasets")
CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

NT = 4
NC_PRIME = 32
SCENARIOS = ["UMi_LOS", "UMa_NLOS"]
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32]
EPOCHS = 300
BATCH_SIZE = 200
LR = 1e-3


def angular_delay_transform(H, nc_prime=NC_PRIME):
    Nt = H.shape[-2]
    F_Nt = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)
    H_angular = np.einsum("...at,ba->...bt", H, F_Nt.conj())
    H_delay = np.fft.ifft(H_angular, axis=-1)
    return H_delay[..., :nc_prime]


def preprocess_with_filtering(raw_path, power_percentile=30):
    with h5py.File(raw_path, "r") as f:
        H_train_raw = f["H_train"][:]
        H_val_raw = f["H_val"][:]
        H_test_raw = f["H_test"][:]

    results = {}
    threshold = None
    for name, H_raw in [("train", H_train_raw), ("val", H_val_raw), ("test", H_test_raw)]:
        H_avg = np.mean(H_raw, axis=1)
        H_ad = angular_delay_transform(H_avg)
        power = np.mean(np.abs(H_ad)**2, axis=(-2, -1))

        if name == "train":
            threshold = np.percentile(power[power > 0], power_percentile)
            print(f"  Power filter threshold (P{power_percentile}): {threshold:.2e}")

        mask = power > threshold
        H_ad_filtered = H_ad[mask]
        print(f"  {name}: {len(H_raw)} -> {len(H_ad_filtered)} ({100*mask.sum()/len(mask):.1f}%)")

        filt_power = np.mean(np.abs(H_ad_filtered)**2, axis=(-2, -1), keepdims=True)
        filt_power = np.maximum(filt_power, 1e-15)
        H_norm = H_ad_filtered / np.sqrt(filt_power)
        X = np.stack([H_norm.real, H_norm.imag], axis=1).astype(np.float32)

        x_power = np.mean(X**2, axis=(1, 2, 3))
        print(f"    X power: min={x_power.min():.4f}, mean={x_power.mean():.4f}")
        results[f"X_{name}"] = X

    return results["X_train"], results["X_val"], results["X_test"]


def train_one(scenario, gamma, X_train, X_val, X_test):
    M = max(1, int(2 * NT * NC_PRIME * gamma))
    tag = f"{scenario}_gamma{gamma:.4f}"
    print(f"\n{'='*60}", flush=True)
    print(f"  {scenario}, gamma={gamma:.4f}, M={M}", flush=True)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}", flush=True)
    print(f"{'='*60}", flush=True)

    model = CsiNetAutoencoder(Nt=NT, Nc_prime=NC_PRIME, compression_ratio=gamma)
    _ = model(tf.zeros([1, 2, NT, NC_PRIME]))

    total_steps = EPOCHS * ((len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR, decay_steps=total_steps)
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

    best_val_nmse = float("inf")
    t0 = time.time()

    for epoch in range(EPOCHS):
        losses = []
        for x_batch in train_ds:
            loss = train_step(x_batch)
            losses.append(float(loss.numpy()))
        train_nmse = np.mean(losses)

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

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"train={10*np.log10(train_nmse+1e-10):.2f} dB, "
                  f"val={10*np.log10(val_nmse+1e-10):.2f} dB, "
                  f"cos={val_cos:.4f}, elapsed={elapsed:.0f}s", flush=True)

    model.load_weights(os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5"))
    test_nmses, test_coss = [], []
    for x_batch in test_ds:
        tn, tc = eval_step(x_batch)
        test_nmses.append(float(tn.numpy()))
        test_coss.append(float(tc.numpy()))

    test_nmse = np.mean(test_nmses)
    test_cos = np.mean(test_coss)
    total_time = time.time() - t0

    print(f"\n  TEST: NMSE={10*np.log10(test_nmse+1e-10):.2f} dB, "
          f"cos={test_cos:.4f}, time={total_time:.0f}s", flush=True)

    return {
        "scenario": scenario, "gamma": gamma, "M": M,
        "test_nmse_dB": float(10 * np.log10(test_nmse + 1e-10)),
        "test_cos_sim": float(test_cos),
    }


if __name__ == "__main__":
    all_results = []
    for scenario in SCENARIOS:
        raw_path = os.path.join(DATA_DIR, f"dataset_{scenario}.h5")
        print(f"\n=== Re-preprocessing {scenario} ===", flush=True)
        X_train, X_val, X_test = preprocess_with_filtering(raw_path, power_percentile=30)

        save_path = os.path.join(DATA_DIR, f"preprocessed_{scenario}_filtered.h5")
        with h5py.File(save_path, "w") as f:
            f.create_dataset("X_train", data=X_train, compression="gzip")
            f.create_dataset("X_val", data=X_val, compression="gzip")
            f.create_dataset("X_test", data=X_test, compression="gzip")
        print(f"  Saved: {save_path}", flush=True)

        for gamma in COMPRESSION_RATIOS:
            r = train_one(scenario, gamma, X_train, X_val, X_test)
            all_results.append(r)

        nmses = [r['test_nmse_dB'] for r in all_results if r['scenario'] == scenario]
        is_mono = all(nmses[i] <= nmses[i+1] for i in range(len(nmses)-1))
        print(f"\n  {scenario} Monotonic: {'PASS' if is_mono else 'FAIL'}", flush=True)

    result_path = os.path.join(CKPT_DIR, "retrain_all_baseline_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=== All Results ===", flush=True)
    for r in all_results:
        print(f"  {r['scenario']:10s} gamma={r['gamma']:.4f} M={r['M']:3d} | "
              f"NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}", flush=True)
