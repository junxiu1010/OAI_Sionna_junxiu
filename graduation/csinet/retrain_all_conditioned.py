#!/usr/bin/env python3
"""
전체 시나리오 Conditioned CsiNet 재학습 — 필터링된 데이터 기반
================================================================
Stage 1: Statistics AE 재학습
Stage 2: Conditioned CsiNet (Phase 1: FiLM only, Phase 2: full fine-tune)
재학습된 baseline 체크포인트에서 weights transfer.
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
from models.stat_autoencoder import StatisticsAutoencoder, vectorize_covariance
from models.conditioned_csinet import ConditionedCsiNet

DATA_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/graduation/csinet/datasets")
CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

NT = 4
NR = 2
NC_PRIME = 32
COV_DIM = NT * NR
COV_VEC_DIM = COV_DIM * (COV_DIM + 1)  # 72
COV_LATENT = 32
PDP_LATENT = 16
COND_DIM = COV_LATENT + PDP_LATENT  # 48

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32]
EPOCHS_STAGE1 = 200
EPOCHS_PHASE1 = 100
EPOCHS_PHASE2 = 400
BATCH_SIZE = 200
LR = 1e-3
PATIENCE = 50


def angular_delay_transform(H, nc_prime=NC_PRIME):
    Nt = H.shape[-2]
    F_Nt = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)
    H_angular = np.einsum("...at,ba->...bt", H, F_Nt.conj())
    H_delay = np.fft.ifft(H_angular, axis=-1)
    return H_delay[..., :nc_prime]


def load_filtered_data(scenario, power_percentile=30):
    """Load raw data, filter by power, normalize, return X + R_H/PDP/loc_idx."""
    raw_path = os.path.join(DATA_DIR, f"dataset_{scenario}.h5")
    with h5py.File(raw_path, "r") as f:
        H_train_raw = f["H_train"][:]
        H_val_raw = f["H_val"][:]
        H_test_raw = f["H_test"][:]
        R_H = f["R_H"][:].astype(np.complex64)
        PDP = f["PDP"][:].astype(np.float32)
        loc_train = f["loc_idx_train"][:]
        loc_val = f["loc_idx_val"][:]
        loc_test = f["loc_idx_test"][:]

    results = {}
    threshold = None
    for name, H_raw, loc in [("train", H_train_raw, loc_train),
                              ("val", H_val_raw, loc_val),
                              ("test", H_test_raw, loc_test)]:
        H_avg = np.mean(H_raw, axis=1)
        H_ad = angular_delay_transform(H_avg)
        power = np.mean(np.abs(H_ad)**2, axis=(-2, -1))

        if name == "train":
            threshold = np.percentile(power[power > 0], power_percentile)

        mask = power > threshold
        H_ad_filtered = H_ad[mask]
        loc_filtered = loc[mask]

        filt_power = np.mean(np.abs(H_ad_filtered)**2, axis=(-2, -1), keepdims=True)
        filt_power = np.maximum(filt_power, 1e-15)
        H_norm = H_ad_filtered / np.sqrt(filt_power)
        X = np.stack([H_norm.real, H_norm.imag], axis=1).astype(np.float32)

        print(f"  {name}: {len(H_raw)} -> {len(X)} ({100*mask.sum()/len(mask):.1f}%)")
        results[f"X_{name}"] = X
        results[f"loc_{name}"] = loc_filtered

    return (results["X_train"], results["X_val"], results["X_test"],
            R_H, PDP,
            results["loc_train"], results["loc_val"], results["loc_test"])


def prepare_condition_data(R_H, PDP, loc_indices):
    R_per_sample = R_H[loc_indices]
    PDP_per_sample = PDP[loc_indices]
    R_vec = vectorize_covariance(tf.constant(R_per_sample)).numpy()
    return R_vec.astype(np.float32), PDP_per_sample.astype(np.float32)


def train_stage1(scenario, R_H, PDP):
    print(f"\n--- Stage 1: Statistics AE for {scenario} ---", flush=True)
    pdp_dim = PDP.shape[-1]
    R_vec = vectorize_covariance(tf.constant(R_H)).numpy().astype(np.float32)
    PDP_f = PDP.astype(np.float32)

    stat_ae = StatisticsAutoencoder(COV_VEC_DIM, pdp_dim, COV_LATENT, PDP_LATENT)
    optimizer = keras.optimizers.Adam(LR)

    n_loc = len(R_vec)
    best_loss = float("inf")
    for epoch in range(EPOCHS_STAGE1):
        idx = np.random.permutation(n_loc)
        losses = []
        for start in range(0, n_loc, 64):
            batch_idx = idx[start:start+64]
            r_batch = tf.constant(R_vec[batch_idx])
            p_batch = tf.constant(PDP_f[batch_idx])
            with tf.GradientTape() as tape:
                r_hat, p_hat, _, _ = stat_ae([r_batch, p_batch], training=True)
                loss = tf.reduce_mean(tf.square(r_batch - r_hat)) + \
                       tf.reduce_mean(tf.square(p_batch - p_hat))
            grads = tape.gradient(loss, stat_ae.trainable_variables)
            optimizer.apply_gradients(zip(grads, stat_ae.trainable_variables))
            losses.append(loss.numpy())
        avg_loss = np.mean(losses)
        if avg_loss < best_loss:
            best_loss = avg_loss
            stat_ae.save_weights(os.path.join(CKPT_DIR, f"stat_ae_{scenario}.weights.h5"))
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f} (best={best_loss:.6f})", flush=True)

    stat_ae.load_weights(os.path.join(CKPT_DIR, f"stat_ae_{scenario}.weights.h5"))
    return stat_ae


def _transfer_layer(src, dst):
    for w_s, w_d in zip(src.weights, dst.weights):
        if w_s.shape == w_d.shape:
            w_d.assign(w_s)


def transfer_baseline_weights(cond_model, scenario, gamma):
    tag = f"{scenario}_gamma{gamma:.4f}"
    bl_ckpt = os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5")
    if not os.path.exists(bl_ckpt):
        print(f"  [Transfer] No baseline: {bl_ckpt}", flush=True)
        return False

    baseline = CsiNetAutoencoder(NT, NC_PRIME, gamma)
    _ = baseline(tf.zeros([1, 2, NT, NC_PRIME]))
    baseline.load_weights(bl_ckpt)

    bl_enc, cd_enc = baseline.encoder, cond_model.encoder
    for bl_l, cd_l in [(bl_enc.conv1, cd_enc.conv1), (bl_enc.bn1, cd_enc.bn1),
                       (bl_enc.conv2, cd_enc.conv2), (bl_enc.bn2, cd_enc.bn2),
                       (bl_enc.conv3, cd_enc.conv3), (bl_enc.bn3, cd_enc.bn3),
                       (bl_enc.dense, cd_enc.dense)]:
        _transfer_layer(bl_l, cd_l)

    bl_dec, cd_dec = baseline.decoder, cond_model.decoder
    _transfer_layer(bl_dec.dense, cd_dec.dense)
    _transfer_layer(bl_dec.conv_out, cd_dec.conv_out)

    for i in range(min(len(bl_dec.refine_blocks), len(cd_dec.refine_blocks))):
        bl_b, cd_b = bl_dec.refine_blocks[i], cd_dec.refine_blocks[i]
        _transfer_layer(bl_b.conv1, cd_b.conv1)
        _transfer_layer(bl_b.bn1, cd_b.bn1)
        if bl_b.conv2 and cd_b.conv2:
            _transfer_layer(bl_b.conv2, cd_b.conv2)
            _transfer_layer(bl_b.bn2, cd_b.bn2)

    print(f"  [Transfer] Loaded from {bl_ckpt}", flush=True)
    return True


def _is_film_or_new(var):
    path = getattr(var, 'path', var.name).lower()
    return any(k in path for k in ["fi_lm_layer", "film", "enc_res", "cond_refine_2", "cond_refine_3"])


def train_stage2(scenario, gamma, stat_ae, X_train, X_val, X_test,
                 R_H, PDP, loc_train, loc_val, loc_test):
    print(f"\n--- Stage 2: Cond-CsiNet {scenario}, gamma={gamma:.4f} ---", flush=True)

    R_tr, P_tr = prepare_condition_data(R_H, PDP, loc_train)
    R_va, P_va = prepare_condition_data(R_H, PDP, loc_val)
    R_te, P_te = prepare_condition_data(R_H, PDP, loc_test)

    cond_train = stat_ae.get_condition_vector(tf.constant(R_tr), tf.constant(P_tr)).numpy()
    cond_val = stat_ae.get_condition_vector(tf.constant(R_va), tf.constant(P_va)).numpy()
    cond_test = stat_ae.get_condition_vector(tf.constant(R_te), tf.constant(P_te)).numpy()

    model = ConditionedCsiNet(NT, NC_PRIME, gamma, COND_DIM)
    _ = model([tf.zeros([1, 2, NT, NC_PRIME]), tf.zeros([1, COND_DIM])])
    transfer_baseline_weights(model, scenario, gamma)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, cond_train))\
        .shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, cond_val))\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tag = f"{scenario}_gamma{gamma:.4f}"
    best_val_nmse = float("inf")
    no_improve = 0
    t0 = time.time()

    # Phase 1: FiLM-only
    print(f"  Phase 1: FiLM warmup ({EPOCHS_PHASE1} epochs)", flush=True)
    all_vars = model.trainable_variables
    film_vars = [v for v in all_vars if _is_film_or_new(v)]
    print(f"  Film vars: {len(film_vars)}/{len(all_vars)}", flush=True)
    opt1 = keras.optimizers.Adam(LR)

    @tf.function
    def train_step_p1(x_b, c_b):
        with tf.GradientTape() as tape:
            x_hat = model([x_b, c_b], training=True)
            loss = nmse_loss(x_b, x_hat)
        grads = tape.gradient(loss, film_vars)
        valid = [(tf.clip_by_norm(g, 1.0), v) for g, v in zip(grads, film_vars) if g is not None]
        if valid:
            opt1.apply_gradients(valid)
        return loss

    @tf.function
    def eval_step(x_b, c_b):
        x_hat = model([x_b, c_b], training=False)
        return nmse_loss(x_b, x_hat), cosine_similarity(x_b, x_hat)

    for epoch in range(EPOCHS_PHASE1):
        losses = []
        for x_b, c_b in train_ds:
            losses.append(float(train_step_p1(x_b, c_b).numpy()))
        tr = np.mean(losses)
        vns = []
        for x_b, c_b in val_ds:
            vn, _ = eval_step(x_b, c_b)
            vns.append(float(vn.numpy()))
        vl = np.mean(vns)
        if vl < best_val_nmse:
            best_val_nmse = vl
            model.save_weights(os.path.join(CKPT_DIR, f"cond_csinet_{tag}_best.weights.h5"))
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  P1 {epoch+1:3d}/{EPOCHS_PHASE1}: "
                  f"tr={10*np.log10(tr+1e-10):.2f}, val={10*np.log10(vl+1e-10):.2f} dB, "
                  f"t={time.time()-t0:.0f}s", flush=True)

    # Phase 2: Full fine-tune
    print(f"  Phase 2: Full fine-tune ({EPOCHS_PHASE2} epochs)", flush=True)
    total_p2 = EPOCHS_PHASE2 * ((len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE)
    lr_base = keras.optimizers.schedules.CosineDecay(LR * 0.05, total_p2)
    lr_film = keras.optimizers.schedules.CosineDecay(LR * 0.5, total_p2)
    opt_base = keras.optimizers.Adam(lr_base)
    opt_film = keras.optimizers.Adam(lr_film)

    all_v2 = model.trainable_variables
    film_ids = set(id(v) for v in all_v2 if _is_film_or_new(v))

    @tf.function
    def train_step_p2(x_b, c_b):
        with tf.GradientTape() as tape:
            x_hat = model([x_b, c_b], training=True)
            loss = nmse_loss(x_b, x_hat)
        all_v = model.trainable_variables
        grads = tape.gradient(loss, all_v)
        bg, fg = [], []
        for g, v in zip(grads, all_v):
            if g is None: continue
            (fg if id(v) in film_ids else bg).append((g, v))
        if bg: opt_base.apply_gradients(bg)
        if fg: opt_film.apply_gradients(fg)
        return loss

    no_improve = 0
    for epoch in range(EPOCHS_PHASE2):
        losses = []
        for x_b, c_b in train_ds:
            losses.append(float(train_step_p2(x_b, c_b).numpy()))
        tr = np.mean(losses)
        vns, vcs = [], []
        for x_b, c_b in val_ds:
            vn, vc = eval_step(x_b, c_b)
            vns.append(float(vn.numpy()))
            vcs.append(float(vc.numpy()))
        vl, vc = np.mean(vns), np.mean(vcs)
        if vl < best_val_nmse:
            best_val_nmse = vl
            model.save_weights(os.path.join(CKPT_DIR, f"cond_csinet_{tag}_best.weights.h5"))
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}", flush=True)
            break
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  P2 {epoch+1:3d}/{EPOCHS_PHASE2}: "
                  f"tr={10*np.log10(tr+1e-10):.2f}, val={10*np.log10(vl+1e-10):.2f} dB, "
                  f"cos={vc:.4f}, t={time.time()-t0:.0f}s", flush=True)

    model.load_weights(os.path.join(CKPT_DIR, f"cond_csinet_{tag}_best.weights.h5"))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, cond_test)).batch(BATCH_SIZE)
    tns, tcs = [], []
    for x_b, c_b in test_ds:
        tn, tc = eval_step(x_b, c_b)
        tns.append(float(tn.numpy()))
        tcs.append(float(tc.numpy()))
    test_nmse, test_cos = np.mean(tns), np.mean(tcs)
    print(f"  TEST: NMSE={10*np.log10(test_nmse+1e-10):.2f} dB, cos={test_cos:.4f}", flush=True)

    return {
        "scenario": scenario, "gamma": gamma,
        "test_nmse_dB": float(10 * np.log10(test_nmse + 1e-10)),
        "test_cos_sim": float(test_cos),
    }


if __name__ == "__main__":
    all_results = []
    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  Processing: {scenario}")
        print(f"{'='*60}", flush=True)

        X_train, X_val, X_test, R_H, PDP, loc_tr, loc_va, loc_te = \
            load_filtered_data(scenario, power_percentile=30)

        stat_ae = train_stage1(scenario, R_H, PDP)

        for gamma in COMPRESSION_RATIOS:
            r = train_stage2(scenario, gamma, stat_ae,
                             X_train, X_val, X_test,
                             R_H, PDP, loc_tr, loc_va, loc_te)
            all_results.append(r)

    result_path = os.path.join(CKPT_DIR, "retrain_conditioned_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=== Conditioned Retraining Summary ===", flush=True)
    for r in all_results:
        print(f"  {r['scenario']:10s} gamma={r['gamma']:.4f} | "
              f"NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}", flush=True)
