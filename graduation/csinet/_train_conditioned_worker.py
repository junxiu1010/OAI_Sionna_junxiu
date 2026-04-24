#!/usr/bin/env python3
"""
Conditioned CsiNet worker — 합리적 학습 전략
============================================
- Transfer: layer-by-layer 정확한 전달 (FiLM은 identity 초기화 유지)
- Phase 1: FiLM warmup (FiLM 변수만 학습, LR=1e-3)
- Phase 2: Full fine-tune (전체 변수, LR=3e-4)
- Data: fulldata (floor=1e-30, 필터링 없음)
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
from models.conditioned_csinet import ConditionedCsiNet
from models.stat_autoencoder import StatisticsAutoencoder

CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
DATA_DIR = "/workspace/graduation/csinet/datasets"

NT = 4
NC_PRIME = 32
COV_LATENT = 16
PDP_LATENT = 8
COND_DIM = COV_LATENT + PDP_LATENT
PHASE1_EPOCHS = 100
PHASE2_EPOCHS = 400
BATCH_SIZE = 200
LR_PHASE1 = 1e-3
LR_PHASE2 = 3e-4


# ─── Data ───────────────────────────────────────────────

def load_fulldata(scenario):
    fulldata_path = os.path.join(DATA_DIR, f"preprocessed_{scenario}_fulldata.h5")
    raw_path = os.path.join(DATA_DIR, f"dataset_{scenario}.h5")

    with h5py.File(fulldata_path, "r") as f:
        X_train = f["X_train"][:]
        X_val = f["X_val"][:]
        X_test = f["X_test"][:]

    with h5py.File(raw_path, "r") as f:
        R_H_all = f["R_H"][:]
        PDP_all = f["PDP"][:]
        loc_train = f["loc_idx_train"][:]
        loc_val = f["loc_idx_val"][:]
        loc_test = f["loc_idx_test"][:]

    return (X_train, X_val, X_test,
            R_H_all[loc_train], R_H_all[loc_val], R_H_all[loc_test],
            PDP_all[loc_train], PDP_all[loc_val], PDP_all[loc_test])


# ─── Statistics AE ──────────────────────────────────────

def get_or_train_stat_ae(scenario, R_H_train, PDP_train, R_H_val, PDP_val):
    ckpt = os.path.join(CKPT_DIR, f"stat_ae_{scenario}_fulldata.weights.h5")

    R_H_flat_tr = R_H_train.reshape(len(R_H_train), -1).astype(np.float32)
    PDP_flat_tr = PDP_train.astype(np.float32)
    cov_dim = R_H_flat_tr.shape[-1]
    pdp_dim = PDP_flat_tr.shape[-1]

    stat_ae = StatisticsAutoencoder(
        cov_dim=cov_dim, pdp_dim=pdp_dim,
        cov_latent=COV_LATENT, pdp_latent=PDP_LATENT)
    _ = stat_ae([R_H_flat_tr[:1], PDP_flat_tr[:1]])

    if os.path.exists(ckpt):
        stat_ae.load_weights(ckpt)
        print(f"  StatAE loaded from cache", flush=True)
        return stat_ae

    print(f"  Training StatAE for {scenario}...", flush=True)
    R_H_flat_val = R_H_val.reshape(len(R_H_val), -1).astype(np.float32)
    PDP_flat_val = PDP_val.astype(np.float32)

    STAT_EPOCHS = 200
    steps = STAT_EPOCHS * ((len(R_H_flat_tr) + BATCH_SIZE - 1) // BATCH_SIZE)
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(1e-3, steps))

    train_ds = tf.data.Dataset.from_tensor_slices((R_H_flat_tr, PDP_flat_tr))\
        .shuffle(len(R_H_flat_tr)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((R_H_flat_val, PDP_flat_val))\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def stat_step(rh, pdp):
        with tf.GradientTape() as tape:
            r_hat, pdp_hat, _, _ = stat_ae([rh, pdp], training=True)
            loss = tf.reduce_mean(tf.square(rh - r_hat)) + tf.reduce_mean(tf.square(pdp - pdp_hat))
        grads = tape.gradient(loss, stat_ae.trainable_variables)
        opt.apply_gradients(zip(grads, stat_ae.trainable_variables))
        return loss

    def stat_val_loss():
        vls = []
        for rh, pdp in val_ds:
            r_hat, pdp_hat, _, _ = stat_ae([rh, pdp], training=False)
            vls.append(float((tf.reduce_mean(tf.square(rh - r_hat)) +
                              tf.reduce_mean(tf.square(pdp - pdp_hat))).numpy()))
        return np.mean(vls)

    best_val = float("inf")
    t0 = time.time()
    for ep in range(STAT_EPOCHS):
        for rh, pdp in train_ds:
            stat_step(rh, pdp)
        if (ep + 1) % 50 == 0 or ep == 0:
            vl = stat_val_loss()
            if vl < best_val:
                best_val = vl
                stat_ae.save_weights(ckpt)
            print(f"  StatAE {ep+1}/{STAT_EPOCHS}: val={vl:.6f}, t={time.time()-t0:.0f}s", flush=True)

    stat_ae.load_weights(ckpt)
    return stat_ae


def encode_conditions(stat_ae, R_H, PDP):
    R_flat = R_H.reshape(len(R_H), -1).astype(np.float32)
    P_flat = PDP.astype(np.float32)
    z_cov = stat_ae.cov_enc(R_flat).numpy()
    z_pdp = stat_ae.pdp_enc(P_flat).numpy()
    return np.concatenate([z_cov, z_pdp], axis=-1).astype(np.float32)


# ─── Layer-by-layer Transfer (정확한 방식) ──────────────

def _transfer_layer(src_layer, dst_layer):
    for w_s, w_d in zip(src_layer.weights, dst_layer.weights):
        if w_s.shape == w_d.shape:
            w_d.assign(w_s)


def transfer_baseline_weights(cond_model, bl_path, gamma):
    bl = CsiNetAutoencoder(Nt=NT, Nc_prime=NC_PRIME, compression_ratio=gamma)
    _ = bl(tf.zeros([1, 2, NT, NC_PRIME]))
    bl.load_weights(bl_path)

    bl_enc, cd_enc = bl.encoder, cond_model.encoder
    for bl_l, cd_l in [(bl_enc.conv1, cd_enc.conv1), (bl_enc.bn1, cd_enc.bn1),
                       (bl_enc.conv2, cd_enc.conv2), (bl_enc.bn2, cd_enc.bn2),
                       (bl_enc.conv3, cd_enc.conv3), (bl_enc.bn3, cd_enc.bn3),
                       (bl_enc.dense, cd_enc.dense)]:
        _transfer_layer(bl_l, cd_l)

    bl_dec, cd_dec = bl.decoder, cond_model.decoder
    _transfer_layer(bl_dec.dense, cd_dec.dense)
    _transfer_layer(bl_dec.conv_out, cd_dec.conv_out)

    for i in range(min(len(bl_dec.refine_blocks), len(cd_dec.refine_blocks))):
        bl_b, cd_b = bl_dec.refine_blocks[i], cd_dec.refine_blocks[i]
        _transfer_layer(bl_b.conv1, cd_b.conv1)
        _transfer_layer(bl_b.bn1, cd_b.bn1)
        if bl_b.conv2 is not None and cd_b.conv2 is not None:
            _transfer_layer(bl_b.conv2, cd_b.conv2)
            _transfer_layer(bl_b.bn2, cd_b.bn2)

    print(f"  [Transfer] Layer-by-layer from baseline (FiLM = identity)", flush=True)


# ─── FiLM variable extraction ──────────────────────────

def get_film_variables(cond_model):
    film_layers = []
    for attr in ['film1', 'film2', 'film3']:
        layer = getattr(cond_model.encoder, attr, None)
        if layer is not None:
            film_layers.append(layer)
    for blk in cond_model.decoder.refine_blocks:
        for attr in ['film1', 'film2']:
            layer = getattr(blk, attr, None)
            if layer is not None:
                film_layers.append(layer)

    film_var_ids = set()
    for fl in film_layers:
        for v in fl.trainable_variables:
            film_var_ids.add(id(v))

    return [v for v in cond_model.trainable_variables if id(v) in film_var_ids]


# ─── Main ──────────────────────────────────────────────

if __name__ == "__main__":
    scenario = sys.argv[1]
    gamma = float(sys.argv[2])
    M = max(1, int(2 * NT * NC_PRIME * gamma))
    tag = f"{scenario}_gamma{gamma:.4f}"

    print(f"{'='*60}")
    print(f"  Conditioned: {scenario}, gamma={gamma:.4f}, M={M}")
    print(f"{'='*60}", flush=True)

    data = load_fulldata(scenario)
    (X_train, X_val, X_test,
     R_H_train, R_H_val, R_H_test,
     PDP_train, PDP_val, PDP_test) = data
    print(f"  X_train: {X_train.shape}, R_H: {R_H_train.shape}", flush=True)

    stat_ae = get_or_train_stat_ae(scenario, R_H_train, PDP_train, R_H_val, PDP_val)
    cond_train = encode_conditions(stat_ae, R_H_train, PDP_train)
    cond_val = encode_conditions(stat_ae, R_H_val, PDP_val)
    cond_test = encode_conditions(stat_ae, R_H_test, PDP_test)

    # Build conditioned model
    cond_model = ConditionedCsiNet(
        Nt=NT, Nc_prime=NC_PRIME, compression_ratio=gamma, cond_dim=COND_DIM)
    dummy_x = tf.zeros([1, 2, NT, NC_PRIME])
    dummy_c = tf.zeros([1, COND_DIM])
    _ = cond_model([dummy_x, dummy_c])

    # Layer-by-layer transfer (FiLM stays identity-initialized)
    bl_path = os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5")
    if not os.path.exists(bl_path):
        print(f"  ERROR: baseline not found: {bl_path}", flush=True)
        sys.exit(1)
    transfer_baseline_weights(cond_model, bl_path, gamma)

    # Extract FiLM variables
    film_vars = get_film_variables(cond_model)
    all_vars = list(cond_model.trainable_variables)
    print(f"  FiLM vars: {len(film_vars)}/{len(all_vars)}", flush=True)

    if len(film_vars) == 0:
        print("  WARNING: No FiLM vars found, using all vars for both phases", flush=True)
        film_vars = all_vars

    # Datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, cond_train))\
        .shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, cond_val))\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, cond_test))\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ckpt_path = os.path.join(CKPT_DIR, f"condcsinet_{tag}_best.weights.h5")
    best_val = float("inf")
    t0 = time.time()

    def eval_model():
        vns, vcs = [], []
        for xb, cb in val_ds:
            xh = cond_model([xb, cb], training=False)
            vns.append(float(nmse_loss(xb, xh).numpy()))
            vcs.append(float(cosine_similarity(xb, xh).numpy()))
        return np.mean(vns), np.mean(vcs)

    # ── Phase 1: FiLM warmup (LR=1e-3, FiLM vars only) ──
    print(f"\n  Phase 1: FiLM warmup ({PHASE1_EPOCHS} ep, LR={LR_PHASE1})", flush=True)

    p1_steps = PHASE1_EPOCHS * ((len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE)
    opt1 = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(LR_PHASE1, p1_steps))

    film_var_list = list(film_vars)

    @tf.function
    def p1_step(xb, cb):
        with tf.GradientTape() as tape:
            xh = cond_model([xb, cb], training=True)
            loss = nmse_loss(xb, xh)
        grads = tape.gradient(loss, film_var_list)
        valid_pairs = [(g, v) for g, v in zip(grads, film_var_list) if g is not None]
        if valid_pairs:
            opt1.apply_gradients(valid_pairs)
        return loss

    for ep in range(PHASE1_EPOCHS):
        ls = []
        for xb, cb in train_ds:
            ls.append(float(p1_step(xb, cb).numpy()))
        vl, vc = eval_model()
        if vl < best_val:
            best_val = vl
            cond_model.save_weights(ckpt_path)
        if (ep + 1) % 25 == 0 or ep == 0:
            print(f"  P1 {ep+1:3d}/{PHASE1_EPOCHS}: "
                  f"tr={10*np.log10(np.mean(ls)+1e-10):.2f}, "
                  f"val={10*np.log10(vl+1e-10):.2f} dB, "
                  f"cos={vc:.4f}, t={time.time()-t0:.0f}s", flush=True)

    # ── Phase 2: Full fine-tune (LR=3e-4, all vars) ──
    print(f"\n  Phase 2: Full fine-tune ({PHASE2_EPOCHS} ep, LR={LR_PHASE2})", flush=True)
    cond_model.load_weights(ckpt_path)

    p2_steps = PHASE2_EPOCHS * ((len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE)
    opt2 = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(LR_PHASE2, p2_steps))

    @tf.function
    def p2_step(xb, cb):
        with tf.GradientTape() as tape:
            xh = cond_model([xb, cb], training=True)
            loss = nmse_loss(xb, xh)
        grads = tape.gradient(loss, all_vars)
        valid_pairs = [(g, v) for g, v in zip(grads, all_vars) if g is not None]
        if valid_pairs:
            opt2.apply_gradients(valid_pairs)
        return loss

    for ep in range(PHASE2_EPOCHS):
        ls = []
        for xb, cb in train_ds:
            ls.append(float(p2_step(xb, cb).numpy()))
        vl, vc = eval_model()
        if vl < best_val:
            best_val = vl
            cond_model.save_weights(ckpt_path)
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  P2 {ep+1:3d}/{PHASE2_EPOCHS}: "
                  f"tr={10*np.log10(np.mean(ls)+1e-10):.2f}, "
                  f"val={10*np.log10(vl+1e-10):.2f} dB, "
                  f"cos={vc:.4f}, t={time.time()-t0:.0f}s", flush=True)

    # ── Test ──
    cond_model.load_weights(ckpt_path)
    tns, tcs = [], []
    for xb, cb in test_ds:
        xh = cond_model([xb, cb], training=False)
        tns.append(float(nmse_loss(xb, xh).numpy()))
        tcs.append(float(cosine_similarity(xb, xh).numpy()))

    test_nmse = np.mean(tns)
    test_cos = np.mean(tcs)
    print(f"\n  TEST: NMSE={10*np.log10(test_nmse+1e-10):.2f} dB, cos={test_cos:.4f}", flush=True)

    result = {
        "scenario": scenario, "gamma": gamma, "M": M,
        "test_nmse_dB": round(float(10 * np.log10(test_nmse + 1e-10)), 2),
        "test_cos_sim": round(float(test_cos), 4),
    }
    result_file = os.path.join(CKPT_DIR, f"conditioned_fulldata_{tag}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
