#!/usr/bin/env python3
"""
Phase 3: Statistics-Conditioned CsiNet 학습 (v3 — Enhanced)
=============================================================
핵심 개선:
  1. COND_DIM 48 (COV_LATENT=32, PDP_LATENT=16)
  2. 2단계 학습: Phase 1 freeze baseline + Phase 2 differential LR
  3. 500 에폭 + early stopping
  4. Statistics AE 재학습 (확대된 용량)

docker exec -e CUDA_VISIBLE_DEVICES=0 -e TF_CPP_MIN_LOG_LEVEL=3 sionna-proxy \
    python3 /workspace/train_conditioned.py
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

DATA_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/csinet_datasets")
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
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 50


def load_data(scenario):
    path = os.path.join(DATA_DIR, f"preprocessed_{scenario}.h5")
    with h5py.File(path, "r") as f:
        X_train = f["X_train"][:].astype(np.float32)
        X_val = f["X_val"][:].astype(np.float32)
        X_test = f["X_test"][:].astype(np.float32)
        R_H = f["R_H"][:].astype(np.complex64)
        PDP = f["PDP"][:].astype(np.float32)
        loc_train = f["loc_idx_train"][:]
        loc_val = f["loc_idx_val"][:]
        loc_test = f["loc_idx_test"][:]
    return X_train, X_val, X_test, R_H, PDP, loc_train, loc_val, loc_test


def prepare_condition_data(R_H, PDP, loc_indices):
    R_per_sample = R_H[loc_indices]
    PDP_per_sample = PDP[loc_indices]
    R_vec = vectorize_covariance(tf.constant(R_per_sample)).numpy()
    return R_vec.astype(np.float32), PDP_per_sample.astype(np.float32)


def train_stage1(scenario, R_H, PDP):
    print(f"\n--- Stage 1: Statistics AE for {scenario} (enhanced) ---", flush=True)
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
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"  Stage1 Epoch {epoch+1}: loss={avg_loss:.6f} (best={best_loss:.6f})", flush=True)

    stat_ae.load_weights(os.path.join(CKPT_DIR, f"stat_ae_{scenario}.weights.h5"))
    return stat_ae


def _transfer_layer(src, dst):
    for w_s, w_d in zip(src.weights, dst.weights):
        if w_s.shape == w_d.shape:
            w_d.assign(w_s)


def transfer_baseline_weights(cond_model, scenario, gamma):
    """Transfer encoder/decoder weights from trained CsiNet baseline.
    Only transfers conv/bn/dense layers; FiLM and new layers start fresh.
    """
    tag = f"{scenario}_gamma{gamma:.4f}"
    bl_ckpt = os.path.join(CKPT_DIR, f"csinet_{tag}_best.weights.h5")
    if not os.path.exists(bl_ckpt):
        print(f"  [Transfer] No baseline checkpoint: {bl_ckpt}", flush=True)
        return False

    baseline = CsiNetAutoencoder(NT, NC_PRIME, gamma)
    _ = baseline(tf.zeros([1, 2, NT, NC_PRIME]))
    baseline.load_weights(bl_ckpt)

    bl_enc = baseline.encoder
    cd_enc = cond_model.encoder
    for bl_layer, cd_layer in [
        (bl_enc.conv1, cd_enc.conv1), (bl_enc.bn1, cd_enc.bn1),
        (bl_enc.conv2, cd_enc.conv2), (bl_enc.bn2, cd_enc.bn2),
        (bl_enc.conv3, cd_enc.conv3), (bl_enc.bn3, cd_enc.bn3),
        (bl_enc.dense, cd_enc.dense),
    ]:
        _transfer_layer(bl_layer, cd_layer)

    bl_dec = baseline.decoder
    cd_dec = cond_model.decoder
    _transfer_layer(bl_dec.dense, cd_dec.dense)
    _transfer_layer(bl_dec.conv_out, cd_dec.conv_out)

    n_bl_refine = len(bl_dec.refine_blocks)
    for i in range(min(n_bl_refine, len(cd_dec.refine_blocks))):
        bl_blk = bl_dec.refine_blocks[i]
        cd_blk = cd_dec.refine_blocks[i]
        _transfer_layer(bl_blk.conv1, cd_blk.conv1)
        _transfer_layer(bl_blk.bn1, cd_blk.bn1)
        if bl_blk.conv2 is not None and cd_blk.conv2 is not None:
            _transfer_layer(bl_blk.conv2, cd_blk.conv2)
            _transfer_layer(bl_blk.bn2, cd_blk.bn2)

    print(f"  [Transfer] Baseline weights loaded from {bl_ckpt}", flush=True)
    return True


def _is_film_or_new(var):
    """Check if variable belongs to FiLM layers, encoder residual, or new refine blocks."""
    path = getattr(var, 'path', var.name).lower()
    return any(k in path for k in ["fi_lm_layer", "enc_res", "cond_refine_2", "cond_refine_3"])


def train_stage2(scenario, gamma, stat_ae):
    print(f"\n--- Stage 2: Conditioned CsiNet {scenario}, gamma={gamma:.4f} ---", flush=True)

    X_train, X_val, X_test, R_H, PDP, loc_train, loc_val, loc_test = load_data(scenario)
    R_train, P_train = prepare_condition_data(R_H, PDP, loc_train)
    R_val, P_val = prepare_condition_data(R_H, PDP, loc_val)
    R_test, P_test = prepare_condition_data(R_H, PDP, loc_test)

    cond_train = stat_ae.get_condition_vector(
        tf.constant(R_train), tf.constant(P_train)).numpy()
    cond_val = stat_ae.get_condition_vector(
        tf.constant(R_val), tf.constant(P_val)).numpy()
    cond_test = stat_ae.get_condition_vector(
        tf.constant(R_test), tf.constant(P_test)).numpy()

    model = ConditionedCsiNet(NT, NC_PRIME, gamma, COND_DIM)
    _ = model([tf.zeros([1, 2, NT, NC_PRIME]), tf.zeros([1, COND_DIM])])

    transfer_baseline_weights(model, scenario, gamma)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, cond_train))\
        .shuffle(50000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, cond_val))\
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tag = f"{scenario}_gamma{gamma:.4f}"
    best_val_nmse = float("inf")
    no_improve = 0
    t0 = time.time()

    # ================================================================
    # Phase 1: Freeze baseline layers, train only FiLM + new layers
    # ================================================================
    print(f"  === Phase 1: FiLM-only warmup ({EPOCHS_PHASE1} epochs) ===", flush=True)

    all_vars = model.trainable_variables
    film_vars = [v for v in all_vars if _is_film_or_new(v)]
    print(f"  Phase1 trainable: {len(film_vars)}/{len(all_vars)} vars", flush=True)

    opt1 = keras.optimizers.Adam(LR)

    @tf.function
    def train_step_phase1(x_b, c_b):
        with tf.GradientTape() as tape:
            x_hat = model([x_b, c_b], training=True)
            loss = nmse_loss(x_b, x_hat)
        grads = tape.gradient(loss, film_vars)
        clipped = [(tf.clip_by_norm(g, 1.0), v) for g, v in zip(grads, film_vars) if g is not None]
        if clipped:
            opt1.apply_gradients(clipped)
        return loss

    @tf.function
    def eval_step(x_b, c_b):
        x_hat = model([x_b, c_b], training=False)
        return nmse_loss(x_b, x_hat), cosine_similarity(x_b, x_hat)

    for epoch in range(EPOCHS_PHASE1):
        epoch_losses = []
        for x_b, c_b in train_ds:
            loss = train_step_phase1(x_b, c_b)
            epoch_losses.append(float(loss.numpy()))
        train_nmse = np.mean(epoch_losses)

        val_nmses, val_coss = [], []
        for x_b, c_b in val_ds:
            vn, vc = eval_step(x_b, c_b)
            val_nmses.append(float(vn.numpy()))
            val_coss.append(float(vc.numpy()))
        val_nmse = np.mean(val_nmses)

        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            model.save_weights(os.path.join(CKPT_DIR, f"cond_csinet_{tag}_best.weights.h5"))
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  P1 Epoch {epoch+1:3d}/{EPOCHS_PHASE1}: "
                  f"train={10*np.log10(train_nmse+1e-10):.2f} dB, "
                  f"val={10*np.log10(val_nmse+1e-10):.2f} dB, "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    # ================================================================
    # Phase 2: Unfreeze all, differential LR
    # ================================================================
    print(f"  === Phase 2: Full fine-tune ({EPOCHS_PHASE2} epochs, differential LR) ===", flush=True)

    total_steps_p2 = EPOCHS_PHASE2 * (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    lr_base = keras.optimizers.schedules.CosineDecay(LR * 0.05, total_steps_p2)
    lr_film = keras.optimizers.schedules.CosineDecay(LR * 0.5, total_steps_p2)

    opt_base = keras.optimizers.Adam(lr_base)
    opt_film = keras.optimizers.Adam(lr_film)

    all_vars_p2 = model.trainable_variables
    film_var_set = set(id(v) for v in all_vars_p2 if _is_film_or_new(v))
    print(f"  Phase2: {len(film_var_set)} FiLM vars / {len(all_vars_p2)} total vars", flush=True)

    @tf.function
    def train_step_phase2(x_b, c_b):
        with tf.GradientTape() as tape:
            x_hat = model([x_b, c_b], training=True)
            loss = nmse_loss(x_b, x_hat)
        all_vars = model.trainable_variables
        grads = tape.gradient(loss, all_vars)

        base_gv = []
        film_gv = []
        for g, v in zip(grads, all_vars):
            if g is None:
                continue
            if id(v) in film_var_set:
                film_gv.append((g, v))
            else:
                base_gv.append((g, v))

        if base_gv:
            opt_base.apply_gradients(base_gv)
        if film_gv:
            opt_film.apply_gradients(film_gv)
        return loss

    no_improve = 0
    for epoch in range(EPOCHS_PHASE2):
        epoch_losses = []
        for x_b, c_b in train_ds:
            loss = train_step_phase2(x_b, c_b)
            epoch_losses.append(float(loss.numpy()))
        train_nmse = np.mean(epoch_losses)

        val_nmses, val_coss = [], []
        for x_b, c_b in val_ds:
            vn, vc = eval_step(x_b, c_b)
            val_nmses.append(float(vn.numpy()))
            val_coss.append(float(vc.numpy()))
        val_nmse = np.mean(val_nmses)
        val_cos = np.mean(val_coss)

        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            model.save_weights(os.path.join(CKPT_DIR, f"cond_csinet_{tag}_best.weights.h5"))
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)", flush=True)
            break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  P2 Epoch {epoch+1:3d}/{EPOCHS_PHASE2}: "
                  f"train={10*np.log10(train_nmse+1e-10):.2f} dB, "
                  f"val={10*np.log10(val_nmse+1e-10):.2f} dB, cos={val_cos:.4f}, "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    # Final evaluation on test set
    model.load_weights(os.path.join(CKPT_DIR, f"cond_csinet_{tag}_best.weights.h5"))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, cond_test)).batch(BATCH_SIZE)
    test_nmses, test_coss = [], []
    for x_b, c_b in test_ds:
        vn, vc = eval_step(x_b, c_b)
        test_nmses.append(float(vn.numpy()))
        test_coss.append(float(vc.numpy()))
    test_nmse = np.mean(test_nmses)
    test_cos = np.mean(test_coss)
    print(f"  TEST: NMSE={10*np.log10(test_nmse+1e-10):.2f} dB, cos={test_cos:.4f}", flush=True)

    return {
        "scenario": scenario, "gamma": gamma,
        "test_nmse_dB": float(10 * np.log10(test_nmse + 1e-10)),
        "test_cos_sim": float(test_cos),
    }


if __name__ == "__main__":
    all_results = []
    for scenario in SCENARIOS:
        X_train, X_val, X_test, R_H, PDP, loc_train, loc_val, loc_test = load_data(scenario)
        stat_ae = train_stage1(scenario, R_H, PDP)
        for gamma in COMPRESSION_RATIOS:
            r = train_stage2(scenario, gamma, stat_ae)
            all_results.append(r)

    with open(os.path.join(CKPT_DIR, "conditioned_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in all_results:
        print(f"  {r['scenario']:10s} gamma={r['gamma']:.4f} | "
              f"NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}", flush=True)
