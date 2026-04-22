#!/usr/bin/env python3
"""
Phase 1: Sionna 기반 CsiNet 학습용 채널 데이터셋 생성
=====================================================
sionna-proxy 컨테이너 내에서 실행:
  docker exec -e CUDA_VISIBLE_DEVICES=0 sionna-proxy python3 /workspace/generate_dataset.py

출력:
  dataset_{scenario}.h5  (per scenario)
    - H_train:    (N_train, Nr, Nt, Nsc)  complex64
    - H_val:      (N_val,   Nr, Nt, Nsc)  complex64
    - H_test:     (N_test,  Nr, Nt, Nsc)  complex64
    - R_H:        (N_loc, Nt*Nr, Nt*Nr)   complex64  (per-location covariance)
    - PDP:        (N_loc, N_tau)           float32    (per-location PDP)
    - loc_idx_*:  location index for each sample
"""

import os, sys, time
import numpy as np
import h5py
import tensorflow as tf
from sionna.phy.channel.tr38901 import UMi, UMa, PanelArray
from sionna.phy.channel import GenerateOFDMChannel
from sionna.phy.ofdm import ResourceGrid

# ── Configuration ──────────────────────────────────────────────
NT = 4          # gNB antennas (2x1 dual-pol)
NR = 2          # UE antennas  (1x1 dual-pol)
FFT_SIZE = 128  # subcarriers for generation (will truncate to active)
NSC_ACTIVE = 72 # active subcarriers (106 PRB too big for memory, use 72 for tractability)
SCS_HZ = 30e3
NUM_OFDM = 14
FC = 3.5e9

NUM_LOCATIONS = 1000
SAMPLES_STAT = 50        # per location, for covariance/PDP estimation
SAMPLES_TRAIN = 60       # per location, for training  -> 60,000 total
SAMPLES_VAL = 5          # per location, for validation -> 5,000 total
SAMPLES_TEST = 5         # per location, for testing    -> 5,000 total
SAMPLES_PER_LOC = SAMPLES_STAT + SAMPLES_TRAIN + SAMPLES_VAL + SAMPLES_TEST

BATCH_SIZE = 32
CELL_RADIUS = 200.0  # meters
BS_HEIGHT = 25.0
UE_HEIGHT = 1.5

SCENARIOS = {
    "UMi_LOS":  {"model": "UMi", "los": True},
    "UMi_NLOS": {"model": "UMi", "los": False},
    "UMa_LOS":  {"model": "UMa", "los": True},
    "UMa_NLOS": {"model": "UMa", "los": False},
}

OUT_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/csinet_datasets")
os.makedirs(OUT_DIR, exist_ok=True)


def create_antenna_arrays():
    bs = PanelArray(num_rows_per_panel=2, num_cols_per_panel=1,
                    polarization="dual", polarization_type="cross",
                    antenna_pattern="38.901", carrier_frequency=FC)
    ut = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                    polarization="dual", polarization_type="cross",
                    antenna_pattern="omni", carrier_frequency=FC)
    return bs, ut


def create_channel_model(scenario_cfg, bs, ut):
    ModelClass = UMi if scenario_cfg["model"] == "UMi" else UMa
    model = ModelClass(carrier_frequency=FC, o2i_model="low",
                       ut_array=ut, bs_array=bs, direction="downlink")
    return model


def generate_location_batch(model, rg, gen_ch, n_samples, los=None):
    """Generate channel samples for ONE random location."""
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    all_h = []

    # Random location within cell
    angle = np.random.uniform(0, 2 * np.pi)
    dist = np.random.uniform(10, CELL_RADIUS)
    x = dist * np.cos(angle)
    y = dist * np.sin(angle)

    for b in range(n_batches):
        bs_this = min(BATCH_SIZE, n_samples - b * BATCH_SIZE)

        ut_loc = tf.constant(np.tile([[[x, y, UE_HEIGHT]]], [bs_this, 1, 1]),
                             dtype=tf.float32)
        bs_loc = tf.constant(np.tile([[[0., 0., BS_HEIGHT]]], [bs_this, 1, 1]),
                             dtype=tf.float32)
        in_state = tf.zeros([bs_this, 1], dtype=tf.bool)
        ut_orient = tf.zeros([bs_this, 1, 3], dtype=tf.float32)
        bs_orient = tf.zeros([bs_this, 1, 3], dtype=tf.float32)
        ut_vel = tf.zeros([bs_this, 1, 3], dtype=tf.float32)

        model.set_topology(ut_loc=ut_loc, bs_loc=bs_loc,
                           in_state=in_state,
                           ut_orientations=ut_orient,
                           bs_orientations=bs_orient,
                           ut_velocities=ut_vel,
                           los=los)

        h = gen_ch(batch_size=bs_this)
        # h: (batch, 1, Nr, 1, Nt, num_ofdm, fft_size)
        h_np = h.numpy()
        # Reshape to (batch, Nr, Nt, fft_size) using first OFDM symbol
        h_np = h_np[:, 0, :, 0, :, 0, :]  # (batch, Nr, Nt, fft_size)
        all_h.append(h_np)

    return np.concatenate(all_h, axis=0)[:n_samples]


def compute_statistics(h_stat):
    """Compute covariance and PDP from statistical samples.
    h_stat: (N, Nr, Nt, Nsc) complex
    """
    N, Nr, Nt, Nsc = h_stat.shape
    # Covariance: E[vec(H) vec(H)^H] per subcarrier, then average
    h_vec = h_stat.reshape(N, Nr * Nt, Nsc)  # (N, Nr*Nt, Nsc)
    R_H = np.zeros((Nr * Nt, Nr * Nt), dtype=np.complex64)
    for sc in range(Nsc):
        h_sc = h_vec[:, :, sc]  # (N, Nr*Nt)
        R_H += (h_sc.conj().T @ h_sc) / N
    R_H /= Nsc

    # PDP: average power per delay tap (via IFFT along subcarrier axis)
    h_delay = np.fft.ifft(h_stat, axis=-1)  # (N, Nr, Nt, Nsc)
    pdp = np.mean(np.abs(h_delay) ** 2, axis=(0, 1, 2))  # (Nsc,)

    return R_H, pdp


def generate_scenario(name, cfg):
    print(f"\n{'='*60}", flush=True)
    print(f"Generating scenario: {name}", flush=True)
    print(f"  Model: {cfg['model']}, LOS: {cfg['los']}", flush=True)
    print(f"  Locations: {NUM_LOCATIONS}, Samples/loc: {SAMPLES_PER_LOC}", flush=True)
    print(f"{'='*60}", flush=True)

    bs, ut = create_antenna_arrays()
    model = create_channel_model(cfg, bs, ut)
    rg = ResourceGrid(num_ofdm_symbols=NUM_OFDM, fft_size=FFT_SIZE,
                      subcarrier_spacing=SCS_HZ,
                      num_tx=1, num_streams_per_tx=NT)
    gen_ch = GenerateOFDMChannel(model, rg)

    all_h_train, all_h_val, all_h_test = [], [], []
    all_R_H, all_PDP = [], []
    loc_idx_train, loc_idx_val, loc_idx_test = [], [], []

    t0 = time.time()
    for loc_i in range(NUM_LOCATIONS):
        h_all = generate_location_batch(model, rg, gen_ch,
                                        SAMPLES_PER_LOC, los=cfg["los"])
        # h_all: (SAMPLES_PER_LOC, Nr, Nt, FFT_SIZE)

        # Truncate to active subcarriers (center)
        start_sc = (FFT_SIZE - NSC_ACTIVE) // 2
        h_active = h_all[:, :, :, start_sc:start_sc + NSC_ACTIVE]

        # Split
        idx = 0
        h_stat = h_active[idx:idx + SAMPLES_STAT]; idx += SAMPLES_STAT
        h_train = h_active[idx:idx + SAMPLES_TRAIN]; idx += SAMPLES_TRAIN
        h_val = h_active[idx:idx + SAMPLES_VAL]; idx += SAMPLES_VAL
        h_test = h_active[idx:idx + SAMPLES_TEST]; idx += SAMPLES_TEST

        # Statistics
        R_H, pdp = compute_statistics(h_stat)
        all_R_H.append(R_H)
        all_PDP.append(pdp)

        all_h_train.append(h_train)
        all_h_val.append(h_val)
        all_h_test.append(h_test)
        loc_idx_train.extend([loc_i] * SAMPLES_TRAIN)
        loc_idx_val.extend([loc_i] * SAMPLES_VAL)
        loc_idx_test.extend([loc_i] * SAMPLES_TEST)

        if (loc_i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (loc_i + 1) * (NUM_LOCATIONS - loc_i - 1)
            print(f"  [{loc_i+1}/{NUM_LOCATIONS}] "
                  f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s", flush=True)

    H_train = np.concatenate(all_h_train, axis=0).astype(np.complex64)
    H_val = np.concatenate(all_h_val, axis=0).astype(np.complex64)
    H_test = np.concatenate(all_h_test, axis=0).astype(np.complex64)
    R_H_all = np.stack(all_R_H, axis=0).astype(np.complex64)
    PDP_all = np.stack(all_PDP, axis=0).astype(np.float32)

    out_path = os.path.join(OUT_DIR, f"dataset_{name}.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("H_train", data=H_train, compression="gzip")
        f.create_dataset("H_val", data=H_val, compression="gzip")
        f.create_dataset("H_test", data=H_test, compression="gzip")
        f.create_dataset("R_H", data=R_H_all)
        f.create_dataset("PDP", data=PDP_all)
        f.create_dataset("loc_idx_train", data=np.array(loc_idx_train, dtype=np.int32))
        f.create_dataset("loc_idx_val", data=np.array(loc_idx_val, dtype=np.int32))
        f.create_dataset("loc_idx_test", data=np.array(loc_idx_test, dtype=np.int32))
        f.attrs["Nt"] = NT
        f.attrs["Nr"] = NR
        f.attrs["Nsc"] = NSC_ACTIVE
        f.attrs["fft_size"] = FFT_SIZE
        f.attrs["scenario"] = name
        f.attrs["num_locations"] = NUM_LOCATIONS

    total_time = time.time() - t0
    print(f"\n  Saved: {out_path}", flush=True)
    print(f"  H_train: {H_train.shape}, H_val: {H_val.shape}, H_test: {H_test.shape}", flush=True)
    print(f"  R_H: {R_H_all.shape}, PDP: {PDP_all.shape}", flush=True)
    print(f"  Total time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    for name, cfg in SCENARIOS.items():
        generate_scenario(name, cfg)
    print("\nAll scenarios done.", flush=True)
