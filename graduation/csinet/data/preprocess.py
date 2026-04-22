#!/usr/bin/env python3
"""
Phase 1: 채널 데이터 전처리 — 각도-지연 도메인 변환
===================================================
generate_dataset.py 출력(.h5)을 읽어 CsiNet 입력 형식으로 변환합니다.

변환:
  H (Nr, Nt, Nsc) complex -> H_tilde (2, Nt, Nc') float32
  1) 각 Rx antenna에 대해 DFT: H_tilde = F_Nt^H @ H @ F_Nc  (angular-delay domain)
  2) 지연 축 절단: 처음 Nc' bin만 유지
  3) 실수부/허수부 분리 -> (2, Nt, Nc')
  4) 전력 기준 정규화

출력:
  preprocessed_{scenario}.h5
    - X_train:  (N, 2, Nt, Nc')  float32
    - X_val, X_test: same
    - loc_idx_train, loc_idx_val, loc_idx_test: int32
    - R_H, PDP: copied from raw dataset
"""

import os, sys
import numpy as np
import h5py

NC_PRIME = 32  # truncated delay bins

RAW_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/csinet_datasets")
OUT_DIR = os.environ.get("CSINET_DATA_DIR", "/workspace/csinet_datasets")


def angular_delay_transform(H, nc_prime=NC_PRIME):
    """
    H: (..., Nt, Nsc) complex -> H_tilde: (..., Nt, nc_prime) complex
    Apply spatial DFT along Nt and IFFT along Nsc (delay domain)
    """
    Nt = H.shape[-2]
    Nsc = H.shape[-1]

    # Spatial DFT (antenna domain -> angular domain)
    F_Nt = np.fft.fft(np.eye(Nt), axis=0) / np.sqrt(Nt)
    # H_angular = F_Nt^H @ H => angular domain along antenna axis
    H_angular = np.einsum("...at,ba->...bt", H, F_Nt.conj())

    # Delay domain via IFFT along subcarrier axis
    H_delay = np.fft.ifft(H_angular, axis=-1)

    # Truncate to first nc_prime delay bins
    H_truncated = H_delay[..., :nc_prime]

    return H_truncated


def normalize_and_split(H_ad):
    """
    H_ad: (..., Nt, Nc') complex -> X: (..., 2, Nt, Nc') float32
    Normalize per sample and split real/imag
    """
    # Per-sample power normalization
    power = np.mean(np.abs(H_ad) ** 2, axis=(-2, -1), keepdims=True)
    power = np.maximum(power, 1e-10)
    H_norm = H_ad / np.sqrt(power)

    # Split real/imag -> (N, 2, Nt, Nc')
    X = np.stack([H_norm.real, H_norm.imag], axis=-3).astype(np.float32)
    return X, power.squeeze()


def preprocess_scenario(scenario_name):
    raw_path = os.path.join(RAW_DIR, f"dataset_{scenario_name}.h5")
    if not os.path.exists(raw_path):
        print(f"  SKIP: {raw_path} not found")
        return

    print(f"\nPreprocessing: {scenario_name}")
    with h5py.File(raw_path, "r") as f:
        H_train = f["H_train"][:]
        H_val = f["H_val"][:]
        H_test = f["H_test"][:]
        R_H = f["R_H"][:]
        PDP = f["PDP"][:]
        loc_train = f["loc_idx_train"][:]
        loc_val = f["loc_idx_val"][:]
        loc_test = f["loc_idx_test"][:]
        Nt = int(f.attrs["Nt"])
        Nr = int(f.attrs["Nr"])

    print(f"  Raw H_train: {H_train.shape}")

    # Process each Rx antenna separately, then pick the "best" or average
    # For CsiNet: typically use first Rx antenna or average across Rx
    # Here: average across Rx for robustness
    # H: (N, Nr, Nt, Nsc) -> average Rx -> (N, Nt, Nsc)
    H_train_avg = np.mean(H_train, axis=1)  # (N, Nt, Nsc)
    H_val_avg = np.mean(H_val, axis=1)
    H_test_avg = np.mean(H_test, axis=1)

    # Angular-delay transform
    H_train_ad = angular_delay_transform(H_train_avg)
    H_val_ad = angular_delay_transform(H_val_avg)
    H_test_ad = angular_delay_transform(H_test_avg)
    print(f"  After angular-delay: {H_train_ad.shape}")

    # Normalize and split
    X_train, _ = normalize_and_split(H_train_ad)
    X_val, _ = normalize_and_split(H_val_ad)
    X_test, _ = normalize_and_split(H_test_ad)
    print(f"  Final X_train: {X_train.shape}")

    out_path = os.path.join(OUT_DIR, f"preprocessed_{scenario_name}.h5")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("X_train", data=X_train, compression="gzip")
        f.create_dataset("X_val", data=X_val, compression="gzip")
        f.create_dataset("X_test", data=X_test, compression="gzip")
        f.create_dataset("R_H", data=R_H)
        f.create_dataset("PDP", data=PDP)
        f.create_dataset("loc_idx_train", data=loc_train)
        f.create_dataset("loc_idx_val", data=loc_val)
        f.create_dataset("loc_idx_test", data=loc_test)
        f.attrs["Nt"] = Nt
        f.attrs["Nr"] = Nr
        f.attrs["Nc_prime"] = NC_PRIME
        f.attrs["scenario"] = scenario_name

    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    scenarios = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
    for sc in scenarios:
        preprocess_scenario(sc)
    print("\nPreprocessing done.")
