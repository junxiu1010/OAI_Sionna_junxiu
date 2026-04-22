#!/usr/bin/env python3
"""
Phase 4: Integration 통합 테스트
=================================
channel_hook -> csinet_engine -> csi_injection 전체 파이프라인 테스트.

docker exec -e CUDA_VISIBLE_DEVICES=0 -e TF_CPP_MIN_LOG_LEVEL=3 sionna-proxy \
    python3 /workspace/test_integration.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from integration.channel_hook import ChannelHook
from integration.csinet_engine import CsiNetInferenceEngine
from integration.csi_injection import CSIInjector, compute_ri_from_H, compute_pmi_from_H, compute_cqi_from_H


def test_channel_hook():
    print("=== Test ChannelHook ===")
    hook = ChannelHook(enabled=True, csi_rs_period=1)

    captured = []
    hook.register_callback(lambda c, u, H: captured.append((c, u, H.shape)))

    # Simulate channel capture (N_SYM=14, Nr=2, Nt=4, FFT=128)
    for i in range(10):
        fake_H = np.random.randn(14, 2, 4, 128) + 1j * np.random.randn(14, 2, 4, 128)
        hook.capture(0, 0, fake_H)

    assert len(captured) == 10, f"Expected 10 captures, got {len(captured)}"
    print(f"  Captured {len(captured)} channels, last shape: {captured[-1][2]}")

    H_latest = hook.get_latest(0, 0)
    assert H_latest is not None
    print(f"  Latest H shape: {H_latest.shape}")

    R_H, pdp = hook.get_statistics(0, 0, n_samples=5)
    assert R_H is not None
    print(f"  Covariance shape: {R_H.shape}, PDP shape: {pdp.shape}")
    print("  PASS\n")


def test_csi_injection():
    print("=== Test CSI Injection ===")
    H_hat = np.random.randn(4, 72) + 1j * np.random.randn(4, 72)
    H_hat = H_hat.astype(np.complex64)

    ri = compute_ri_from_H(H_hat)
    print(f"  RI: {ri}")
    assert ri in [1, 2]

    pmi = compute_pmi_from_H(H_hat)
    print(f"  PMI beams: {pmi['i1_beam_indices']}, phase shape: {np.array(pmi['i2_phase']).shape}")

    cqi = compute_cqi_from_H(H_hat)
    print(f"  CQI: {cqi}")
    assert 0 <= cqi <= 15

    injector = CSIInjector()
    report = injector.process_channel(0, 0, H_hat)
    print(f"  Report keys: {list(report.keys())}")
    print(f"  Precoding weights shape: {report['precoding_weights'].shape}")
    print("  PASS\n")


def test_csinet_engine():
    print("=== Test CsiNet Engine ===")
    ckpt_dir = "/workspace/csinet_checkpoints"
    if not os.path.exists(os.path.join(ckpt_dir, "csinet_UMi_LOS_gamma0.2500_best.weights.h5")):
        print("  SKIP: No checkpoint found")
        return

    engine = CsiNetInferenceEngine(
        mode="baseline",
        compression_ratio=1/4,
        checkpoint_dir=ckpt_dir,
        scenario="UMi_LOS"
    )

    H_freq = np.random.randn(2, 4, 72) + 1j * np.random.randn(2, 4, 72)
    H_freq = H_freq.astype(np.complex64)
    H_hat, codeword = engine.encode_decode(H_freq)
    print(f"  H_hat shape: {H_hat.shape}, codeword dim: {codeword.shape}")
    print(f"  H_hat dtype: {H_hat.dtype}")
    print("  PASS\n")


def test_full_pipeline():
    print("=== Test Full Pipeline ===")
    ckpt_dir = "/workspace/csinet_checkpoints"
    if not os.path.exists(os.path.join(ckpt_dir, "csinet_UMi_LOS_gamma0.2500_best.weights.h5")):
        print("  SKIP: No checkpoint found")
        return

    hook = ChannelHook(enabled=True, csi_rs_period=1)
    engine = CsiNetInferenceEngine(mode="baseline", compression_ratio=1/4,
                                   checkpoint_dir=ckpt_dir, scenario="UMi_LOS")
    injector = CSIInjector()

    results = []
    def on_capture(cell_idx, ue_idx, H):
        H_hat, cw = engine.encode_decode(H)
        report = injector.process_channel(cell_idx, ue_idx, H_hat)
        results.append(report)

    hook.register_callback(on_capture)

    for i in range(5):
        fake_H = np.random.randn(14, 2, 4, 128) + 1j * np.random.randn(14, 2, 4, 128)
        hook.capture(0, 0, fake_H.astype(np.complex64))

    assert len(results) == 5
    print(f"  Processed {len(results)} slots through full pipeline")
    print(f"  Last report: RI={results[-1]['ri']}, CQI={results[-1]['cqi']}")
    print(f"  Precoding weights shape: {results[-1]['precoding_weights'].shape}")
    print("  PASS\n")


if __name__ == "__main__":
    test_channel_hook()
    test_csi_injection()
    test_csinet_engine()
    test_full_pipeline()
    print("All integration tests PASSED!")
