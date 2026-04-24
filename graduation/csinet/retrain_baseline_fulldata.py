#!/usr/bin/env python3
"""
Baseline CsiNet 재학습 — floor=1e-30, 필터링 없음, 데이터 100% 사용
=================================================================
subprocess로 각 모델을 개별 GPU 세션에서 학습.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys, json, subprocess

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32]
CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")

WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_train_baseline_worker.py")


def main():
    all_results = []

    for sc in SCENARIOS:
        for gamma in COMPRESSION_RATIOS:
            result_file = os.path.join(CKPT_DIR, f"baseline_fulldata_{sc}_gamma{gamma:.4f}.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    r = json.load(f)
                all_results.append(r)
                print(f"\n>>> SKIP (already done): {sc} gamma={gamma:.4f} "
                      f"NMSE={r['test_nmse_dB']:.2f} dB", flush=True)
                continue

            print(f"\n>>> Launching: {sc} gamma={gamma:.4f}", flush=True)
            cmd = [sys.executable, WORKER_SCRIPT, sc, str(gamma)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout, end="", flush=True)
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if "TEST:" in line or "Training:" in line or "Scenario:" in line:
                        print(f"  [stderr] {line}")

            result_file = os.path.join(CKPT_DIR, f"baseline_fulldata_{sc}_gamma{gamma:.4f}.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    r = json.load(f)
                all_results.append(r)
                print(f"  => NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}", flush=True)
            else:
                print(f"  => FAILED (no result file)", flush=True)

    combined = os.path.join(CKPT_DIR, "retrain_baseline_fulldata_results.json")
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*60}")
    print("  BASELINE FULLDATA RETRAINING SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['scenario']:10s} gamma={r['gamma']:.4f} M={r['M']:3d} | "
              f"NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}")


if __name__ == "__main__":
    main()
