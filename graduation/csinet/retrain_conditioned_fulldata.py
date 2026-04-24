#!/usr/bin/env python3
"""
Conditioned CsiNet 재학습 — subprocess 방식, fulldata
"""
import os, sys, json, subprocess

SCENARIOS = ["UMi_LOS", "UMi_NLOS", "UMa_NLOS"]
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32]
CKPT_DIR = os.environ.get("CSINET_CKPT_DIR", "/workspace/csinet_checkpoints")
WORKER = os.path.join(os.path.dirname(__file__), "_train_conditioned_worker.py")


def main():
    all_results = []

    for sc in SCENARIOS:
        for gamma in COMPRESSION_RATIOS:
            tag = f"{sc}_gamma{gamma:.4f}"
            result_file = os.path.join(CKPT_DIR, f"conditioned_fulldata_{tag}.json")

            if os.path.exists(result_file):
                with open(result_file) as f:
                    r = json.load(f)
                all_results.append(r)
                print(f"\n>>> SKIP: {sc} gamma={gamma:.4f} "
                      f"NMSE={r['test_nmse_dB']:.2f} dB", flush=True)
                continue

            print(f"\n>>> Launching: {sc} gamma={gamma:.4f}", flush=True)
            sys.stdout.flush()
            result = subprocess.run(
                [sys.executable, WORKER, sc, str(gamma)])
            if result.returncode != 0:
                print(f"  [ERROR] exit={result.returncode}", flush=True)

            if os.path.exists(result_file):
                with open(result_file) as f:
                    r = json.load(f)
                all_results.append(r)
                print(f"  => NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}", flush=True)
            else:
                print(f"  => FAILED", flush=True)

    combined = os.path.join(CKPT_DIR, "retrain_conditioned_fulldata_results.json")
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*60}")
    print("  CONDITIONED FULLDATA RETRAINING SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r['scenario']:10s} gamma={r['gamma']:.4f} M={r['M']:3d} | "
              f"NMSE={r['test_nmse_dB']:.2f} dB, cos={r['test_cos_sim']:.4f}")


if __name__ == "__main__":
    main()
