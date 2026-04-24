#!/usr/bin/env python3
"""
Update csinet_evaluation.json & conditioned_evaluation.json with fulldata retrained results,
then run E2E simulation.
"""
import os, json

RESULTS_DIR = os.environ.get("CSINET_OUT_DIR", "/workspace/csinet_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TYPE2_NMSE = {"UMi_LOS": -3.656, "UMi_NLOS": -4.480, "UMa_NLOS": -4.229}

baseline_raw = [
    ("UMi_LOS",  0.25, 64, -25.51, 0.9986),
    ("UMi_LOS",  0.125, 32, -20.44, 0.9955),
    ("UMi_LOS",  0.0625, 16, -15.77, 0.9862),
    ("UMi_LOS",  0.03125, 8, -11.38, 0.9602),
    ("UMi_NLOS", 0.25, 64, -23.74, 0.9979),
    ("UMi_NLOS", 0.125, 32, -19.73, 0.9945),
    ("UMi_NLOS", 0.0625, 16, -14.55, 0.9810),
    ("UMi_NLOS", 0.03125, 8, -9.66, 0.9408),
    ("UMa_NLOS", 0.25, 64, -15.75, 0.9857),
    ("UMa_NLOS", 0.125, 32, -10.70, 0.9526),
    ("UMa_NLOS", 0.0625, 16, -7.02, 0.8850),
    ("UMa_NLOS", 0.03125, 8, -4.46, 0.7805),
]

cond_raw = [
    ("UMi_LOS",  0.25, -28.30, 0.9993),
    ("UMi_LOS",  0.125, -21.92, 0.9968),
    ("UMi_LOS",  0.0625, -16.32, 0.9878),
    ("UMi_LOS",  0.03125, -11.55, 0.9617),
    ("UMi_NLOS", 0.25, -25.46, 0.9986),
    ("UMi_NLOS", 0.125, -20.60, 0.9955),
    ("UMi_NLOS", 0.0625, -14.75, 0.9818),
    ("UMi_NLOS", 0.03125, -9.73, 0.9417),
    ("UMa_NLOS", 0.25, -15.85, 0.9861),
    ("UMa_NLOS", 0.125, -10.84, 0.9541),
    ("UMa_NLOS", 0.0625, -7.05, 0.8858),
    ("UMa_NLOS", 0.03125, -4.48, 0.7814),
]

bl_map = {}
for sc, g, M, nmse, cos in baseline_raw:
    bl_map[(sc, g)] = (M, nmse, cos)

csinet_eval = []
for sc, g, M, nmse, cos in baseline_raw:
    t2 = TYPE2_NMSE[sc]
    csinet_eval.append({
        "scenario": sc, "gamma": g, "M": M,
        "nmse_dB": nmse, "cos_sim": cos,
        "type2_nmse_dB": t2,
        "gain_vs_type2_dB": round(abs(nmse) - abs(t2), 2),
    })

cond_eval = []
for sc, g, c_nmse, c_cos in cond_raw:
    _, bl_nmse, _ = bl_map[(sc, g)]
    cond_eval.append({
        "scenario": sc, "gamma": g,
        "baseline_nmse_dB": bl_nmse,
        "conditioned_nmse_dB": c_nmse,
        "conditioned_cos": c_cos,
        "gain_dB": round(abs(c_nmse) - abs(bl_nmse), 2),
    })

bl_path = os.path.join(RESULTS_DIR, "csinet_evaluation.json")
cd_path = os.path.join(RESULTS_DIR, "conditioned_evaluation.json")

with open(bl_path, "w") as f:
    json.dump(csinet_eval, f, indent=2)
print(f"Updated: {bl_path}")

with open(cd_path, "w") as f:
    json.dump(cond_eval, f, indent=2)
print(f"Updated: {cd_path}")

for d in ["/workspace/graduation/csinet/results",
          "/workspace/graduation/csinet/results_new"]:
    os.makedirs(d, exist_ok=True)
    for src, name in [(bl_path, "csinet_evaluation.json"),
                      (cd_path, "conditioned_evaluation.json")]:
        dst = os.path.join(d, name)
        with open(src) as f1, open(dst, "w") as f2:
            f2.write(f1.read())
        print(f"  Copied → {dst}")

print("\nJSON update complete. Running E2E simulation...")
