"""
PITNN DAB Converter — Export File Inspector
===========================================
Inspects the three exported files and prints all values needed
to hard-code normalisation constants into any target language.

Required files (same folder):
    pitnn_scripted.pt
    pitnn_mu.npy
    pitnn_sigma.npy

Run:
    python pitnn_inspect_exports.py

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University — All Rights Reserved
"""

import numpy as np
import os, sys

FEATURE_LABELS = [
    "V1       (V)         ",
    "V2       (V)         ",
    "iL       (A)         ",
    "phi1     (rad)       ",
    "phi2     (rad)       ",
    "phi3     (rad)       ",
    "Pref     (W)         ",
    "V1*V2/Vnom^2 (ratio) ",
]


def inspect_npy(path, label):
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return None
    arr = np.load(path).astype(np.float32)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {label}: shape={arr.shape}  dtype=float32  size={size_kb:.1f} KB")
    return arr


def print_table(mu, sigma):
    print()
    print("Normalisation Constants — Feature Table")
    print("=" * 68)
    print(f"  {'Feature':<26} {'mu':>18} {'sigma':>18}")
    print("-" * 68)
    for i, label in enumerate(FEATURE_LABELS):
        print(f"  {label:<26} {mu[i]:>18.8f} {sigma[i]:>18.8f}")
    print()


def print_cpp_arrays(mu, sigma):
    print("C++ / C Arrays (copy into pitnn_inference.cpp or embedded code)")
    print("=" * 68)
    mu_str    = ",\n    ".join(f"{v:.8f}f" for v in mu)
    sigma_str = ",\n    ".join(f"{v:.8f}f" for v in sigma)
    print(f"static const float MU[8] = {{\n    {mu_str}\n}};")
    print()
    print(f"static const float SIGMA[8] = {{\n    {sigma_str}\n}};")
    print()


def print_python_arrays(mu, sigma):
    print("Python / NumPy Arrays")
    print("=" * 68)
    print(f"mu    = np.array({list(mu.round(8))}, dtype=np.float32)")
    print(f"sigma = np.array({list(sigma.round(8))}, dtype=np.float32)")
    print()


def print_matlab_arrays(mu, sigma):
    print("MATLAB Arrays")
    print("=" * 68)
    mu_str    = "; ".join(f"{v:.8f}" for v in mu)
    sigma_str = "; ".join(f"{v:.8f}" for v in sigma)
    print(f"mu    = [{mu_str}];")
    print(f"sigma = [{sigma_str}];")
    print()
    print("% Normalise input x (1x8 vector):")
    print("x_norm = (x - mu) ./ sigma;")
    print()


def print_json(mu, sigma):
    import json
    d = {"mu": list(float(v) for v in mu),
         "sigma": list(float(v) for v in sigma),
         "features": [l.strip() for l in FEATURE_LABELS]}
    print("JSON (for web / REST API integration)")
    print("=" * 68)
    print(json.dumps(d, indent=2))
    print()


def inspect_torchscript(path):
    print()
    print("TorchScript Model")
    print("=" * 68)
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

    try:
        import torch
        model = torch.jit.load(path, map_location="cpu")
        model.eval()

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Test inference
        dummy = torch.zeros(1, 20, 8)
        with torch.no_grad():
            out = model(dummy)
        print(f"  Input shape : {list(dummy.shape)}")
        print(f"  Output shape: {list(out.shape)}")
        print(f"  Test output : {out.numpy().squeeze().round(4)}")
        print(f"  Status      : OK — model loads and runs correctly")
    except ImportError:
        print("  (torch not installed — cannot verify model, but file exists)")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    print("=" * 68)
    print("  PITNN Export File Inspector")
    print("=" * 68)

    # ── Check files exist ──────────────────────────────────────────────────
    print("\nFile Status:")
    print("-" * 68)
    mu    = inspect_npy("pitnn_mu.npy",    "pitnn_mu.npy   ")
    sigma = inspect_npy("pitnn_sigma.npy", "pitnn_sigma.npy")

    onnx_exists = os.path.exists("pitnn_model.onnx")
    pt_exists   = os.path.exists("pitnn_scripted.pt")
    print(f"  pitnn_scripted.pt : {'FOUND' if pt_exists else 'NOT FOUND'}")
    print(f"  pitnn_model.onnx  : {'FOUND' if onnx_exists else 'NOT FOUND — run: python pitnn_deploy.py --mode export'}")

    if mu is None or sigma is None:
        print("\nERROR: mu.npy and sigma.npy are required. "
              "Run: python pitnn_deploy.py --mode export")
        sys.exit(1)

    # ── Feature table ──────────────────────────────────────────────────────
    print_table(mu, sigma)

    # ── Code snippets ──────────────────────────────────────────────────────
    print_cpp_arrays(mu, sigma)
    print_python_arrays(mu, sigma)
    print_matlab_arrays(mu, sigma)
    print_json(mu, sigma)

    # ── TorchScript model check ────────────────────────────────────────────
    if pt_exists:
        inspect_torchscript("pitnn_scripted.pt")

    # ── Quick-reference usage summary ─────────────────────────────────────
    print()
    print("Quick-Reference Usage")
    print("=" * 68)
    print("""
  Option 1 — Python (TorchScript):
      python pitnn_inference.py

  Option 2 — C++ (LibTorch):
      mkdir build && cd build
      cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
      make && ./pitnn_inference

  Option 3 — ONNX Runtime (any platform):
      pip install onnxruntime
      python pitnn_onnx_inference.py

  All three options produce identical phi_TPS outputs.
  Input  : (1, 20, 8) float32 — last 20 switching-cycle states
  Output : (1, 3)     float32 — [phi1, phi2, phi3] in radians
""")


if __name__ == "__main__":
    main()
