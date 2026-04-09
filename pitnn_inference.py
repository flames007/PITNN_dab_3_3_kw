"""
PITNN DAB Converter — Python Inference (Option 1)
==================================================
Runs the trained PITNN using only the three exported files.
No dependency on pitnn_dab.py or any training code.

Required files (all in the same folder as this script):
    pitnn_scripted.pt   — TorchScript model
    pitnn_mu.npy        — normalisation means
    pitnn_sigma.npy     — normalisation standard deviations

Install:
    pip install torch numpy

Run:
    python pitnn_inference.py

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University — All Rights Reserved
"""

import time
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match values used during training)
# ─────────────────────────────────────────────────────────────────────────────
V1_NOM    = 800.0    # V  — nominal primary bus voltage
V2_NOM    = 800.0    # V  — nominal secondary bus voltage
FSW       = 100e3    # Hz — switching frequency
PI        = 3.141592653589793
PHI12_MIN = PI * 0.65   # 2.0420 rad — lower bound for phi1/phi2
PHI12_MAX = PI * 0.99   # 3.1102 rad — upper bound for phi1/phi2
PHI_MIN   = 0.02         # rad — lower bound for phi3
PHI3_MAX  = 1.50         # rad — upper bound for phi3
SEQ_LEN   = 20           # number of past switching cycles the model sees
N_FEAT    = 8            # [V1, V2, iL, phi1, phi2, phi3, Pref, V1V2/Vnom2]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL AND NORMALISATION STATS
# ─────────────────────────────────────────────────────────────────────────────

def load_pitnn(model_path="pitnn_scripted.pt",
               mu_path="pitnn_mu.npy",
               sigma_path="pitnn_sigma.npy",
               device=None):
    """
    Load the TorchScript model and normalisation constants.
    Returns (model, mu, sigma, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[PITNN] Loading model from: {model_path}")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    mu    = np.load(mu_path).astype(np.float32)
    sigma = np.load(sigma_path).astype(np.float32)

    print(f"[PITNN] Ready on {device}")
    print(f"[PITNN] Input features: [V1, V2, iL, φ1, φ2, φ3, Pref, V1·V2/V²nom]")
    print(f"[PITNN] φ1,φ2 ∈ [{PHI12_MIN:.4f}, {PHI12_MAX:.4f}] rad  (predicted)")
    print(f"[PITNN] φ3    ∈ [{PHI_MIN:.4f},   {PHI3_MAX:.4f}]  rad  (predicted)")
    return model, mu, sigma, device


# ─────────────────────────────────────────────────────────────────────────────
# REAL-TIME CONTROLLER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PITNNInference:
    """
    Stateful real-time PITNN controller.
    Maintains a rolling 20-step history buffer.
    Call step() once per switching cycle.

    Usage:
        ctrl = PITNNInference()
        phi1, phi2, phi3 = ctrl.step(V1=800, V2=798, iL=27.5, Pref=20000)
    """

    def __init__(self, model_path="pitnn_scripted.pt",
                 mu_path="pitnn_mu.npy",
                 sigma_path="pitnn_sigma.npy",
                 device=None):
        self.model, self.mu, self.sigma, self.device = load_pitnn(
            model_path, mu_path, sigma_path, device
        )
        # Rolling buffer: last SEQ_LEN normalised state vectors
        self._buffer    = np.zeros((SEQ_LEN, N_FEAT), dtype=np.float32)
        self._phi1_prev = PI * 0.95   # initial phi1 estimate (nominal)
        self._phi2_prev = PI * 0.95   # initial phi2 estimate (nominal)
        self._phi3_prev = 0.22        # initial phi3 estimate

    def reset(self):
        """Clear history buffer. Call when starting a new scenario."""
        self._buffer    = np.zeros((SEQ_LEN, N_FEAT), dtype=np.float32)
        self._phi1_prev = PI * 0.95
        self._phi2_prev = PI * 0.95
        self._phi3_prev = 0.22

    def _normalise(self, feat: np.ndarray) -> np.ndarray:
        """Apply training normalisation: (x - mu) / sigma."""
        return ((feat - self.mu) / self.sigma).astype(np.float32)

    def step(self, V1: float, V2: float, iL: float, Pref: float,
             phi_prev: tuple = None) -> tuple:
        """
        Run one PITNN inference step.

        Parameters
        ----------
        V1        : float — primary DC bus voltage (V)
        V2        : float — secondary DC bus voltage (V)
        iL        : float — inductor current (A)
        Pref      : float — power reference from outer control loop (W)
        phi_prev  : (phi1, phi2, phi3) previous outputs (rad); uses internal
                    state if None

        Returns
        -------
        (phi1, phi2, phi3) : floats in radians — all three independently predicted
            phi1 ∈ [PHI12_MIN, PHI12_MAX]  — primary bridge inner duty
            phi2 ∈ [PHI12_MIN, PHI12_MAX]  — secondary bridge inner duty
            phi3 ∈ [PHI_MIN,   PHI3_MAX]   — external phase shift → gate drive
        """
        if phi_prev is not None:
            phi1_p, phi2_p, phi3_p = phi_prev
        else:
            phi1_p = self._phi1_prev
            phi2_p = self._phi2_prev
            phi3_p = self._phi3_prev

        # Build 8-feature state vector using previous predicted angles
        v_ratio = float(V1 * V2) / (V1_NOM * V2_NOM)
        feat = np.array([
            V1, V2, iL,
            phi1_p, phi2_p, phi3_p,
            Pref, v_ratio
        ], dtype=np.float32)

        # Shift buffer left, append new normalised state at the end
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = self._normalise(feat)

        # Single PITNN forward pass
        x = torch.from_numpy(self._buffer).unsqueeze(0).to(self.device)
        with torch.no_grad():
            phi_out = self.model(x).cpu().numpy().squeeze()

        phi1 = float(phi_out[0])
        phi2 = float(phi_out[1])
        phi3 = float(phi_out[2])
        self._phi1_prev = phi1
        self._phi2_prev = phi2
        self._phi3_prev = phi3
        return phi1, phi2, phi3

    def phi3_to_delay_us(self, phi3: float) -> float:
        """Convert φ3 (rad) to gate drive phase delay in microseconds."""
        return phi3 / (2.0 * PI * FSW) * 1e6

    def phi1_to_duty_pct(self, phi1: float) -> float:
        """Convert φ1 (rad) to inner duty cycle percentage."""
        return (phi1 / PI) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: PRINT NORMALISATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

def print_normalisation_constants(mu_path="pitnn_mu.npy",
                                  sigma_path="pitnn_sigma.npy"):
    """
    Print exact mu and sigma values for use in C++, MATLAB, or embedded code.
    Copy these numbers directly into your target-language implementation.
    """
    mu    = np.load(mu_path).astype(np.float32)
    sigma = np.load(sigma_path).astype(np.float32)

    labels = ["V1 (V)", "V2 (V)", "iL (A)", "φ1 (rad)",
              "φ2 (rad)", "φ3 (rad)", "Pref (W)", "V1V2/Vnom²"]

    print("\nNormalisation Constants (copy into C++/MATLAB/embedded code)")
    print("=" * 60)
    print(f"{'Feature':<16} {'mu':>20} {'sigma':>20}")
    print("-" * 60)
    for i, label in enumerate(labels):
        print(f"{label:<16} {mu[i]:>20.8f} {sigma[i]:>20.8f}")
    print()
    print("C++ arrays:")
    mu_str    = ", ".join(f"{v:.8f}f" for v in mu)
    sigma_str = ", ".join(f"{v:.8f}f" for v in sigma)
    print(f"float mu[8]    = {{{mu_str}}};")
    print(f"float sigma[8] = {{{sigma_str}}};")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — runs without any hardware
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 66)
    print("  PITNN Inference — Python Demo (pitnn_inference.py)")
    print("=" * 66)

    # Print the normalisation constants first
    print_normalisation_constants()

    # Create controller
    ctrl = PITNNInference()

    # Test operating conditions matching the paper's inference table
    scenarios = [
        (800, 800, 10000, "10kW nominal"),
        (760, 840,  8000, "8kW voltage variation"),
        (800, 800, 20000, "20kW"),
        (800, 800,  5000, "5kW light load"),
        (880, 720, 50000, "50kW high voltage"),
        (800, 800, 30000, "30kW mid load"),
        (840, 760, 40000, "40kW asymmetric V"),
        (720, 880, 15000, "15kW off-voltage"),
    ]

    print(f"\n{'Condition':<22} {'V1':>5} {'V2':>5} {'Pref':>7}  "
          f"{'φ1 (rad)':>9}  {'φ3 (rad)':>9}  {'delay (µs)':>10}  {'duty%':>6}  {'t (µs)':>8}")
    print("-" * 92)

    for V1, V2, Pref, label in scenarios:
        ctrl.reset()
        # Warm up the buffer with a plausible iL estimate
        iL_est = float(V1 * V2 / (V1_NOM * V2_NOM) * 10.7 * 0.3)

        # Run 5 cycles to fill the buffer before measuring
        for _ in range(5):
            ctrl.step(float(V1), float(V2), iL_est, float(Pref))

        # Timed inference
        t0 = time.perf_counter()
        phi1, phi2, phi3 = ctrl.step(float(V1), float(V2), iL_est, float(Pref))
        t_us = (time.perf_counter() - t0) * 1e6

        delay_us = ctrl.phi3_to_delay_us(phi3)
        duty_pct = ctrl.phi1_to_duty_pct(phi1)

        print(f"{label:<22} {V1:>5.0f} {V2:>5.0f} {Pref:>7.0f}  "
              f"{phi1:>9.4f}  {phi3:>9.4f}  {delay_us:>10.3f}  {duty_pct:>5.1f}%  {t_us:>8.1f}")

    print(f"\nAll three angles φ1, φ2, φ3 independently predicted each cycle.")
    print(f"Gate drive: φ3 → phase delay  |  φ1 → primary duty  |  φ2 → secondary duty")


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION EXAMPLE — drop this into your own control loop
# ─────────────────────────────────────────────────────────────────────────────

def integration_example():
    """
    Minimal integration example.
    Replace the simulated sensor values with your real ADC readings.
    """
    ctrl = PITNNInference()

    # Your real-time loop — call at fsw = 100kHz
    Pref = 20000.0   # W — from your outer PI voltage controller

    for cycle in range(20):
        # ── READ SENSORS (replace with real ADC calls) ────────────────────
        V1_meas = 800.0 + (cycle % 3) * 1.5     # simulated noise
        V2_meas = 798.0 + (cycle % 2) * 1.0
        iL_meas = 27.5  + (cycle % 4) * 0.5

        # ── PITNN INFERENCE ───────────────────────────────────────────────
        phi1, phi2, phi3 = ctrl.step(V1_meas, V2_meas, iL_meas, Pref)

        # ── CONVERT TO GATE DRIVE TIMING ─────────────────────────────────
        phase_delay_s = phi3 / (2.0 * PI * FSW)      # seconds
        inner_duty    = phi1 / PI                     # fraction (0 to 1)

        # ── APPLY TO PWM HARDWARE (replace with your hardware API) ────────
        # pwm.set_primary_duty(inner_duty)
        # pwm.set_phase_delay(phase_delay_s)
        # pwm.commit()

        if cycle % 5 == 0:
            print(f"Cycle {cycle:3d}: φ3={phi3:.4f} rad  "
                  f"delay={phase_delay_s*1e6:.3f}µs  duty={inner_duty*100:.1f}%")


if __name__ == "__main__":
    run_demo()
    print("\n" + "=" * 66)
    print("  Integration Example (replace sensor stubs with real ADC reads)")
    print("=" * 66)
    integration_example()
