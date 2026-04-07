"""
pitnn_plecs_cosim.py
=====================================================================
PITNN DAB Converter — PLECS Python Co-Simulation Script
Method 2: Python Simulation Script running inside PLECS.

HOW TO USE IN PLECS
────────────────────
1. In PLECS, go to:
     Simulation → Simulation Scripts → Add Script

2. Set the script language to Python and point it to this file.

3. In your PLECS schematic:
     - Add a "Simulation Script" block
     - Set Input signals  (4): V1_meas, V2_meas, iL_meas, Pref
     - Set Output signals (5): phi1, phi2, phi3, delay_us, duty_pct
     - Connect phi3 output to your Phase Shift Modulator block

4. Set PITNN_DIR below to the folder containing:
     pitnn_inference.py
     pitnn_scripted.pt
     pitnn_mu.npy
     pitnn_sigma.npy

5. Run the simulation. PLECS will call pitnn_step() at each
   sample time of the Simulation Script block.

PLECS Simulation Script API
─────────────────────────────
   plecs.mdlInitializeSizes()    → called once at startup
   plecs.mdlStart()              → called when sim starts
   plecs.mdlOutputs(t, *u)      → called every sample period
   plecs.mdlTerminate()          → called when sim stops

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University — All Rights Reserved
=====================================================================
"""

import sys
import os

# ── SET THIS PATH to your project folder ─────────────────────────
PITNN_DIR = r"C:\Users\Nzead\Documents\Research\Power electronics control system\Simulation\pitnn_dab"
# ─────────────────────────────────────────────────────────────────

sys.path.insert(0, PITNN_DIR)

# These imports happen once when PLECS loads the script
from pitnn_inference import PITNNInference
import numpy as np

# Global controller — persists for the entire simulation run
_ctrl = None


def plecs_setup():
    """
    Called by PLECS before the simulation starts.
    Initialises the PITNN controller and loads the model.
    """
    global _ctrl

    model_path = os.path.join(PITNN_DIR, "pitnn_scripted.pt")
    mu_path    = os.path.join(PITNN_DIR, "pitnn_mu.npy")
    sigma_path = os.path.join(PITNN_DIR, "pitnn_sigma.npy")

    for path, label in [(model_path, "pitnn_scripted.pt"),
                        (mu_path,    "pitnn_mu.npy"),
                        (sigma_path, "pitnn_sigma.npy")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[PITNN] Required file not found: {path}\n"
                f"Run 'python pitnn_deploy.py --mode export' first."
            )

    _ctrl = PITNNInference(model_path, mu_path, sigma_path)
    print(f"[PITNN Co-Sim] Controller ready  |  PLECS integration active")
    print(f"[PITNN Co-Sim] Model: {model_path}")


def plecs_step(V1, V2, iL, Pref):
    """
    Main inference function — called by the PLECS Simulation Script
    block at every sample time.

    Parameters (from PLECS input signals)
    ──────────────────────────────────────
    V1   : primary DC bus voltage   (V)
    V2   : secondary DC bus voltage (V)
    iL   : inductor current         (A)
    Pref : power reference          (W)  from outer PI control loop

    Returns (to PLECS output signals)
    ──────────────────────────────────
    phi1      : inner duty phi1 = 2.9845 rad (fixed)
    phi2      : inner duty phi2 = 2.9845 rad (fixed)
    phi3      : external phase shift (rad) — connect to Phase Shift Modulator
    delay_us  : phi3 converted to gate drive time delay (µs)
    duty_pct  : phi1 converted to inner duty cycle (%)
    """
    global _ctrl

    # Lazy initialisation if setup was not called
    if _ctrl is None:
        plecs_setup()

    phi1, phi2, phi3 = _ctrl.step(
        float(V1), float(V2), float(iL), float(Pref)
    )

    delay_us = _ctrl.phi3_to_delay_us(phi3)
    duty_pct = _ctrl.phi1_to_duty_pct(phi1)

    return phi1, phi2, phi3, delay_us, duty_pct


def plecs_reset():
    """Called when the PLECS simulation is reset."""
    global _ctrl
    if _ctrl is not None:
        _ctrl.reset()
        print("[PITNN Co-Sim] Controller buffer reset")


# ── PLECS Simulation Script callbacks ─────────────────────────────
# PLECS calls these automatically when the script is active.

def mdlInitializeSizes():
    """Tell PLECS how many inputs and outputs this script has."""
    plecs.numContStates   = 0
    plecs.numDiscStates   = 0
    plecs.numInputs       = 4    # V1, V2, iL, Pref
    plecs.numOutputs      = 5    # phi1, phi2, phi3, delay_us, duty_pct
    plecs.directFeedthrough = True
    plecs.sampleTime      = [1.0 / 100e3, 0]  # 100kHz sample rate


def mdlStart():
    plecs_setup()


def mdlOutputs(t, *u):
    """
    Called at every sample time step.
    u[0]=V1, u[1]=V2, u[2]=iL, u[3]=Pref
    Returns list of 5 output values.
    """
    if len(u) < 4:
        return [2.9845, 2.9845, 0.22, 0.35, 95.0]

    V1, V2, iL, Pref = u[0], u[1], u[2], u[3]
    phi1, phi2, phi3, delay_us, duty_pct = plecs_step(V1, V2, iL, Pref)
    return [phi1, phi2, phi3, delay_us, duty_pct]


def mdlTerminate():
    plecs_reset()
    print("[PITNN Co-Sim] Simulation ended")
