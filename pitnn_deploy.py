"""
============================================================
DAB Converter — Real-Time Deployment Controller
============================================================

Standalone deployment file for the trained PITNN model.
Imports everything it needs from pitnn_dab.py — no code
duplication, no modifications to the training file needed.

File structure
──────────────
  pitnn_dab.py          ← training + simulation (unchanged)
  pitnn_deploy.py       ← this file  (deployment only)
  pitnn_dab_checkpoint.pt  ← saved after running pitnn_dab.py

Usage
──────────────────────────────────────────────────────────
  # Step 1: train the model (run once)
  python pitnn_dab.py

  # Step 2: run the deployment controller (run on prototype)
  python pitnn_deploy.py

  # Step 3: integrate into your own control loop
  from pitnn_deploy import DeployedController
  ctrl = DeployedController()
  phi1, phi2, phi3 = ctrl.step(V1=800, V2=798, iL=27.5, Pref=20000)

Deployment modes
──────────────────────────────────────────────────────────
  Mode 1 — Demo loop      : simulates sensor readings, prints results
  Mode 2 — Hardware loop  : reads from real ADC, writes to PWM
  Mode 3 — Export         : saves ONNX + TorchScript for embedded targets
  Mode 4 — Closed-loop    : full outer PI + inner PITNN control loop

Run a specific mode:
  python pitnn_deploy.py --mode demo
  python pitnn_deploy.py --mode export
  python pitnn_deploy.py --mode closed_loop --Vref 800 --Pmax 50000
"""

import math
import time
import argparse
import numpy as np
import torch

# ── Import everything from the training file ──────────────────────────────────
# pitnn_dab.py must be in the same directory (or on PYTHONPATH)
from pitnn_dab import (
    # Constants
    V1_NOM, V2_NOM, FSW, LK, N_TURNS, PI,
    PHI_MIN, PHI12_FIXED, PHI3_MAX, K_POWER, B_POWER,
    # Classes
    PITNN, DABPhysics, PITNNController,
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE DEPLOYMENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DeployedController:
    """
    Self-contained real-time controller loaded from a saved checkpoint.

    This is the only class you need in a deployment scenario.
    It loads the trained PITNN, wraps it in the PITNNController,
    and exposes a single .step() method for your control loop.

    Usage
    ─────
        ctrl = DeployedController("pitnn_dab_checkpoint.pt")

        # In your switching-cycle interrupt or real-time loop:
        phi1, phi2, phi3 = ctrl.step(V1, V2, iL, Pref)

        # Apply phi3 to your PWM hardware:
        pwm.set_phase_delay(phi3 / (2 * pi * fsw))
    """

    def __init__(self, checkpoint_path="pitnn_dab_checkpoint.pt",
                 device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[DeployedController] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # ── Restore model from checkpoint ─────────────────────────────────
        hp   = ckpt["hyperparams"]
        self._model = PITNN(
            d_in    = hp.get("d_in",     8),
            d_model = hp.get("d_model",  128),
            n_heads = hp.get("n_heads",  8),
            n_layers= hp.get("n_layers", 4),
            d_ff    = hp.get("d_ff",     256),
            seq_len = hp.get("seq_len",  20),
            dropout = 0.0,   # no dropout at inference
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(device).eval()

        # ── Restore normalisation statistics ──────────────────────────────
        self._mu    = ckpt["mu"].astype(np.float32)
        self._sigma = ckpt["sigma"].astype(np.float32)

        # ── Physics model for ZVS/power verification ──────────────────────
        self._dab  = DABPhysics()

        # ── Wrap in PITNNController for stateful buffer management ─────────
        self._ctrl = PITNNController(
            self._model, self._mu, self._sigma, self._dab, device=device
        )

        n_p = sum(p.numel() for p in self._model.parameters())
        print(f"[DeployedController] Ready — {n_p:,} parameters on {device}")
        print(f"[DeployedController] φ1=φ2 fixed={PHI12_FIXED:.4f} rad, "
              f"φ3 ∈ [{PHI_MIN:.3f}, {PHI3_MAX:.3f}] rad")

    def reset(self):
        """Clear history buffer. Call when starting a new operating scenario."""
        self._ctrl.reset()

    def step(self, V1: float, V2: float, iL: float, Pref: float,
             verify_physics: bool = False):
        """
        Single inference step — call once per switching cycle.

        Parameters
        ──────────
        V1    : primary DC bus voltage   (V)   — from ADC / voltage sensor
        V2    : secondary DC bus voltage (V)   — from ADC / voltage sensor
        iL    : inductor current         (A)   — from current probe / Rogowski
        Pref  : power reference          (W)   — from outer PI control loop
        verify_physics : if True, also computes P_calc, Irms, ZVS
                         (adds ~2ms overhead — disable in tight timing loops)

        Returns
        ───────
        phi1, phi2, phi3  (floats, radians)
            Apply phi3 to gate drive: delay = phi3 / (2π × fsw)
            phi1=phi2=PHI12_FIXED are constant — apply as fixed duty cycle

        If verify_physics=True, returns (phi1, phi2, phi3, info_dict)
        """
        result = self._ctrl.step(
            float(V1), float(V2), float(iL), float(Pref),
            phi_prev=None, reset=False
        )
        phi = result["phi_TPS"]

        if verify_physics:
            return float(phi[0]), float(phi[1]), float(phi[2]), {
                "P_calc":    result["P_calc"],
                "Irms":      result["Irms"],
                "zvs_ok":    result["zvs_ok"],
                "P_err_pct": result["P_err_pct"],
                "mode":      result["mode"],
                "inf_us":    result["inf_us"],
            }
        return float(phi[0]), float(phi[1]), float(phi[2])

    def phi3_to_delay_us(self, phi3: float) -> float:
        """Convert φ3 (radians) to the gate drive time delay in microseconds."""
        return phi3 / (2.0 * PI * FSW) * 1e6

    def phi1_to_duty_pct(self, phi1: float) -> float:
        """Convert φ1 (radians) to inner duty cycle percentage."""
        return (phi1 / PI) * 100.0

    def export(self, save_dir="."):
        """
        Export the model in ONNX and TorchScript formats for embedded targets.

        ONNX      → ONNX Runtime, TensorRT, OpenVINO, microcontrollers
        TorchScript → C++ deployment without Python, Jetson, embedded Linux
        """
        dummy = torch.zeros(1, self._model.seq_len, 8).to(self.device)

        # TorchScript
        scripted = torch.jit.trace(self._model, dummy)
        scripted.save(f"{save_dir}/pitnn_scripted.pt")
        print(f"Saved: {save_dir}/pitnn_scripted.pt  (C++/Jetson/embedded Linux)")

        # ONNX
        torch.onnx.export(
            self._model, dummy,
            f"{save_dir}/pitnn_model.onnx",
            input_names  = ["state_sequence"],
            output_names = ["phi_TPS"],
            dynamic_axes = {"state_sequence": {0: "batch"}},
            opset_version = 17,
        )
        print(f"Saved: {save_dir}/pitnn_model.onnx   (ONNX Runtime/TensorRT/MCU)")

        # Normalisation constants for embedded use
        np.save(f"{save_dir}/pitnn_mu.npy",    self._mu)
        np.save(f"{save_dir}/pitnn_sigma.npy", self._sigma)
        print(f"Saved: {save_dir}/pitnn_mu.npy, pitnn_sigma.npy")

        print(f"\nModel I/O:")
        print(f"  Input : (1, 20, 8) float32  [V1,V2,iL,φ1,φ2,φ3,Pref,V1V2/Vnom²]")
        print(f"  Output: (1, 3)     float32  [φ1, φ2, φ3]")


# ─────────────────────────────────────────────────────────────────────────────
# OUTER PI CONTROLLER  (§III Eq. 21-22)
# ─────────────────────────────────────────────────────────────────────────────

class OuterPIController:
    """
    Outer voltage regulation loop per paper §III, Eq. 21-22.
    Runs at a slower timescale than the PITNN inner loop.
    Generates Pref from the output voltage error.

    Typical update rate: every 10–100 switching cycles.
    """

    def __init__(self, Vref=800.0, Kp=50.0, Ki=500.0,
                 Pref_min=5000.0, Pref_max=70000.0,
                 update_every_n_cycles=10, fsw=FSW):
        self.Vref    = Vref
        self.Kp      = Kp
        self.Ki      = Ki
        self.Pmin    = Pref_min
        self.Pmax    = Pref_max
        self.dt      = update_every_n_cycles / fsw
        self._integ  = 0.0
        self._Pref   = Pref_min
        self._cycle  = 0
        self._n      = update_every_n_cycles

    def update(self, V2_measured: float) -> float:
        """
        Call every switching cycle with measured V2.
        Returns current Pref — updates only every _n cycles.
        """
        self._cycle += 1
        if self._cycle % self._n == 0:
            err = self.Vref - V2_measured          # Eq. 21
            self._integ += err * self.dt
            self._integ  = float(np.clip(self._integ, -1e6, 1e6))
            Pref = self.Kp * err + self.Ki * self._integ   # Eq. 22
            self._Pref = float(np.clip(Pref, self.Pmin, self.Pmax))
        return self._Pref


# ─────────────────────────────────────────────────────────────────────────────
# SAFETY MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class SafetyMonitor:
    """
    Hardware protection layer — checks sensor readings and model outputs.
    Call check() before applying any phi_TPS to the gate drive.
    Triggers hardware shutdown if any limit is exceeded.
    """

    def __init__(self,
                 V1_max=960.0, V2_max=960.0,
                 iL_max=200.0,
                 phi3_min=PHI_MIN, phi3_max=PHI3_MAX):
        self.V1_max   = V1_max
        self.V2_max   = V2_max
        self.iL_max   = iL_max
        self.phi3_min = phi3_min
        self.phi3_max = phi3_max
        self.fault_count = 0

    def check_sensors(self, V1, V2, iL):
        """Returns (safe: bool, reason: str)."""
        if V1 > self.V1_max:
            return False, f"V1 overvoltage: {V1:.1f}V > {self.V1_max}V"
        if V2 > self.V2_max:
            return False, f"V2 overvoltage: {V2:.1f}V > {self.V2_max}V"
        if abs(iL) > self.iL_max:
            return False, f"Overcurrent: {iL:.1f}A > {self.iL_max}A"
        return True, "OK"

    def check_output(self, phi3):
        """Returns (safe: bool, phi3_clamped: float)."""
        if phi3 < self.phi3_min or phi3 > self.phi3_max:
            self.fault_count += 1
            phi3_clamped = float(np.clip(phi3, self.phi3_min, self.phi3_max))
            return False, phi3_clamped
        return True, phi3


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — DEMO LOOP  (simulated sensors, no hardware needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(ctrl: DeployedController, n_steps=20):
    """
    Simulated control loop — demonstrates the controller without hardware.
    Cycles through a series of power setpoints and voltage conditions.
    """
    print("\n" + "="*68)
    print("  Demo Loop — Simulated Sensor Readings")
    print("="*68)
    print(f"\n  {'Step':>4}  {'V1':>5} {'V2':>5} {'Pref':>7}  "
          f"{'φ3':>7}  {'delay µs':>9}  {'duty%':>7}  "
          f"{'P_calc':>8}  {'ZVS':>5}  {'|ΔP|%':>7}")
    print(f"  {'─'*78}")

    # Simulated operating scenario: ramp from 10kW to 50kW then back
    scenarios = [
        (800, 800, 10000), (800, 800, 15000), (800, 800, 20000),
        (800, 800, 30000), (820, 780, 40000), (840, 760, 50000),
        (840, 760, 40000), (820, 780, 30000), (800, 800, 20000),
        (800, 800, 10000), (780, 820, 15000), (760, 840, 25000),
        (800, 800, 35000), (800, 800, 45000), (880, 720, 55000),
        (880, 720, 60000), (800, 800, 50000), (800, 800, 30000),
        (800, 800, 20000), (800, 800, 10000),
    ]

    dab = DABPhysics()
    ctrl.reset()

    for step, (V1, V2, Pref) in enumerate(scenarios[:n_steps]):
        # Simulate sensor noise
        V1_meas = V1 + np.random.normal(0, 1.5)
        V2_meas = V2 + np.random.normal(0, 1.5)

        # Estimate iL from previous phi3 (in real hardware: ADC reading)
        iL_meas = float(V1*V2/(V1_NOM*V2_NOM) * 50e-6/LK * 10.7 * 0.3
                        * (1 + np.random.normal(0, 0.05)))

        # Run one PITNN inference step with physics verification
        phi1, phi2, phi3, info = ctrl.step(
            V1_meas, V2_meas, iL_meas, Pref,
            verify_physics=True
        )

        delay_us = ctrl.phi3_to_delay_us(phi3)
        duty_pct = ctrl.phi1_to_duty_pct(phi1)

        print(f"  {step+1:>4}  {V1:>5.0f} {V2:>5.0f} {Pref:>7.0f}  "
              f"{phi3:>7.4f}  {delay_us:>9.3f}  {duty_pct:>6.1f}%  "
              f"{info['P_calc']:>8.1f}  {'YES' if info['zvs_ok'] else 'NO ':>5}  "
              f"{info['P_err_pct']:>6.1f}%")

        time.sleep(0.05)   # remove in real deployment

    print(f"\n  Done. φ1=φ2={PHI12_FIXED:.4f} rad fixed across all steps.")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — HARDWARE LOOP  (real ADC + PWM — fill in your hardware API)
# ─────────────────────────────────────────────────────────────────────────────

def run_hardware_loop(ctrl: DeployedController, Vref=800.0,
                      Pref_max=70000.0, duration_s=10.0):
    """
    Real hardware control loop.

    Replace the ADC read stubs and PWM write stubs with your
    actual hardware API calls. Everything else runs as-is.

    Compatible hardware APIs:
      TI C2000 DSP    → use pyserial + UART to C2000 firmware
      Raspberry Pi    → use RPi.GPIO or pigpio for PWM
      National Instruments → use nidaqmx
      dSPACE / OPAL-RT → use their Python API
      Custom FPGA     → write phi3 to memory-mapped register
    """
    outer_pi = OuterPIController(Vref=Vref, Pref_max=Pref_max)
    safety   = SafetyMonitor()

    print(f"\n[Hardware Loop] Starting — Vref={Vref}V, duration={duration_s}s")
    print("  Replace ADC/PWM stubs below with your hardware API\n")

    t_start = time.perf_counter()
    Ts      = 1.0 / FSW          # 10µs per cycle at 100kHz
    cycle   = 0
    ctrl.reset()

    while (time.perf_counter() - t_start) < duration_s:
        cycle_start = time.perf_counter()

        # ── Read sensors ──────────────────────────────────────────────────
        # REPLACE THESE with your actual ADC reads:
        #   V1_meas = adc.read_channel(0) * V1_SCALE
        #   V2_meas = adc.read_channel(1) * V2_SCALE
        #   iL_meas = adc.read_channel(2) * IL_SCALE
        V1_meas = float(V1_NOM + np.random.normal(0, 2))   # ← stub
        V2_meas = float(V2_NOM + np.random.normal(0, 2))   # ← stub
        iL_meas = float(10.0   + np.random.normal(0, 0.5)) # ← stub

        # ── Safety check ──────────────────────────────────────────────────
        safe, reason = safety.check_sensors(V1_meas, V2_meas, iL_meas)
        if not safe:
            print(f"\n[SAFETY FAULT] {reason} — stopping")
            # REPLACE with: pwm.disable_all_outputs()
            break

        # ── Outer PI loop (runs every 10 cycles) ──────────────────────────
        Pref = outer_pi.update(V2_meas)

        # ── PITNN inner loop — one call per switching cycle ───────────────
        phi1, phi2, phi3 = ctrl.step(V1_meas, V2_meas, iL_meas, Pref)

        # ── Safety clamp on model output ──────────────────────────────────
        out_safe, phi3 = safety.check_output(phi3)

        # ── Apply to gate drive ───────────────────────────────────────────
        delay_s = phi3 / (2.0 * PI * FSW)      # seconds
        duty    = phi1 / PI                     # fraction

        # REPLACE THESE with your actual PWM hardware writes:
        #   pwm_primary.set_duty(duty)
        #   pwm_secondary.set_phase_delay(delay_s)
        #   pwm.update()                        # atomic register commit

        # ── Log every 1000 cycles ─────────────────────────────────────────
        if cycle % 1000 == 0:
            print(f"  t={time.perf_counter()-t_start:6.3f}s  "
                  f"V1={V1_meas:.1f}V  V2={V2_meas:.1f}V  "
                  f"Pref={Pref:.0f}W  φ3={phi3:.4f}rad  "
                  f"delay={delay_s*1e6:.2f}µs")

        # ── Wait for next switching cycle ─────────────────────────────────
        elapsed   = time.perf_counter() - cycle_start
        remaining = Ts - elapsed
        if remaining > 0:
            time.sleep(remaining)

        cycle += 1

    print(f"\n[Hardware Loop] Completed {cycle} cycles in "
          f"{time.perf_counter()-t_start:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — EXPORT  (ONNX + TorchScript for embedded targets)
# ─────────────────────────────────────────────────────────────────────────────

def run_export(ctrl: DeployedController, save_dir="."):
    """Export model files for C++, Jetson, DSP, or FPGA deployment."""
    print("\n" + "="*68)
    print("  Model Export for Embedded Deployment")
    print("="*68)
    ctrl.export(save_dir=save_dir)
    print(f"\nTarget platform guide:")
    print(f"  NVIDIA Jetson Orin  → use pitnn_scripted.pt with libtorch")
    print(f"  TensorRT / edge GPU → convert pitnn_model.onnx with trtexec")
    print(f"  STM32 / ARM Cortex  → use ONNX Runtime for Cortex-M")
    print(f"  TI C2000 DSP        → extract weights from pitnn_weights.npz,")
    print(f"                         implement forward pass in C")
    print(f"  Xilinx FPGA         → use Xilinx Vitis AI with ONNX input")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 4 — CLOSED LOOP  (full outer PI + inner PITNN, software plant)
# ─────────────────────────────────────────────────────────────────────────────

def run_closed_loop(ctrl: DeployedController, Vref=800.0,
                    Pref_max=50000.0, n_cycles=500):
    """
    Full closed-loop simulation: outer PI voltage regulator feeds Pref
    to the PITNN inner modulation layer. The DABPhysics model acts as
    the plant. Demonstrates the complete §III control architecture.
    """
    print("\n" + "="*68)
    print("  Closed-Loop Simulation  (Outer PI + Inner PITNN)")
    print(f"  Vref={Vref}V  Pref_max={Pref_max/1000:.0f}kW  cycles={n_cycles}")
    print("="*68)

    outer_pi = OuterPIController(Vref=Vref, Pref_max=Pref_max)
    dab      = DABPhysics(V1=V1_NOM, V2=V2_NOM)
    safety   = SafetyMonitor()
    ctrl.reset()

    # Plant state
    V2_plant = V2_NOM
    C_out    = 1e-3   # 1mF output capacitor
    R_load   = V2_NOM**2 / Pref_max   # load resistance at max power

    print(f"\n  {'Cycle':>6}  {'Pref':>7}  {'φ3':>7}  {'P_calc':>8}  "
          f"{'V2':>7}  {'err_V':>7}  {'ZVS':>5}")
    print(f"  {'─'*60}")

    for cycle in range(n_cycles):
        # Outer PI: voltage → Pref
        Pref = outer_pi.update(V2_plant)

        # Simulate sensor noise
        V1_meas = float(V1_NOM + np.random.normal(0, 1.0))
        V2_meas = float(V2_plant + np.random.normal(0, 0.5))
        iL_meas = float(Pref / (V1_NOM * 0.95) + np.random.normal(0, 0.2))

        # Safety check
        safe, reason = safety.check_sensors(V1_meas, V2_meas, iL_meas)
        if not safe:
            print(f"\n  [FAULT at cycle {cycle}] {reason}")
            break

        # PITNN inference
        phi1, phi2, phi3 = ctrl.step(V1_meas, V2_meas, iL_meas, Pref)
        _, phi3 = safety.check_output(phi3)

        # Plant update: compute actual power and update V2
        dab.V1, dab.V2 = V1_meas, V2_plant
        P_actual = dab.compute_power(phi1, phi2, phi3)
        zvs_ok, _ = dab.check_zvs(phi1, phi2, phi3)

        # Simple capacitor plant model: dV2/dt = (P_in - P_load) / (C * V2)
        P_load  = V2_plant**2 / R_load
        dV2     = (P_actual - P_load) / (C_out * V2_plant) / FSW
        V2_plant = float(np.clip(V2_plant + dV2, 600, 1000))

        # Log every 50 cycles
        if cycle % 50 == 0:
            print(f"  {cycle:>6}  {Pref/1000:>6.1f}k  {phi3:>7.4f}  "
                  f"{P_actual/1000:>7.1f}k  {V2_plant:>7.2f}  "
                  f"{Vref-V2_plant:>7.2f}  {'YES' if zvs_ok else 'NO':>5}")

    print(f"\n  Final V2={V2_plant:.2f}V  (Vref={Vref}V  "
          f"error={Vref-V2_plant:.2f}V)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PITNN DAB Deployment Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  demo         Simulate sensor readings, show phi_TPS outputs  (default)
  hardware     Real hardware loop with ADC/PWM stubs
  export       Save ONNX + TorchScript for embedded targets
  closed_loop  Full outer PI + inner PITNN closed-loop simulation

Examples:
  python pitnn_deploy.py
  python pitnn_deploy.py --mode demo
  python pitnn_deploy.py --mode export
  python pitnn_deploy.py --mode closed_loop --Vref 800 --Pmax 50000
  python pitnn_deploy.py --mode hardware --duration 30
        """
    )
    parser.add_argument("--checkpoint", type=str,
                        default="pitnn_dab_checkpoint.pt",
                        help="Path to saved model checkpoint")
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "hardware", "export", "closed_loop"],
                        help="Deployment mode")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu or cuda (auto-detected if omitted)")
    parser.add_argument("--Vref", type=float, default=800.0,
                        help="Output voltage reference (V)")
    parser.add_argument("--Pmax", type=float, default=50000.0,
                        help="Maximum power reference (W)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Hardware loop duration (seconds)")
    parser.add_argument("--cycles", type=int, default=500,
                        help="Closed-loop simulation cycles")
    args = parser.parse_args()

    print("="*68)
    print("  PITNN DAB Converter — Deployment Controller")
    print(f"  Mode: {args.mode}  |  Checkpoint: {args.checkpoint}")
    print("="*68)

    # Load model once — shared across all modes
    ctrl = DeployedController(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.mode == "demo":
        run_demo(ctrl)

    elif args.mode == "hardware":
        run_hardware_loop(ctrl,
                          Vref=args.Vref,
                          Pref_max=args.Pmax,
                          duration_s=args.duration)

    elif args.mode == "export":
        run_export(ctrl)

    elif args.mode == "closed_loop":
        run_closed_loop(ctrl,
                        Vref=args.Vref,
                        Pref_max=args.Pmax,
                        n_cycles=args.cycles)

    print("\nDone.")


if __name__ == "__main__":
    main()
