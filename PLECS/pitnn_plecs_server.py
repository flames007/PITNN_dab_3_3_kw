"""
pitnn_plecs_server.py
=====================================================================
PITNN Inference Server for PLECS Co-Simulation
Option B: PI regulates delivered power P_measured = V2 × I_out
          PI output → P_ref_corrected → PITNN → (φ1, φ2, φ3)

Signal flow:
    PLECS sends:  [V1, V2, I_out, P_ref_ext]   (4 floats, CSV over TCP)
    Server:       1. Computes P_measured = V2 × I_out
                  2. PI corrects P_ref:  P_ref_corr = P_ref_ext + PI(P_err)
                  3. PITNN forward pass → (φ1, φ2, φ3)
    PLECS gets:   [phi1, phi2, phi3, P_ref_corr, P_measured]  (5 floats, CSV)

PLECS C-Script calls this server every PITNN_UPDATE_CYCLES switching
cycles (500 µs at fsw=100kHz) using a TCP socket on localhost:9876.

Usage:
    python pitnn_plecs_server.py
    python pitnn_plecs_server.py --checkpoint pitnn_dab_checkpoint.pt
    python pitnn_plecs_server.py --port 9876 --warmup 10 --device cuda

Converter spec:
    V1=400V, V2=250V, n=1.6, Lk=40µH, fsw=100kHz, P_rated=3.3kW
=====================================================================
"""

import os as _os
_DEFAULT_CKPT = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),   # folder this script lives in
    "..",                                            # one level up
    "pitnn_dab_checkpoint.pt"
)
import argparse
import socket
import struct
import time
import math
import threading
import numpy as np
import torch
import sys
from pathlib import Path

# Get the absolute path of the parent directory
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.insert(0, str(parent_dir))

# ── Import PITNN infrastructure ───────────────────────────────────────────────
from pitnn_dab import (
    PITNN, DABPhysics, PITNNController,
    V1_NOM, V2_NOM, FSW, LK, N_TURNS, PI,
    PHI12_MIN, PHI12_MAX, PHI_MIN, PHI3_MAX, PHI12_NOM,
    P_RATED,
)

# ─────────────────────────────────────────────────────────────────────────────
# PI CONTROLLER  (runs inside the server, Option B)
# ─────────────────────────────────────────────────────────────────────────────

class PowerPI:
    """
    Discrete PI controller that corrects the external power reference
    based on measured delivery error.

    P_ref_corrected = P_ref_ext + Kp*e + Ki*integral(e)*dt

    Anti-windup: integral is clamped so P_ref_corrected never exceeds
    [P_MIN, P_MAX]. Reset is called whenever P_ref_ext changes by more
    than STEP_THRESHOLD watts to avoid integrator wind-up during steps.

    dt = PITNN_UPDATE_CYCLES / fsw = 50 / 100e3 = 500µs
    """

    def __init__(self,
                 Kp: float = 0.05,
                 Ki: float = 2.0,
                 dt: float = 50.0 / 100e3,
                 P_min: float = 500.0,
                 P_max: float = 3300.0,
                 step_threshold: float = 100.0):
        self.Kp  = Kp
        self.Ki  = Ki
        self.dt  = dt
        self.P_min = P_min
        self.P_max = P_max
        self.step_threshold = step_threshold

        self._integral  = 0.0
        self._prev_pref = 0.0
        self._int_max   = (P_max - P_min) / max(Ki * dt, 1e-9)

    def reset(self):
        self._integral = 0.0

    def step(self, P_ref_ext: float, P_measured: float) -> float:
        """
        One PI update step.

        Parameters
        ----------
        P_ref_ext   : External power reference (W) — from PLECS signal source
        P_measured  : Delivered power V2 × I_out (W) — from PLECS sensors

        Returns
        -------
        P_ref_corr  : Corrected power reference clamped to [P_min, P_max]
        """
        # Reset integrator on large reference step to avoid windup
        if abs(P_ref_ext - self._prev_pref) > self.step_threshold:
            self._integral = 0.0
        self._prev_pref = P_ref_ext

        error = P_ref_ext - P_measured
        self._integral = float(np.clip(
            self._integral + error * self.dt,
            -self._int_max, self._int_max
        ))

        P_corr = P_ref_ext + self.Kp * error + self.Ki * self._integral
        return float(np.clip(P_corr, self.P_min, self.P_max))


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE SERVER
# ─────────────────────────────────────────────────────────────────────────────

class PITNNServer:
    """
    TCP server that wraps the PITNN controller and PowerPI for PLECS.

    Protocol (both directions: newline-terminated ASCII CSV):
        PLECS → Server:  "V1,V2,I_out,P_ref_ext\n"
        Server → PLECS:  "phi1,phi2,phi3,P_ref_corr,P_measured\n"

    Thread-safe: each PLECS connection gets its own controller state.
    Only one PLECS client is expected at a time.
    """

    def __init__(self,
                 checkpoint_path: str,
                 host: str = "127.0.0.1",
                 port: int = 9876,
                 device: str = "cpu",
                 n_warmup: int = 10,
                 pi_kp: float = 0.05,
                 pi_ki: float = 2.0):

        self.host     = host
        self.port     = port
        self.device   = device
        self.n_warmup = n_warmup

        # ── Load checkpoint ───────────────────────────────────────────────────
        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path,
                          map_location=device, weights_only=False)
        hp   = ckpt["hyperparams"]

        self.mu    = ckpt["mu"].astype(np.float32)
        self.sigma = ckpt["sigma"].astype(np.float32)

        model = PITNN(
            d_in    = hp.get("d_in",    8),
            d_model = hp.get("d_model", 128),
            n_heads = hp.get("n_heads", 8),
            n_layers= hp.get("n_layers",4),
            d_ff    = hp.get("d_ff",    256),
            seq_len = hp.get("seq_len", 20),
            dropout = 0.0,   # inference — dropout off
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self.model = model.to(device)

        # Warm up the model (first inference is slow due to JIT compilation)
        dummy = torch.zeros(1, 20, 8, device=device)
        with torch.no_grad():
            for _ in range(5):
                self.model(dummy)
        print(f"  Model loaded: 565K params | device={device}")

        self.pi_kp = pi_kp
        self.pi_ki = pi_ki

        self._lock   = threading.Lock()
        self._server = None
        self._stats  = {"calls": 0, "inf_us_mean": 0.0, "errors": 0}

    # ── Per-connection handler ─────────────────────────────────────────────────
    def _handle_client(self, conn: socket.socket, addr):
        print(f"  [Server] PLECS connected from {addr}")

        dab  = DABPhysics()
        ctrl = PITNNController(self.model, self.mu, self.sigma, dab,
                               device=self.device)
        pi   = PowerPI(Kp=self.pi_kp, Ki=self.pi_ki)

        # State
        phi_cur   = None
        primed    = False
        call_idx  = 0

        try:
            buf = ""
            while True:
                chunk = conn.recv(256).decode("ascii", errors="replace")
                if not chunk:
                    break
                buf += chunk

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    # ── Parse incoming message ────────────────────────────────
                    try:
                        parts = [float(x) for x in line.split(",")]
                        if len(parts) < 4:
                            raise ValueError(f"Expected 4 values, got {len(parts)}")
                        V1, V2, I_out, P_ref_ext = parts[:4]
                    except Exception as e:
                        self._stats["errors"] += 1
                        conn.sendall(
                            f"{PHI12_NOM:.6f},{PHI12_NOM:.6f},0.100000,"
                            f"{P_ref_ext if 'P_ref_ext' in dir() else 500.0:.2f},"
                            f"0.00\n".encode()
                        )
                        continue

                    # ── Separate RAW plant measurements from NN-safe inputs ─────
                    # Raw values are the actual PLECS sensor values. These must be used
                    # for physical delivered-power feedback.
                    V1_raw        = float(V1)
                    V2_raw        = float(V2)
                    I_out_raw     = float(I_out)
                    P_ref_ext_raw = float(P_ref_ext)

                    # Use the raw output voltage/current to measure actual delivered power.
                    # Do NOT compute P_measured using the clamped V2_nn value; that was the
                    # reason P_meas stayed around 220*Iout even when Vout collapsed below 220 V.
                    I_out_meas = max(I_out_raw, 0.0)
                    P_measured = V2_raw * I_out_meas
                    if P_measured < 0.0:
                        P_measured = 0.0

                    # Clamp only the values that are sent into the trained PITNN/DABPhysics
                    # model. This keeps the NN inside its training range without corrupting
                    # the physical power feedback.
                    V1_nn     = float(np.clip(V1_raw, 360.0, 440.0))
                    V2_nn     = float(np.clip(V2_raw, 220.0, 280.0))
                    I_out_nn  = I_out_meas
                    P_ref_ext = float(np.clip(P_ref_ext_raw, 500.0, 3300.0))

                    # ── PI correction uses TRUE measured output power ──────────
                    P_ref_corr = pi.step(P_ref_ext, P_measured)

                    # ── Prime controller on first call ────────────────────────
                    dab.V1, dab.V2 = V1_nn, V2_nn
                    if not primed:
                        phi_seed = dab.solve_optimal_phi(P_ref_corr)
                        ctrl.reset()
                        ctrl.prime(V1_nn, V2_nn, P_ref_corr, phi_seed)
                        phi_cur = phi_seed
                        # Warm-up steps
                        for _ in range(self.n_warmup):
                            r = ctrl.step(V1_nn, V2_nn, None, P_ref_corr, phi_cur)
                            phi_cur = r["phi_TPS"]
                        primed = True

                    # ── PITNN forward pass ────────────────────────────────────
                    t0 = time.perf_counter()
                    r  = ctrl.step(V1_nn, V2_nn, None, P_ref_corr, phi_cur)
                    inf_us = (time.perf_counter() - t0) * 1e6

                    phi_cur = r["phi_TPS"]
                    phi1, phi2, phi3 = float(phi_cur[0]), float(phi_cur[1]), float(phi_cur[2])

                    # ── Update stats ──────────────────────────────────────────
                    call_idx += 1
                    with self._lock:
                        n = self._stats["calls"]
                        self._stats["inf_us_mean"] = (
                            self._stats["inf_us_mean"] * n + inf_us) / (n + 1)
                        self._stats["calls"] = n + 1

                    # ── Send response to PLECS ────────────────────────────────
                    response = (
                        f"{phi1:.6f},{phi2:.6f},{phi3:.6f},"
                        f"{P_ref_corr:.4f},{P_measured:.4f}\n"
                    )
                    conn.sendall(response.encode("ascii"))

                    # ── Console log every 200 calls (~100ms) ──────────────────
                    if call_idx % 200 == 0:
                        print(f"  [t={call_idx*0.5:.0f}ms] "
                              f"P_ext={P_ref_ext:.0f}W  "
                              f"P_meas={P_measured:.0f}W  "
                              f"P_corr={P_ref_corr:.0f}W  "
                              f"φ3={phi3:.3f}  "
                              f"ZVS={'YES' if r['zvs_ok'] else 'NO'}  "
                              f"inf={inf_us:.1f}µs")

        except Exception as e:
            print(f"  [Server] Client error: {e}")
        finally:
            conn.close()
            print(f"  [Server] PLECS disconnected. "
                  f"Total calls: {call_idx}  "
                  f"Mean inference: {self._stats['inf_us_mean']:.1f}µs")

    # ── Main server loop ───────────────────────────────────────────────────────
    def run(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)

        print(f"\n{'='*60}")
        print(f"  PITNN PLECS Server — Option B (PI power regulation)")
        print(f"  Listening on {self.host}:{self.port}")
        print(f"  Converter: V1={V1_NOM:.0f}V / V2={V2_NOM:.0f}V / "
              f"P_rated={P_RATED:.0f}W / fsw={FSW/1e3:.0f}kHz")
        print(f"  PI gains: Kp={self.pi_kp}  Ki={self.pi_ki}")
        print(f"  Warmup steps per connection: {self.n_warmup}")
        print(f"  Waiting for PLECS to connect ...")
        print(f"{'='*60}\n")

        try:
            while True:
                conn, addr = self._server.accept()
                t = threading.Thread(
                    target=self._handle_client,
                    args=(conn, addr),
                    daemon=True,
                )
                t.start()
        except KeyboardInterrupt:
            print("\n  [Server] Shutting down.")
        finally:
            self._server.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PITNN PLECS Inference Server — Option B PI power control")
    parser.add_argument("--checkpoint", default=_DEFAULT_CKPT,
                        help="Path to trained PITNN checkpoint "
                             "(default: ../pitnn_dab_checkpoint.pt relative to this script)")
    parser.add_argument("--host",    default="127.0.0.1",
                        help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port",    type=int, default=9876,
                        help="TCP port (default: 9876)")
    parser.add_argument("--device",  default="cpu",
                        help="Torch device: cpu or cuda (default: cpu)")
    parser.add_argument("--warmup",  type=int, default=10,
                        help="Controller warm-up steps on connect (default: 10)")
    parser.add_argument("--kp",      type=float, default=0.05,
                        help="PI proportional gain (default: 0.05)")
    parser.add_argument("--ki",      type=float, default=2.0,
                        help="PI integral gain (default: 2.0)")
    args = parser.parse_args()

    server = PITNNServer(
        checkpoint_path = args.checkpoint,
        host     = args.host,
        port     = args.port,
        device   = args.device,
        n_warmup = args.warmup,
        pi_kp    = args.kp,
        pi_ki    = args.ki,
    )
    server.run()


if __name__ == "__main__":
    main()
