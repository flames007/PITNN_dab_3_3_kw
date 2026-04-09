"""
pitnn_uart_server.py
=====================================================================
PITNN DAB Converter — UART Server for TI C2000 DSP
Runs on the companion PC connected to the C2000 via USB-to-UART.

HARDWARE CONNECTION
────────────────────
  PC USB port  →  USB-to-UART adapter (3.3V logic, e.g. CP2102 or FTDI)
  Adapter TX   →  C2000 GPIO29 (SCI-A RX)
  Adapter RX   →  C2000 GPIO28 (SCI-A TX)
  Adapter GND  →  C2000 GND
  (Do NOT connect 3.3V — C2000 is self-powered)

PROTOCOL
─────────
  C2000 → PC  : 16 bytes  [V1, V2, iL, Pref]  as 4 × float32 little-endian
  PC    → C2000: 12 bytes  [phi1, phi2, phi3]  as 3 × float32 little-endian

HOW TO USE
───────────
  Step 1 — Connect the USB-to-UART adapter between PC and C2000.

  Step 2 — Find the COM port Windows assigned to the adapter:
             Device Manager → Ports (COM & LPT) → note the COMx number.
             Update COM_PORT below.

  Step 3 — Start this server BEFORE powering the C2000:
             python pitnn_uart_server.py

  Step 4 — Power on the C2000. The firmware (pitnn_c2000.c) will
             start sending ADC data immediately. This server responds
             with phi1, phi2, phi3 each time.

  Step 5 — To stop: Ctrl+C. The C2000 will hold its last phi values.

TIMING
───────
  The C2000 sends every PITNN_UPDATE_CYCLES switching cycles (default 50).
  At fsw=100kHz and 50 cycles that is one update per 500µs.
  Round-trip latency on USB-UART at 921600 baud is ~350µs for 28 bytes,
  leaving ~150µs margin. Increase PITNN_UPDATE_CYCLES if you see timeouts.

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University — All Rights Reserved
=====================================================================
"""

import serial
import struct
import time
import os
import sys
import argparse

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
PITNN_DIR = r"C:\Users\Nzead\Documents\Research\Power electronics control system\Simulation\pitnn_dab"
COM_PORT  = "COM3"        # ← change to your USB-UART adapter COM port
BAUD_RATE = 921600        # must match SciaRegs baud in pitnn_c2000_setup.c
TIMEOUT_S = 0.005         # 5ms receive timeout — increase if seeing dropped packets
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, PITNN_DIR)
from pitnn_inference import PITNNInference


def run_server(com_port: str, baud_rate: int):
    print("=" * 62)
    print("  PITNN UART Server — TI C2000 DSP Integration")
    print("=" * 62)
    print(f"  Port     : {com_port}  @  {baud_rate} baud")
    print(f"  Protocol : send 16 B [V1,V2,iL,Pref], recv 12 B [φ1,φ2,φ3]")
    print(f"  Model    : {PITNN_DIR}")
    print()

    # Load PITNN
    ctrl = PITNNInference(
        model_path = os.path.join(PITNN_DIR, "pitnn_scripted.pt"),
        mu_path    = os.path.join(PITNN_DIR, "pitnn_mu.npy"),
        sigma_path = os.path.join(PITNN_DIR, "pitnn_sigma.npy"),
    )
    # Prime with a nominal operating point so the first inference is meaningful
    ctrl.prime(V1=800.0, V2=800.0, Pref=20000.0)
    print("[Server] PITNN loaded and primed — ready")

    # Open serial port
    try:
        ser = serial.Serial(
            port        = com_port,
            baudrate    = baud_rate,
            bytesize    = serial.EIGHTBITS,
            parity      = serial.PARITY_NONE,
            stopbits    = serial.STOPBITS_ONE,
            timeout     = TIMEOUT_S,
        )
    except serial.SerialException as e:
        print(f"\n[ERROR] Cannot open {com_port}: {e}")
        print("  Check: Device Manager → Ports to find the correct COM port.")
        sys.exit(1)

    print(f"[Server] {com_port} open — waiting for C2000 data...")
    print(f"[Server] Press Ctrl+C to stop\n")

    n_calls   = 0
    n_timeout = 0
    total_us  = 0.0
    t_start   = time.perf_counter()

    try:
        while True:
            # ── Receive 16 bytes from C2000: [V1, V2, iL, Pref] ──────────────
            raw = _recv_exact(ser, 16)

            if raw is None:
                # Timeout — C2000 may not have started yet or UART misaligned
                n_timeout += 1
                if n_timeout % 200 == 0:
                    print(f"[Server] Waiting for C2000 ... ({n_timeout} timeouts)")
                # Flush input buffer to re-sync after misalignment
                ser.reset_input_buffer()
                continue

            n_timeout = 0   # reset timeout counter on successful receive

            try:
                V1, V2, iL, Pref = struct.unpack('<4f', raw)
            except struct.error:
                ser.reset_input_buffer()
                continue

            # Basic sanity check — reject obviously corrupt packets
            if not (500.0 < V1 < 1100.0 and 500.0 < V2 < 1100.0
                    and abs(iL) < 300.0 and 0.0 < Pref < 100000.0):
                ser.reset_input_buffer()
                continue

            # ── PITNN inference ───────────────────────────────────────────────
            t0 = time.perf_counter()
            phi1, phi2, phi3 = ctrl.step(V1, V2, iL, Pref)
            inf_us = (time.perf_counter() - t0) * 1e6

            n_calls  += 1
            total_us += inf_us

            # ── Send 12 bytes back to C2000: [phi1, phi2, phi3] ──────────────
            response = struct.pack('<3f', phi1, phi2, phi3)
            ser.write(response)

            # ── Log every 1000 calls (~every 0.5s at 2kHz update rate) ───────
            if n_calls % 1000 == 0:
                elapsed = time.perf_counter() - t_start
                rate    = n_calls / elapsed
                avg_us  = total_us / n_calls
                print(f"[Server] {n_calls:>7,} calls | "
                      f"{rate:.0f} Hz | "
                      f"avg inf {avg_us:.1f}µs | "
                      f"V1={V1:.1f}V V2={V2:.1f}V "
                      f"Pref={Pref/1000:.1f}kW | "
                      f"φ1={phi1:.4f} φ3={phi3:.4f}")

    except KeyboardInterrupt:
        print(f"\n[Server] Stopped — {n_calls:,} total calls")
        avg = total_us / max(n_calls, 1)
        print(f"[Server] Average PITNN latency: {avg:.1f} µs")

    finally:
        ser.close()


def _recv_exact(ser: serial.Serial, n: int):
    """Read exactly n bytes, return None on timeout or short read."""
    data = ser.read(n)
    return data if len(data) == n else None


def main():
    parser = argparse.ArgumentParser(
        description="PITNN UART Server for TI C2000 DSP"
    )
    parser.add_argument("--port",  default=COM_PORT,  help="COM port (e.g. COM3)")
    parser.add_argument("--baud",  default=BAUD_RATE, type=int,
                        help="Baud rate (must match C2000 firmware)")
    args = parser.parse_args()
    run_server(args.port, args.baud)


if __name__ == "__main__":
    main()
