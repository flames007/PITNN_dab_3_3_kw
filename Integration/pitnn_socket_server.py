"""
pitnn_socket_server.py
=====================================================================
PITNN DAB Converter — TCP Socket Server for PLECS RT / HIL
Method 4: Runs on a PC alongside a PLECS real-time or HIL target.

The PLECS simulation (or a C-Script block) connects as a TCP client,
sends sensor readings, and receives phase shift angles each cycle.

HOW TO USE
───────────
Step 1 — Start this server BEFORE launching PLECS:
    python pitnn_socket_server.py

Step 2 — In PLECS, use the C-Script socket client code from
    pitnn_socket_client.h to connect to this server.

Step 3 — Run the PLECS simulation. The server processes each
    inference request and returns phi1, phi2, phi3 in real time.

PROTOCOL
─────────
  Client → Server : 4 x float32 = 16 bytes  [V1, V2, iL, Pref]
  Server → Client : 3 x float32 = 12 bytes  [phi1, phi2, phi3]

  All floats are little-endian IEEE 754 single precision.

CONFIGURATION
──────────────
  HOST : "127.0.0.1" for same machine, or PC IP for remote HIL board
  PORT : 9999 (must match PLECS C-Script client)

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University — All Rights Reserved
=====================================================================
"""

import socket
import struct
import time
import os
import sys

# ── SET THIS PATH ─────────────────────────────────────────────────
PITNN_DIR = r"C:\Users\Nzead\Documents\Research\Power electronics control system\Simulation\pitnn_dab"
HOST      = "127.0.0.1"   # Change to PC LAN IP for remote HIL
PORT      = 9999
# ─────────────────────────────────────────────────────────────────

sys.path.insert(0, PITNN_DIR)
from pitnn_inference import PITNNInference


def run_server():
    """Start the PITNN TCP socket server."""

    # Load model
    print("=" * 60)
    print("  PITNN Socket Server — PLECS RT / HIL Integration")
    print("=" * 60)

    ctrl = PITNNInference(
        model_path = os.path.join(PITNN_DIR, "pitnn_scripted.pt"),
        mu_path    = os.path.join(PITNN_DIR, "pitnn_mu.npy"),
        sigma_path = os.path.join(PITNN_DIR, "pitnn_sigma.npy"),
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        print(f"\n[Server] Listening on {HOST}:{PORT}")
        print(f"[Server] Waiting for PLECS to connect...")
        print(f"[Server] Protocol: send 16 bytes (4×float32), receive 12 bytes (3×float32)")
        print(f"[Server] Press Ctrl+C to stop\n")

        while True:
            try:
                conn, addr = srv.accept()
                print(f"[Server] PLECS connected from {addr}")
                _handle_connection(conn, ctrl)
                print(f"[Server] Connection closed — waiting for reconnect...")
            except KeyboardInterrupt:
                print("\n[Server] Stopped by user")
                break


def _handle_connection(conn: socket.socket, ctrl: PITNNInference):
    """Handle one PLECS client connection."""
    n_calls    = 0
    total_us   = 0.0
    ctrl.reset()

    with conn:
        while True:
            # ── Receive 16 bytes: [V1, V2, iL, Pref] ─────────────
            data = _recv_exact(conn, 16)
            if data is None:
                break

            V1, V2, iL, Pref = struct.unpack("<4f", data)

            # ── PITNN inference ───────────────────────────────────
            t0 = time.perf_counter()
            phi1, phi2, phi3 = ctrl.step(V1, V2, iL, Pref)
            t_us = (time.perf_counter() - t0) * 1e6

            n_calls  += 1
            total_us += t_us

            # ── Send 12 bytes: [phi1, phi2, phi3] ────────────────
            response = struct.pack("<3f", phi1, phi2, phi3)
            try:
                conn.sendall(response)
            except BrokenPipeError:
                break

            # Log every 10,000 calls
            if n_calls % 10000 == 0:
                avg_us = total_us / n_calls
                print(f"[Server] Calls: {n_calls:,}  Avg latency: {avg_us:.2f}µs  "
                      f"Last: phi3={phi3:.4f} rad  Pref={Pref:.0f}W")

    print(f"[Server] Session ended — {n_calls:,} calls, "
          f"avg {total_us/max(n_calls,1):.2f}µs/call")


def _recv_exact(conn: socket.socket, n_bytes: int):
    """Receive exactly n_bytes from socket, or return None on disconnect."""
    data = b""
    while len(data) < n_bytes:
        try:
            chunk = conn.recv(n_bytes - len(data))
        except ConnectionResetError:
            return None
        if not chunk:
            return None
        data += chunk
    return data


if __name__ == "__main__":
    run_server()
