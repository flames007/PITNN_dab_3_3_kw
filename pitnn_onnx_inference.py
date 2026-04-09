"""
PITNN DAB Converter — ONNX Runtime Inference (Option 3)
========================================================
Runs the trained PITNN using ONNX Runtime.
Works on any platform: Windows, Linux, Raspberry Pi, ARM Cortex,
NVIDIA Jetson with TensorRT, Intel with OpenVINO.

Required files (same folder as this script):
    pitnn_model.onnx    — ONNX model (export with: python pitnn_deploy.py --mode export)
    pitnn_mu.npy        — normalisation means
    pitnn_sigma.npy     — normalisation standard deviations

Install:
    pip install onnxruntime numpy          # CPU only
    pip install onnxruntime-gpu numpy      # GPU (CUDA)

Run:
    python pitnn_onnx_inference.py

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University — All Rights Reserved
"""

import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match values used during training)
# ─────────────────────────────────────────────────────────────────────────────
V1_NOM    = 800.0
V2_NOM    = 800.0
FSW       = 100e3
PI        = 3.141592653589793
PHI12_MIN = PI * 0.65   # 2.0420 rad — lower bound for phi1/phi2
PHI12_MAX = PI * 0.99   # 3.1102 rad — upper bound for phi1/phi2
PHI_MIN   = 0.02         # rad — lower bound for phi3
PHI3_MAX  = 1.50
SEQ_LEN   = 20
N_FEAT    = 8


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ONNX SESSION
# ─────────────────────────────────────────────────────────────────────────────

def load_onnx_session(model_path="pitnn_model.onnx",
                      mu_path="pitnn_mu.npy",
                      sigma_path="pitnn_sigma.npy"):
    """
    Load the ONNX Runtime inference session and normalisation constants.

    Execution providers are tried in order of preference:
      1. TensorrtExecutionProvider  — NVIDIA GPU via TensorRT (fastest)
      2. CUDAExecutionProvider      — NVIDIA GPU via CUDA
      3. CPUExecutionProvider       — fallback, always available

    Returns (session, mu, sigma).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "pip install onnxruntime        # CPU\n"
            "pip install onnxruntime-gpu    # GPU with CUDA"
        )

    providers = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Only keep providers that are actually available
    available = ort.get_available_providers()
    providers  = [p for p in providers if p in available]

    print(f"[ONNX] Loading model: {model_path}")
    print(f"[ONNX] Execution providers: {providers}")
    session = ort.InferenceSession(model_path, providers=providers)

    mu    = np.load(mu_path).astype(np.float32)
    sigma = np.load(sigma_path).astype(np.float32)

    # Verify model I/O shape
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"[ONNX] Input  : {inp.name}  shape={inp.shape}  type={inp.type}")
    print(f"[ONNX] Output : {out.name}  shape={out.shape}  type={out.type}")

    return session, mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# ONNX CONTROLLER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PITNNOnnxInference:
    """
    Stateful real-time PITNN controller using ONNX Runtime.
    Identical interface to PITNNInference (pitnn_inference.py).
    Maintains a rolling 20-step history buffer.

    Usage:
        ctrl = PITNNOnnxInference()
        phi1, phi2, phi3 = ctrl.step(V1=800, V2=798, iL=27.5, Pref=20000)
    """

    def __init__(self, model_path="pitnn_model.onnx",
                 mu_path="pitnn_mu.npy",
                 sigma_path="pitnn_sigma.npy"):
        self.session, self.mu, self.sigma = load_onnx_session(
            model_path, mu_path, sigma_path
        )
        self._input_name  = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        self._buffer      = np.zeros((SEQ_LEN, N_FEAT), dtype=np.float32)
        self._phi1_prev   = PI * 0.95   # initial phi1 estimate (nominal)
        self._phi2_prev   = PI * 0.95   # initial phi2 estimate (nominal)
        self._phi3_prev   = 0.22        # initial phi3 estimate

    def reset(self):
        """Clear history buffer."""
        self._buffer    = np.zeros((SEQ_LEN, N_FEAT), dtype=np.float32)
        self._phi1_prev = PI * 0.95
        self._phi2_prev = PI * 0.95
        self._phi3_prev = 0.22

    def _normalise(self, feat: np.ndarray) -> np.ndarray:
        return ((feat - self.mu) / self.sigma).astype(np.float32)

    def step(self, V1: float, V2: float, iL: float, Pref: float,
             phi_prev: tuple = None) -> tuple:
        """
        Run one inference step. Call once per switching cycle.

        Parameters
        ----------
        V1        : primary bus voltage (V)
        V2        : secondary bus voltage (V)
        iL        : inductor current (A)
        Pref      : power reference (W)
        phi_prev  : (phi1, phi2, phi3) previous outputs (rad); uses internal
                    state if None

        Returns
        -------
        (phi1, phi2, phi3) in radians — all three independently predicted
            phi1 ∈ [PHI12_MIN, PHI12_MAX]  — primary bridge inner duty
            phi2 ∈ [PHI12_MIN, PHI12_MAX]  — secondary bridge inner duty
            phi3 ∈ [PHI_MIN,   PHI3_MAX]   — external phase shift
        """
        if phi_prev is not None:
            phi1_p, phi2_p, phi3_p = phi_prev
        else:
            phi1_p = self._phi1_prev
            phi2_p = self._phi2_prev
            phi3_p = self._phi3_prev

        v_ratio = float(V1 * V2) / (V1_NOM * V2_NOM)
        feat = np.array([
            V1, V2, iL, phi1_p, phi2_p, phi3_p, Pref, v_ratio
        ], dtype=np.float32)

        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = self._normalise(feat)

        # ONNX Runtime inference — shape must be (1, 20, 8)
        x = self._buffer[np.newaxis]    # (1, 20, 8)
        result = self.session.run(
            [self._output_name],
            {self._input_name: x}
        )[0].squeeze()   # (3,)

        phi1 = float(result[0])
        phi2 = float(result[1])
        phi3 = float(result[2])
        self._phi1_prev = phi1
        self._phi2_prev = phi2
        self._phi3_prev = phi3
        return phi1, phi2, phi3

    def phi3_to_delay_us(self, phi3: float) -> float:
        return phi3 / (2.0 * PI * FSW) * 1e6

    def phi1_to_duty_pct(self, phi1: float) -> float:
        return (phi1 / PI) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM GUIDE
# ─────────────────────────────────────────────────────────────────────────────

PLATFORM_GUIDE = """
Platform-Specific Setup
═══════════════════════

NVIDIA Jetson (Orin / AGX / Nano):
    pip install onnxruntime-gpu
    # Or use TensorRT for maximum speed:
    # trtexec --onnx=pitnn_model.onnx --saveEngine=pitnn.trt --fp16
    # Then load pitnn.trt with the TensorRT Python API

Intel CPU / OpenVINO:
    pip install onnxruntime
    # Or convert to OpenVINO IR format:
    # mo --input_model pitnn_model.onnx --output_dir ir_model/

Raspberry Pi / ARM Cortex-A:
    pip install onnxruntime
    # Uses optimised NEON SIMD kernels automatically

STM32 / Cortex-M (bare metal):
    # Use ONNX Runtime for Microcontrollers (ORT Micro):
    # https://onnxruntime.ai/docs/build/inferencing.html#ort-mobile-and-ort-for-iot
    # Or use STM32Cube.AI to convert pitnn_model.onnx

MATLAB:
    # Import with MATLAB Deep Learning Toolbox:
    net = importONNXNetwork('pitnn_model.onnx', 'OutputLayerType', 'regression');
    phi_TPS = predict(net, input_sequence);

LabVIEW / NI Hardware:
    # Convert via MATLAB or use NI's AI inference support
    # See: ni.com/en/support/documentation/supplemental/21/using-onnx-models.html
"""


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK — compare ONNX vs fallback timing
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(ctrl, n_warmup=10, n_bench=100):
    """Measure inference latency statistics over n_bench calls."""
    ctrl.reset()
    V1, V2, iL, Pref = 800.0, 800.0, 27.5, 20000.0

    # Warm up
    for _ in range(n_warmup):
        ctrl.step(V1, V2, iL, Pref)

    # Benchmark
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        ctrl.step(V1, V2, iL, Pref)
        times.append((time.perf_counter() - t0) * 1e6)

    times = np.array(times)
    print(f"\nInference latency over {n_bench} calls:")
    print(f"  Mean   : {times.mean():.2f} µs")
    print(f"  Median : {np.median(times):.2f} µs")
    print(f"  Min    : {times.min():.2f} µs")
    print(f"  Max    : {times.max():.2f} µs")
    print(f"  Std    : {times.std():.2f} µs")
    print(f"  fsw=100kHz → Ts=10µs  →  {'FAST ENOUGH' if times.mean() < 10 else 'TOO SLOW for cycle-by-cycle — run every N cycles'}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 66)
    print("  PITNN Inference — ONNX Runtime Demo (pitnn_onnx_inference.py)")
    print("=" * 66)
    print(PLATFORM_GUIDE)

    try:
        ctrl = PITNNOnnxInference()
    except Exception as e:
        print(f"\nCould not load ONNX model: {e}")
        print("Run first:  python pitnn_deploy.py --mode export")
        return

    # Test conditions
    scenarios = [
        (800, 800, 10000, "10kW nominal"),
        (760, 840,  8000, "8kW V-variation"),
        (800, 800, 20000, "20kW"),
        (800, 800,  5000, "5kW light load"),
        (880, 720, 50000, "50kW high V"),
        (800, 800, 30000, "30kW mid load"),
        (840, 760, 40000, "40kW asymm V"),
        (720, 880, 15000, "15kW off-voltage"),
    ]

    print(f"\n{'Condition':<22} {'V1':>5} {'V2':>5} {'Pref':>7}  "
          f"{'φ3 (rad)':>9}  {'delay (µs)':>10}  {'duty%':>6}  {'t (µs)':>8}")
    print("-" * 84)

    for V1, V2, Pref, label in scenarios:
        ctrl.reset()
        iL_est = float(V1 * V2 / (V1_NOM * V2_NOM) * 10.7 * 0.3)

        for _ in range(5):   # warm up buffer
            ctrl.step(float(V1), float(V2), iL_est, float(Pref))

        t0 = time.perf_counter()
        phi1, phi2, phi3 = ctrl.step(float(V1), float(V2), iL_est, float(Pref))
        t_us = (time.perf_counter() - t0) * 1e6

        print(f"{label:<22} {V1:>5.0f} {V2:>5.0f} {Pref:>7.0f}  "
              f"{phi3:>9.4f}  {ctrl.phi3_to_delay_us(phi3):>10.3f}  "
              f"{ctrl.phi1_to_duty_pct(phi1):>5.1f}%  {t_us:>8.1f}")

    # Benchmark
    print()
    ctrl_bench = PITNNOnnxInference()
    benchmark(ctrl_bench)

    # Minimal integration example
    print("\n--- Integration Example ---")
    ctrl.reset()
    Pref = 20000.0
    for cycle in range(10):
        V1_meas = 800.0 + (cycle % 3) * 1.5
        V2_meas = 798.0 + (cycle % 2) * 1.0
        iL_meas = 27.5  + (cycle % 4) * 0.5

        phi1, phi2, phi3 = ctrl.step(V1_meas, V2_meas, iL_meas, Pref)
        delay_us = ctrl.phi3_to_delay_us(phi3)

        if cycle % 3 == 0:
            print(f"Cycle {cycle:3d}: φ3={phi3:.4f} rad  delay={delay_us:.3f}µs")


if __name__ == "__main__":
    run_demo()
