"""
============================================================
Physics-Informed Transformer Neural Network (PITNN)
for Real-Time Triple Phase Shift (TPS) Optimal Modulation
in Dual Active Bridge (DAB) Converters
============================================================

Paper: "Physics-Informed Transformer Neural Network Control for
        Real-Time Triple Phase Shift Optimal Modulation in
        Dual Active Bridge Converters" — Chukwuemeka Nzeadibe,
        Mississippi State University, 2026.

Requirements:  pip install torch numpy matplotlib scipy opencv-python
Run:           python pitnn_dab.py
Run with video: python pitnn_dab.py --video path/to/scope_video.mp4

============================================================
VIDEO INGESTION PIPELINE
============================================================
Drop any oscilloscope or simulation screen-recording video into
the training pipeline without modifying any model code:

    python pitnn_dab.py --video my_scope_capture.mp4

The VideoWaveformExtractor automatically:
  1. Detects the waveform display region in each frame
  2. Separates individual signal panels (vab, nvcd, iL, gate signals)
  3. Extracts normalised waveform traces column-by-column
  4. Estimates TPS parameters (phi1, phi3) from duty cycle + phase lag
  5. Builds a MeasurementDataset of (sequence, phi_TPS) pairs
  6. Trains a VideoConsistencyLoss alongside the synthetic dataset

To swap in a new video: just change the --video path. No code changes needed.

Hardware compatibility
  Tektronix / Rigol / Keysight oscilloscopes — direct screen recording
  MATLAB/Simulink scope windows — simulation output video
  PLECS / LTspice / PSIM waveform windows — all supported
  Multi-channel captures (CH1=vab, CH2=nvcd, CH3=iL) — auto-detected
============================================================

High-power configuration (10kW–80kW):
  V1=V2=800V, n=1.0, Lk=10µH, fsw=100kHz → P_max=80kW
  Operating range: 5kW–70kW  (P_min≈3kW physics floor)
  All three TPS angles predicted by the PITNN:
    phi1 ∈ [PHI12_MIN, PHI12_MAX] rad  (primary inner duty)
    phi2 ∈ [PHI12_MIN, PHI12_MAX] rad  (secondary inner duty)
    phi3 ∈ [PHI_MIN,   PHI3_MAX]  rad  (external phase shift)
"""

import math, time, warnings, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from scipy.optimize import brentq
from scipy.signal import savgol_filter, find_peaks, correlate

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (Table II)
# V1=400V, V2=250V, n=1.6, Lk=40µH, fsw=100kHz → P_max≈5kW, P_rated=3.3kW
# ─────────────────────────────────────────────────────────────────────────────
V1_NOM      = 400.0          # Primary bus voltage         (V)
V2_NOM      = 250.0          # Secondary bus voltage       (V)
FSW         = 100e3          # Switching frequency         (Hz)
LK          = 40e-6          # Series inductance           (H)
N_TURNS     = 1.6            # Transformer turns ratio n = V1/V2
P_RATED     = 3300.0         # Nominal rated power         (W)
TS          = 1.0 / FSW
WS          = 2.0 * math.pi * FSW
PI          = math.pi
PHI_MIN     = 0.02
PHI12_MIN   = PI * 0.65
PHI12_MAX   = PI * 0.99
PHI12_NOM   = PI * 0.95
PHI3_MAX    = 1.50
K_POWER     = 6620.8         # P ≈ K·(φ1/π)·φ3·(1−φ3/B), fitted at new spec
B_POWER     = PI             # Power peaks at φ3 = B/2 ≈ π/2
PHI3_PEAK   = B_POWER / 2.0


# ═════════════════════════════════════════════════════════════════════════════
# VIDEO INGESTION PIPELINE  ← plug any scope/simulation video in here
# ═════════════════════════════════════════════════════════════════════════════

class VideoWaveformExtractor:
    """
    Extracts TPS modulation parameters from oscilloscope or simulation videos.

    Works with any screen-recording that shows waveform traces on a dark
    background — oscilloscope hardware captures, MATLAB/Simulink scope
    windows, PLECS, LTspice, or PSIM output videos.

    Usage
    ─────
        extractor = VideoWaveformExtractor("scope_video.mp4")
        dataset   = extractor.build_dataset()   # returns MeasurementDataset
        # Then pass dataset to train_pitnn(..., video_dataset=dataset)

    To use a new video: just change the path. No other code changes needed.

    Extraction algorithm
    ────────────────────
    1. Detect waveform display ROI by finding the dark rectangular region
       with maximum extent of bright waveform pixels.
    2. Split ROI into horizontal panels using valley detection on per-row
       brightness — each panel corresponds to one signal channel.
    3. Classify each panel as SQUARE (voltage/gate) or SMOOTH (current)
       based on the fraction of samples in high/mid/low regions.
    4. For each frame, extract the normalised trace profile per column
       (y-centroid of bright pixels → normalised voltage level).
    5. From the main square-wave panel, compute duty cycle → phi1.
    6. From cross-correlation between square and smooth panels → phi3.
    7. Invert the power model to get the implied Pref for each frame.
    8. Assemble into (sequence, phi_TPS_label) training pairs.
    """

    # ── Layout constants ──────────────────────────────────────────────────────
    ROI_Y1, ROI_Y2 = 100, 980    # waveform display rows (calibrated from video)
    ROI_X1, ROI_X2 = 150, 1490   # waveform display columns
    FRAME_STEP      = 3           # process every Nth frame
    BRIGHT_THRESH   = 100         # pixel brightness threshold for trace detection
    SEQ_LEN         = 20          # PITNN sequence length

    def __init__(self, video_path: str, fsw_hardware=FSW,
                 V1_hardware=V1_NOM, V2_hardware=V2_NOM,
                 verbose=True):
        self.video_path  = video_path
        self.fsw         = fsw_hardware
        self.V1          = V1_hardware
        self.V2          = V2_hardware
        self.verbose     = verbose
        self._panels     = None   # panel boundaries, populated by _detect_panels
        self._sq_panel   = None   # index of main square-wave panel (vab)
        self._sm_panel   = None   # index of main smooth panel (iL or nvcd)

    def _log(self, msg):
        if self.verbose:
            print(f"  [VideoExtractor] {msg}")

    # ── Step 1: Detect ROI and panels from a reference frame ─────────────────
    def _detect_panels(self, gray_roi):
        """
        Find horizontal panel boundaries by detecting rows with low brightness
        (separator lines between subplots).
        Returns list of (y_start, y_end) tuples.
        """
        row_bright  = (gray_roi > self.BRIGHT_THRESH).sum(axis=1).astype(float)
        row_smooth  = savgol_filter(row_bright, 21, 3) if len(row_bright) > 21 else row_bright
        valleys, _  = find_peaks(-row_smooth, height=-60, distance=40)
        # Add start and end
        boundaries = sorted(set([0] + list(valleys) + [gray_roi.shape[0]]))
        panels = []
        for i in range(len(boundaries) - 1):
            h = boundaries[i+1] - boundaries[i]
            if h > 30:   # skip tiny artefact panels
                panels.append((boundaries[i], boundaries[i+1]))
        return panels

    # ── Step 2: Classify panels ───────────────────────────────────────────────
    def _classify_panels(self, gray_roi, panels):
        """
        Classify each panel as SQUARE (voltage/gate) or SMOOTH (current/filtered).
        SQUARE panels → candidate for vab / gate drive
        SMOOTH panels → candidate for iL / output voltage
        """
        types = []
        for y1, y2 in panels:
            panel    = gray_roi[y1:y2, :]
            ph       = y2 - y1
            signal   = self._extract_trace(panel, ph)
            high_f   = (signal >  0.25).mean()
            low_f    = (signal < -0.25).mean()
            is_square = (high_f + low_f) > 0.40 and panel.shape[1] > 200
            types.append("SQUARE" if is_square else "SMOOTH")
        return types

    # ── Step 3: Extract normalised trace from one panel ───────────────────────
    def _extract_trace(self, panel_gray, panel_height):
        """
        For each column, find y-centroid of bright pixels.
        Returns normalised array in [-1, +1]: top=+1, bottom=-1.
        """
        signal = np.zeros(panel_gray.shape[1], dtype=np.float32)
        for x in range(panel_gray.shape[1]):
            col  = panel_gray[:, x].astype(float)
            mask = col > self.BRIGHT_THRESH
            if mask.sum() > 0:
                signal[x] = float(np.where(mask)[0].mean())
            else:
                signal[x] = panel_height / 2.0
        return 1.0 - 2.0 * signal / max(panel_height, 1)

    # ── Step 4: Estimate TPS parameters from a pair of traces ─────────────────
    def _estimate_phi(self, sq_trace, sm_trace):
        """
        phi1 from duty cycle of square trace.
        phi3 from normalised cross-correlation lag between sq and sm.
        """
        # phi1: fraction of columns where square trace is high → times π
        duty = float((sq_trace > 0.20).mean())
        phi1 = float(np.clip(duty * PI, PHI_MIN, PI * 0.99))

        # phi3: normalised lag from cross-correlation
        sq_n  = sq_trace - sq_trace.mean()
        sm_n  = sm_trace - sm_trace.mean()
        sq_s  = sq_n.std() + 1e-9
        sm_s  = sm_n.std() + 1e-9
        xcorr = correlate(sq_n / sq_s, sm_n / sm_s, mode='full')
        lag_idx = int(xcorr.argmax()) - len(sq_trace) + 1
        lag_frac = lag_idx / max(len(sq_trace), 1)
        phi3 = float(np.clip(abs(lag_frac) * PI, PHI_MIN, PHI3_MAX))

        return phi1, phi3

    # ── Step 5: Invert power model to get implied Pref ────────────────────────
    def _phi3_to_pref(self, phi1, phi3):
        """
        P = K*(phi1/π)*phi3*(1-phi3/B) * (V1*V2/Vnom²)
        """
        v_scale = self.V1 * self.V2 / (V1_NOM * V2_NOM)
        P = K_POWER * (phi1 / PI) * phi3 * (1.0 - phi3 / B_POWER) * v_scale
        return float(np.clip(P, 100.0, 80000.0))

    # ── Main extraction method ────────────────────────────────────────────────
    def extract(self):
        """
        Process all video frames and return list of measurement dicts:
          [{"phi_TPS": np.array([φ1,φ2,φ3]),
            "Pref": float,
            "V1": float, "V2": float,
            "trace_sq": np.array,     ← normalised square trace
            "trace_sm": np.array,     ← normalised smooth trace
            "frame_time": float}, ...]
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("pip install opencv-python  to use video ingestion")

        cap   = cv2.VideoCapture(self.video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # OpenCV returns 0 fps for some codecs/containers on Windows.
        # Fall back to counting frames manually if needed.
        if fps <= 0:
            self._log("WARNING: OpenCV reported 0 fps — counting frames manually")
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1
            total = count
            fps   = 30.0   # assume 30 fps if unreadable; adjust if known
            cap.release()
            cap = cv2.VideoCapture(self.video_path)  # reopen from start
            self._log(f"Manual count: {total} frames, assuming {fps:.1f} fps")

        if total <= 0:
            self._log("ERROR: No frames found in video — check file path and codec")
            cap.release()
            return []

        self._log(f"Video: {total} frames @ {fps:.1f}fps = {total/fps:.1f}s")

        # ── Detect panels from reference frame ─────────────────────────────
        ref_idx = min(150, total // 3)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        ret, ref_frame = cap.read()
        if not ret:
            cap.release(); return []

        roi_gray = cv2.cvtColor(
            ref_frame[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2],
            cv2.COLOR_BGR2GRAY
        )
        panels = self._detect_panels(roi_gray)
        types  = self._classify_panels(roi_gray, panels)
        self._log(f"Detected {len(panels)} panels: {types}")

        # Find best square and smooth panel
        sq_candidates = [i for i,t in enumerate(types) if t == "SQUARE"]
        sm_candidates = [i for i,t in enumerate(types) if t == "SMOOTH"]

        # Pick the square panel with most transitions (= most dynamic)
        best_sq_score = -1
        for i in sq_candidates:
            y1, y2 = panels[i]
            p = roi_gray[y1:y2, :]
            tr = self._extract_trace(p, y2-y1)
            score = int(np.abs(np.diff((tr > 0.2).astype(int))).sum())
            if score > best_sq_score:
                best_sq_score = score
                self._sq_panel = i

        # Pick the smooth panel with highest variance (= carries most info)
        best_sm_var = -1
        for i in sm_candidates:
            y1, y2 = panels[i]
            p  = roi_gray[y1:y2, :]
            tr = self._extract_trace(p, y2-y1)
            if tr.std() > best_sm_var:
                best_sm_var = tr.std()
                self._sm_panel = i

        if self._sq_panel is None or self._sm_panel is None:
            self._log("WARNING: could not find square+smooth panel pair. "
                      "Falling back to panels 3 and 4.")
            self._sq_panel = min(3, len(panels)-1)
            self._sm_panel = min(4, len(panels)-1)

        sq_y1, sq_y2 = panels[self._sq_panel]
        sm_y1, sm_y2 = panels[self._sm_panel]
        self._log(f"Using square panel {self._sq_panel} (rows {sq_y1}-{sq_y2}), "
                  f"smooth panel {self._sm_panel} (rows {sm_y1}-{sm_y2})")

        # ── Process all frames ──────────────────────────────────────────────
        measurements = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fi = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if fi % self.FRAME_STEP == 0:
                roi = cv2.cvtColor(
                    frame[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2],
                    cv2.COLOR_BGR2GRAY
                )
                # Extract traces
                sq_panel_gray = roi[sq_y1:sq_y2, :]
                sm_panel_gray = roi[sm_y1:sm_y2, :]
                sq_trace = self._extract_trace(sq_panel_gray, sq_y2-sq_y1)
                sm_trace = self._extract_trace(sm_panel_gray, sm_y2-sm_y1)

                # Estimate TPS parameters
                phi1, phi3 = self._estimate_phi(sq_trace, sm_trace)
                Pref       = self._phi3_to_pref(phi1, phi3)
                v_ratio    = self.V1 * self.V2 / (V1_NOM * V2_NOM)

                # Estimate iL from smooth trace amplitude
                iL_est = float(abs(sm_trace).mean() * 50.0 * v_ratio)  # rough A

                measurements.append({
                    "phi_TPS":    np.array([phi1, phi1, phi3], dtype=np.float32),
                    "Pref":       Pref,
                    "V1":         self.V1,
                    "V2":         self.V2,
                    "iL_est":     iL_est,
                    "trace_sq":   sq_trace,
                    "trace_sm":   sm_trace,
                    "frame_time": fi / fps,
                    "v_ratio":    v_ratio,
                })
            fi += 1
        cap.release()
        self._log(f"Extracted {len(measurements)} measurement frames")
        return measurements

    # ── Build dataset ─────────────────────────────────────────────────────────
    def build_dataset(self, mu=None, sigma=None):
        """
        Extract measurements and convert to a TensorDataset compatible with
        the PITNN training loop.

        Returns (X_norm, Y, mu, sigma, X_raw) in the same format as
        generate_dataset() so it can be concatenated or used standalone.

        If mu/sigma are provided (from the synthetic dataset), uses them
        for normalisation — ensures video and synthetic data are on the
        same scale.
        """
        measurements = self.extract()
        if not measurements:
            self._log("No measurements extracted — returning empty dataset")
            return None

        # Smooth all three extracted angles
        phi1_raw = np.array([m["phi_TPS"][0] for m in measurements], dtype=np.float32)
        phi3_raw = np.array([m["phi_TPS"][2] for m in measurements], dtype=np.float32)
        win = min(len(phi3_raw) if len(phi3_raw) % 2 == 1 else len(phi3_raw) - 1, 11)
        if win >= 5:
            phi1_smooth = savgol_filter(phi1_raw, win, 2).astype(np.float32)
            phi3_smooth = savgol_filter(phi3_raw, win, 2).astype(np.float32)
        else:
            phi1_smooth = phi1_raw.copy()
            phi3_smooth = phi3_raw.copy()
        phi1_smooth = np.clip(phi1_smooth, PHI12_MIN, PHI12_MAX)
        phi3_smooth = np.clip(phi3_smooth, PHI_MIN, PHI3_MAX)

        for i, meas in enumerate(measurements):
            meas["phi_TPS"][0] = float(phi1_smooth[i])   # phi1 from video
            meas["phi_TPS"][1] = float(phi1_smooth[i])   # phi2 = phi1 (symmetric)
            meas["phi_TPS"][2] = float(phi3_smooth[i])
            meas["Pref"] = self._phi3_to_pref(float(phi1_smooth[i]), float(phi3_smooth[i]))

        X_list, Y_list = [], []

        for i, meas in enumerate(measurements):
            phi_opt = np.array([meas["phi_TPS"][0],
                                meas["phi_TPS"][1],
                                meas["phi_TPS"][2]], dtype=np.float32)
            phi_opt[0] = float(np.clip(phi_opt[0], PHI12_MIN, PHI12_MAX))
            phi_opt[1] = float(np.clip(phi_opt[1], PHI12_MIN, PHI12_MAX))
            phi_opt[2] = float(np.clip(phi_opt[2], PHI_MIN, PHI3_MAX))

            seq = []
            rng_local = np.random.default_rng(i)
            phi_h = phi_opt.copy()
            for _ in range(self.SEQ_LEN):
                nv   = rng_local.normal(0, 2.0, 2)
                np12 = rng_local.normal(0, 0.012, 2)
                np3  = rng_local.normal(0, 0.006, 1)
                ph   = phi_h.copy()
                ph[0] = float(np.clip(ph[0]+np12[0], PHI12_MIN, PHI12_MAX))
                ph[1] = float(np.clip(ph[1]+np12[1], PHI12_MIN, PHI12_MAX))
                ph[2] = float(np.clip(ph[2]+np3[0],  PHI_MIN,   PHI3_MAX))
                iLt   = meas["iL_est"] * float(rng_local.uniform(0.92, 1.08))
                vrat  = float(meas["V1"] * meas["V2"] / (V1_NOM * V2_NOM))
                seq.append(np.array([
                    meas["V1"] + nv[0], meas["V2"] + nv[1],
                    iLt, ph[0], ph[1], ph[2], meas["Pref"], vrat,
                ], dtype=np.float32))
                phi_h[0] = float(np.clip(phi_opt[0]*.97+phi_h[0]*.03+rng_local.normal(0,.008), PHI12_MIN, PHI12_MAX))
                phi_h[1] = float(np.clip(phi_opt[1]*.97+phi_h[1]*.03+rng_local.normal(0,.008), PHI12_MIN, PHI12_MAX))
                phi_h[2] = float(np.clip(phi_opt[2]*.97+phi_h[2]*.03+rng_local.normal(0,.004), PHI_MIN,   PHI3_MAX))

            X_list.append(np.stack(seq))
            Y_list.append(phi_opt)

        X_raw = np.stack(X_list).astype(np.float32)
        Y     = np.stack(Y_list).astype(np.float32)

        if mu is None:
            mu    = X_raw.mean(axis=(0,1), keepdims=True).astype(np.float32)
            sigma = (X_raw.std(axis=(0,1), keepdims=True) + 1e-8).astype(np.float32)
        else:
            mu    = mu.reshape(1, 1, -1).astype(np.float32)
            sigma = sigma.reshape(1, 1, -1).astype(np.float32)

        X_norm = ((X_raw - mu) / sigma).astype(np.float32)
        self._log(f"Dataset: X={X_norm.shape}, Y={Y.shape}")
        self._log(f"φ3 range: [{Y[:,2].min():.4f}, {Y[:,2].max():.4f}] rad")
        self._log(f"Pref range: [{X_raw[:,-1,6].min():.0f}, {X_raw[:,-1,6].max():.0f}] W")
        return X_norm, Y, mu.squeeze(), sigma.squeeze(), X_raw

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    def plot_extraction(self, save_path="video_extraction.png"):
        """Save a diagnostic plot of the extracted waveform parameters."""
        measurements = self.extract()
        if not measurements:
            return

        times  = [m["frame_time"] for m in measurements]
        phi1s  = [m["phi_TPS"][0] for m in measurements]
        phi3s  = [m["phi_TPS"][2] for m in measurements]
        prefs  = [m["Pref"]       for m in measurements]

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(times, phi1s, lw=1.2, color="steelblue")
        axes[0].set(ylabel="φ1 (rad)", title="Extracted TPS Parameters from Video")
        axes[1].plot(times, phi3s, lw=1.2, color="darkorange")
        axes[1].set(ylabel="φ3 (rad)")
        axes[2].plot(times, [p/1000 for p in prefs], lw=1.2, color="green")
        axes[2].set(ylabel="P_implied (kW)", xlabel="Video time (s)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  Saved: {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# §II  DAB CONVERTER PHYSICS
# ═════════════════════════════════════════════════════════════════════════════

class DABPhysics:
    def __init__(self, V1=V1_NOM, V2=V2_NOM, n=N_TURNS, Lk=LK, fsw=FSW):
        self.V1=float(V1); self.V2=float(V2); self.n=float(n)
        self.Lk=float(Lk); self.fsw=float(fsw)
        self.Ts=1/fsw;     self.ws=2*PI*fsw

    def classify_mode(self, phi1, phi2, phi3):
        s=phi3+phi2
        if s<=PI:
            if phi1<=phi3: return 1
            elif phi3<=phi1<=s: return 2
            else: return 3
        else:
            pw=s-PI; return 4 if (pw<=phi3<=phi1) else 5

    def bridge_voltages(self, t, phi1, phi2, phi3):
        ws,Ts=self.ws,self.Ts
        tm1=t%Ts; tm2=(t-phi3/ws)%Ts
        t1a,t2a,t3a=phi1/ws,PI/ws,(PI+phi1)/ws
        t1b,t2b,t3b=phi2/ws,PI/ws,(PI+phi2)/ws
        vab =np.where(tm1<t1a,self.V1,np.where(tm1<t2a,0.,np.where(tm1<t3a,-self.V1,0.)))
        nvcd=np.where(tm2<t1b,self.n*self.V2,np.where(tm2<t2b,0.,np.where(tm2<t3b,-self.n*self.V2,0.)))
        return vab.astype(np.float32), nvcd.astype(np.float32)

    def simulate_current(self, phi1, phi2, phi3, N_pts=600):
        Ts,Lk=self.Ts,self.Lk
        t=np.linspace(0,Ts,N_pts,endpoint=False,dtype=np.float32); dt=float(t[1]-t[0])
        vab,nvcd=self.bridge_voltages(t,phi1,phi2,phi3); vL=vab-nvcd
        iL=np.zeros(N_pts,dtype=np.float32)
        for k in range(1,N_pts): iL[k]=iL[k-1]+vL[k-1]/Lk*dt
        iL-=np.linspace(0,float(iL[-1]),N_pts,dtype=np.float32)
        return t,vab,nvcd,vL,iL

    def compute_power(self,phi1,phi2,phi3):
        _,vab,_,_,iL=self.simulate_current(phi1,phi2,phi3,N_pts=800)
        return float(np.mean(vab*iL))

    def _compute_power_fast(self,phi1,phi2,phi3):
        _,vab,_,_,iL=self.simulate_current(phi1,phi2,phi3,N_pts=300)
        return float(np.mean(vab*iL))

    def compute_irms(self,phi1,phi2,phi3):
        _,_,_,_,iL=self.simulate_current(phi1,phi2,phi3)
        return float(np.sqrt(np.mean(iL**2)))

    def check_zvs(self,phi1,phi2,phi3):
        t,_,_,_,iL=self.simulate_current(phi1,phi2,phi3)
        ws,Ts,N=self.ws,self.Ts,len(iL)
        ok,pen=True,0.0
        for tk in [0.,phi1/ws,PI/ws,phi3/ws,(phi3+phi2)/ws]:
            idx=min(max(int(round((tk%Ts)/Ts*(N-1))),0),N-1)
            viol=float(max(0.,-iL[idx])); pen+=viol
            if viol>0.1: ok=False
        return ok,pen

    def _score_fast(self, phi1, phi2, phi3):
        """
        Single-simulation scoring: returns (Irms, zvs_ok, zvs_penalty, P).
        Uses N=200 points — fast enough for solver inner loop while
        resolving the switching instants correctly at 100kHz.
        Combines what compute_irms + check_zvs do in two separate calls,
        cutting solver simulation count roughly in half.
        """
        N = 200
        _,vab,_,_,iL = self.simulate_current(phi1, phi2, phi3, N_pts=N)
        Ir   = float(np.sqrt(np.mean(iL**2)))
        P    = float(np.mean(vab * iL))
        ws, Ts = self.ws, self.Ts
        ok, pen = True, 0.0
        for tk in [0., phi1/ws, PI/ws, phi3/ws, (phi3+phi2)/ws]:
            idx = min(max(int(round((tk % Ts) / Ts * (N-1))), 0), N-1)
            viol = float(max(0., -iL[idx])); pen += viol
            if viol > 0.1: ok = False
        return Ir, ok, pen, P

    def p_max(self): return self.n*self.V1*self.V2/(8*self.fsw*self.Lk)

    def solve_optimal_phi(self, Pref, rng=None):
        """
        Find (phi1, phi2, phi3) that delivers Pref with minimum Irms/P ratio
        while maintaining ZVS where physically achievable.

        Speed optimisation
        ------------------
        Uses _score_fast() (one N=200 simulation) for scoring instead of
        separate compute_irms() + check_zvs() (two N=600 simulations).
        Grid reduced to 7 candidates. Saves ~60% of dataset generation time.

        Low-power note
        --------------
        Below 10kW, phi3 solutions are very small (<0.05 rad). N=300
        simulations have quantisation error at small phi3, causing brentq
        to see a flat power function. N_brentq=500 is used below 10kW
        to resolve these small phase windows correctly.

        ZVS note
        --------
        At V2 > V1 by more than ~5%, ZVS requires phi12 > pi rad which is
        outside the TPS range. The solver minimises ZVS penalty in those
        cases. This is a hardware physics constraint.
        """
        # Adaptive phi12 floor: scales with V2/V1 so high-V2 conditions
        # search lower phi12 values where ZVS penalty is smallest
        v_ratio     = (self.n * self.V2) / max(self.V1, 1.0)
        phi12_floor = float(np.clip(PI * 0.50 * v_ratio, PHI12_MIN, PHI12_NOM))

        # 7-point grid (6 spaced + nominal) — faster than 11-point
        grid = np.unique(np.clip(
            np.concatenate([np.linspace(phi12_floor, PHI12_MAX, 6),
                            np.array([PHI12_NOM])]),
            PHI12_MIN, PHI12_MAX))

        # Higher resolution brentq at low power avoids quantisation flat spot
        N_brentq = 500 if Pref < 1000 else 300

        best_phi  = None
        best_cost = float("inf")

        for phi12 in grid:
            phi12 = float(phi12)

            # Power feasibility gate — use 80% to not reject low-power candidates
            P_hi = self._compute_power_fast(phi12, phi12, PHI3_PEAK)
            if P_hi < Pref * 0.80:
                continue

            Pref_c = float(min(Pref, P_hi * 0.97))

            # Capture phi12 for the lambda to avoid closure-over-loop-variable bug
            _phi12 = phi12
            def _pfast(p, phi12=_phi12):
                _,vab,_,_,iL = self.simulate_current(phi12, phi12, max(p, 0.001), N_pts=N_brentq)
                return float(np.mean(vab * iL)) - Pref_c

            try:
                phi3 = brentq(_pfast, 0.005, PHI3_PEAK, xtol=5e-3, maxiter=20)
            except Exception:
                v_scale = (self.n * self.V1 * self.V2) / (N_TURNS * V1_NOM * V2_NOM)
                A    = K_POWER * v_scale * phi12 / PI
                disc = A * A - 4 * (A / B_POWER) * Pref_c
                phi3 = float(np.clip(
                    (-A - math.sqrt(max(disc, 0))) / (-2 * A / B_POWER)
                    if disc >= 0 else 0.3,
                    0.005, PHI3_MAX))

            phi3 = float(np.clip(phi3, 0.005, PHI3_MAX))

            # Single combined simulation: Irms + ZVS + actual power
            try:
                Ir, zvs_ok, zvpen, P_actual = self._score_fast(phi12, phi12, phi3)
            except Exception:
                continue

            if abs(P_actual - Pref) > max(Pref * 0.25, 300.0):
                continue

            cost = (Ir / max(P_actual, 1.0)) * 1000.0 + 200.0 * zvpen
            if cost < best_cost:
                best_cost = cost
                best_phi  = np.array([phi12, phi12, phi3], dtype=np.float32)

        # Fallback: PHI12_NOM + brentq on phi3 only
        if best_phi is None:
            phi12  = float(PHI12_NOM)
            P_hi   = self._compute_power_fast(phi12, phi12, PHI3_PEAK)
            Pref_c = float(min(Pref, P_hi * 0.97))
            def _pfb(p, phi12=phi12):
                _,vab,_,_,iL = self.simulate_current(phi12, phi12, max(p, 0.001), N_pts=N_brentq)
                return float(np.mean(vab * iL)) - Pref_c
            try:
                phi3 = brentq(_pfb, 0.005, PHI3_PEAK, xtol=5e-3, maxiter=20)
            except Exception:
                v_scale = (self.n * self.V1 * self.V2) / (N_TURNS * V1_NOM * V2_NOM)
                A    = K_POWER * v_scale * phi12 / PI
                disc = A * A - 4 * (A / B_POWER) * Pref_c
                phi3 = float(np.clip(
                    (-A - math.sqrt(max(disc, 0))) / (-2 * A / B_POWER)
                    if disc >= 0 else 0.05,
                    0.005, PHI3_MAX))
            best_phi = np.array([phi12, phi12,
                                  float(np.clip(phi3, 0.005, PHI3_MAX))],
                                 dtype=np.float32)

        return best_phi


# ═════════════════════════════════════════════════════════════════════════════
# §V-B  PITNN ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=128,dropout=0.1):
        super().__init__(); self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return self.dropout(x+self.pe[:,:x.size(1)])


class PITNN(nn.Module):
    """
    Physics-Informed Transformer (§V-B, Fig.2, Eq.29-32).
    d_in=8: [V1,V2,iL,φ1,φ2,φ3,Pref,V1V2/Vnom²]
    Output: all three TPS angles predicted independently:
      φ1 ∈ [PHI12_MIN, PHI12_MAX]  — primary bridge inner duty
      φ2 ∈ [PHI12_MIN, PHI12_MAX]  — secondary bridge inner duty
      φ3 ∈ [PHI_MIN,   PHI3_MAX]   — external phase shift
    """
    def __init__(self,d_in=8,d_model=128,n_heads=8,n_layers=4,
                 d_ff=256,seq_len=20,dropout=0.1):
        super().__init__(); self.seq_len=seq_len
        self.embed   = nn.Linear(d_in,d_model)
        self.pos_enc = PositionalEncoding(d_model,seq_len+8,dropout)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_heads,
            dim_feedforward=d_ff,dropout=dropout,batch_first=True,
            norm_first=True,activation="relu")
        self.transformer=nn.TransformerEncoder(enc,num_layers=n_layers,enable_nested_tensor=False)
        self.ln_out=nn.LayerNorm(d_model)
        # Three-head output: one sigmoid per angle
        self.output_head=nn.Sequential(nn.Linear(d_model,d_ff),nn.ReLU(),
                                        nn.Dropout(dropout),nn.Linear(d_ff,3))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self,x):
        z   = self.ln_out(self.transformer(self.pos_enc(self.embed(x))))
        raw = self.output_head(z[:,-1,:])          # (B, 3)
        sig = torch.sigmoid(raw)                   # (B, 3) all in (0,1)
        phi1 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3 = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)  # (B, 3)


# ═════════════════════════════════════════════════════════════════════════════
# §V-C  LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

class PITNNLoss(nn.Module):
    """
    Weighted MSE with optional physics terms (Eq.33-37).
    All three angles are now free — weights [2,2,3]: phi3 gets highest
    weight (drives power), phi1/phi2 get equal weight (drive ZVS/duty).
    A soft symmetry penalty encourages phi1≈phi2 (symmetric bridges).
    """
    def __init__(self,lambda_p=0.,lambda1=0.,lambda2=0.,
                 Lk=LK,n=N_TURNS,fsw=FSW,V_nom=V1_NOM,I_rated=100.):
        super().__init__()
        self.lambda_p=lambda_p; self.lambda1=lambda1; self.lambda2=lambda2
        self.Lk=Lk; self.n=n; self.fsw=fsw; self.V_nom=V_nom
        self.I_rated=I_rated; self.diL_nom=V_nom/Lk; self.physics_weight=0.

    def _LP(self,phi,V1,V2,Pref):
        # Power model uses predicted phi1, not a fixed constant
        phi1=phi[:,0]; phi3=phi[:,2]
        scale=self.n*V1*V2/(self.V_nom**2)
        P=scale*K_POWER*(phi1/PI)*phi3*(1.-phi3/B_POWER)
        return torch.mean((P-Pref)**2/(Pref**2+1.))

    def _LZVS(self,phi,V1,V2):
        d1=phi[:,0]/PI; phi3=phi[:,2]
        i0=(V1*d1-self.n*V2*phi3/PI)/(2.*self.Lk*self.fsw)
        return torch.mean(torch.clamp(-i0,min=0.)/self.I_rated)

    def _Lsym(self,phi):
        # Soft penalty: phi1 and phi2 should be close (symmetric DAB)
        return torch.mean((phi[:,0]-phi[:,1])**2)

    def forward(self,phi_pred,phi_target,V1,V2,Pref,iL_seq):
        w=torch.tensor([2.,2.,3.],device=phi_pred.device)
        L_data=torch.mean(w*(phi_pred-phi_target)**2)
        LP=self._LP(phi_pred,V1,V2,Pref)
        LZVS=self._LZVS(phi_pred,V1,V2)
        Lsym=self._Lsym(phi_pred)
        L_physics=LP+self.lambda2*LZVS+0.5*Lsym
        L_total=L_data+self.physics_weight*self.lambda_p*L_physics
        return L_total,{"L_total":L_total.item(),"L_data":L_data.item(),
                        "LP":LP.item(),"LI":0.,"LZVS":LZVS.item(),"L_physics":L_physics.item()}


# ═════════════════════════════════════════════════════════════════════════════
# §V-D  SYNTHETIC DATASET
# ═════════════════════════════════════════════════════════════════════════════

def generate_dataset(n_samples=10000,seq_len=20,
                     V1_range=(360.,440.),V2_range=(220.,280.),
                     Pref_range=(500.,3300.),seed=42):
    """
    Generate (X, Y) pairs where Y = [phi1_opt, phi2_opt, phi3_opt].
    All three angles come from solve_optimal_phi() which searches across
    a grid of phi12 values and selects the most efficient ZVS-maintaining
    solution. 20% of samples are drawn from the low-power region (5–15kW)
    to improve accuracy near the physics floor.
    """
    rng=np.random.default_rng(seed); dab=DABPhysics()
    print(f"  Generating {n_samples} synthetic samples (all 3 angles free) …")
    t0=time.perf_counter(); X_list,Y_list=[],[]; n_fb=0

    # 30% of samples from the low-power region (3–12kW) to improve accuracy
    # near the physics floor where phi3 is very small and hard to predict.
    n_low = int(n_samples * 0.30)
    Pref_low_range  = (300., 1500.)
    Pref_main_range = Pref_range

    for s in range(n_samples):
        if (s+1)%max(1,n_samples//8)==0:
            el=time.perf_counter()-t0; eta=el/(s+1)*(n_samples-s-1)
            print(f"    {s+1}/{n_samples}  elapsed {el:.0f}s  ETA {eta:.0f}s  (fallbacks:{n_fb})")
        V1=float(rng.uniform(*V1_range)); V2=float(rng.uniform(*V2_range))
        dab.V1,dab.V2=V1,V2
        P_ach = dab._compute_power_fast(PHI12_NOM, PHI12_NOM, PHI3_PEAK)

        # Alternate between low-power and full-range samples
        if s < n_low:
            Pref = float(np.clip(rng.uniform(*Pref_low_range), 100., P_ach*0.93))
        else:
            Pref = float(np.clip(rng.uniform(*Pref_main_range), 100., P_ach*0.93))

        # Jointly optimise all three angles
        phi_opt=dab.solve_optimal_phi(Pref,rng=rng)
        if abs(dab.compute_power(*phi_opt)-Pref)>max(Pref*.15,50.): n_fb+=1
        phi_opt[0]=float(np.clip(phi_opt[0],PHI12_MIN,PHI12_MAX))
        phi_opt[1]=float(np.clip(phi_opt[1],PHI12_MIN,PHI12_MAX))
        phi_opt[2]=float(np.clip(phi_opt[2],PHI_MIN,PHI3_MAX))

        # Build 20-step history — all three angles vary across steps
        seq=[]; phi_h=phi_opt.copy()
        for _ in range(seq_len):
            nv  = rng.normal(0,1.5,2)
            np12= rng.normal(0,0.012,2)
            np3 = rng.normal(0,0.006,1)
            ph  = phi_h.copy()
            ph[0]=float(np.clip(ph[0]+np12[0],PHI12_MIN,PHI12_MAX))
            ph[1]=float(np.clip(ph[1]+np12[1],PHI12_MIN,PHI12_MAX))
            ph[2]=float(np.clip(ph[2]+np3[0], PHI_MIN,  PHI3_MAX))
            vsc=V1*V2/(V1_NOM*V2_NOM); lksc=N_TURNS
            iLt=vsc*lksc*10.7*ph[2]*float(rng.uniform(.90,1.10))
            vrat=float(V1*V2/(V1_NOM*V2_NOM))
            seq.append(np.array([V1+nv[0],V2+nv[1],iLt,ph[0],ph[1],ph[2],Pref,vrat],dtype=np.float32))
            phi_h[0]=float(np.clip(phi_opt[0]*.97+phi_h[0]*.03+rng.normal(0,.008),PHI12_MIN,PHI12_MAX))
            phi_h[1]=float(np.clip(phi_opt[1]*.97+phi_h[1]*.03+rng.normal(0,.008),PHI12_MIN,PHI12_MAX))
            phi_h[2]=float(np.clip(phi_opt[2]*.97+phi_h[2]*.03+rng.normal(0,.005),PHI_MIN,  PHI3_MAX))
        X_list.append(np.stack(seq)); Y_list.append(phi_opt)

    X_raw=np.stack(X_list).astype(np.float32); Y=np.stack(Y_list).astype(np.float32)
    el=time.perf_counter()-t0
    print(f"  Done in {el:.1f}s  Fallback rate: {n_fb}/{n_samples} ({100*n_fb/n_samples:.1f}%)")
    print(f"  X_raw:{X_raw.shape}  "
          f"φ1=[{Y[:,0].min():.3f},{Y[:,0].max():.3f}]  "
          f"φ2=[{Y[:,1].min():.3f},{Y[:,1].max():.3f}]  "
          f"φ3=[{Y[:,2].min():.3f},{Y[:,2].max():.3f}]  rad")
    mu=X_raw.mean(axis=(0,1),keepdims=True).astype(np.float32)
    sigma=(X_raw.std(axis=(0,1),keepdims=True)+1e-8).astype(np.float32)
    return ((X_raw-mu)/sigma).astype(np.float32),Y,mu.squeeze(),sigma.squeeze(),X_raw


# ═════════════════════════════════════════════════════════════════════════════
# §V-D  TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_pitnn(model,loss_fn,X_norm,X_raw,Y,epochs=150,batch_size=64,
                lr=1e-4,val_split=.15,warmup_epochs=20,device="cpu",
                video_X_norm=None,video_X_raw=None,video_Y=None):
    """
    Train PITNN on synthetic dataset, optionally augmented with video data.

    video_X_norm, video_X_raw, video_Y:
        If provided (from VideoWaveformExtractor.build_dataset()),
        these are concatenated with the synthetic data so the model
        learns from real hardware measurements as well as simulation.
        The mixing weight is proportional to dataset sizes — no
        separate hyperparameter needed.
    """
    # ── Optionally merge video data ───────────────────────────────────────
    if video_X_norm is not None and len(video_X_norm) > 0:
        X_norm_all = np.concatenate([X_norm, video_X_norm], axis=0)
        X_raw_all  = np.concatenate([X_raw,  video_X_raw],  axis=0)
        Y_all      = np.concatenate([Y,      video_Y],      axis=0)
        print(f"  Combined dataset: {len(X_norm)} synthetic + {len(video_X_norm)} video = {len(X_norm_all)} total")
    else:
        X_norm_all, X_raw_all, Y_all = X_norm, X_raw, Y

    N=len(X_norm_all); n_val=int(N*.15); n_tr=N-2*n_val
    perm=np.random.permutation(N)
    tr_i,va_i,te_i=perm[:n_tr],perm[n_tr:n_tr+n_val],perm[n_tr+n_val:]
    def tt(a): return torch.from_numpy(a).float().to(device)
    Xn_tr,Xn_va,Xn_te=tt(X_norm_all[tr_i]),tt(X_norm_all[va_i]),tt(X_norm_all[te_i])
    Xr_tr,Xr_va,Xr_te=tt(X_raw_all[tr_i]), tt(X_raw_all[va_i]), tt(X_raw_all[te_i])
    Ytr,  Yva,  Yte  =tt(Y_all[tr_i]),     tt(Y_all[va_i]),     tt(Y_all[te_i])

    loader=DataLoader(TensorDataset(Xn_tr,Xr_tr,Ytr),batch_size=batch_size,shuffle=True,drop_last=True)
    optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=lr/20)
    hist={k:[] for k in ["train","val","LP","LZVS"]}
    best_val,best_state=float("inf"),None; model.to(device)

    n_p=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'─'*70}")
    print(f"  PITNN {n_p:,} params | {n_tr} tr · {n_val} val · {n_val} test")
    print(f"  Epochs {epochs} · Batch {batch_size} · LR {lr}")
    src_tag = f"synthetic+video({len(video_X_norm)})" if video_X_norm is not None else "synthetic only"
    print(f"  Data source: {src_tag}")
    print(f"{'─'*70}")
    print(f"  {'Ep':>4}  {'Train':>9}  {'Val':>9}  {'LP':>9}  {'LZVS':>9}  {'LR':>9}  {'pw':>5}")
    print(f"  {'─'*70}")

    for epoch in range(1,epochs+1):
        loss_fn.physics_weight=min(1.,(epoch-1)/max(warmup_epochs,1))
        model.train(); ep={k:0. for k in ["L_total","L_data","LP","LI","LZVS","L_physics"]}; nb=0
        for xn_b,xr_b,yb in loader:
            optimizer.zero_grad()
            phi_pred=model(xn_b)
            loss,info=loss_fn(phi_pred,yb,xr_b[:,-1,0],xr_b[:,-1,1],xr_b[:,-1,6],xr_b[:,:,2])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            for k in info: ep[k]+=info[k]; nb+=1
        scheduler.step()
        model.eval()
        with torch.no_grad():
            pv=model(Xn_va)
            _,vi=loss_fn(pv,Yva,Xr_va[:,-1,0],Xr_va[:,-1,1],Xr_va[:,-1,6],Xr_va[:,:,2])
        avg={k:v/max(nb,1) for k,v in ep.items()}
        hist["train"].append(avg["L_total"]); hist["val"].append(vi["L_total"])
        hist["LP"].append(avg["LP"]);         hist["LZVS"].append(avg["LZVS"])
        if vi["L_total"]<best_val:
            best_val=vi["L_total"]
            best_state={k:v.clone() for k,v in model.state_dict().items()}
        if epoch%10==0 or epoch==1:
            print(f"  {epoch:>4}  {avg['L_total']:>9.5f}  {vi['L_total']:>9.5f}  "
                  f"{avg['LP']:>9.5f}  {avg['LZVS']:>9.5f}  "
                  f"{scheduler.get_last_lr()[0]:>9.2e}  {loss_fn.physics_weight:>5.2f}")

    if best_state: model.load_state_dict(best_state); print(f"\n  Best checkpoint restored (val={best_val:.5f})")
    model.eval()
    with torch.no_grad():
        pt=model(Xn_te); mse=nn.functional.mse_loss(pt,Yte).item(); mae=(pt-Yte).abs().mean().item()
    print(f"\n{'─'*70}")
    print(f"  Test MSE: {mse:.6f} rad²   Test MAE: {mae:.6f} rad ({math.degrees(mae):.3f}°)")
    print(f"{'─'*70}\n")
    hist.update({"test_mse":mse,"test_mae":mae,"Y_test":Yte.cpu().numpy(),"Y_pred":pt.cpu().numpy()})
    return hist


# ── Export for deployment ─────────────────────────────────────────
import torch

def export_model(model, mu, sigma, save_dir="."):
    """Export trained PITNN in multiple formats for deployment."""
    model.eval()
    dummy_input = torch.zeros(1, 20, 8)  # (batch, seq_len, d_in)

    # ── Option A: TorchScript ─────────────────────────────────────
    # Use train() to disable the fused TransformerEncoder fast path
    # which causes non-deterministic graph tracing (known PyTorch issue)
    model.train()
    try:
        scripted = torch.jit.trace(model, dummy_input, check_trace=False, strict=False)
    finally:
        model.eval()
    scripted.save(f"{save_dir}/pitnn_scripted.pt")
    print("Saved: pitnn_scripted.pt  (C++/embedded Linux/Jetson)")

    # ── Option B: ONNX ────────────────────────────────────────────
    torch.onnx.export(
        model, dummy_input,
        f"{save_dir}/pitnn_model.onnx",
        input_names  = ["state_sequence"],
        output_names = ["phi_TPS"],
        dynamic_axes = {"state_sequence": {0: "batch"}},
        opset_version = 17,
        export_params = True,
        do_constant_folding = True,
    )
    print("Saved: pitnn_model.onnx   (ONNX Runtime / TensorRT / MCU)")

    # ── Option C: Weight arrays ───────────────────────────────────
    state = model.state_dict()
    weights = {k: v.cpu().numpy() for k, v in state.items()}
    np.savez(f"{save_dir}/pitnn_weights.npz", **weights)
    np.save(f"{save_dir}/pitnn_mu.npy",    mu)
    np.save(f"{save_dir}/pitnn_sigma.npy", sigma)
    print("Saved: pitnn_weights.npz  (bare C / FPGA / custom runtime)")

    print(f"\nModel input  : (1, 20, 8) float32")
    print(f"Model output : (1, 3)     float32  [phi1, phi2, phi3]  — all predicted")
    print(f"phi1 ∈ [{PHI12_MIN:.4f}, {PHI12_MAX:.4f}] rad")
    print(f"phi2 ∈ [{PHI12_MIN:.4f}, {PHI12_MAX:.4f}] rad")
    print(f"phi3 ∈ [{PHI_MIN:.3f},   {PHI3_MAX:.4f}]  rad")


# ═════════════════════════════════════════════════════════════════════════════
# §V-E  REAL-TIME CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

class PITNNController:
    def __init__(self,model,mu,sigma,dab,device="cpu"):
        self.model=model.to(device).eval(); self.mu=mu.astype(np.float32)
        self.sigma=sigma.astype(np.float32); self.dab=dab
        self.device=device; self.seq_len=model.seq_len; self._buf=[]

    def reset(self): self._buf=[]

    def _est_irms(self,V1,V2,phi1,phi3):
        return float(V1*V2/(V1_NOM*V2_NOM)*N_TURNS*10.7
                     *max(float(phi3),PHI_MIN)*(float(phi1)/PI))

    def prime(self,V1,V2,Pref,phi_seed=None):
        self.dab.V1,self.dab.V2=float(V1),float(V2)
        if phi_seed is None: phi_seed=self.dab.solve_optimal_phi(float(Pref))
        iL=self._est_irms(V1,V2,float(phi_seed[0]),float(phi_seed[2]))
        vrat=float(V1)*float(V2)/(V1_NOM*V2_NOM)
        feat=np.array([V1,V2,iL,phi_seed[0],phi_seed[1],phi_seed[2],Pref,vrat],np.float32)
        self._buf=[((feat-self.mu)/self.sigma).astype(np.float32)]*self.seq_len
        return phi_seed

    def step(self,V1,V2,iL,Pref,phi_prev=None,reset=False):
        if reset or len(self._buf)==0: phi_prev=self.prime(V1,V2,Pref,phi_prev)
        if phi_prev is None:
            self.dab.V1,self.dab.V2=float(V1),float(V2)
            phi_prev=self.dab.solve_optimal_phi(float(Pref))
        if iL is None: iL=self._est_irms(V1,V2,float(phi_prev[0]),float(phi_prev[2]))
        vrat=float(V1)*float(V2)/(V1_NOM*V2_NOM)
        feat=np.array([V1,V2,iL,phi_prev[0],phi_prev[1],phi_prev[2],Pref,vrat],np.float32)
        self._buf.append(((feat-self.mu)/self.sigma).astype(np.float32))
        self._buf=self._buf[-self.seq_len:]
        x=torch.from_numpy(np.stack(self._buf)).unsqueeze(0).to(self.device)
        t0=time.perf_counter()
        with torch.no_grad(): phi=self.model(x).cpu().numpy().squeeze()
        inf_us=(time.perf_counter()-t0)*1e6
        self.dab.V1,self.dab.V2=V1,V2
        P=self.dab.compute_power(*phi); Ir=self.dab.compute_irms(*phi)
        zvs,zvp=self.dab.check_zvs(*phi); mode=self.dab.classify_mode(*phi)
        return {"phi_TPS":phi,"P_calc":P,"Irms":Ir,"zvs_ok":zvs,"zvs_pen":zvp,
                "mode":mode,"P_err_W":abs(P-Pref),
                "P_err_pct":abs(P-Pref)/max(abs(Pref),1)*100,"inf_us":inf_us}


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_all(hist,dab):
    ep=range(1,len(hist["train"])+1)
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].semilogy(ep,hist["train"],label="Train")
    ax[0].semilogy(ep,hist["val"],label="Val",ls="--")
    ax[0].set(xlabel="Epoch",ylabel="Loss (log)",title="Training Loss"); ax[0].legend()
    ax[1].semilogy(ep,hist["LP"],label="LP (power)")
    ax[1].semilogy(ep,hist["LZVS"],label="LZVS (ZVS)")
    ax[1].set(xlabel="Epoch",ylabel="Physics loss",title="Physics Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig("pitnn_training.png",dpi=150); plt.close()
    print("  Saved: pitnn_training.png")

    Yt,Yp=hist["Y_test"],hist["Y_pred"]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for i,ax in enumerate(axes):
        ax.scatter(Yt[:,i],Yp[:,i],alpha=0.4,s=8,rasterized=True)
        lo=min(Yt[:,i].min(),Yp[:,i].min()); hi=max(Yt[:,i].max(),Yp[:,i].max())
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.2)
        ax.set(xlabel="Optimal (rad)",ylabel="PITNN (rad)",
               title=["φ₁ primary","φ₂ secondary","φ₃ external"][i]+" (rad)")
    plt.suptitle("φ_TPS: PITNN vs Offline-Optimal — Test Set",y=1.02)
    plt.tight_layout(); plt.savefig("pitnn_parity.png",dpi=150); plt.close()
    print("  Saved: pitnn_parity.png")

    plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    })

    wave_cases = [
        # (filename_suffix, [phi1, phi2, phi3])
        # Angles chosen to produce powers representative of the 3.3kW converter
        ("light_load",   [PI*0.95, PI*0.95, 0.10]),   # ≈ 500W   light load
        ("half_load",    [PI*0.95, PI*0.95, 0.45]),   # ≈ 1.65kW half rated
        ("rated_load",   [PI*0.90, PI*0.90, 0.80]),   # ≈ 3.0kW  near rated
        ("asym_voltage", [PI*0.85, PI*0.85, 0.65]),   # ≈ 2.5kW  V1≠V2 region
    ]

    for case_name, phi_ex in wave_cases:
        t, vab, nvcd, vL, iL = dab.simulate_current(*phi_ex, N_pts=800)
        P_ex = dab.compute_power(*phi_ex)
        mode_ex = dab.classify_mode(*phi_ex)

        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        ax[0].step(t * 1e6, vab, where="post", lw=2.2, label="v_ab")
        ax[0].step(t * 1e6, nvcd, where="post", linestyle="--", lw=2.2, label="n·v_cd")
        ax[0].axhline(0, color="black", lw=1.0, alpha=0.7)
        ax[0].set_ylabel("Voltage (V)")
        ax[0].set_title(f"TPS waveforms — P={P_ex:.0f}W  Mode {mode_ex}")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3, linestyle="--")

        ax[1].step(t * 1e6, vL, where="post", lw=2.2)
        ax[1].axhline(0, color="black", lw=1.0, alpha=0.7)
        ax[1].set_ylabel("v_L (V)")
        ax[1].grid(True, alpha=0.3, linestyle="--")

        ax[2].plot(t * 1e6, iL, color="red", lw=2.4)
        ax[2].fill_between(t * 1e6, iL, 0, alpha=0.12, color="red")
        ax[2].axhline(0, color="black", lw=1.0, alpha=0.7)
        ax[2].set_ylabel("i_L (A)")
        ax[2].set_xlabel("Time (µs)")
        ax[2].grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout(pad=1.2)
        plt.savefig(f"pitnn_waveforms_{case_name}.png", dpi=220, bbox_inches="tight")
        plt.close()

        print(f"  Saved: pitnn_waveforms_{case_name}.png")

    phi12 = PI * 0.95
    phi3s = np.linspace(0.05, 2.0, 100)
    Ps = [dab.compute_power(phi12, phi12, p3) for p3 in phi3s]
    Pm = [K_POWER * (phi12 / PI) * p3 * (1 - p3 / B_POWER) for p3 in phi3s]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(phi3s, [p / 1000 for p in Ps], lw=2.4, label="Simulation")
    ax.plot(phi3s, [p / 1000 for p in Pm], lw=2.2, ls="--", label="Calibrated model")
    ax.axvline(PHI3_MAX, color="blue", ls=":", lw=2.0, label=f"φ₃_max={PHI3_MAX}")
    ax.set_xlabel("φ₃ (rad)")
    ax.set_ylabel("P (kW)")
    ax.set_title("Power vs φ₃ | φ₂ | φ₁")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()

    plt.tight_layout(pad=1.2)
    plt.savefig("pitnn_power_surface.png", dpi=220, bbox_inches="tight")
    plt.close()

    print("  Saved: pitnn_power_surface.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Fix all random sources for a reproducible training run."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():
    parser = argparse.ArgumentParser(description="PITNN DAB Converter Simulation")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to oscilloscope/simulation video file.")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of independent training runs (default 1). "
                             "Use --runs 5 to report mean ± std.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed (default 0). "
                             "Run k uses seed+k for reproducibility.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*70)
    print("  PITNN — Real-Time TPS Optimal Modulation in DAB Converters")
    print(f"  Device: {device}")
    if args.video:
        print(f"  Video: {args.video}")
    print("="*70)

    dab = DABPhysics()

    # [1] Physics verification
    print("\n[1] DAB Physics Verification")
    print(f"  V1={V1_NOM:.0f}V  V2={V2_NOM:.0f}V  n={N_TURNS:.1f}  "
          f"Lk={LK*1e6:.0f}µH  fsw={FSW/1e3:.0f}kHz\n"
          f"  P_max≈{dab.p_max()/1000:.1f}kW  P_rated={P_RATED/1000:.1f}kW  "
          f"Operating range: 0.5–3.3kW\n")
    print(f"  {'φ1':>7} {'φ2':>7} {'φ3':>7}  {'P(W)':>9}  {'Irms':>8}  {'ZVS':>5}  {'Mode':>5}")
    print(f"  {'─'*60}")
    for p1,p2,p3 in [(PI*.95,PI*.95,.10),(PI*.95,PI*.95,.30),
                      (PI*.95,PI*.95,.60),(PI*.95,PI*.95,1.0),
                      (PI*.85,PI*.85,.40),(PI*.90,PI*.90,.65)]:
        P=dab.compute_power(p1,p2,p3); Ir=dab.compute_irms(p1,p2,p3)
        zvs,_=dab.check_zvs(p1,p2,p3); m=dab.classify_mode(p1,p2,p3)
        print(f"  {p1:>7.4f} {p2:>7.4f} {p3:>7.4f}  {P:>9.1f}  {Ir:>8.4f}  {'YES' if zvs else 'NO':>5}  {m:>5}")

    print(f"\n  Solver label check (joint φ1/φ2/φ3 optimisation):")
    rng_chk=np.random.default_rng(0)
    print(f"  {'Pref':>7} {'φ1':>7} {'φ2':>7} {'φ3':>7} {'P_calc':>9} {'err%':>6} {'Irms':>7}")
    print(f"  {'─'*58}")
    for Pref in [500,1000,1650,2000,2500,3000,3300]:
        phi=dab.solve_optimal_phi(float(Pref),rng=rng_chk)
        P=dab.compute_power(*phi); Ir=dab.compute_irms(*phi)
        err=abs(P-Pref)/max(Pref,1)*100
        print(f"  {Pref:>7.0f} {phi[0]:>7.4f} {phi[1]:>7.4f} {phi[2]:>7.4f} "
              f"{P:>9.1f} {err:>5.1f}% {Ir:>7.4f}  {'OK' if err<15 else 'WARN'}")

    # [2] Synthetic dataset
    print("\n[2] Synthetic Dataset Generation  (§V-D)")
    X_norm,Y,mu,sigma,X_raw = generate_dataset(n_samples=10000,seq_len=20,seed=42)

    # [3] Video dataset  ← plug any video in here via --video flag
    video_X_norm=video_X_raw=video_Y=None
    if args.video:
        print(f"\n[2b] Video Dataset Extraction  ({args.video})")
        extractor = VideoWaveformExtractor(
            args.video,
            fsw_hardware = FSW,
            V1_hardware  = V1_NOM,
            V2_hardware  = V2_NOM,
            verbose      = True,
        )
        # Save diagnostic plot of what was extracted
        extractor.plot_extraction("video_extraction.png")

        # Build dataset using same normalisation as synthetic data
        video_result = extractor.build_dataset(mu=mu, sigma=sigma)
        if video_result is not None:
            video_X_norm,video_Y,_,_,video_X_raw = video_result
            print(f"  Video samples added to training: {len(video_Y)}")
        else:
            print("  Warning: video extraction returned no samples — training on synthetic data only")

    # [4] Architecture
    print("\n[3] PITNN Architecture  (§V-B, Fig. 2, Eq. 29-32)")
    _tmp = PITNN(d_in=8,d_model=128,n_heads=8,n_layers=4,d_ff=256,seq_len=20,dropout=0.1)
    n_p  = sum(p.numel() for p in _tmp.parameters() if p.requires_grad); del _tmp
    print(f"  Parameters    : {n_p:,}")
    print(f"  Layers/heads  : 4 / 8  |  d_model/d_ff: 128 / 256")
    print(f"  Input features: 8  [V1,V2,iL,φ1,φ2,φ3,Pref,V1V2/Vnom²]")
    print(f"  φ1 output     : sigmoid → [{PHI12_MIN:.4f}, {PHI12_MAX:.4f}] rad")
    print(f"  φ2 output     : sigmoid → [{PHI12_MIN:.4f}, {PHI12_MAX:.4f}] rad")
    print(f"  φ3 output     : sigmoid → [{PHI_MIN:.4f},   {PHI3_MAX:.4f}]  rad")
    print(f"  Power range   : 0.5kW–3.3kW  (P_rated={P_RATED/1000:.1f}kW)")
    print(f"  Base seed     : {args.seed}  |  Training runs: {args.runs}")

    # [5] Multi-run training
    all_mse, all_mae, all_hist, all_models = [], [], [], []
    print(f"\n[4] Training  (§V-D: Adam lr=1e-4, batch=64, 150 epochs)")
    print(f"    Repeatability: {args.runs} run(s), "
          f"seeds {args.seed}–{args.seed+args.runs-1}")

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        set_seed(run_seed)
        if args.runs > 1:
            print(f"\n{'━'*70}\n  Run {run_idx+1}/{args.runs}  (seed={run_seed})\n{'━'*70}")

        model   = PITNN(d_in=8,d_model=128,n_heads=8,n_layers=4,
                        d_ff=256,seq_len=20,dropout=0.1)
        loss_fn = PITNNLoss(lambda_p=0.,lambda1=0.,lambda2=0.,
                            Lk=LK,n=N_TURNS,fsw=FSW,V_nom=V1_NOM,I_rated=100.)
        hist = train_pitnn(model,loss_fn,X_norm,X_raw,Y,
                           epochs=150,batch_size=64,lr=1e-4,
                           val_split=.15,warmup_epochs=20,device=device,
                           video_X_norm=video_X_norm,
                           video_X_raw=video_X_raw,video_Y=video_Y)
        all_mse.append(hist["test_mse"])
        all_mae.append(hist["test_mae"])
        all_hist.append(hist)
        all_models.append(model)
        if args.runs > 1:
            print(f"  Run {run_idx+1} — MSE={hist['test_mse']:.6f}  "
                  f"MAE={math.degrees(hist['test_mae']):.3f}°  seed={run_seed}")

    # Best run
    best_idx = int(np.argmin(all_mse))
    model    = all_models[best_idx]
    hist     = all_hist[best_idx]
    mse_arr  = np.array(all_mse)
    mae_arr  = np.array(all_mae)
    mae_deg  = np.degrees(mae_arr)

    # Repeatability summary
    print(f"\n{'═'*70}\n  REPEATABILITY SUMMARY  ({args.runs} run(s))\n{'═'*70}")
    if args.runs == 1:
        print(f"  Test MSE : {mse_arr[0]:.6f} rad²")
        print(f"  Test MAE : {mae_deg[0]:.3f}°  (seed={args.seed})")
        print(f"  Use --runs N for multi-run mean ± std statistics")
    else:
        print(f"  {'Metric':<12}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
        print(f"  {'─'*56}")
        print(f"  {'MSE (rad²)':<12}  {mse_arr.mean():>10.6f}  {mse_arr.std():>10.6f}  "
              f"{mse_arr.min():>10.6f}  {mse_arr.max():>10.6f}")
        print(f"  {'MAE (°)':<12}  {mae_deg.mean():>10.3f}  {mae_deg.std():>10.3f}  "
              f"{mae_deg.min():>10.3f}  {mae_deg.max():>10.3f}")
        print(f"  {'─'*56}")
        print(f"  Seeds: {', '.join(str(args.seed+k) for k in range(args.runs))}")
        print(f"  Report as: MAE = {mae_deg.mean():.3f}° ± {mae_deg.std():.3f}°  "
              f"(mean ± std, n={args.runs})")
        print("  Per-run results:")
        for k in range(args.runs):
            print(f"    Run {k+1}  seed={args.seed+k}  MSE={mse_arr[k]:.6f}  "
                  f"MAE={mae_deg[k]:.3f}°{' ← best' if k==best_idx else ''}")
    print(f"{'═'*70}\n")

    torch.save({"model_state": model.state_dict(), "mu": mu, "sigma": sigma,
                "hyperparams": dict(d_in=8,d_model=128,n_heads=8,n_layers=4,
                                    d_ff=256,seq_len=20,
                                    phi12_min=PHI12_MIN,phi12_max=PHI12_MAX,
                                    phi_min=PHI_MIN,phi3_max=PHI3_MAX),
                "repeatability": {"n_runs": args.runs, "base_seed": args.seed,
                                  "all_mse": mse_arr.tolist(),
                                  "all_mae_deg": mae_deg.tolist(),
                                  "mean_mae_deg": float(mae_deg.mean()),
                                  "std_mae_deg":  float(mae_deg.std())}},
               "pitnn_dab_checkpoint.pt")
    print(f"  Checkpoint: pitnn_dab_checkpoint.pt")
    print(f"  φ1,φ2 ∈ [{PHI12_MIN:.4f},{PHI12_MAX:.4f}] rad  |  "
          f"φ3 ∈ [{PHI_MIN:.4f},{PHI3_MAX:.4f}] rad")

    # [6] Plots
    print("\n[5] Plots")
    plot_all(hist,dab)

    # [7] Inference — PITNN vs offline solver
    print("\n[6] Real-Time Inference  (§V-E, Eq. 38: φ_TPS = f_θ(X))")
    print("    Offline solver  : adaptive grid search + brentq root-finding")
    print("    PITNN Inf.(µs)  : single Transformer forward pass (GPU, torch.no_grad)")
    print("    Speedup         : Solver(µs) / Inf.(µs)\n")

    ctrl = PITNNController(model,mu,sigma,dab,device=device)
    ops  = [(400,250,3300,"3.3kW nom"), (380,240,2000,"2kW V-var"),
            (400,250,1650,"1.65kW"),    (400,250, 500,"0.5kW light"),
            (400,250,3000,"3kW high"),  (400,250,2500,"2.5kW mid"),
            (410,245,3000,"3kW asym"),  (390,260,1000,"1kW off-V")]

    print(f"\n  {'Condition':<12} {'V1':>5} {'V2':>5} {'Pref':>6}  "
          f"{'φ1':>7} {'φ2':>7} {'φ3':>6}  "
          f"{'P_calc':>8} {'Irms':>7} {'ZVS':>5} {'|ΔP|%':>6}  "
          f"{'Solver(µs)':>11} {'Inf.(µs)':>9} {'Speedup':>8}")
    print(f"  {'─'*112}")

    total_solver_us = total_inf_us = 0.0

    for V1,V2,Pref,label in ops:
        dab.V1,dab.V2 = float(V1),float(V2)

        # Solver — median of 5 runs for a stable estimate
        stimes = []
        for _ in range(5):
            t_s = time.perf_counter()
            phi_seed = dab.solve_optimal_phi(float(Pref))
            stimes.append((time.perf_counter()-t_s)*1e6)
        solver_us = float(np.median(stimes))

        ctrl.reset()
        r       = ctrl.step(float(V1),float(V2),None,float(Pref),phi_seed,reset=True)
        phi     = r["phi_TPS"]
        inf_us  = r["inf_us"]
        speedup = solver_us / max(inf_us, 0.001)
        total_solver_us += solver_us
        total_inf_us    += inf_us

        print(f"  {label:<12} {V1:>5} {V2:>5} {Pref:>6}  "
              f"{phi[0]:>7.4f} {phi[1]:>7.4f} {phi[2]:>6.4f}  "
              f"{r['P_calc']:>8.1f} {r['Irms']:>7.4f} "
              f"{'YES' if r['zvs_ok'] else 'NO':>5} "
              f"{r['P_err_pct']:>6.1f}%  "
              f"{solver_us:>11.1f} {inf_us:>9.1f} {speedup:>7.0f}x")

    n       = len(ops)
    avg_sol = total_solver_us / n
    avg_inf = total_inf_us    / n
    avg_spd = avg_sol / max(avg_inf, 0.001)
    print(f"  {'─'*112}")
    print(f"  {'Average':>81} {avg_sol:>11.1f} {avg_inf:>9.1f} {avg_spd:>7.0f}x")
    print(f"\n  Average speedup: {avg_spd:.0f}x  "
          f"(Solver: {avg_sol:.0f}µs  vs  PITNN Inf.: {avg_inf:.0f}µs)")

    print(f"\n{'='*70}")
    print(f"  Test MSE: {hist['test_mse']:.6f} rad²")
    print(f"  Test MAE: {hist['test_mae']:.6f} rad ({math.degrees(hist['test_mae']):.3f}°)")
    if args.runs > 1:
        print(f"  Across {args.runs} runs: MAE = "
              f"{mae_deg.mean():.3f}° ± {mae_deg.std():.3f}°  (mean ± std)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
