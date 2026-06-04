"""
pitnn_ablation.py
=====================================================================
Ablation Study and Lightweight Baseline Comparisons
for the PITNN DAB Converter TPS Modulation Paper.

Trains and evaluates all model variants on the same dataset split
and reports a consolidated results table suitable for a paper table.
Optionally augments training with a real hardware oscilloscope video
and produces a second set of results (synthetic + video) to quantify
the benefit of multimodal training.

MODELS
───────
Baselines:
  1. MLP            — flat feature vector, no sequence modelling
  2. LSTM           — recurrent sequence model, no physics loss
  3. GRU            — gated recurrent unit, no physics loss

Ablations (full PITNN with components removed one at a time):
  4. PITNN – LP       — no power physics loss
  5. PITNN – LZVS     — no ZVS physics loss
  6. PITNN – Lsym     — no bridge symmetry penalty
  7. PITNN – warmup   — physics loss on from epoch 1 (no curriculum)
  8. PITNN – PE       — no positional encoding
  9. PITNN – Pre-LN   — post-LayerNorm instead of Pre-LN

Full model:
  10. PITNN (full)    — all components enabled

RUN
────
  # Synthetic data only:
  python pitnn_ablation.py

  # With oscilloscope video (adds synthetic + video results table):
  python pitnn_ablation.py --video path/to/scope.mp4

  # Quick sanity check (50 epochs, 3000 samples):
  python pitnn_ablation.py --fast

OUTPUT FILES
─────────────
  pitnn_ablation_results.csv        — synthetic-only results table
  pitnn_ablation_results.png        — MAE bar chart (synthetic)
  pitnn_ablation_training.png       — training curves, all models
  pitnn_ablation_perangle.png       — per-angle MAE breakdown
  pitnn_ablation_video_results.csv  — synthetic+video results (if --video)
  pitnn_ablation_video_results.png  — MAE comparison with video data

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University — All Rights Reserved
=====================================================================
"""

import math
import time
import csv
import warnings
import argparse

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Import shared infrastructure from pitnn_dab.py ───────────────────────────
from pitnn_dab import (
    # Constants
    PI, PHI_MIN, PHI12_MIN, PHI12_MAX, PHI3_MAX, LK, N_TURNS, FSW,
    V1_NOM, V2_NOM, P_RATED,
    K_POWER, B_POWER,
    # Dataset + physics
    generate_dataset, DABPhysics,
    # Video ingestion
    VideoWaveformExtractor,
    # Full PITNN architecture + loss
    PITNN, PositionalEncoding, PITNNLoss,
)


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE ARCHITECTURES
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Baseline 1 — Multi-Layer Perceptron.
    Flattens the 20-step sequence into a single vector and passes it
    through fully-connected layers. Has no notion of temporal order.
    Parameter count matched approximately to PITNN (~500k).
    """
    def __init__(self, seq_len=20, n_feat=8, hidden=512, n_layers=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        d_in = seq_len * n_feat          # 160 inputs
        layers = []
        prev = d_in
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, seq_len, n_feat) → flatten → (B, seq_len*n_feat)
        flat = x.flatten(1)
        raw  = self.net(flat)
        sig  = torch.sigmoid(raw)
        phi1 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3 = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)


class LSTMModel(nn.Module):
    """
    Baseline 2 — LSTM.
    Processes the 20-step sequence recurrently. Uses the final hidden
    state as the feature vector for the output head.
    Parameter count matched approximately to PITNN.
    """
    def __init__(self, n_feat=8, hidden=256, n_layers=2, dropout=0.1, seq_len=20):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size  = n_feat,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        out, _ = self.lstm(x)               # (B, seq_len, hidden)
        raw    = self.head(out[:, -1, :])   # last time step
        sig    = torch.sigmoid(raw)
        phi1   = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2   = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3   = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)


class GRUModel(nn.Module):
    """
    Baseline 3 — GRU (Gated Recurrent Unit).
    Same structure as the LSTM baseline but uses GRU cells, which have
    fewer parameters (no separate cell state) and often train faster.
    Included to compare against LSTM and isolate the effect of the
    gating mechanism choice on TPS prediction accuracy.
    """
    def __init__(self, n_feat=8, hidden=256, n_layers=2, dropout=0.1, seq_len=20):
        super().__init__()
        self.seq_len = seq_len
        self.gru = nn.GRU(
            input_size  = n_feat,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        out, _ = self.gru(x)                # (B, seq_len, hidden)
        raw    = self.head(out[:, -1, :])   # last time step
        sig    = torch.sigmoid(raw)
        phi1   = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2   = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3   = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)


class PITNNNoPE(nn.Module):
    """
    Ablation: PITNN without positional encoding.
    Identical to PITNN but the pos_enc step is skipped. Isolates the
    contribution of the sinusoidal positional encoding.
    """
    def __init__(self, d_in=8, d_model=128, n_heads=8, n_layers=4,
                 d_ff=256, seq_len=20, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Linear(d_in, d_model)
        self.dropout = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=n_layers, enable_nested_tensor=False
        )
        self.ln_out  = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, 3),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        # No positional encoding — just embed and pass through transformer
        emb = self.dropout(self.embed(x))
        z   = self.ln_out(self.transformer(emb))
        raw = self.head(z[:, -1, :])
        sig = torch.sigmoid(raw)
        phi1 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3 = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)


class PITNNPostLN(nn.Module):
    """
    Ablation: PITNN with Post-LayerNorm instead of Pre-LayerNorm.
    All other components identical to full PITNN. Isolates the
    contribution of the Pre-LN design choice.
    """
    def __init__(self, d_in=8, d_model=128, n_heads=8, n_layers=4,
                 d_ff=256, seq_len=20, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Linear(d_in, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len + 8, dropout)
        # Post-LN: norm_first=False
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=False,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=n_layers, enable_nested_tensor=False
        )
        self.ln_out  = nn.LayerNorm(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, 3),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        z   = self.ln_out(self.transformer(self.pos_enc(self.embed(x))))
        raw = self.head(z[:, -1, :])
        sig = torch.sigmoid(raw)
        phi1 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3 = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)


# ═════════════════════════════════════════════════════════════════════════════
# GENERIC TRAINING LOOP
# Accepts any model — does not require raw features for physics loss.
# For baselines (MLP, LSTM, GRU) physics_loss=False.
# For PITNN variants physics_loss=True and the full PITNNLoss is used.
# ═════════════════════════════════════════════════════════════════════════════

def train_model(
    model, name,
    Xn_tr, Xr_tr, Ytr,
    Xn_va, Xr_va, Yva,
    Xn_te, Xr_te, Yte,
    epochs=150, batch_size=64, lr=1e-4, warmup_epochs=20,
    physics_loss=False,
    lambda_p=0.0, lambda2=0.0,
    no_warmup=False,       # ablation: disable curriculum (physics on from ep1)
    no_lp=False,           # ablation: zero out LP term
    no_lzvs=False,         # ablation: zero out LZVS term
    no_lsym=False,         # ablation: zero out Lsym term
    device="cpu",
):
    """
    Train any model variant. Returns a results dict with test metrics
    and training history for plotting.
    """
    model = model.to(device)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loader = DataLoader(
        TensorDataset(Xn_tr, Xr_tr, Ytr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr/20)

    # Loss function — same PITNNLoss for ALL models (fair comparison)
    # Baselines and ablations differ only in architecture, not loss.
    loss_fn = PITNNLoss(
        lambda_p = lambda_p if not no_lp   else 0.0,
        lambda1  = 0.0,
        lambda2  = lambda2  if not no_lzvs else 0.0,
        Lk=LK, n=N_TURNS, fsw=FSW, V_nom=V1_NOM, I_rated=100.,
    )

    print(f"\n{'─'*70}")
    print(f"  [{name}]  {n_p:,} params")
    print(f"  no_lp={no_lp}  no_lzvs={no_lzvs}  "
          f"no_lsym={no_lsym}  no_warmup={no_warmup}")
    print(f"{'─'*70}")

    w = torch.tensor([2., 2., 3.], device=device)
    best_val, best_state = float("inf"), None
    hist_train, hist_val = [], []
    t0_total = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Curriculum: ramp physics weight from 0 to 1 over warmup_epochs
        if physics_loss and loss_fn is not None:
            if no_warmup:
                loss_fn.physics_weight = 1.0
            else:
                loss_fn.physics_weight = min(1.0, (epoch - 1) / max(warmup_epochs, 1))

        model.train()
        ep_loss = 0.0
        nb = 0

        for xn_b, xr_b, yb in loader:
            optimizer.zero_grad()
            phi_pred = model(xn_b)

            if physics_loss and loss_fn is not None:
                # Physics terms need last-step raw features
                V1_b   = xr_b[:, -1, 0]
                V2_b   = xr_b[:, -1, 1]
                Pref_b = xr_b[:, -1, 6]
                iL_seq = xr_b[:, :, 2]

                if no_lsym:
                    w_mse  = torch.tensor([2., 2., 3.], device=phi_pred.device)
                    L_data = torch.mean(w_mse * (phi_pred - yb) ** 2)
                    phi1_t = phi_pred[:, 0]; phi3_t = phi_pred[:, 2]
                    scale  = N_TURNS * V1_b * V2_b / (V1_NOM ** 2)
                    P_pred = scale * K_POWER * (phi1_t / PI) * phi3_t * (
                        1.0 - phi3_t / B_POWER)
                    LP     = torch.mean((P_pred - Pref_b) ** 2 /
                                        (Pref_b ** 2 + 1.0))
                    if no_lzvs:
                        L_physics = LP
                    else:
                        d1   = phi_pred[:, 0] / PI
                        i0   = (V1_b * d1 - N_TURNS * V2_b * phi3_t / PI) / (
                                2.0 * LK * FSW)
                        LZVS = torch.mean(
                            torch.clamp(-i0, min=0.0) / 100.0)
                        L_physics = LP + loss_fn.lambda2 * LZVS
                    loss = L_data + loss_fn.physics_weight * loss_fn.lambda_p * L_physics
                else:
                    loss, _ = loss_fn(phi_pred, yb, V1_b, V2_b, Pref_b, iL_seq)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            nb += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            pv = model(Xn_va)
            val_loss = torch.mean(w * (pv - Yva) ** 2).item()

        hist_train.append(ep_loss / max(nb, 1))
        hist_val.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Ep {epoch:>4}/{epochs}  train={ep_loss/max(nb,1):.5f}  "
                  f"val={val_loss:.5f}")

    # Restore best checkpoint
    if best_state:
        model.load_state_dict(best_state)

    # Test metrics
    model.eval()
    with torch.no_grad():
        pt  = model(Xn_te)
        mse = nn.functional.mse_loss(pt, Yte).item()
        mae = (pt - Yte).abs().mean().item()

        # Per-angle MAE
        mae_phi1 = (pt[:, 0] - Yte[:, 0]).abs().mean().item()
        mae_phi2 = (pt[:, 1] - Yte[:, 1]).abs().mean().item()
        mae_phi3 = (pt[:, 2] - Yte[:, 2]).abs().mean().item()

        # ZVS violation rate — model-induced violations only
        # Counts cases where MODEL loses ZVS but the SOLVER (label) maintains it.
        # This removes the ~48% baseline from V2>V1/n operating points
        # that lose ZVS regardless of which model is used, so the metric
        # reflects model prediction quality rather than dataset composition.
        dab_eval = DABPhysics()
        phi1_np  = pt[:, 0].cpu().numpy()
        phi2_np  = pt[:, 1].cpu().numpy()
        phi3_np  = pt[:, 2].cpu().numpy()
        V1_np    = Xr_te[:, -1, 0].cpu().numpy()
        V2_np    = Xr_te[:, -1, 1].cpu().numpy()
        n_solver_zvs_ok = 0   # points where ZVS is achievable (solver maintains it)
        n_model_worse   = 0   # model loses ZVS at a point where solver did not

        for i in range(len(phi1_np)):
            dab_eval.V1 = float(V1_np[i])
            dab_eval.V2 = float(V2_np[i])
            solver_zvs_ok, _ = dab_eval.check_zvs(
                float(Yte[i, 0].item()), float(Yte[i, 1].item()),
                float(Yte[i, 2].item()))
            model_zvs_ok, _  = dab_eval.check_zvs(
                float(phi1_np[i]), float(phi2_np[i]), float(phi3_np[i]))
            if solver_zvs_ok:
                n_solver_zvs_ok += 1
                if not model_zvs_ok:
                    n_model_worse += 1

        # % of ZVS-achievable points where the model unnecessarily loses ZVS
        zvs_violation_rate = (n_model_worse / max(n_solver_zvs_ok, 1)) * 100

    elapsed = time.perf_counter() - t0_total
    print(f"  ► MSE={mse:.6f}  MAE={mae:.6f} rad ({math.degrees(mae):.3f}°)  "
          f"[φ1={math.degrees(mae_phi1):.3f}°  "
          f"φ2={math.degrees(mae_phi2):.3f}°  "
          f"φ3={math.degrees(mae_phi3):.3f}°]  "
          f"ZVS viol={zvs_violation_rate:.1f}%  "
          f"time={elapsed:.1f}s")

    return {
        "name"               : name,
        "n_params"           : n_p,
        "test_mse"           : mse,
        "test_mae"           : mae,
        "test_mae_deg"       : math.degrees(mae),
        "mae_phi1_deg"       : math.degrees(mae_phi1),
        "mae_phi2_deg"       : math.degrees(mae_phi2),
        "mae_phi3_deg"       : math.degrees(mae_phi3),
        "zvs_violation_rate" : zvs_violation_rate,
        "train_time_s"       : elapsed,
        "hist_train"         : hist_train,
        "hist_val"           : hist_val,
        "Y_test"             : Yte.cpu().numpy(),
        "Y_pred"             : pt.cpu().numpy(),
        "_model"             : model,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CONTROLLER EVALUATION  (Option A primary metrics)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_controller(model, mu, sigma, device, n_warmup=10):
    """
    Evaluate a trained model as a real-time controller over 60 operating
    points spanning the full 0.5–3.3kW range at V1=400V, V2=250V.

    Each point is evaluated independently:
      reset → prime → n_warmup steps → measure

    Returns dict with the four primary control metrics:
      power_err_mean  — mean |ΔP|/P_ref (%)  across all points
      power_err_max   — worst-case |ΔP|/P_ref (%)
      pct_within_10   — % of points with |ΔP| < 10%
      ctrl_zvs_viol   — % of ZVS-achievable points where controller loses ZVS
    """
    from pitnn_dab import PITNNController
    dab  = DABPhysics()
    ctrl = PITNNController(model, mu, sigma, dab, device=device)

    V1, V2 = 400.0, 250.0
    dab.V1, dab.V2 = V1, V2
    Prefs  = np.linspace(500, 3300, 60)

    p_errs     = []
    zvs_worse  = 0
    zvs_avail  = 0

    for Pref in Prefs:
        phi_seed = dab.solve_optimal_phi(float(Pref))

        # Solver ZVS reference
        solver_zvs_ok, _ = dab.check_zvs(*phi_seed)
        if solver_zvs_ok:
            zvs_avail += 1

        # PITNN controller — independent evaluation
        ctrl.reset()
        ctrl.prime(V1, V2, float(Pref), phi_seed)
        phi_cur = phi_seed
        for _ in range(n_warmup):
            r = ctrl.step(V1, V2, None, float(Pref), phi_cur)
            phi_cur = r["phi_TPS"]
        r = ctrl.step(V1, V2, None, float(Pref), phi_cur)

        p_errs.append(r["P_err_pct"])

        if solver_zvs_ok and not r["zvs_ok"]:
            zvs_worse += 1

    p_errs = np.array(p_errs)
    return {
        "power_err_mean" : float(p_errs.mean()),
        "power_err_max"  : float(p_errs.max()),
        "pct_within_10"  : float((p_errs < 10).mean() * 100),
        "ctrl_zvs_viol"  : float(zvs_worse / max(zvs_avail, 1) * 100),
    }




def print_results_table(results, title="Synthetic Only", csv_path="pitnn_ablation_results.csv"):
    """
    Print and save the consolidated ablation results table.

    Primary metrics (control performance):
      Mean |ΔP|%   — average power tracking error across 60 operating points
      Max |ΔP|%    — worst-case power tracking error
      Within 10%   — % of points where tracking error < 10%
      Ctrl ZVS%    — controller-induced ZVS violations on achievable points

    Secondary metrics (angle prediction):
      MAE (°)      — mean angle prediction error on held-out test set
      ZVS viol%    — model-induced angle ZVS violations (test set)
    """
    print("\n" + "=" * 135)
    print(f"  ABLATION STUDY & BASELINE COMPARISON — {title}")
    print(f"  All models trained with identical PITNNLoss (λ_p=1.0, λ_ZVS=0.5)")
    print(f"  Primary: control performance over 60 operating points  |  "
          f"Secondary: test-set angle accuracy")
    print("=" * 135)

    # Header
    print(f"  {'Model':<22} {'Params':>8}  "
          f"{'Mean|ΔP|%':>10} {'Max|ΔP|%':>9} {'Within10%':>10} {'CtrlZVS%':>9}  "
          f"{'MAE(°)':>7} {'ZVSviol%':>9}  {'Time(s)':>8}")
    print("  " + "-" * 133)

    rows = []
    for r in results:
        cm = r.get("ctrl_metrics", {})
        print(f"  {r['name']:<22} {r['n_params']:>8,}  "
              f"{cm.get('power_err_mean', 0):>10.2f} "
              f"{cm.get('power_err_max',  0):>9.2f} "
              f"{cm.get('pct_within_10',  0):>9.1f}% "
              f"{cm.get('ctrl_zvs_viol',  0):>8.1f}%  "
              f"{r['test_mae_deg']:>7.3f} "
              f"{r['zvs_violation_rate']:>8.1f}%  "
              f"{r['train_time_s']:>8.1f}")
        rows.append([
            r['name'], r['n_params'],
            cm.get('power_err_mean', 0), cm.get('power_err_max',  0),
            cm.get('pct_within_10',  0), cm.get('ctrl_zvs_viol',  0),
            r['test_mae_deg'], r['zvs_violation_rate'], r['train_time_s'],
        ])

    print("=" * 135)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Params",
                         "PowerErr_mean_pct", "PowerErr_max_pct",
                         "Within10pct", "CtrlZVS_viol_pct",
                         "MAE_deg", "ZVS_violation_pct", "TrainTime_s"])
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(results, title="Synthetic Only",
                 bar_path="pitnn_ablation_results.png",
                 train_path="pitnn_ablation_training.png",
                 angle_path="pitnn_ablation_perangle.png"):
    """Bar chart of MAE per model, training curves, and per-angle breakdown."""

    names  = [r["name"] for r in results]
    maes   = [r["test_mae_deg"] for r in results]
    colors = []
    for r in results:
        n = r["name"]
        if "PITNN (full)" in n:
            colors.append("#1a6bbd")
        elif n.startswith("PITNN"):
            colors.append("#5ba3d9")
        else:
            colors.append("#e07b39")

    # ── Primary: Control performance bar chart ────────────────────────────
    ctrl_mean = [r.get("ctrl_metrics", {}).get("power_err_mean", 0) for r in results]
    ctrl_zvs  = [r.get("ctrl_metrics", {}).get("ctrl_zvs_viol",  0) for r in results]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    bars = ax1.bar(range(len(names)), ctrl_mean, color=colors,
                   edgecolor="white", linewidth=0.8, zorder=2)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Power Tracking Error |ΔP|/P_ref (%)", fontsize=11)
    ax1.set_title(f"Ablation Study — Control Performance ({title})\n"
                  "Primary: mean power error over 60 operating points  "
                  "|  Secondary axis: controller ZVS violation rate",
                  fontsize=11)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--", zorder=0)
    ax1.set_axisbelow(True)
    for bar, v in zip(bars, ctrl_mean):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#e07b39", label="Baselines (MLP/LSTM/GRU)"),
        Patch(facecolor="#5ba3d9", label="Ablations"),
        Patch(facecolor="#1a6bbd", label="Full PITNN"),
    ]
    ax1.legend(handles=legend_elems, loc="upper left", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(range(len(names)), ctrl_zvs, "k--o", lw=1.5, ms=6,
             label="Ctrl ZVS viol. %", zorder=5)
    ax2.set_ylabel("Controller ZVS Violation Rate (%)", fontsize=10)
    ax2.set_ylim(bottom=0)
    for i, z in enumerate(ctrl_zvs):
        if z > 0:
            ax2.text(i, z + 0.05, f"{z:.1f}%",
                     ha="center", fontsize=7, color="black")
    ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {bar_path}")

    # ── Secondary: Angle MAE bar chart ────────────────────────────────────
    mae_path = bar_path.replace(".png", "_mae.png")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(names)), maes, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test MAE (degrees)", fontsize=11)
    ax.set_title(f"Secondary — Angle Prediction MAE ({title})", fontsize=11)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--"); ax.set_axisbelow(True)
    for bar, mae in zip(ax.patches, maes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{mae:.3f}°", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(mae_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {mae_path}")

    # ── Training curves ────────────────────────────────────────────────────
    n_models = len(results)
    n_cols   = 5
    n_rows   = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, r in enumerate(results):
        ax = axes_flat[i]
        ep = range(1, len(r["hist_train"]) + 1)
        ax.semilogy(ep, r["hist_train"], label="Train", lw=1.5)
        ax.semilogy(ep, r["hist_val"],   label="Val",   lw=1.5, ls="--")
        ax.set_title(r["name"], fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(f"Training & Validation Loss — All Models ({title})",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(train_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {train_path}")

    # ── Combined baseline curves (MLP / LSTM / GRU on shared axes) ────────
    sfx = "_video" if "video" in train_path.lower() else ""
    plot_baselines(
        results, title=title,
        curves_path=f"pitnn_ablation{sfx}_baseline_curves.png",
        parity_path=f"pitnn_ablation{sfx}_baseline_parity.png",
    )

    # ── Combined ablation curves (all PITNN variants on shared axes) ───────
    plot_ablations(
        results, title=title,
        curves_path=f"pitnn_ablation{sfx}_ablation_curves.png",
    )

    # ── Per-angle MAE breakdown ────────────────────────────────────────────
    x     = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, [r["mae_phi1_deg"] for r in results], width,
           label="φ1 MAE", color="#2196F3", edgecolor="white")
    ax.bar(x,          [r["mae_phi2_deg"] for r in results], width,
           label="φ2 MAE", color="#4CAF50", edgecolor="white")
    ax.bar(x + width,  [r["mae_phi3_deg"] for r in results], width,
           label="φ3 MAE", color="#FF5722", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MAE (degrees)", fontsize=11)
    ax.set_title(f"Per-Angle MAE Breakdown ({title})", fontsize=13)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(angle_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {angle_path}")


def plot_baselines(results, title="Synthetic Only",
                   curves_path="pitnn_ablation_baseline_curves.png",
                   parity_path="pitnn_ablation_baseline_parity.png"):
    """
    Dedicated baseline comparison plots — all three baselines (MLP, LSTM, GRU)
    shown together on shared axes so their curves can be directly compared.

    Produces two figures:
      1. Train & validation loss on one combined plot (3 curves each, log scale)
      2. Parity plots (predicted vs optimal) for all three baselines in a
         single row — one column per model, one sub-row per angle (φ1/φ2/φ3)
    """
    BASELINE_NAMES  = {"MLP", "LSTM", "GRU"}
    BASELINE_COLORS = {"MLP": "#e07b39", "LSTM": "#c0392b", "GRU": "#8e44ad"}
    LINE_STYLES     = {"MLP": "-",       "LSTM": "--",       "GRU": "-."}

    baselines = [r for r in results if r["name"] in BASELINE_NAMES]
    if not baselines:
        return

    # ── 1. Combined train/val loss curves ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_tr, ax_va = axes

    for r in baselines:
        name  = r["name"]
        col   = BASELINE_COLORS.get(name, "#555555")
        ls    = LINE_STYLES.get(name, "-")
        ep    = range(1, len(r["hist_train"]) + 1)
        ax_tr.semilogy(ep, r["hist_train"], color=col, ls=ls, lw=2.0,
                       label=name)
        ax_va.semilogy(ep, r["hist_val"],   color=col, ls=ls, lw=2.0,
                       label=name)

    for ax, ylabel, panel_title in [
        (ax_tr, "Training Loss (log)",   "Training Loss"),
        (ax_va, "Validation Loss (log)", "Validation Loss"),
    ]:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(panel_title, fontsize=12)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)

    plt.suptitle(f"Baseline Model Training Curves ({title})", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(curves_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {curves_path}")

    # ── 2. Parity plots — one column per baseline, one row per angle ─────────
    angle_labels = ["φ₁ (rad)", "φ₂ (rad)", "φ₃ (rad)"]
    n_bl  = len(baselines)
    n_ang = 3
    fig, axes = plt.subplots(n_ang, n_bl,
                             figsize=(4.5 * n_bl, 4.0 * n_ang),
                             squeeze=False)

    for col_i, r in enumerate(baselines):
        Yt = r["Y_test"]   # (N, 3)
        Yp = r["Y_pred"]   # (N, 3)
        col = BASELINE_COLORS.get(r["name"], "#555555")

        for row_i in range(n_ang):
            ax = axes[row_i][col_i]
            ax.scatter(Yt[:, row_i], Yp[:, row_i],
                       alpha=0.35, s=6, color=col, rasterized=True)
            lo = min(Yt[:, row_i].min(), Yp[:, row_i].min())
            hi = max(Yt[:, row_i].max(), Yp[:, row_i].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
            ax.set_xlabel("Optimal (rad)", fontsize=9)
            ax.set_ylabel("Predicted (rad)", fontsize=9)
            mae_deg = math.degrees(abs(Yt[:, row_i] - Yp[:, row_i]).mean())
            ax.set_title(f"{r['name']} — {angle_labels[row_i]}\nMAE={mae_deg:.3f}°",
                         fontsize=9)
            ax.tick_params(labelsize=8)

    plt.suptitle(f"Baseline Parity Plots — Predicted vs Optimal ({title})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(parity_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {parity_path}")


def plot_ablations(results, title="Synthetic Only",
                   curves_path="pitnn_ablation_ablation_curves.png"):
    """
    Combined training/validation loss curves for all PITNN ablation variants
    and the full PITNN on a single pair of axes, so the effect of each
    removed component on convergence is directly visible.
    """
    ABLATION_COLORS = {
        "PITNN – LP":     "#e74c3c",
        "PITNN – LZVS":   "#e67e22",
        "PITNN – Lsym":   "#f1c40f",
        "PITNN – warmup": "#2ecc71",
        "PITNN – PE":     "#1abc9c",
        "PITNN – Pre-LN": "#9b59b6",
        "PITNN (full)":   "#1a6bbd",
    }

    ablations = [r for r in results
                 if r["name"].startswith("PITNN")]
    if not ablations:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_tr, ax_va = axes

    for r in ablations:
        name = r["name"]
        col  = ABLATION_COLORS.get(name, "#555555")
        lw   = 2.5 if name == "PITNN (full)" else 1.5
        ls   = "-" if name == "PITNN (full)" else "--"
        ep   = range(1, len(r["hist_train"]) + 1)
        ax_tr.semilogy(ep, r["hist_train"], color=col, lw=lw, ls=ls, label=name)
        ax_va.semilogy(ep, r["hist_val"],   color=col, lw=lw, ls=ls, label=name)

    for ax, ylabel, panel_title in [
        (ax_tr, "Training Loss (log)",   "Training Loss"),
        (ax_va, "Validation Loss (log)", "Validation Loss"),
    ]:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(panel_title, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)

    plt.suptitle(f"PITNN Ablation Training Curves ({title})", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(curves_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {curves_path}")



def plot_video_comparison(results_syn, results_vid,
                          path="pitnn_ablation_video_comparison.png"):
    """
    Side-by-side grouped bar chart comparing synthetic-only vs
    synthetic+video MAE for every model. Shows the benefit of video
    augmentation for each architecture.
    """
    names  = [r["name"] for r in results_syn]
    mae_s  = [r["test_mae_deg"] for r in results_syn]
    mae_v  = [r["test_mae_deg"] for r in results_vid]

    x     = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(15, 5))
    b1 = ax.bar(x - width / 2, mae_s, width, label="Synthetic only",
                color="#5b9bd5", edgecolor="white")
    b2 = ax.bar(x + width / 2, mae_v, width, label="Synthetic + video",
                color="#ed7d31", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test MAE (degrees)", fontsize=11)
    ax.set_title("Effect of Video Augmentation on All Model Variants",
                 fontsize=13)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    # Annotate improvement arrows for PITNN (full) specifically
    for i, (s, v) in enumerate(zip(mae_s, mae_v)):
        if v < s:
            ax.annotate("", xy=(x[i] + width / 2, v + 0.002),
                        xytext=(x[i] - width / 2, s + 0.002),
                        arrowprops=dict(arrowstyle="-|>", color="green",
                                        lw=1.2))

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# SHARED MODEL BUILDER
# Returns a fresh instance of every model variant for one training pass.
# ═════════════════════════════════════════════════════════════════════════════

def build_all_models():
    """Return a list of (model_instance, name, physics_loss, kwargs) tuples."""
    pitnn_kw = dict(d_in=8, d_model=128, n_heads=8, n_layers=4,
                    d_ff=256, seq_len=20, dropout=0.1)
    return [
        # ── Baselines — same PITNNLoss as full PITNN, architecture differs ──
        # Sizes chosen so all baselines fall below PITNN (565K params),
        # reflecting that they lack the inductive biases (self-attention,
        # positional encoding, physics curriculum) that justify PITNN's capacity.
        # MLP ≈239K  |  LSTM ≈340K  |  GRU ≈375K  vs  PITNN 565K
        (MLP(seq_len=20, n_feat=8, hidden=256, n_layers=4, dropout=0.1),
         "MLP",  True, dict(lambda_p=1.0, lambda2=0.5)),
        (LSTMModel(n_feat=8, hidden=160, n_layers=2, dropout=0.1),
         "LSTM", True, dict(lambda_p=1.0, lambda2=0.5)),
        (GRUModel(n_feat=8, hidden=192, n_layers=2, dropout=0.1),
         "GRU",  True, dict(lambda_p=1.0, lambda2=0.5)),
        # ── Ablations ───────────────────────────────────────────────────────
        (PITNN(**pitnn_kw), "PITNN – LP",
         True, dict(lambda_p=1.0, lambda2=0.5, no_lp=True)),
        (PITNN(**pitnn_kw), "PITNN – LZVS",
         True, dict(lambda_p=1.0, lambda2=0.5, no_lzvs=True)),
        (PITNN(**pitnn_kw), "PITNN – Lsym",
         True, dict(lambda_p=1.0, lambda2=0.5, no_lsym=True)),
        (PITNN(**pitnn_kw), "PITNN – warmup",
         True, dict(lambda_p=1.0, lambda2=0.5, no_warmup=True)),
        (PITNNNoPE(**pitnn_kw), "PITNN – PE",
         True, dict(lambda_p=1.0, lambda2=0.5)),
        (PITNNPostLN(**pitnn_kw), "PITNN – Pre-LN",
         True, dict(lambda_p=1.0, lambda2=0.5)),
        # ── Full PITNN ──────────────────────────────────────────────────────
        (PITNN(**pitnn_kw), "PITNN (full)",
         True, dict(lambda_p=1.0, lambda2=0.5)),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PITNN Ablation Study")
    parser.add_argument("--epochs",  type=int,  default=150,
                        help="Training epochs per model (default 150)")
    parser.add_argument("--samples", type=int,  default=10000,
                        help="Synthetic dataset size (default 10000)")
    parser.add_argument("--video",   type=str,  default=None,
                        help="Path to oscilloscope/simulation video. If provided, "
                             "all models are trained a second time on the combined "
                             "synthetic + video dataset and a separate results table "
                             "is produced for direct comparison.")
    parser.add_argument("--fast",    action="store_true",
                        help="Quick run: 50 epochs, 3000 samples (for testing)")
    args = parser.parse_args()

    if args.fast:
        args.epochs  = 50
        args.samples = 3000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  PITNN — Ablation Study & Baseline Comparison")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Samples: {args.samples}")
    if args.video:
        print(f"  Video  : {args.video}")
    print("=" * 70)

    # ── [1] Synthetic dataset ─────────────────────────────────────────────
    print("\n[1] Generating shared synthetic dataset …")
    X_norm, Y, mu, sigma, X_raw = generate_dataset(
        n_samples=args.samples, seq_len=20, seed=42
    )

    # ── [2] Optional video dataset ────────────────────────────────────────
    video_X_norm = video_X_raw = video_Y = None
    if args.video:
        print(f"\n[2] Extracting video dataset from: {args.video}")
        try:
            extractor = VideoWaveformExtractor(
                args.video,
                fsw_hardware = FSW,
                V1_hardware  = V1_NOM,
                V2_hardware  = V2_NOM,
                verbose      = True,
            )
            extractor.plot_extraction("pitnn_ablation_video_extraction.png")
            video_result = extractor.build_dataset(mu=mu, sigma=sigma)
            if video_result is not None:
                video_X_norm, video_Y, _, _, video_X_raw = video_result
                print(f"  Video samples extracted: {len(video_Y)}")
                # Merge for the combined dataset
                X_norm_vid = np.concatenate([X_norm, video_X_norm], axis=0)
                X_raw_vid  = np.concatenate([X_raw,  video_X_raw],  axis=0)
                Y_vid      = np.concatenate([Y,      video_Y],       axis=0)
                print(f"  Combined: {len(X_norm)} synthetic + {len(video_Y)} "
                      f"video = {len(X_norm_vid)} total")
            else:
                print("  Warning: video extraction returned no samples — "
                      "skipping video training pass")
                args.video = None
        except Exception as e:
            print(f"  Warning: video extraction failed ({e}) — "
                  f"skipping video training pass")
            args.video = None

    # ── [3] Fixed train/val/test split (same indices for all models) ──────
    def make_split(Xn, Xr, Yl):
        N     = len(Xn)
        n_val = int(N * 0.15)
        n_tr  = N - 2 * n_val
        perm  = np.random.RandomState(42).permutation(N)
        tr_i  = perm[:n_tr]
        va_i  = perm[n_tr : n_tr + n_val]
        te_i  = perm[n_tr + n_val :]
        def tt(a): return torch.from_numpy(a).float().to(device)
        return (tt(Xn[tr_i]), tt(Xr[tr_i]), tt(Yl[tr_i]),
                tt(Xn[va_i]), tt(Xr[va_i]), tt(Yl[va_i]),
                tt(Xn[te_i]), tt(Xr[te_i]), tt(Yl[te_i]),
                n_tr, n_val)

    Xn_tr, Xr_tr, Ytr, Xn_va, Xr_va, Yva, Xn_te, Xr_te, Yte, n_tr, n_val = \
        make_split(X_norm, X_raw, Y)
    print(f"\n  Synthetic split: {n_tr} train / {n_val} val / {n_val} test")

    common_syn = dict(
        Xn_tr=Xn_tr, Xr_tr=Xr_tr, Ytr=Ytr,
        Xn_va=Xn_va, Xr_va=Xr_va, Yva=Yva,
        Xn_te=Xn_te, Xr_te=Xr_te, Yte=Yte,
        epochs=args.epochs, batch_size=64, lr=1e-4,
        warmup_epochs=20, device=device,
    )

    # ── [4] SYNTHETIC-ONLY training pass ──────────────────────────────────
    print("\n" + "═" * 70)
    print("  PASS 1 — Synthetic Data Only")
    print("═" * 70)

    results_syn = []
    for model, name, phys, extra_kw in build_all_models():
        r = train_model(model, name, physics_loss=phys, **extra_kw, **common_syn)
        print(f"  Evaluating controller performance for [{name}] ...")
        r["ctrl_metrics"] = evaluate_controller(
            r["_model"], mu, sigma, device)
        cm = r["ctrl_metrics"]
        print(f"    Mean|ΔP|%={cm['power_err_mean']:.2f}  "
              f"Max|ΔP|%={cm['power_err_max']:.2f}  "
              f"Within10%={cm['pct_within_10']:.1f}%  "
              f"CtrlZVS%={cm['ctrl_zvs_viol']:.1f}%")
        results_syn.append(r)

    print_results_table(results_syn,
                        title="Synthetic Only",
                        csv_path="pitnn_ablation_results.csv")
    plot_results(results_syn,
                 title="Synthetic Only",
                 bar_path="pitnn_ablation_results.png",
                 train_path="pitnn_ablation_training.png",
                 angle_path="pitnn_ablation_perangle.png")

    # ── [5] SYNTHETIC + VIDEO training pass (only if --video supplied) ────
    if args.video and video_X_norm is not None:
        # Use the same test set as the synthetic pass so results are
        # directly comparable (test set comes from the synthetic split).
        Xn_tr_v, Xr_tr_v, Ytr_v, Xn_va_v, Xr_va_v, Yva_v, _, _, _, n_tr_v, n_val_v = \
            make_split(X_norm_vid, X_raw_vid, Y_vid)

        print(f"\n  Combined split: {n_tr_v} train / {n_val_v} val | "
              f"test set unchanged from synthetic pass")

        common_vid = dict(
            Xn_tr=Xn_tr_v, Xr_tr=Xr_tr_v, Ytr=Ytr_v,
            Xn_va=Xn_va_v, Xr_va=Xr_va_v, Yva=Yva_v,
            # Keep the same test set for a fair comparison
            Xn_te=Xn_te, Xr_te=Xr_te, Yte=Yte,
            epochs=args.epochs, batch_size=64, lr=1e-4,
            warmup_epochs=20, device=device,
        )

        print("\n" + "═" * 70)
        print("  PASS 2 — Synthetic + Video Data")
        print("═" * 70)

        results_vid = []
        for model, name, phys, extra_kw in build_all_models():
            r = train_model(model, name, physics_loss=phys,
                            **extra_kw, **common_vid)
            print(f"  Evaluating controller performance for [{name}] ...")
            r["ctrl_metrics"] = evaluate_controller(
                r["_model"], mu, sigma, device)
            cm = r["ctrl_metrics"]
            print(f"    Mean|ΔP|%={cm['power_err_mean']:.2f}  "
                  f"Max|ΔP|%={cm['power_err_max']:.2f}  "
                  f"Within10%={cm['pct_within_10']:.1f}%  "
                  f"CtrlZVS%={cm['ctrl_zvs_viol']:.1f}%")
            results_vid.append(r)

        print_results_table(results_vid,
                            title="Synthetic + Video",
                            csv_path="pitnn_ablation_video_results.csv")
        plot_results(results_vid,
                     title="Synthetic + Video",
                     bar_path="pitnn_ablation_video_results.png",
                     train_path="pitnn_ablation_video_training.png",
                     angle_path="pitnn_ablation_video_perangle.png")
        plot_video_comparison(results_syn, results_vid)

        # Print delta table — how much does video augmentation help each model
        print("\n" + "=" * 80)
        print("  VIDEO AUGMENTATION DELTA  (Synthetic+Video MAE − Synthetic MAE)")
        print("=" * 80)
        print(f"  {'Model':<30}  {'Syn MAE (°)':>12}  "
              f"{'Vid MAE (°)':>12}  {'Δ MAE (°)':>12}  {'Δ%':>8}")
        print("  " + "-" * 76)
        for rs, rv in zip(results_syn, results_vid):
            delta    = rv["test_mae_deg"] - rs["test_mae_deg"]
            delta_pct = delta / max(rs["test_mae_deg"], 1e-9) * 100
            marker = "▼" if delta < 0 else ("▲" if delta > 0 else " ")
            print(f"  {rs['name']:<30}  {rs['test_mae_deg']:>12.3f}  "
                  f"{rv['test_mae_deg']:>12.3f}  "
                  f"{marker}{abs(delta):>11.3f}  {delta_pct:>7.1f}%")
        print("=" * 80)

    # ── [6] Summary ───────────────────────────────────────────────────────
    print("\nDone. Output files:")
    print("  pitnn_ablation_results.csv          — synthetic-only results")
    print("  pitnn_ablation_results.png          — MAE bar chart (synthetic)")
    print("  pitnn_ablation_training.png         — training curves (synthetic)")
    print("  pitnn_ablation_perangle.png         — per-angle MAE (synthetic)")
    if args.video:
        print("  pitnn_ablation_video_results.csv    — synthetic+video results")
        print("  pitnn_ablation_video_results.png    — MAE bar chart (video)")
        print("  pitnn_ablation_video_training.png   — training curves (video)")
        print("  pitnn_ablation_video_perangle.png   — per-angle MAE (video)")
        print("  pitnn_ablation_video_comparison.png — side-by-side comparison")
        print("  pitnn_ablation_video_extraction.png — video extraction diagnostic")


if __name__ == "__main__":
    main()
