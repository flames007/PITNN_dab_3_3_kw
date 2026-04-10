"""
pitnn_ablation.py
=====================================================================
Ablation Study and Lightweight Baseline Comparisons
for the PITNN DAB Converter TPS Modulation Paper.

Trains and evaluates all model variants on the same dataset split
and reports a consolidated results table suitable for a paper table.

MODELS
───────
Baselines:
  1. MLP            — flat feature vector, no sequence modelling
  2. LSTM           — recurrent sequence model, no physics loss
  3. VanillaTransformer — Transformer with no physics loss terms

Ablations (full PITNN with components removed one at a time):
  4. PITNN – LP       — no power physics loss
  5. PITNN – LZVS     — no ZVS physics loss
  6. PITNN – Lsym     — no bridge symmetry penalty
  7. PITNN – warmup   — physics loss on from epoch 1 (no curriculum)
  8. PITNN – PE       — no positional encoding
  9. PITNN – PreLN    — post-LayerNorm (standard) instead of Pre-LN

Full model:
  10. PITNN (full)    — all components enabled

RUN
────
  python pitnn_ablation.py

All models are trained on the same 10,000-sample synthetic dataset
with identical train/val/test splits (seed=42).
Results are written to:
  pitnn_ablation_results.csv   — machine-readable table
  pitnn_ablation_results.png   — bar chart comparison
  pitnn_ablation_training.png  — training curves for all models

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
    PI, PHI_MIN, PHI12_MIN, PHI12_MAX, PHI3_MAX, LK, N_TURNS, FSW, V1_NOM,
    K_POWER, B_POWER,
    # Dataset + physics
    generate_dataset, DABPhysics,
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
    def __init__(self, n_feat=8, hidden=256, n_layers=2, dropout=0.1):
        super().__init__()
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
        # x: (B, seq_len, n_feat)
        out, _ = self.lstm(x)         # out: (B, seq_len, hidden)
        raw    = self.head(out[:, -1, :])   # use last time step
        sig    = torch.sigmoid(raw)
        phi1   = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2   = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3   = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)


class VanillaTransformer(nn.Module):
    """
    Baseline 3 — Vanilla Transformer (no physics loss, standard Post-LN).
    Same architecture as PITNN but trained with pure MSE loss and using
    standard post-LayerNorm TransformerEncoderLayer. Isolates the
    contribution of the physics-informed loss and Pre-LN design.
    """
    def __init__(self, d_in=8, d_model=128, n_heads=8, n_layers=4,
                 d_ff=256, seq_len=20, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Linear(d_in, d_model)
        # Sinusoidal positional encoding (same as PITNN)
        self.pos_enc = PositionalEncoding(d_model, seq_len + 8, dropout)
        # Standard Post-LN (norm_first=False, PyTorch default)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=False,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=n_layers, enable_nested_tensor=False
        )
        self.ln_out    = nn.LayerNorm(d_model)
        self.head      = nn.Sequential(
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
# For baselines (MLP, LSTM, VanillaTransformer) physics_loss=False.
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

    # Loss function — physics terms enabled only for PITNN variants
    if physics_loss:
        loss_fn = PITNNLoss(
            lambda_p  = lambda_p if not no_lp   else 0.0,
            lambda1   = 0.0,
            lambda2   = lambda2  if not no_lzvs else 0.0,
            Lk=LK, n=N_TURNS, fsw=FSW, V_nom=V1_NOM, I_rated=100.,
        )
    else:
        loss_fn = None   # pure MSE for baselines

    print(f"\n{'─'*70}")
    print(f"  [{name}]  {n_p:,} params")
    print(f"  physics_loss={physics_loss}  "
          f"no_lp={no_lp}  no_lzvs={no_lzvs}  "
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

                loss, info = loss_fn(phi_pred, yb, V1_b, V2_b, Pref_b, iL_seq)

                # Optional ablation: zero out Lsym contribution
                if no_lsym:
                    # Recompute without symmetry term
                    loss = info["L_data"] + loss_fn.physics_weight * (
                        info["LP"] + (info["LZVS"] if not no_lzvs else 0.0)
                    )
                    if isinstance(loss, float):
                        loss = torch.tensor(loss, device=device, requires_grad=False)
            else:
                # Pure weighted MSE for baselines
                loss = torch.mean(w * (phi_pred - yb) ** 2)

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

    elapsed = time.perf_counter() - t0_total
    print(f"  ► MSE={mse:.6f}  MAE={mae:.6f} rad ({math.degrees(mae):.3f}°)  "
          f"[φ1={math.degrees(mae_phi1):.3f}°  "
          f"φ2={math.degrees(mae_phi2):.3f}°  "
          f"φ3={math.degrees(mae_phi3):.3f}°]  "
          f"time={elapsed:.1f}s")

    return {
        "name"      : name,
        "n_params"  : n_p,
        "test_mse"  : mse,
        "test_mae"  : mae,
        "test_mae_deg" : math.degrees(mae),
        "mae_phi1_deg" : math.degrees(mae_phi1),
        "mae_phi2_deg" : math.degrees(mae_phi2),
        "mae_phi3_deg" : math.degrees(mae_phi3),
        "train_time_s" : elapsed,
        "hist_train"   : hist_train,
        "hist_val"     : hist_val,
        "Y_test"       : Yte.cpu().numpy(),
        "Y_pred"       : pt.cpu().numpy(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_results_table(results):
    """Print and save the consolidated ablation results table."""
    print("\n" + "=" * 100)
    print("  ABLATION STUDY & BASELINE COMPARISON — Results Table")
    print("=" * 100)
    hdr = (f"  {'Model':<30} {'Params':>8}  {'MSE (rad²)':>12}  "
           f"{'MAE (°)':>9}  {'φ1 MAE(°)':>10}  "
           f"{'φ2 MAE(°)':>10}  {'φ3 MAE(°)':>10}  {'Time(s)':>8}")
    print(hdr)
    print("  " + "-" * 98)

    rows = []
    for r in results:
        row = (f"  {r['name']:<30} {r['n_params']:>8,}  "
               f"{r['test_mse']:>12.6f}  "
               f"{r['test_mae_deg']:>9.3f}  "
               f"{r['mae_phi1_deg']:>10.3f}  "
               f"{r['mae_phi2_deg']:>10.3f}  "
               f"{r['mae_phi3_deg']:>10.3f}  "
               f"{r['train_time_s']:>8.1f}")
        print(row)
        rows.append([
            r['name'], r['n_params'], r['test_mse'],
            r['test_mae_deg'], r['mae_phi1_deg'],
            r['mae_phi2_deg'], r['mae_phi3_deg'],
            r['train_time_s'],
        ])

    print("=" * 100)

    # Save CSV
    with open("pitnn_ablation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Params", "MSE_rad2",
                         "MAE_deg", "MAE_phi1_deg", "MAE_phi2_deg",
                         "MAE_phi3_deg", "TrainTime_s"])
        writer.writerows(rows)
    print("  Saved: pitnn_ablation_results.csv")


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(results):
    """Bar chart of MAE per model and training curves."""

    names  = [r["name"] for r in results]
    maes   = [r["test_mae_deg"] for r in results]
    colors = []
    for r in results:
        n = r["name"]
        if "PITNN (full)" in n:
            colors.append("#1a6bbd")        # blue — full model
        elif n.startswith("PITNN"):
            colors.append("#5ba3d9")        # light blue — ablations
        else:
            colors.append("#e07b39")        # orange — baselines

    # ── Bar chart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(names)), maes, color=colors, edgecolor="white",
                  linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Test MAE (degrees)", fontsize=11)
    ax.set_title("Ablation Study and Baseline Comparison — Test MAE", fontsize=13)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    # Annotate bars
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{mae:.3f}°", ha="center", va="bottom", fontsize=8)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#e07b39", label="Baselines"),
        Patch(facecolor="#5ba3d9", label="Ablations"),
        Patch(facecolor="#1a6bbd", label="Full PITNN"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("pitnn_ablation_results.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  Saved: pitnn_ablation_results.png")

    # ── Training curves ────────────────────────────────────────────────────
    n_models = len(results)
    n_cols   = 5
    n_rows   = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3),
                             sharey=False)
    axes_flat = axes.flatten() if n_models > 1 else [axes]

    for i, r in enumerate(results):
        ax = axes_flat[i]
        epochs = range(1, len(r["hist_train"]) + 1)
        ax.semilogy(epochs, r["hist_train"], label="Train", lw=1.5)
        ax.semilogy(epochs, r["hist_val"],   label="Val",   lw=1.5, ls="--")
        ax.set_title(r["name"], fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Training & Validation Loss Curves — All Models", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("pitnn_ablation_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pitnn_ablation_training.png")

    # ── Per-angle MAE breakdown ────────────────────────────────────────────
    x      = np.arange(len(names))
    width  = 0.25
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
    ax.set_title("Per-Angle MAE Breakdown — All Models", fontsize=13)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("pitnn_ablation_perangle.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  Saved: pitnn_ablation_perangle.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PITNN Ablation Study")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs per model (default 150)")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Dataset size (default 10000)")
    parser.add_argument("--fast", action="store_true",
                        help="Quick run: 50 epochs, 3000 samples (for testing)")
    args = parser.parse_args()

    if args.fast:
        args.epochs  = 50
        args.samples = 3000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  PITNN — Ablation Study & Baseline Comparison")
    print(f"  Device: {device}  |  Epochs: {args.epochs}  |  Samples: {args.samples}")
    print("=" * 70)

    # ── [1] Generate dataset (one shared dataset for all models) ─────────
    print("\n[1] Generating shared dataset …")
    X_norm, Y, mu, sigma, X_raw = generate_dataset(
        n_samples=args.samples, seq_len=20, seed=42
    )

    # ── [2] Split into train / val / test (identical across all models) ──
    N     = len(X_norm)
    n_val = int(N * 0.15)
    n_tr  = N - 2 * n_val
    perm  = np.random.RandomState(42).permutation(N)
    tr_i  = perm[:n_tr]
    va_i  = perm[n_tr : n_tr + n_val]
    te_i  = perm[n_tr + n_val :]

    def tt(a): return torch.from_numpy(a).float().to(device)

    Xn_tr, Xn_va, Xn_te = tt(X_norm[tr_i]), tt(X_norm[va_i]), tt(X_norm[te_i])
    Xr_tr, Xr_va, Xr_te = tt(X_raw[tr_i]),  tt(X_raw[va_i]),  tt(X_raw[te_i])
    Ytr,   Yva,   Yte   = tt(Y[tr_i]),       tt(Y[va_i]),      tt(Y[te_i])

    print(f"  Split: {n_tr} train / {n_val} val / {n_val} test")

    # ── Common training kwargs ────────────────────────────────────────────
    common = dict(
        Xn_tr=Xn_tr, Xr_tr=Xr_tr, Ytr=Ytr,
        Xn_va=Xn_va, Xr_va=Xr_va, Yva=Yva,
        Xn_te=Xn_te, Xr_te=Xr_te, Yte=Yte,
        epochs=args.epochs, batch_size=64, lr=1e-4,
        warmup_epochs=20, device=device,
    )

    results = []

    # ══════════════════════════════════════════════════════════════════════
    # BASELINES
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 70)
    print("  BASELINES")
    print("═" * 70)

    # 1. MLP
    results.append(train_model(
        MLP(seq_len=20, n_feat=8, hidden=512, n_layers=4, dropout=0.1),
        "MLP", physics_loss=False, **common,
    ))

    # 2. LSTM
    results.append(train_model(
        LSTMModel(n_feat=8, hidden=256, n_layers=2, dropout=0.1),
        "LSTM", physics_loss=False, **common,
    ))

    # 3. Vanilla Transformer (Post-LN, no physics loss)
    results.append(train_model(
        VanillaTransformer(d_in=8, d_model=128, n_heads=8, n_layers=4,
                           d_ff=256, seq_len=20, dropout=0.1),
        "Vanilla Transformer", physics_loss=False, **common,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # ABLATIONS  (full PITNN with one component removed at a time)
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 70)
    print("  ABLATIONS")
    print("═" * 70)

    pitnn_kwargs = dict(
        d_in=8, d_model=128, n_heads=8, n_layers=4,
        d_ff=256, seq_len=20, dropout=0.1,
    )

    # 4. PITNN without power physics loss (LP = 0)
    results.append(train_model(
        PITNN(**pitnn_kwargs), "PITNN – LP",
        physics_loss=True, lambda_p=1.0, lambda2=0.5,
        no_lp=True, **common,
    ))

    # 5. PITNN without ZVS loss (LZVS = 0)
    results.append(train_model(
        PITNN(**pitnn_kwargs), "PITNN – LZVS",
        physics_loss=True, lambda_p=1.0, lambda2=0.5,
        no_lzvs=True, **common,
    ))

    # 6. PITNN without symmetry penalty (Lsym = 0)
    results.append(train_model(
        PITNN(**pitnn_kwargs), "PITNN – Lsym",
        physics_loss=True, lambda_p=1.0, lambda2=0.5,
        no_lsym=True, **common,
    ))

    # 7. PITNN without curriculum warmup (physics loss from epoch 1)
    results.append(train_model(
        PITNN(**pitnn_kwargs), "PITNN – warmup",
        physics_loss=True, lambda_p=1.0, lambda2=0.5,
        no_warmup=True, **common,
    ))

    # 8. PITNN without positional encoding
    results.append(train_model(
        PITNNNoPE(**pitnn_kwargs), "PITNN – PE",
        physics_loss=True, lambda_p=1.0, lambda2=0.5, **common,
    ))

    # 9. PITNN with Post-LN instead of Pre-LN
    results.append(train_model(
        PITNNPostLN(**pitnn_kwargs), "PITNN – Pre-LN",
        physics_loss=True, lambda_p=1.0, lambda2=0.5, **common,
    ))

    # ══════════════════════════════════════════════════════════════════════
    # FULL PITNN
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 70)
    print("  FULL PITNN")
    print("═" * 70)

    results.append(train_model(
        PITNN(**pitnn_kwargs), "PITNN (full)",
        physics_loss=True, lambda_p=1.0, lambda2=0.5,
        warmup_epochs=20, **common,
    ))

    # ── Results ───────────────────────────────────────────────────────────
    print_results_table(results)
    plot_results(results)

    print("\nDone. Output files:")
    print("  pitnn_ablation_results.csv     — full results table")
    print("  pitnn_ablation_results.png     — MAE bar chart")
    print("  pitnn_ablation_training.png    — training curves")
    print("  pitnn_ablation_perangle.png    — per-angle MAE breakdown")


if __name__ == "__main__":
    main()
