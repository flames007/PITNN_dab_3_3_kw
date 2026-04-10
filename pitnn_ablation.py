"""
pitnn_ablation.py
=====================================================================
PITNN Ablation Study
for the PITNN DAB Converter TPS Modulation Paper.

Trains the full PITNN and six ablated variants on the same dataset
split. Each variant removes exactly one design component so its
individual contribution can be quantified.

VARIANTS
─────────
  1. PITNN – LP       no power physics loss (LP = 0)
  2. PITNN – LZVS     no ZVS soft constraint (LZVS = 0)
  3. PITNN – Lsym     no bridge symmetry penalty (Lsym = 0)
  4. PITNN – warmup   physics loss active from epoch 1 (no curriculum)
  5. PITNN – PE       no sinusoidal positional encoding
  6. PITNN – Pre-LN   Post-LayerNorm instead of Pre-LayerNorm
  7. PITNN (full)     all components enabled  <- reference

RUN
----
  # Synthetic data only:
  python pitnn_ablation.py

  # With oscilloscope video (trains all variants twice and reports delta):
  python pitnn_ablation.py --video path/to/scope.mp4

  # Quick sanity check (50 epochs, 3000 samples):
  python pitnn_ablation.py --fast

OUTPUT FILES
-------------
  pitnn_ablation_results.csv        -- results table (synthetic only)
  pitnn_ablation_results.png        -- horizontal MAE bar chart
  pitnn_ablation_curves.png         -- overlaid train/val loss curves
  pitnn_ablation_perangle.png       -- per-angle MAE breakdown
  pitnn_ablation_parity.png         -- predicted vs optimal scatter
  pitnn_ablation_video_results.csv  -- results table (synthetic + video)
  pitnn_ablation_video_results.png  -- MAE bar chart (video)
  pitnn_ablation_video_curves.png   -- overlaid curves (video)
  pitnn_ablation_video_perangle.png -- per-angle MAE (video)
  pitnn_ablation_video_parity.png   -- parity plots (video)
  pitnn_ablation_video_comparison.png -- side-by-side delta chart

Copyright (c) 2026 Chukwuemeka Nzeadibe
Mississippi State University -- All Rights Reserved
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
    PI, PHI_MIN, PHI12_MIN, PHI12_MAX, PHI3_MAX, LK, N_TURNS, FSW, V1_NOM,
    K_POWER, B_POWER,
    generate_dataset, DABPhysics,
    VideoWaveformExtractor,
    PITNN, PositionalEncoding, PITNNLoss,
)


# =============================================================================
# ABLATION ARCHITECTURES
# Each class removes exactly one component from the full PITNN.
# =============================================================================

class PITNNNoPE(nn.Module):
    """
    PITNN without positional encoding.
    Identical to PITNN but the sinusoidal PE step is skipped entirely.
    The embed output is passed directly to the TransformerEncoder with
    only a dropout applied, so the model has no information about where
    in the 20-step sequence each state vector sits.
    Isolates the contribution of temporal position information.
    """
    def __init__(self, d_in=8, d_model=128, n_heads=8, n_layers=4,
                 d_ff=256, seq_len=20, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Linear(d_in, d_model)
        self.drop    = nn.Dropout(dropout)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=n_layers, enable_nested_tensor=False)
        self.ln_out = nn.LayerNorm(d_model)
        self.head   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, 3),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        z   = self.ln_out(self.transformer(self.drop(self.embed(x))))
        raw = self.head(z[:, -1, :])
        sig = torch.sigmoid(raw)
        phi1 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 0:1]
        phi2 = PHI12_MIN + (PHI12_MAX - PHI12_MIN) * sig[:, 1:2]
        phi3 = PHI_MIN   + (PHI3_MAX  - PHI_MIN)   * sig[:, 2:3]
        return torch.cat([phi1, phi2, phi3], dim=1)


class PITNNPostLN(nn.Module):
    """
    PITNN with Post-LayerNorm instead of Pre-LayerNorm.
    All other components -- architecture, loss, training -- identical to
    the full PITNN. Isolates the training-stability benefit of Pre-LN.
    """
    def __init__(self, d_in=8, d_model=128, n_heads=8, n_layers=4,
                 d_ff=256, seq_len=20, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Linear(d_in, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len + 8, dropout)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=False,  # Post-LN
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=n_layers, enable_nested_tensor=False)
        self.ln_out = nn.LayerNorm(d_model)
        self.head   = nn.Sequential(
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


# =============================================================================
# MODEL REGISTRY
# =============================================================================

def build_ablation_variants():
    """
    Returns a fresh list of (model, name, extra_kwargs) for every ablation
    variant plus the full PITNN. Called once per training pass so each pass
    gets brand-new randomly-initialised models.
    """
    kw   = dict(d_in=8, d_model=128, n_heads=8, n_layers=4,
                d_ff=256, seq_len=20, dropout=0.1)
    base = dict(lambda_p=1.0, lambda2=0.5)

    return [
        (PITNN(**kw),       "PITNN - LP",      {**base, "no_lp":     True}),
        (PITNN(**kw),       "PITNN - LZVS",    {**base, "no_lzvs":   True}),
        (PITNN(**kw),       "PITNN - Lsym",    {**base, "no_lsym":   True}),
        (PITNN(**kw),       "PITNN - warmup",  {**base, "no_warmup": True}),
        (PITNNNoPE(**kw),   "PITNN - PE",      {**base}),
        (PITNNPostLN(**kw), "PITNN - Pre-LN",  {**base}),
        (PITNN(**kw),       "PITNN (full)",    {**base}),
    ]


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model(
    model, name,
    Xn_tr, Xr_tr, Ytr,
    Xn_va, Xr_va, Yva,
    Xn_te, Xr_te, Yte,
    epochs=150, batch_size=64, lr=1e-4, warmup_epochs=20,
    lambda_p=1.0, lambda2=0.5,
    no_warmup=False,
    no_lp=False,
    no_lzvs=False,
    no_lsym=False,
    device="cpu",
):
    model  = model.to(device)
    n_p    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loader = DataLoader(TensorDataset(Xn_tr, Xr_tr, Ytr),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr/20)
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
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss_fn.physics_weight = (1.0 if no_warmup else
                                  min(1.0, (epoch - 1) / max(warmup_epochs, 1)))

        model.train()
        ep_loss, nb = 0.0, 0

        for xn_b, xr_b, yb in loader:
            opt.zero_grad()
            phi_pred = model(xn_b)
            V1_b, V2_b = xr_b[:, -1, 0], xr_b[:, -1, 1]
            Pref_b     = xr_b[:, -1, 6]
            iL_seq     = xr_b[:, :, 2]

            if no_lsym:
                # Recompute loss from tensors to preserve the autograd graph.
                # The info dict from loss_fn contains .item() floats which have
                # no grad_fn and cannot be backpropagated through.
                w_mse  = torch.tensor([2., 2., 3.], device=phi_pred.device)
                L_data = torch.mean(w_mse * (phi_pred - yb) ** 2)
                phi1_t = phi_pred[:, 0]
                phi3_t = phi_pred[:, 2]
                scale  = N_TURNS * V1_b * V2_b / (V1_NOM ** 2)
                P_pred = scale * K_POWER * (phi1_t / PI) * phi3_t * (
                         1.0 - phi3_t / B_POWER)
                LP     = torch.mean((P_pred - Pref_b) ** 2 / (Pref_b ** 2 + 1.0))
                if no_lzvs:
                    L_phys = LP
                else:
                    d1     = phi_pred[:, 0] / PI
                    i0     = (V1_b * d1 - N_TURNS * V2_b * phi3_t / PI) / (
                             2.0 * LK * FSW)
                    LZVS   = torch.mean(torch.clamp(-i0, min=0.0) / 100.0)
                    L_phys = LP + loss_fn.lambda2 * LZVS
                loss = L_data + loss_fn.physics_weight * loss_fn.lambda_p * L_phys
            else:
                loss, _ = loss_fn(phi_pred, yb, V1_b, V2_b, Pref_b, iL_seq)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1

        sched.step()
        model.eval()
        with torch.no_grad():
            pv       = model(Xn_va)
            val_loss = torch.mean(w * (pv - Yva) ** 2).item()

        hist_train.append(ep_loss / max(nb, 1))
        hist_val.append(val_loss)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch in (1, epochs):
            print(f"  Ep {epoch:>4}/{epochs}  "
                  f"train={ep_loss/max(nb,1):.5f}  val={val_loss:.5f}")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pt       = model(Xn_te)
        mse      = nn.functional.mse_loss(pt, Yte).item()
        mae      = (pt - Yte).abs().mean().item()
        mae_phi1 = (pt[:, 0] - Yte[:, 0]).abs().mean().item()
        mae_phi2 = (pt[:, 1] - Yte[:, 1]).abs().mean().item()
        mae_phi3 = (pt[:, 2] - Yte[:, 2]).abs().mean().item()

    elapsed = time.perf_counter() - t0
    print(f"  > MSE={mse:.6f}  MAE={mae:.6f} rad ({math.degrees(mae):.3f} deg)  "
          f"[phi1={math.degrees(mae_phi1):.3f} deg  "
          f"phi2={math.degrees(mae_phi2):.3f} deg  "
          f"phi3={math.degrees(mae_phi3):.3f} deg]  time={elapsed:.1f}s")

    return {
        "name"         : name,
        "n_params"     : n_p,
        "test_mse"     : mse,
        "test_mae"     : mae,
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


# =============================================================================
# RESULTS TABLE
# =============================================================================

def print_results_table(results, title="Synthetic Only",
                        csv_path="pitnn_ablation_results.csv"):
    print("\n" + "=" * 100)
    print(f"  PITNN ABLATION STUDY -- {title}")
    print("=" * 100)
    print(f"  {'Model':<26} {'Params':>8}  {'MSE (rad2)':>12}  "
          f"{'MAE (deg)':>10}  {'phi1 MAE':>10}  "
          f"{'phi2 MAE':>10}  {'phi3 MAE':>10}  {'Time(s)':>8}")
    print("  " + "-" * 96)

    rows = []
    for r in results:
        marker = "  <- best" if r["name"] == "PITNN (full)" else ""
        print(f"  {r['name']:<26} {r['n_params']:>8,}  "
              f"{r['test_mse']:>12.6f}  "
              f"{r['test_mae_deg']:>10.3f}  "
              f"{r['mae_phi1_deg']:>10.3f}  "
              f"{r['mae_phi2_deg']:>10.3f}  "
              f"{r['mae_phi3_deg']:>10.3f}  "
              f"{r['train_time_s']:>8.1f}{marker}")
        rows.append([r['name'], r['n_params'], r['test_mse'],
                     r['test_mae_deg'], r['mae_phi1_deg'],
                     r['mae_phi2_deg'], r['mae_phi3_deg'],
                     r['train_time_s']])

    print("=" * 100)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(
            [["Model", "Params", "MSE_rad2", "MAE_deg",
              "MAE_phi1_deg", "MAE_phi2_deg", "MAE_phi3_deg", "TrainTime_s"]]
            + rows)
    print(f"  Saved: {csv_path}")


# =============================================================================
# PLOTS
# Consistent colour and line-style per variant across all figures.
# =============================================================================

VARIANT_STYLE = {
    "PITNN - LP":      {"color": "#e74c3c", "ls": "--",  "lw": 1.6},
    "PITNN - LZVS":    {"color": "#e67e22", "ls": "--",  "lw": 1.6},
    "PITNN - Lsym":    {"color": "#f1c40f", "ls": "--",  "lw": 1.6},
    "PITNN - warmup":  {"color": "#2ecc71", "ls": "--",  "lw": 1.6},
    "PITNN - PE":      {"color": "#1abc9c", "ls": "-.",  "lw": 1.6},
    "PITNN - Pre-LN":  {"color": "#9b59b6", "ls": "-.",  "lw": 1.6},
    "PITNN (full)":    {"color": "#1a6bbd", "ls": "-",   "lw": 2.8},
}

def _style(name):
    return VARIANT_STYLE.get(name, {"color": "#555555", "ls": "-", "lw": 1.5})


def plot_bar(results, title, path):
    """Horizontal MAE bar chart — full PITNN highlighted with a reference line."""
    names  = [r["name"] for r in results]
    maes   = [r["test_mae_deg"] for r in results]
    colors = [_style(n)["color"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(names)), maes, color=colors,
                   edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Test MAE (degrees)", fontsize=11)
    ax.set_title(f"PITNN Ablation Study -- Test MAE ({title})", fontsize=12)
    ax.xaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{mae:.3f} deg", va="center", fontsize=8.5)

    full_mae = next((r["test_mae_deg"] for r in results
                     if r["name"] == "PITNN (full)"), None)
    if full_mae is not None:
        ax.axvline(full_mae, color="#1a6bbd", lw=1.2, ls=":",
                   label=f"Full PITNN ({full_mae:.3f} deg)")
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_curves(results, title, path):
    """
    Overlaid train and validation loss curves for all variants on one figure.
    Left panel = training loss, right panel = validation loss.
    Full PITNN is drawn thicker so it stands out as the reference baseline.
    """
    fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        st = _style(r["name"])
        ep = range(1, len(r["hist_train"]) + 1)
        ax_tr.semilogy(ep, r["hist_train"], label=r["name"],
                       color=st["color"], ls=st["ls"], lw=st["lw"])
        ax_va.semilogy(ep, r["hist_val"],   label=r["name"],
                       color=st["color"], ls=st["ls"], lw=st["lw"])

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

    plt.suptitle(f"PITNN Ablation -- Training & Validation Loss ({title})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_perangle(results, title, path):
    """Grouped bar chart of per-angle MAE for every variant."""
    names = [r["name"] for r in results]
    x     = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, [r["mae_phi1_deg"] for r in results], width,
           label="phi1 MAE", color="#2196F3", edgecolor="white")
    ax.bar(x,          [r["mae_phi2_deg"] for r in results], width,
           label="phi2 MAE", color="#4CAF50", edgecolor="white")
    ax.bar(x + width,  [r["mae_phi3_deg"] for r in results], width,
           label="phi3 MAE", color="#FF5722", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("MAE (degrees)", fontsize=11)
    ax.set_title(f"Per-Angle MAE Breakdown ({title})", fontsize=12)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_parity(results, title, path):
    """
    Predicted vs optimal scatter for all variants.
    Layout: one row per angle (phi1, phi2, phi3), one column per variant.
    Each panel title includes the per-angle MAE.
    """
    angle_labels = ["phi1 (rad)", "phi2 (rad)", "phi3 (rad)"]
    n_var = len(results)
    n_ang = 3

    fig, axes = plt.subplots(n_ang, n_var,
                             figsize=(3.8 * n_var, 3.8 * n_ang),
                             squeeze=False)

    for col, r in enumerate(results):
        Yt        = r["Y_test"]
        Yp        = r["Y_pred"]
        col_color = _style(r["name"])["color"]

        for row in range(n_ang):
            ax = axes[row][col]
            ax.scatter(Yt[:, row], Yp[:, row],
                       alpha=0.3, s=5, color=col_color, rasterized=True)
            lo = min(Yt[:, row].min(), Yp[:, row].min())
            hi = max(Yt[:, row].max(), Yp[:, row].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2)
            mae_deg = math.degrees(abs(Yt[:, row] - Yp[:, row]).mean())
            ax.set_title(f"{r['name']}\n{angle_labels[row]}"
                         f"  MAE={mae_deg:.3f} deg", fontsize=7.5)
            ax.set_xlabel("Optimal (rad)", fontsize=7)
            ax.set_ylabel("Predicted (rad)", fontsize=7)
            ax.tick_params(labelsize=6)

    plt.suptitle(f"Parity Plots -- Predicted vs Optimal ({title})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_video_comparison(results_syn, results_vid,
                          path="pitnn_ablation_video_comparison.png"):
    """
    Grouped bar chart comparing synthetic-only vs synthetic+video MAE
    for every ablation variant. Green arrows highlight improvements.
    """
    names = [r["name"] for r in results_syn]
    mae_s = [r["test_mae_deg"] for r in results_syn]
    mae_v = [r["test_mae_deg"] for r in results_vid]
    x     = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width / 2, mae_s, width, label="Synthetic only",
           color="#5b9bd5", edgecolor="white")
    ax.bar(x + width / 2, mae_v, width, label="Synthetic + video",
           color="#ed7d31", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Test MAE (degrees)", fontsize=11)
    ax.set_title("Effect of Video Augmentation -- All Ablation Variants",
                 fontsize=12)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)

    for i, (s, v) in enumerate(zip(mae_s, mae_v)):
        if v < s:
            ax.annotate("", xy=(x[i] + width / 2, v + 0.002),
                        xytext=(x[i] - width / 2, s + 0.002),
                        arrowprops=dict(arrowstyle="-|>", color="green", lw=1.2))

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run_pass(variants, common_kw, label, pfx):
    """Train all variants and save all plots for one data pass."""
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)

    results = []
    for model, name, extra_kw in variants:
        results.append(train_model(model, name, **extra_kw, **common_kw))

    print_results_table(results, title=label, csv_path=f"{pfx}_results.csv")
    plot_bar(results,      title=label, path=f"{pfx}_results.png")
    plot_curves(results,   title=label, path=f"{pfx}_curves.png")
    plot_perangle(results, title=label, path=f"{pfx}_perangle.png")
    plot_parity(results,   title=label, path=f"{pfx}_parity.png")
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PITNN Ablation Study")
    parser.add_argument("--epochs",  type=int, default=150,
                        help="Training epochs per variant (default 150)")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Synthetic dataset size (default 10000)")
    parser.add_argument("--video",   type=str, default=None,
                        help="Oscilloscope video path -- enables synthetic+video pass")
    parser.add_argument("--fast",    action="store_true",
                        help="50 epochs, 3000 samples (quick test)")
    args = parser.parse_args()

    if args.fast:
        args.epochs  = 50
        args.samples = 3000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("  PITNN Ablation Study")
    print(f"  Device : {device}  |  Epochs : {args.epochs}  |"
          f"  Samples: {args.samples}")
    if args.video:
        print(f"  Video  : {args.video}")
    print("=" * 70)

    # ── [1] Synthetic dataset ─────────────────────────────────────────────
    print("\n[1] Generating synthetic dataset ...")
    X_norm, Y, mu, sigma, X_raw = generate_dataset(
        n_samples=args.samples, seq_len=20, seed=42)

    # ── [2] Optional video dataset ────────────────────────────────────────
    X_norm_vid = X_raw_vid = Y_vid = None
    if args.video:
        print(f"\n[2] Extracting video dataset from: {args.video}")
        try:
            ext = VideoWaveformExtractor(
                args.video, fsw_hardware=FSW,
                V1_hardware=V1_NOM, V2_hardware=V1_NOM, verbose=True)
            ext.plot_extraction("pitnn_ablation_video_extraction.png")
            vr = ext.build_dataset(mu=mu, sigma=sigma)
            if vr is not None:
                v_Xn, v_Y, _, _, v_Xr = vr
                print(f"  Video samples: {len(v_Y)}")
                X_norm_vid = np.concatenate([X_norm, v_Xn], axis=0)
                X_raw_vid  = np.concatenate([X_raw,  v_Xr], axis=0)
                Y_vid      = np.concatenate([Y,      v_Y],  axis=0)
                print(f"  Combined: {len(X_norm)} + {len(v_Y)} = {len(X_norm_vid)}")
            else:
                print("  Warning: no video samples extracted -- skipping video pass")
                args.video = None
        except Exception as e:
            print(f"  Warning: video extraction failed ({e}) -- skipping")
            args.video = None

    # ── [3] Fixed train/val/test split ────────────────────────────────────
    def make_split(Xn, Xr, Yl):
        N     = len(Xn)
        n_val = int(N * 0.15)
        n_tr  = N - 2 * n_val
        perm  = np.random.RandomState(42).permutation(N)
        ti = perm[:n_tr]
        vi = perm[n_tr : n_tr + n_val]
        ei = perm[n_tr + n_val :]
        tt = lambda a: torch.from_numpy(a).float().to(device)
        return (tt(Xn[ti]), tt(Xr[ti]), tt(Yl[ti]),
                tt(Xn[vi]), tt(Xr[vi]), tt(Yl[vi]),
                tt(Xn[ei]), tt(Xr[ei]), tt(Yl[ei]),
                n_tr, n_val)

    Xn_tr,Xr_tr,Ytr, Xn_va,Xr_va,Yva, Xn_te,Xr_te,Yte, n_tr, n_val = \
        make_split(X_norm, X_raw, Y)
    print(f"\n  Split: {n_tr} train / {n_val} val / {n_val} test")

    common_syn = dict(
        Xn_tr=Xn_tr, Xr_tr=Xr_tr, Ytr=Ytr,
        Xn_va=Xn_va, Xr_va=Xr_va, Yva=Yva,
        Xn_te=Xn_te, Xr_te=Xr_te, Yte=Yte,
        epochs=args.epochs, batch_size=64, lr=1e-4,
        warmup_epochs=20, device=device,
    )

    # ── [4] Synthetic-only pass ───────────────────────────────────────────
    results_syn = run_pass(
        build_ablation_variants(), common_syn,
        label="Synthetic Only", pfx="pitnn_ablation",
    )

    # ── [5] Synthetic + video pass ────────────────────────────────────────
    if args.video and X_norm_vid is not None:
        Xn_tr_v,Xr_tr_v,Ytr_v, Xn_va_v,Xr_va_v,Yva_v, _,_,_, n_tr_v, n_val_v = \
            make_split(X_norm_vid, X_raw_vid, Y_vid)
        print(f"\n  Video split: {n_tr_v} train / {n_val_v} val"
              f"  |  test set shared with synthetic pass")

        common_vid = dict(
            Xn_tr=Xn_tr_v, Xr_tr=Xr_tr_v, Ytr=Ytr_v,
            Xn_va=Xn_va_v, Xr_va=Xr_va_v, Yva=Yva_v,
            Xn_te=Xn_te,   Xr_te=Xr_te,   Yte=Yte,    # same test set for fairness
            epochs=args.epochs, batch_size=64, lr=1e-4,
            warmup_epochs=20, device=device,
        )

        results_vid = run_pass(
            build_ablation_variants(), common_vid,
            label="Synthetic + Video", pfx="pitnn_ablation_video",
        )

        plot_video_comparison(results_syn, results_vid)

        # Delta table
        print("\n" + "=" * 80)
        print("  VIDEO AUGMENTATION DELTA  (Synthetic+Video minus Synthetic)")
        print("=" * 80)
        print(f"  {'Variant':<26}  {'Syn MAE(deg)':>13}  "
              f"{'Vid MAE(deg)':>13}  {'Delta(deg)':>12}  {'Delta%':>8}")
        print("  " + "-" * 74)
        for rs, rv in zip(results_syn, results_vid):
            d    = rv["test_mae_deg"] - rs["test_mae_deg"]
            dpct = d / max(rs["test_mae_deg"], 1e-9) * 100
            mark = "v" if d < 0 else ("^" if d > 0 else " ")
            print(f"  {rs['name']:<26}  {rs['test_mae_deg']:>13.3f}  "
                  f"{rv['test_mae_deg']:>13.3f}  "
                  f"{mark}{abs(d):>11.3f}  {dpct:>7.1f}%")
        print("=" * 80)

    # ── [6] Summary ───────────────────────────────────────────────────────
    print("\nDone. Output files:")
    for f in ["pitnn_ablation_results.csv",
              "pitnn_ablation_results.png",
              "pitnn_ablation_curves.png",
              "pitnn_ablation_perangle.png",
              "pitnn_ablation_parity.png"]:
        print(f"  {f}")
    if args.video:
        for f in ["pitnn_ablation_video_results.csv",
                  "pitnn_ablation_video_results.png",
                  "pitnn_ablation_video_curves.png",
                  "pitnn_ablation_video_perangle.png",
                  "pitnn_ablation_video_parity.png",
                  "pitnn_ablation_video_comparison.png",
                  "pitnn_ablation_video_extraction.png"]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
