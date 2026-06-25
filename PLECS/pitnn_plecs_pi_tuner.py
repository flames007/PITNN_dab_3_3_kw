"""
pitnn_plecs_pi_tuner.py
=====================================================================
PI Gain Tuner for the PITNN+PI Power Controller (Option B)

Computes recommended Kp and Ki gains based on:
  - The PITNN's measured power tracking error characteristics
  - The converter's power-to-angle small-signal gain
  - The desired closed-loop bandwidth and phase margin

Also runs a discrete-time step response simulation to validate
stability before connecting to PLECS.

Usage:
    python pitnn_plecs_pi_tuner.py
    python pitnn_plecs_pi_tuner.py --checkpoint pitnn_dab_checkpoint.pt
    python pitnn_plecs_pi_tuner.py --kp 0.05 --ki 2.0 --plot

Converter spec:
    V1=400V, V2=250V, P_rated=3.3kW, fsw=100kHz
    Controller update period: dt = 50/100e3 = 500µs
=====================================================================
"""

import argparse
import math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONVERTER PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
V1_NOM   = 400.0
V2_NOM   = 250.0
FSW      = 100e3
LK       = 40e-6
N_TURNS  = 1.6
P_RATED  = 3300.0
P_MIN    = 500.0
DT       = 50.0 / FSW          # 500µs controller update period
PI_CONST = math.pi


# ─────────────────────────────────────────────────────────────────────────────
# SMALL-SIGNAL POWER GAIN  dP/dP_ref
# ─────────────────────────────────────────────────────────────────────────────

def compute_plant_gain(P_op: float = 1650.0) -> float:
    """
    Estimate the small-signal DC gain of the PITNN+converter plant:
        G = dP_delivered / dP_ref

    At steady state the PITNN tracks power with ~2% mean error,
    so G ≈ 0.98. The gain is slightly load-dependent but constant
    enough for PI design purposes.

    A more precise estimate perturbs P_ref by ±5% and measures
    the output power change using the DABPhysics model.
    """
    try:
        from pitnn_dab import DABPhysics
        dab = DABPhysics()
        dP  = P_op * 0.05    # 5% perturbation
        phi_lo = dab.solve_optimal_phi(P_op - dP)
        phi_hi = dab.solve_optimal_phi(P_op + dP)
        P_lo   = dab.compute_power(*phi_lo)
        P_hi   = dab.compute_power(*phi_hi)
        G = (P_hi - P_lo) / (2.0 * dP)
        return float(G)
    except Exception:
        return 0.98    # fallback if pitnn_dab not available


# ─────────────────────────────────────────────────────────────────────────────
# GAIN DESIGN
# ─────────────────────────────────────────────────────────────────────────────

def design_pi_gains(G: float,
                    bw_hz: float = 20.0,
                    phase_margin_deg: float = 60.0) -> dict:
    """
    Design PI gains for a given plant gain G and desired bandwidth.

    Uses the standard PI tuning formula for a unity-feedback system
    with a first-order plant approximation:

        C(z) = Kp + Ki*dt / (1 - z^-1)
        G_plant(s) ≈ G / (tau*s + 1)

    where tau is estimated from the PITNN's settling behaviour
    (~10 update steps → tau ≈ 10 * dt = 5ms).

    Parameters
    ----------
    G               : Plant DC gain (dP_out / dP_ref ≈ 0.98)
    bw_hz           : Desired closed-loop bandwidth (Hz)
    phase_margin_deg: Desired phase margin (degrees)

    Returns
    -------
    dict with Kp, Ki, tau, bw_hz, phase_margin_deg
    """
    omega_c = 2.0 * math.pi * bw_hz    # desired crossover (rad/s)
    tau     = 10.0 * DT                 # estimated plant time constant (5ms)

    # At crossover, |G_plant(j*omega_c)| * |C(j*omega_c)| = 1
    # For PI: |C(j*omega)| ≈ Kp at frequencies >> Ki/Kp
    # So: Kp ≈ 1 / (G * |1 / (j*omega_c*tau + 1)|)
    plant_mag = G / math.sqrt(1.0 + (omega_c * tau) ** 2)
    Kp = 1.0 / plant_mag

    # Ki placed one decade below crossover for phase margin
    # omega_i = omega_c / 10  →  Ki = Kp * omega_i
    omega_i = omega_c / 10.0
    Ki = Kp * omega_i

    # Anti-windup limit: integral clamped to ±(P_MAX-P_MIN)/2
    int_max = (P_RATED - P_MIN) / 2.0 / max(Ki * DT, 1e-9)

    return {
        "Kp"             : round(Kp, 4),
        "Ki"             : round(Ki, 4),
        "tau_ms"         : tau * 1e3,
        "bw_hz"          : bw_hz,
        "phase_margin_deg": phase_margin_deg,
        "omega_c_rad_s"  : omega_c,
        "int_max_W"      : int_max,
        "plant_gain_G"   : G,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISCRETE-TIME STEP RESPONSE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_step_response(Kp: float,
                            Ki: float,
                            P_steps: list = None,
                            n_steps_each: int = 60,
                            G: float = 0.98,
                            tau: float = None,
                            noise_std: float = 10.0) -> dict:
    """
    Simulate the closed-loop PI+PITNN step response in discrete time.

    The PITNN+converter plant is approximated as a first-order system:
        P_out[k] = (1 - alpha)*P_out[k-1] + alpha*G*P_ref_corr[k-1]
    where alpha = DT/tau.

    Parameters
    ----------
    Kp, Ki       : PI gains to evaluate
    P_steps      : List of power reference steps (W)
    n_steps_each : Number of controller updates per step
    G            : Plant DC gain
    tau          : Plant time constant (s); defaults to 10*DT
    noise_std    : Measurement noise standard deviation (W)

    Returns
    -------
    dict with time, P_ref, P_out, P_corr, error arrays
    """
    if P_steps is None:
        P_steps = [500.0, 1650.0, 3300.0, 1650.0, 500.0]
    if tau is None:
        tau = 10.0 * DT

    alpha  = DT / tau
    int_max= (P_RATED - P_MIN) / 2.0 / max(Ki * DT, 1e-9)

    rng = np.random.default_rng(42)

    N = len(P_steps) * n_steps_each
    t_arr     = np.zeros(N)
    Pref_arr  = np.zeros(N)
    Pcorr_arr = np.zeros(N)
    Pout_arr  = np.zeros(N)
    Perr_arr  = np.zeros(N)

    P_out   = P_steps[0]
    integ   = 0.0
    prev_ref= P_steps[0]
    k = 0

    for P_ref_ext in P_steps:
        # Reset integrator on step
        if abs(P_ref_ext - prev_ref) > 100.0:
            integ = 0.0
        prev_ref = P_ref_ext

        for _ in range(n_steps_each):
            t = k * DT
            # Measurement noise
            P_meas = P_out + rng.normal(0, noise_std)
            P_meas = max(0.0, P_meas)

            # PI correction
            error  = P_ref_ext - P_meas
            integ  = float(np.clip(integ + error * DT, -int_max, int_max))
            P_corr = float(np.clip(P_ref_ext + Kp*error + Ki*integ,
                                   P_MIN, P_RATED))

            # Plant update (first-order approximation)
            P_out = (1.0 - alpha) * P_out + alpha * G * P_corr

            t_arr[k]     = t * 1e3   # ms
            Pref_arr[k]  = P_ref_ext
            Pcorr_arr[k] = P_corr
            Pout_arr[k]  = P_out
            Perr_arr[k]  = abs(P_out - P_ref_ext) / max(P_ref_ext, 1) * 100
            k += 1

    return {
        "t_ms"   : t_arr,
        "P_ref"  : Pref_arr / 1000,    # kW
        "P_out"  : Pout_arr / 1000,
        "P_corr" : Pcorr_arr / 1000,
        "P_err"  : Perr_arr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_step_response(sim: dict, Kp: float, Ki: float,
                       save_path: str = "pitnn_pi_step_response.png"):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].step(sim["t_ms"], sim["P_ref"],  "k--", lw=1.8,
                 label="P_ref (ext)", where="post")
    axes[0].plot(sim["t_ms"], sim["P_out"],  "b-",  lw=2.0,
                 label="P_out (delivered)")
    axes[0].plot(sim["t_ms"], sim["P_corr"], "g:",  lw=1.4,
                 label="P_corr (PI output)")
    axes[0].set_ylabel("Power (kW)", fontsize=11)
    axes[0].set_title(
        f"PITNN+PI Closed-Loop Step Response  —  Kp={Kp}  Ki={Ki}\n"
        f"V1={V1_NOM:.0f}V / V2={V2_NOM:.0f}V / fsw=100kHz / "
        f"dt={DT*1e6:.0f}µs / P_rated={P_RATED:.0f}W",
        fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, ls="--")
    axes[0].set_ylim(bottom=0)

    axes[1].plot(sim["t_ms"], sim["P_err"], "r-", lw=1.8)
    axes[1].axhline(10, color="k", ls="--", lw=1.2, label="10% threshold")
    axes[1].fill_between(sim["t_ms"], sim["P_err"], 0,
                         where=np.array(sim["P_err"]) <= 10,
                         alpha=0.15, color="blue", label="Within 10%")
    axes[1].fill_between(sim["t_ms"], sim["P_err"], 0,
                         where=np.array(sim["P_err"]) > 10,
                         alpha=0.25, color="red", label=">10% error")
    axes[1].set_ylabel("|ΔP| / P_ref (%)", fontsize=11)
    axes[1].set_xlabel("Time (ms)", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, ls="--")
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def check_stability(Kp: float, Ki: float, G: float = 0.98) -> dict:
    """
    Check discrete-time closed-loop stability by computing the
    characteristic equation poles.

    For a PI controller with first-order plant:
        Open-loop pulse transfer function:
        L(z) = C(z) * G_plant(z)

    The closed-loop poles must lie inside the unit circle.
    """
    tau   = 10.0 * DT
    alpha = DT / tau

    # Discrete plant: G_plant(z) = alpha*G / (z - (1-alpha))
    # PI: C(z) = (Kp*(z-1) + Ki*DT*z) / (z-1)
    #          = ((Kp + Ki*DT)*z - Kp) / (z-1)

    # Closed-loop characteristic polynomial coefficients
    # z^2 + a1*z + a0 = 0
    a1 = -(2.0 - alpha - alpha*G*(Kp + Ki*DT))
    a0 = (1.0 - alpha + alpha*G*Kp)

    disc = a1**2 - 4.0*a0
    if disc >= 0:
        r1 = (-a1 + math.sqrt(disc)) / 2.0
        r2 = (-a1 - math.sqrt(disc)) / 2.0
        poles = [r1, r2]
        stable = all(abs(p) < 1.0 for p in poles)
    else:
        mag = math.sqrt(a0)
        poles = [complex(-a1/2, math.sqrt(-disc)/2),
                 complex(-a1/2, -math.sqrt(-disc)/2)]
        stable = mag < 1.0

    return {
        "stable"     : stable,
        "poles"      : poles,
        "pole_mags"  : [abs(p) for p in poles],
        "gain_margin": 1.0 / (G * Kp) if Kp > 0 else float("inf"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PI gain tuner for PITNN+PI power controller (Option B)")
    parser.add_argument("--kp",   type=float, default=None,
                        help="Override Kp (default: auto-computed)")
    parser.add_argument("--ki",   type=float, default=None,
                        help="Override Ki (default: auto-computed)")
    parser.add_argument("--bw",   type=float, default=20.0,
                        help="Desired closed-loop bandwidth Hz (default: 20)")
    parser.add_argument("--plot", action="store_true",
                        help="Save step response plot")
    args = parser.parse_args()

    print("=" * 65)
    print("  PITNN+PI Power Controller — Gain Tuner (Option B)")
    print(f"  Converter: {V1_NOM:.0f}V/{V2_NOM:.0f}V  "
          f"P_rated={P_RATED:.0f}W  dt={DT*1e6:.0f}µs")
    print("=" * 65)

    # Step 1: Plant gain
    print("\n[1] Estimating plant gain ...")
    G = compute_plant_gain(P_op=1650.0)
    print(f"  G = dP_out/dP_ref ≈ {G:.4f}  "
          f"({'from DABPhysics' if abs(G-0.98)>0.001 else 'fallback estimate'})")

    # Step 2: Gain design
    print(f"\n[2] Designing PI gains  (target bandwidth: {args.bw:.0f} Hz) ...")
    design = design_pi_gains(G, bw_hz=args.bw)
    Kp = args.kp if args.kp is not None else design["Kp"]
    Ki = args.ki if args.ki is not None else design["Ki"]
    print(f"  Recommended:  Kp = {design['Kp']}   Ki = {design['Ki']}")
    if args.kp or args.ki:
        print(f"  Using:        Kp = {Kp}   Ki = {Ki}  (user override)")
    print(f"  Plant tau ≈ {design['tau_ms']:.1f} ms  "
          f"(≈10 controller update periods)")
    print(f"  Anti-windup limit: ±{design['int_max_W']:.0f} W")

    # Step 3: Stability check
    print("\n[3] Checking closed-loop stability ...")
    stab = check_stability(Kp, Ki, G)
    print(f"  Stable: {stab['stable']}")
    print(f"  Pole magnitudes: "
          f"{[f'{m:.4f}' for m in stab['pole_mags']]}")
    if not stab["stable"]:
        print("  ⚠ WARNING: System is UNSTABLE with these gains.")
        print("    Try reducing Kp or Ki, or lowering --bw.")
    else:
        print(f"  All poles inside unit circle — system is stable ✓")

    # Step 4: Step response simulation
    print("\n[4] Simulating step response ...")
    sim = simulate_step_response(Kp, Ki, G=G)
    err = np.array(sim["P_err"])
    print(f"  Mean |ΔP|% : {err.mean():.2f}%")
    print(f"  Max  |ΔP|% : {err.max():.2f}%")
    print(f"  Within 10% : {(err < 10).mean()*100:.1f}%")

    # Step 5: Save plot
    if args.plot:
        print("\n[5] Saving step response plot ...")
        plot_step_response(sim, Kp, Ki)

    # Step 6: Print recommended server command
    print("\n" + "=" * 65)
    print("  RECOMMENDED SERVER COMMAND")
    print("=" * 65)
    print(f"  python pitnn_plecs_server.py "
          f"--kp {Kp} --ki {Ki} --device cpu")
    print()
    print("  PLECS C-Script sample time:  500e-6")
    print("  PLECS C-Script inputs:       4  [V1, V2, I_out, P_ref_ext]")
    print("  PLECS C-Script outputs:      5  [phi1, phi2, phi3, P_corr, P_meas]")
    print()
    print("  C2000 header equivalents:")
    print(f"    #define PITNN_PI_KP  {Kp}f")
    print(f"    #define PITNN_PI_KI  {Ki}f")
    print("=" * 65)


if __name__ == "__main__":
    main()
