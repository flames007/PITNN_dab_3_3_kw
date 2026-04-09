/*
 * pitnn_cscript.h
 * ===============================================================
 * PITNN DAB Converter — PLECS C-Script Block Integration
 * Method 1: Runs inside PLECS with no external process needed.
 *
 * HOW TO USE IN PLECS
 * ────────────────────
 * 1. In your PLECS schematic, add a C-Script block
 *    (Components → Control → C-Script)
 *
 * 2. Set block ports:
 *      Inputs  (4): V1, V2, iL, Pref
 *      Outputs (5): phi1, phi2, phi3, delay_us, duty_pct
 *
 * 3. Copy the three sections below into the C-Script editor:
 *      Section 1 → "Declarations"
 *      Section 2 → "Output function"
 *      Section 3 → "Reset function"
 *
 * 4. NOTE: The Transformer forward pass (564k parameters) cannot
 *    run inside a C-Script directly. This file handles the buffer
 *    management and normalisation. For full inference either:
 *      (a) Use Method 2 (Python co-simulation) — recommended
 *      (b) Use Method 4 (socket server) — for RT/HIL targets
 *    If you have embedded the weights as C arrays (see comments
 *    in the Output function), replace the placeholder with real
 *    matrix multiply calls to your weight arrays.
 *
 * Copyright (c) 2026 Chukwuemeka Nzeadibe
 * Mississippi State University — All Rights Reserved
 * ===============================================================
 */


/* ═══════════════════════════════════════════════════════════════
   SECTION 1 — DECLARATIONS
   Paste everything below this line into the "Declarations" tab
   ═══════════════════════════════════════════════════════════════ */

#include <math.h>
#include <string.h>

/* ── Controller constants ─────────────────────────────────────── */
#define PITNN_SEQ_LEN    20
#define PITNN_N_FEAT      8
#define PITNN_PI_F        3.14159265358979f
#define PITNN_PHI12_MIN   2.04203522f   /* PI * 0.65 — lower bound phi1/phi2  */
#define PITNN_PHI12_MAX   3.11017082f   /* PI * 0.99 — upper bound phi1/phi2  */
#define PITNN_PHI12_NOM   2.98451302f   /* PI * 0.95 — nominal seed           */
#define PITNN_PHI_MIN     0.02f         /* lower bound phi3                   */
#define PITNN_PHI3_MAX    1.50f
#define PITNN_V1_NOM    800.0f
#define PITNN_V2_NOM    800.0f
#define PITNN_FSW    100000.0f

/*
 * Normalisation constants extracted from pitnn_mu.npy / pitnn_sigma.npy
 * NOTE: phi1/phi2 now vary — these values MUST be refreshed after retraining.
 * Run: python pitnn_inspect_exports.py  and paste the C arrays printed there.
 * Feature order: [V1, V2, iL, phi1, phi2, phi3, Pref, V1V2/Vnom2]
 */
static const float PITNN_MU[PITNN_N_FEAT] = {
    800.02514648f,    /* V1   (V)        */
    800.20202637f,    /* V2   (V)        */
     25.38187981f,    /* iL   (A)        */
      2.99313879f,    /* phi1 (rad)      */
      2.99313879f,    /* phi2 (rad)      */
      0.47853482f,    /* phi3 (rad)      */
  37593.66796875f,    /* Pref (W)        */
      1.00037384f     /* V1V2/Vnom2      */
};

static const float PITNN_SIGMA[PITNN_N_FEAT] = {
     46.25230789f,    /* V1              */
     46.19021606f,    /* V2              */
     15.18008804f,    /* iL              */
      0.00862482f,    /* phi1            */
      0.00862482f,    /* phi2            */
      0.29205424f,    /* phi3            */
  18769.15234375f,    /* Pref            */
      0.08148149f     /* V1V2/Vnom2      */
};

/* Rolling history buffer — persists between simulation time steps */
static float pitnn_buffer[PITNN_SEQ_LEN][PITNN_N_FEAT];
static float pitnn_phi1_prev = PITNN_PHI12_NOM;  /* previous phi1 output */
static float pitnn_phi2_prev = PITNN_PHI12_NOM;  /* previous phi2 output */
static float pitnn_phi3_prev = 0.22f;             /* previous phi3 output */
static int   pitnn_init_done  = 0;

/* Safety clamp helper */
static float pitnn_clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 2 — OUTPUT FUNCTION
   Paste everything below this line into the "Output function" tab
   ═══════════════════════════════════════════════════════════════ */

{
    int step, f;
    float feat[PITNN_N_FEAT];
    float feat_norm[PITNN_N_FEAT];
    float phi1, phi2, phi3, v_ratio;

    /* Initialise buffer on first call */
    if (!pitnn_init_done) {
        memset(pitnn_buffer, 0, sizeof(pitnn_buffer));
        pitnn_phi1_prev = PITNN_PHI12_NOM;
        pitnn_phi2_prev = PITNN_PHI12_NOM;
        pitnn_phi3_prev = 0.22f;
        pitnn_init_done = 1;
    }

    /* ── Read inputs ──────────────────────────────────────────── */
    float V1   = (float)u[0];   /* Primary bus voltage   (V) */
    float V2   = (float)u[1];   /* Secondary bus voltage (V) */
    float iL   = (float)u[2];   /* Inductor current      (A) */
    float Pref = (float)u[3];   /* Power reference       (W) */

    /* ── Build 8-feature state vector ────────────────────────── */
    v_ratio  = (V1 * V2) / (PITNN_V1_NOM * PITNN_V2_NOM);
    feat[0]  = V1;
    feat[1]  = V2;
    feat[2]  = iL;
    feat[3]  = pitnn_phi1_prev;   /* previous predicted phi1 */
    feat[4]  = pitnn_phi2_prev;   /* previous predicted phi2 */
    feat[5]  = pitnn_phi3_prev;   /* previous predicted phi3 */
    feat[6]  = Pref;
    feat[7]  = v_ratio;

    /* ── Normalise: (x - mu) / sigma ─────────────────────────── */
    for (f = 0; f < PITNN_N_FEAT; f++)
        feat_norm[f] = (feat[f] - PITNN_MU[f]) / PITNN_SIGMA[f];

    /* ── Update rolling buffer (shift left, append new state) ─── */
    for (step = 0; step < PITNN_SEQ_LEN - 1; step++)
        for (f = 0; f < PITNN_N_FEAT; f++)
            pitnn_buffer[step][f] = pitnn_buffer[step + 1][f];
    for (f = 0; f < PITNN_N_FEAT; f++)
        pitnn_buffer[PITNN_SEQ_LEN - 1][f] = feat_norm[f];

    /*
     * ── INFERENCE ─────────────────────────────────────────────
     *
     * The normalised buffer (pitnn_buffer[20][8]) is now ready.
     * The model predicts [phi1, phi2, phi3] independently.
     * Replace the placeholder assignments below with your inference call.
     *
     * OPTION A — Embedded weights (advanced):
     *   Call your matrix-multiply forward pass here.
     *   Export weights from pitnn_scripted.pt using:
     *     python -c "
     *       import torch, numpy as np
     *       m = torch.jit.load('pitnn_scripted.pt')
     *       for k,v in m.state_dict().items():
     *           np.savetxt(k.replace('.','_')+'.csv', v.cpu().numpy().reshape(-1,max(1,v.shape[-1])))
     *     "
     *   Then load each CSV as a C array and implement matmul.
     *
     * OPTION B — Method 2 (Python co-simulation):
     *   Remove this C-Script inference block entirely.
     *   Use pitnn_plecs_cosim.py as a Simulation Script instead.
     *
     * OPTION C — Method 4 (socket server):
     *   Call pitnn_send_recv() here (see pitnn_socket_client.h)
     *   to send the buffer to pitnn_socket_server.py and receive
     *   [phi1, phi2, phi3].
     *
     * Placeholder — replace with real inference:
     */
    phi1 = pitnn_phi1_prev;
    phi2 = pitnn_phi2_prev;
    phi3 = pitnn_phi3_prev;

    /* ── Safety clamp all three outputs ──────────────────────── */
    phi1 = pitnn_clamp(phi1, PITNN_PHI12_MIN, PITNN_PHI12_MAX);
    phi2 = pitnn_clamp(phi2, PITNN_PHI12_MIN, PITNN_PHI12_MAX);
    phi3 = pitnn_clamp(phi3, PITNN_PHI_MIN,   PITNN_PHI3_MAX);
    pitnn_phi1_prev = phi1;
    pitnn_phi2_prev = phi2;
    pitnn_phi3_prev = phi3;

    /* ── Write outputs ────────────────────────────────────────── */
    y[0] = (double)phi1;                                              /* phi1 (rad)  */
    y[1] = (double)phi2;                                              /* phi2 (rad)  */
    y[2] = (double)phi3;                                              /* phi3 (rad)  */
    y[3] = (double)(phi3 / (2.0f * PITNN_PI_F * PITNN_FSW) * 1e6f); /* delay (µs)  */
    y[4] = (double)(phi1 / PITNN_PI_F * 100.0f);                     /* duty %      */
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 3 — RESET FUNCTION
   Paste everything below this line into the "Reset function" tab
   ═══════════════════════════════════════════════════════════════ */

{
    memset(pitnn_buffer, 0, sizeof(pitnn_buffer));
    pitnn_phi1_prev = PITNN_PHI12_NOM;
    pitnn_phi2_prev = PITNN_PHI12_NOM;
    pitnn_phi3_prev = 0.22f;
    pitnn_init_done  = 0;
}
