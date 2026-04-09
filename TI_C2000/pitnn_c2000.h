/*
 * pitnn_c2000.h
 * =====================================================================
 * PITNN DAB Converter — TI C2000 DSP Header
 * Constants, types, and function declarations for the PITNN firmware.
 *
 * Target:  TI F28379D (Delfino) — also compatible with F2837xS, F2807x
 * IDE:     Code Composer Studio 12.x
 * C std:   C99
 *
 * Copyright (c) 2026 Chukwuemeka Nzeadibe
 * Mississippi State University — All Rights Reserved
 * =====================================================================
 */

#ifndef PITNN_C2000_H
#define PITNN_C2000_H

#include <stdint.h>
#include <stdbool.h>

/* ═══════════════════════════════════════════════════════════════════════════
   SYSTEM CONSTANTS — match your hardware and pitnn_dab.py training config
   ═══════════════════════════════════════════════════════════════════════════ */

/* Converter parameters */
#define PITNN_V1_NOM        800.0f      /* Nominal primary bus voltage  (V)  */
#define PITNN_V2_NOM        800.0f      /* Nominal secondary bus voltage (V) */
#define PITNN_FSW           100000.0f   /* Switching frequency          (Hz) */
#define PITNN_PI_F          3.14159265f

/* Phase angle bounds (must match pitnn_dab.py constants) */
#define PITNN_PHI12_MIN     2.04203522f /* PI * 0.65 — lower bound phi1/phi2 */
#define PITNN_PHI12_MAX     3.11017082f /* PI * 0.99 — upper bound phi1/phi2 */
#define PITNN_PHI12_NOM     2.98451302f /* PI * 0.95 — nominal seed          */
#define PITNN_PHI3_MIN      0.02f       /* Lower bound phi3                  */
#define PITNN_PHI3_MAX      1.50f       /* Upper bound phi3                  */

/* ── ADJUST THESE TO YOUR HARDWARE ───────────────────────────────────────── */

/* ADC scaling — convert raw ADC counts to physical units
 * Formula: physical = (ADC_count / ADC_MAX) * ADC_REF_V / DIVIDER_RATIO
 * Example for 12-bit ADC, 3.0V reference, 1:300 voltage divider:
 *   V1_SCALE = (3.0 / 4095.0) * 300.0 = 0.21978... ≈ 0.2198f              */
#define PITNN_ADC_V1_SCALE  0.2198f     /* Counts → primary bus voltage  (V) */
#define PITNN_ADC_V2_SCALE  0.2198f     /* Counts → secondary bus voltage (V)*/
#define PITNN_ADC_IL_SCALE  0.0732f     /* Counts → inductor current     (A) */
                                        /* Example: 300A/4095 counts * gain  */

/* ePWM period register value for fsw = 100kHz at EPWMCLK = 200MHz
 * TBPRD = EPWMCLK / (2 * fsw) = 200e6 / (2 * 100e3) = 1000               */
#define PITNN_EPWM_PERIOD   1000U

/* SCI baud rate divider — must match pitnn_uart_server.py BAUD_RATE
 * Formula: BRR = LSPCLK / (baud * 8) - 1
 * For LSPCLK=25MHz (200MHz sysclk / 8), baud=921600:
 *   BRR = 25000000 / (921600 * 8) - 1 = 2.39 → round to 2 → actual ~1.04Mbaud
 * Adjust for your actual LSPCLK. A USB-UART adapter typically accepts 1–4% error. */
#define PITNN_SCI_BRR_HIGH  0x0000U
#define PITNN_SCI_BRR_LOW   0x0002U     /* → ~921600 baud at LSPCLK=25MHz   */

/* How often to request a PITNN update — every N switching cycles.
 * At fsw=100kHz and N=50, update rate = 2kHz, period = 500µs.
 * Round-trip UART latency at 921600 baud for 28 bytes ≈ 350µs.
 * Set N >= 40 to ensure the response arrives before the next request.       */
#define PITNN_UPDATE_CYCLES 50U

/* Outer PI controller gains — tune for your converter and load step specs */
#define PITNN_PI_KP         50.0f       /* Proportional gain             */
#define PITNN_PI_KI         500.0f      /* Integral gain                 */
#define PITNN_PI_PREF_MIN   5000.0f     /* Minimum power reference   (W) */
#define PITNN_PI_PREF_MAX   70000.0f    /* Maximum power reference   (W) */
#define PITNN_VREF          800.0f      /* Output voltage setpoint   (V) */

/* ── END OF USER-CONFIGURABLE SECTION ────────────────────────────────────── */

/* Derived constants */
#define PITNN_DT            (PITNN_UPDATE_CYCLES / PITNN_FSW)   /* PI update dt (s) */


/* ═══════════════════════════════════════════════════════════════════════════
   DATA TYPES
   ═══════════════════════════════════════════════════════════════════════════ */

/* Holds the three PITNN output angles */
typedef struct {
    float phi1;     /* Primary bridge inner duty (rad)   ∈ [PHI12_MIN, PHI12_MAX] */
    float phi2;     /* Secondary bridge inner duty (rad) ∈ [PHI12_MIN, PHI12_MAX] */
    float phi3;     /* External phase shift (rad)        ∈ [PHI3_MIN,  PHI3_MAX]  */
} PITNN_Phi_t;

/* Outer PI controller state */
typedef struct {
    float integral;
    float Pref;
    uint16_t cycle_count;
} PITNN_PI_t;

/* UART transaction state */
typedef struct {
    uint8_t  tx_buf[16];    /* [V1, V2, iL, Pref] as 4×float32 */
    uint8_t  rx_buf[12];    /* [phi1, phi2, phi3] as 3×float32  */
    uint8_t  rx_count;      /* bytes received so far            */
    bool     tx_pending;    /* true while waiting for response  */
    uint32_t tx_cycle;      /* cycle number of last TX          */
} PITNN_UART_t;


/* ═══════════════════════════════════════════════════════════════════════════
   FUNCTION DECLARATIONS
   ═══════════════════════════════════════════════════════════════════════════ */

/* pitnn_c2000_setup.c */
void PITNN_InitSystem(void);    /* PLL, clocks, watchdog                      */
void PITNN_InitADC(void);       /* ADC-A (V1), ADC-B (V2), ADC-C (iL)        */
void PITNN_InitSCI(void);       /* SCI-A for UART to companion PC             */
void PITNN_InitEPWM(void);      /* ePWM1 (primary), ePWM2 (secondary + phase) */
void PITNN_InitPIE(void);       /* PIE vector table and interrupt enables      */

/* pitnn_c2000.c */
void    PITNN_PIController(PITNN_PI_t *pi, float V2_meas);
void    PITNN_SendToPC(PITNN_UART_t *uart, float V1, float V2,
                       float iL, float Pref);
bool    PITNN_RecvFromPC(PITNN_UART_t *uart, PITNN_Phi_t *phi);
void    PITNN_ApplyPWM(const PITNN_Phi_t *phi);
void    PITNN_ClampPhi(PITNN_Phi_t *phi);
__interrupt void epwm1_isr(void);
__interrupt void sciaRx_isr(void);

#endif /* PITNN_C2000_H */
