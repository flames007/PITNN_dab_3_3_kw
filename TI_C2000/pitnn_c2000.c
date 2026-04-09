/*
 * pitnn_c2000.c
 * =====================================================================
 * PITNN DAB Converter — TI C2000 DSP Main Firmware
 *
 * This file contains:
 *   - ePWM1 interrupt (100kHz) — ADC reads, PI loop, UART trigger
 *   - SCI-A RX interrupt — receives phi1/phi2/phi3 from companion PC
 *   - Outer PI voltage controller
 *   - PWM register update for TPS modulation
 *
 * Target:  TI F28379D (Delfino dual-core)
 *          Runs on CPU1. CPU2 unused.
 * IDE:     Code Composer Studio 12.x
 * Include: pitnn_c2000.h, pitnn_c2000_setup.c
 *          F2837xD_Device.h (from C2000Ware)
 *
 * ADC channel assignments:
 *   ADCA-SOC0 → ADCINA0 → V1  (primary bus, via voltage divider)
 *   ADCB-SOC0 → ADCINB0 → V2  (secondary bus, via voltage divider)
 *   ADCC-SOC0 → ADCINC0 → iL  (inductor current, via Hall sensor)
 *
 * ePWM assignments:
 *   ePWM1A/B → Primary H-bridge   (inner duty = phi1/π)
 *   ePWM2A/B → Secondary H-bridge (inner duty = phi2/π, phase = phi3)
 *
 * Copyright (c) 2026 Chukwuemeka Nzeadibe
 * Mississippi State University — All Rights Reserved
 * =====================================================================
 */

#include "pitnn_c2000.h"
#include "F2837xD_Device.h"     /* C2000Ware device header — from your CCS project */
#include <string.h>
#include <math.h>

/* ── Global state ─────────────────────────────────────────────────────────── */

/* Current best phi values — updated by SCI-A RX interrupt, applied every cycle */
static volatile PITNN_Phi_t g_phi = {
    .phi1 = PITNN_PHI12_NOM,    /* Start at nominal inner duty */
    .phi2 = PITNN_PHI12_NOM,
    .phi3 = 0.22f               /* Start at a safe low-power phase shift */
};

static PITNN_PI_t   g_pi   = { .integral = 0.0f,
                                .Pref     = PITNN_PI_PREF_MIN,
                                .cycle_count = 0U };

static PITNN_UART_t g_uart = { .rx_count    = 0U,
                                .tx_pending  = false,
                                .tx_cycle    = 0U };

/* Cycle counter — incremented in ePWM ISR */
static volatile uint32_t g_cycle = 0U;


/* ═══════════════════════════════════════════════════════════════════════════
   main()
   ═══════════════════════════════════════════════════════════════════════════ */

void main(void)
{
    /* One-time peripheral initialisation (in pitnn_c2000_setup.c) */
    PITNN_InitSystem();
    PITNN_InitADC();
    PITNN_InitSCI();
    PITNN_InitEPWM();
    PITNN_InitPIE();

    /* Enable global interrupts and real-time debug */
    EINT;
    ERTM;

    /* Spin forever — all work done in interrupts */
    for (;;) {
        /* Feed watchdog so it does not reset the DSP */
        EALLOW;
        WdRegs.WDKEY.bit.WDKEY = 0x55U;
        WdRegs.WDKEY.bit.WDKEY = 0xAAU;
        EDIS;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
   ePWM1 ISR — runs at fsw = 100kHz
   Reads ADC, runs outer PI, sends to PC every PITNN_UPDATE_CYCLES cycles,
   applies the most recent phi values to the PWM registers every cycle.
   ═══════════════════════════════════════════════════════════════════════════ */

#pragma CODE_SECTION(epwm1_isr, ".TI.ramfunc")   /* execute from RAM for speed */
__interrupt void epwm1_isr(void)
{
    g_cycle++;

    /* ── 1. Read ADC results (conversions triggered by ePWM SOC) ─────────── */
    float V1  = (float)AdcaResultRegs.ADCRESULT0 * PITNN_ADC_V1_SCALE;
    float V2  = (float)AdcbResultRegs.ADCRESULT0 * PITNN_ADC_V2_SCALE;
    float iL  = (float)AdccResultRegs.ADCRESULT0 * PITNN_ADC_IL_SCALE;

    /* iL is bipolar — centre the signed current around ADC midpoint */
    iL -= (4096.0f * PITNN_ADC_IL_SCALE / 2.0f);

    /* ── 2. Outer PI voltage controller (every PITNN_UPDATE_CYCLES cycles) ── */
    if (g_cycle % PITNN_UPDATE_CYCLES == 0U) {
        PITNN_PIController(&g_pi, V2);
    }

    /* ── 3. Send sensor data to PC (every PITNN_UPDATE_CYCLES cycles) ─────── */
    /*    Only send when not already waiting for a response to avoid flooding.  */
    if ((g_cycle % PITNN_UPDATE_CYCLES == 1U) && !g_uart.tx_pending) {
        PITNN_SendToPC(&g_uart, V1, V2, iL, g_pi.Pref);
    }

    /* ── 4. Apply most recent phi to PWM registers (every cycle) ──────────── */
    PITNN_ApplyPWM(&g_phi);

    /* ── 5. Clear interrupt flags ──────────────────────────────────────────── */
    EPwm1Regs.ETCLR.bit.INT = 1U;
    PieCtrlRegs.PIEACK.all  = PIEACK_GROUP3;
}


/* ═══════════════════════════════════════════════════════════════════════════
   SCI-A RX ISR — fires for each received byte from the PC
   Accumulates bytes into rx_buf. On receiving all 12 bytes, unpacks
   phi1/phi2/phi3 and updates g_phi atomically.
   ═══════════════════════════════════════════════════════════════════════════ */

#pragma CODE_SECTION(sciaRx_isr, ".TI.ramfunc")
__interrupt void sciaRx_isr(void)
{
    /* Read received byte into buffer */
    g_uart.rx_buf[g_uart.rx_count++] = (uint8_t)(SciaRegs.SCIRXBUF.all & 0xFFU);

    /* Once all 12 bytes have arrived, unpack and update phi */
    if (g_uart.rx_count >= 12U) {
        PITNN_Phi_t new_phi;

        /* Unpack three little-endian float32 values */
        memcpy(&new_phi.phi1, &g_uart.rx_buf[0], sizeof(float));
        memcpy(&new_phi.phi2, &g_uart.rx_buf[4], sizeof(float));
        memcpy(&new_phi.phi3, &g_uart.rx_buf[8], sizeof(float));

        /* Safety clamp before applying */
        PITNN_ClampPhi(&new_phi);

        /* Atomic update of g_phi (written in ISR, read in ISR — no race) */
        g_phi = new_phi;

        /* Ready for next transaction */
        g_uart.rx_count   = 0U;
        g_uart.tx_pending = false;
    }

    /* Clear RX interrupt flag */
    SciaRegs.SCIFFRX.bit.RXFFINTCLR = 1U;
    PieCtrlRegs.PIEACK.all = PIEACK_GROUP9;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_PIController
   Simple PI voltage regulator: V2_meas → Pref
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_PIController(PITNN_PI_t *pi, float V2_meas)
{
    float err = PITNN_VREF - V2_meas;

    pi->integral += err * PITNN_DT;

    /* Anti-windup: clamp integral */
    float integral_max = PITNN_PI_PREF_MAX / PITNN_PI_KI;
    if (pi->integral >  integral_max) pi->integral =  integral_max;
    if (pi->integral < -integral_max) pi->integral = -integral_max;

    float Pref = PITNN_PI_KP * err + PITNN_PI_KI * pi->integral;

    /* Clamp output to valid power range */
    if (Pref < PITNN_PI_PREF_MIN) Pref = PITNN_PI_PREF_MIN;
    if (Pref > PITNN_PI_PREF_MAX) Pref = PITNN_PI_PREF_MAX;

    pi->Pref = Pref;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_SendToPC
   Pack [V1, V2, iL, Pref] as 16 bytes and transmit over SCI-A.
   Uses FIFO-based TX — fills FIFO and returns immediately.
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_SendToPC(PITNN_UART_t *uart, float V1, float V2,
                    float iL, float Pref)
{
    /* Pack four float32 values into tx_buf in little-endian order */
    memcpy(&uart->tx_buf[0],  &V1,   sizeof(float));
    memcpy(&uart->tx_buf[4],  &V2,   sizeof(float));
    memcpy(&uart->tx_buf[8],  &iL,   sizeof(float));
    memcpy(&uart->tx_buf[12], &Pref, sizeof(float));

    /* Transmit byte-by-byte into SCI TX FIFO */
    uint8_t i;
    for (i = 0U; i < 16U; i++) {
        /* Wait while FIFO is full (should not block at this baud/update rate) */
        while (SciaRegs.SCIFFTX.bit.TXFFST >= 15U) { /* spin */ }
        SciaRegs.SCITXBUF.all = (uint16_t)uart->tx_buf[i];
    }

    uart->tx_pending = true;
    uart->rx_count   = 0U;
    uart->tx_cycle   = g_cycle;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_ApplyPWM
   Convert phi1, phi2, phi3 to ePWM register values and write them.
   Called every switching cycle with the most recently received phi values.

   ePWM configuration (up-down count, centre-aligned PWM):
     Primary bridge   — ePWM1, TBPRD = PITNN_EPWM_PERIOD
     Secondary bridge — ePWM2, phase-shifted relative to ePWM1

   Inner duty conversion:   CMPA = phi / π × TBPRD
   Phase shift conversion:  TBPHS = phi3 / (2π) × 2×TBPRD
   ═══════════════════════════════════════════════════════════════════════════ */

#pragma CODE_SECTION(PITNN_ApplyPWM, ".TI.ramfunc")
void PITNN_ApplyPWM(const PITNN_Phi_t *phi)
{
    /* Inner duty for primary bridge (ePWM1) */
    uint16_t cmpa1 = (uint16_t)(phi->phi1 / PITNN_PI_F
                                 * (float)PITNN_EPWM_PERIOD);

    /* Inner duty for secondary bridge (ePWM2) */
    uint16_t cmpa2 = (uint16_t)(phi->phi2 / PITNN_PI_F
                                 * (float)PITNN_EPWM_PERIOD);

    /* Phase shift: phi3 in radians → TBPHS counts
     * Up-down counter period = 2×TBPRD counts per switching cycle
     * TBPHS = phi3 / (2π) × 2×TBPRD                                        */
    uint16_t tbphs = (uint16_t)(phi->phi3 / (2.0f * PITNN_PI_F)
                                 * 2.0f * (float)PITNN_EPWM_PERIOD);

    /* Clamp to valid register range */
    if (cmpa1 > PITNN_EPWM_PERIOD) cmpa1 = PITNN_EPWM_PERIOD;
    if (cmpa2 > PITNN_EPWM_PERIOD) cmpa2 = PITNN_EPWM_PERIOD;
    if (tbphs > 2U * PITNN_EPWM_PERIOD) tbphs = 2U * PITNN_EPWM_PERIOD;

    /* Write to shadow registers — loaded at TBCTR=0 (zero event) */
    EALLOW;
    EPwm1Regs.CMPA.bit.CMPA = cmpa1;
    EPwm2Regs.CMPA.bit.CMPA = cmpa2;
    EPwm2Regs.TBPHS.bit.TBPHS = tbphs;
    EDIS;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_ClampPhi
   Hardware safety clamp — applied to every PITNN response before use.
   Prevents out-of-range values from corrupting the PWM registers even if
   the UART receives a corrupt packet.
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_ClampPhi(PITNN_Phi_t *phi)
{
    if (phi->phi1 < PITNN_PHI12_MIN) phi->phi1 = PITNN_PHI12_MIN;
    if (phi->phi1 > PITNN_PHI12_MAX) phi->phi1 = PITNN_PHI12_MAX;
    if (phi->phi2 < PITNN_PHI12_MIN) phi->phi2 = PITNN_PHI12_MIN;
    if (phi->phi2 > PITNN_PHI12_MAX) phi->phi2 = PITNN_PHI12_MAX;
    if (phi->phi3 < PITNN_PHI3_MIN)  phi->phi3 = PITNN_PHI3_MIN;
    if (phi->phi3 > PITNN_PHI3_MAX)  phi->phi3 = PITNN_PHI3_MAX;
}
