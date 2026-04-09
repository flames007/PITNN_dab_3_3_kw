/*
 * pitnn_c2000_setup.c
 * =====================================================================
 * PITNN DAB Converter — TI C2000 Peripheral Initialisation
 *
 * Configures all hardware peripherals needed for the PITNN controller:
 *   PLL   → 200MHz system clock from 20MHz XTAL
 *   ADC-A → V1 (primary bus voltage)
 *   ADC-B → V2 (secondary bus voltage)
 *   ADC-C → iL (inductor current)
 *   SCI-A → UART to companion PC (GPIO28/GPIO29)
 *   ePWM1 → Primary H-bridge PWM
 *   ePWM2 → Secondary H-bridge PWM + phase shift
 *   PIE   → Interrupt routing for ePWM1 and SCI-A RX
 *
 * All settings use registers directly — no driverlib dependency.
 *
 * ── HOW TO ADAPT TO YOUR BOARD ────────────────────────────────────
 * 1. Adjust PLL multipliers if your XTAL is not 20MHz (search PLLSYSCTL).
 * 2. Adjust ADC channel assignments to match your PCB routing.
 * 3. Adjust PITNN_SCI_BRR_HIGH/LOW in pitnn_c2000.h for your LSPCLK.
 * 4. Adjust ePWM action-qualifier logic for your H-bridge gate driver
 *    polarity (active-high or active-low).
 *
 * Copyright (c) 2026 Chukwuemeka Nzeadibe
 * Mississippi State University — All Rights Reserved
 * =====================================================================
 */

#include "pitnn_c2000.h"
#include "F2837xD_Device.h"

/* Forward declarations of ISRs (defined in pitnn_c2000.c) */
extern __interrupt void epwm1_isr(void);
extern __interrupt void sciaRx_isr(void);


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_InitSystem
   PLL → 200MHz, disable watchdog during init, enable peripheral clocks.
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_InitSystem(void)
{
    EALLOW;

    /* Disable watchdog during initialisation */
    WdRegs.WDCR.all = 0x0068U;

    /* PLL: 20MHz XTAL × 10 = 200MHz SYSCLKOUT
     * PLLSYSCTL: PLLMULT=10, PLLDIV=1, PLLCLKEN=1                          */
    ClkCfgRegs.CLKSRCCTL1.bit.OSCCLKSRCSEL = 0U;    /* Internal XTAL        */
    ClkCfgRegs.SYSPLLCTL1.bit.PLLEN        = 0U;    /* Disable PLL first    */
    ClkCfgRegs.SYSPLLMULT.bit.IMULT        = 10U;   /* × 10 multiplier      */
    ClkCfgRegs.SYSPLLMULT.bit.FMULT        = 0U;    /* No fractional mult   */
    ClkCfgRegs.SYSPLLCTL1.bit.PLLEN        = 1U;    /* Enable PLL           */

    /* Wait for PLL lock */
    while (ClkCfgRegs.SYSPLLSTS.bit.LOCKS != 1U) { /* spin */ }

    ClkCfgRegs.SYSCLKDIVSEL.bit.PLLSYSCLKDIV = 0U; /* Divide by 1 → 200MHz */

    /* Enable peripheral clocks */
    CpuSysRegs.PCLKCR0.bit.CLA1         = 0U;
    CpuSysRegs.PCLKCR2.bit.EPWM1        = 1U;
    CpuSysRegs.PCLKCR2.bit.EPWM2        = 1U;
    CpuSysRegs.PCLKCR8.bit.SCI_A        = 1U;
    CpuSysRegs.PCLKCR13.bit.ADC_A       = 1U;
    CpuSysRegs.PCLKCR13.bit.ADC_B       = 1U;
    CpuSysRegs.PCLKCR13.bit.ADC_C       = 1U;

    /* Low-speed clock: LSPCLK = SYSCLKOUT / 8 = 25MHz (for SCI baud calc) */
    ClkCfgRegs.LOSPCP.bit.LSPCLKDIV = 3U;          /* Divide by 2^3 = 8    */

    /* Re-enable watchdog (kicked in main loop) */
    WdRegs.WDCR.all = 0x0028U;

    EDIS;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_InitADC
   Three ADC modules, 12-bit, triggered by ePWM1 SOC event.
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_InitADC(void)
{
    EALLOW;

    /* Power up all three ADC modules */
    AdcaRegs.ADCCTL1.bit.ADCPWDNZ = 1U;
    AdcbRegs.ADCCTL1.bit.ADCPWDNZ = 1U;
    AdccRegs.ADCCTL1.bit.ADCPWDNZ = 1U;

    /* Short delay for ADC power-up (≥1ms at 200MHz = 200,000 cycles) */
    {
        volatile uint32_t d;
        for (d = 0U; d < 250000U; d++) { /* spin */ }
    }

    /* Resolution: 12-bit single-ended */
    AdcaRegs.ADCCTL2.bit.RESOLUTION = 0U;  /* 12-bit */
    AdcaRegs.ADCCTL2.bit.SIGNALMODE = 0U;  /* single-ended */
    AdcbRegs.ADCCTL2.bit.RESOLUTION = 0U;
    AdcbRegs.ADCCTL2.bit.SIGNALMODE = 0U;
    AdccRegs.ADCCTL2.bit.RESOLUTION = 0U;
    AdccRegs.ADCCTL2.bit.SIGNALMODE = 0U;

    /* Calibrate ADCs (uses factory trim values) */
    AdcaRegs.ADCCTL1.bit.INTPULSEPOS = 1U;  /* End-of-conversion pulse      */
    AdcbRegs.ADCCTL1.bit.INTPULSEPOS = 1U;
    AdccRegs.ADCCTL1.bit.INTPULSEPOS = 1U;

    /* ADC-A SOC0: ADCINA0 → V1, triggered by ePWM1 SOCA */
    AdcaRegs.ADCSOC0CTL.bit.CHSEL    = 0U;  /* ADCINA0 */
    AdcaRegs.ADCSOC0CTL.bit.TRIGSEL  = 5U;  /* ePWM1 SOCA */
    AdcaRegs.ADCSOC0CTL.bit.ACQPS    = 14U; /* 75ns sample window at 200MHz */

    /* ADC-B SOC0: ADCINB0 → V2, triggered by ePWM1 SOCA */
    AdcbRegs.ADCSOC0CTL.bit.CHSEL    = 0U;  /* ADCINB0 */
    AdcbRegs.ADCSOC0CTL.bit.TRIGSEL  = 5U;  /* ePWM1 SOCA */
    AdcbRegs.ADCSOC0CTL.bit.ACQPS    = 14U;

    /* ADC-C SOC0: ADCINC0 → iL, triggered by ePWM1 SOCA */
    AdccRegs.ADCSOC0CTL.bit.CHSEL    = 0U;  /* ADCINC0 */
    AdccRegs.ADCSOC0CTL.bit.TRIGSEL  = 5U;  /* ePWM1 SOCA */
    AdccRegs.ADCSOC0CTL.bit.ACQPS    = 14U;

    EDIS;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_InitSCI
   SCI-A at 921600 baud on GPIO28 (TX) / GPIO29 (RX).
   8 data bits, no parity, 1 stop bit.
   RX FIFO interrupt at 1 byte (fires for every received byte).
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_InitSCI(void)
{
    EALLOW;

    /* Route GPIO28 → SCI-A TX, GPIO29 → SCI-A RX */
    GpioCtrlRegs.GPAPUD.bit.GPIO28   = 0U;  /* Enable pull-up on GPIO28     */
    GpioCtrlRegs.GPAPUD.bit.GPIO29   = 0U;  /* Enable pull-up on GPIO29     */
    GpioCtrlRegs.GPAQSEL2.bit.GPIO28 = 3U;  /* Async qualification          */
    GpioCtrlRegs.GPAQSEL2.bit.GPIO29 = 3U;
    GpioCtrlRegs.GPAMUX2.bit.GPIO28  = 1U;  /* Mux to SCI-A TX (mode 1)    */
    GpioCtrlRegs.GPAMUX2.bit.GPIO29  = 1U;  /* Mux to SCI-A RX (mode 1)    */

    /* Reset SCI-A */
    SciaRegs.SCICTL1.bit.SWRESET = 0U;

    /* Baud rate: PITNN_SCI_BRR_HIGH:LOW (see pitnn_c2000.h comments) */
    SciaRegs.SCIHBAUD.all = PITNN_SCI_BRR_HIGH;
    SciaRegs.SCILBAUD.all = PITNN_SCI_BRR_LOW;

    /* 8N1: 8 data bits, no parity, 1 stop bit */
    SciaRegs.SCICCR.all = 0x0007U;

    /* Enable TX and RX */
    SciaRegs.SCICTL1.all = 0x0003U;

    /* TX FIFO: enable FIFO mode, reset TX FIFO */
    SciaRegs.SCIFFTX.all = 0xE040U;    /* SCIFFENA=1, TXFIFOXRESET=1        */

    /* RX FIFO: interrupt when 1 byte received (fires per byte) */
    SciaRegs.SCIFFRX.all = 0x2021U;    /* RXFFIENA=1, RXFFIL=1              */

    /* Release SCI-A from reset */
    SciaRegs.SCICTL1.bit.SWRESET = 1U;

    EDIS;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_InitEPWM
   ePWM1 — primary H-bridge, up-down count, 100kHz, triggers ADC SOC.
   ePWM2 — secondary H-bridge, up-down count, 100kHz, phase-shifted.

   Both run in up-down (centre-aligned) mode with:
     TBPRD = PITNN_EPWM_PERIOD = 1000
     Dead-band: 50 counts = 500ns at 100MHz EPWMCLK (adjust for your driver)

   Action qualifiers (active-HIGH gate driver assumed):
     Count-UP   crossing CMPA → force A high
     Count-DOWN crossing CMPA → force A low
     (B is complementary via dead-band module)
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_InitEPWM(void)
{
    EALLOW;

    /* Disable SYNCO to stop ePWM clocks during config */
    CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 0U;

    /* ── ePWM1 — Primary bridge ──────────────────────────────────────────── */

    /* Time-base: up-down count, period=1000, no phase shift, EPWMCLK/1     */
    EPwm1Regs.TBCTL.bit.CTRMODE    = TB_COUNT_UPDOWN;
    EPwm1Regs.TBCTL.bit.PHSEN      = TB_DISABLE;       /* No phase sync      */
    EPwm1Regs.TBCTL.bit.SYNCOSEL   = TB_CTR_ZERO;      /* Sync out at zero   */
    EPwm1Regs.TBCTL.bit.HSPCLKDIV  = TB_DIV1;
    EPwm1Regs.TBCTL.bit.CLKDIV     = TB_DIV1;
    EPwm1Regs.TBPRD                = PITNN_EPWM_PERIOD;
    EPwm1Regs.TBPHS.bit.TBPHS      = 0U;
    EPwm1Regs.TBCTR                = 0U;

    /* Counter-compare: initial duty at PHI12_NOM/π */
    EPwm1Regs.CMPA.bit.CMPA = (uint16_t)(PITNN_PHI12_NOM / 3.14159265f
                                          * (float)PITNN_EPWM_PERIOD);

    /* Action-qualifier: A goes HIGH on count-up CMPA, LOW on count-down CMPA */
    EPwm1Regs.AQCTLA.bit.CAU = AQ_SET;
    EPwm1Regs.AQCTLA.bit.CAD = AQ_CLEAR;

    /* Dead-band: 50 counts ≈ 500ns (rising and falling edge delay)          */
    EPwm1Regs.DBCTL.bit.OUT_MODE  = DB_FULL_ENABLE;
    EPwm1Regs.DBCTL.bit.POLSEL    = DB_ACTV_HIC;   /* B inverted (active-hi)  */
    EPwm1Regs.DBCTL.bit.IN_MODE   = DBA_ALL;
    EPwm1Regs.DBRED.bit.DBRED     = 50U;            /* 50 counts rising delay  */
    EPwm1Regs.DBFED.bit.DBFED     = 50U;            /* 50 counts falling delay */

    /* ADC SOC trigger: SOCA on zero event (start of each cycle) */
    EPwm1Regs.ETSEL.bit.SOCAEN    = 1U;
    EPwm1Regs.ETSEL.bit.SOCASEL   = ET_CTR_ZERO;
    EPwm1Regs.ETPS.bit.SOCAPRD    = ET_1ST;         /* Trigger every period   */

    /* Interrupt: INT on zero event — drives control ISR */
    EPwm1Regs.ETSEL.bit.INTEN     = 1U;
    EPwm1Regs.ETSEL.bit.INTSEL    = ET_CTR_ZERO;
    EPwm1Regs.ETPS.bit.INTPRD     = ET_1ST;

    /* ── ePWM2 — Secondary bridge ───────────────────────────────────────── */

    EPwm2Regs.TBCTL.bit.CTRMODE   = TB_COUNT_UPDOWN;
    EPwm2Regs.TBCTL.bit.PHSEN     = TB_ENABLE;        /* Receives sync from ePWM1 */
    EPwm2Regs.TBCTL.bit.PHSDIR    = 1U;               /* Count UP after sync      */
    EPwm2Regs.TBCTL.bit.SYNCOSEL  = TB_SYNC_IN;       /* Pass sync through         */
    EPwm2Regs.TBCTL.bit.HSPCLKDIV = TB_DIV1;
    EPwm2Regs.TBCTL.bit.CLKDIV    = TB_DIV1;
    EPwm2Regs.TBPRD               = PITNN_EPWM_PERIOD;

    /* Initial phase: phi3=0.22 rad → TBPHS counts */
    EPwm2Regs.TBPHS.bit.TBPHS    = (uint16_t)(0.22f / (2.0f * 3.14159265f)
                                               * 2.0f * (float)PITNN_EPWM_PERIOD);
    EPwm2Regs.TBCTR               = 0U;

    /* Counter-compare: initial duty at PHI12_NOM/π */
    EPwm2Regs.CMPA.bit.CMPA = (uint16_t)(PITNN_PHI12_NOM / 3.14159265f
                                          * (float)PITNN_EPWM_PERIOD);

    EPwm2Regs.AQCTLA.bit.CAU  = AQ_SET;
    EPwm2Regs.AQCTLA.bit.CAD  = AQ_CLEAR;

    EPwm2Regs.DBCTL.bit.OUT_MODE = DB_FULL_ENABLE;
    EPwm2Regs.DBCTL.bit.POLSEL   = DB_ACTV_HIC;
    EPwm2Regs.DBCTL.bit.IN_MODE  = DBA_ALL;
    EPwm2Regs.DBRED.bit.DBRED    = 50U;
    EPwm2Regs.DBFED.bit.DBFED    = 50U;

    /* Re-enable time-base clock sync */
    CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 1U;

    EDIS;
}


/* ═══════════════════════════════════════════════════════════════════════════
   PITNN_InitPIE
   Route ePWM1 INT → CPU INT3 (Group 3, vector 1)
   Route SCI-A RX  → CPU INT9 (Group 9, vector 1)
   ═══════════════════════════════════════════════════════════════════════════ */

void PITNN_InitPIE(void)
{
    EALLOW;

    /* Disable and clear all CPU interrupts */
    DINT;
    IER = 0x0000U;
    IFR = 0x0000U;

    /* Initialise PIE control registers to default (from C2000Ware) */
    PieCtrlRegs.PIECTRL.bit.ENPIE = 1U;    /* Enable PIE block */

    /* Install ISR vectors into PIE table */
    PieVectTable.EPWM1_INT  = &epwm1_isr;  /* PIE Group 3, INT1 */
    PieVectTable.SCIA_RX_INT = &sciaRx_isr; /* PIE Group 9, INT1 */

    /* Enable PIE group interrupts */
    PieCtrlRegs.PIEIER3.bit.INTx1 = 1U;    /* ePWM1 INT */
    PieCtrlRegs.PIEIER9.bit.INTx1 = 1U;    /* SCI-A RX  */

    /* Enable CPU interrupt lines */
    IER |= M_INT3;     /* Group 3 — ePWM1 */
    IER |= M_INT9;     /* Group 9 — SCI-A */

    /* Re-enable global interrupts */
    EINT;

    EDIS;
}
