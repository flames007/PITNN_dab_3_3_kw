#define TPS_FS      100000.0
#define TPS_TS      (1.0 / TPS_FS)
#define TPS_DT      50e-9
#define TPS_TWO_PI  6.2831853071795864769

static double tps_clamp(double x, double xmin, double xmax)
{
    if (x < xmin) return xmin;
    if (x > xmax) return xmax;
    return x;
}

static double tps_wrap_time(double x)
{
    while (x < 0.0) {
        x += TPS_TS;
    }

    while (x >= TPS_TS) {
        x -= TPS_TS;
    }

    return x;
}

static double tps_wrap_phase(double x)
{
    while (x < 0.0) {
        x += TPS_TWO_PI;
    }

    while (x >= TPS_TWO_PI) {
        x -= TPS_TWO_PI;
    }

    return x;
}

static void tps_leg_gates(double t, double phase, double *upper, double *lower)
{
    double delay;
    double tau;
    double cmd;
    double d0;
    double d1;

    phase = tps_wrap_phase(phase);

    delay = phase / TPS_TWO_PI * TPS_TS;
    tau = tps_wrap_time(t - delay);

    cmd = (tau < 0.5 * TPS_TS) ? 1.0 : 0.0;

    d0 = tau;
    if ((TPS_TS - tau) < d0) {
        d0 = TPS_TS - tau;
    }

    d1 = tau - 0.5 * TPS_TS;
    if (d1 < 0.0) {
        d1 = -d1;
    }

    if ((d0 < TPS_DT) || (d1 < TPS_DT)) {
        *upper = 0.0;
        *lower = 0.0;
    } else {
        *upper = cmd;
        *lower = 1.0 - cmd;
    }
}