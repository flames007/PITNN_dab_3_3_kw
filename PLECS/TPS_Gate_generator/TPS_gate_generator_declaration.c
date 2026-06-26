#include <math.h>

#define FSW     100000.0
#define TSW     (1.0 / FSW)
#define TWO_PI  6.2831853071795864769

static double wrap_time(double x)
{
    while (x >= TSW) x -= TSW;
    while (x < 0.0)  x += TSW;
    return x;
}

static int square_state(double t)
{
    double tau = wrap_time(t);
    return (tau < 0.5 * TSW) ? 1 : 0;
}