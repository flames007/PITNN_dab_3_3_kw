double phi1 = InputSignal(0, 0);
double phi2 = InputSignal(1, 0);
double phi3 = InputSignal(2, 0);

double pL_u, pL_l;
double pR_u, pR_l;
double sL_u, sL_l;
double sR_u, sR_l;

/* Keep angles in a valid range. */
phi1 = tps_clamp(phi1, 0.0, TPS_TWO_PI);
phi2 = tps_clamp(phi2, 0.0, TPS_TWO_PI);
phi3 = tps_clamp(phi3, 0.0, TPS_TWO_PI);

/* TPS phase convention:
   Primary left leg      = 0
   Primary right leg     = phi1
   Secondary left leg    = phi3
   Secondary right leg   = phi3 + phi2
*/
tps_leg_gates(CurrentTime, 0.0,         &pL_u, &pL_l);
tps_leg_gates(CurrentTime, phi1,        &pR_u, &pR_l);
tps_leg_gates(CurrentTime, phi3,        &sL_u, &sL_l);
tps_leg_gates(CurrentTime, phi3 + phi2, &sR_u, &sR_l);

/* Output order matched to your MOSFET names. */
OutputSignal(0, 0) = pL_u;   /* FETD  */
OutputSignal(1, 0) = pL_l;   /* FETD1 */

OutputSignal(2, 0) = pR_u;   /* FETD2 */
OutputSignal(3, 0) = pR_l;   /* FETD3 */

OutputSignal(4, 0) = sL_u;   /* FETD4 */
OutputSignal(5, 0) = sL_l;   /* FETD5 */

OutputSignal(6, 0) = sR_u;   /* FETD7 */
OutputSignal(7, 0) = sR_l;   /* FETD6 */