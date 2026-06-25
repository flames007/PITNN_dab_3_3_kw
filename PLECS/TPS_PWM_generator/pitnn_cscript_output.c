double V1, V2, I_out, P_ref;

V1    = InputSignal(0, 0);
V2    = InputSignal(1, 0);
I_out = InputSignal(2, 0);
P_ref = InputSignal(3, 0);

do_output(V1, V2, I_out, P_ref);

OutputSignal(0, 0) = g_phi1;
OutputSignal(1, 0) = g_phi2;
OutputSignal(2, 0) = g_phi3;
OutputSignal(3, 0) = g_p_corr;
OutputSignal(4, 0) = g_p_meas;