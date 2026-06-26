double phi1 = InputSignal(0, 0);
double phi2 = InputSignal(0, 1);
double phi3 = InputSignal(0, 2);

double d1 = (phi1 / TWO_PI) * TSW;
double d2 = (phi2 / TWO_PI) * TSW;
double d3 = (phi3 / TWO_PI) * TSW;

int pL = square_state(CurrentTime);
int pR = square_state(CurrentTime - d1);

int sL = square_state(CurrentTime + d3);
int sR = square_state(CurrentTime + d3 - d2);

OutputSignal(0, 0) = pL;
OutputSignal(1, 0) = 1 - pL;
OutputSignal(2, 0) = pR;
OutputSignal(3, 0) = 1 - pR;

OutputSignal(4, 0) = sL;
OutputSignal(5, 0) = 1 - sL;
OutputSignal(6, 0) = sR;
OutputSignal(7, 0) = 1 - sR;