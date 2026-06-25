/*
 * =====================================================================
 * PLECS C-Script — PITNN Inference Client (Option B: PI power control)
 *
 * PASTE THIS ENTIRE FILE into the DECLARATIONS tab only.
 * The Start / Output / Terminate tabs have their own separate files.
 *
 * Block settings (Setup tab):
 *   Number of inputs  : 4
 *   Number of outputs : 5
 *   Sample time       : 500e-6
 *   Direct feedthrough: checked
 *
 * Inputs:
 *   0 = V1        Primary bus voltage      (V)
 *   1 = V2        Secondary bus voltage    (V)
 *   2 = I_out     Output current           (A)
 *   3 = P_ref_ext External power reference (W)
 *
 * Outputs:
 *   0 = phi1       Primary duty angle      (rad)
 *   1 = phi2       Secondary duty angle    (rad)
 *   2 = phi3       External phase shift    (rad)
 *   3 = P_ref_corr PI-corrected reference  (W)
 *   4 = P_measured V2 x I_out             (W)
 * =====================================================================
 */

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef unsigned int sock_t;
  #define SOCK_INVALID  ((sock_t)(~0))
  #define sock_close(s) closesocket(s)
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  typedef int sock_t;
  #define SOCK_INVALID  (-1)
  #define sock_close(s) close(s)
#endif

#include <stdio.h>
#include <string.h>
#include <math.h>

/* ── Server parameters ───────────────────────────────────────────── */
#define SERVER_PORT     9876
#define RECV_BUF_SIZE   256
#define SEND_BUF_SIZE   128

/* ── Angle and power bounds ──────────────────────────────────────── */
#define PHI12_NOM   2.98451302
#define PHI12_MIN   2.04203522
#define PHI12_MAX   3.11017082
#define PHI3_SAFE   0.10
#define PHI3_MIN    0.02
#define PHI3_MAX    1.50
#define P_MIN       500.0
#define P_MAX       3300.0
#define V1_LO       360.0
#define V1_HI       440.0
#define V2_LO       220.0
#define V2_HI       280.0

/* ── Persistent state ────────────────────────────────────────────── */
static sock_t g_sock      = (sock_t)(-1);
static int    g_connected = 0;
static int    g_started   = 0;
static int    g_wsinit    = 0;
static double g_phi1      = PHI12_NOM;
static double g_phi2      = PHI12_NOM;
static double g_phi3      = PHI3_SAFE;
static double g_p_corr    = P_MIN;
static double g_p_meas    = 0.0;
static int    g_calls     = 0;
static int    g_errors    = 0;
static char   g_recvbuf[RECV_BUF_SIZE];
static char   g_sendbuf[SEND_BUF_SIZE];

/* ── Helpers ─────────────────────────────────────────────────────── */
static double clamp_val(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static int server_connect(void)
{
    struct sockaddr_in addr;

#ifdef _WIN32
    if (!g_wsinit) {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            printf("[CScript] WSAStartup failed\n");
            return 0;
        }
        g_wsinit = 1;
    }
#endif

    g_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_sock == SOCK_INVALID) {
        printf("[CScript] socket() failed\n");
        return 0;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(SERVER_PORT);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(g_sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        printf("[CScript] Cannot connect on port %d "
               "— is pitnn_plecs_server.py running?\n", SERVER_PORT);
        sock_close(g_sock);
        g_sock = SOCK_INVALID;
        return 0;
    }

    printf("[CScript] Connected to PITNN server on 127.0.0.1:%d\n",
           SERVER_PORT);
    return 1;
}

static int server_query(double V1, double V2,
                        double I_out, double P_ref)
{
    int    n, total, r;
    double p1, p2, p3, pc, pm;

    n = sprintf(g_sendbuf, "%.4f,%.4f,%.4f,%.4f\n",
                V1, V2, I_out, P_ref);
    if (send(g_sock, g_sendbuf, n, 0) < 0)
        return 0;

    total = 0;
    while (total < RECV_BUF_SIZE - 1) {
        r = recv(g_sock, g_recvbuf + total, 1, 0);
        if (r <= 0) return 0;
        if (g_recvbuf[total] == '\n') {
            g_recvbuf[total] = '\0';
            break;
        }
        total++;
    }

    if (sscanf(g_recvbuf, "%lf,%lf,%lf,%lf,%lf",
               &p1, &p2, &p3, &pc, &pm) != 5)
        return 0;

    g_phi1   = clamp_val(p1, PHI12_MIN, PHI12_MAX);
    g_phi2   = clamp_val(p2, PHI12_MIN, PHI12_MAX);
    g_phi3   = clamp_val(p3, PHI3_MIN,  PHI3_MAX);
    g_p_corr = clamp_val(pc, P_MIN, P_MAX);
    g_p_meas = (pm < 0.0) ? 0.0 : pm;
    return 1;
}

static void do_start(void)
{
    g_phi1      = PHI12_NOM;
    g_phi2      = PHI12_NOM;
    g_phi3      = PHI3_SAFE;
    g_p_corr    = P_MIN;
    g_p_meas    = 0.0;
    g_calls     = 0;
    g_errors    = 0;
    g_connected = server_connect();
    if (!g_connected) {
        printf("[CScript] WARNING: No server — "
               "outputs will hold fallback angles.\n");
    }
}

static void do_output(double V1_raw, double V2_raw,
                      double I_out_raw, double P_ref_raw)
{
    double V1, V2, I_out, P_ref;

    V1    = clamp_val(V1_raw,    V1_LO, V1_HI);
    V2    = clamp_val(V2_raw,    V2_LO, V2_HI);
    I_out = (I_out_raw < 0.0) ? 0.0 : I_out_raw;
    P_ref = clamp_val(P_ref_raw, P_MIN, P_MAX);

    if (g_connected) {
        if (server_query(V1, V2, I_out, P_ref)) {
            g_errors = 0;
        } else {
            g_errors++;
            if (g_errors == 1)
                printf("[CScript] Server query failed at call %d\n",
                       g_calls);
            if (g_errors > 20) {
                sock_close(g_sock);
                g_sock      = SOCK_INVALID;
                g_connected = server_connect();
                g_errors    = 0;
            }
        }
    }
    g_calls++;
}

static void do_terminate(void)
{
    if (g_sock != SOCK_INVALID) {
        sock_close(g_sock);
        g_sock = SOCK_INVALID;
    }
#ifdef _WIN32
    if (g_wsinit) {
        WSACleanup();
        g_wsinit = 0;
    }
#endif
    printf("[CScript] Done. Calls: %d  Errors: %d\n",
           g_calls, g_errors);
}
