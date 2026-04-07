/*
 * pitnn_socket_client.h
 * ================================================================
 * PITNN DAB Converter — PLECS C-Script Socket Client
 * Method 4: Connects to pitnn_socket_server.py from a C-Script block.
 *
 * HOW TO USE
 * ──────────
 * 1. Start pitnn_socket_server.py on the inference PC.
 * 2. In PLECS, add a C-Script block with:
 *      Inputs  (4): V1, V2, iL, Pref
 *      Outputs (3): phi1, phi2, phi3
 * 3. Copy Sections 1-3 into the C-Script editor tabs.
 * 4. Set PITNN_HOST to match the server's IP address.
 *    Use "127.0.0.1" if server runs on the same machine.
 *
 * PROTOCOL
 * ─────────
 *   Send    : 4 x float32 = 16 bytes  [V1, V2, iL, Pref]
 *   Receive : 3 x float32 = 12 bytes  [phi1, phi2, phi3]
 *
 * ================================================================
 */


/* ═══════════════════════════════════════════════════════════════
   SECTION 1 — DECLARATIONS
   ═══════════════════════════════════════════════════════════════ */

/* Platform socket headers */
#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef SOCKET sock_t;
  #define SOCK_INVALID INVALID_SOCKET
  #define SOCK_ERR     SOCKET_ERROR
#else
  #include <sys/socket.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  typedef int sock_t;
  #define SOCK_INVALID (-1)
  #define SOCK_ERR     (-1)
  #define closesocket  close
#endif

#include <string.h>
#include <stdio.h>

/* ── Server connection settings ───────────────────────────────── */
#define PITNN_HOST  "127.0.0.1"   /* IP of machine running pitnn_socket_server.py */
#define PITNN_PORT  9999          /* Must match PORT in pitnn_socket_server.py     */

#define PITNN_PHI12     2.98451302f
#define PITNN_PHI_MIN   0.05f
#define PITNN_PHI3_MAX  1.50f

/* Persistent socket handle */
static sock_t  pitnn_sock    = SOCK_INVALID;
static int     pitnn_connected = 0;
static float   pitnn_phi3_prev = 0.22f;

/* Helper: send exactly n bytes */
static int pitnn_send_all(sock_t s, const char *buf, int n) {
    int sent = 0;
    while (sent < n) {
        int r = send(s, buf + sent, n - sent, 0);
        if (r <= 0) return -1;
        sent += r;
    }
    return 0;
}

/* Helper: receive exactly n bytes */
static int pitnn_recv_all(sock_t s, char *buf, int n) {
    int recvd = 0;
    while (recvd < n) {
        int r = recv(s, buf + recvd, n - recvd, 0);
        if (r <= 0) return -1;
        recvd += r;
    }
    return 0;
}

/* Connect to pitnn_socket_server.py */
static int pitnn_connect(void) {
    struct sockaddr_in addr;
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) return -1;
#endif
    pitnn_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (pitnn_sock == SOCK_INVALID) return -1;

    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(PITNN_PORT);
    inet_pton(AF_INET, PITNN_HOST, &addr.sin_addr);

    if (connect(pitnn_sock, (struct sockaddr*)&addr, sizeof(addr)) == SOCK_ERR) {
        closesocket(pitnn_sock);
        pitnn_sock = SOCK_INVALID;
        return -1;
    }
    pitnn_connected = 1;
    return 0;
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 2 — OUTPUT FUNCTION
   ═══════════════════════════════════════════════════════════════ */

{
    float V1   = (float)u[0];
    float V2   = (float)u[1];
    float iL   = (float)u[2];
    float Pref = (float)u[3];
    float phi3 = pitnn_phi3_prev;

    /* Connect on first call */
    if (!pitnn_connected) {
        if (pitnn_connect() != 0) {
            /* Connection failed — output safe defaults */
            y[0] = PITNN_PHI12;
            y[1] = PITNN_PHI12;
            y[2] = PITNN_PHI3_MAX * 0.15f;   /* safe low-power phi3 */
            return;
        }
    }

    /* ── Send 16 bytes: [V1, V2, iL, Pref] ────────────────────── */
    float send_buf[4] = {V1, V2, iL, Pref};
    if (pitnn_send_all(pitnn_sock, (char*)send_buf, 16) != 0) {
        pitnn_connected = 0;
        closesocket(pitnn_sock);
        pitnn_sock = SOCK_INVALID;
    }

    /* ── Receive 12 bytes: [phi1, phi2, phi3] ──────────────────── */
    float recv_buf[3] = {PITNN_PHI12, PITNN_PHI12, phi3};
    if (pitnn_connected) {
        if (pitnn_recv_all(pitnn_sock, (char*)recv_buf, 12) != 0) {
            pitnn_connected = 0;
            closesocket(pitnn_sock);
            pitnn_sock = SOCK_INVALID;
        }
    }

    /* ── Clamp and store ───────────────────────────────────────── */
    phi3 = recv_buf[2];
    if (phi3 < PITNN_PHI_MIN)  phi3 = PITNN_PHI_MIN;
    if (phi3 > PITNN_PHI3_MAX) phi3 = PITNN_PHI3_MAX;
    pitnn_phi3_prev = phi3;

    y[0] = (double)recv_buf[0];   /* phi1 */
    y[1] = (double)recv_buf[1];   /* phi2 */
    y[2] = (double)phi3;          /* phi3 */
}


/* ═══════════════════════════════════════════════════════════════
   SECTION 3 — RESET FUNCTION
   ═══════════════════════════════════════════════════════════════ */

{
    if (pitnn_sock != SOCK_INVALID) {
        closesocket(pitnn_sock);
        pitnn_sock = SOCK_INVALID;
    }
    pitnn_connected = 0;
    pitnn_phi3_prev = 0.22f;
#ifdef _WIN32
    WSACleanup();
#endif
}
