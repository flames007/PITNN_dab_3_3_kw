/*
 * PITNN DAB Converter — C++ Inference (Option 2)
 * ================================================
 * Real-time PITNN controller using LibTorch.
 * Runs on NVIDIA Jetson, embedded Linux, or any platform
 * with the LibTorch C++ library installed.
 *
 * Required files (same directory as executable):
 *     pitnn_scripted.pt   — TorchScript model
 *     pitnn_mu.npy        — normalisation means    (load via load_npy_array)
 *     pitnn_sigma.npy     — normalisation std devs (load via load_npy_array)
 *
 * Build:
 *     mkdir build && cd build
 *     cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
 *     make -j4
 *
 * Run:
 *     ./pitnn_inference
 *
 * Copyright (c) 2026 Chukwuemeka Nzeadibe
 * Mississippi State University — All Rights Reserved
 */

#include <torch/script.h>
#include <iostream>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS  (must match values used during training)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr float V1_NOM    = 800.0f;
static constexpr float V2_NOM    = 800.0f;
static constexpr float FSW       = 100000.0f;
static constexpr float PI_F      = 3.14159265358979f;
static constexpr float PHI12_MIN = PI_F * 0.65f;   // 2.0420 rad — lower bound phi1/phi2
static constexpr float PHI12_MAX = PI_F * 0.99f;   // 3.1102 rad — upper bound phi1/phi2
static constexpr float PHI12_NOM = PI_F * 0.95f;   // nominal seed for buffer priming
static constexpr float PHI_MIN   = 0.02f;           // lower bound for phi3
static constexpr float PHI3_MAX  = 1.50f;
static constexpr int   SEQ_LEN   = 20;
static constexpr int   N_FEAT    = 8;

// ─────────────────────────────────────────────────────────────────────────────
// NORMALISATION CONSTANTS
// NOTE: phi1/phi2 now vary across [PHI12_MIN, PHI12_MAX] — the mu/sigma values
// below MUST be regenerated after retraining. Run:
//     python pitnn_inspect_exports.py
// and paste the printed C++ arrays here.
// Feature order: [V1, V2, iL, phi1, phi2, phi3, Pref, V1V2/Vnom2]
// ─────────────────────────────────────────────────────────────────────────────
static const float MU[N_FEAT] = {
    800.02514648f,   // V1   (V)
    800.20202637f,   // V2   (V)
     25.38187981f,   // iL   (A)
      2.99313903f,   // phi1 (rad)
      2.99313903f,   // phi2 (rad)
      0.47853500f,   // phi3 (rad)
  37593.66796875f,   // Pref (W)
      1.00037396f    // V1*V2/Vnom2
};

static const float SIGMA[N_FEAT] = {
     46.25230789f,   // V1
     46.19021606f,   // V2
     15.18008804f,   // iL
      0.00862500f,   // phi1
      0.00862500f,   // phi2
      0.29205400f,   // phi3
  18769.15234375f,   // Pref
      0.08148100f    // V1*V2/Vnom2
};


// ─────────────────────────────────────────────────────────────────────────────
// SIMPLE .npy READER  (for float32 1D arrays only)
// ─────────────────────────────────────────────────────────────────────────────
static bool load_npy_float32(const std::string& path,
                              std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; return false; }

    // Skip npy magic (6) + version (2) + header_len (2) = 10 bytes, then header
    char magic[7]; f.read(magic, 6);
    uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint16_t hlen; f.read((char*)&hlen, 2);
    std::string header(hlen, ' '); f.read(&header[0], hlen);

    // Find the shape from the header string
    size_t pos = header.find("'shape': (");
    if (pos == std::string::npos) { std::cerr << "Bad npy header\n"; return false; }
    size_t end = header.find(")", pos);
    std::string shape_str = header.substr(pos + 10, end - pos - 10);
    // Remove trailing comma if present
    if (!shape_str.empty() && shape_str.back() == ',')
        shape_str.pop_back();
    size_t n = std::stoul(shape_str);

    out.resize(n);
    f.read((char*)out.data(), n * sizeof(float));
    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// PITNN CONTROLLER CLASS
// ─────────────────────────────────────────────────────────────────────────────
class PITNNController {
public:
    // Constructor — loads model from disk once
    explicit PITNNController(const std::string& model_path = "pitnn_scripted.pt",
                              bool use_cuda = false) {
        _device = (use_cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;

        std::cout << "[PITNN] Loading: " << model_path << std::endl;
        try {
            _model = torch::jit::load(model_path, _device);
            _model.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error(
                std::string("Failed to load model: ") + e.what());
        }

        // Copy compile-time constants into instance arrays
        for (int i = 0; i < N_FEAT; i++) {
            _mu[i]    = MU[i];
            _sigma[i] = SIGMA[i];
        }

        reset();
        std::cout << "[PITNN] Ready on "
                  << (_device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    }

    // Clear history buffer — call when starting a new operating scenario
    void reset() {
        for (int t = 0; t < SEQ_LEN; t++)
            for (int f = 0; f < N_FEAT; f++)
                _buffer[t][f] = 0.0f;
        _phi1_prev = PHI12_NOM;   // seed with nominal inner duty
        _phi2_prev = PHI12_NOM;
        _phi3_prev = 0.22f;
    }

    /*
     * step() — call once per switching cycle
     *
     * Parameters:
     *   V1    : primary DC bus voltage (V)   — from ADC
     *   V2    : secondary DC bus voltage (V) — from ADC
     *   iL    : inductor current (A)         — from current sensor
     *   Pref  : power reference (W)          — from outer PI loop
     *
     * Returns:
     *   {phi1, phi2, phi3} in radians — all three independently predicted
     *   phi1 ∈ [PHI12_MIN, PHI12_MAX] — primary bridge inner duty
     *   phi2 ∈ [PHI12_MIN, PHI12_MAX] — secondary bridge inner duty
     *   phi3 ∈ [PHI_MIN,   PHI3_MAX]  — external phase shift → gate drive delay
     */
    std::array<float, 3> step(float V1, float V2, float iL, float Pref) {

        // Build 8-feature state vector using previous predicted angles
        float v_ratio = (V1 * V2) / (V1_NOM * V2_NOM);
        float feat[N_FEAT] = {
            V1, V2, iL, _phi1_prev, _phi2_prev, _phi3_prev, Pref, v_ratio
        };

        // Normalise: (x - mu) / sigma
        float feat_norm[N_FEAT];
        for (int i = 0; i < N_FEAT; i++)
            feat_norm[i] = (feat[i] - _mu[i]) / _sigma[i];

        // Shift buffer left by one step, append new state at end
        for (int t = 0; t < SEQ_LEN - 1; t++)
            for (int f = 0; f < N_FEAT; f++)
                _buffer[t][f] = _buffer[t+1][f];
        for (int f = 0; f < N_FEAT; f++)
            _buffer[SEQ_LEN-1][f] = feat_norm[f];

        // Build input tensor (1, SEQ_LEN, N_FEAT) — copy from buffer
        auto opts  = torch::TensorOptions().dtype(torch::kFloat32);
        auto input = torch::from_blob(_buffer,
                                       {1, SEQ_LEN, N_FEAT}, opts)
                         .clone()
                         .to(_device);

        // PITNN forward pass
        torch::NoGradGuard no_grad;
        auto output = _model.forward({input}).toTensor()
                            .squeeze()
                            .to(torch::kCPU);

        float phi1 = output[0].item<float>();
        float phi2 = output[1].item<float>();
        float phi3 = output[2].item<float>();

        _phi1_prev = phi1;
        _phi2_prev = phi2;
        _phi3_prev = phi3;
        return {phi1, phi2, phi3};
    }

    // Convert phi3 (rad) to gate drive phase delay in microseconds
    static float phi3_to_delay_us(float phi3) {
        return phi3 / (2.0f * PI_F * FSW) * 1e6f;
    }

    // Convert phi1 (rad) to inner duty cycle percentage
    static float phi1_to_duty_pct(float phi1) {
        return (phi1 / PI_F) * 100.0f;
    }

private:
    torch::jit::script::Module _model;
    torch::Device _device{torch::kCPU};
    float _mu[N_FEAT];
    float _sigma[N_FEAT];
    float _buffer[SEQ_LEN][N_FEAT];
    float _phi1_prev;
    float _phi2_prev;
    float _phi3_prev;
};


// ─────────────────────────────────────────────────────────────────────────────
// DEMO
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "====================================================" << std::endl;
    std::cout << "  PITNN DAB Converter — C++ Inference (Option 2)"     << std::endl;
    std::cout << "====================================================" << std::endl;

    // Optional: load mu/sigma from .npy files at runtime
    // (overrides the compile-time constants above if files are found)
    std::vector<float> mu_runtime, sigma_runtime;
    if (load_npy_float32("pitnn_mu.npy", mu_runtime) &&
        load_npy_float32("pitnn_sigma.npy", sigma_runtime)) {
        std::cout << "[PITNN] Loaded normalisation from .npy files" << std::endl;
    } else {
        std::cout << "[PITNN] Using compile-time normalisation constants" << std::endl;
    }

    // Create controller (set use_cuda=true if CUDA is available)
    PITNNController ctrl("pitnn_scripted.pt", /*use_cuda=*/true);

    // Test operating conditions
    struct Scenario {
        float V1, V2, Pref;
        const char* label;
    };
    Scenario scenarios[] = {
        {800, 800, 10000, "10kW nominal"},
        {760, 840,  8000, "8kW V-variation"},
        {800, 800, 20000, "20kW"},
        {800, 800,  5000, "5kW light load"},
        {880, 720, 50000, "50kW high V"},
        {800, 800, 30000, "30kW mid load"},
        {840, 760, 40000, "40kW asymm V"},
        {720, 880, 15000, "15kW off-voltage"},
    };

    printf("\n%-22s %5s %5s %7s  %9s  %10s  %6s  %8s\n",
           "Condition", "V1", "V2", "Pref",
           "phi3(rad)", "delay(us)", "duty%", "t(us)");
    printf("%s\n", std::string(80, '-').c_str());

    for (const auto& s : scenarios) {
        ctrl.reset();
        float iL_est = s.V1 * s.V2 / (V1_NOM * V2_NOM) * 10.7f * 0.3f;

        // Warm up the buffer
        for (int w = 0; w < 5; w++)
            ctrl.step(s.V1, s.V2, iL_est, s.Pref);

        // Timed inference
        auto t0   = std::chrono::high_resolution_clock::now();
        auto phi  = ctrl.step(s.V1, s.V2, iL_est, s.Pref);
        auto t1   = std::chrono::high_resolution_clock::now();
        float t_us = std::chrono::duration<float, std::micro>(t1 - t0).count();

        float delay_us  = PITNNController::phi3_to_delay_us(phi[2]);
        float duty_pct  = PITNNController::phi1_to_duty_pct(phi[0]);

        printf("%-22s %5.0f %5.0f %7.0f  %9.4f  %10.3f  %5.1f%%  %8.1f\n",
               s.label, s.V1, s.V2, s.Pref,
               phi[2], delay_us, duty_pct, t_us);
    }

    std::cout << std::endl;
    std::cout << "All three angles phi1, phi2, phi3 independently predicted each cycle."
              << std::endl;
    std::cout << "Gate drive: phi3 -> phase delay  |  phi1 -> primary duty  |  phi2 -> secondary duty"
              << std::endl;

    // ── Integration example: minimal real-time loop ───────────────────────
    std::cout << "\n--- Integration Example (replace sensor stubs with real reads) ---\n";
    ctrl.reset();
    float Pref = 20000.0f;

    for (int cycle = 0; cycle < 10; cycle++) {
        // Replace these with real ADC reads
        float V1_meas = 800.0f + (cycle % 3) * 1.5f;
        float V2_meas = 798.0f + (cycle % 2) * 1.0f;
        float iL_meas = 27.5f  + (cycle % 4) * 0.5f;

        auto phi = ctrl.step(V1_meas, V2_meas, iL_meas, Pref);

        float phase_delay_s = phi[2] / (2.0f * PI_F * FSW);
        float duty          = phi[0] / PI_F;

        if (cycle % 3 == 0) {
            printf("Cycle %3d: phi3=%.4f rad  delay=%.3fus  duty=%.1f%%\n",
                   cycle, phi[2], phase_delay_s * 1e6f, duty * 100.0f);
        }

        // Apply to gate drive hardware:
        // pwm_set_primary_duty(duty);
        // pwm_set_phase_delay(phase_delay_s);
        // pwm_commit();
    }

    return 0;
}
