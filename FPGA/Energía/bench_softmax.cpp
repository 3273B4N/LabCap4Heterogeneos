#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <chrono>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#ifdef FP16
typedef __fp16 T;
static const char* PREC = "fp16";
#else
typedef float T;
static const char* PREC = "fp32";
#endif

using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
struct Row { double fill, h2d, launch, wait, d2h, gather, total; };
struct St { double med, mn, mx; };
static St stat(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double med = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return {med, v.front(), v.back()};
}
static long long epoch_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "uso: " << argv[0] << " <xclbin> <parts 1|4> <variante> <MHz> <csv>"
                     " [--reps M] [--warmup W] [--energy S] [--energy-mode full|kernel|e2e]\n"
                     "  full  : llamada completa con relleno y lectura/conversion en el host\n"
                     "  kernel: solo lanzamiento y espera del kernel, con los buffers ya cargados\n"
                     "  e2e   : sync a dispositivo + kernel + sync de vuelta (sin relleno ni conversion);\n"
                     "          equivale al end-to-end de la GPU\n";
        return 1;
    }
    const std::string xclbin = argv[1];
    const int P = std::atoi(argv[2]);
    const std::string variant = argv[3], mhz = argv[4], csv = argv[5];
    int reps = 30, warmup = 5;
    double energy_s = 0.0;
    std::string emode = "full";
    for (int i = 6; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--energy" && i + 1 < argc) energy_s = std::atof(argv[++i]);
        else if (a == "--energy-mode" && i + 1 < argc) emode = argv[++i];
    }
    if ((P != 1 && P != 4) || reps < 5) {
        std::cerr << "parts debe ser 1 o 4 y reps >= 5\n";
        return 1;
    }
    if (emode != "full" && emode != "kernel" && emode != "e2e") {
        std::cerr << "--energy-mode debe ser full, kernel o e2e\n";
        return 1;
    }
    const int Ns[4] = {10, 100, 500, 1000};

    // entrada comun: semilla 42, uniforme en [-4,4]; se usan los primeros N elementos
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
    std::vector<T> xin(1000);
    {
        std::ofstream fi("input_seed42.txt");
        for (int i = 0; i < 1000; i++) {
            double d = dist(gen);
            xin[i] = (T)d;
            fi << std::setprecision(17) << d << "\n";
        }
    }

    try {
        auto dev  = xrt::device(0);
        auto uuid = dev.load_xclbin(xclbin);
        auto krn  = xrt::kernel(dev, uuid, "softmax_hls");

        const int PER = (P == 1) ? 1000 : 256;
        std::vector<xrt::bo> bx, by;
        std::vector<T*> px, py;
        for (int k = 0; k < P; k++) {
            bx.emplace_back(dev, PER * sizeof(T), krn.group_id(k));
            by.emplace_back(dev, PER * sizeof(T), krn.group_id(P + k));
            px.push_back(bx[k].map<T*>());
            py.push_back(by[k].map<T*>());
            for (int j = 0; j < PER; j++) { px[k][j] = (T)0; py[k][j] = (T)0; }
        }
        std::vector<double> out(1000);

        auto fail = [&]() {
            std::cerr << "el kernel NO termino (timeout).\n"
                         "recuperar con: sudo xmutil unloadapp && sudo xmutil loadapp <app>\n";
            return 2;
        };
        auto launch = [&](int N) {
            xrt::run r;
            if (P == 1) r = krn(bx[0], by[0], (unsigned int)N);
            else r = krn(bx[0], bx[1], bx[2], bx[3], by[0], by[1], by[2], by[3], (unsigned int)N);
            return r;
        };
        // llamada completa, con tiempo por etapa
        auto call = [&](int N, Row& r) -> bool {
            auto t0 = clk::now();
            for (int i = 0; i < N; i++) px[i % P][i / P] = xin[i];
            auto t1 = clk::now();
            for (auto& b : bx) b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto t2 = clk::now();
            auto run = launch(N);
            auto t3 = clk::now();
            auto st = run.wait(3000u);
            auto t4 = clk::now();
            if (static_cast<int>(st) != 4) return false;   // 4 = COMPLETED
            for (auto& b : by) b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto t5 = clk::now();
            for (int i = 0; i < N; i++) out[i] = (double)py[i % P][i / P];
            auto t6 = clk::now();
            r = {us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4), us(t4, t5), us(t5, t6), us(t0, t6)};
            return true;
        };
        // solo el kernel (buffers ya cargados)
        auto kcall = [&](int N) -> bool {
            auto run = launch(N);
            return static_cast<int>(run.wait(3000u)) == 4;
        };
        // end-to-end: sync a dispositivo + kernel + sync de vuelta, sin relleno ni conversion en el host
        auto ecall = [&](int N) -> bool {
            for (auto& b : bx) b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto run = launch(N);
            if (static_cast<int>(run.wait(3000u)) != 4) return false;
            for (auto& b : by) b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            return true;
        };
        auto check = [&](int N, double& sum, double& eabs, bool& amok) {
            std::vector<double> ref(N);
            double m = -1e30, s = 0.0;
            for (int i = 0; i < N; i++) m = std::max(m, (double)xin[i]);
            for (int i = 0; i < N; i++) { ref[i] = std::exp((double)xin[i] - m); s += ref[i]; }
            sum = 0; eabs = 0;
            int am = 0, ar = 0;
            for (int i = 0; i < N; i++) {
                ref[i] /= s;
                sum += out[i];
                eabs = std::max(eabs, std::fabs(out[i] - ref[i]));
                if (out[i] > out[am]) am = i;
                if (ref[i] > ref[ar]) ar = i;
            }
            amok = (am == ar);
        };

        std::ofstream f(csv);
        f << "variant,prec,mhz,N,rep,t_fill_us,t_h2d_us,t_launch_us,t_wait_us,t_d2h_us,t_gather_us,t_total_us\n";
        std::printf("\n== %s %s %s MHz parts=%d warmup=%d reps=%d ==\n",
                    variant.c_str(), PREC, mhz.c_str(), P, warmup, reps);

        for (int N : Ns) {
            Row r;
            for (int w = 0; w < warmup; w++) if (!call(N, r)) return fail();
            std::vector<double> tot, fil, h2d, exe, d2h, gat;
            for (int k = 0; k < reps; k++) {
                if (!call(N, r)) return fail();
                f << variant << "," << PREC << "," << mhz << "," << N << "," << k << ","
                  << r.fill << "," << r.h2d << "," << r.launch << "," << r.wait << ","
                  << r.d2h << "," << r.gather << "," << r.total << "\n";
                tot.push_back(r.total); fil.push_back(r.fill); h2d.push_back(r.h2d);
                exe.push_back(r.launch + r.wait); d2h.push_back(r.d2h); gat.push_back(r.gather);
            }
            double sum, eabs; bool amok;
            check(N, sum, eabs, amok);
            St st = stat(tot);
            std::printf("N=%4d total med/min/max = %.1f/%.1f/%.1f us | fill %.1f | h2d %.1f | "
                        "launch+wait %.1f | d2h %.1f | gather %.1f | suma=%.6f err_abs=%.3g argmax=%d\n",
                        N, st.med, st.mn, st.mx, stat(fil).med, stat(h2d).med, stat(exe).med,
                        stat(d2h).med, stat(gat).med, sum, eabs, (int)amok);
        }

        if (energy_s > 0) {
            std::ofstream fe(csv + "_energy.csv");
            fe << "variant,prec,mhz,N,mode,calls,t_start_epoch_ns,t_end_epoch_ns\n";
            for (int N : Ns) {
                Row r;
                if (!call(N, r)) return fail();          // deja los buffers cargados
                long long calls = 0;
                auto tstop = clk::now() + std::chrono::duration_cast<clk::duration>(
                                 std::chrono::duration<double>(energy_s));
                long long ts = epoch_ns();
                while (clk::now() < tstop) {
                    bool ok = (emode == "kernel") ? kcall(N) : (emode == "e2e") ? ecall(N) : call(N, r);
                    if (!ok) return fail();
                    calls++;
                }
                long long te = epoch_ns();
                fe << variant << "," << PREC << "," << mhz << "," << N << "," << emode << ","
                   << calls << "," << ts << "," << te << "\n";
                std::printf("energia N=%d modo=%s llamadas=%lld inicio=%lld fin=%lld\n",
                            N, emode.c_str(), calls, ts, te);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "excepcion: " << e.what() << "\n";
        return 3;
    }
    return 0;
}