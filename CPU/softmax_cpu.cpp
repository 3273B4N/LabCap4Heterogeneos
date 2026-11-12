#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>

#include <chrono>

// Calcula el softmax en FP32, primero busca el máximo, después calcula las exponenciales con su suma y divide cada valor entre esa suma.
static void softmax_fp32(const float *x, float *y, int n) {
    float max_value = x[0];
    for (int i = 0; i < n; i++) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i] - max_value);
        sum += y[i];
    }

    for (int i = 0; i < n; i++) {
        y[i] /= sum;
    }
}

// Calcula el softmax en FP16. Los datos se guardan en half, pero las cuentas se hacen en float porque la CPU de la Nano no opera en FP16. Con sum_fp16 la suma se acumula en half.
static void softmax_fp16(const __fp16 *x, __fp16 *y, int n, bool sum_fp16) {
    float max_value = x[0];
    for (int i = 0; i < n; i++) {
        if (x[i] > max_value) {
            max_value = x[i];
        }
    }

    float sum = 0.0f;
    __fp16 sum_h = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = expf((float)x[i] - max_value);
        y[i] = e;
        if (sum_fp16) {
            sum_h = sum_h + y[i];
        } else {
            sum += e;
        }
    }
    if (sum_fp16) {
        sum = sum_h;
    }

    for (int i = 0; i < n; i++) {
        y[i] = (float)y[i] / sum;
    }
}

// Carga los primeros n valores del vector. Devuelve false si no logra leerlos todos.
static bool read_vector(const char *path, float *x, int n) {
    FILE *f = std::fopen(path, "r");
    if (!f) {
        return false;
    }
    int i = 0;
    while (i < n && std::fscanf(f, "%f", &x[i]) == 1) {
        i++;
    }
    std::fclose(f);
    return i == n;
}

// Calcula el softmax en double para usarlo como referencia al medir el error.
static void softmax_reference(const float *input, double *output, int n) {
    double max_value = input[0];
    for (int i = 1; i < n; ++i) {
        max_value = std::fmax(max_value, (double)input[i]);
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        output[i] = std::exp((double)input[i] - max_value);
        sum += output[i];
    }

    for (int i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

// Revisa que la suma sea aproximadamente 1, que los valores estén entre 0 y 1 y que el máximo quede en la misma posición que en la referencia. También calcula el error absoluto y relativo.
static bool verify_result(const float *cpu, const double *ref, int n, double tolerance,
                          double *max_abs, double *max_rel, double *sum) {
    *max_abs = 0.0;
    *max_rel = 0.0;
    *sum = 0.0;
    bool ok = true;
    int max_cpu = 0;
    int max_ref = 0;

    for (int i = 0; i < n; ++i) {
        *sum += cpu[i];

        double error = std::fabs((double)cpu[i] - ref[i]);
        if (error > *max_abs) {
            *max_abs = error;
        }
        if (error / ref[i] > *max_rel) {
            *max_rel = error / ref[i];
        }

        if (cpu[i] < 0.0f || cpu[i] > 1.0f) {
            std::fprintf(stderr, "Valor fuera de [0,1] en indice %d: %.6f\n", i, cpu[i]);
            ok = false;
        }
        if (cpu[i] > cpu[max_cpu]) max_cpu = i;
        if (ref[i] > ref[max_ref]) max_ref = i;
    }

    if (std::fabs(*sum - 1.0) > tolerance) {
        std::fprintf(stderr, "La suma no es 1: %.7f\n", *sum);
        ok = false;
    }
    if (cpu[max_ref] != cpu[max_cpu]) {
        std::fprintf(stderr, "Posicion del maximo distinta: cpu %d, referencia %d\n", max_cpu, max_ref);
        ok = false;
    }

    return ok;
}

// Lee los argumentos, prepara los datos, hace un calentamiento y luego mide varias corridas e imprime los resultados.
int main(int argc, char **argv) {
    int n = 100;
    int reps = 10000;
    int runs = 5;
    const char *precision = "fp32";
    const char *measure = "function";
    const char *path = "Vector_n1000_20.txt";
    bool sum_fp16 = false;

    for (int i = 1; i < argc; ++i){
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc){
            n = std::atoi(argv[i + 1]);
            i ++;
        } else if (std::strcmp(argv[i], "--measure") == 0 && i + 1 < argc){
            measure = argv[i + 1];
            i ++;
        } else if (std::strcmp(argv[i], "--precision") == 0 && i + 1 < argc){
            precision = argv[i + 1];
            i ++;
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc){
            reps = std::atoi(argv[i + 1]);
            i ++;
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc){
            runs = std::atoi(argv[i + 1]);
            i ++;
        } else if (std::strcmp(argv[i], "--vector") == 0 && i + 1 < argc){
            path = argv[i + 1];
            i ++;
        } else if (std::strcmp(argv[i], "--sum-fp16") == 0){
            sum_fp16 = true;
        } else {
            std::fprintf(stderr, "Argumento desconocido: %s\n", argv[i]);
            std::fprintf(stderr, "Uso: %s --precision fp32|fp16 --n N --reps R "
                                 "[--measure function|program] [--runs 5] [--sum-fp16] [--vector archivo]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    bool fp16_mode = std::strcmp(precision, "fp16") == 0;
    if (!fp16_mode && std::strcmp(precision, "fp32") != 0) {
        std::fprintf(stderr, "Solo se puede modo FP16 o FP32\n");
        return EXIT_FAILURE;
    }
    bool program_mode = std::strcmp(measure, "program") == 0;
    if (!program_mode && std::strcmp(measure, "function") != 0) {
        std::fprintf(stderr, "--measure debe ser function o program\n");
        return EXIT_FAILURE;
    }
    if (sum_fp16 && !fp16_mode) {
        std::fprintf(stderr, "--sum-fp16 solo se usa con --precision fp16\n");
        return EXIT_FAILURE;
    }
    const char *label = sum_fp16 ? "fp16sum" : precision;
    if (n < 1 || n > 1000 || reps < 1 || runs < 1) {
        std::fprintf(stderr, "n debe estar entre 1 y 1000\n");
        return EXIT_FAILURE;
    }

    float *h_input = static_cast<float *>(std::malloc(n * sizeof(float)));
    float *h_output = static_cast<float *>(std::malloc(n * sizeof(float)));
    double *h_reference = static_cast<double *>(std::malloc(n * sizeof(double)));
    __fp16 *h_input_h = static_cast<__fp16 *>(std::malloc(n * sizeof(__fp16)));
    __fp16 *h_output_h = static_cast<__fp16 *>(std::malloc(n * sizeof(__fp16)));
    double *times = static_cast<double *>(std::malloc(runs * sizeof(double)));

    if (!h_input || !h_output || !h_reference || !h_input_h || !h_output_h || !times) {
        std::fprintf(stderr, "No se pudo reservar memoria\n");
        return EXIT_FAILURE;
    }

    if (!read_vector(path, h_input, n)) {
        std::fprintf(stderr, "No se pudo leer %d valores de %s\n", n, path);
        return EXIT_FAILURE;
    }
    std::printf("primeros valores del vector: %.4f %.4f %.4f %.4f %.4f\n",
                h_input[0], h_input[1 % n], h_input[2 % n], h_input[3 % n], h_input[4 % n]);

    for (int i = 0; i < n; ++i) {
        h_input_h[i] = h_input[i];
    }

    for (int rep = 0; rep < reps; ++rep) {
        if (fp16_mode) {
            softmax_fp16(h_input_h, h_output_h, n, sum_fp16);
        } else {
            softmax_fp32(h_input, h_output, n);
        }
    }

    double tolerance = fp16_mode ? 1e-2 : 1e-4;
    double max_abs = 0.0, max_rel = 0.0, sum = 0.0;
    bool ok = true;

    for (int run = 0; run < runs; ++run) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (program_mode) {
            if (!read_vector(path, h_input, n)) {
                std::fprintf(stderr, "No se pudo leer %s\n", path);
                return EXIT_FAILURE;
            }
            for (int i = 0; i < n; ++i) {
                h_input_h[i] = h_input[i];
            }
        }

        for (int rep = 0; rep < reps; ++rep) {
            if (fp16_mode) {
                softmax_fp16(h_input_h, h_output_h, n, sum_fp16);
            } else {
                softmax_fp32(h_input, h_output, n);
            }
        }

        if (fp16_mode) {
            for (int i = 0; i < n; ++i) {
                h_output[i] = h_output_h[i];
            }
        }

        if (program_mode) {
            softmax_reference(h_input, h_reference, n);
            ok = verify_result(h_output, h_reference, n, tolerance, &max_abs, &max_rel, &sum) && ok;
        }

        auto stop_time = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(stop_time - start_time).count();
        times[run] = total_ms * 1000.0 / reps;

        if (!program_mode) {
            softmax_reference(h_input, h_reference, n);
            ok = verify_result(h_output, h_reference, n, tolerance, &max_abs, &max_rel, &sum) && ok;
        }

        std::printf("%s, %d, %d, %s, %.4f, %.4f, %.3e, %d, %.3e, %.7f\n", label, n, reps, measure,
                    total_ms, times[run], max_abs, run, max_rel, sum);
    }

    std::sort(times, times + runs);
    std::printf("softmax precision=%s n=%d reps=%d measure=%s: %s max_error=%.3e rel_error=%.3e\n",
                label, n, reps, measure,
                ok ? "OK" : "ERROR", max_abs, max_rel);
    std::printf("us por llamada: mediana %.4f  min %.4f  max %.4f\n",
                times[runs / 2], times[0], times[runs - 1]);

    std::free(h_input);
    std::free(h_output);
    std::free(h_reference);
    std::free(h_input_h);
    std::free(h_output_h);
    std::free(times);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}