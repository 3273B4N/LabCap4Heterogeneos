#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include <cstring>


#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

__global__ void softmax_rows_kernel(const float *input, float *output, int rows,
                                    int cols) {
    extern __shared__ float cache[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
	    local_max = fmaxf(local_max, input[row * cols + col]);
    }

    cache[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
	cache[tid] = fmaxf(cache[tid], cache[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = cache[0];

    __syncthreads();
    
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
	output[idx] = expf(input[idx] - row_max);
	local_sum += output[idx];
    }

    cache[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
		cache[tid] += cache[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = cache[0];

    for (int col = tid; col < cols; col += blockDim.x) {
        int idx = row * cols + col;
        output[idx] = output[idx] / row_sum;
    }
}

static void fill_matrix(float *x, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            x[idx] = 0.01f * static_cast<float>((row * 17 + col * 31) % 101);
        }
    }
}

static void cpu_softmax(const float *input, float *output, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        float max_value = input[row * cols];
        for (int col = 1; col < cols; ++col) {
            max_value = std::fmax(max_value, input[row * cols + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            output[idx] = std::exp(input[idx] - max_value);
            sum += output[idx];
        }

        for (int col = 0; col < cols; ++col) {
            output[row * cols + col] /= sum;
        }
    }
}

static bool verify_result(const float *gpu, const float *cpu, int count) {
    for (int i = 0; i < count; ++i) {
        if (std::fabs(gpu[i] - cpu[i]) > 1e-4f) {
            std::fprintf(stderr,
                         "Error en indice %d: obtenido %.6f, esperado %.6f\n",
                         i, gpu[i], cpu[i]);
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv) {
    int n = 100;
    int reps = 1;
    const char *precision = "fp32";
    const char *measure = "kernel";

    for (int i = 1; i < argc; ++i){
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc){
            n = std::atoi(argv[i + 1]);
            i ++;
        } else if (std::strcmp(argv[i], "--measure") == 0 && i + 1 < argc){
            measure = argv[i +1];
            i ++;
        } else if (std::strcmp(argv[i], "--precision") == 0 && i + 1 < argc){
            precision = argv[i + 1];
            i ++;
        } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc){
            reps = std::atoi(argv[i + 1]);
            i ++;
        } 
    }

    if (std::strcmp(precision, "fp32") != 0) {
        std::fprintf(stderr, "Es una prueba solo con fp32\n");
        return EXIT_FAILURE;
    }

    int rows = 1;
    int cols = n;

    int count = rows * cols;
    size_t bytes = static_cast<size_t>(count) * sizeof(float);

    float *h_input = static_cast<float *>(std::malloc(bytes));
    float *h_output = static_cast<float *>(std::malloc(bytes));
    float *h_reference = static_cast<float *>(std::malloc(bytes));

    if (!h_input || !h_output || !h_reference) {
        std::fprintf(stderr, "No se pudo reservar memoria en CPU\n");
        return EXIT_FAILURE;
    }

    fill_matrix(h_input, rows, cols);
    cpu_softmax(h_input, h_reference, rows, cols);

    float *d_input = nullptr;
    float *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    int threads_per_block = 256;
    size_t shared_bytes = static_cast<size_t>(threads_per_block) * sizeof(float);

    softmax_rows_kernel<<<rows, threads_per_block, shared_bytes>>>(
        d_input, d_output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_result(h_output, h_reference, count);
    std::printf("softmax precision=%s n=%d reps=%d measure=%s: %s\n", precision, n, reps, measure,
                ok ? "OK" : "ERROR");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_input);
    std::free(h_output);
    std::free(h_reference);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
