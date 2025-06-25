#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation again for interview prep
// Naive version. Each thread works on an output index of C.
// This requires size_k multiplications of A and B values.
// This version simply directly loads them from global memory.

namespace matmul_baseline {

constexpr int32_t TILE_DIM_M = 32;
constexpr int32_t TILE_DIM_N = 32;

// Naive version of matmul
__global__ void matmul_baseline(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) 
{
    // Calculate output index
    int32_t out_i = blockIdx.y * TILE_DIM_M + threadIdx.y;
    int32_t out_j = blockIdx.x * TILE_DIM_N + threadIdx.x;

    float sum = 0.0;
    for (int k = 0; k < size_k; k++) {
        sum += a[out_i * size_k + k] * b[k * size_k + out_j];
    }
    c[out_i * size_k + out_j] = sum;

}

void launch_matmul_baseline(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */
    int32_t GRID_DIM_X = (size_j + TILE_DIM_N - 1) / TILE_DIM_N;
    int32_t GRID_DIM_Y = (size_i + TILE_DIM_M - 1) / TILE_DIM_M;
    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
    dim3 block_dim(TILE_DIM_N, TILE_DIM_M);
    matmul_baseline<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
}

};

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation again for interview prep... use shared memory
// Bring in TILE_DIM_M x TILE_DIM_K of A, and TILE_DIM_K x TILE_DIM_N of B
// to shared memory. Perform matmul up to TILE_DIM_K.
// Then, bring in the next TILE_DIM_K amount of A and B.
// Increases arithmetic intensity by ~TILE_DIM_K.

namespace matmul_shmem {

constexpr int32_t TILE_DIM_M = 32;
constexpr int32_t TILE_DIM_K = 32;
constexpr int32_t TILE_DIM_N = 32;


__global__ void matmul_shmem(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) 
{
    // Get pointer to shared memory
    extern __shared__ unsigned char shmem[];

    float* shmem_a = reinterpret_cast<float*>(shmem);
    float* shmem_b = reinterpret_cast<float*>(shmem + sizeof(float) * TILE_DIM_M * TILE_DIM_K);
    float sum = 0;

    a = a + (blockIdx.y * TILE_DIM_M) * size_k;
    b = b + blockIdx.x * TILE_DIM_N;

    // Load to shmem
    // Since M,N,K are conveniently the same size, we can simply load one of A and one of B per thread.
    for (int tile_idx = 0; tile_idx < size_k / TILE_DIM_K; tile_idx++) {
        shmem_a[threadIdx.y * TILE_DIM_K + threadIdx.x] = a[threadIdx.y * size_k + threadIdx.x];
        shmem_b[threadIdx.y * TILE_DIM_N + threadIdx.x] = b[threadIdx.y * size_j + threadIdx.x];

        __syncthreads();

        for (int k = 0; k < TILE_DIM_K; k++) {
            sum += shmem_a[threadIdx.y * TILE_DIM_K + k] * shmem_b[k * TILE_DIM_N + threadIdx.x];
        }

        __syncthreads();

        // shift a and b pointers
        a = a + TILE_DIM_K;
        b = b + TILE_DIM_K * size_j;
    }

    int32_t out_offset_x = TILE_DIM_N * blockIdx.x;
    int32_t out_offset_y = TILE_DIM_M * blockIdx.y;
    c = c + out_offset_y * size_j + out_offset_x;
    c[threadIdx.y * size_j + threadIdx.x] = sum;
}

void launch_matmul_shmem(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */
    int32_t GRID_DIM_X = (size_j + TILE_DIM_N - 1) / TILE_DIM_N;
    int32_t GRID_DIM_Y = (size_i + TILE_DIM_M - 1) / TILE_DIM_M;
    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
    dim3 block_dim(TILE_DIM_N, TILE_DIM_M);
    // Allocate shmem
    int32_t shmemSize = (TILE_DIM_M * TILE_DIM_K + TILE_DIM_K * TILE_DIM_N) * sizeof(float);
    matmul_shmem<<<grid_dim, block_dim, shmemSize>>>(size_i, size_j, size_k, a, b, c);
}

};

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

namespace matmul_l1_reg {

constexpr int32_t UTILE_SZ = 8;
constexpr int32_t NTHREADS_Y = 16;
constexpr int32_t NTHREADS_X = 16;
constexpr int32_t TILE_DIM_M = NTHREADS_Y * UTILE_SZ;
constexpr int32_t TILE_DIM_N = NTHREADS_X * UTILE_SZ;
constexpr int32_t TILE_DIM_K = 16;

typedef struct {
    float a_tile[TILE_DIM_K][TILE_DIM_M];
    float b_tile[TILE_DIM_K][TILE_DIM_N];
} ShMem;


__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your GPU code here */
    int32_t TILE_IDX_I = blockIdx.y;
    int32_t TILE_IDX_J = blockIdx.x;
    int32_t INPUT_TILE_CNT = (size_k + TILE_DIM_K - 1) / TILE_DIM_K;

    extern __shared__ ShMem shmem[];

    float sum[UTILE_SZ][UTILE_SZ] = {0.0};
    //float a_regs[UTILE_SZ];
    float a_regs[UTILE_SZ];
    float b_regs[UTILE_SZ];

    int32_t c_tile_I = TILE_IDX_I * TILE_DIM_M;
    int32_t c_tile_J = TILE_IDX_J * TILE_DIM_N;
    int32_t a_tile_I = c_tile_I;
    int32_t b_tile_J = c_tile_J;

    for (int32_t tile_idx = 0; tile_idx < INPUT_TILE_CNT; tile_idx++) {
        // Load the input tiles into shared memory.
        int32_t a_tile_k = tile_idx * TILE_DIM_K;
        int32_t b_tile_k = tile_idx * TILE_DIM_K;
        for (int32_t a_shmem_i = threadIdx.y; a_shmem_i < TILE_DIM_M; a_shmem_i += NTHREADS_Y) {
            for (int32_t a_shmem_k = threadIdx.x; a_shmem_k < TILE_DIM_K; a_shmem_k += NTHREADS_X) {
                shmem->a_tile[a_shmem_k][a_shmem_i] = a[(a_tile_I + a_shmem_i) * size_k + (a_tile_k + a_shmem_k)];
            }
        }

        for (int32_t b_shmem_k = threadIdx.y; b_shmem_k < TILE_DIM_K; b_shmem_k += NTHREADS_Y) {
            for (int32_t b_shmem_j = threadIdx.x; b_shmem_j < TILE_DIM_N; b_shmem_j += NTHREADS_X) {
                shmem->b_tile[b_shmem_k][b_shmem_j] = b[(b_tile_k + b_shmem_k) * size_j + (b_tile_J + b_shmem_j)];
            }
        }

        __syncthreads();

        // Compute the output tile.
        for (int32_t k = 0; k < TILE_DIM_K; k++) {
            for (int32_t i = 0; i < UTILE_SZ; i++) {
                a_regs[i] = shmem->a_tile[k][threadIdx.y * UTILE_SZ + i];
            }

            for (int32_t j = 0; j < UTILE_SZ; j++) {
                b_regs[j] = shmem->b_tile[k][threadIdx.x * UTILE_SZ + j];
            }

            for (int32_t i = 0; i < UTILE_SZ; i++) {
                //a_regs[i] = shmem->a_tile[threadIdx.y * UTILE_SZ + i][k];
                for (int32_t j = 0; j < UTILE_SZ; j++) {
                    //sum[i][j] += a_regs[i] * b_regs[j]; 
                    sum[i][j] += a_regs[i] * b_regs[j];    
                    //sum[i][j] += shmem->a_tile[threadIdx.y * UTILE_SZ + i][k] * shmem->b_tile[k][threadIdx.x * UTILE_SZ + j];
                }
            }
        }

        __syncthreads();
    }

    for (int32_t i = 0; i < UTILE_SZ; i++) {
        for (int32_t j = 0; j < UTILE_SZ; j++) {
            // where does each thread's utile begin?
            int32_t c_utile_I = c_tile_I + UTILE_SZ * threadIdx.y;
            int32_t c_utile_J = c_tile_J + UTILE_SZ * threadIdx.x;
            if (c_utile_I + i < size_i && c_utile_J + j < size_j) {
                c[(c_utile_I + i) * size_j + (c_utile_J + j)] = sum[i][j];
            }
        }
    }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */
    int32_t GRID_DIM_X = (size_j + TILE_DIM_N - 1) / TILE_DIM_N;
    int32_t GRID_DIM_Y = (size_i + TILE_DIM_M - 1) / TILE_DIM_M;
    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
    dim3 block_dim(NTHREADS_X, NTHREADS_Y);
    auto SHMEM_SIZE = sizeof(ShMem);
    matmul_l1_reg<<<grid_dim, block_dim, SHMEM_SIZE>>>(size_i, size_j, size_k, a, b, c);
    
}

}; // namespace matmul_l1_reg

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_k * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_k * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_i * size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_i; ++i) {
            for (int32_t j = 0; j < size_j; ++j) {
                float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
                mse += diff * diff;
                ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
            }
        }
        mse /= size_i * size_j;
        ref_mean_square /= size_i * size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulBaseline {
    constexpr static char const *name = "matmul_baseline";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_baseline::launch_matmul_baseline(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulShmem {
    constexpr static char const *name = "matmul_shmem";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_shmem::launch_matmul_shmem(size_i, size_j, size_k, a, b, c);
    }
};


struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("MATMUL_TEST_DATA_DIR")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulBaseline>(test_data_dir, saved_results);
    run_all_tests<MatmulShmem>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
