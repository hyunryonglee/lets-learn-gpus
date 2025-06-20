#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include <cublas_v2.h>

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

namespace transpose_baseline {

constexpr int32_t TILE_DIM_M = 32;
constexpr int32_t TILE_DIM_N = 32;

__global__ void transpose_baseline(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float *c) 
{
    // Calculate output index
    int32_t out_i = blockIdx.y * TILE_DIM_M + threadIdx.y;
    int32_t out_j = blockIdx.x * TILE_DIM_N + threadIdx.x;

    c[out_j * size_j + out_i] = a[out_i * size_j + out_j];

}

void launch_transpose_baseline(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float *c) {
    /* TODO: your CPU code here */
    int32_t GRID_DIM_X = (size_j + TILE_DIM_N - 1) / TILE_DIM_N;
    int32_t GRID_DIM_Y = (size_i + TILE_DIM_M - 1) / TILE_DIM_M;
    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
    dim3 block_dim(TILE_DIM_N, TILE_DIM_M);
    transpose_baseline<<<grid_dim, block_dim>>>(size_i, size_j, a, c);
}

};

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation again for interview prep

namespace transpose_shmem {

constexpr int32_t TILE_DIM_M = 32;
constexpr int32_t TILE_DIM_N = 32;

__global__ void transpose_shmem(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float *c) 
{
    extern __shared__ unsigned char sh[];
    float* shmem = reinterpret_cast<float*>(sh);
    // Calculate output index
    //int32_t out_i = blockIdx.y * TILE_DIM_M + threadIdx.x;
    //int32_t out_j = blockIdx.x * TILE_DIM_N + threadIdx.y;
    //int32_t in_i = blockIdx.x * TILE_DIM_N + threadIdx.x;
    //int32_t in_j = blockIdx.y * TILE_DIM_M + threadIdx.y;

    a = a + TILE_DIM_M * blockIdx.y * size_j + TILE_DIM_N * blockIdx.x;
    c = c + TILE_DIM_N * blockIdx.x * size_i + TILE_DIM_M * blockIdx.y;

    shmem[threadIdx.x * (TILE_DIM_M + 1) + threadIdx.y] = a[threadIdx.y * size_j + threadIdx.x];
    __syncthreads();
    c[threadIdx.y * size_i + threadIdx.x] = shmem[threadIdx.y * (TILE_DIM_M + 1) + threadIdx.x];
    __syncthreads();

}

void launch_transpose_shmem(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float *c) {
    /* TODO: your CPU code here */
    int32_t GRID_DIM_X = (size_j + TILE_DIM_N - 1) / TILE_DIM_N;
    int32_t GRID_DIM_Y = (size_i + TILE_DIM_M - 1) / TILE_DIM_M;
    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
    dim3 block_dim(TILE_DIM_N, TILE_DIM_M);
    int32_t shmemSize = sizeof(float) * TILE_DIM_M * (TILE_DIM_N + 1);
    transpose_shmem<<<grid_dim, block_dim, shmemSize>>>(size_i, size_j, a, c);
}

};

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation again for interview prep

namespace transpose_shmem_coalescing {

constexpr int32_t TILE_DIM_M = 64;
constexpr int32_t TILE_DIM_N = 64;
constexpr int32_t U = 4; // utile size, where utile is 1xU.

__global__ void transpose_shmem_coalescing(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float *c) 
{
    extern __shared__ unsigned char sh[];
    float* shmem = reinterpret_cast<float*>(sh);
    a = a + TILE_DIM_N * blockIdx.x + TILE_DIM_M * size_j * blockIdx.y;
    c = c + TILE_DIM_N * blockIdx.x * size_i + TILE_DIM_M * blockIdx.y;

    for (int32_t i = 0; i < TILE_DIM_N; i += TILE_DIM_N / U) {
        shmem[(threadIdx.x + i) * (TILE_DIM_M + 1) + threadIdx.y] = a[threadIdx.y * size_j + threadIdx.x + i];
    }

    __syncthreads();

    for (int32_t u = 0; u < TILE_DIM_N; u += TILE_DIM_N / U) {
        c[threadIdx.y * size_i + threadIdx.x + u] =  shmem[threadIdx.y * (TILE_DIM_M + 1) + threadIdx.x + u];
    }
}

void launch_transpose_shmem_coalescing(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float *c) {
    /* TODO: your CPU code here */
    int32_t GRID_DIM_X = (size_j + TILE_DIM_N - 1) / TILE_DIM_N;
    int32_t GRID_DIM_Y = (size_i + TILE_DIM_M - 1) / TILE_DIM_M;
    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
    dim3 block_dim(TILE_DIM_N/U, TILE_DIM_M);
    int32_t shmemSize = sizeof(float) * TILE_DIM_M * (TILE_DIM_N + 1);
    transpose_shmem_coalescing<<<grid_dim, block_dim, shmemSize>>>(size_i, size_j, a, c);
}

};


namespace transpose_cublas {

void launch_transpose_cublas(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float *c) {

    float const alpha(1.0);
    float const beta(0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_T, size_j, size_i, &alpha, a, size_j, &beta, a, size_j, c, size_j );
    cublasDestroy(handle);
}

};


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

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, a_gpu, c_gpu);

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

        printf("  size %4d * %4d:\n", size_i, size_j);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, a_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

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
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, true}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{16384, 16384, true}});
}

struct TransposeBaseline {
    constexpr static char const *name = "transpose_baseline";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float *c) {
        transpose_baseline::launch_transpose_baseline(size_i, size_j, a, c);
    }
};

struct TransposeShmem {
    constexpr static char const *name = "transpose_shmem";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float *c) {
        transpose_shmem::launch_transpose_shmem(size_i, size_j, a, c);
    }
};


struct TransposeShmemCoalescing {
    constexpr static char const *name = "transpose_shmem_coalescing";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float *c) {
        transpose_shmem_coalescing::launch_transpose_shmem_coalescing(size_i, size_j, a, c);
    }
};

struct TransposeShmemCublas {
    constexpr static char const *name = "transpose_cublas";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float *c) {
        transpose_cublas::launch_transpose_cublas(size_i, size_j, a, c);
    }
};


int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("MATMUL_TEST_DATA_DIR")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<TransposeShmemCublas>(test_data_dir, saved_results);
    run_all_tests<TransposeBaseline>(test_data_dir, saved_results);
    run_all_tests<TransposeShmem>(test_data_dir, saved_results);
    run_all_tests<TransposeShmemCoalescing>(test_data_dir, saved_results);

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
