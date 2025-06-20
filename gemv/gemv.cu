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
#include <cassert>

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

// One thread per output
namespace gemv_baseline {

constexpr int32_t BLOCK_SZ = 128;

__global__ void gemv_baseline(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float const *x,
    float *c) {

    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int j = 0; j < size_j; j++) {
        sum += a[idx * size_j + j] * x[j];
    }

    c[idx] = sum;
}

void launch_gemv_baseline(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float const *x,
    float *c) {

    int32_t NUM_BLOCKS = size_i / BLOCK_SZ;
    gemv_baseline<<<NUM_BLOCKS,BLOCK_SZ>>>(size_i, size_j, a, x, c);
}

};


// One warp per output
// Note 32 threads per warp
namespace gemv_shmem {

constexpr int32_t BLOCK_SZ = 128;
constexpr int32_t WARP_SZ = 32;

__global__ void gemv_shmem(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float const *x,
    float *c) {

    int32_t warp_idx = threadIdx.y;
    int32_t warp_thread_idx = threadIdx.x;
    int32_t idx = blockIdx.x * blockDim.y + warp_idx;

    extern __shared__ unsigned char shmem[];
    float* sh = reinterpret_cast<float*>(shmem + WARP_SZ * sizeof(float) * warp_idx * 2);
    float* shBuf = reinterpret_cast<float*>(shmem + WARP_SZ * sizeof(float) * (warp_idx * 2 + 1));

    float sum = 0;
    for (int j = warp_thread_idx; j < size_j; j += WARP_SZ) {
        assert(idx * size_j + j < size_i * size_j);
        sum += a[idx * size_j + j] * x[j];
    }

    // Reduction across threads in a warp
    // Example with shmem
    sh[warp_thread_idx] = sum;
    __syncthreads();
    float* src = sh;
    float* dst = shBuf;
    for (int j = WARP_SZ / 2; j > 0; j >>= 1) {
        if (warp_thread_idx < j) {
            dst[warp_thread_idx] = src[warp_thread_idx] + src[warp_thread_idx + j];
        }
        float* tmp = src;
        src = dst;
        dst = tmp;
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x == 0)
        c[idx] = src[0];
}

void launch_gemv_shmem(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float const *x,
    float *c) {
    int32_t BLOCK_DIM_X = WARP_SZ;
    int32_t BLOCK_DIM_Y = BLOCK_SZ / BLOCK_DIM_X;
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    int32_t NUM_BLOCKS = size_i / BLOCK_DIM_Y;
    int32_t shmemSz = sizeof(float) * BLOCK_SZ * 2;
    gemv_shmem<<<NUM_BLOCKS,block_dim,shmemSz>>>(size_i, size_j, a, x, c);
}

};

// One warp per output
// Note 32 threads per warp
namespace gemv_warp {

constexpr int32_t BLOCK_SZ = 128;
constexpr int32_t WARP_SZ = 32;

__global__ void gemv_warp(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float const *x,
    float *c) {

    int32_t warp_idx = threadIdx.y;
    int32_t warp_thread_idx = threadIdx.x;
    int32_t idx = blockIdx.x * blockDim.y + warp_idx;

    float sum = 0;
    for (int j = warp_thread_idx; j < size_j; j += WARP_SZ) {
        assert(idx * size_j + j < size_i * size_j);
        sum += a[idx * size_j + j] * x[j];
    }

    // Reduction across threads in a warp
    // Example with direct communication between registers of threads in a warp.
    for (int offset = 16; offset > 0; offset /= 2) {
        sum +=  __shfl_down_sync(0xffffffff, sum, offset); 
    }

    if (threadIdx.x == 0)
        c[idx] = sum;
}

void launch_gemv_warp(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float const *x,
    float *c) {
    int32_t BLOCK_DIM_X = WARP_SZ;
    int32_t BLOCK_DIM_Y = BLOCK_SZ / BLOCK_DIM_X;
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    int32_t NUM_BLOCKS = size_i / BLOCK_DIM_Y;
    gemv_warp<<<NUM_BLOCKS,block_dim>>>(size_i, size_j, a, x, c);
}

};

namespace gemv_cublas {

#define CHECK_CUBLAS_ERROR(call)                                        \
{                                                                       \
    cublasStatus_t status = call;                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                             \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                 \
                __FILE__, __LINE__, status);                            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

void launch_gemv_cublas(
    int32_t size_i,
    int32_t size_j,
    float const *a,
    float const *x,
    float *c) {

    float const alpha(1.0);
    float const beta(0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    CHECK_CUBLAS_ERROR(
    cublasSgemv( handle, 
        CUBLAS_OP_T, 
        size_j, 
        size_i, 
        &alpha, 
        a, 
        size_j, 
        x,
        1,
        &beta,
        c,
        1
    ));
    cublasDestroy(handle);
}

};



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
        auto b = read_data(path_prefix + "_b.bin", size_j);
        auto c = read_data(path_prefix + "_c.bin", size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_j * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[j] - c[j];
            mse += diff * diff;
            ref_mean_square += c[j] * c[j];
        }
        mse /= size_j;
        ref_mean_square /= size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d:\n", size_i, size_j);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, a_gpu, b_gpu, c_gpu);
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

struct GemvCublas {
    constexpr static char const *name = "gemv_cublas";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float const *x,
        float *c) {
        gemv_cublas::launch_gemv_cublas(size_i, size_j, a, x, c);
    }
};

struct GemvBaseline {
    constexpr static char const *name = "gemv_baseline";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float const *x,
        float *c) {
        gemv_baseline::launch_gemv_baseline(size_i, size_j, a, x, c);
    }
};

struct GemvShmem {
    constexpr static char const *name = "gemv_shmem";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float const *x,
        float *c) {
        gemv_shmem::launch_gemv_shmem(size_i, size_j, a, x, c);
    }
};

struct GemvWarp {
    constexpr static char const *name = "gemv_warp";
    static void
    run(int32_t size_i,
        int32_t size_j,
        float const *a,
        float const *x,
        float *c) {
        gemv_warp::launch_gemv_warp(size_i, size_j, a, x, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("MATMUL_TEST_DATA_DIR")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<GemvWarp>(test_data_dir, saved_results);
    run_all_tests<GemvShmem>(test_data_dir, saved_results);
    run_all_tests<GemvBaseline>(test_data_dir, saved_results);
    run_all_tests<GemvCublas>(test_data_dir, saved_results);

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
