#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>
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

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_memcpy_waitall() {
    asm volatile("cp.async.wait_all;\n" ::);
}

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

/*
    // OPTIONAL: Uncomment this block to include your kernel implementation
    // from Lab 4 for easy comparison.

    ////////////////////////////////////////////////////////////////////////////////
    // GPU Implementation with Reuse in L1/Shmem and Registers (Baseline from Lab 4)

    #define HAS_LAB_4_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 4 kernel!

    namespace matmul_l1_reg {

    __global__ void matmul_l1_reg(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        // TODO: your GPU code here
    }

    void launch_matmul_l1_reg(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        // TODO: your CPU code here
    }

    } // namespace matmul_l1_reg
*/

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace matmul_improved {

constexpr int32_t UTILE_SZ = 8;
constexpr int32_t NTHREADS_Y = 8;
constexpr int32_t NTHREADS_X = 16;
constexpr int32_t TILE_DIM_M = NTHREADS_Y * UTILE_SZ;
constexpr int32_t TILE_DIM_N = NTHREADS_X * UTILE_SZ;
constexpr int32_t TILE_DIM_K = 16;

typedef struct {
    float a_tile[TILE_DIM_K][TILE_DIM_M];
    float b_tile[TILE_DIM_K][TILE_DIM_N];
} ShMem;

typedef struct {
    float a_tile[TILE_DIM_K * UTILE_SZ][NTHREADS_Y];
    float b_tile[TILE_DIM_K * UTILE_SZ][NTHREADS_X];
} ShMem2;

typedef struct __align__(16) {
    float a_tile[2][TILE_DIM_M][TILE_DIM_K];
    float b_tile[2][TILE_DIM_K][TILE_DIM_N];
} ShMemDB;

/*
__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    int32_t TILE_IDX_I = blockIdx.y;
    int32_t TILE_IDX_J = blockIdx.x;
    int32_t INPUT_TILE_CNT = (size_k + TILE_DIM_K - 1) / TILE_DIM_K;

    //extern __shared__ ShMem shmem[];
    extern __shared__ ShMem2 shmem[];

    float sum[UTILE_SZ][UTILE_SZ] = {0.0};
    float a_regs[UTILE_SZ];
    float b_regs[UTILE_SZ];

    int32_t c_tile_I = TILE_IDX_I * TILE_DIM_M;
    int32_t c_tile_J = TILE_IDX_J * TILE_DIM_N;
    int32_t a_tile_I = c_tile_I;
    int32_t b_tile_J = c_tile_J;

    a += a_tile_I * size_k;
    b += b_tile_J;

    int32_t linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int32_t a_y_idx = linear_idx / TILE_DIM_K;
    int32_t a_x_idx = linear_idx % TILE_DIM_K;
    int32_t a_shmem_j_idx = linear_idx / (UTILE_SZ * TILE_DIM_K);
    int32_t a_shmem_i_idx = UTILE_SZ * (linear_idx % TILE_DIM_K) + ((linear_idx % (UTILE_SZ * TILE_DIM_K)) / TILE_DIM_K);
    int32_t b_y_idx = linear_idx / TILE_DIM_N;
    int32_t b_x_idx = linear_idx % TILE_DIM_N;
    int32_t b_shmem_j_idx = (linear_idx % UTILE_SZ) + UTILE_SZ * (linear_idx / TILE_DIM_N);
    int32_t b_shmem_i_idx = (linear_idx / UTILE_SZ) % NTHREADS_Y;
    //int32_t b_shmem_x_idx = UTILE_SZ * (linear_idx % TILE_DIM_K) + (linear_idx / TILE_DIM_K);
    int32_t a_stride = blockDim.x * blockDim.y / TILE_DIM_K;
    int32_t b_stride = blockDim.x * blockDim.y / TILE_DIM_N;
    int32_t a_shmem_stride = blockDim.x * blockDim.y / (UTILE_SZ * TILE_DIM_K);
    int32_t b_shmem_stride = blockDim.x * blockDim.y / NTHREADS_X;

    for (int32_t tile_idx = 0; tile_idx < INPUT_TILE_CNT; tile_idx++) {
        int32_t a_shmem_offset = 0;
        for (int32_t offset = 0; offset < TILE_DIM_M; offset += a_stride) {
            assert(a_shmem_i_idx < TILE_DIM_K * UTILE_SZ);
            assert(a_shmem_offset + a_shmem_j_idx < NTHREADS_X);
            shmem->a_tile[a_shmem_i_idx][a_shmem_j_idx + a_shmem_offset] = a[(a_y_idx + offset) * size_k + a_x_idx];
            a_shmem_offset += a_shmem_stride;
        }
        assert(a_shmem_offset == NTHREADS_X);

        int32_t b_shmem_offset = 0;
        for (int32_t offset = 0; offset < TILE_DIM_K; offset += b_stride) {
            assert(b_shmem_offset + b_shmem_i_idx < TILE_DIM_K * UTILE_SZ);
            assert(b_shmem_j_idx < NTHREADS_Y);
            shmem->b_tile[b_shmem_offset + b_shmem_i_idx][b_shmem_j_idx] = b[(b_y_idx + offset) * size_j + b_x_idx];
            b_shmem_offset += b_shmem_stride;
        }
        assert(b_shmem_offset == UTILE_SZ * TILE_DIM_K);


        __syncthreads();

        a += TILE_DIM_K;
        b += TILE_DIM_K * size_j;

        // Compute the output tile.
        for (int32_t k = 0; k < TILE_DIM_K; k++) {
            for (int32_t i = 0; i < UTILE_SZ; i++) {
                a_regs[i] = shmem->a_tile[UTILE_SZ * k + i][threadIdx.y];
                b_regs[i] = shmem->b_tile[UTILE_SZ * k + i][threadIdx.x];
            }

            for (int32_t i = 0; i < UTILE_SZ; i++) {
                for (int32_t j = 0; j < UTILE_SZ; j++) {
                    sum[i][j] += a_regs[i] * b_regs[j];
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
*/


// under 7ms
__launch_bounds__(NTHREADS_X * NTHREADS_Y,2)
__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    int32_t TILE_IDX_I = blockIdx.y;
    int32_t TILE_IDX_J = blockIdx.x;
    int32_t INPUT_TILE_CNT = (size_k + TILE_DIM_K - 1) / TILE_DIM_K;

    extern __shared__ ShMem shmem[]; // double buffer

    float sum[UTILE_SZ][UTILE_SZ] = {0.0};
    float a_regs[UTILE_SZ];
    float b_regs[UTILE_SZ];

    int32_t c_tile_I = TILE_IDX_I * TILE_DIM_M;
    int32_t c_tile_J = TILE_IDX_J * TILE_DIM_N;
    int32_t a_tile_I = c_tile_I;
    int32_t b_tile_J = c_tile_J;

    a += a_tile_I * size_k;
    b += b_tile_J;

    int32_t linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int32_t a_y_idx = linear_idx / TILE_DIM_K;
    int32_t a_x_idx = linear_idx % TILE_DIM_K;
    int32_t b_y_idx = linear_idx / TILE_DIM_N;
    int32_t b_x_idx = linear_idx % TILE_DIM_N;
    int32_t a_stride = blockDim.x * blockDim.y / TILE_DIM_K;
    int32_t b_stride = blockDim.x * blockDim.y / TILE_DIM_N;

    for (int32_t tile_idx = 0; tile_idx < INPUT_TILE_CNT; tile_idx++) {

        for (int32_t offset = 0; offset < TILE_DIM_M; offset += a_stride) {
            shmem->a_tile[a_x_idx][a_y_idx + offset] = a[(a_y_idx + offset) * size_k + a_x_idx];
        }
        for (int32_t offset = 0; offset < TILE_DIM_K; offset += b_stride) {
            shmem->b_tile[b_y_idx + offset][b_x_idx] = b[(b_y_idx + offset) * size_j + b_x_idx];
        }


        __syncthreads();

        a += TILE_DIM_K;
        b += TILE_DIM_K * size_j;

        // Compute the output tile.
        for (int32_t k = 0; k < TILE_DIM_K; k++) {
            for (int32_t i = 0; i < UTILE_SZ; i++) {
                a_regs[i] = shmem->a_tile[k][threadIdx.y * UTILE_SZ + i];
                b_regs[i] = shmem->b_tile[k][threadIdx.x * UTILE_SZ + i];
            }

            for (int32_t i = 0; i < UTILE_SZ; i++) {
                for (int32_t j = 0; j < UTILE_SZ; j++) {
                    sum[i][j] += a_regs[i] * b_regs[j];
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


/*
__launch_bounds__(NTHREADS_X * NTHREADS_Y,2)
__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    int32_t TILE_IDX_I = blockIdx.y;
    int32_t TILE_IDX_J = blockIdx.x;
    int32_t INPUT_TILE_CNT = (size_k + TILE_DIM_K - 1) / TILE_DIM_K;

    extern __shared__ ShMemDB shmem[]; // double buffer

    float sum[UTILE_SZ][UTILE_SZ] = {0.0};
    float a_regs[UTILE_SZ];
    float b_regs[UTILE_SZ];

    int32_t c_tile_I = TILE_IDX_I * TILE_DIM_M;
    int32_t c_tile_J = TILE_IDX_J * TILE_DIM_N;
    int32_t a_tile_I = c_tile_I;
    int32_t b_tile_J = c_tile_J;

    a += a_tile_I * size_k;
    b += b_tile_J;
    int32_t cnt = 0;

    int32_t linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int32_t a_y_idx = (linear_idx * 4) / TILE_DIM_K;
    int32_t a_x_idx = (linear_idx * 4) % TILE_DIM_K;
    int32_t b_y_idx = (linear_idx * 4) / TILE_DIM_N;
    int32_t b_x_idx = (linear_idx * 4) % TILE_DIM_N;
    int32_t a_stride = blockDim.x * blockDim.y * 4 / TILE_DIM_K; // each thread loads 4 elements
    int32_t b_stride = blockDim.x * blockDim.y * 4 / TILE_DIM_N; // each thread loads 4 elements
    for (int32_t offset = 0; offset < TILE_DIM_M; offset += a_stride) {
        cp_async4(&(shmem->a_tile[cnt][a_x_idx][a_y_idx + offset]), &a[(a_y_idx + offset) * size_k + a_x_idx]);
    }
    for (int32_t offset = 0; offset < TILE_DIM_K; offset += b_stride) {
        cp_async4(&(shmem->b_tile[cnt][b_y_idx + offset][b_x_idx]), &b[(b_y_idx + offset) * size_j + b_x_idx]);
    }
    a += TILE_DIM_K;
    b += TILE_DIM_K * size_j;
    cnt ^= 1;
    async_memcpy_waitall();
    __syncthreads();

    for (int32_t tile_idx = 0; tile_idx < INPUT_TILE_CNT; tile_idx++) {

        if (tile_idx < INPUT_TILE_CNT - 1) {
            for (int32_t offset = 0; offset < TILE_DIM_M; offset += a_stride) {
                cp_async4(&(shmem->a_tile[cnt][a_x_idx][a_y_idx + offset]), &a[(a_y_idx + offset) * size_k + a_x_idx]);
                //shmem->a_tile[a_x_idx][a_y_idx + offset] = a[(a_y_idx + offset) * size_k + a_x_idx];
            }
            for (int32_t offset = 0; offset < TILE_DIM_K; offset += b_stride) {
                cp_async4(&(shmem->b_tile[cnt][b_y_idx + offset][b_x_idx]), &b[(b_y_idx + offset) * size_j + b_x_idx]);
                //shmem->b_tile[b_y_idx + offset][b_x_idx] = b[(b_y_idx + offset) * size_j + b_x_idx];
            }
        }


        //__syncthreads();

        a += TILE_DIM_K;
        b += TILE_DIM_K * size_j;
        cnt ^= 1;

        // Compute the output tile.
        for (int32_t k = 0; k < TILE_DIM_K; k++) {
            for (int32_t i = 0; i < UTILE_SZ; i++) {
                a_regs[i] = shmem->a_tile[cnt][k][threadIdx.y * UTILE_SZ + i];
                b_regs[i] = shmem->b_tile[cnt][k][threadIdx.x * UTILE_SZ + i];
            }

            for (int32_t i = 0; i < UTILE_SZ; i++) {
                for (int32_t j = 0; j < UTILE_SZ; j++) {
                    sum[i][j] += a_regs[i] * b_regs[j];
                }
            }
        }

        async_memcpy_waitall();
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
*/


void launch_matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {

    int32_t GRID_DIM_X = (size_j + TILE_DIM_N - 1) / TILE_DIM_N;
    int32_t GRID_DIM_Y = (size_i + TILE_DIM_M - 1) / TILE_DIM_M;
    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
    dim3 block_dim(NTHREADS_X, NTHREADS_Y);
    auto SHMEM_SIZE = sizeof(ShMem);
    //auto SHMEM_SIZE = sizeof(ShMemDB);
    matmul_improved<<<grid_dim, block_dim, SHMEM_SIZE>>>(size_i, size_j, size_k, a, b, c);
}

}; // namespace matmul_improved

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k

namespace matmul_improved_reduce {

/* TODO: your GPU kernels here... */

constexpr int32_t UTILE_SZ = 8;
constexpr int32_t NTHREADS_Y = 8;
constexpr int32_t NTHREADS_X = 16;
constexpr int32_t TILE_DIM_M = NTHREADS_Y * UTILE_SZ;
constexpr int32_t TILE_DIM_N = NTHREADS_X * UTILE_SZ;
constexpr int32_t PARTIAL_SUM_K_DIM = 128; // this is the size of the partial summing dimension (i.e., each TB accumulates 128 pixels deep then stores to global memory)
constexpr int32_t TILE_DIM_K = 16; // this is tile size of A and B input tiles brought into shmem

typedef struct {
    float a_tile[TILE_DIM_K][TILE_DIM_M];
    float b_tile[TILE_DIM_K][TILE_DIM_N];
} ShMem;

int32_t ceilDiv(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    /* TODO: your CPU code here */
    return sizeof(float) * size_i * size_j * ceilDiv(size_k, PARTIAL_SUM_K_DIM);
}

//version without partial summing
__launch_bounds__(NTHREADS_X * NTHREADS_Y,2)
__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    int32_t TILE_IDX_I = blockIdx.y;
    int32_t TILE_IDX_J = blockIdx.x;
    int32_t INPUT_TILE_CNT = (size_k + TILE_DIM_K - 1) / TILE_DIM_K; // number of tiles to fetch to shmem from A and B.

    extern __shared__ ShMem shmem[]; // double buffer

    float sum[UTILE_SZ][UTILE_SZ] = {0.0};
    float a_regs[UTILE_SZ];
    float b_regs[UTILE_SZ];

    int32_t c_tile_I = TILE_IDX_I * TILE_DIM_M;
    int32_t c_tile_J = TILE_IDX_J * TILE_DIM_N;
    int32_t a_tile_I = c_tile_I;
    int32_t b_tile_J = c_tile_J;

    a += a_tile_I * size_k;
    b += b_tile_J;

    int32_t linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int32_t a_y_idx = linear_idx / TILE_DIM_K;
    int32_t a_x_idx = linear_idx % TILE_DIM_K;
    int32_t b_y_idx = linear_idx / TILE_DIM_N;
    int32_t b_x_idx = linear_idx % TILE_DIM_N;
    int32_t a_stride = blockDim.x * blockDim.y / TILE_DIM_K;
    int32_t b_stride = blockDim.x * blockDim.y / TILE_DIM_N;

    for (int32_t tile_idx = 0; tile_idx < INPUT_TILE_CNT; tile_idx++) {

        for (int32_t offset = 0; offset < TILE_DIM_M; offset += a_stride) {
            shmem->a_tile[a_x_idx][a_y_idx + offset] = a[(a_y_idx + offset) * size_k + a_x_idx];
        }
        for (int32_t offset = 0; offset < TILE_DIM_K; offset += b_stride) {
            shmem->b_tile[b_y_idx + offset][b_x_idx] = b[(b_y_idx + offset) * size_j + b_x_idx];
        }


        __syncthreads();

        a += TILE_DIM_K;
        b += TILE_DIM_K * size_j;

        // Compute the output tile.
        for (int32_t k = 0; k < TILE_DIM_K; k++) {
            for (int32_t i = 0; i < UTILE_SZ; i++) {
                a_regs[i] = shmem->a_tile[k][threadIdx.y * UTILE_SZ + i];
                b_regs[i] = shmem->b_tile[k][threadIdx.x * UTILE_SZ + i];
            }

            for (int32_t i = 0; i < UTILE_SZ; i++) {
                for (int32_t j = 0; j < UTILE_SZ; j++) {
                    sum[i][j] += a_regs[i] * b_regs[j];
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

            //if (c_utile_I == 0 && c_utile_J == 0) {
            //    printf("sum[0][0] = %f\n", sum[0][0]);
            //}
        }
    }
}

__global__ void matmul_improved_partial_sum(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *workspace_c
) {
    int32_t TILE_IDX_I = blockIdx.y;
    int32_t TILE_IDX_J = blockIdx.x;
    int32_t PARTIAL_SUM_K_IDX = blockIdx.z;
    int32_t INPUT_TILE_CNT = (PARTIAL_SUM_K_DIM + TILE_DIM_K - 1) / TILE_DIM_K;

    extern __shared__ ShMem shmem[];

    float sum[UTILE_SZ][UTILE_SZ] = {0.0};
    float a_regs[UTILE_SZ] = {0.0};
    float b_regs[UTILE_SZ] = {0.0};

    int32_t c_tile_I = TILE_IDX_I * TILE_DIM_M;
    int32_t c_tile_J = TILE_IDX_J * TILE_DIM_N;
    int32_t a_tile_I = c_tile_I;
    int32_t b_tile_J = c_tile_J;

    a += a_tile_I * size_k;
    b += b_tile_J;

    // offset a and b by PARTIAL_SUM_K_IDX
    a += PARTIAL_SUM_K_IDX * TILE_DIM_K * INPUT_TILE_CNT;
    b += PARTIAL_SUM_K_IDX * TILE_DIM_K * size_j * INPUT_TILE_CNT;

    int32_t linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int32_t a_y_idx = linear_idx / TILE_DIM_K;
    int32_t a_x_idx = linear_idx % TILE_DIM_K;
    int32_t b_y_idx = linear_idx / TILE_DIM_N;
    int32_t b_x_idx = linear_idx % TILE_DIM_N;
    int32_t a_stride = blockDim.x * blockDim.y / TILE_DIM_K;
    int32_t b_stride = blockDim.x * blockDim.y / TILE_DIM_N;
    //printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d", blockIdx.x, blockIdx.y, blockIdx.z);

    for (int32_t tile_idx = 0; tile_idx < INPUT_TILE_CNT; tile_idx++) {

        for (int32_t offset = 0; offset < TILE_DIM_M; offset += a_stride) {
            shmem->a_tile[a_x_idx][a_y_idx + offset] = a[(a_y_idx + offset) * size_k + a_x_idx];
        }
        for (int32_t offset = 0; offset < TILE_DIM_K; offset += b_stride) {
            shmem->b_tile[b_y_idx + offset][b_x_idx] = b[(b_y_idx + offset) * size_j + b_x_idx];
        }

        __syncthreads();

        a += TILE_DIM_K;
        b += TILE_DIM_K * size_j;

        // Compute the output tile.
        for (int32_t k = 0; k < TILE_DIM_K; k++) {
            for (int32_t i = 0; i < UTILE_SZ; i++) {
                a_regs[i] = shmem->a_tile[k][threadIdx.y * UTILE_SZ + i];
                b_regs[i] = shmem->b_tile[k][threadIdx.x * UTILE_SZ + i];
            }

            for (int32_t i = 0; i < UTILE_SZ; i++) {
                for (int32_t j = 0; j < UTILE_SZ; j++) {
                    sum[i][j] += a_regs[i] * b_regs[j];
                }
            }
        }

        __syncthreads();
    }

    // when storing, store the partial sum to the workspace buffer.
    int32_t workspace_offset = PARTIAL_SUM_K_IDX * size_i * size_j;
    workspace_c += workspace_offset;
    for (int32_t i = 0; i < UTILE_SZ; i++) {
        for (int32_t j = 0; j < UTILE_SZ; j++) {
            // where does each thread's utile begin?
            int32_t c_utile_I = c_tile_I + UTILE_SZ * threadIdx.y;
            int32_t c_utile_J = c_tile_J + UTILE_SZ * threadIdx.x;
            if (c_utile_I + i < size_i && c_utile_J + j < size_j) {
                workspace_c[(c_utile_I + i) * size_j + (c_utile_J + j)] = sum[i][j];
            }

            //if (c_utile_I == 0 && c_utile_J == 0) {
            //    printf("PARTIAL SUM %u sum[0][0] = %f\n", PARTIAL_SUM_K_IDX, sum[0][0]);
            //}
        }
    }
}

__global__ void reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float* c,
    float* workspace_c
) {
    // reduce the partial sums in the workspace buffer to the final output.
    // works on pixel i,j
    int32_t i = blockIdx.y;
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    int32_t PARTIAL_SUM_K_CNT = (size_k + PARTIAL_SUM_K_DIM - 1) / PARTIAL_SUM_K_DIM;
    float* workspace_cur = workspace_c;
    for (int32_t k = 0; k < PARTIAL_SUM_K_CNT; k++) {
        sum += workspace_cur[i * size_j + j];
        workspace_cur += size_i * size_j;
    }
    //if (i == 0 && j == 0) {
    //    printf("sum[0][0] = %f\n", sum);
    //}
    c[i * size_j + j] = sum;
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    /* TODO: your CPU code here */
    int32_t GRID_DIM_X = ceilDiv(size_j, TILE_DIM_N);
    int32_t GRID_DIM_Y = ceilDiv(size_i, TILE_DIM_M);
    // Do split-K only if we don't have enough thread blocks to keep the SMs busy.
    if (GRID_DIM_X * GRID_DIM_Y >= 48) {
        dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);
        dim3 block_dim(NTHREADS_X, NTHREADS_Y);
        auto SHMEM_SIZE = sizeof(ShMem);
        matmul_improved<<<grid_dim, block_dim, SHMEM_SIZE>>>(size_i, size_j, size_k, a, b, c);
    } else {
        int32_t GRID_DIM_Z = ceilDiv(size_k, PARTIAL_SUM_K_DIM);
        //printf("GRID_DIM_X = %d, GRID_DIM_Y = %d, GRID_DIM_Z = %d\n", GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z);
        dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z);
        dim3 block_dim(NTHREADS_X, NTHREADS_Y);
        auto SHMEM_SIZE = sizeof(ShMem);
        matmul_improved_partial_sum<<<grid_dim, block_dim, SHMEM_SIZE>>>(size_i, size_j, size_k, a, b, (float*)workspace);
        // have minimum # warps per block (so 32 * 4 threads per threadblock),
        // and have these reduce across K dimension.
        int32_t THREADS_PER_BLOCK = 128;
        int32_t REDUCE_GRID_DIM_X = size_j / THREADS_PER_BLOCK;
        int32_t REDUCE_GRID_DIM_Y = size_i;
        dim3 reduce_grid_dim(REDUCE_GRID_DIM_X, REDUCE_GRID_DIM_Y);
        dim3 reduce_block_dim(THREADS_PER_BLOCK);
        reduce<<<reduce_grid_dim, reduce_block_dim, SHMEM_SIZE>>>(size_i, size_j, size_k, c, (float*)workspace);
    }
}

}; // namespace matmul_improved_reduce

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

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
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

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

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

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

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

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-5) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            4,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
            },
            [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
            });

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_4_BASELINE_IMPL

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

#endif

struct MatmulImproved {
    constexpr static char const *name = "matmul_improved";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_4_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulL1Reg>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulImproved>(phase, data, configs));
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";
    if (char *c_str_test_data_dir = std::getenv("MATMUL_TEST_DATA_DIR_2")) {
        test_data_dir = c_str_test_data_dir;
    }

    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
        {1, 3072, 3072},
        {256, 256, 256},
        {256, 256, 1024},
        {256, 256, 8192},
        {128, 128, 32768},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            auto const &first = results.at(i);
            auto const &second = results.at(j);
            printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
            printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
            printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
            for (auto const &config : configs) {
                auto size_i = config.size_i;
                auto size_j = config.size_j;
                auto size_k = config.size_k;
                printf("  %6d  %6d  %6d", size_i, size_j, size_k);
                auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
                auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
                if (it_first != first.elapsed_ms.end() &&
                    it_second != second.elapsed_ms.end()) {
                    printf("  %6.02fx", it_first->second / it_second->second);
                } else {
                    printf("  %7s", "-");
                }
                printf("\n");
            }
        }
    }

    write_json_results("out/results.json", results);

    return 0;
}