#include "utils.h"
#include <stdlib.h>
#include <immintrin.h> // AVX2 + FMA Intrinsics
#include <algorithm>   // std::min
#include <string.h>    // memset

#define MAX_BLOCK_SIZE 32

// =========================================================================
// PROBLEM A-1: 6 Loop Permutations (Baseline)
// =========================================================================

void matrix_multiplication(double** m1, double** m2, double** result, int N) {
    // Change this call to test different problems:
    //   Problem A: matrix_multiplication_ijk (or any permutation)
    //   Problem B: transposed_matrix_multiplication(m1, m2, result, N);
    //   Problem C: block_matrix_multiplication(m1, m2, result, N, B);
    int B = (N >= MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : N);
    block_matrix_multiplication(m1, m2, result, B, N);
}

void matrix_multiplication_ijk(double** m1, double** m2, double** result, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] += sum;
        }
    }
}

void matrix_multiplication_ikj(double** m1, double** m2, double** result, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double r = m1[i][k];
            for (int j = 0; j < N; j++) {
                result[i][j] += r * m2[k][j];
            }
        }
    }
}

void matrix_multiplication_jik(double** m1, double** m2, double** result, int N) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] += sum;
        }
    }
}

void matrix_multiplication_jki(double** m1, double** m2, double** result, int N) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            double r = m2[k][j];
            for (int i = 0; i < N; i++) {
                result[i][j] += m1[i][k] * r;
            }
        }
    }
}

void matrix_multiplication_kij(double** m1, double** m2, double** result, int N) {
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            double r = m1[i][k];
            for (int j = 0; j < N; j++) {
                result[i][j] += r * m2[k][j];
            }
        }
    }
}

void matrix_multiplication_kji(double** m1, double** m2, double** result, int N) {
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < N; j++) {
            double r = m2[k][j];
            for (int i = 0; i < N; i++) {
                result[i][j] += m1[i][k] * r;
            }
        }
    }
}

// =========================================================================
// PROBLEM B-1: Transpose Optimization (Blocked transpose + AVX2 FMA)
// =========================================================================

void transpose(double** m, double** mt, int N) {
    int b = (N >= MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : N);
    for (int i = 0; i < N; i += b) {
        int i_end = std::min(i + b, N);
        for (int j = 0; j < N; j += b) {
            int j_end = std::min(j + b, N);
            for (int ii = i; ii < i_end; ii++) {
                for (int jj = j; jj < j_end; jj++) {
                    mt[jj][ii] = m[ii][jj];
                }
            }
        }
    }
}

void transposed_matrix_multiplication(double** m1, double** m2, double** result, int N) {
    // Allocate transpose with 32-byte alignment for AVX loads
    double** mt = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        mt[i] = (double*)_mm_malloc(N * sizeof(double), 32);
    }

    transpose(m2, mt, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256d sum_vec = _mm256_setzero_pd();
            int k = 0;

            // Main AVX2 + FMA vectorized loop (4 doubles at a time)
            for (; k <= N - 4; k += 4) {
                __m256d a_vec = _mm256_loadu_pd(&m1[i][k]);
                __m256d b_vec = _mm256_load_pd(&mt[j][k]);   // aligned
                sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
            }

            double temp[4];
            _mm256_storeu_pd(temp, sum_vec);
            double total = temp[0] + temp[1] + temp[2] + temp[3];

            // Scalar cleanup for remainder
            for (; k < N; k++) {
                total += m1[i][k] * mt[j][k];
            }

            result[i][j] += total;
        }
    }

    for (int i = 0; i < N; i++) _mm_free(mt[i]);
    free(mt);
}

// =========================================================================
// PROBLEM C-1: Blocked + Packed B + AVX2 FMA + 4x4 Micro-kernel
// =========================================================================

// 4x4 micro-kernel: computes a 4-row x 4-col tile of C using packed B
__attribute__((always_inline))
inline void kernel_4x4(int K_len,
                       double* A_row0, double* A_row1,
                       double* A_row2, double* A_row3,
                       double* B_ptr,
                       double* C_row0, double* C_row1,
                       double* C_row2, double* C_row3) {

    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    int k = 0;

    // Unrolled by 4 for deeper instruction pipeline
    for (; k <= K_len - 4; k += 4) {
        __m256d b0 = _mm256_load_pd(B_ptr + (k + 0) * 4);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row0 + k + 0), b0, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row1 + k + 0), b0, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row2 + k + 0), b0, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row3 + k + 0), b0, c3);

        __m256d b1 = _mm256_load_pd(B_ptr + (k + 1) * 4);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row0 + k + 1), b1, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row1 + k + 1), b1, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row2 + k + 1), b1, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row3 + k + 1), b1, c3);

        __m256d b2 = _mm256_load_pd(B_ptr + (k + 2) * 4);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row0 + k + 2), b2, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row1 + k + 2), b2, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row2 + k + 2), b2, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row3 + k + 2), b2, c3);

        __m256d b3 = _mm256_load_pd(B_ptr + (k + 3) * 4);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row0 + k + 3), b3, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row1 + k + 3), b3, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row2 + k + 3), b3, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row3 + k + 3), b3, c3);
    }

    // Cleanup for remaining k
    for (; k < K_len; k++) {
        __m256d bv = _mm256_load_pd(B_ptr + k * 4);
        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row0 + k), bv, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row1 + k), bv, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row2 + k), bv, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(A_row3 + k), bv, c3);
    }

    // Accumulate into C (unaligned since C comes from init.cpp's malloc)
    _mm256_storeu_pd(C_row0, _mm256_add_pd(_mm256_loadu_pd(C_row0), c0));
    _mm256_storeu_pd(C_row1, _mm256_add_pd(_mm256_loadu_pd(C_row1), c1));
    _mm256_storeu_pd(C_row2, _mm256_add_pd(_mm256_loadu_pd(C_row2), c2));
    _mm256_storeu_pd(C_row3, _mm256_add_pd(_mm256_loadu_pd(C_row3), c3));
}

// Scalar fallback for edge tiles that don't fit a full 4x4 kernel
static inline void scalar_block_multiply(double** m1, double** m2, double** result,
                                         int i_start, int i_end,
                                         int j_start, int j_end,
                                         int k_start, int k_end) {
    for (int i = i_start; i < i_end; i++) {
        for (int j = j_start; j < j_end; j++) {
            double sum = 0.0;
            for (int k = k_start; k < k_end; k++) {
                sum += m1[i][k] * m2[k][j];
            }
            result[i][j] += sum;
        }
    }
}

void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N) {
    // For very small N, just do scalar ikj (cache-friendly, no AVX overhead)
    if (N < 4) {
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < N; k++) {
                double r = m1[i][k];
                for (int j = 0; j < N; j++) {
                    result[i][j] += r * m2[k][j];
                }
            }
        }
        return;
    }

    // Clamp block size
    if (B > N) B = N;
    if (B > MAX_BLOCK_SIZE) B = MAX_BLOCK_SIZE;

    // --- Contiguous + padded layout to fix two critical performance issues: ---
    // 1) Pointer-of-pointers: each row from init.cpp is separately malloc'd,
    //    causing TLB misses and defeating hardware prefetching.
    // 2) Cache set conflicts: for power-of-2 N (e.g. 4096), row stride = N*8
    //    is a multiple of the L1 cache size, so ALL rows map to the same cache
    //    set, causing pathological conflict misses. Padding by 8 doubles (1
    //    cache line) makes consecutive rows land in different sets.
    int ld = N + 8;  // leading dimension with padding
    double* A = (double*)_mm_malloc((size_t)N * ld * sizeof(double), 32);
    double* C = (double*)_mm_malloc((size_t)N * ld * sizeof(double), 32);
    double* packedB = (double*)_mm_malloc(B * B * sizeof(double), 32);

    // Copy m1 into contiguous A, zero out C
    for (int i = 0; i < N; i++) {
        memcpy(A + (size_t)i * ld, m1[i], N * sizeof(double));
        memset(C + (size_t)i * ld, 0, ld * sizeof(double));
    }

    // j-k-i block loop order: C tiles stay in L3 between k-iterations,
    // packed B is reused across all i-blocks.
    for (int jj = 0; jj < N; jj += B) {
        int j_end = std::min(jj + B, N);
        int J_len = j_end - jj;

        for (int kk = 0; kk < N; kk += B) {
            int k_end = std::min(kk + B, N);
            int K_len = k_end - kk;

            // Pack B[kk:k_end, jj:j_end] from m2 into column-panel layout
            int idx = 0;
            for (int j = jj; j < j_end; j += 4) {
                for (int k = kk; k < k_end; k++) {
                    for (int x = 0; x < 4; x++) {
                        packedB[idx++] = (j + x < j_end) ? m2[k][j + x] : 0.0;
                    }
                }
            }

            // Compute using contiguous A and C (no pointer chasing)
            for (int ii = 0; ii < N; ii += B) {
                int i_end = std::min(ii + B, N);
                int I_len = i_end - ii;
                int i_main = ii + (I_len / 4) * 4;
                int j_main = jj + (J_len / 4) * 4;

                for (int i = ii; i < i_main; i += 4) {
                    for (int j = jj; j < j_main; j += 4) {
                        int strip_idx = (j - jj) / 4;
                        double* B_ptr = packedB + strip_idx * K_len * 4;

                        kernel_4x4(K_len,
                                   A + (size_t)i * ld + kk,
                                   A + (size_t)(i+1) * ld + kk,
                                   A + (size_t)(i+2) * ld + kk,
                                   A + (size_t)(i+3) * ld + kk,
                                   B_ptr,
                                   C + (size_t)i * ld + j,
                                   C + (size_t)(i+1) * ld + j,
                                   C + (size_t)(i+2) * ld + j,
                                   C + (size_t)(i+3) * ld + j);
                    }

                    // Right edge columns: scalar fallback
                    if (j_main < j_end) {
                        for (int i2 = i; i2 < i + 4 && i2 < i_end; i2++) {
                            for (int j2 = j_main; j2 < j_end; j2++) {
                                double sum = 0.0;
                                for (int k = kk; k < k_end; k++) {
                                    sum += A[(size_t)i2 * ld + k] * m2[k][j2];
                                }
                                C[(size_t)i2 * ld + j2] += sum;
                            }
                        }
                    }
                }

                // Bottom edge rows: scalar fallback
                if (i_main < i_end) {
                    for (int i2 = i_main; i2 < i_end; i2++) {
                        for (int j2 = jj; j2 < j_end; j2++) {
                            double sum = 0.0;
                            for (int k = kk; k < k_end; k++) {
                                sum += A[(size_t)i2 * ld + k] * m2[k][j2];
                            }
                            C[(size_t)i2 * ld + j2] += sum;
                        }
                    }
                }
            }
        }
    }

    // Copy C back into result
    for (int i = 0; i < N; i++) {
        double* C_row = C + (size_t)i * ld;
        double* R_row = result[i];
        for (int j = 0; j < N; j++) {
            R_row[j] += C_row[j];
        }
    }

    _mm_free(A);
    _mm_free(C);
    _mm_free(packedB);
}
