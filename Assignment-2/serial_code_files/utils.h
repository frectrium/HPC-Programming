#ifndef UTILS_H
#define UTILS_H
#include <time.h>

// =========================================================================
// PROBLEM A-1: The 6 Loop Permutations
// =========================================================================
void matrix_multiplication_ijk(double** m1, double** m2, double** result, int N);
void matrix_multiplication_ikj(double** m1, double** m2, double** result, int N);
void matrix_multiplication_jik(double** m1, double** m2, double** result, int N);
void matrix_multiplication_jki(double** m1, double** m2, double** result, int N);
void matrix_multiplication_kij(double** m1, double** m2, double** result, int N);
void matrix_multiplication_kji(double** m1, double** m2, double** result, int N);

// Compatibility Wrapper (Defaults to Standard ijk)
void matrix_multiplication(double** m1, double** m2, double** result, int N);

// =========================================================================
// PROBLEM B-1: Transpose Optimization
// =========================================================================
void transpose(double** m, double** mt, int N);
void transposed_matrix_multiplication(double** m1, double** m2, double** result, int N);

// =========================================================================
// PROBLEM C-1: Optimized Blocked Matrix Multiplication
// =========================================================================
void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N);

#endif
