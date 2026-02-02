#include "utils.h"

void copy_kernel(double *x, double *y, size_t Np) {
    for (size_t p = 0; p < Np; p++) x[p] = y[p];
}

void scale_kernel(double *x, double *y, double scalar, size_t Np) {
    for (size_t p = 0; p < Np; p++) x[p] = scalar * y[p];
}

void add_kernel(double *x, double *y, double *S, size_t Np) {
    for (size_t p = 0; p < Np; p++) S[p] = x[p] + y[p];
}

void triad_kernel(double *x, double *y, double *v, double *S, size_t Np) {
    for (size_t p = 0; p < Np; p++) {
        // Vector Triad: 3 Reads (x,y,v), 1 Write (S), 2 FLOPs
        S[p] = x[p] + v[p] * y[p];
    }
}

void energy_kernel(double *v, double *S, double mass, size_t Np) {
    double factor = 0.5 * mass;
    for (size_t p = 0; p < Np; p++) {
        S[p] = factor * v[p] * v[p];
    }
}

// TASK 4: Memory Only
void triad_memory_only(double *x, double *y, double *v, double *S, size_t Np) {
    for (size_t p = 0; p < Np; p++) {
        S[p] = x[p] + v[p] + y[p]; 
    }
}

// TASK 4: Compute Only
void triad_compute_only(size_t Np) {
    double a = 1.0001, b = 1.0002, c = 1.0003;
    double result = 0.0;

    for (size_t p = 0; p < Np; p++) {
        result += a + b * c; 
    }
    
    volatile double sink = result; 
}