#ifndef UTILS_H
#define UTILS_H

#include <stddef.h> // For size_t

// Standard Kernels
void copy_kernel(double *x, double *y, size_t Np);
void scale_kernel(double *x, double *y, double scalar, size_t Np);
void add_kernel(double *x, double *y, double *S, size_t Np);

// Your Custom "Vector Quad" Triad: S = X + V*Y
void triad_kernel(double *x, double *y, double *v, double *S, size_t Np);

void energy_kernel(double *v, double *S, double mass, size_t Np);

// Analysis Kernels
void triad_memory_only(double *x, double *y, double *v, double *S, size_t Np);
void triad_compute_only(size_t Np);

#endif