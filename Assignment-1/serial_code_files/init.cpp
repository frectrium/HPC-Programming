#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"

void init_vectors(size_t Np, double **x, double **y, double **v, double **S) {
    // size_t is better for large arrays than int
    *x = (double*) malloc(Np * sizeof(double));
    *y = (double*) malloc(Np * sizeof(double));
    *v = (double*) malloc(Np * sizeof(double));
    *S = (double*) malloc(Np * sizeof(double));

    // Check for allocation failure (Critical for 2^29)
    if (!*x || !*y || !*v || !*S) {
        fprintf(stderr, "Error: Failed to allocate memory for Np = %lu\n", Np);
        exit(1);
    }

    for (size_t i = 0; i < Np; i++) {
        (*x)[i] = (double) rand() / (double) RAND_MAX;
        (*y)[i] = (double) rand() / (double) RAND_MAX;
        (*v)[i] = (double) rand() / (double) RAND_MAX;
        (*S)[i] = 0.0;
    }
}