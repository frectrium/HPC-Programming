#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"

void init_matrices(int Np, double ***m1, double ***m2, double ***result) {
    
    int i, j;

    // Allocate row pointers
    *m1     = (double**)malloc(Np * sizeof(double*));
    *m2     = (double**)malloc(Np * sizeof(double*));
    *result = (double**)malloc(Np * sizeof(double*));

    // Allocate rows
    for (i = 0; i < Np; i++) {
        (*m1)[i]     = (double*)malloc(Np * sizeof(double));
        (*m2)[i]     = (double*)malloc(Np * sizeof(double));
        (*result)[i] = (double*)malloc(Np * sizeof(double));
    }

    // Initialize values
    for (i = 0; i < Np; i++) {
        for (j = 0; j < Np; j++) {
            (*m1)[i][j] = (double)(rand() % 10);
            (*m2)[i][j] = (double)(rand() % 10); 
            (*result)[i][j] = 0.0;
        }
    }
}

void free_matrices(int Np, double** m1, double** m2, double** result) {
    
    for (int i = 0; i < Np; i++) {
        free(m1[i]);
        free(m2[i]);
        free(result[i]);
    }

    free(m1);
    free(m2);
    free(result);
}