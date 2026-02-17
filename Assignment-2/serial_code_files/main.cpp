#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "init.h"
#include "utils.h"

#define CLK CLOCK_MONOTONIC

int main() {

    // structs to measure timings
    struct timespec start_e2e, end_e2e, start_alg, end_alg;

    int minProbSize = 1 << 1;  // pow(2, 1)
    int maxProbSize = 1 << 12; // pow(2, 12)

    double** m1;
    double** m2;
    double** mt;
    double** result;

    // print the output order
    printf("ProblemSize, E2ETime, AlgoTime\n");

    for (int Np = minProbSize; Np <= maxProbSize; Np *= 2) {

        clock_gettime(CLK, &start_e2e);

        // Initialize Matrices
        init_matrices(Np, &m1, &m2, &result);

        // Block Size   
        // int B = (Np >= 32? 32 : Np);

        clock_gettime(CLK, &start_alg);
        matrix_multiplication(m1, m2, result, Np);
        clock_gettime(CLK, &end_alg);

        clock_gettime(CLK, &end_e2e);


        double e2e_time = (end_e2e.tv_sec - start_e2e.tv_sec) + (end_e2e.tv_nsec - start_e2e.tv_nsec) * 1e-9;
        double alg_time = (end_alg.tv_sec - start_alg.tv_sec) + (end_alg.tv_nsec - start_alg.tv_nsec) * 1e-9;

        printf("%d, %.9lf, %.9lf\n", Np, e2e_time, alg_time);
        fflush(stdout); 

        free_matrices(Np, m1, m2, result);
    }

    return 0;
}
