/*
 * main_exp02.cpp — Experiment 02: Consistency Across Configurations
 *
 * Compile: gcc -O3 -march=native -fopenmp -o exp02 main_exp02.cpp utils.cpp init.cpp -lm
 * Run:     ./exp02
 * Output:  consistency_results.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(void) {
    omp_set_num_threads(4);

    int grid_nx[3] = {250, 500, 1000};
    int grid_ny[3] = {100, 200,  400};
    long fixed_count = 100000000L;
    Maxiter = 10;

    FILE *fp = fopen("consistency_results.csv", "w");
    if (!fp) { printf("Cannot open consistency_results.csv\n"); return 1; }
    fprintf(fp, "Grid_NX,Grid_NY,Iter,Interp_Time_s\n");

    printf("=== EXPERIMENT 2: CONSISTENCY (N=10^8) ===\n");
    printf("%-12s %-6s %s\n", "Grid", "Iter", "Interp_Time(s)");

    for (int g = 0; g < 3; g++) {
        NX = grid_nx[g]; NY = grid_ny[g];
        GRID_X = NX + 1; GRID_Y = NY + 1;
        dx = 1.0 / NX; dy = 1.0 / NY;
        NUM_Points = (int)fixed_count;

        double *mesh_value = (double *)calloc((size_t)GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points *)calloc(NUM_Points, sizeof(Points));

        if (!mesh_value || !points) {
            printf("Alloc failed for grid %dx%d\n", NX, NY);
            free(mesh_value); free(points); continue;
        }

        initializepoints(points);

        for (int iter = 0; iter < Maxiter; iter++) {
            clock_t t0 = clock();
            interpolation(mesh_value, points);
            clock_t t1 = clock();
            double t = (double)(t1 - t0) / CLOCKS_PER_SEC;

            const char *gname = (NX==250 ? "250x100" : NX==500 ? "500x200" : "1000x400");
            printf("%-12s %-6d %.6f\n", gname, iter + 1, t);
            fprintf(fp, "%d,%d,%d,%.6f\n", NX, NY, iter + 1, t);
        }

        free(mesh_value); free(points);
    }

    fclose(fp);
    printf("Done. Results in consistency_results.csv\n");
    return 0;
}
