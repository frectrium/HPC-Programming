/*
 * main.cpp — PIC Interpolation driver
 *
 * Changes from original:
 *   - Uses omp_get_wtime() instead of clock() when compiled with -fopenmp
 *     (clock() measures CPU time across ALL threads, which is wrong for speedup)
 *   - Prints thread count for parallel builds
 *   - Calls interpolation_cleanup() to free persistent buffers
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "init.h"
#include "utils.h"

/* Global variables */
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    /* Open binary file for reading */
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file\n");
        exit(1);
    }

    /* Read grid dimensions */
    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);

    /* Read number of Points and max iterations */
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    /* Since Number of points will be 1 more than number of cells */
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    /* Allocate memory for grid and Points */
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

    double total_time = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {

        /* Read scattered points from file */
        read_points(file, points);

        memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);

#ifdef _OPENMP
        double start = omp_get_wtime();
#else
        clock_t start = clock();
#endif

        /* Perform interpolation */
        interpolation(mesh_value, points);

#ifdef _OPENMP
        double end = omp_get_wtime();
        total_time += (end - start);
#else
        clock_t end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
#endif
    }

    save_mesh(mesh_value);

#ifdef _OPENMP
    printf("Threads: %d, Total interpolation time = %lf seconds\n",
           omp_get_max_threads(), total_time);
#else
    printf("Total interpolation time (serial) = %lf seconds\n", total_time);
#endif

    /* Cleanup */
    interpolation_cleanup();
    free(mesh_value);
    free(points);
    fclose(file);

    return 0;
}
