/*
 * main_exp03_parallel.cpp — Experiment 03: Mover (Parallel Component)
 *
 * Records interpolation + serial mover + parallel mover timings per iteration.
 * Computes speedup as T_serial / T_parallel (wall clock).
 *
 * Compile: gcc -O3 -march=native -fopenmp -o exp03p main_exp03_parallel.cpp utils.cpp init.cpp -lm
 * Run:     ./exp03p
 * Output:  mover_results.csv
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

    NX = 1000; NY = 400; Maxiter = 10; NUM_Points = 14000000;
    GRID_X = NX + 1; GRID_Y = NY + 1;
    dx = 1.0 / NX; dy = 1.0 / NY;

    double *mesh_value = (double *)calloc((size_t)GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *)calloc(NUM_Points, sizeof(Points));

    if (!mesh_value || !points) {
        printf("Allocation failed\n");
        free(mesh_value); free(points); return 1;
    }

    initializepoints(points);

    FILE *fp = fopen("mover_results.csv", "w");
    if (!fp) { printf("Cannot open mover_results.csv\n"); return 1; }
    fprintf(fp, "Iter,Interp_Time_s,Serial_Mover_Time_s,"
            "Parallel_Mover_Clock_s,Parallel_Mover_Wall_s,"
            "Total_Serial_s,Total_Parallel_Wall_s,Speedup_Wall\n");

    printf("=== EXPERIMENT 3 (PARALLEL): 1000x400, 14M particles ===\n");
    printf("Iter\tInterp\t\tSer.Mover\tPar(clock)\tPar(wall)\tSpeedup\n");

    for (int iter = 0; iter < Maxiter; iter++) {
        clock_t si = clock();
        interpolation(mesh_value, points);
        clock_t ei = clock();

        clock_t sm = clock();
        mover_serial(points, dx, dy);
        clock_t em = clock();

        double wt0 = omp_get_wtime();
        clock_t sp = clock();
        mover_parallel(points, dx, dy);
        clock_t ep = clock();
        double wt1 = omp_get_wtime();

        double interp_time   = (double)(ei - si) / CLOCKS_PER_SEC;
        double serial_time   = (double)(em - sm) / CLOCKS_PER_SEC;
        double par_time_clk  = (double)(ep - sp) / CLOCKS_PER_SEC;
        double par_time_wall = wt1 - wt0;
        double total_serial  = interp_time + serial_time;
        double total_par     = interp_time + par_time_wall;
        double speedup       = (par_time_wall > 0.0) ? serial_time / par_time_wall : 0.0;

        printf("%d\t%lf\t%lf\t%lf\t%lf\t%.2f\n",
               iter + 1, interp_time, serial_time,
               par_time_clk, par_time_wall, speedup);
        fprintf(fp, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%.4f\n",
                iter + 1, interp_time, serial_time,
                par_time_clk, par_time_wall,
                total_serial, total_par, speedup);
    }

    fclose(fp);
    free(mesh_value); free(points);
    printf("Done. Results in mover_results.csv\n");
    return 0;
}
