/*
 * main_exp03_serial.cpp — Experiment 03: Mover (Serial Component)
 *
 * Records interpolation time + serial mover time per iteration.
 *
 * Compile: gcc -O3 -march=native -fopenmp -o exp03s main_exp03_serial.cpp utils.cpp init.cpp -lm
 * Run:     ./exp03s
 * Output:  mover_results.csv (serial columns)
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
    fprintf(fp, "Iter,Interp_Time_s,Serial_Mover_Time_s,Total_Serial_s\n");

    printf("=== EXPERIMENT 3 (SERIAL): 1000x400, 14M particles ===\n");
    printf("Iter\tInterp\t\tSer.Mover\tTotal\n");

    for (int iter = 0; iter < Maxiter; iter++) {
        clock_t si = clock();
        interpolation(mesh_value, points);
        clock_t ei = clock();

        clock_t sm = clock();
        mover_serial(points, dx, dy);
        clock_t em = clock();

        double interp_time = (double)(ei - si) / CLOCKS_PER_SEC;
        double serial_time = (double)(em - sm) / CLOCKS_PER_SEC;
        double total = interp_time + serial_time;

        printf("%d\t%lf\t%lf\t%lf\n", iter + 1, interp_time, serial_time, total);
        fprintf(fp, "%d,%lf,%lf,%lf\n", iter + 1, interp_time, serial_time, total);
    }

    fclose(fp);
    free(mesh_value); free(points);
    printf("Done. Results in mover_results.csv\n");
    return 0;
}
