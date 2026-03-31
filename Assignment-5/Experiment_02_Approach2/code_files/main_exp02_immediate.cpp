
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

int main(int argc, char **argv) {
    /* approach: immediate */
    int grids[][2] = { {250,100}, {500,200}, {1000,400} };
    int ngrids = 3;
    int thread_counts[] = { 1, 2, 4, 8, 16 };
    int ntc = 5;
    Maxiter = 10;
    NUM_Points = 14000000;

    printf("Grid_NX,Grid_NY,Num_Threads,Total_Interp_Time_s,Total_Mover_Time_s,Total_Time_s,Mover_Speedup\n");
    fflush(stdout);

    for (int gi = 0; gi < ngrids; gi++) {
        NX = grids[gi][0];
        NY = grids[gi][1];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double serial_mover_time = 0.0;

        for (int ti = 0; ti < ntc; ti++) {
            int nt = thread_counts[ti];

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Threads %d: allocating...\n", NX, NY, nt);
            fflush(stderr);

            double *mesh_value = (double *)calloc((long)GRID_X * GRID_Y, sizeof(double));
            Points *points = (Points *)calloc((long)NUM_Points, sizeof(Points));
            if (!mesh_value || !points) {
                printf("%d,%d,%d,ALLOC_FAIL,ALLOC_FAIL,ALLOC_FAIL,0.0\n", NX, NY, nt);
                fflush(stdout);
                if (mesh_value) free(mesh_value);
                if (points) free(points);
                continue;
            }

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Threads %d: initializing 14M particles...\n", NX, NY, nt);
            fflush(stderr);

            srand(12345);
            initializepoints(points);

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Threads %d: running %d iters...\n", NX, NY, nt, Maxiter);
            fflush(stderr);

            double total_interp = 0.0;
            double total_mover = 0.0;

            for (int iter = 0; iter < Maxiter; iter++) {
                memset(mesh_value, 0, (long)GRID_X * GRID_Y * sizeof(double));

                clock_t ci0 = clock();
                interpolation(mesh_value, points);
                clock_t ci1 = clock();
                total_interp += (double)(ci1 - ci0) / CLOCKS_PER_SEC;

                double wt0 = omp_get_wtime();
                
                if (nt == 1) {
                    mover_serial_immediate(points, dx, dy);
                } else {
                    mover_parallel_immediate(points, dx, dy, nt);
                }

                double wt1 = omp_get_wtime();
                total_mover += (wt1 - wt0);
            }

            if (nt == 1) {
                serial_mover_time = total_mover;
            }

            double speedup = (serial_mover_time > 0.0) ? serial_mover_time / total_mover : 1.0;

            printf("%d,%d,%d,%.6f,%.6f,%.6f,%.4f\n", NX, NY, nt,
                   total_interp, total_mover, total_interp + total_mover, speedup);
            fflush(stdout);

            free(mesh_value);
            free(points);

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Threads %d: DONE\n", NX, NY, nt);
            fflush(stderr);
        }
    }
    return 0;
}
