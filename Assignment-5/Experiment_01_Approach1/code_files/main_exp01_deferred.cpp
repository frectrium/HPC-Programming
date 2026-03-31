
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
    /* approach: deferred */
    int grids[][2] = { {250,100}, {500,200}, {1000,400} };
    int ngrids = 3;

    long pcounts[] = { 100L, 10000L, 1000000L, 100000000L };
    int npcounts = 4;

    Maxiter = 10;

    printf("Grid_NX,Grid_NY,Particle_Count,Total_Interp_Time_s,Total_Mover_Time_s,Total_Time_s\n");
    fflush(stdout);

    for (int gi = 0; gi < ngrids; gi++) {
        NX = grids[gi][0];
        NY = grids[gi][1];
        GRID_X = NX + 1;
        GRID_Y = NY + 1;
        dx = 1.0 / NX;
        dy = 1.0 / NY;

        for (int pi = 0; pi < npcounts; pi++) {
            long pc = pcounts[pi];
            NUM_Points = (int)pc;

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Particles %ld: allocating...\n", NX, NY, pc);
            fflush(stderr);

            /* Allocate */
            double *mesh_value = (double *)calloc((long)GRID_X * GRID_Y, sizeof(double));
            Points *points = (Points *)calloc(pc, sizeof(Points));
            if (!mesh_value || !points) {
                printf("%d,%d,%ld,ALLOC_FAIL,ALLOC_FAIL,ALLOC_FAIL\n", NX, NY, pc);
                fflush(stdout);
                if (mesh_value) free(mesh_value);
                if (points) free(points);
                continue;
            }

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Particles %ld: initializing...\n", NX, NY, pc);
            fflush(stderr);

            /* Initialize once */
            srand(12345);
            initializepoints(points);

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Particles %ld: running %d iters...\n", NX, NY, pc, Maxiter);
            fflush(stderr);

            double total_interp = 0.0;
            double total_mover = 0.0;

            for (int iter = 0; iter < Maxiter; iter++) {
                /* Zero mesh each iteration */
                memset(mesh_value, 0, (long)GRID_X * GRID_Y * sizeof(double));

                clock_t t0 = clock();
                interpolation(mesh_value, points);
                clock_t t1 = clock();

                clock_t t2 = clock();
                mover_serial_deferred(points, dx, dy);
                clock_t t3 = clock();

                total_interp += (double)(t1 - t0) / CLOCKS_PER_SEC;
                total_mover  += (double)(t3 - t2) / CLOCKS_PER_SEC;
            }

            printf("%d,%d,%ld,%.6f,%.6f,%.6f\n", NX, NY, pc,
                   total_interp, total_mover, total_interp + total_mover);
            fflush(stdout);

            free(mesh_value);
            free(points);

            fprintf(stderr, "[PROGRESS] Grid %dx%d, Particles %ld: DONE\n", NX, NY, pc);
            fflush(stderr);
        }
    }
    return 0;
}
