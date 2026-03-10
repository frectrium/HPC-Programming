#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    // Fixed Parameters
    NX = 200;
    NY = 200;
    Maxiter = 10;
    NUM_Points = 100000000;

    // Since Number of points will be 1 more than number of cells
    GRID_X = NX + 1;
    GRID_Y = NY + 1;

    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Fix Number of Threads
    omp_set_num_threads(4);

    // Allocate memory for grid and Points
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

    initializepoints(points);

    printf("Iter\tInterp\t\tMover\t\tTotal\n");

    for (int iter = 0; iter < Maxiter; iter++) {

        // Interpolation timing
        clock_t start_interp = clock();
        interpolation(mesh_value, points);
        clock_t end_interp = clock();

        // Mover timing
        clock_t start_move = clock();
        mover_serial(points, dx, dy);
        clock_t end_move = clock();

        double interp_time = (double)(end_interp - start_interp) / CLOCKS_PER_SEC;
        double move_time = (double)(end_move - start_move) / CLOCKS_PER_SEC;
        double total = interp_time + move_time;

        printf("%d\t%lf\t%lf\t%lf\n", iter+1, interp_time, move_time, total);
    }

    // Free memory
    free(mesh_value);
    free(points);

    return 0;
}
