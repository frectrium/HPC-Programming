#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    // Open binary file for reading
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file\n");
        exit(1);
    }

    // Read grid dimensions
    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);

    // Read number of Points and max iterations
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    // Since Number of points will be 1 more than number of cells
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Allocate memory for grid and Points
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

    double total_int_time = 0.0;
    double total_norm_time = 0.0;
    double total_move_time = 0.0;
    double total_denorm_time = 0.0;

    // Read scattered points from file
    read_points(file, points);

    for (int iter = 0; iter < Maxiter; iter++) {

        double t0 = omp_get_wtime();

        // Perform interpolation
        interpolation(mesh_value, points);

        double t1 = omp_get_wtime();

        normalization(mesh_value);

        double t3 = omp_get_wtime();

        // Perform interpolation
        mover(mesh_value, points);

        double t4 = omp_get_wtime();

        denormalization(mesh_value);

        double t5 = omp_get_wtime();

        total_int_time += (t1 - t0);
        total_norm_time += (t3 - t1);
        total_move_time += (t4 - t3);
        total_denorm_time += (t5 - t4);
    }

    save_mesh(mesh_value);
    printf("Total Interpolation Time = %lf seconds\n", total_int_time);
    printf("Total Normalization Time = %lf seconds\n", total_norm_time);
    printf("Total Mover Time = %lf seconds\n", total_move_time);
    printf("Total Denormalization Time = %lf seconds\n", total_denorm_time);
    printf("Total Algorithm Time = %lf seconds\n", total_int_time + total_norm_time + total_move_time + total_denorm_time);
    printf("Total Number of Voids = %lld\n", void_count(points));
    
    // Free memory
    free(mesh_value);
    free(points);
    fclose(file);

    return 0;
}