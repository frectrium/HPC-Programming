#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

    double total_time = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {

        // Initialize Points randomly
        // Use it to randomly initalize points inside the domain
        // you can also use this to generate values instead of using input_filemaker.c
        // initializepoints(Points);
        
        // Read scattered points from file
        read_points(file, points);
        
        memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);

        clock_t start = clock();

        // Perform interpolation
        interpolation(mesh_value, points);

        clock_t end = clock();

        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    save_mesh(mesh_value);
    printf("Total interpolation time (serial) = %lf seconds\n", total_time);

    // Free memory
    free(mesh_value);
    free(points);
    fclose(file);

    return 0;
}
