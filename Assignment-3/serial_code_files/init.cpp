#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"

// Random particle initialization (optional)
void initializepoints(Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        points[i].x = (double) rand() / RAND_MAX;
        points[i].y = (double) rand() / RAND_MAX;
    }
}

// Read particle positions from binary file
void read_points(FILE *file, Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        fread(&points[i].x, sizeof(double), 1, file);
        fread(&points[i].y, sizeof(double), 1, file);
    }
}