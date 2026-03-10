#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"

// Random particle initialization 
void initializepoints(Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        points[i].x = (double) rand() / RAND_MAX;
        points[i].y = (double) rand() / RAND_MAX;
    }
}
