#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "utils.h"

double min_val, max_val;

// Scatter: particles -> mesh (bilinear)
void interpolation(double *mesh_value, Points *points) {
    memset(mesh_value, 0, sizeof(double) * GRID_X * GRID_Y);
    for (int p = 0; p < NUM_Points; p++) {
        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int ix = (int)(x / dx);
        int iy = (int)(y / dy);
        if (ix >= NX) ix = NX - 1;
        if (iy >= NY) iy = NY - 1;
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;

        double lx = x - ix * dx;
        double ly = y - iy * dy;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx        * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx        * ly;

        int base = iy * GRID_X + ix;
        mesh_value[base]              += w00;
        mesh_value[base + 1]          += w10;
        mesh_value[base + GRID_X]     += w01;
        mesh_value[base + GRID_X + 1] += w11;
    }
}

void normalization(double *mesh_value) {
    int N = GRID_X * GRID_Y;
    double mn =  DBL_MAX;
    double mx = -DBL_MAX;
    for (int i = 0; i < N; i++) {
        if (mesh_value[i] < mn) mn = mesh_value[i];
        if (mesh_value[i] > mx) mx = mesh_value[i];
    }
    min_val = mn;
    max_val = mx;

    double range = mx - mn;
    if (range == 0.0) {
        for (int i = 0; i < N; i++) mesh_value[i] = 0.0;
        return;
    }
    for (int i = 0; i < N; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - mn) / range - 1.0;
    }
}

// Gather: mesh -> particles, then update positions
void mover(double *mesh_value, Points *points) {
    for (int p = 0; p < NUM_Points; p++) {
        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int ix = (int)(x / dx);
        int iy = (int)(y / dy);
        if (ix >= NX) ix = NX - 1;
        if (iy >= NY) iy = NY - 1;
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;

        double lx = x - ix * dx;
        double ly = y - iy * dy;

        double w00 = (dx - lx) * (dy - ly);
        double w10 = lx        * (dy - ly);
        double w01 = (dx - lx) * ly;
        double w11 = lx        * ly;

        int base = iy * GRID_X + ix;
        double Fi = w00 * mesh_value[base]
                  + w10 * mesh_value[base + 1]
                  + w01 * mesh_value[base + GRID_X]
                  + w11 * mesh_value[base + GRID_X + 1];

        double xn = x + Fi * dx;
        double yn = y + Fi * dy;

        if (xn < 0.0 || xn >= 1.0 || yn < 0.0 || yn >= 1.0) {
            points[p].is_void = true;
        } else {
            points[p].x = xn;
            points[p].y = yn;
        }
    }
}

void denormalization(double *mesh_value) {
    int N = GRID_X * GRID_Y;
    double range = max_val - min_val;
    if (range == 0.0) {
        for (int i = 0; i < N; i++) mesh_value[i] = min_val;
        return;
    }
    for (int i = 0; i < N; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * 0.5 * range + min_val;
    }
}

long long int void_count(Points *points) {
    long long int voids = 0;
    for (int i = 0; i < NUM_Points; i++) {
        voids += (int)points[i].is_void;
    }
    return voids;
}

void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}