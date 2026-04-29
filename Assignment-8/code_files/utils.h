#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

extern double min_val, max_val;

/* PIC kernels — operate on whichever NUM_Points the caller has set.
 * For MPI runs, main.cpp sets NUM_Points to the per-rank LOCAL particle count
 * after Scatterv; the kernels are then automatically per-rank correct. */
void interpolation(double *mesh_value, Points *points);
void mover(double *mesh_value, Points *points);

/* Min/max + normalize split so MPI can Allreduce min/max between the two. */
void mesh_minmax(const double *mesh_value, double *out_min, double *out_max);
void normalize_with_minmax(double *mesh_value, double mn, double mx);

/* Convenience wrapper used by serial baseline (computes local min/max). */
void normalization(double *mesh_value);
void denormalization(double *mesh_value);

long long int void_count(Points *points);
void save_mesh(double *mesh_value);

#endif
