#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* PIC operations */
void interpolation(double *mesh_value, Points *points);
void save_mesh(double *mesh_value);

/* Cleanup persistent buffers (call at program end) */
void interpolation_cleanup(void);

#endif
