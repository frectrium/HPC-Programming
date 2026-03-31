#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

/* Interpolation (serial, optimized for Haswell — from Assignment 04) */
void interpolation(double *mesh_value, Points *points);

/* Assignment 05 Movers — with deletion and insertion */

/* Deferred insertion: move all, compact, reinsert */
void mover_serial_deferred(Points *points, double deltaX, double deltaY);
void mover_parallel_deferred(Points *points, double deltaX, double deltaY, int nthreads);

/* Immediate replacement: replace out-of-bounds on the spot */
void mover_serial_immediate(Points *points, double deltaX, double deltaY);
void mover_parallel_immediate(Points *points, double deltaX, double deltaY, int nthreads);

/* Legacy wrappers (called by main.cpp) */
void mover_serial(Points *points, double deltaX, double deltaY);
void mover_parallel(Points *points, double deltaX, double deltaY);

/* Old Assignment 04 mover — no deletion, periodic wrap-around */
void mover_parallel_no_delete(Points *points, double deltaX, double deltaY, int nthreads);

void save_mesh(double *mesh_value);

#endif